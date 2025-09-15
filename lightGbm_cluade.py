"""
Melbourne Pedestrian Count Prediction Model
============================================
Predicts pedestrian counts based on location (latitude/longitude) and temporal features
using XGBoost and LightGBM with GPU support and memory optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
import gc
import joblib
from haversine import haversine, Unit
warnings.filterwarnings('ignore')
import math
# Optional: For GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU is available")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not installed. Running on CPU.")

class PedestrianCountPredictor:
    """
    A class for predicting pedestrian counts based on location and temporal features
    """
    
    def __init__(self, model_type='lightgbm', use_gpu=False):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        model_type : str, 'lightgbm' or 'xgboost'
        use_gpu : bool, whether to use GPU if available
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.sensor_locations = None
        
    def _compute_global_spatial_bins(self, df, bins=10):
        lat = pd.to_numeric(df['latitude'], errors='coerce').astype('float64')
        lon = pd.to_numeric(df['longitude'], errors='coerce').astype('float64')

        lat_q = max(1, min(bins, lat.nunique()))
        lon_q = max(1, min(bins, lon.nunique()))

        # Quantile bins; drop duplicates if values repeat
        df['lat_zone'], lat_bins = pd.qcut(
            lat, q=lat_q, labels=False, retbins=True, duplicates='drop'
        )
        df['lon_zone'], lon_bins = pd.qcut(
            lon, q=lon_q, labels=False, retbins=True, duplicates='drop'
        )

        self.lat_bins = lat_bins.tolist()
        self.lon_bins = lon_bins.tolist()

        # Fallback if there’s effectively no spread
        if len(lat_bins) < 2:
            v = lat.dropna().iloc[0] if lat.notna().any() else 0.0
            lat_bins = np.array([v - 1e-6, v + 1e-6])
            df['lat_zone'] = 0
        if len(lon_bins) < 2:
            v = lon.dropna().iloc[0] if lon.notna().any() else 0.0
            lon_bins = np.array([v - 1e-6, v + 1e-6])
            df['lon_zone'] = 0

        df['lat_zone'] = df['lat_zone'].astype('Int8')
        df['lon_zone'] = df['lon_zone'].astype('Int8')

        # save for prediction time
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        return df

    def _haversine_vec(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        lat1 = np.asarray(lat1, dtype='float64')
        lon1 = np.asarray(lon1, dtype='float64')
        lat2 = float(lat2)  # ensure scalar float, not 'latitude'
        lon2 = float(lon2)

        lat1r = np.radians(lat1)
        lon1r = np.radians(lon1)
        lat2r = np.radians(lat2)
        lon2r = np.radians(lon2)

        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
        return (2 * R * np.arcsin(np.sqrt(a))).astype('float32')




    def load_and_preprocess_data(self, file_path: str):
        """
        Load the joined dataset (parquet preferred) and build stable spatial bins.
        """
        import os
        print("Loading data...")

        if file_path.lower().endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            # Fallback to CSV; auto-parse common timestamp cols if present
            parse_cols = [c for c in ["timestamp_local", "timestamp_utc", "date"] if c in pd.read_csv(file_path, nrows=1).columns]
            df = pd.read_csv(file_path, parse_dates=parse_cols if parse_cols else None)

        # Required columns
        req = {"location_id", "timestamp_local", "count", "latitude", "longitude"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Input missing columns: {sorted(missing)}")

        # Ensure Melbourne timezone (don’t strip tz)
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
        if getattr(df["timestamp_local"].dt, "tz", None) is None:
            df["timestamp_local"] = df["timestamp_local"].dt.tz_localize("Australia/Melbourne", nonexistent="NaT", ambiguous="NaT")
        else:
            df["timestamp_local"] = df["timestamp_local"].dt.tz_convert("Australia/Melbourne")
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
        if getattr(df["timestamp_local"].dt, "tz", None) is None:
            df["timestamp_local"] = df["timestamp_local"].dt.tz_localize(
                "Australia/Melbourne", nonexistent="NaT", ambiguous="NaT"
            )
        else:
            df["timestamp_local"] = df["timestamp_local"].dt.tz_convert("Australia/Melbourne")

        # 👇 drop rows where timestamp_local is NaT (DST gaps / bad parses)
        df = df.dropna(subset=["timestamp_local"]).copy()

        # Time features (safe now)
        df["hour"] = df["timestamp_local"].dt.hour
        df["dow"]  = df["timestamp_local"].dt.dayofweek

        # 👇 fill NA in the boolean before casting to integer
        is_weekend_bool = df["dow"].ge(5).fillna(False)
        df["is_weekend"] = is_weekend_bool.astype("uint8")

        # If you build other booleans, also use the filled mask
        df["is_business_hours"] = (df["hour"].between(9, 17) & ~is_weekend_bool).astype("uint8")
        df["is_peak_hour"] = (df["hour"].isin([7,8,9,17,18,19]) & ~is_weekend_bool).astype("uint8")

        # Basic time features (downstream code also adds more)
        t = df["timestamp_local"]
        df["hour"]       = t.dt.hour.astype("Int8")
        df["dow"]        = t.dt.dayofweek.astype("Int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("uint8")

        # Coerce numerics
        for c in ["latitude", "longitude", "count"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "count"])

        # Build **once** the global spatial bins and save edges
        df = self._compute_global_spatial_bins(df, bins=10)

        print(f"Data loaded: {len(df):,} rows from {df['location_id'].nunique()} sensors")
        return df

    
    def _process_chunk(self, chunk):
        """Normalize columns, build timestamp, then add features safely."""

        # 1) Normalize column names from Melbourne datasets → your pipeline names
        rename_map = {
            'Location_ID': 'location_id',
            'Sensor_Name': 'sensor_name',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Direction_1': 'dir1',
            'Direction_2': 'dir2',
            'Total_of_Directions': 'count',
            'Sensing_Date': 'sensing_date',  # date-only column
            'HourDay': 'hour'                # 0..23
        }
        chunk = chunk.rename(columns=rename_map)

        # 2) Ensure we have a usable timestamp_local
        if 'timestamp_local' not in chunk.columns:
            if {'sensing_date', 'hour'}.issubset(chunk.columns):
                # Coerce hour to [0..23]
                chunk['hour'] = pd.to_numeric(chunk['hour'], errors='coerce').clip(0, 23).fillna(0).astype('Int16')
                base_date = pd.to_datetime(chunk['sensing_date'], errors='coerce')
                chunk['timestamp_local'] = base_date + pd.to_timedelta(chunk['hour'].astype('int16'), unit='h')
            elif 'date' in chunk.columns:
                chunk['timestamp_local'] = pd.to_datetime(chunk['date'], errors='coerce')
            else:
                # Try common fallbacks if your joined file used another name
                for cand in ['Timestamp', 'timestamp', 'time', 'datetime']:
                    if cand in chunk.columns:
                        chunk['timestamp_local'] = pd.to_datetime(chunk[cand], errors='coerce')
                        break

        # Force datetime dtype (handles strings and mixed)
        if 'timestamp_local' in chunk.columns:
            chunk['timestamp_local'] = pd.to_datetime(chunk['timestamp_local'], errors='coerce')
            # drop tz if present (make naive)
            try:
                if chunk['timestamp_local'].dt.tz is not None:
                    chunk['timestamp_local'] = chunk['timestamp_local'].dt.tz_localize(None)
            except Exception:
                pass
        else:
            raise ValueError("No timestamp column found. Expected 'timestamp_local' or (Sensing_Date + HourDay).")

        # 3) Now it's safe to use .dt
        t = chunk['timestamp_local']
        if not pd.api.types.is_datetime64_any_dtype(t):
            raise ValueError("timestamp_local could not be parsed to datetime; check input format.")

        chunk['year']       = t.dt.year.astype('Int16')
        chunk['month']      = t.dt.month.astype('Int8')
        chunk['day']        = t.dt.day.astype('Int8')
        chunk['hour']       = t.dt.hour.astype('Int8')      # overwrites raw hour for consistency
        chunk['minute']     = t.dt.minute.astype('Int8')
        chunk['dayofyear']  = t.dt.dayofyear.astype('Int16')
        chunk['weekofyear'] = t.dt.isocalendar().week.astype('Int16')  # safe & allows <NA>
        chunk['dow']        = t.dt.dayofweek.astype('Int8')
        chunk['is_weekend'] = (chunk['dow'] >= 5).astype('boolean')  # nullable boolean, allows <NA>

        # later:
        # when you build spatial bins, allow NA too
        

        # 4) Ensure counts/dirs exist
        if 'count' not in chunk.columns:
            if {'dir1', 'dir2'}.issubset(chunk.columns):
                chunk['count'] = (chunk['dir1'].fillna(0) + chunk['dir2'].fillna(0)).astype('float32')
            else:
                raise ValueError("Missing 'count' (or 'Total_of_Directions') and also 'dir1/dir2'.")

        for c in ['dir1', 'dir2']:
            if c in chunk.columns:
                chunk[c] = chunk[c].fillna(0).astype('float32')

        # Directional derived features
        chunk['total_dir'] = chunk.get('dir1', 0) + chunk.get('dir2', 0)
        chunk['dir_ratio'] = np.where(chunk['total_dir'] > 0,
                                    chunk.get('dir1', 0) / chunk['total_dir'],
                                    0.5).astype('float32')

        # 5) Ensure lat/lon present (your joined CSV should already have them)
        if 'latitude' not in chunk.columns:
            chunk['latitude'] = np.nan
        if 'longitude' not in chunk.columns:
            chunk['longitude'] = np.nan

        return chunk
    
    def create_location_features(self, df):
        """
        Create location-based features including distance to key sensors
        
        Parameters:
        -----------
        df : pandas DataFrame
        """
        print("Creating location features...")
        
        # Store unique sensor locations for later predictions
        self.sensor_locations = df[['location_id', 'latitude', 'longitude', 'sensor_name']].drop_duplicates()
        
        # Calculate statistics for each location
        location_stats = df.groupby('location_id').agg({
            'count': ['mean', 'std', 'median', 'min', 'max'],
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        location_stats.columns = ['_'.join(col).strip('_') for col in location_stats.columns.values]
        
        # Merge back to main dataframe
        df = df.merge(location_stats, left_on='location_id', right_on='location_id', 
                     suffixes=('', '_loc_stats'), how='left')
        
        # Create spatial clusters (simple grid-based)

        df['spatial_cluster'] = (df['lat_zone'].astype('Int16') * 10 + df['lon_zone'].astype('Int16')).astype('Int16')
        
        # Distance to city center (Melbourne CBD approximate center)
        cbd_lat, cbd_lon = -37.8136, 144.9631

        df['dist_to_cbd'] = self._haversine_vec(df['latitude'].to_numpy(),
                                        df['longitude'].to_numpy(),
                                        cbd_lat, cbd_lon)
        df[['latitude','longitude']] = df[['latitude','longitude']].apply(pd.to_numeric, errors='coerce')

        # one row per location_id with numeric lat/lon
        coords = (self.sensor_locations
                .dropna(subset=['latitude','longitude'])
                .drop_duplicates('location_id')
                .set_index('location_id')[['latitude','longitude']]
                .astype('float64'))

        top_locations = (df.groupby('location_id')['count']
                        .mean().nlargest(5).index.tolist())

        for loc_id in top_locations:
            if loc_id in coords.index:
                lat2 = float(coords.at[loc_id, 'latitude'])   # <- explicit cell access
                lon2 = float(coords.at[loc_id, 'longitude'])
                df[f'dist_to_loc_{loc_id}'] = self._haversine_vec(
                    df['latitude'].to_numpy(), df['longitude'].to_numpy(), lat2, lon2
                )
            else:
                # location_id without coordinates; skip it
                continue
        
        return df
    
    def create_temporal_features(self, df):
        print("Creating temporal features...")

        # --- cyclic encodings (OK if hour/dow have NaN; results become NaN floats) ---
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
        df['dow_sin']  = np.sin(2 * np.pi * df['dow'] / 7).astype('float32')
        df['dow_cos']  = np.cos(2 * np.pi * df['dow'] / 7).astype('float32')
        df['month_sin'] = np.sin(2 * np.pi * df.get('month', 1) / 12).astype('float32')
        df['month_cos'] = np.cos(2 * np.pi * df.get('month', 1) / 12).astype('float32')

        # Time of day categories
        df['time_category'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        ).astype('category')

        # --- IMPORTANT: work with nullable boolean; fill NAs to False for masks ---
        is_weekend = df['is_weekend'].astype('boolean').fillna(False)

        df['is_business_hours'] = (
            df['hour'].ge(9) & df['hour'].le(17) & (~is_weekend)
        ).astype('boolean')

        df['is_peak_hour'] = (
            df['hour'].isin([7, 8, 9, 17, 18, 19]) & (~is_weekend)
        ).astype('boolean')

        # Optional: convert boolean features to integers (saves memory, plays nice with sklearn)
        for col in ['is_weekend', 'is_business_hours', 'is_peak_hour']:
            df[col] = df[col].fillna(False).astype('uint8')

        return df
    
    def create_lag_features(self, df, lag_periods=[1, 24, 168]):
        """
        Create lag features for time series patterns
        
        Parameters:
        -----------
        df : pandas DataFrame
        lag_periods : list of int, hours to lag
        """
        print("Creating lag features...")
        
        # Sort by location and time
        df = df.sort_values(['location_id', 'timestamp_local'])
        
        for lag in lag_periods:
            df[f'count_lag_{lag}h'] = df.groupby('location_id')['count'].shift(lag)
            
        # Rolling statistics
        for window in [24, 168]:  # Daily and weekly windows
            g = df.groupby('location_id')['count']
            df[f'count_roll_mean_{window}h'] = g.transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1)).astype('float32')
            df[f'count_roll_std_{window}h']  = g.transform(lambda x: x.rolling(window, min_periods=2).std().shift(1)).astype('float32')

        
        return df
    
    def prepare_features(self, df):
        """
        Prepare all features for modeling
        
        Parameters:
        -----------
        df : pandas DataFrame
        """
        # Create all feature types
        df = self.create_location_features(df)
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        
        # Select feature columns
        feature_cols = [
            # Location / geo
            "latitude", "longitude", "dist_to_cbd", "spatial_cluster",
            "lat_zone", "lon_zone",
            # Time
            "hour", "dow", "is_weekend", "is_business_hours", "is_peak_hour",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            # Directional if present
            "dir1", "dir2", "total_dir", "dir_ratio",
            # Lags & rolling
            "count_lag_1h", "count_lag_24h", "count_lag_168h",
            "count_roll_mean_24h", "count_roll_std_24h",
            "count_roll_mean_168h", "count_roll_std_168h",
        ]
            
        '''
        # Add distance features
        dist_cols = [col for col in df.columns if col.startswith('dist_to_loc_')]
        feature_cols.extend(dist_cols)
        
        # Add temporal features if they exist
        if 'year' in df.columns:
            feature_cols.extend(['year', 'month', 'day', 'dayofyear', 'weekofyear'])
        
        # Filter to existing columns
        '''
        self.feature_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"Total features: {len(self.feature_cols)}")
        
        return df
    
    def train_model(self, df, valid_days=28):
        print(f"Training {self.model_type} model...")

        df = self.prepare_features(df)
        df = df.dropna(subset=self.feature_cols + ["count"]).sort_values("timestamp_local")

        # Time-based split
        cut = df["timestamp_local"].max() - pd.Timedelta(days=valid_days)
        train = df[df["timestamp_local"] <= cut]
        test  = df[df["timestamp_local"]  > cut]

        X_train, y_train = train[self.feature_cols].values, train["count"].values
        X_test,  y_test  = test[self.feature_cols].values,  test["count"].values

        if self.model_type == "lightgbm":
            params = {
                "objective": "tweedie",
                "tweedie_variance_power": 1.2,
                "metric": ["rmse", "mae"],
                "learning_rate": 0.05,
                "num_leaves": 64,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "verbose": -1,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test,  label=y_test, reference=train_data)
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=3000,
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
            )
        else:
            # XGBoost version if you really want it (kept simple)
            import xgboost as xgb
            dtrain, dvalid = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_test, label=y_test)
            self.model = xgb.train(
                {"objective": "reg:tweedie", "tweedie_variance_power": 1.2, "eta": 0.05, "max_depth": 8},
                dtrain, num_boost_round=3000, evals=[(dvalid, "valid")], early_stopping_rounds=200, verbose_eval=200
            )

        # Evaluate
        pred = (self.model.predict(X_test, num_iteration=self.model.best_iteration)
                if self.model_type == "lightgbm" else
                self.model.predict(xgb.DMatrix(X_test)))
        mae  = mean_absolute_error(y_test, pred)
        rmse = math.sqrt(mean_squared_error(y_test, pred))
        wape = (np.abs(y_test - pred).sum() / (y_test.sum() + 1e-9))
        print(f"\nHoldout  MAE={mae:.2f}  RMSE={rmse:.2f}  WAPE={wape:.2%}")
        return self

    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, cv_folds):
        """Train LightGBM model"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

        tuning_space = {
            'num_leaves': [31, 63, 127,255,511],
            'max_depth': [-1, 5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 30],
            'reg_alpha': [0.0, 0.1, 0.2],
            'reg_lambda': [0.0, 0.1, 0.2],
            'min_split_gain': [0.0, 0.1, 0.2]
        }
        
        # Add GPU parameters if available
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train with early stopping
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, cv_folds):
        """Train XGBoost model"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Add GPU parameters if available
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
        
        # Create DMatrix for efficiency
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': [self.feature_cols[int(k[1:])] for k in importance.keys()],
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : array-like
        y_test : array-like
        """
        if self.model_type == 'lightgbm':
            y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        else:
            dtest = xgb.DMatrix(X_test)
            y_pred = self.model.predict(dtest)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("Model Performance on Test Set:")
        print("="*50)
        print(f"MAE:  {mae:.2f} pedestrians")
        print(f"RMSE: {rmse:.2f} pedestrians")
        print(f"R²:   {r2:.4f}")
        
        # Calculate MAPE for non-zero counts
        mask = y_test > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            print(f"MAPE: {mape:.2f}%")
    
    def _zone_from_bins(self, value, bins):
        if bins is None or len(bins) < 2 or pd.isna(value):
            return 0
        idx = int(np.digitize([float(value)], bins, right=False)[0] - 1)
        return int(np.clip(idx, 0, len(bins) - 2))

    def predict_location(self, latitude, longitude, timestamp, dow=None, is_weekend=None, hour=None):
        # Parse timestamp and force Melbourne tz
        ts = pd.to_datetime(timestamp)
        ts = ts.tz_localize("Australia/Melbourne") if ts.tzinfo is None else ts.tz_convert("Australia/Melbourne")

        # Temporal
        if dow is None: dow = ts.dayofweek
        if is_weekend is None: is_weekend = (dow >= 5)
        if hour is None: hour = ts.hour

        # Nearest sensor (for info)
        distances = self.sensor_locations.apply(
            lambda row: haversine((latitude, longitude), (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS),
            axis=1
        )
        nearest_sensor = self.sensor_locations.iloc[distances.idxmin()]

        # Zones via saved bins
        lat_zone = self._zone_from_bins(latitude, getattr(self, "lat_bins", None))
        lon_zone = self._zone_from_bins(longitude, getattr(self, "lon_bins", None))

        # Feature vector (lags/rolling unknown → NaN, not zero)
        feat = {
            "latitude": latitude,
            "longitude": longitude,
            "hour": hour,
            "dow": dow,
            "is_weekend": int(is_weekend),
            "dist_to_cbd": haversine((latitude, longitude), (-37.8136, 144.9631), unit=Unit.KILOMETERS),
            "hour_sin": np.sin(2*np.pi*hour/24),
            "hour_cos": np.cos(2*np.pi*hour/24),
            "dow_sin":  np.sin(2*np.pi*dow/7),
            "dow_cos":  np.cos(2*np.pi*dow/7),
            "month_sin": np.sin(2*np.pi*ts.month/12),
            "month_cos": np.cos(2*np.pi*ts.month/12),
            "is_business_hours": int((hour >= 9) and (hour <= 17) and (not is_weekend)),
            "is_peak_hour": int((hour in [7,8,9,17,18,19]) and (not is_weekend)),
            "lat_zone": lat_zone,
            "lon_zone": lon_zone,
            "spatial_cluster": lat_zone * 10 + lon_zone,
            "dir1": 0.0, "dir2": 0.0, "total_dir": 0.0, "dir_ratio": 0.5,
            "count_lag_1h": np.nan, "count_lag_24h": np.nan, "count_lag_168h": np.nan,
            "count_roll_mean_24h": np.nan, "count_roll_std_24h": np.nan,
            "count_roll_mean_168h": np.nan, "count_roll_std_168h": np.nan,
        }
        # Align to trained columns
        X = np.array([[feat.get(c, np.nan) for c in self.feature_cols]], dtype=float)

        # Predict (LightGBM handles NaNs)
        if self.model_type == "lightgbm":
            yhat = float(self.model.predict(X, num_iteration=getattr(self.model, "best_iteration", None))[0])
        else:
            import xgboost as xgb
            yhat = float(self.model.predict(xgb.DMatrix(X))[0])

        yhat = max(0.0, yhat)  # non-negative
        return {
            "predicted_count": int(round(yhat)),
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
            "nearest_sensor": nearest_sensor.get("sensor_name", ""),
            "distance_to_nearest_sensor_km": round(float(distances.min()), 2),
            "confidence_interval": {"lower": int(max(0, yhat * 0.8)), "upper": int(yhat * 1.2)},
        }

    
    def save_model(self, filepath):
        model_data = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "sensor_locations": self.sensor_locations,
            "model_type": self.model_type,
            "lat_bins": getattr(self, "lat_bins", None),
            "lon_bins": getattr(self, "lon_bins", None),
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.model            = model_data["model"]
        self.feature_cols     = model_data["feature_cols"]
        self.sensor_locations = model_data["sensor_locations"]
        self.model_type       = model_data["model_type"]
        self.lat_bins         = model_data.get("lat_bins")
        self.lon_bins         = model_data.get("lon_bins")
        print(f"Model loaded from {filepath}")
        return self
# ---- drop this into your file (e.g., under PedestrianCountPredictor class) ----


    def forecast_next_hours_known_sensors(predictor, history_df, H=6):
        """
        predictor : PedestrianCountPredictor already loaded with .load_model(...)
        history_df: DataFrame with ['location_id','timestamp_local','count','latitude','longitude']
                    timestamp_local must be tz-aware in Australia/Melbourne
        H         : number of future hours to predict (recursive)

        Returns DataFrame: [location_id, timestamp_local, pred, horizon_h]
        """
        # basic checks
        need = {"location_id","timestamp_local","count","latitude","longitude"}
        missing = need - set(history_df.columns)
        if missing:
            raise ValueError(f"history_df missing columns: {sorted(missing)}")

        wk = history_df.copy()
        # ensure Melbourne tz
        ts = pd.to_datetime(wk["timestamp_local"], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("Australia/Melbourne", nonexistent="NaT", ambiguous="NaT")
        else:
            ts = ts.dt.tz_convert("Australia/Melbourne")
        wk["timestamp_local"] = ts
        wk = wk.dropna(subset=["timestamp_local"]).sort_values(["location_id","timestamp_local"])

        out = []
        for step in range(1, H+1):
            t_next = wk["timestamp_local"].max() + pd.Timedelta(hours=1)

            base = (wk.groupby("location_id", as_index=False)
                    .tail(1)[["location_id","latitude","longitude"]])
            base["timestamp_local"] = t_next

            # time features (same as training)
            base["hour"] = base["timestamp_local"].dt.hour
            base["dow"]  = base["timestamp_local"].dt.dayofweek
            is_weekend = base["dow"] >= 5
            base["is_weekend"]       = is_weekend.astype("uint8")
            base["is_business_hours"]= (base["hour"].between(9,17) & ~is_weekend).astype("uint8")
            base["is_peak_hour"]     = (base["hour"].isin([7,8,9,17,18,19]) & ~is_weekend).astype("uint8")
            base["hour_sin"] = np.sin(2*np.pi*base["hour"]/24)
            base["hour_cos"] = np.cos(2*np.pi*base["hour"]/24)
            base["dow_sin"]  = np.sin(2*np.pi*base["dow"]/7)
            base["dow_cos"]  = np.cos(2*np.pi*base["dow"]/7)
            base["month_sin"]= np.sin(2*np.pi*base["timestamp_local"].dt.month/12)
            base["month_cos"]= np.cos(2*np.pi*base["timestamp_local"].dt.month/12)

            # spatial features
            def _zone(val, bins):
                if bins is None or len(bins) < 2 or pd.isna(val): return 0
                idx = int(np.digitize([float(val)], bins, right=False)[0] - 1)
                return int(np.clip(idx, 0, len(bins)-2))
            base["lat_zone"] = [ _zone(v, predictor.lat_bins) for v in base["latitude"] ]
            base["lon_zone"] = [ _zone(v, predictor.lon_bins) for v in base["longitude"] ]
            base["spatial_cluster"] = base["lat_zone"]*10 + base["lon_zone"]

            # distance to CBD (same as training)
            def hav(lat1, lon1, lat2=-37.8136, lon2=144.9631):
                R=6371.0088
                lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
                return 2*R*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
            base["dist_to_cbd"] = hav(base["latitude"].astype(float), base["longitude"].astype(float))

            # directional placeholders (your model may or may not use them)
            for c in ["dir1","dir2","total_dir","dir_ratio"]:
                base[c] = 0.0

            # lags/rollings from history
            g = wk.groupby("location_id")["count"]
            for L in (1,24,168):
                lastL = g.apply(lambda s, L=L: s.iloc[-L] if len(s) >= L else np.nan)
                base[f"count_lag_{L}h"] = base["location_id"].map(lastL.to_dict()).astype("float32")
            for w in (24,168):
                m = g.apply(lambda s, w=w: s.iloc[-w:].mean() if len(s) >= 1 else np.nan)
                s = g.apply(lambda s, w=w: s.iloc[-w:].std()  if len(s) >= 2 else 0.0)
                base[f"count_roll_mean_{w}h"] = base["location_id"].map(m.to_dict()).astype("float32")
                base[f"count_roll_std_{w}h"]  = base["location_id"].map(s.to_dict()).astype("float32")

            # line up features and predict
            X = base[[c for c in predictor.feature_cols]].to_numpy(dtype=float)
            yhat = predictor.model.predict(X, num_iteration=getattr(predictor.model, "best_iteration", None))
            step_out = base[["location_id"]].copy()
            step_out["timestamp_local"] = t_next
            step_out["pred"] = yhat
            step_out["horizon_h"] = step
            out.append(step_out)

            # recursive: append as next observed hour
            add = step_out.rename(columns={"pred":"count"})[["location_id","timestamp_local","count"]]
            add = add.merge(wk.drop_duplicates("location_id")[["location_id","latitude","longitude"]], on="location_id", how="left")
            wk = pd.concat([wk, add], ignore_index=True).sort_values(["location_id","timestamp_local"])

        return pd.concat(out, ignore_index=True).sort_values(["horizon_h","location_id"])


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    # Use 'lightgbm' for LightGBM or 'xgboost' for XGBoost
    predictor = PedestrianCountPredictor(model_type='lightgbm', use_gpu=False)
    
    # Load and preprocess data
    # Replace with your actual file path
    df = predictor.load_and_preprocess_data('data/processed/melbourne_pedestrian_hourly_joined.csv')

    # Train model
    predictor.train_model(df)  
    
    # Save model
    predictor.save_model('pedestrian_count_model_GPU_2.pkl')
    
    # Example prediction for a new location
    '''
    prediction = predictor.predict_location(
        latitude=-37.814,
        longitude=144.963,
        timestamp='2024-03-15 14:00:00'
    )
    '''
    

    # Batch prediction example
    print("\n" + "="*50)
    print("Batch Predictions for Multiple Locations:")
    print("="*50)
    
