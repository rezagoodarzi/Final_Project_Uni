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

# Optional: For GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
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
        
    def load_and_preprocess_data(self, file_path, chunksize=50000):
        """
        Load and preprocess the data with memory optimization
        
        Parameters:
        -----------
        file_path : str, path to the CSV file
        chunksize : int, number of rows to read at once
        """
        print("Loading data in chunks for memory efficiency...")
        
        # Define dtypes for memory optimization
        dtypes = {
            # hourly joined CSV may already be normalized; these are safe
            'Location_ID': 'Int64',        # allow NA just in case
            'location_id': 'Int64',
            'Sensor_Name': 'category',
            'sensor_name': 'category',
            'Latitude': 'float32',
            'Longitude': 'float32',
            'latitude': 'float32',
            'longitude': 'float32',
            'Direction_1': 'float32',
            'Direction_2': 'float32',
            'dir1': 'float32',
            'dir2': 'float32',
            'Total_of_Directions': 'float32',
            'count': 'float32',
            'HourDay': 'Int16'  # 0..23 if present
    }
        # Read data in chunks
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtypes, 
                                 parse_dates=['timestamp_local', 'timestamp_utc', 'date']):
            # Process each chunk
            chunk = self._process_chunk(chunk)
            chunks.append(chunk)
            
            # Clear memory
            gc.collect()
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
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

        chunk['year']       = t.dt.year.astype('int16')
        chunk['month']      = t.dt.month.astype('int8')
        chunk['day']        = t.dt.day.astype('int8')
        chunk['hour']       = t.dt.hour.astype('int8')     # overwrite any raw hour to keep consistency
        chunk['minute']     = t.dt.minute.astype('int8')
        chunk['dayofyear']  = t.dt.dayofyear.astype('int16')
        # isocalendar().week is UInt32 → cast to int8 (max 53)
        chunk['weekofyear'] = t.dt.isocalendar().week.astype('int8')
        chunk['dow']        = t.dt.dayofweek.astype('int8')
        chunk['is_weekend'] = (chunk['dow'] >= 5)

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
        df['lat_zone'] = pd.cut(df['latitude'], bins=10, labels=False).astype('int8')
        df['lon_zone'] = pd.cut(df['longitude'], bins=10, labels=False).astype('int8')
        df['spatial_cluster'] = df['lat_zone'] * 10 + df['lon_zone']
        
        # Distance to city center (Melbourne CBD approximate center)
        cbd_lat, cbd_lon = -37.8136, 144.9631
        df['dist_to_cbd'] = df.apply(
            lambda row: haversine((row['latitude'], row['longitude']), 
                                (cbd_lat, cbd_lon), unit=Unit.KILOMETERS),
            axis=1
        ).astype('float32')
        
        # Calculate distances to top 5 busiest locations
        top_locations = df.groupby('location_id')['count'].mean().nlargest(5).index
        for loc_id in top_locations:
            loc_data = self.sensor_locations[self.sensor_locations['location_id'] == loc_id].iloc[0]
            df[f'dist_to_loc_{loc_id}'] = df.apply(
                lambda row: haversine((row['latitude'], row['longitude']), 
                                    (loc_data['latitude'], loc_data['longitude']), 
                                    unit=Unit.KILOMETERS),
                axis=1
            ).astype('float32')
        
        return df
    
    def create_temporal_features(self, df):
        """
        Create advanced temporal features
        
        Parameters:
        -----------
        df : pandas DataFrame
        """
        print("Creating temporal features...")
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7).astype('float32')
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7).astype('float32')
        df['month_sin'] = np.sin(2 * np.pi * df.get('month', 1) / 12).astype('float32')
        df['month_cos'] = np.cos(2 * np.pi * df.get('month', 1) / 12).astype('float32')
        
        # Time of day categories
        df['time_category'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['night', 'morning', 'afternoon', 'evening']).astype('category')
        
        # Business hours flag
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                   (~df['is_weekend'])).astype('bool')
        
        # Peak hours flag
        df['is_peak_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19])) & 
                              (~df['is_weekend'])).astype('bool')
        
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
            df[f'count_roll_mean_{window}h'] = df.groupby('location_id')['count'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ).astype('float32')
            
            df[f'count_roll_std_{window}h'] = df.groupby('location_id')['count'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).astype('float32')
        
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
            # Location features
            'latitude', 'longitude', 'dist_to_cbd', 'spatial_cluster',
            'lat_zone', 'lon_zone',
            # Temporal features
            'hour', 'dow', 'is_weekend', 'is_business_hours', 'is_peak_hour',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            # Directional features
            'dir1', 'dir2', 'total_dir', 'dir_ratio',
            # Location statistics
            'count_mean', 'count_std', 'count_median', 'count_min', 'count_max',
            # Lag features
            'count_lag_1h', 'count_lag_24h', 'count_lag_168h',
            'count_roll_mean_24h', 'count_roll_std_24h',
            'count_roll_mean_168h', 'count_roll_std_168h'
        ]
        
        # Add distance features
        dist_cols = [col for col in df.columns if col.startswith('dist_to_loc_')]
        feature_cols.extend(dist_cols)
        
        # Add temporal features if they exist
        if 'year' in df.columns:
            feature_cols.extend(['year', 'month', 'day', 'dayofyear', 'weekofyear'])
        
        # Filter to existing columns
        self.feature_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"Total features: {len(self.feature_cols)}")
        
        return df
    
    def train_model(self, df, test_size=0.2, cv_folds=5):
        """
        Train the prediction model
        
        Parameters:
        -----------
        df : pandas DataFrame
        test_size : float, proportion of data for testing
        cv_folds : int, number of cross-validation folds
        """
        print(f"Training {self.model_type} model...")
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=self.feature_cols + ['count'])
        
        # Prepare X and y
        X = df[self.feature_cols].values
        y = df['count'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Train model based on type
        if self.model_type == 'lightgbm':
            self._train_lightgbm(X_train, y_train, X_test, y_test, cv_folds)
        else:
            self._train_xgboost(X_train, y_train, X_test, y_test, cv_folds)
        
        # Evaluate on test set
        self.evaluate(X_test, y_test)
        
        # Clear memory
        gc.collect()
        
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
    
    def predict_location(self, latitude, longitude, timestamp, 
                        dow=None, is_weekend=None, hour=None):
        """
        Predict pedestrian count for a new location
        
        Parameters:
        -----------
        latitude : float
        longitude : float
        timestamp : str or datetime
        dow : int, day of week (0-6), optional
        is_weekend : bool, optional
        hour : int, hour of day (0-23), optional
        
        Returns:
        --------
        dict : prediction results
        """
        # Parse timestamp if string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Extract temporal features if not provided
        if dow is None:
            dow = timestamp.dayofweek
        if is_weekend is None:
            is_weekend = dow >= 5
        if hour is None:
            hour = timestamp.hour
        
        # Find nearest sensor for location statistics
        distances = self.sensor_locations.apply(
            lambda row: haversine((latitude, longitude), 
                                (row['latitude'], row['longitude']), 
                                unit=Unit.KILOMETERS),
            axis=1
        )
        nearest_sensor_idx = distances.idxmin()
        nearest_sensor = self.sensor_locations.iloc[nearest_sensor_idx]
        
        # Create feature vector
        features = {
            'latitude': latitude,
            'longitude': longitude,
            'hour': hour,
            'dow': dow,
            'is_weekend': is_weekend,
            'dist_to_cbd': haversine((latitude, longitude), 
                                    (-37.8136, 144.9631), 
                                    unit=Unit.KILOMETERS),
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * dow / 7),
            'dow_cos': np.cos(2 * np.pi * dow / 7),
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
            'is_business_hours': (hour >= 9) and (hour <= 17) and (not is_weekend),
            'is_peak_hour': (hour in [7, 8, 9, 17, 18, 19]) and (not is_weekend),
            'lat_zone': int(pd.cut([latitude], bins=10, labels=False)[0]),
            'lon_zone': int(pd.cut([longitude], bins=10, labels=False)[0]),
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'dayofyear': timestamp.dayofyear,
            'weekofyear': timestamp.isocalendar()[1]
        }
        
        # Add default values for missing features
        for col in self.feature_cols:
            if col not in features:
                features[col] = 0
        
        # Create feature array
        X_pred = np.array([[features[col] for col in self.feature_cols]])
        X_pred = self.scaler.transform(X_pred)
        
        # Make prediction
        if self.model_type == 'lightgbm':
            prediction = self.model.predict(X_pred, num_iteration=self.model.best_iteration)[0]
        else:
            dpred = xgb.DMatrix(X_pred)
            prediction = self.model.predict(dpred)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        result = {
            'predicted_count': int(round(prediction)),
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'nearest_sensor': nearest_sensor['sensor_name'],
            'distance_to_nearest_sensor_km': round(distances.iloc[nearest_sensor_idx], 2),
            'confidence_interval': {
                'lower': int(max(0, prediction * 0.8)),
                'upper': int(prediction * 1.2)
            }
        }
        
        return result
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'sensor_locations': self.sensor_locations,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.sensor_locations = model_data['sensor_locations']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")
        return self


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    # Use 'lightgbm' for LightGBM or 'xgboost' for XGBoost
    predictor = PedestrianCountPredictor(model_type='lightgbm', use_gpu=False)
    
    # Load and preprocess data
    # Replace with your actual file path
    df = predictor.load_and_preprocess_data('data/processed/melbourne_pedestrian_hourly_joined.csv')

    # Train model
    predictor.train_model(df, test_size=0.2, cv_folds=5)
    
    # Save model
    predictor.save_model('pedestrian_count_model.pkl')
    
    # Example prediction for a new location
    prediction = predictor.predict_location(
        latitude=-37.814,
        longitude=144.963,
        timestamp='2024-03-15 14:00:00'
    )
    
    print("\n" + "="*50)
    print("Prediction for New Location:")
    print("="*50)
    print(f"Location: ({prediction['latitude']}, {prediction['longitude']})")
    print(f"Time: {prediction['timestamp']}")
    print(f"Predicted Count: {prediction['predicted_count']} pedestrians")
    print(f"Confidence Interval: {prediction['confidence_interval']['lower']}-{prediction['confidence_interval']['upper']}")
    print(f"Nearest Sensor: {prediction['nearest_sensor']}")
    print(f"Distance to Nearest Sensor: {prediction['distance_to_nearest_sensor_km']} km")
    
    # Batch prediction example
    print("\n" + "="*50)
    print("Batch Predictions for Multiple Locations:")
    print("="*50)
    
    locations = [
        (-37.8136, 144.9631),  # Melbourne CBD
        (-37.8183, 144.9671),  # Flinders Street Station area
        (-37.8102, 144.9628),  # Queen Victoria Market area
    ]
    
    for lat, lon in locations:
        pred = predictor.predict_location(
            latitude=lat,
            longitude=lon,
            timestamp='2024-03-15 17:30:00'  # Friday evening peak hour
        )
        print(f"({lat:.4f}, {lon:.4f}): {pred['predicted_count']} pedestrians")