import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

# -------- Mini loader/wrapper (no need to import your training file) --------
class MiniPedestrianPredictor:
    def __init__(self, model_path: str):
        blob = joblib.load(model_path)
        # pulled from your save_model()
        self.model = blob["model"]
        self.scaler = blob.get("scaler", None)
        self.feature_cols = blob["feature_cols"]
        self.sensor_locations = blob["sensor_locations"]
        self.model_type = blob.get("model_type", "lightgbm")

        # reconstruct spatial bins from available sensor coords
        self._reconstruct_bins_from_sensors(bins=10)

    # ---------- utilities ----------
    @staticmethod
    def _to_ts(x):
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(x)
        return pd.to_datetime(x, errors="coerce")

    def _reconstruct_bins_from_sensors(self, bins=10):
        sl = (self.sensor_locations
              .dropna(subset=["latitude", "longitude"])
              .drop_duplicates("location_id"))
        lat = pd.to_numeric(sl["latitude"], errors="coerce").astype("float64")
        lon = pd.to_numeric(sl["longitude"], errors="coerce").astype("float64")

        # derive quantile bins; drop duplicates if values repeat
        lat_q = max(1, min(bins, lat.nunique()))
        lon_q = max(1, min(bins, lon.nunique()))
        # using qcut just to get consistent edges
        _, lat_bins = pd.qcut(lat, q=lat_q, labels=False, retbins=True, duplicates="drop")
        _, lon_bins = pd.qcut(lon, q=lon_q, labels=False, retbins=True, duplicates="drop")

        # if degenerate, make tiny interval to avoid errors
        if len(lat_bins) < 2:
            v = float(lat.dropna().iloc[0]) if lat.notna().any() else 0.0
            lat_bins = np.array([v - 1e-6, v + 1e-6], dtype="float64")
        if len(lon_bins) < 2:
            v = float(lon.dropna().iloc[0]) if lon.notna().any() else 0.0
            lon_bins = np.array([v - 1e-6, v + 1e-6], dtype="float64")

        self.lat_bins = lat_bins
        self.lon_bins = lon_bins

    @staticmethod
    def _zone_from_bins(value, bins):
        if bins is None or len(bins) < 2 or pd.isna(value):
            return 0
        idx = int(np.digitize([float(value)], bins, right=False)[0] - 1)
        return int(np.clip(idx, 0, len(bins) - 2))

    @staticmethod
    def _haversine_vec(lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        lat1 = np.asarray(lat1, dtype="float64")
        lon1 = np.asarray(lon1, dtype="float64")
        lat2 = float(lat2)
        lon2 = float(lon2)
        lat1r = np.radians(lat1); lon1r = np.radians(lon1)
        lat2r = np.radians(lat2); lon2r = np.radians(lon2)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
        return (2 * R * np.arcsin(np.sqrt(a))).astype("float32")

    # ---------- single-point predict ----------
    def predict_location(self, latitude, longitude, timestamp):
        ts = self._to_ts(timestamp)
        if pd.isna(ts):
            raise ValueError(f"Could not parse timestamp: {timestamp}")

        dow = int(ts.dayofweek)
        is_weekend = int(dow >= 5)
        hour = int(ts.hour)

        # nearest sensor (index-safe)
        sensors = (self.sensor_locations
                   .dropna(subset=["latitude", "longitude"])
                   .reset_index(drop=True))
        if sensors.empty:
            nearest_sensor_name = "N/A"
            nearest_km = float("nan")
        else:
            dists = self._haversine_vec(
                sensors["latitude"].to_numpy(), sensors["longitude"].to_numpy(),
                latitude, longitude
            )
            nn = int(np.nanargmin(dists))
            nearest_sensor_name = str(sensors.iloc[nn]["sensor_name"])
            nearest_km = float(dists[nn])

        # zones & cluster via saved/reconstructed bins
        lat_zone = self._zone_from_bins(latitude, getattr(self, "lat_bins", None))
        lon_zone = self._zone_from_bins(longitude, getattr(self, "lon_bins", None))
        spatial_cluster = lat_zone * 10 + lon_zone

        # build feature vector expected by the trained model
        feat = {
            # location features
            "latitude": float(latitude),
            "longitude": float(longitude),
            "lat_zone": lat_zone,
            "lon_zone": lon_zone,
            "spatial_cluster": spatial_cluster,
            "dist_to_cbd": float(self._haversine_vec([latitude], [longitude], -37.8136, 144.9631)[0]),
            # temporal features
            "hour": hour,
            "dow": dow,
            "is_weekend": is_weekend,
            "hour_sin": np.sin(2*np.pi*hour/24.0),
            "hour_cos": np.cos(2*np.pi*hour/24.0),
            "dow_sin": np.sin(2*np.pi*dow/7.0),
            "dow_cos": np.cos(2*np.pi*dow/7.0),
            "month_sin": np.sin(2*np.pi*ts.month/12.0),
            "month_cos": np.cos(2*np.pi*ts.month/12.0),
            "is_business_hours": int((hour >= 9) and (hour <= 17) and (not bool(is_weekend))),
            "is_peak_hour": int((hour in [7,8,9,17,18,19]) and (not bool(is_weekend))),
            # directionals (not known at predict time → zeros)
            "dir1": 0.0, "dir2": 0.0, "total_dir": 0.0, "dir_ratio": 0.5,
            # lag/rolling features not available out-of-sample → zeros
            "count_lag_1h": 0.0, "count_lag_24h": 0.0, "count_lag_168h": 0.0,
            "count_roll_mean_24h": 0.0, "count_roll_std_24h": 0.0,
            "count_roll_mean_168h": 0.0, "count_roll_std_168h": 0.0,
            # per-location stats unknown for arbitrary coords → zeros
            "count_mean": 0.0, "count_std": 0.0, "count_median": 0.0,
            "count_min": 0.0, "count_max": 0.0,
            # date-derived (if your model included them)
            "year": ts.year, "month": ts.month, "day": ts.day,
            "dayofyear": ts.dayofyear, "weekofyear": int(ts.isocalendar()[1]),
        }

        # make sure all trained features exist in the right order
        X = np.array([[feat.get(c, 0) for c in self.feature_cols]], dtype="float32")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # predict
        if self.model_type == "lightgbm":
            y = float(self.model.predict(X, num_iteration=getattr(self.model, "best_iteration", None))[0])
        else:
            import xgboost as xgb
            y = float(self.model.predict(xgb.DMatrix(X))[0])

        y = max(0.0, y)
        return {
            "predicted_count": int(round(y)),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "nearest_sensor": nearest_sensor_name,
            "distance_to_nearest_sensor_km": round(nearest_km, 2) if np.isfinite(nearest_km) else None,
            "ci_lower": int(max(0, y * 0.8)),
            "ci_upper": int(y * 1.2),
        }

    # ---------- batch predict from CSV ----------
    def predict_csv(self, csv_path, out_path=None, lat_col="latitude", lon_col="longitude", ts_col="timestamp"):
        df = pd.read_csv(csv_path)
        out = []
        for i, row in df.iterrows():
            res = self.predict_location(row[lat_col], row[lon_col], row[ts_col])
            out.append(res)
        out_df = pd.DataFrame(out)
        if out_path:
            out_df.to_csv(out_path, index=False)
        return out_df


def main():
    ap = argparse.ArgumentParser(description="Test pedestrian count model")
    ap.add_argument("--model", default="pedestrian_count_model.pkl", help="Path to saved model")
    ap.add_argument("--lat", type=float, help="Latitude for single prediction")
    ap.add_argument("--lon", type=float, help="Longitude for single prediction")
    ap.add_argument("--time", type=str, help="Timestamp (e.g., 2024-03-15 14:00:00)")
    ap.add_argument("--csv", type=str, help="Optional: CSV file with columns latitude,longitude,timestamp")
    ap.add_argument("--out", type=str, help="Optional: path to save batch predictions CSV")
    args = ap.parse_args()

    predictor = MiniPedestrianPredictor(args.model)

    if args.csv:
        out_df = predictor.predict_csv(args.csv, out_path=args.out)
        print(out_df.head().to_string(index=False))
        if args.out:
            print(f"\nSaved predictions to {args.out}")
    elif args.lat is not None and args.lon is not None and args.time:
        res = predictor.predict_location(args.lat, args.lon, args.time)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        # demo
        demo = predictor.predict_location(-37.814, 144.963, "2024-03-15 14:00:00")
        print(json.dumps(demo, ensure_ascii=False, indent=2))
        print("\nTip: pass --lat --lon --time for your own point, or --csv file.csv for batch.")
        
if __name__ == "__main__":
    main()
