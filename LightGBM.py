# pip install pandas pyarrow numpy scikit-learn lightgbm holidays
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import holidays
import math

DATA = Path("data/processed")
DF = pd.read_parquet(DATA/"melbourne_pedestrian_hourly_joined.parquet")

# ---- 1) ensure continuous hourly grid per sensor ----
def make_full_grid(df):
    df = df.copy()
    df = df.sort_values(["location_id","timestamp_local"])
    # full hourly index per sensor between its min/max
    rows = []
    for loc, g in df.groupby("location_id", sort=False):
        idx = pd.date_range(g["timestamp_local"].min(), g["timestamp_local"].max(), freq="H", tz=g["timestamp_local"].dt.tz)
        g2 = g.set_index("timestamp_local").reindex(idx)
        g2["location_id"] = loc
        # forward fill static cols
        for col in ["sensor_name","latitude","longitude"]:
            if col in g.columns:
                g2[col] = g[col].iloc[0]
        g2["timestamp_local"] = g2.index
        rows.append(g2.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    # counts: missing hour means no record; Melbourne feed usually puts 0 for none → fill with 0
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0)
    # recompute UTC from local (keep local for features)
    out["timestamp_utc"] = out["timestamp_local"].dt.tz_convert("UTC")
    return out

df = make_full_grid(DF)

# ---- 2) time features (local) ----
df["hour"] = df["timestamp_local"].dt.hour
df["dow"]  = df["timestamp_local"].dt.dayofweek
df["dom"]  = df["timestamp_local"].dt.day
df["week"] = df["timestamp_local"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

# Cyclic encodings
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)

# Public holidays (Australia/VIC)
try:
    au_holidays = holidays.Australia(subdiv="VIC")
    df["is_holiday"] = df["timestamp_local"].dt.date.map(lambda d: int(d in au_holidays))
except Exception:
    df["is_holiday"] = 0  # if library unavailable, default to 0

# ---- 3) lag & rolling features per sensor ----
def add_lags(group, lags=(1,2,3,24,25,168), roll_windows=(3,6,24)):
    g = group.sort_values("timestamp_local").copy()
    for L in lags:
        g[f"lag_{L}"] = g["count"].shift(L)
    for w in roll_windows:
        g[f"rollmean_{w}h"] = g["count"].rolling(w, min_periods=1).mean().shift(1)
        g[f"rollstd_{w}h"]  = g["count"].rolling(w, min_periods=1).std().shift(1)
    # week-over-week delta
    g["wow_delta"] = g["count"].shift(0) - g["count"].shift(168)
    return g

df = df.groupby("location_id", group_keys=False).apply(add_lags)

# ---- 4) spatial neighbor features from lat/long (k-NN) ----
# Build neighbor graph once (static); use haversine distances
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

sensors = df.drop_duplicates("location_id")[["location_id","latitude","longitude"]].dropna()
coords = np.radians(sensors[["latitude","longitude"]].values)  # NearestNeighbors wants radians for haversine
nbrs = NearestNeighbors(n_neighbors=6, metric="haversine").fit(coords)  # 1 self + 5 neighbors
dist, idx = nbrs.kneighbors(coords)

# map location_id -> list of neighbor ids (exclude self index 0)
loc_ids = sensors["location_id"].values
neighbor_map = {loc_ids[i]: list(loc_ids[idx[i,1:]]) for i in range(len(loc_ids))}

# neighbor mean at previous hour
df["nbr_mean_lag1"] = np.nan
# To compute efficiently, create a dict of previous-hour counts per sensor
for loc, g in df.groupby("location_id", sort=False):
    df.loc[g.index, "lag1_tmp"] = g["count"].shift(1).values

# aggregate by timestamp across each sensor's neighbor set
# (simple loop is OK for tens of sensors; vectorization is possible if needed)
df["nbr_mean_lag1"] = 0.0
df["nbr_cnt"] = 0
df = df.sort_values(["timestamp_local","location_id"])
for ts, gts in df.groupby("timestamp_local"):
    lookup = gts.set_index("location_id")["lag1_tmp"].to_dict()
    for loc in gts["location_id"]:
        nbs = neighbor_map.get(loc, [])
        vals = [lookup.get(n) for n in nbs]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if vals:
            df.loc[(df["timestamp_local"]==ts) & (df["location_id"]==loc), "nbr_mean_lag1"] = float(np.mean(vals))
            df.loc[(df["timestamp_local"]==ts) & (df["location_id"]==loc), "nbr_cnt"] = len(vals)
df.drop(columns=["lag1_tmp","nbr_cnt"], inplace=True)

# ---- 5) final training table ----
# drop first max lag horizon to avoid NaNs
max_lag = 168
df_model = df[df["timestamp_local"] >= (df["timestamp_local"].min() + pd.Timedelta(hours=max_lag))].copy()

# target options:
# (A) raw counts with Tweedie/Poisson objective
y = df_model["count"].astype(float)

# features
feat_cols = [
    "hour","dow","dom","week","is_weekend","is_holiday",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "lag_1","lag_2","lag_3","lag_24","lag_25","lag_168",
    "rollmean_3h","rollmean_6h","rollmean_24h",
    "rollstd_3h","rollstd_6h","rollstd_24h",
    "wow_delta",
    "nbr_mean_lag1",
    "latitude","longitude",
]
# Categorical sensor id (lets the model learn per-sensor bias)
df_model["location_id"] = df_model["location_id"].astype("category")
feat_cols += ["location_id"]

X = df_model[feat_cols]

# ---- 6) time-based split (e.g., last 28 days as test) ----
split_time = df_model["timestamp_local"].max() - pd.Timedelta(days=28)
train_idx = df_model["timestamp_local"] <= split_time
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[~train_idx], y[~train_idx]

# ---- 7) train LightGBM (Tweedie) ----
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=["location_id"])
valid_set = lgb.Dataset(X_test,  label=y_test,  categorical_feature=["location_id"])

params = dict(
    objective="tweedie",   # good for non-negative, right-skewed counts
    tweedie_variance_power=1.2,  # 1–1.5 often works; tune on validation
    metric=["rmse","mae"],
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=100,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    max_depth=-1,
    verbose=-1,
)
model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, valid_set],
    valid_names=["train","valid"],
    early_stopping_rounds=200,
    verbose_eval=250,
)

# ---- 8) evaluate ----
pred = model.predict(X_test, num_iteration=model.best_iteration)
mae = mean_absolute_error(y_test, pred)
rmse = math.sqrt(mean_squared_error(y_test, pred))
wape = (np.abs(y_test - pred).sum() / (y_test.sum() + 1e-9))
print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  WAPE={wape:.3%}")

# (Optional) quantiles for uncertainty
# params_quant = {**params, "objective":"quantile", "alpha":0.9}  # train P90 model similarly
