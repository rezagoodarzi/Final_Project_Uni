# -*- coding: utf-8 -*-
"""
Melbourne Pedestrian Counting — join attachment + monthly table + locations
Outputs a single time series with lat/lon (and optional H3 r9) ready for GeoAI/ML.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytz
from dateutil import parser

# -------- settings --------
DATA = Path("data")
RAW  = DATA
OUT  = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

USE_H3 = True           # set False if you don't want H3 r9
H3_RES = 9
BOUNDARY_DATE_LOCAL = "2022-12-14 23:59:59"   # Attachment covers up to this (inclusive), local time

ATTACHMENT_FILE = RAW / "Pedestrian_Counting_System_Monthly_counts_per_hour_may_2009_to_14_dec_2022.csv"
MONTHLY_FILE    = RAW / "pedestrian-counting-system-monthly-counts-per-hour.csv"
LOCATIONS_FILE  = RAW / "pedestrian-counting-system-sensor-locations.csv"

MEL_TZ = pytz.timezone("Australia/Melbourne")

# -------- helpers --------
def to_local_and_utc(dt_series_naive_local):
    """Localize a naive Melbourne time series and produce both local & UTC."""
    dt_local = pd.to_datetime(dt_series_naive_local, errors="coerce", infer_datetime_format=True)
    # Localize to Melbourne; handle DST gaps/ambiguity as NaT (we won't force-fill here).
    dt_local = dt_local.dt.tz_localize(MEL_TZ, nonexistent="NaT", ambiguous="NaT")
    dt_utc = dt_local.dt.tz_convert("UTC")
    return dt_local, dt_utc

def build_ts_from_date_and_hour(date_series, hour_series):
    """Combine Sensing_Date (date-like) + HourDay (hour label/number) into a tz-aware Melbourne timestamp."""
    d = pd.to_datetime(date_series, errors="coerce")
    # HourDay can be int (0..23) or strings like '0', '00:00', '12AM'
    hr_raw = hour_series.astype(str).str.strip().str.lower()
    # Try numeric first, else parse first 2 digits, else fallback to pandas to_datetime(time)
    hr_num = pd.to_numeric(hr_raw, errors="coerce")
    # If still NaN, try to regex extract hour from 'hh:mm' or 'h am/pm'
    mask_nan = hr_num.isna()
    if mask_nan.any():
        # Extract leading digits
        guess = hr_raw.str.extract(r'(\d{1,2})', expand=False)
        hr_num2 = pd.to_numeric(guess, errors="coerce")
        hr_num = hr_num.fillna(hr_num2)

    # Coerce to integer 0..23 and clip
    hr = hr_num.fillna(0).astype(int).clip(0, 23)
    # Combine
    dt_naive = d + pd.to_timedelta(hr, unit="h")
    return to_local_and_utc(dt_naive)

def safe_int_like(s):
    try:
        return s.astype("Int64")
    except Exception:
        return pd.to_numeric(s, errors="coerce").astype("Int64")

# -------- 1) Locations --------
locs = pd.read_csv(LOCATIONS_FILE)
# Normalize key columns
# Location_ID, Sensor_Description, Sensor_Name, Latitude, Longitude, ...
col = {c.lower(): c for c in locs.columns}

locs = locs.rename(columns={
    col.get("location_id", "Location_ID"): "location_id",
    col.get("sensor_name", "Sensor_Name"): "sensor_name",
    col.get("latitude", "Latitude"): "latitude",
    col.get("longitude", "Longitude"): "longitude",
})
# Keep essentials + handy metadata
keep_locs = ["location_id", "sensor_name", "latitude", "longitude"]
extra_locs = [c for c in ["Status", "Location_Type", "Installation_Date", "Note", "Direction_1", "Direction_2"] if c in locs.columns]
locs = locs[keep_locs + extra_locs].dropna(subset=["location_id", "latitude", "longitude"]).copy()

# Ensure types
locs["location_id"] = locs["location_id"].astype(str).str.strip()
locs["sensor_name"] = locs["sensor_name"].astype(str).str.strip()

# Build name→id map (helps resolve Sensor_ID in the attachment when needed)
name_to_id = (locs.dropna(subset=["sensor_name", "location_id"])
                   .drop_duplicates(subset=["sensor_name"])
                   .set_index("sensor_name")["location_id"]
                   .to_dict())

# -------- 2) Attachment (≤ 2022-12-14) --------
att = pd.read_csv(ATTACHMENT_FILE)

# Normalize columns: ID, Date_Time, Year, Month, Mdate, Day, Time, Sensor_ID, Sensor_Name, Hourly_Counts
cols = {c.lower(): c for c in att.columns}
att = att.rename(columns={
    cols.get("sensor_id", "Sensor_ID"): "sensor_id_raw",
    cols.get("sensor_name", "Sensor_Name"): "sensor_name",
    cols.get("date_time", "Date_Time"): "date_time",
    cols.get("hourly_counts", "Hourly_Counts"): "count",
})

# Build timestamps
att_local, att_utc = to_local_and_utc(att["date_time"])
att["timestamp_local"] = att_local
att["timestamp_utc"]   = att_utc

# Resolve a unified location_id:
# 1) If sensor_id_raw matches a Location_ID in locs, use it.
# 2) Else map by sensor_name via name_to_id (fallback).
att["sensor_id_raw"] = att["sensor_id_raw"].astype(str).str.strip()
att["sensor_name"]   = att["sensor_name"].astype(str).str.strip()

# Try direct match (some datasets use same numeric IDs)
direct_ids = set(locs["location_id"].unique())
att["location_id"] = np.where(att["sensor_id_raw"].isin(direct_ids), att["sensor_id_raw"], np.nan)

# Fallback by name where location_id is missing
mask_missing = att["location_id"].isna() & att["sensor_name"].notna()
att.loc[mask_missing, "location_id"] = att.loc[mask_missing, "sensor_name"].map(name_to_id)

# Keep essentials
att = att[["location_id", "sensor_name", "timestamp_local", "timestamp_utc", "count"]].dropna(subset=["timestamp_utc", "location_id"])

# Limit to boundary (inclusive)
boundary_local = pd.Timestamp(BOUNDARY_DATE_LOCAL).tz_localize(MEL_TZ)
att = att[att["timestamp_local"] <= boundary_local].copy()

# Clean counts
att["count"] = pd.to_numeric(att["count"], errors="coerce")
att = att.dropna(subset=["count"])
att = att[att["count"] >= 0]

# -------- 3) Monthly table (≥ 2022-12-15) --------
mon = pd.read_csv(MONTHLY_FILE)

# Columns: ID, Location_ID, Sensing_Date, HourDay, Direction_1, Direction_2, Total_of_Directions, Sensor_Name, Location
cols = {c.lower(): c for c in mon.columns}
mon = mon.rename(columns={
    cols.get("location_id", "Location_ID"): "location_id",
    cols.get("sensing_date", "Sensing_Date"): "sensing_date",
    cols.get("hourday", "HourDay"): "hourday",
    cols.get("direction_1", "Direction_1"): "dir1",
    cols.get("direction_2", "Direction_2"): "dir2",
    cols.get("total_of_directions", "Total_of_Directions"): "total",
    cols.get("sensor_name", "Sensor_Name"): "sensor_name",
})

# Timestamp from Sensing_Date + HourDay (local)
mon_local, mon_utc = build_ts_from_date_and_hour(mon["sensing_date"], mon["hourday"])
mon["timestamp_local"] = mon_local
mon["timestamp_utc"]   = mon_utc

# Counts: prefer 'total', else sum dir1+dir2
for c in ["dir1", "dir2", "total"]:
    if c in mon.columns:
        mon[c] = pd.to_numeric(mon[c], errors="coerce")

if "total" in mon.columns and mon["total"].notna().any():
    mon["count"] = mon["total"]
else:
    mon["count"] = mon[["dir1", "dir2"]].fillna(0).sum(axis=1)

# Clean + limit to after boundary (strictly greater)
mon["location_id"] = mon["location_id"].astype(str).str.strip()
mon = mon.dropna(subset=["location_id", "timestamp_utc"])
mon = mon[mon["timestamp_local"] > boundary_local].copy()
mon = mon[(mon["count"].notna()) & (mon["count"] >= 0)]

# Keep essentials (+ keep dir splits as optional features)
keep_mon = ["location_id", "sensor_name", "timestamp_local", "timestamp_utc", "count"]
for extra in ["dir1", "dir2"]:
    if extra in mon.columns:
        keep_mon.append(extra)
mon = mon[keep_mon]

# -------- 4) Combine both periods --------
counts = pd.concat([att, mon], ignore_index=True)
# Drop any accidental dupes on (location_id, timestamp_utc)
counts = counts.sort_values(["location_id", "timestamp_utc"]).drop_duplicates(
    subset=["location_id", "timestamp_utc"], keep="last"
)

# -------- 5) Join Locations --------
df = counts.merge(locs[["location_id", "latitude", "longitude", "sensor_name"]].rename(columns={"sensor_name":"sensor_name_loc"}),
                  on="location_id", how="left")

# Prefer the name coming from counts, but fill from locations if missing
df["sensor_name"] = df["sensor_name"].fillna(df["sensor_name_loc"])
df = df.drop(columns=["sensor_name_loc"])

# Flag missing coords (should be rare)
df["has_coords"] = df["latitude"].notna() & df["longitude"].notna()

# -------- 6) Optional H3 --------
if USE_H3:
    import h3
    def _to_h3(lat, lon, res):
        try:
            return h3.geo_to_h3(lat, lon, res)
        except Exception:
            return np.nan
    df["h3_r9"] = np.where(
        df["has_coords"],
        [ _to_h3(lat, lon, H3_RES) for lat, lon in zip(df["latitude"], df["longitude"]) ],
        np.nan
    )

# -------- 7) Time features (use LOCAL time for daily/weekly patterns) --------
df["hour"]        = df["timestamp_local"].dt.hour
df["dow"]         = df["timestamp_local"].dt.dayofweek  # Mon=0
df["is_weekend"]  = df["dow"].isin([5,6])
df["date"]        = df["timestamp_local"].dt.date

# -------- 8) Final clean + save --------
df = df.dropna(subset=["timestamp_utc", "location_id", "count"]).copy()
df = df.sort_values(["location_id", "timestamp_utc"]).reset_index(drop=True)

OUT_FILE_PARQUET = OUT / "melbourne_pedestrian_hourly_joined.csv"
OUT_FILE_SAMPLE  = OUT / "melbourne_sample_head.csv"

df.to_csv(OUT_FILE_PARQUET, index=False)
df.head(50).to_csv(OUT_FILE_SAMPLE, index=False)

print("✅ Done!")
print("Rows: ", len(df))
print("Sensors (location_id):", df["location_id"].nunique())
print("With coords:", df["has_coords"].mean().round(3))
if USE_H3:
    print("Unique H3 r9 hexes:", df["h3_r9"].nunique())
print("Date range (local):", df["timestamp_local"].min(), "→", df["timestamp_local"].max())
