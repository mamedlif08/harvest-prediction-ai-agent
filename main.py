"""
Harvest Prediction Agent (Google ADK + LiteLlm, Claude Sonnet 5)
----------------------------------------------------------------
Train on local files (e.g. the 2025 season), predict on raw MinIO data (e.g. 2026).

  python main.py
  You: cotton_geozones_az.csv climate_features_SYNTHETIC.csv vegetation_indices_SYNTHETIC.csv

Core rule: THE CODE PICKS THE MODEL, THE LLM EXPLAINS THE PICK.
A deterministic benchmark trains every shortlisted candidate (ridge, random forest,
XGBoost) on an identical split, ranks them by held-out MAE in days and runs automatic
checks. The LLM only reads the result; register_model refuses anything the benchmark
did not select.

Workflow (each step is a tool; the order is enforced in code):
  1. prepare_training_data -> loads + merges the local files, target = days_to_harvest
                              (harvest date - sowing date), rebuilds every feature from the
                              raw MinIO observations and drops what cannot be rebuilt
  2. profile_dataset       -> 6 signals + model shortlist
  3. make_splits           -> newest season = test (season-out), CV grouped by geozone
  4. baseline              -> GDD rule (or calendar rule), MAE in days
  5. run_benchmark         -> ranked comparison table + checks + gate
  6. explain               -> grouped permutation importance of the winner
  7. register_model        -> only the benchmark winner, only if the gate passed
  8. predict_from_minio    -> harvest date per geozone, sowing date shifted -5..+5 days (averaged)

Outputs: predictions_minio.csv (geozone_id, sowing_date, days_to_harvest),
feature_importance.csv, benchmark_<run_id>.json/.md, models/*.joblib

Run without the LLM (deterministic, same tools):
  python main.py --no-llm cotton_geozones_az.csv climate_features_SYNTHETIC.csv vegetation_indices_SYNTHETIC.csv
  python main.py --no-llm --no-minio ...   # benchmark only, MinIO is not read

.env: ANTHROPIC_API_KEY, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
Optional: MINIO_SKIP_PARTS (default "pixels,_state"), MINIO_CHUNK_SIZE, HARVEST_LOG_FILE,
          HARVEST_GATE_MAE_DAYS (default 4.0), MODEL_DIR (default "models")
"""

import io
import os
import json
import posixpath
import re
import warnings
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv(override=True)  # .env wins over stale system-level variables

# LiteLLM's background logging worker prints harmless TimeoutError tracebacks — keep the console clean
import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.CRITICAL)
os.environ.setdefault("LITELLM_LOG", "CRITICAL")

MODEL = "claude-sonnet-5"
MODEL_DIR = os.getenv("MODEL_DIR", "models")
REPORT_TITLE = os.getenv("HARVEST_REPORT_TITLE", "Harvest Forecast Report — Cotton, Azerbaijan")
RANDOM_STATE = 42

STATE = {
    "files": {},             # alias -> pd.DataFrame (all uploaded files "as is")
    "primary_alias": None,   # alias of the primary (target-containing) file
    "merged_aliases": [],    # which other files have already been merged into the working df
    "minio_alias": None,
    "df": None,               # pd.DataFrame — the WORKING dataframe (after all merges)
    "target": None,           # name of the target column (days_to_harvest)
    "id_like_cols": [],       # columns that look like ids (not features, just keys)
    "feature_cols": [],       # final feature list of the registered model
    "feature_medians": {},    # medians of training features, reused when predicting on MinIO data
    "dropped_cols": {},
    "engineered": {},
    "feature_sources": {},    # {model_feature: source_column}
    "model": None,
    "model_name": None,
    "task_type": "regression",
    "metrics": {},
    "requested_feature_sources": None,
    # benchmark workflow
    "prep": None,             # result of prepare_training_data (features, MinIO matrix, report)
    "profile": None,
    "splits": None,
    "baseline": None,
    "runs": {},
    "explained": {},          # run_id -> model name explain() was run on
    "registered": None,
}



def _looks_like_id(series: pd.Series, col_name: str) -> bool:
    name_hits = any(tok in col_name.lower() for tok in ["id", "uuid", "code"])
    uniqueness = series.nunique(dropna=True) / max(len(series), 1)
    return name_hits and uniqueness > 0.9


def _looks_like_geometry(series: pd.Series) -> bool:
    sample = series.dropna().head(5)
    for v in sample:
        if isinstance(v, bytes):
            return True
        if isinstance(v, dict) and ("wkb" in v or "wkt" in v):
            return True
        if isinstance(v, str) and v.strip().upper().startswith(("POLYGON", "MULTIPOLYGON", "POINT")):
            return True
    return False


def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(10)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def _extract_polygon(geom_field):
    from shapely import wkb, wkt
    if geom_field is None:
        return None
    try:
        if isinstance(geom_field, dict):
            if "wkb" in geom_field:
                return wkb.loads(geom_field["wkb"])
            elif "wkt" in geom_field:
                return wkt.loads(geom_field["wkt"])
        elif isinstance(geom_field, bytes):
            return wkb.loads(geom_field)
        elif isinstance(geom_field, str):
            return wkt.loads(geom_field)
    except Exception:
        pass
    return None


def _profile_df(df: pd.DataFrame) -> list:
    profile = []
    for c in df.columns:
        profile.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "null_pct": round(df[c].isna().mean() * 100, 1),
            "n_unique": int(df[c].nunique(dropna=True)),
            "looks_like_id": _looks_like_id(df[c], c),
            "looks_like_geometry": _looks_like_geometry(df[c]),
            "looks_like_date": _looks_like_date(df[c]),
        })
    return profile


def load_file(path: str, alias: Optional[str] = None, as_primary: Optional[bool] = None) -> dict:
    """Loads a CSV or Excel file and returns its profile: columns, dtypes,
    null %, row count, and a few sample rows.

    Call this for EVERY local file the user provides — including additional
    "feature" files, not just the primary one with the target. (Data that
    lives in MinIO is loaded with load_minio_folder instead.)

    Args:
        path: path to the file (.csv, .xlsx, .xls)
        alias: short name for the file (e.g. "weather", "soil"). If not
            given — derived from the file name without extension. Used
            later in merge_feature_file to refer to this file.
        as_primary: whether this file should be treated as the PRIMARY one
            (the one holding the target, which becomes the working
            dataframe STATE["df"]). If not given explicitly: True for the
            very first file loaded in the session, False for all
            subsequent ones (they are simply registered and wait for
            merge_feature_file). If the user explicitly says "this is the
            primary file" / "this has the target" — pass True.
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if not alias:
        alias = os.path.splitext(os.path.basename(path))[0]

    if alias in STATE["files"]:
        return {"error": f"Alias '{alias}' is already taken by another loaded file. Use a different alias."}

    STATE["files"][alias] = df

    is_primary = as_primary if as_primary is not None else (STATE["primary_alias"] is None)

    result = {
        "alias": alias,
        "is_primary": is_primary,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns_profile": _profile_df(df),
        "sample_rows": json.loads(df.head(3).to_json(orient="records")),
        "already_loaded_files": {a: list(d.columns) for a, d in STATE["files"].items()},
    }

    if is_primary:
        STATE["primary_alias"] = alias
        STATE["merged_aliases"] = []
        STATE["df"] = df.copy()
        STATE["target"] = None
        STATE["id_like_cols"] = [c for c in df.columns if _looks_like_id(df[c], c)]
        STATE["feature_cols"] = []
        STATE["feature_medians"] = {}
        STATE["dropped_cols"] = {}
        STATE["engineered"] = {}
        STATE["model"] = None
        STATE["metrics"] = {}
        STATE["requested_feature_sources"] = None
        result["note"] = "This file became the PRIMARY (working) one. Next, ask for the target, and if there aren't enough features, check whether there are more feature files for merge_feature_file."
    else:
        result["note"] = (
            f"File loaded under alias '{alias}', but did NOT become the working dataframe "
            f"(the primary one is already '{STATE['primary_alias']}'). To use its "
            f"columns as features, call merge_feature_file(alias='{alias}', ...)."
        )

    return result


def list_loaded_files() -> dict:
    """Shows all files loaded so far (aliases, columns, row counts), which
    one is primary (contains the target), and which have already been
    merged into the working dataframe. Call this if it's unclear what has
    already been loaded, or before deciding where to source features from.
    """
    if not STATE["files"]:
        return {"error": "No files loaded yet"}

    return {
        "primary_alias": STATE["primary_alias"],
        "merged_aliases": STATE["merged_aliases"],
        "minio_alias": STATE["minio_alias"],
        "working_df_columns": list(STATE["df"].columns) if STATE["df"] is not None else [],
        "files": {
            alias: {
                "n_rows": len(df),
                "columns": list(df.columns),
                "merged_already": alias in STATE["merged_aliases"] or alias == STATE["primary_alias"],
            }
            for alias, df in STATE["files"].items()
        },
    }


def merge_feature_file(alias: str, join_key_main: str, join_key_file: Optional[str] = None, how: str = "left") -> dict:
    """Joins columns from a previously loaded file (by alias) into the
    working dataframe on a shared key (e.g. geozone_id). Afterwards, the
    columns from the merged file become available as regular columns of the
    working df — usable with engineer_date_column / engineer_geometry_column
    / one_hot_encode_column / train_model, just like columns of the primary
    file.

    Call this as many times as needed to attach additional feature files.

    Args:
        alias: alias of the file (from load_file) whose columns should be merged in
        join_key_main: name of the key column in the WORKING dataframe (e.g. geozone_id)
        join_key_file: name of the key column in the file alias, if different
            from join_key_main. If not given — the same name is used.
        how: join type — "left" (default, keep all rows of the primary
            file), "inner" (only matching ids), "right"
    """
    if STATE["df"] is None:
        return {"error": "The primary file must be loaded first (load_file)"}
    if alias not in STATE["files"]:
        return {"error": f"No loaded file with alias '{alias}'. Available: {list(STATE['files'].keys())}"}
    if alias == STATE["primary_alias"]:
        return {"error": f"'{alias}' is already the primary file, no need to merge it into itself"}
    if alias in STATE["merged_aliases"]:
        return {"error": f"'{alias}' has already been merged into the working df"}

    join_key_file = join_key_file or join_key_main
    main_df = STATE["df"]
    feature_df = STATE["files"][alias]

    if join_key_main not in main_df.columns:
        return {"error": f"Key column '{join_key_main}' is not in the working df. Available: {list(main_df.columns)}"}
    if join_key_file not in feature_df.columns:
        return {"error": f"Key column '{join_key_file}' is not in file '{alias}'. Available: {list(feature_df.columns)}"}

    before_rows = len(main_df)
    before_cols = set(main_df.columns)

    merged = main_df.merge(
        feature_df,
        left_on=join_key_main,
        right_on=join_key_file,
        how=how,
        suffixes=("", f"__{alias}"),
    )

    # if the key was duplicated under a different name (join_key_file != join_key_main),
    # the duplicate key column from the right file can be dropped, it's no longer needed
    if join_key_file != join_key_main and join_key_file in merged.columns:
        merged = merged.drop(columns=[join_key_file])

    new_cols = [c for c in merged.columns if c not in before_cols]
    unmatched = int(merged[new_cols[0]].isna().sum()) if new_cols else 0

    STATE["df"] = merged
    STATE["merged_aliases"].append(alias)

    return {
        "merged_alias": alias,
        "how": how,
        "rows_before": before_rows,
        "rows_after": len(merged),
        "new_columns": new_cols,
        "rows_without_match_in_feature_file": unmatched,
        "warning": (
            f"{unmatched} rows found no match in '{alias}' by key — these rows will "
            f"have NaN in the new columns. They will be filled with the median in "
            f"train_model; flag the unmatched count in the final summary."
            if unmatched else None
        ),
    }


def set_target(column: str) -> dict:
    """Sets the target column — the number the models predict (days_to_harvest).

    Args:
        column: exact name of the target column (must be numeric)
    """
    df = STATE["df"]
    if df is None:
        return {"error": "Call load_file first"}
    if column not in df.columns:
        return {"error": f"Column '{column}' is not in the working df. Available: {list(df.columns)}"}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Target '{column}' must be numeric (days), got {df[column].dtype}"}

    STATE["target"] = column
    STATE["task_type"] = "regression"
    return {
        "target": column,
        "task_type": "regression",
        "rows_with_target": int(df[column].notna().sum()),
        "rows_to_predict": int(df[column].isna().sum()),
    }



def engineer_geometry_column(column: str) -> dict:
    """Turns a geometry column (WKT/WKB/dict) into 3 numeric features:
    <col>_centroid_x, <col>_centroid_y, <col>_area. Adds them to the
    dataframe and registers them as engineered features.

    Args:
        column: name of the geometry column
    """
    df = STATE["df"]
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    parsed = df[column].apply(_extract_polygon)
    cx, cy, area = [], [], []
    for poly in parsed:
        if poly is not None and not poly.is_empty:
            c = poly.centroid
            cx.append(c.x); cy.append(c.y); area.append(poly.area)
        else:
            cx.append(0.0); cy.append(0.0); area.append(0.0)

    df[f"{column}_centroid_x"] = cx
    df[f"{column}_centroid_y"] = cy
    df[f"{column}_area"] = area
    STATE["engineered"][column] = "geometry"
    for _c in (f"{column}_centroid_x", f"{column}_centroid_y", f"{column}_area"):
        STATE["feature_sources"][_c] = column

    new_cols = [f"{column}_centroid_x", f"{column}_centroid_y", f"{column}_area"]
    return {"created_columns": new_cols, "note": "The original geometry column itself is NOT a feature, use only the 3 new columns"}


def engineer_date_column(column: str) -> dict:
    """Turns a date column into numeric features: <col>_dayofyear, <col>_month.
    The date itself cannot be used as a feature (not numeric).

    IMPORTANT: if this is the TARGET date column — after calling this
    function you must call set_target("<column>_dayofyear") so the target
    becomes numeric instead of a date string.

    Args:
        column: name of the date column
    """
    df = STATE["df"]
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    parsed = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_dayofyear"] = parsed.dt.dayofyear
    df[f"{column}_month"] = parsed.dt.month
    STATE["engineered"][column] = "date"
    STATE["feature_sources"][f"{column}_dayofyear"] = column
    STATE["feature_sources"][f"{column}_month"] = column

    new_cols = [f"{column}_dayofyear", f"{column}_month"]
    result = {"created_columns": new_cols}
    if STATE["target"] == column:
        result["warning"] = (
            f"'{column}' is the current target, and it's a date string, not a number! "
            f"Be sure to call set_target('{column}_dayofyear') as the next step, "
            f"otherwise train_model will refuse to run."
        )
    return result


def one_hot_encode_column(column: str) -> dict:
    """One-hot encodes a categorical column (for low cardinality, roughly
    up to 15-20 unique values). Creates columns like <col>_<value>.

    WARNING: if the user restricted the feature sources (see
    register_requested_feature_sources) and this column is NOT among them,
    first ask the user whether they want it included in the model, and only
    call this tool after they agree.

    Args:
        column: name of the categorical column
    """
    df = STATE["df"]
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    n_unique = df[column].nunique(dropna=True)
    dummies = pd.get_dummies(df[column], prefix=column)
    STATE["df"] = pd.concat([df, dummies], axis=1)
    STATE["engineered"][column] = "onehot"
    for _c in dummies.columns:
        STATE["feature_sources"][_c] = column

    return {
        "n_unique_values": int(n_unique),
        "created_columns": list(dummies.columns),
        "warning": "If n_unique_values > ~20, one-hot may not be optimal — consider frequency encoding" if n_unique > 20 else None,
    }


def _get_minio_client():
    from minio import Minio
    endpoint = (os.getenv("MINIO_ENDPOINT") or "").replace("https://", "").replace("http://", "").strip().strip("/")
    if not endpoint:
        raise RuntimeError("MINIO_ENDPOINT is not set in .env")

    def _clean(value):
        return (value or "").strip().strip('"').strip("'")

    return Minio(
        endpoint,
        access_key=_clean(os.getenv("MINIO_ACCESS_KEY")),
        secret_key=_clean(os.getenv("MINIO_SECRET_KEY")),
        secure=os.getenv("MINIO_SECURE", "false").strip().lower() == "true",
    )



# ---------------------------------------------------------------------------
# MinIO pipeline
#   - only files that really contain a needed column are read (and only those columns)
#   - per-pixel folders and internal _state copies are skipped by default
#   - every feature is rebuilt from raw daily observations relative to the sowing date
# ---------------------------------------------------------------------------
import logging
import time
import traceback

MINIO_BUCKET = "sandbox"
MINIO_PREFIXES = ["yieldprediction/", "yieldprediction_batch2/"]  # every folder is searched for geozones

# Path segments that are NOT read. "pixels" = per-pixel rasters (huge, field-level data already exists),
# "_state" = internal copies of the same field-level tables. Set MINIO_SKIP_PARTS= (empty) to read everything.
MINIO_SKIP_PARTS = [p.strip() for p in os.getenv("MINIO_SKIP_PARTS", "pixels,_state").split(",") if p.strip()]
MINIO_CHUNK_SIZE = int(os.getenv("MINIO_CHUNK_SIZE", "200000"))
SCHEMA_SAMPLE_ROWS = int(os.getenv("SCHEMA_SAMPLE_ROWS", "300"))

HEAT_STRESS_TMAX_C = 35.0   # assumed definition of a "heat stress day": daily tmax above this value
MAX_NAN_SHARE = 0.5         # a feature that cannot be computed for more than this share of geozones is dropped

LOG_FILE = os.getenv("HARVEST_LOG_FILE", "harvest_pipeline.log")
logger = logging.getLogger("harvest_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for _h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        _h.setFormatter(_fmt)
        logger.addHandler(_h)


def _log(message: str):
    logger.info(message)


RAW_ALIASES = {
    "tmax": ["tmax", "temperature_max", "max_temp", "temp_max", "t_max", "tmax_c"],
    "tmin": ["tmin", "temperature_min", "min_temp", "temp_min", "t_min", "tmin_c"],
    "precip": ["precipitation", "precip", "rainfall", "rain", "prcp", "precipitation_mm"],
    "gdd": ["gdd", "daily_gdd", "gdd_daily", "gdd_from_tmax_tmin"],
    "cgdd": ["cgdd", "cum_gdd", "cumulative_gdd", "gdd_cum", "gdd_cumulative"],
    "ndvi": ["ndvi", "ndvi_mean", "ndvi_value"],
    "evi": ["evi", "evi_mean", "evi_value"],
    "bsi": ["bsi", "bsi_mean", "bsi_value"],
    "ndwi": ["ndwi", "ndwi_mean", "ndwi_value"],
}
VAR_TOKENS = {
    "tmax": "tmax", "tmin": "tmin",
    "precip": "precip", "precipitation": "precip", "rain": "precip", "rainfall": "precip",
    "gdd": "gdd", "ndvi": "ndvi", "evi": "evi", "bsi": "bsi", "ndwi": "ndwi",
}
SOWING_TOKENS = {"sow", "sowing", "plant", "planting", "seed", "seeding"}
HARVEST_TOKENS = {"harvest", "end", "finish", "yield"}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _find_col(columns, names):
    normed = {_norm(c): c for c in columns if not str(c).startswith("_")}
    for name in names:
        if _norm(name) in normed:
            return normed[_norm(name)]
    return None


def _cat_norm(value) -> str:
    """'101.0' -> '101' so category codes and geozone ids compare equal across files."""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def _days(ts) -> int:
    return int((pd.Timestamp(ts).normalize() - pd.Timestamp("1970-01-01")).days)


def _date_like_columns(df: pd.DataFrame) -> list:
    out = []
    for c in df.columns:
        if str(c).startswith("_") or not any(k in _norm(c) for k in ("date", "time")):
            continue
        sample = df[c].dropna().head(2000)
        if sample.empty:
            continue
        if pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.5:
            out.append(c)
    return out


def _id_col(columns) -> Optional[str]:
    for c in columns:
        low = str(c).lower()
        if "id" in low and ("geozone" in low or "zone" in low):
            return c
    return None


# ---------------------------------------------------------------------------
# MinIO access
# ---------------------------------------------------------------------------

def _list_minio_csvs(bucket: str, prefixes: list):
    client = _get_minio_client()
    kept, skipped = [], 0
    skip = set(MINIO_SKIP_PARTS)
    for prefix in prefixes:
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            name = obj.object_name
            if not name.lower().endswith(".csv"):
                continue
            if set(name.split("/")) & skip:
                skipped += 1
                continue
            kept.append((prefix, name))
    return kept, skipped


def _read_schema_samples(bucket: str, objects: list):
    """Reads the first rows of every kept file: column names, dtypes and a small sample."""
    client = _get_minio_client()
    file_cols, frames, failed = {}, [], {}
    for i, (prefix, name) in enumerate(objects, start=1):
        resp = None
        try:
            resp = client.get_object(bucket, name)
            df = pd.read_csv(resp, encoding="utf-8-sig", nrows=SCHEMA_SAMPLE_ROWS)
            file_cols[name] = list(df.columns)
            idc = _id_col(df.columns)
            ids = df[idc] if idc else pd.Series(posixpath.basename(posixpath.dirname(name)), index=df.index)
            df["_minio_id"] = ids.map(_cat_norm)
            frames.append(df)
            _log(f"[SCHEMA {i}/{len(objects)}] {name} | columns={len(df.columns)}")
        except Exception as exc:
            failed[name] = str(exc)
            logger.warning("[SCHEMA] cannot read %s | %s", name, exc)
        finally:
            if resp is not None:
                try:
                    resp.close()
                    resp.release_conn()
                except Exception:
                    pass
    if not frames:
        raise RuntimeError(f"No readable MinIO CSV files. Failed: {failed}")
    return file_cols, pd.concat(frames, ignore_index=True, sort=False), failed


def _load_needed(bucket: str, objects: list, file_cols: dict, value_like: set, needed_all: set):
    """Reads ONLY the files that contain a needed value column, and ONLY the needed columns."""
    client = _get_minio_client()
    frames, used_files, failed = [], [], {}
    started = time.time()
    to_read = [(p, n) for p, n in objects if n in file_cols and (set(file_cols[n]) & value_like)]
    _log(f"[LOAD] {len(to_read)} of {len(objects)} files contain needed columns")

    for i, (prefix, name) in enumerate(to_read, start=1):
        cols = file_cols[name]
        use = [c for c in cols if c in needed_all]
        idc = _id_col(cols)
        usecols = list(dict.fromkeys(use + ([idc] if idc else [])))
        value_in_file = [c for c in cols if c in value_like]
        rows = 0
        resp = None
        try:
            resp = client.get_object(bucket, name)
            for chunk in pd.read_csv(resp, encoding="utf-8-sig", usecols=usecols, chunksize=MINIO_CHUNK_SIZE):
                ids = chunk[idc] if idc else pd.Series(posixpath.basename(posixpath.dirname(name)), index=chunk.index)
                chunk["_minio_id"] = ids.map(_cat_norm)
                if idc and idc not in use:
                    chunk = chunk.drop(columns=[idc])
                chunk = chunk.dropna(subset=value_in_file, how="all")
                chunk["_minio_batch"] = prefix
                rows += len(chunk)
                frames.append(chunk)
            used_files.append(name)
        except Exception as exc:
            failed[name] = str(exc)
            logger.warning("[LOAD] failed %s | %s", name, exc)
        finally:
            if resp is not None:
                try:
                    resp.close()
                    resp.release_conn()
                except Exception:
                    pass
        _log(f"[LOAD {i}/{len(to_read)}] {name} | rows kept={rows:,} | columns={usecols} | elapsed={time.time() - started:.1f}s")

    if not frames:
        raise RuntimeError("None of the MinIO files contained the needed columns.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    zones_by_batch = {p: set(raw.loc[raw["_minio_batch"] == p, "_minio_id"]) for p in MINIO_PREFIXES}
    data_cols = [c for c in raw.columns if c != "_minio_batch"]
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=data_cols).reset_index(drop=True)
    return raw, used_files, failed, zones_by_batch, n_before - len(raw)


def _pick_minio_sowing(sample: pd.DataFrame, date_cols: list) -> Optional[str]:
    """A column named like sowing/planting/seeding wins. Otherwise: the date column that is constant per geozone."""
    named = [c for c in date_cols if set(_norm(c).split("_")) & SOWING_TOKENS]
    if named:
        return named[0]
    best, best_score = None, 1.5
    for c in date_cols:
        score = sample.groupby("_minio_id")[c].nunique().mean()
        if score <= best_score:
            best, best_score = c, score
    return best


def _ts_date_col(sample: pd.DataFrame, source: str, date_cols: list) -> Optional[str]:
    """Observation-date column that belongs to the rows where `source` has values."""
    sub = sample[sample[source].notna()]
    best, best_score = None, 1.0
    for c in date_cols:
        parsed = pd.to_datetime(sub[c], errors="coerce")
        if parsed.notna().mean() < 0.5:
            continue
        score = parsed.groupby(sub["_minio_id"]).nunique().mean()
        if score > best_score:
            best, best_score = c, score
    return best


# ---------------------------------------------------------------------------
# Feature planning and reconstruction
# ---------------------------------------------------------------------------

def _plan_feature(feature: str, raw: pd.DataFrame, raw_cols: list, date_cols: list, train_sowing: str) -> dict:
    """Decides how one training feature is rebuilt from raw MinIO data (or why it cannot be)."""
    prov = STATE.get("feature_sources", {}).get(feature, feature)
    kind = STATE["engineered"].get(prov)
    f = _norm(feature)
    toks = f.split("_")

    if prov == train_sowing and (f.endswith("_dayofyear") or f.endswith("_month")):
        part = "dayofyear" if f.endswith("_dayofyear") else "month"
        return {"method": "sowing_date", "part": part, "raw_column": "sowing date", "desc": f"{part} of the sowing date (shifted +/- tolerance)"}

    if kind == "geometry":
        col = _find_col(raw_cols, [prov, "geometry_wkt", "geometry", "wkt", "geom"])
        if not col:
            return {"unavailable": "no geometry column in MinIO"}
        part = "centroid_x" if f.endswith("centroid_x") else "centroid_y" if f.endswith("centroid_y") else "area"
        return {"method": "geometry", "col": col, "part": part, "raw_column": col, "desc": f"{part} of the polygon in '{col}'"}

    if kind == "onehot":
        names = [prov] + (["crop", "crop_type", "crop_code", "crop_id"] if "crop" in _norm(prov) else [])
        col = _find_col(raw_cols, names)
        if not col:
            return {"unavailable": f"no '{prov}' (category) column in MinIO"}
        wanted = _cat_norm(feature[len(prov) + 1:])
        train_cats = {_cat_norm(k[len(prov) + 1:]) for k, v in STATE["feature_sources"].items() if v == prov}
        raw_vals = {_cat_norm(v) for v in raw[col].dropna().unique()}
        if not (raw_vals & train_cats):
            return {"unavailable": f"MinIO column '{col}' has other category values than the training data"}
        return {"method": "onehot", "col": col, "wanted": wanted, "raw_column": col, "desc": f"1 if '{col}' == {wanted}, else 0"}

    heat = "heat" in toks and "stress" in toks
    var = "tmax" if heat else next((VAR_TOKENS[t] for t in toks if t in VAR_TOKENS), None)
    window = next((int(t[:-1]) for t in toks if re.fullmatch(r"\d+d", t)), None)

    if var is None:
        col = _find_col(raw_cols, [feature])
        if col and pd.api.types.is_numeric_dtype(raw[col]):
            return {"method": "static", "col": col, "raw_column": col, "desc": f"value of raw column '{col}'"}
        return {"unavailable": "no raw MinIO column with this name"}

    assumed = None
    if heat:
        stat = "heat_days"
        assumed = f"heat_stress_days assumed = number of days with tmax > {HEAT_STRESS_TMAX_C:g} C"
    elif "peak" in toks:
        stat = "peak" if ("at" not in toks or var == "ndvi") else "at_peak"
        if stat == "at_peak":
            assumed = f"{feature} assumed = {var} on the day when ndvi is highest after sowing"
    elif "at" in toks and "sowing" in toks:
        stat = "at_sowing"
    elif "at" in toks and window:
        stat = "at_day"
    elif set(toks) & {"cum", "cumulative", "total", "sum"}:
        stat = "sum"
    else:
        stat = "mean"
        if window is None:
            assumed = f"{feature} assumed = mean over all observations from sowing onward"

    col = _find_col(raw_cols, RAW_ALIASES[var])
    if var == "gdd" and stat == "sum" and not col:
        col = _find_col(raw_cols, RAW_ALIASES["cgdd"])
        stat = "cum_last" if col else stat
    if var == "gdd" and not col:
        col = _find_col(raw_cols, RAW_ALIASES["cgdd"])
    if not col:
        return {"unavailable": f"no raw '{var}' column in MinIO"}

    dc = _ts_date_col(raw, col, date_cols)
    if dc is None and (stat in ("at_day", "at_sowing", "at_peak") or window is not None):
        return {"unavailable": f"no observation-date column found for '{col}'"}

    plan = {"method": "ts", "stat": stat, "col": col, "date_col": dc, "window": window, "raw_column": col, "assumed": assumed}
    if stat == "at_peak":
        pcol = _find_col(raw_cols, RAW_ALIASES["ndvi"])
        pdc = _ts_date_col(raw, pcol, date_cols) if pcol else None
        if not pcol or pdc is None:
            return {"unavailable": "ndvi series needed to locate the peak is missing in MinIO"}
        plan["peak_col"], plan["peak_date_col"] = pcol, pdc
    win = f" over the first {window} days after sowing" if window and stat in ("mean", "sum", "cum_last", "heat_days") else ""
    plan["desc"] = {
        "mean": f"mean of '{col}'{win or ' from sowing onward'}",
        "sum": f"sum of '{col}'{win}",
        "cum_last": f"last cumulative value of '{col}'{win}",
        "heat_days": f"days with '{col}' > {HEAT_STRESS_TMAX_C:g}{win}",
        "peak": f"maximum of '{col}' after sowing",
        "at_day": f"value of '{col}' about {window} days after sowing",
        "at_sowing": f"value of '{col}' at sowing",
        "at_peak": f"value of '{col}' on the ndvi peak day",
    }[stat]
    return plan


def _plan_columns(plan: dict) -> set:
    cols = set()
    for key in ("col", "date_col", "peak_col", "peak_date_col"):
        if plan.get(key):
            cols.add(plan[key])
    return cols


def _series(g: pd.DataFrame, col: str, date_col: Optional[str]):
    d = g[[col] + ([date_col] if date_col else [])].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])
    if date_col:
        d["_d"] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=["_d"]).sort_values("_d")
        days = ((d["_d"].dt.normalize() - pd.Timestamp("1970-01-01")).dt.days).values.astype("int64")
        return days, d[col].values.astype(float)
    return None, d[col].values.astype(float)


def _geozone_cache(g: pd.DataFrame, plans: dict) -> dict:
    cache = {"ts": {}, "static": {}, "geom": {}, "cat": {}}
    for plan in plans.values():
        m = plan["method"]
        if m == "ts":
            keys = [(plan["col"], plan["date_col"])]
            if plan.get("peak_col"):
                keys.append((plan["peak_col"], plan["peak_date_col"]))
            for key in keys:
                if key not in cache["ts"]:
                    cache["ts"][key] = _series(g, *key)
        elif m == "geometry" and plan["col"] not in cache["geom"]:
            vals = g[plan["col"]].dropna()
            poly = _extract_polygon(vals.iloc[0]) if not vals.empty else None
            if poly is None or poly.is_empty:
                cache["geom"][plan["col"]] = None
            else:
                c = poly.centroid
                cache["geom"][plan["col"]] = {"centroid_x": float(c.x), "centroid_y": float(c.y), "area": float(poly.area)}
        elif m == "onehot" and plan["col"] not in cache["cat"]:
            vals = g[plan["col"]].dropna().map(_cat_norm)
            cache["cat"][plan["col"]] = vals.mode().iloc[0] if not vals.empty else None
        elif m == "static" and plan["col"] not in cache["static"]:
            vals = pd.to_numeric(g[plan["col"]], errors="coerce").dropna()
            cache["static"][plan["col"]] = float(vals.iloc[0]) if not vals.empty else None
    return cache


def _nearest(delta: np.ndarray, vals: np.ndarray, target: float, max_gap: float = 20.0):
    if len(vals) == 0:
        return None
    i = int(np.argmin(np.abs(delta - target)))
    return float(vals[i]) if abs(float(delta[i]) - target) <= max_gap else None


def _eval_ts(plan: dict, cache: dict, sd: int):
    days, vals = cache["ts"][(plan["col"], plan["date_col"])]
    if len(vals) == 0:
        return None
    stat, window = plan["stat"], plan.get("window")
    if days is None:
        if stat in ("at_day", "at_sowing", "at_peak"):
            return None
        sel = vals
    else:
        delta = days - sd
        if stat == "at_day":
            return _nearest(delta, vals, window)
        if stat == "at_sowing":
            return _nearest(delta, vals, 0)
        mask = delta >= 0
        if window:
            mask = mask & (delta <= window)
        if stat == "at_peak":
            pdays, pvals = cache["ts"][(plan["peak_col"], plan["peak_date_col"])]
            if pdays is None or len(pvals) == 0:
                return None
            pdelta = pdays - sd
            pmask = pdelta >= 0
            if window:
                pmask = pmask & (pdelta <= window)
            if not pmask.any():
                return None
            peak_day = pdays[pmask][int(np.argmax(pvals[pmask]))]
            return _nearest(days - peak_day, vals, 0)
        sel = vals[mask]
    if len(sel) == 0:
        return None
    if stat == "mean":
        return float(sel.mean())
    if stat == "sum":
        return float(sel.sum())
    if stat == "cum_last":
        return float(sel[-1])
    if stat == "peak":
        return float(sel.max())
    if stat == "heat_days":
        return float((sel > HEAT_STRESS_TMAX_C).sum())
    return None


def _eval_plan(plan: dict, cache: dict, sow_shifted: pd.Timestamp):
    m = plan["method"]
    if m == "sowing_date":
        return float(sow_shifted.dayofyear if plan["part"] == "dayofyear" else sow_shifted.month)
    if m == "geometry":
        geo = cache["geom"].get(plan["col"])
        return None if geo is None else geo[plan["part"]]
    if m == "onehot":
        cat = cache["cat"].get(plan["col"])
        return None if cat is None else float(cat == plan["wanted"])
    if m == "static":
        return cache["static"].get(plan["col"])
    return _eval_ts(plan, cache, _days(sow_shifted))


# ---------------------------------------------------------------------------
# Benchmark settings — the gate is set by agronomy, not by the model
# ---------------------------------------------------------------------------

GATE = {
    "mae_days_max": float(os.getenv("HARVEST_GATE_MAE_DAYS", "4.0")),  # wider than the planning window -> useless
    "overfit_ratio_max": 1.5,           # holdout MAE / CV MAE
    "identity_top_k": 3,                # an identity-like feature in the top k blocks
    "proxy_top_k": 3,                   # region in the top k warns
    "xgboost_retry_improvement": 0.05,  # feedback edge: trees beat ridge by more than 5 %
    "noise_margin_days": 0.25,          # winner margin below this is "inside the noise"
}
# Crop / variety id columns are excluded by design: the harvest date must not be driven by the crop code
EXCLUDED_CROP_COLUMNS = {"crop_id", "crop", "crop_type", "crop_code", "crop_nomenclature_id", "crop_name",
                         "variety", "variety_id", "cultivar"}
FORBIDDEN_TIME_COLUMNS = {"season", "year", "operating_year", "il", "season_id", "year_id", "crop_year"}
GDD_BASE_C = 15.6                                    # cotton base temperature for growing degree days


def _is_time_identity(col: str) -> bool:
    """season / year columns let the model memorise the year — never features."""
    n = _norm(col)
    return n in FORBIDDEN_TIME_COLUMNS or n.endswith("_year") or n.startswith("year_")
IDENTITY_TOKENS = ("geozone", "field", "zone_id", "objectid", "uuid")
MODEL_COMPLEXITY = {"ridge": 0, "random_forest": 1, "xgboost": 2}
MATURITY_HINTS = ("gdd", "ndvi", "psri", "tmax", "evi")


def _gdd_from_temps(frame: pd.DataFrame, tmax_c: str, tmin_c: str) -> pd.Series:
    tmax = pd.to_numeric(frame[tmax_c], errors="coerce")
    tmin = pd.to_numeric(frame[tmin_c], errors="coerce")
    return ((tmax + tmin) / 2 - GDD_BASE_C).clip(lower=0)


def _make_model(name: str):
    if name == "ridge":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", RidgeCV(alphas=np.logspace(-2, 3, 12)))])
    if name == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                                         n_jobs=-1, random_state=RANDOM_STATE))])
    if name == "xgboost":
        return xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                                n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model '{name}'")


def _jsonable(obj):
    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else round(float(o), 4)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.strftime("%Y-%m-%d")
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.loads(json.dumps(obj, default=_default))


def _labelled() -> pd.DataFrame:
    df = STATE["df"]
    return df[df["days_to_harvest"].notna()]


def _X(frame: pd.DataFrame, feats: list) -> pd.DataFrame:
    return frame[feats].apply(pd.to_numeric, errors="coerce").astype(float)


def _groups(frame: pd.DataFrame) -> pd.Series:
    """Geozone id used to group CV folds so that no geozone straddles a fold."""
    idc = _id_col(frame.columns)
    if idc is None:
        idc = next((c for c in STATE["id_like_cols"] if c in frame.columns), None)
    if idc is None:
        return pd.Series(frame.index.astype(str), index=frame.index)
    return frame[idc].map(_cat_norm)


def _group_folds(groups: pd.Series, n_max: int = 5) -> list:
    n = min(n_max, groups.nunique())
    if n < 2:
        return []
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups.values))


def _grouped_oof(name: str, X: pd.DataFrame, y: pd.Series, folds: list) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, va in folds:
        oof[va] = clone(_make_model(name)).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return np.clip(oof, 1, None)


def _source_of(col: str) -> str:
    return STATE["feature_sources"].get(col, col)


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> dict:
    """Increase of MAE when all columns of one source column are shuffled together, in %."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = mean_absolute_error(y, model.predict(X))
    groups = {}
    for c in X.columns:
        groups.setdefault(_source_of(c), []).append(c)
    raw = {}
    for src, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
            deltas.append(mean_absolute_error(y, model.predict(Xp)) - base)
        raw[src] = max(float(np.mean(deltas)), 0.0)
    total = sum(raw.values()) or 1.0
    return dict(sorted(((k, round(v / total * 100, 2)) for k, v in raw.items()), key=lambda kv: -kv[1]))


def _reset_workflow():
    STATE.update({"prep": None, "profile": None, "splits": None, "baseline": None, "runs": {},
                  "explained": {}, "registered": None, "model": None, "model_name": None, "metrics": {}})


# ---------------------------------------------------------------------------
# TOOL 1 — data preparation (local files + MinIO feature rebuild)
# ---------------------------------------------------------------------------

def prepare_training_data(
    file_paths: list,
    use_minio: bool = True,
    sowing_tolerance_days: int = 5,
    bucket: str = MINIO_BUCKET,
) -> dict:
    """Step 1 of the workflow. Loads the training files (first = primary file
    with sowing and harvest dates, the rest = feature files merged by the shared
    geozone id), builds the target days_to_harvest = harvest date - sowing date,
    engineers all usable columns, then (use_minio=True) reads both MinIO folders
    and rebuilds every training feature from the raw daily observations relative
    to the sowing date. Features that cannot be rebuilt from MinIO are dropped
    with a reason, so the model is only trained on features available at
    prediction time. Never asks the user anything.

    Args:
        file_paths: training files; the first one is the primary file
        use_minio: read MinIO and rebuild features (False = benchmark only, no prediction)
        sowing_tolerance_days: +/- window applied to the MinIO sowing date (default 5)
        bucket: MinIO bucket
    """
    if not file_paths:
        return {"error": "No training files supplied"}
    t0 = time.time()
    _log("=" * 72)
    _log(f"PREPARE STARTED | files={file_paths} | use_minio={use_minio}")

    STATE.update({
        "files": {}, "primary_alias": None, "merged_aliases": [], "minio_alias": None, "df": None,
        "target": None, "feature_cols": [], "feature_medians": {}, "dropped_cols": {},
        "engineered": {}, "feature_sources": {},
    })
    _reset_workflow()

    # ---- 1) training data -------------------------------------------------
    loaded = []
    for i, path in enumerate(file_paths):
        r = load_file(path, alias=f"file_{i}", as_primary=(i == 0))
        if "error" in r:
            return r
        loaded.append(f"file_{i}")

    not_merged = []
    for alias in loaded[1:]:
        main, other = STATE["df"], STATE["files"][alias]
        common = [c for c in main.columns if c in other.columns and "id" in c.lower() and _looks_like_id(other[c], c)]
        if not common:
            common = [c for c in main.columns if c in other.columns and ("id" in c.lower() or "zone" in c.lower())]
        r = merge_feature_file(alias, common[0], common[0], "left") if common else {"error": "no shared key"}
        if "error" in r:
            not_merged.append({"file": alias, "reason": r["error"]})

    df = STATE["df"]
    _log(f"[1] training data ready | rows={len(df):,} | columns={len(df.columns)}")
    date_cols_train = []
    for c in df.columns:
        if any(k in _norm(c) for k in ("date", "time")) and not pd.api.types.is_numeric_dtype(df[c]):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.5:
                date_cols_train.append((c, parsed))
    harvest = [x for x in date_cols_train if set(_norm(x[0]).split("_")) & HARVEST_TOKENS]
    if harvest:
        target_raw = harvest[0][0]
    elif len(date_cols_train) >= 2:
        target_raw = max(date_cols_train, key=lambda x: x[1].median())[0]
    else:
        return {"error": "Could not identify a harvest/target date column automatically."}
    others = [x for x in date_cols_train if x[0] != target_raw]
    named = [x for x in others if set(_norm(x[0]).split("_")) & SOWING_TOKENS]
    if named:
        sow_raw = named[0][0]
    elif others:
        sow_raw = min(others, key=lambda x: x[1].median())[0]
    else:
        return {"error": "Could not identify the training sowing date column automatically."}

    # Target in DAYS: how many days from sowing until harvest
    sow_dt = pd.to_datetime(df[sow_raw], errors="coerce")
    harv_dt = pd.to_datetime(df[target_raw], errors="coerce")
    days = (harv_dt - sow_dt).dt.days.astype(float)
    invalid = int(((days <= 0) | (days > 365)).sum())
    days[(days <= 0) | (days > 365)] = np.nan
    df["days_to_harvest"] = days
    df["_season"] = sow_dt.dt.year
    set_target("days_to_harvest")
    engineer_date_column(sow_raw)
    STATE["train_sowing_col"] = sow_raw
    _log(f"[1] target=days_to_harvest ({target_raw!r} - {sow_raw!r}) | invalid rows set to NaN={invalid}")

    candidates = [sow_raw + "_dayofyear", sow_raw + "_month"]
    skipped_train = []
    cat_words = {"crop", "type", "class", "category", "code", "id", "kind", "variety"}
    for c in list(df.columns):
        if c in (target_raw, sow_raw, "days_to_harvest") or c.startswith(target_raw + "_") \
                or c.startswith(sow_raw + "_") or c.startswith("_"):
            continue
        s = STATE["df"][c]
        if _norm(c) in EXCLUDED_CROP_COLUMNS:
            skipped_train.append({"feature": c, "reason": "excluded by design: crop id is not used as a feature"})
            continue
        if _is_time_identity(c):
            skipped_train.append({"feature": c, "reason": "forbidden: season/year lets the model memorise the year"})
            continue
        if s.nunique(dropna=True) <= 1:
            skipped_train.append({"feature": c, "reason": "constant column (no information)"})
            continue
        if _looks_like_id(s, c):
            skipped_train.append({"feature": c, "reason": "identifier column (forbidden: memorises the field)"})
            continue
        if _looks_like_geometry(s):
            candidates += engineer_geometry_column(c).get("created_columns", [])
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 15 and set(_norm(c).split("_")) & cat_words:
                candidates += one_hot_encode_column(c).get("created_columns", [])
            else:
                candidates.append(c)
        elif _looks_like_date(s):
            candidates += engineer_date_column(c).get("created_columns", [])
        elif s.nunique(dropna=True) <= 20:
            candidates += one_hot_encode_column(c).get("created_columns", [])
        else:
            skipped_train.append({"feature": c, "reason": "high-cardinality text column"})
    candidates = list(dict.fromkeys(candidates))
    _log(f"[1] {len(candidates)} candidate features: {candidates}")

    removed = list(skipped_train)
    prep = {
        "file_paths": file_paths, "target_raw": target_raw, "sow_raw": sow_raw, "not_merged": not_merged,
        "invalid_targets": invalid, "sowing_tolerance_days": sowing_tolerance_days,
        "plans": {}, "mat": None, "meta": None, "sowing": {}, "minio": None, "minio_sow": None,
    }

    if not use_minio:
        prep["features"] = candidates
        prep["removed"] = removed
        STATE["prep"] = prep
        return _prepare_summary(prep, t0)

    # ---- 2) MinIO: list files, read schema ----------------------------------
    _log("[2] listing MinIO objects")
    try:
        objects, n_skipped = _list_minio_csvs(bucket, MINIO_PREFIXES)
    except Exception as exc:
        return {"error": f"MinIO listing failed: {exc}"}
    if not objects:
        return {"error": f"No CSV files found in '{bucket}' under {MINIO_PREFIXES} (after skipping {MINIO_SKIP_PARTS})"}
    _log(f"[2] CSV files to inspect={len(objects)} | skipped by path ({MINIO_SKIP_PARTS})={n_skipped}")

    try:
        file_cols, sample, schema_failed = _read_schema_samples(bucket, objects)
    except Exception as exc:
        return {"error": f"MinIO schema read failed: {exc}"}
    raw_cols = [c for c in sample.columns if not c.startswith("_")]
    date_cols = _date_like_columns(sample)
    minio_sow = _pick_minio_sowing(sample, date_cols)
    if not minio_sow:
        return {"error": "No sowing-date column found in the MinIO files.", "date_like_columns": date_cols, "minio_columns": raw_cols}
    _log(f"[2] MinIO sowing column={minio_sow!r} | date-like columns={date_cols}")

    # daily GDD is computed from tmax/tmin when MinIO has no ready GDD column
    tmax_c, tmin_c = _find_col(raw_cols, RAW_ALIASES["tmax"]), _find_col(raw_cols, RAW_ALIASES["tmin"])
    gdd_calc = None
    if not _find_col(raw_cols, RAW_ALIASES["gdd"] + RAW_ALIASES["cgdd"]) and tmax_c and tmin_c:
        gdd_calc = "gdd_from_tmax_tmin"
        sample[gdd_calc] = _gdd_from_temps(sample, tmax_c, tmin_c)
        raw_cols.append(gdd_calc)
        _log(f"[2] no GDD column in MinIO -> daily GDD computed from {tmax_c!r}/{tmin_c!r} (base {GDD_BASE_C} C)")

    # ---- 3) plan every feature ---------------------------------------------
    plans = {}
    for feat in candidates:
        plan = _plan_feature(feat, sample, raw_cols, date_cols, sow_raw)
        if "unavailable" in plan:
            removed.append({"feature": feat, "reason": plan["unavailable"]})
            _log(f"[3] {feat}: NOT AVAILABLE ({plan['unavailable']})")
        else:
            plans[feat] = plan
            _log(f"[3] {feat}: {plan['desc']}")

    value_like = {minio_sow}
    needed_all = {minio_sow}
    for plan in plans.values():
        cols = _plan_columns(plan)
        if gdd_calc in cols:
            cols = (cols - {gdd_calc}) | {tmax_c, tmin_c}
        needed_all |= cols
        value_like |= {c for c in cols if c not in date_cols}

    # ---- 4) read only what is needed -----------------------------------------
    try:
        raw, used_files, load_failed, zones_by_batch, dup_removed = _load_needed(bucket, objects, file_cols, value_like, needed_all)
    except Exception as exc:
        return {"error": f"MinIO load failed: {exc}"}
    _log(f"[4] raw rows={len(raw):,} | geozones={raw['_minio_id'].nunique():,} | files read={len(used_files)} | duplicate rows removed={dup_removed:,}")

    if gdd_calc and tmax_c in raw.columns and tmin_c in raw.columns:
        raw[gdd_calc] = _gdd_from_temps(raw, tmax_c, tmin_c)

    sowing = {}
    if minio_sow in raw.columns:
        for gid, dates in raw.groupby("_minio_id")[minio_sow]:
            parsed = pd.to_datetime(dates, errors="coerce").dropna()
            if not parsed.empty:
                sowing[gid] = parsed.mode().iloc[0]
    all_ids = set(raw["_minio_id"].unique())
    groups = {gid: g for gid, g in raw.groupby("_minio_id") if gid in sowing}
    n_no_sowing = len(all_ids) - len(groups)
    if not groups:
        return {"error": "No geozone has a usable sowing date in MinIO.", "minio_sowing_column": minio_sow}
    _log(f"[4] geozones with sowing date={len(groups):,} | without={n_no_sowing:,}")

    # ---- 5) rebuild features for every geozone and sowing shift ---------------
    shifts = list(range(-sowing_tolerance_days, sowing_tolerance_days + 1))
    rows, meta = [], []
    for k, (gid, g) in enumerate(groups.items(), start=1):
        cache = _geozone_cache(g, plans)
        for sh in shifts:
            ss = sowing[gid] + pd.Timedelta(days=sh)
            rows.append({f: (np.nan if v is None else v) for f, v in ((f, _eval_plan(p, cache, ss)) for f, p in plans.items())})
            meta.append((gid, sh, ss))
        if k % 100 == 0 or k == len(groups):
            _log(f"[5] features rebuilt for {k}/{len(groups)} geozones")
    mat = pd.DataFrame(rows, columns=list(plans)).astype(float)
    zero_rows = [i for i, m in enumerate(meta) if m[1] == 0]
    nan_share = mat.iloc[zero_rows].isna().mean() if zero_rows else pd.Series(dtype=float)
    for feat in list(plans):
        if nan_share.get(feat, 0) > MAX_NAN_SHARE:
            removed.append({"feature": feat, "reason": f"could not be computed for {nan_share[feat] * 100:.0f}% of MinIO geozones"})
            _log(f"[5] {feat}: dropped, missing for {nan_share[feat] * 100:.0f}% of geozones")
            del plans[feat]
    feats_final = [f for f in candidates if f in plans]
    if not [f for f in feats_final if not f.startswith(sow_raw + "_")]:
        return {
            "error": "After rebuilding features from MinIO only sowing-date features remain; predicting would be meaningless.",
            "features_removed": removed,
            "minio_columns": raw_cols,
        }

    overlap = sum(1 for gid in all_ids if sum(gid in z for z in zones_by_batch.values()) > 1)
    prep.update({
        "features": feats_final, "removed": removed, "plans": plans, "mat": mat, "meta": meta,
        "sowing": sowing, "minio_sow": minio_sow,
        "assumed": sorted({p["assumed"] for p in plans.values() if p.get("assumed")}),
        "minio": {
            "csv_files_inspected": len(objects),
            "files_skipped_by_path": n_skipped,
            "skipped_path_parts": MINIO_SKIP_PARTS,
            "files_read_for_features": len(used_files),
            "files_failed": {**schema_failed, **load_failed},
            "geozones_per_folder": {p: len(z) for p, z in zones_by_batch.items()},
            "geozones_total": len(all_ids),
            "geozones_in_more_than_one_folder": overlap,
            "geozones_with_sowing_date": len(groups),
            "geozones_skipped_no_sowing_date": n_no_sowing,
            "duplicate_rows_removed": dup_removed,
        },
    })
    STATE["prep"] = prep
    return _prepare_summary(prep, t0)


def _prepare_summary(prep: dict, t0: float) -> dict:
    lab = _labelled()
    seasons = sorted(int(s) for s in lab["_season"].dropna().unique())
    out = {
        "status": "ready",
        "target": f"days_to_harvest = {prep['target_raw']} - {prep['sow_raw']} (days)",
        "labelled_rows": len(lab),
        "geozones": int(_groups(lab).nunique()),
        "seasons": seasons,
        "days_to_harvest_mean": round(float(lab["days_to_harvest"].mean()), 1),
        "invalid_target_rows_dropped": prep["invalid_targets"],
        "files_not_merged": prep["not_merged"],
        "features_kept": prep["features"],
        "features_removed": prep["removed"],
        "assumed_definitions": prep.get("assumed", []),
        "minio": prep["minio"] or "not read (use_minio=False) — benchmark only, no MinIO prediction possible",
        "minio_sowing_column": prep["minio_sow"],
        "elapsed_s": round(time.time() - t0, 1),
        "next_step": "profile_dataset",
    }
    if len(seasons) < 2:
        out["warning"] = (f"Only one season ({seasons}) has harvest dates: a season-out test is impossible. "
                          f"make_splits will fall back to a held-out 20 % of geozones — weaker evidence. "
                          f"Add older seasons to get an honest season-out test.")
    _log(f"[PREPARE] labelled rows={len(lab)} | seasons={seasons} | features={len(prep['features'])}")
    return _jsonable(out)


# ---------------------------------------------------------------------------
# TOOL 2 — profile
# ---------------------------------------------------------------------------

def profile_dataset() -> dict:
    """Step 2. Measures six signals (rows, rows_per_feature, linearity_gap,
    missing_pct, n_fields, seasons) and turns them into a shortlist of models
    worth training. No model may be named before this runs. The newest season
    is excluded from the measurement when there is more than one.
    """
    prep = STATE["prep"]
    if prep is None:
        return {"error": "Call prepare_training_data first"}
    lab = _labelled()
    feats = prep["features"]
    seasons = sorted(lab["_season"].dropna().unique())
    dev = lab[lab["_season"] < seasons[-1]] if len(seasons) > 1 else lab
    X, y, groups = _X(dev, feats), dev["days_to_harvest"], _groups(dev)

    rows, n_features = len(lab), len(feats)
    missing_pct = float(X.isna().mean().mean() * 100) if n_features else 0.0
    n_fields = int(_groups(lab).nunique())

    linearity_gap = None
    folds = _group_folds(groups)
    if folds and len(dev) >= 20:
        r2_ridge = r2_score(y, _grouped_oof("ridge", X, y, folds))
        quick_rf = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("model", RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE))])
        oof = np.full(len(y), np.nan)
        for tr, va in folds:
            oof[va] = clone(quick_rf).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        linearity_gap = float(r2_score(y, oof) - r2_ridge)

    signals = {
        "rows": rows, "n_features": n_features,
        "rows_per_feature": round(rows / max(n_features, 1), 1),
        "linearity_gap": None if linearity_gap is None else round(linearity_gap, 3),
        "missing_pct": round(missing_pct, 1),
        "n_fields": n_fields, "seasons": len(seasons),
    }

    shortlist = ["ridge", "random_forest"]
    reasons = ["ridge: always kept as the floor",
               "random_forest: always a challenger (never fewer than two candidates)"]
    warnings_ = []
    if len(seasons) < 2:
        warnings_.append("fewer than 2 seasons: no season can be held out (fallback: held-out geozones)")
    if signals["rows_per_feature"] < 10:
        warnings_.append("rows_per_feature < 10: cut features before adding model capacity")
    if n_fields < 30:
        warnings_.append("few geozones: the holdout is noisy to rank on")
    if rows < 500:
        reasons.append("under 500 rows: ridge is the favourite")
    if linearity_gap is not None:
        if linearity_gap < 0.03:
            reasons.append(f"linearity_gap {linearity_gap:.3f} < 0.03: probably close to linear")
        elif linearity_gap > 0.10:
            reasons.append(f"linearity_gap {linearity_gap:.3f} > 0.10: clear curvature, trees should win")
    xgb_reasons = []
    if rows > 5000:
        xgb_reasons.append("over 5 000 rows, boosting is affordable")
    if missing_pct > 15:
        xgb_reasons.append("over 15 % missing, XGBoost splits on missingness natively")
    if linearity_gap is not None and linearity_gap > 0.10 and rows >= 500:
        xgb_reasons.append("clear curvature with enough rows")
    if xgb_reasons:
        shortlist.append("xgboost")
        reasons.append("xgboost: " + "; ".join(xgb_reasons))
    else:
        reasons.append("xgboost: pruned for now (the benchmark feedback can bring it back)")

    STATE["profile"] = {"signals": signals, "shortlist": shortlist}
    _log(f"[PROFILE] {signals} | shortlist={shortlist}")
    return _jsonable({"signals": signals, "shortlist": shortlist, "reasons": reasons,
                      "warnings": warnings_, "next_step": "make_splits"})


# ---------------------------------------------------------------------------
# TOOL 3 — splits
# ---------------------------------------------------------------------------

def make_splits() -> dict:
    """Step 3. Season-out split: the newest season is the test set, CV folds
    inside the training seasons are grouped by geozone id. With only one season
    it falls back to holding out 20 % of the geozones (and says so loudly).
    Id, season and year columns can never be features.
    """
    if STATE["profile"] is None:
        return {"error": "Call profile_dataset first"}
    lab = _labelled().reset_index(drop=True)
    feats = STATE["prep"]["features"]
    groups = _groups(lab)
    seasons = sorted(int(s) for s in lab["_season"].dropna().unique())

    if len(seasons) >= 2:
        mode = "season_out"
        test_mask = (lab["_season"] == seasons[-1]).values
        warning = None
    else:
        mode = "geozone_holdout_fallback"
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        _, te = next(gss.split(lab, groups=groups.values))
        test_mask = np.zeros(len(lab), dtype=bool)
        test_mask[te] = True
        warning = ("NOT a season-out test: only one season is available, so 20 % of the geozones of the same "
                   "season are held out. Scores will look better than on a truly unseen year.")

    train, test = lab[~test_mask].reset_index(drop=True), lab[test_mask].reset_index(drop=True)
    g_train = _groups(train)
    folds = _group_folds(g_train)
    if not folds or test.empty:
        return {"error": "Not enough geozones to build grouped CV folds and a test set"}
    leaked = set(g_train) & set(_groups(test)) if mode != "season_out" else set()
    STATE["splits"] = {
        "mode": mode, "train": train, "test": test, "folds": folds,
        "X_train": _X(train, feats), "y_train": train["days_to_harvest"],
        "X_test": _X(test, feats), "y_test": test["days_to_harvest"],
        "train_seasons": sorted(int(s) for s in train["_season"].dropna().unique()),
        "test_season": seasons[-1] if mode == "season_out" else None,
    }
    STATE["baseline"], STATE["runs"] = None, {}
    return _jsonable({
        "mode": mode, "warning": warning,
        "train_seasons": STATE["splits"]["train_seasons"], "test_season": STATE["splits"]["test_season"],
        "train_rows": len(train), "test_rows": len(test),
        "train_geozones": int(g_train.nunique()), "test_geozones": int(_groups(test).nunique()),
        "geozones_in_both_train_and_test": len(leaked),
        "cv_folds": len(folds), "cv_grouped_by": "geozone id",
        "features": feats, "next_step": "baseline",
    })


# ---------------------------------------------------------------------------
# TOOL 4 — baseline
# ---------------------------------------------------------------------------

def _find_gdd_column(frame: pd.DataFrame) -> tuple:
    """A cumulative-GDD column with a window in its name, e.g. cum_gdd_150d -> ('cum_gdd_150d', 150)."""
    for c in frame.columns:
        toks = _norm(c).split("_")
        if "gdd" in toks and pd.api.types.is_numeric_dtype(frame[c]):
            win = next((int(t[:-1]) for t in toks if re.fullmatch(r"\d+d", t)), None)
            if win and frame[c].notna().mean() > 0.5:
                return c, win
    return None, None


def baseline() -> dict:
    """Step 4. Rule of thumb, no machine learning. GDD rule: learn the average
    heat sum needed until harvest on the training rows (daily GDD rate x days),
    then days_to_harvest = needed GDD / the geozone's daily GDD rate. Falls back
    to the average season length when no GDD column exists. Every model is
    reported against this MAE.
    """
    sp = STATE["splits"]
    if sp is None:
        return {"error": "Call make_splits first"}
    train, test, y_tr, y_te = sp["train"], sp["test"], sp["y_train"], sp["y_test"]
    gdd_col, window = _find_gdd_column(STATE["df"])
    mean_days = float(y_tr.mean())
    if gdd_col:
        rate_tr = pd.to_numeric(train[gdd_col], errors="coerce") / window
        rate_te = pd.to_numeric(test[gdd_col], errors="coerce") / window
        need = float((rate_tr * y_tr).mean())
        pred = (need / rate_te.where(rate_te > 0.1)).fillna(mean_days)
        method = (f"GDD rule: harvest after {need:.0f} GDD; days = {need:.0f} / daily rate "
                  f"({gdd_col} / {window})")
    else:
        pred = pd.Series(mean_days, index=test.index)
        method = f"calendar rule: harvest {mean_days:.0f} days after sowing (no GDD column found)"
    pred = pred.clip(lower=1).values
    mae = float(mean_absolute_error(y_te, pred))
    STATE["baseline"] = {"mae_days": mae, "method": method}
    _log(f"[BASELINE] {method} | MAE={mae:.2f}")
    return _jsonable({"method": method, "mae_days": round(mae, 2),
                      "p90_error_days": round(float(np.quantile(np.abs(y_te - pred), 0.9)), 2),
                      "next_step": "run_benchmark"})


# ---------------------------------------------------------------------------
# TOOL 5 — benchmark + checks + gate
# ---------------------------------------------------------------------------

def _evaluate(name: str, sp: dict, baseline_mae: float) -> dict:
    X_tr, y_tr, X_te, y_te = sp["X_train"], sp["y_train"], sp["X_test"], sp["y_test"]
    oof = _grouped_oof(name, X_tr, y_tr, sp["folds"])
    cv_mae = float(mean_absolute_error(y_tr, oof))
    model = _make_model(name).fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 1, None)
    err = np.abs(y_te.values - pred)
    mae = float(err.mean())
    train_mae = float(mean_absolute_error(y_tr, model.predict(X_tr)))   # internal only, never for ranking
    importance = _permutation_importance(model, X_te, y_te)
    top_ident = list(importance)[: GATE["identity_top_k"]]
    top_proxy = list(importance)[: GATE["proxy_top_k"]]

    checks = []

    def add(check, fired, severity, detail):
        checks.append({"check": check, "status": severity if fired else "pass", "detail": detail})

    add("threshold", mae > GATE["mae_days_max"], "block", f"mae_days {mae:.2f} vs max {GATE['mae_days_max']}")
    add("baseline", mae >= baseline_mae, "block", f"mae_days {mae:.2f} vs baseline {baseline_mae:.2f}")
    ratio = mae / cv_mae if cv_mae > 0 else np.inf
    add("overfit", ratio > GATE["overfit_ratio_max"], "block",
        f"holdout/cv = {ratio:.2f} (max {GATE['overfit_ratio_max']})")
    ident = [f for f in top_ident if any(t in _norm(f) for t in IDENTITY_TOKENS)]
    add("identity", bool(ident), "block", f"identity-like features in top {GATE['identity_top_k']}: {ident or 'none'}")
    proxy = [f for f in top_proxy if "region" in _norm(f)]
    add("proxy", bool(proxy), "warn", f"region-like features in top {GATE['proxy_top_k']}: {proxy or 'none'}")
    checks.append({"check": "stage", "status": "n/a",
                   "detail": "one row per geozone (no per-pass rows), error per crop stage cannot be measured"})

    return {
        "model": name, "fitted": model,
        "mae_days": mae, "cv_mae_days": cv_mae, "vs_baseline": mae - baseline_mae,
        "p90_error_days": float(np.quantile(err, 0.9)), "train_mae_days_internal": train_mae,
        "importance_pct": importance, "checks": checks,
        "blocked_by": [c["check"] for c in checks if c["status"] == "block"],
        "warnings": [c["check"] for c in checks if c["status"] == "warn"],
    }


def _results_table(ranked: list, baseline_mae: float, baseline_name: str) -> str:
    lines = ["| Model | MAE days | CV MAE | vs. baseline | p90 error | Outcome |",
             "|---|---|---|---|---|---|"]
    for r in ranked:
        lines.append(f"| {r['model']} | {r['mae_days']:.2f} | {r['cv_mae_days']:.2f} | {r['vs_baseline']:+.2f} | "
                     f"{r['p90_error_days']:.2f} | {r['outcome']} |")
    lines.append(f"| {baseline_name} (baseline) | {baseline_mae:.2f} | — | — | — | — |")
    return "\n".join(lines)


def _propose_features() -> list:
    not_fixable = ("identifier", "constant", "forbidden", "high-cardinality", "excluded")
    removed = [x["feature"] for x in (STATE["prep"] or {}).get("removed", [])
               if not str(x["reason"]).startswith(not_fixable)]
    props = []
    if removed:
        props.append(f"make these features available in MinIO so they are not dropped: {removed}")
    props += [
        "add more seasons with recorded harvest dates (enables a real season-out test)",
        "add per-pass satellite time series (NDVI decline, PSRI rise, days since NDVI peak)",
        "add PSRI / NDMI / MSAVI indices and accumulated GDD up to the observation date",
        "add field context: region, soil type, irrigation dates",
    ]
    return props


def run_benchmark(shortlist: Optional[list] = None) -> dict:
    """Step 5. Trains every shortlisted model on the identical split, scores
    each on held-out MAE in days, runs the automatic checks and the gate and
    ranks the models. The ranking is final.

    Args:
        shortlist: names from {"ridge", "random_forest", "xgboost"}; default =
            the shortlist from profile_dataset. Ridge is always included.
    """
    if STATE["profile"] is None:
        return {"error": "Call profile_dataset first"}
    if STATE["splits"] is None:
        return {"error": "Call make_splits first"}
    if STATE["baseline"] is None:
        return {"error": "Call baseline first — every model is reported against it"}

    shortlist = list(shortlist or STATE["profile"]["shortlist"])
    unknown = [m for m in shortlist if m not in MODEL_COMPLEXITY]
    if unknown:
        return {"error": f"Unknown models {unknown}. Allowed: {list(MODEL_COMPLEXITY)}"}
    if "ridge" not in shortlist:
        shortlist.insert(0, "ridge")
    if len(shortlist) < 2:
        shortlist.append("random_forest")

    sp, base = STATE["splits"], STATE["baseline"]
    base_mae = base["mae_days"]
    t0 = time.time()
    results = {}
    for name in shortlist:
        _log(f"[BENCHMARK] training {name}")
        results[name] = _evaluate(name, sp, base_mae)

    feedback = None
    best = min(results.values(), key=lambda r: r["mae_days"])
    improvement = (results["ridge"]["mae_days"] - best["mae_days"]) / results["ridge"]["mae_days"]
    if improvement > GATE["xgboost_retry_improvement"] and "xgboost" not in results:
        _log(f"[BENCHMARK] feedback: {best['model']} beats ridge by {improvement:.1%}, adding xgboost")
        results["xgboost"] = _evaluate("xgboost", sp, base_mae)
        shortlist.append("xgboost")
        feedback = (f"{best['model']} beat ridge by {improvement:.1%} (> {GATE['xgboost_retry_improvement']:.0%}); "
                    f"xgboost was added and benchmarked")

    ranked = sorted(results.values(), key=lambda r: r["mae_days"])
    winner = next((r for r in ranked if not r["blocked_by"]), None)
    for r in ranked:
        if r is winner:
            r["outcome"] = "Selected" + (f" (warn: {', '.join(r['warnings'])})" if r["warnings"] else "")
        elif r["blocked_by"]:
            r["outcome"] = "Blocked: " + ", ".join(r["blocked_by"])
        else:
            r["outcome"] = "Passed"

    notes = []
    if sp["mode"] != "season_out":
        notes.append("split is NOT season-out (only one season) — treat these numbers as optimistic")
    if winner:
        for r in ranked:
            if r is winner or r["blocked_by"] or MODEL_COMPLEXITY[r["model"]] >= MODEL_COMPLEXITY[winner["model"]]:
                continue
            margin = r["mae_days"] - winner["mae_days"]
            if margin < GATE["noise_margin_days"]:
                notes.append(f"{winner['model']} beats the simpler {r['model']} by only {margin:.2f} days — inside "
                             f"the noise; a sane team ships {r['model']} unless the margin holds over several seasons")

    gate = {"passed": winner is not None}
    if winner is None:
        gate["gap"] = {
            "best_model": best["model"], "best_mae_days": round(best["mae_days"], 2),
            "days_over_threshold": round(best["mae_days"] - GATE["mae_days_max"], 2),
            "days_vs_baseline": round(best["mae_days"] - base_mae, 2),
            "blocked_by": best["blocked_by"],
        }
        gate["proposed_features"] = _propose_features()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    STATE["runs"][run_id] = {"results": results, "winner": winner["model"] if winner else None, "gate": gate,
                             "best": best["model"], "ranked": [r["model"] for r in ranked]}
    STATE["last_run_id"] = run_id
    baseline_name = "GDD rule" if base["method"].startswith("GDD") else "Calendar rule"
    table_md = _results_table(ranked, base_mae, baseline_name)

    public = []
    for r in ranked:
        p = {k: v for k, v in r.items() if k not in ("fitted", "train_mae_days_internal")}
        for k in ("mae_days", "cv_mae_days", "vs_baseline", "p90_error_days"):
            p[k] = round(p[k], 2)
        p["importance_pct"] = dict(list(p["importance_pct"].items())[:10])
        public.append(p)
    report = {
        "run_id": run_id, "split_mode": sp["mode"], "test_season": sp["test_season"],
        "baseline": {"method": base["method"], "mae_days": round(base_mae, 2)},
        "shortlist_trained": shortlist, "feedback_retry": feedback,
        "ranking_metric": "held-out mae_days (lower is better)",
        "comparison_table_markdown": table_md,
        "models": public, "winner": winner["model"] if winner else None,
        "gate": gate, "gate_thresholds": GATE, "notes": notes,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"benchmark_{run_id}.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, ensure_ascii=False, indent=2)
    with open(f"benchmark_{run_id}.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Benchmark {run_id}\n\nSplit: {sp['mode']}\n\n{table_md}\n")
    _log(f"[BENCHMARK] winner={report['winner']} | gate passed={gate['passed']}\n{table_md}")
    report["next_step"] = ("explain(run_id) on the winner, then register_model(winner, run_id)" if winner
                           else "do NOT register; report the gap and the proposed features")
    return _jsonable(report)



def explain(run_id: str, model_name: Optional[str] = None) -> dict:
    """Step 6. Feature importance of the winner (grouped permutation importance
    on the held-out rows, % of the MAE increase) plus review flags. Must run on
    the winner before register_model.

    Args:
        run_id: id returned by run_benchmark
        model_name: defaults to the winner of that run
    """
    run = STATE["runs"].get(run_id)
    if run is None:
        return {"error": f"Unknown run_id '{run_id}'. Known: {list(STATE['runs'])}"}
    name = model_name or run["winner"] or run["best"]   # no winner -> explain the best-ranked model (preview)
    if name not in run["results"]:
        return {"error": f"Model '{name}' was not trained in {run_id}"}
    imp = run["results"][name]["importance_pct"]
    plans = STATE["prep"]["plans"]
    how = {}   # source column -> descriptions of ALL model columns built from it
    for f, p in plans.items():
        descs = how.setdefault(_source_of(f), [])
        if p["desc"] not in descs:
            descs.append(p["desc"])
    how = {k: "; ".join(v) for k, v in how.items()}
    table = [{"feature": f, "importance_percent": v, "how_rebuilt_from_minio": how.get(f, "—")} for f, v in imp.items()]
    top5 = list(imp)[:5]
    flags = []
    if not any(any(h in _norm(f) for h in MATURITY_HINTS) for f in top5):
        flags.append("no heat / vegetation feature (GDD, NDVI, PSRI, tmax, EVI) in the top 5 — check the features "
                     "before trusting the score")
    if imp and max(imp.values()) > 50:
        flags.append(f"'{top5[0]}' carries {imp[top5[0]]}% of the model — a human should review it before it reaches a farmer")
    if name == run["winner"]:
        STATE["explained"][run_id] = name
    return _jsonable({"run_id": run_id, "model": name, "is_winner": name == run["winner"],
                      "features_by_importance": table, "review_flags": flags,
                      "method": "grouped permutation importance on the held-out rows"})


def get_gate() -> dict:
    """Returns the gate thresholds and check rules (set by agronomy)."""
    return _jsonable({
        "ship_if": f"mae_days <= {GATE['mae_days_max']} and mae_days < baseline.mae_days and no blocking check fires",
        "thresholds": GATE, "blocking_checks": ["threshold", "baseline", "overfit", "identity"],
        "warning_checks": ["proxy"],
    })


def _fit_final(name: str, run: dict) -> dict:
    """Refits a benchmarked model on all labelled rows and returns what prediction needs."""
    feats = STATE["prep"]["features"]
    lab = _labelled()
    X_all = _X(lab, feats)
    r = run["results"][name]
    return {
        "model": _make_model(name).fit(X_all, lab["days_to_harvest"]), "model_name": name, "feature_cols": feats,
        "feature_medians": {c: float(X_all[c].median()) for c in feats if X_all[c].notna().any()},
        "feature_ranges": {c: [float(X_all[c].min()), float(X_all[c].max())] for c in feats if X_all[c].notna().any()},
        "metrics": {"holdout_mae_days": round(r["mae_days"], 2), "cv_mae_days": round(r["cv_mae_days"], 2),
                    "baseline_mae_days": round(STATE["baseline"]["mae_days"], 2)},
    }


def register_model(name: str, run_id: str) -> dict:
    """Step 7. Promotes the model for prediction. Refuses anything the benchmark
    did not select, anything after a failed gate, and a winner that was not explained.

    Args:
        name: model name — must equal the winner of the run
        run_id: id returned by run_benchmark
    """
    run = STATE["runs"].get(run_id)
    if run is None:
        return {"error": f"Unknown run_id '{run_id}'"}
    if run["winner"] is None or not run["gate"]["passed"]:
        return {"error": "REFUSED: the gate did not pass in this run — nothing may be registered"}
    if name != run["winner"]:
        return {"error": f"REFUSED: '{name}' is not the benchmark winner ('{run['winner']}'). The ranking cannot be changed."}
    if STATE["explained"].get(run_id) != name:
        return {"error": f"REFUSED: run explain(run_id='{run_id}') on the winner before registering"}

    STATE.update(_fit_final(name, run))
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"harvest_{name}_{run_id}.joblib")
    feats, final = STATE["feature_cols"], STATE["model"]
    joblib.dump({"model": final, "features": feats, "medians": STATE["feature_medians"],
                 "ranges": STATE["feature_ranges"],
                 "metrics": STATE["metrics"], "run_id": run_id, "gate": GATE}, path)
    STATE["registered"] = {"name": name, "run_id": run_id, "path": path}
    _log(f"[REGISTER] {name} -> {path}")
    return _jsonable({"registered": name, "path": path, **STATE["metrics"],
                      "note": "evaluated on the held-out rows, then refit on all labelled rows",
                      "next_step": "predict_from_minio"})


def predict_from_minio(output_path: str = "predictions_minio.csv") -> dict:
    """Step 8. ALWAYS produces the prediction table for every MinIO geozone:
    days_to_harvest for every sowing shift -N..+N, converted to dates (shifted
    sowing date + predicted days) and averaged per geozone.

    - Accuracy target met + model registered -> the registered model is used.
    - Accuracy target not met -> the best-ranked benchmark model is used
      (nothing is registered) and its real accuracy is reported.
    Columns: geozone_id, sowing_date, days_to_harvest.

    Also writes feature_importance.csv.

    Args:
        output_path: prediction CSV path
    """
    prep = STATE["prep"]
    if prep is None:
        return {"error": "Call prepare_training_data first"}
    if prep["mat"] is None:
        return {"error": "MinIO was not read (prepare_training_data was called with use_minio=False)"}

    reg = STATE["registered"]
    if reg is not None:
        model, feats = STATE["model"], STATE["feature_cols"]
        medians, ranges, metrics = STATE["feature_medians"], STATE["feature_ranges"], STATE["metrics"]
        model_name, run_id = reg["name"], reg["run_id"]
        gate_status = "accuracy target reached"
    else:
        run_id = STATE.get("last_run_id")
        run = STATE["runs"].get(run_id) if run_id else None
        if run is None:
            return {"error": "Run the benchmark first (run_benchmark)"}
        if run["winner"] is not None:
            return {"error": f"The gate passed — call explain('{run_id}') and register_model('{run['winner']}', '{run_id}') first"}
        fit = _fit_final(run["best"], run)
        model, feats = fit["model"], fit["feature_cols"]
        medians, ranges, metrics = fit["feature_medians"], fit["feature_ranges"], fit["metrics"]
        model_name = run["best"]
        blocked = ", ".join(run["results"][model_name]["blocked_by"])
        gate_status = (f"accuracy target not reached: mean error {metrics['holdout_mae_days']} days "
                       f"(target {GATE['mae_days_max']} days; failed checks: {blocked})")

    mat, meta = prep["mat"], prep["meta"]
    X = mat[feats].copy()
    filled = {}   # feature -> number of GEOZONES that needed a median fill
    gids = pd.Series([m[0] for m in meta])
    for c in feats:
        miss = X[c].isna().values
        if miss.any():
            filled[c] = int(gids[miss].nunique())
            X[c] = X[c].fillna(medians.get(c, 0.0))

    # MinIO values outside the training range would make the model extrapolate -> clip and report
    shift_report, shift_warnings = {}, []
    for c, (lo, hi) in ranges.items():
        outside = float(((X[c] < lo) | (X[c] > hi)).mean() * 100)
        if outside > 0:
            shift_report[c] = {"train_range": [round(lo, 3), round(hi, 3)],
                               "minio_range": [round(float(X[c].min()), 3), round(float(X[c].max()), 3)],
                               "pct_rows_outside_and_clipped": round(outside, 1)}
            if outside > 20:
                shift_warnings.append(f"{c}: {outside:.0f}% of MinIO rows outside the training range — "
                                      f"check how this feature is rebuilt from MinIO")
        X[c] = X[c].clip(lo, hi)

    days = np.clip(np.round(model.predict(X)), 1, 365)
    sow_shifted = pd.Series([m[2] for m in meta])
    harvest_dates = sow_shifted + pd.to_timedelta(days, unit="D")
    ordinals = harvest_dates.map(lambda t: t.toordinal()).values
    agg = pd.DataFrame({"gid": pd.Series([m[0] for m in meta]), "ord": ordinals}).groupby("gid", sort=False)["ord"]
    mean_ord, spread = agg.mean().round().astype(int), agg.max() - agg.min()
    sowing = prep["sowing"]
    sow_dates = [sowing[g] for g in mean_ord.index]
    harvest = [pd.Timestamp.fromordinal(int(o)) for o in mean_ord.values]
    out = pd.DataFrame({
        "geozone_id": list(mean_ord.index),
        "sowing_date": [d.date() for d in sow_dates],
        # main answer: how many days from the (real) sowing date until harvest
        "days_to_harvest": [int((h - s.normalize()).days) for h, s in zip(harvest, sow_dates)],
    })
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    expl = explain(run_id, model_name)
    imp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "feature_importance.csv")
    pd.DataFrame(expl["features_by_importance"]).to_csv(imp_path, index=False, encoding="utf-8-sig")
    _log(f"[PREDICT] predictions={len(out):,} | {os.path.abspath(output_path)}")
    return _jsonable({
        "status": "success", "gate_status": gate_status, "model": model_name, "metrics": metrics,
        "days_to_harvest_stats": {"min": int(out["days_to_harvest"].min()), "mean": round(float(out["days_to_harvest"].mean()), 1),
                                  "max": int(out["days_to_harvest"].max())},
        "accuracy_target_met": reg is not None,
        "accuracy_target_days": GATE["mae_days_max"],
        "geozones_predicted": len(out), "sowing_tolerance_days": prep["sowing_tolerance_days"],
        "avg_spread_days_across_shifts": round(float(spread.mean()), 2),
        "geozones_median_filled_per_feature": filled,
        "distribution_shift_clipped": shift_report, "distribution_shift_warnings": shift_warnings,
        "minio": prep["minio"], "output_path": output_path, "feature_importance_csv": imp_path,
        "sample": out.head(10).astype(str).to_dict(orient="records"),
    })


# ---------------------------------------------------------------------------
# Deterministic run (no LLM) — same tools, same order
# ---------------------------------------------------------------------------

def run_all(file_paths: list, use_minio: bool = True) -> None:
    def show(title, obj):
        print(f"\n=== {title} ===")
        print(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, default=str))

    r = prepare_training_data(file_paths, use_minio=use_minio)
    show("PREPARE", r)
    if "error" in r:
        return
    show("PROFILE", profile_dataset())
    r = make_splits()
    show("SPLITS", {k: v for k, v in r.items() if k != "features"})
    if "error" in r:
        return
    show("BASELINE", baseline())
    b = run_benchmark()
    show("BENCHMARK", b["comparison_table_markdown"])
    show("CHECKS", {m["model"]: m["checks"] for m in b["models"]})
    show("GATE", b["gate"])
    for note in b["notes"] + ([b["feedback_retry"]] if b["feedback_retry"] else []):
        print("NOTE:", note)
    show("EXPLAIN", explain(b["run_id"]))
    if b["winner"]:
        show("REGISTER", register_model(b["winner"], b["run_id"]))
    else:
        print("\nAccuracy target not reached — model not registered; forecast uses the best-ranked model.")
    if use_minio:
        p = predict_from_minio()
        show("PREDICT", {k: v for k, v in p.items() if k != "sample"})
        if "sample" in p:
            print("\n" + pd.DataFrame(p["sample"]).to_string(index=False))


# ---------------------------------------------------------------------------
# Agent + CLI
# ---------------------------------------------------------------------------

INSTRUCTION = f"""
You are the harvest prediction agent. The code picks the model, you explain the pick.
You never see raw rows and never fit a model by hand — everything comes from the tools.

NEVER ask the user questions. When the user gives training file paths, run in this order:
prepare_training_data(file_paths) -> profile_dataset -> make_splits -> baseline -> run_benchmark
-> explain(run_id) -> register_model(winner, run_id) ONLY if a winner exists -> predict_from_minio.
predict_from_minio is ALWAYS called (when MinIO was read): it uses the registered model, or the
best-ranked model when the accuracy target was not reached. The user always gets the forecast table.
The first file is the primary file (sowing + harvest dates), the rest are feature files.
Use use_minio=False only if the user explicitly says not to read MinIO.

MUST
- profile_dataset before naming any model
- make_splits before training (season-out, grouped by geozone)
- compute the baseline first; report every model against it
- compare on held-out mae_days only, never a training score
- run explain() on the winner before register_model()

GATE
- ship if mae_days <= threshold and mae_days < baseline.mae_days and no blocking check fires
- else do NOT register; still produce the forecast with the best-ranked model and report its real accuracy

MAY NOT
- change the ranking — the benchmark ranks by held-out MAE
- declare a winner when the benchmark returned none
- quote a training score, or any number that is not in a tool result
- invent an MAE, a feature name, or a threshold

FINAL REPORT — it goes to agronomists, so write it as a clean professional forecast report.
Use plain agronomic language. Do NOT use internal words such as gate, blocked, preview, approved,
registered, run_id, tool, threshold check, benchmark rules. Use ONLY numbers returned by the tools —
never round an error down or claim a better accuracy than the tools report.

Start the report with exactly this title line: "# {REPORT_TITLE}".
Never guess a country, region or crop from file names or ids (e.g. "_az" is NOT Arizona).

## 1. Harvest forecast
- Markdown table of the sample rows: geozone_id | sowing_date | days_to_harvest (only these 3 columns)
- One line: "Forecast file: <output_path> (<geozones_predicted> geozones)."
- One line: "Expected accuracy: ±<metrics.holdout_mae_days> days (mean absolute error on held-out geozones)."
- days_to_harvest range across all geozones (days_to_harvest_stats).

## 2. Model and accuracy
- Which model produced the forecast and why it was chosen (lowest mean error in days).
- Comparison table with columns: Model | Mean error, days | Cross-validation error, days |
  Difference vs GDD rule, days | 90 % of errors below, days. Add the GDD-rule row.
- If accuracy_target_met is false, write EXACTLY this one sentence and nothing else about it:
  "Target accuracy: ±<accuracy_target_days> days; current accuracy: ±<metrics.holdout_mae_days> days."
  Do not mention production, registration, approval, or why a model was or was not chosen for production.
- Describe the comparison with the GDD rule neutrally (numbers only, no words like "comfortably").

## 3. Main factors
- Top features by importance (feature, %, how it is computed from the satellite/weather data),
  described in agronomic terms.

## 4. Data notes
- Seasons used for training and how the model was validated (if only one season: "validated on
  20 % of geozones held out from the same season; a multi-season check will be added when more
  seasons are available").
- Features not available in the satellite/weather data for the forecast season and therefore not used.
- Features whose forecast-season values lie outside the training range (distribution_shift_warnings).
- Assumed definitions (e.g. heat stress day = tmax > 35 °C).

## 5. Next steps to improve accuracy
- The proposed features / data from the tools, phrased as recommendations.

If a tool returns an error, show the exact error text and stop.
"""


def build_agent():
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    return Agent(
        name="harvest_predict_agent",
        model=LiteLlm(model=f"anthropic/{MODEL}", api_key=os.getenv("ANTHROPIC_API_KEY")),
        description="Benchmarks ridge / random forest / XGBoost on days to harvest, gates the winner against a "
                    "GDD baseline, explains it and predicts harvest dates for MinIO geozones.",
        instruction=INSTRUCTION,
        tools=[prepare_training_data, profile_dataset, make_splits, baseline, run_benchmark,
               explain, get_gate, register_model, predict_from_minio],
    )


async def _run_cli():
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    agent = build_agent()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="harvest_predict_agent", user_id="local_user")
    runner = Runner(agent=agent, app_name="harvest_predict_agent", session_service=session_service)

    print("Harvest Predict Agent. Type training file paths (comma or space separated), or 'exit'.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        # If the input is just existing file paths, turn it into a clear instruction for the agent
        paths = [p.strip().strip('"').strip("'") for p in re.split(r"[,\s]+", user_input) if p.strip()]
        if paths and all(os.path.exists(p) for p in paths):
            user_input = f"Run the full workflow on these training files: {json.dumps(paths)}"

        content = types.Content(role="user", parts=[types.Part(text=user_input)])
        try:
            async for event in runner.run_async(user_id="local_user", session_id=session.id, new_message=content):
                if not event.is_final_response():
                    continue
                text = None
                if event.content and event.content.parts:
                    text = "".join(p.text for p in event.content.parts
                                   if getattr(p, "text", None) and not getattr(p, "thought", False))
                print(f"\nAgent: {text}" if text else "\n[!] Final response is empty.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception:
            logger.error("Fatal agent error\n%s", traceback.format_exc())
            print(f"\n[ERROR] full traceback in: {os.path.abspath(LOG_FILE)}")


if __name__ == "__main__":
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Harvest prediction agent")
    ap.add_argument("files", nargs="*", help="training files (with --no-llm)")
    ap.add_argument("--no-llm", action="store_true", help="run the deterministic workflow without the LLM")
    ap.add_argument("--no-minio", action="store_true", help="do not read MinIO (benchmark only)")
    args = ap.parse_args()
    if args.no_llm:
        if not args.files:
            ap.error("--no-llm needs training file paths")
        try:
            run_all(args.files, use_minio=not args.no_minio)
        except Exception:
            logger.error("Fatal pipeline error\n%s", traceback.format_exc())
            print(f"\n[ERROR] full traceback in: {os.path.abspath(LOG_FILE)}")
    else:
        asyncio.run(_run_cli())