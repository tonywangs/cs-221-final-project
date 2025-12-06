from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

CATEGORICAL_FEATURES = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
NUMERIC_FEATURES = ['age','trestbps','chol','thalach','oldpeak']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'num'

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Convert '?' to NaN, coerce columns to numeric
    df = df.replace('?', np.nan)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            # Keep non-numeric columns as-is
            pass
    # Some UCI loads come as float with .0 for categorical; cast later
    return df

def load_uci_heart(source: str = "ucimlrepo",
                   local_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Returns X (features df), y (Series), and metadata dict.
    Tries ucimlrepo; falls back to local CSV if provided or './heart.csv'.
    """
    df = None
    meta = {"source": None}
    if source == "ucimlrepo":
        try:
            from ucimlrepo import fetch_ucirepo
            heart = fetch_ucirepo(id=45)
            # try to pull a canonical table; if not available, fallback to URL
            if hasattr(heart, "data") and hasattr(heart.data, "original"):
                raw = heart.data.original
                if isinstance(raw, pd.DataFrame):
                    df = raw.copy()
            if df is None and "data_url" in heart.metadata:
                df = pd.read_csv(heart.metadata['data_url'])
            meta["source"] = "ucimlrepo"
        except Exception as e:
            df = None
    if df is None:
        # local fallback
        path = local_path or "./heart.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not load dataset. Install 'ucimlrepo' or place a CSV at {path}")
        df = pd.read_csv(path)
        meta["source"] = "local"

    df = _coerce_numeric(df)
    # column alignment: some variants use different column names; normalize to Cleveland schema
    expected = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
    # If df already has these, great. If not, try to map via case-insensitive match.
    if set(expected) <= set(df.columns):
        df = df[expected]
    else:
        lower = {c.lower(): c for c in df.columns}
        if set(expected) <= set(lower.keys()):
            df = df[[lower[c] for c in expected]]
            df.columns = expected
        else:
            raise ValueError("Input CSV columns do not match UCI Cleveland schema.")

    # Final coercions - fill NaN in categorical features before converting to int
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Fill NaN with most frequent value (mode) for categorical features
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                # If all NaN, fill with 0
                df[col] = df[col].fillna(0)
            df[col] = df[col].astype(float).round().astype(int)
    
    X = df.drop(columns=[TARGET]).copy()
    # Fill NaN in target if any, then convert to int
    if df[TARGET].isna().any():
        # For target, use forward fill then backward fill, or 0 as last resort
        df[TARGET] = df[TARGET].ffill().bfill().fillna(0)
    y = df[TARGET].astype(int).copy()
    return X, y, meta

def make_targets(y: pd.Series, task: str = "binary") -> pd.Series:
    if task == "binary":
        return (y > 0).astype(int)
    elif task == "multiclass":
        return y.astype(int)
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")
