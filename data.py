"""Corpus loading, label detection and dataset auditing."""

import gc

import numpy as np
import pandas as pd

from .config import CFG


def find_label_column(df):
    """Locate the binary anomaly label column.

    Honours ``FORCE_LABEL_COL`` first, then a list of common names, then falls
    back to any object column whose values look like attack/anomaly/normal
    tokens. Returns ``None`` when nothing plausible is found.
    """
    if CFG["FORCE_LABEL_COL"] and CFG["FORCE_LABEL_COL"] in df.columns:
        return CFG["FORCE_LABEL_COL"]
    for c in CFG["LABEL_CANDIDATES"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            vals = set(str(x).lower() for x in df[c].dropna().unique()[:100])
            if any(tok in vals for tok in ("attack", "anomaly", "normal")):
                return c
    return None


def coerce_label_series(s):
    """Map an arbitrary label column onto {0, 1}."""
    try:
        arr = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
        if set(np.unique(arr)) <= {0.0, 1.0}:
            return arr.astype(int)
        return (arr > 0).astype(int)
    except Exception:
        pass
    return (s.fillna("").astype(str).str.lower()
              .map(lambda x: 1 if ("attack" in x or "anomaly" in x) else 0)
              .astype(int))


def audit_dataset(df, nb_cols, sw_cols, label_col):
    """Print the corpus shape, label balance and episode count before training.

    The episode count matters: the split allocator needs at least three
    separated anomaly bursts to give every split some anomaly mass.
    """
    print("\n================ DATASET AUDIT ================")
    print("[AUDIT] Rows:", len(df), "Cols:", df.shape[1])
    print("[AUDIT] Label distribution (rows):")
    print(df[label_col].value_counts(dropna=False))
    print(f"[AUDIT] {CFG['NET_SUFFIX']} numeric cols:", len(nb_cols))
    print(f"[AUDIT] {CFG['SEN_SUFFIX']} numeric cols:", len(sw_cols))
    y = df[label_col].to_numpy()
    n_episodes = int((y[0] == 1)) + int(np.sum((y[1:] == 1) & (y[:-1] == 0)))
    print(f"[AUDIT] Anomaly episodes (contiguous runs of 1s): {n_episodes}")
    print("================================================\n")


def filter_low_variance(df, cols, thr=1e-12):
    """Drop constant columns, which contribute nothing and destabilise scaling."""
    if len(cols) == 0:
        return cols
    var = df[cols].var(numeric_only=True)
    return [c for c in cols if (var.get(c, 0.0) > thr)]


def load_corpus(data_path):
    """Read the fused corpus and return everything downstream stages need.

    Returns
    -------
    X_np : (N, F) float32 array of features, network stream first
    y_rows : (N,) int array of row labels
    nb_cols, sw_cols : the retained column names of each stream
    """
    df = pd.read_csv(data_path)
    df.dropna(axis=1, how="all", inplace=True)

    label_col = find_label_column(df)
    if label_col is None:
        raise SystemExit("[ERROR] No label column found. Set FORCE_LABEL_COL.")
    df[label_col] = coerce_label_series(df[label_col])

    net_sfx = CFG["NET_SUFFIX"].lower()
    sen_sfx = CFG["SEN_SUFFIX"].lower()
    all_cols = list(df.columns)
    nb_cols = [c for c in all_cols
               if str(c).lower().endswith(net_sfx)
               and pd.api.types.is_numeric_dtype(df[c])]
    sw_cols = [c for c in all_cols
               if str(c).lower().endswith(sen_sfx)
               and pd.api.types.is_numeric_dtype(df[c])]
    audit_dataset(df, nb_cols, sw_cols, label_col)

    if len(nb_cols) == 0 or len(sw_cols) == 0:
        raise SystemExit("[ERROR] Hybrid features not found - fix the "
                         "fusion/export stage first.")

    nb_cols = filter_low_variance(df, nb_cols)
    sw_cols = filter_low_variance(df, sw_cols)
    if len(nb_cols) == 0 or len(sw_cols) == 0:
        raise SystemExit("[ERROR] Hybrid columns vanished after low-variance "
                         "filtering.")

    features = nb_cols + sw_cols
    print(f"[INFO] Network ({CFG['NET_SUFFIX']}) columns: {len(nb_cols)} | "
          f"Process ({CFG['SEN_SUFFIX']}) columns: {len(sw_cols)}")

    y_rows = df[label_col].to_numpy().astype(int)
    X_np = (df[features].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32))     # one (N, F) float32 array
    del df
    gc.collect()
    return X_np, y_rows, nb_cols, sw_cols
