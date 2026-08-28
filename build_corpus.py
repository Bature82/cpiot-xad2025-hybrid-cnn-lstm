# CPIoT-XAD2025 Re-Fusion v5 (SELF-DIAGNOSING LOADER)
#
# WHY v5: v4.1 still hit "50.5% of timestamps failed" - one SWaT file is
# structured differently from the other. Instead of guessing formats one
# at a time, v5 AUTO-DETECTS, PER FILE:
#
#   1. the delimiter               (",", ";", tab, "|")
#   2. the number of junk rows     (SWaT exports often carry a stage-name
#      above the real header        row like "P1,P1,P2,..." or a blank
#                                   line ABOVE the actual column header -
#                                   then every 'timestamp' is garbage)
#   3. the timestamp column        (by name, or column 0 as fallback)
#   4. the timestamp format        (cascade: day-first 12h/24h, US
#                                   month-first 12h/24h, milliseconds,
#                                   ISO, generic day-first, Excel serials;
#                                   NBSP/multi-space cleaned first)
#
# It scores every (delimiter x skiprows) combination by HOW MANY SAMPLE
# TIMESTAMPS ACTUALLY PARSE and uses the best one for that file. If a
# file still cannot reach 80% parse success, the run stops BEFORE fusing
# and prints that file's raw header and sample values so the cause is
# visible instead of hidden.
#
# Everything else from v4/v4.1 is kept:
#   - 'A ttack' label typo fix (whitespace stripped before matching)
#   - no global median imputation (NaNs exported; pipeline imputes
#     train-only)
#   - quality gates: <=1% parse failures, >=1000 anomalous rows,
#     >=10 anomaly episodes - the export refuses to produce another
#     unusable corpus
#   - optional episode-preserving downsampling (DOWNSAMPLE_SECONDS)

import os, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------- Paths --------------------
NB_IOT_FOLDER = "/content/drive/MyDrive/Fusion/N-BaIoT Dataset/N-BaIoT Dataset"
SWAT_FOLDER   = "/content/drive/MyDrive/Fusion/SWaT.A1 & A2_Dec 2015/Physical"
OUT_DIR = "/content/drive/MyDrive/Fusion/fused_output_v5"
os.makedirs(OUT_DIR, exist_ok=True)

SWAT_CLEAN_CSV = os.path.join(OUT_DIR, "SWaT_clean_numeric.csv")
NB_AGG_CSV     = os.path.join(OUT_DIR, "NBaiot_agg_to_T.csv")
FUSED_CSV      = os.path.join(OUT_DIR, "CPIoT-XAD2025_fused_hybrid_v5.csv")
REPORT_TXT     = os.path.join(OUT_DIR, "fusion_report_v5.txt")

# -------------------- Config --------------------
CHUNK_SIZE = 50000
TS_CANDIDATES = ["timestamp", "t_stamp", "tstamp", "time", "date", "datetime"]
LABEL_CANDIDATES = ["label", "attack", "is_anomaly", "anomaly", "normal/attack", "normalattack"]

DOWNSAMPLE_SECONDS = 1        # your corpus is ~31k rows (Jul-2019 collection) - keep every row.
                              # Set to 5 only if you later switch to the Dec-2015 data (~945k rows).
POST_FUSION_MISSING_POLICY = "none"   # export NaNs; pipeline imputes train-only

# Quality gates
MAX_TS_PARSE_FAIL_FRAC = 0.01
MIN_FILE_PARSE_FRAC    = 0.80   # per-file: below this the file is declared unusable
MIN_ANOMALY_EPISODES   = 10
MIN_ANOMALY_ROWS       = 1000

SEPS      = [",", ";", "\t", "|"]
SKIPROWS  = [0, 1, 2]           # junk rows above the real header
TS_FORMATS = [
    "%d/%m/%Y %I:%M:%S %p",     # 28/12/2015 10:29:14 AM   (SWaT native)
    "%d/%m/%Y %H:%M:%S",        # 28/12/2015 22:29:14
    "%m/%d/%Y %I:%M:%S %p",     # 12/28/2015 10:29:14 AM   (Excel US locale)
    "%m/%d/%Y %H:%M:%S",        # 12/28/2015 22:29:14
    "%d/%m/%Y %I:%M:%S.%f %p",  # with milliseconds
    "%Y-%m-%d %H:%M:%S",        # ISO
    "%d/%m/%y %I:%M:%S %p",     # two-digit year variants
    "%d/%m/%y %H:%M:%S",
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%y %H:%M:%S",
    "%d/%m/%Y %I:%M %p",        # no seconds
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",        # dash-separated
    "%Y-%m-%dT%H:%M:%SZ",       # ISO 8601 with T separator + Z suffix (SWaT Jul-2019)
    "%Y-%m-%dT%H:%M:%S",        # ISO 8601 with T separator
    "%Y-%m-%dT%H:%M:%S.%fZ",    # ISO 8601 with milliseconds + Z
    "%d/%m/%Y",                 # date only (midnight rows)
    "%m/%d/%Y",
]

# -------------------- Small helpers --------------------
def list_csv_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv")])

def strip_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_by_name(cols, candidates):
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None

def clean_ts_strings(series: pd.Series) -> pd.Series:
    """Normalise whitespace: NBSP -> space, collapse runs, strip, uppercase
    (so 'am'/'pm' match %p)."""
    s = series.fillna("").astype(str)   # fillna first: modern pandas keeps NaN through astype(str)
    s = s.str.replace(" ", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
    return s

EMPTY_TOKENS = {"", "NAN", "NONE", "NULL", "NA", "N/A", "NAT"}

def parse_ts_cascade(series: pd.Series):
    """Multi-format cascade. Returns (parsed, n_failed, n_empty, failing_samples).
    Empty/placeholder rows (blank lines, 'nan') are separated from real
    parse failures - they are dropped but do NOT count toward the gate."""
    s = clean_ts_strings(series)
    empty = s.isin(EMPTY_TOKENS)
    ts = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m = ~empty
    for fmt in TS_FORMATS:
        if not m.any():
            break
        ts.loc[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
        m = ts.isna() & ~empty
    if m.any():   # ISO 8601 catch-all (mixed offsets/fractions), tz stripped
        try:
            iso = pd.to_datetime(s[m], format="ISO8601", utc=True, errors="coerce")
            ts.loc[m] = iso.dt.tz_convert(None)
        except (ValueError, TypeError):
            pass
        m = ts.isna() & ~empty
    if m.any():   # generic day-first (handles remaining oddities)
        ts.loc[m] = pd.to_datetime(s[m], dayfirst=True, errors="coerce")
        m = ts.isna() & ~empty
    if m.any():   # Excel serial dates from xlsx->csv exports
        num = pd.to_numeric(s[m], errors="coerce")
        ok = num.notna() & (num > 20000) & (num < 60000)
        if ok.any():
            ts.loc[num[ok].index] = pd.to_datetime(num[ok], unit="D", origin="1899-12-30")
        m = ts.isna() & ~empty
    samples = list(pd.unique(s[m]))[:10] if m.any() else []
    return ts, int(m.sum()), int(empty.sum()), samples

def coerce_binary_label(series):
    """'A ttack' fix: strip ALL whitespace before matching."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().any():
        x = x.fillna(0)
        u = set(np.unique(x.values))
        if u.issubset({0, 1}):
            return x.astype(int)
        return (x > 0).astype(int)
    s = (series.fillna("").astype(str).str.lower()
               .str.replace(r"\s+", "", regex=True))
    return s.map(lambda v: 1 if ("attack" in v or "anomaly" in v) else 0).astype(int)

def salvage_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = s.astype(str).str.strip().str.lower()
    x = x.replace({
        "": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan, "na": np.nan, "n/a": np.nan,
        "bad input": np.nan, "badinput": np.nan, "inf": np.nan, "+inf": np.nan, "-inf": np.nan
    })
    x = x.str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce").astype(float)

def episode_audit(labels):
    y = np.asarray(labels).astype(int)
    d = np.flatnonzero(np.diff(np.r_[0, y, 0]))
    eps = list(zip(d[0::2], d[1::2]))
    gaps = [eps[i+1][0] - eps[i][1] for i in range(len(eps) - 1)] if len(eps) > 1 else []
    return eps, gaps

# -------------------- Per-file structure detection --------------------
def detect_read_config(fp):
    """Try every (separator x skiprows) combination on a sample and score it
    by the fraction of sample timestamps that actually parse. Returns the
    best config dict, or one with parse_frac=0 if nothing works."""
    best = {"sep": ",", "skiprows": 0, "ts_col": None, "positional_ts": False,
            "parse_frac": 0.0, "ncol": 0, "samples": [], "header_preview": []}
    for sep in SEPS:
        for skip in SKIPROWS:
            try:
                df = pd.read_csv(fp, nrows=30, sep=sep, skiprows=skip, engine="python")
            except Exception:
                continue
            df = strip_cols(df)
            df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]
            if df.shape[1] < 3 or len(df) == 0:
                continue
            # candidate timestamp column: by name, else column 0
            ts_col = find_by_name(df.columns, TS_CANDIDATES)
            positional = False
            if ts_col is None:
                ts_col = df.columns[0]
                positional = True
            ts, nfail, nempty, samples = parse_ts_cascade(df[ts_col])
            frac = 1.0 - nfail / max(len(df) - nempty, 1)
            # prefer: higher parse fraction; then named ts col; then more cols
            key = (frac, 0 if positional else 1, min(df.shape[1], 500))
            best_key = (best["parse_frac"], 0 if best["positional_ts"] else 1, min(best["ncol"], 500))
            if key > best_key:
                best = {"sep": sep, "skiprows": skip, "ts_col": ts_col,
                        "positional_ts": positional, "parse_frac": frac,
                        "ncol": df.shape[1], "samples": samples,
                        "header_preview": [str(c) for c in df.columns[:8]]}
    return best

# -------------------- SWaT loader (v5) --------------------
def load_swat_clean_numeric():
    files = list_csv_files(SWAT_FOLDER)
    if not files:
        raise FileNotFoundError(f"No SWaT CSVs in {SWAT_FOLDER}")

    # ---- 1) detect per-file structure ----
    configs, unusable = {}, {}
    print("[SWaT] Detecting per-file structure (delimiter, junk rows, timestamp format)...")
    for fp in files:
        cfg = detect_read_config(fp)
        fname = os.path.basename(fp)
        configs[fp] = cfg
        print(f"  {fname}: sep={cfg['sep']!r} skiprows={cfg['skiprows']} "
              f"ts_col={cfg['ts_col']!r}{' (positional)' if cfg['positional_ts'] else ''} "
              f"sample-parse={cfg['parse_frac']:.0%} cols={cfg['ncol']}")
        if cfg["parse_frac"] < MIN_FILE_PARSE_FRAC:
            unusable[fname] = cfg

    if unusable:
        msg = ["The following SWaT file(s) could not be parsed under ANY "
               "delimiter/skiprows/format combination:"]
        for fname, cfg in unusable.items():
            msg.append(f"  - {fname}: best sample-parse={cfg['parse_frac']:.0%}, "
                       f"header preview={cfg['header_preview']}, "
                       f"unparsed samples={cfg['samples']}")
        msg.append("Open one of these files and inspect its first lines - the "
                   "header preview and samples above show what the loader saw.")
        raise RuntimeError("\n".join(msg))

    # ---- 2) stream-load each file with ITS OWN config ----
    chunks, total_rows_read, total_ts_failed, total_empty = [], 0, 0, 0
    per_file = {}
    unparsed_counter = {}   # raw string -> count (capped)
    for fp in tqdm(files, desc="Loading SWaT (v5.1)"):
        cfg = configs[fp]
        fname = os.path.basename(fp)
        per_file[fname] = [0, 0, 0, []]   # rows, failed, empty, samples
        for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, sep=cfg["sep"],
                                 skiprows=cfg["skiprows"], engine="python"):
            chunk = strip_cols(chunk)
            chunk = chunk.loc[:, [c for c in chunk.columns if not str(c).lower().startswith("unnamed")]]
            if cfg["ts_col"] in chunk.columns:
                ts_col = cfg["ts_col"]
            elif cfg["positional_ts"] and chunk.shape[1] > 0:
                ts_col = chunk.columns[0]
            else:
                continue
            chunk = chunk.rename(columns={ts_col: "timestamp"})
            raw_ts = chunk["timestamp"].copy()
            total_rows_read += len(chunk)
            per_file[fname][0] += len(chunk)
            chunk["timestamp"], n_failed, n_empty, samples = parse_ts_cascade(chunk["timestamp"])
            total_ts_failed += n_failed
            total_empty += n_empty
            per_file[fname][1] += n_failed
            per_file[fname][2] += n_empty
            if n_failed and len(unparsed_counter) < 1000:
                bad = clean_ts_strings(raw_ts)[chunk["timestamp"].isna()]
                for v, c in bad.value_counts().head(50).items():
                    if v not in EMPTY_TOKENS:
                        unparsed_counter[(fname, v)] = unparsed_counter.get((fname, v), 0) + int(c)
            if samples and len(per_file[fname][3]) < 10:
                per_file[fname][3] = (per_file[fname][3] + samples)[:10]
            chunk = chunk.dropna(subset=["timestamp"])
            label_col = find_by_name(chunk.columns, LABEL_CANDIDATES)
            if label_col is not None:
                chunk["label"] = coerce_binary_label(chunk[label_col])
                if label_col != "label":
                    chunk = chunk.drop(columns=[label_col])
            else:
                chunk["label"] = 0    # normal-run file
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("SWaT produced no data after timestamp parsing.")

    report_lines = []
    for fname, (rows, failed, empty, samples) in per_file.items():
        denom = max(rows - empty, 1)
        line = f"{fname}: rows={rows} empty={empty} failed={failed} ({failed / denom:.2%} of non-empty)"
        if samples:
            line += f" | UNPARSED SAMPLES: {samples}"
        report_lines.append(line)
    print("\n[SWaT] Per-file timestamp parse report:")
    for line in report_lines:
        print("  " + line)

    # persist the full unparsed-value census for offline inspection
    if unparsed_counter:
        up = pd.DataFrame([(f, v, c) for (f, v), c in unparsed_counter.items()],
                          columns=["file", "raw_value", "count"]).sort_values("count", ascending=False)
        up_path = os.path.join(OUT_DIR, "unparsed_timestamps.csv")
        up.to_csv(up_path, index=False)
        print(f"[SWaT] Unparsed-value census written to: {up_path}")

    fail_frac = total_ts_failed / max(total_rows_read - total_empty, 1)
    print(f"[SWaT] rows read: {total_rows_read} | empty: {total_empty} | "
          f"parse failures: {total_ts_failed} ({fail_frac:.3%} of non-empty)")
    if fail_frac > MAX_TS_PARSE_FAIL_FRAC:
        # diagnostics INSIDE the exception so a traceback paste carries them
        raise RuntimeError(
            f"{fail_frac:.1%} of SWaT timestamps failed to parse.\n"
            f"PER-FILE REPORT:\n  " + "\n  ".join(report_lines) +
            "\nFull census of unparsed values saved to "
            f"{os.path.join(OUT_DIR, 'unparsed_timestamps.csv')} - "
            "share the UNPARSED SAMPLES above to get the format added.")

    # concat aligns different schemas across files (union of columns, NaN-filled)
    sw = pd.concat(chunks, ignore_index=True)
    sw = strip_cols(sw).sort_values("timestamp").reset_index(drop=True)

    feat_cols = [c for c in sw.columns if c not in ("timestamp", "label")]
    for c in feat_cols:
        sw[c] = salvage_numeric_series(sw[c])
    sw.dropna(axis=1, how="all", inplace=True)

    sw_feat = [c for c in sw.columns if c not in ("timestamp", "label")]
    n_anom = int(sw["label"].sum())
    print("[SWaT] Clean shape:", sw.shape, "| sensor features:", len(sw_feat),
          "| anomalous rows:", n_anom)
    if len(sw_feat) == 0:
        raise RuntimeError("SWaT cleaning resulted in NO sensor feature columns.")
    if n_anom < MIN_ANOMALY_ROWS:
        raise RuntimeError(
            f"Only {n_anom} anomalous rows survived cleaning (expected tens of "
            f"thousands from the SWaT attack file). Verify the attack CSV is in "
            f"SWAT_FOLDER and its label column is present.")

    if DOWNSAMPLE_SECONDS and DOWNSAMPLE_SECONDS > 1:
        print(f"[SWaT] Downsampling to {DOWNSAMPLE_SECONDS}-second bins ...")
        sw = sw.set_index("timestamp")
        agg = {c: "mean" for c in sw.columns if c != "label"}
        agg["label"] = "max"
        sw = sw.resample(f"{DOWNSAMPLE_SECONDS}s").agg(agg)
        sw = sw.dropna(how="all").reset_index()
        sw = sw.dropna(subset=[c for c in sw.columns if c != "timestamp"], how="all")
        sw["label"] = sw["label"].fillna(0).astype(int)
        print("[SWaT] After downsampling:", sw.shape, "| anomalous rows:", int(sw["label"].sum()))
    return sw

# -------------------- N-BaIoT count + aggregate --------------------
def count_nbaiot_rows():
    nb_files = list_csv_files(NB_IOT_FOLDER)
    if not nb_files:
        raise FileNotFoundError(f"No N-BaIoT CSVs in {NB_IOT_FOLDER}")
    total = 0
    for fp in tqdm(nb_files, desc="Counting N-BaIoT rows"):
        for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, low_memory=False):
            total += len(chunk)
    return total

def aggregate_nbaiot_to_bins(total_rows, target_bins):
    nb_files = list_csv_files(NB_IOT_FOLDER)
    sample = strip_cols(pd.read_csv(nb_files[0], nrows=50, low_memory=False))
    sample_num = sample.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    numeric_cols = sample_num.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns detected in N-BaIoT sample.")
    bin_sums   = {c: np.zeros(target_bins, dtype=np.float64) for c in numeric_cols}
    bin_counts = {c: np.zeros(target_bins, dtype=np.int64) for c in numeric_cols}
    current_index = 0
    for fp in tqdm(nb_files, desc="Aggregating N-BaIoT"):
        for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = strip_cols(chunk)
            for c in numeric_cols:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            n = len(chunk)
            if n == 0:
                continue
            idx_global = np.arange(current_index, current_index + n)
            bin_idx = np.clip((idx_global / total_rows * target_bins).astype(int),
                              0, target_bins - 1)
            for col in numeric_cols:
                if col in chunk.columns:
                    vals = chunk[col].to_numpy(dtype=np.float64)
                    mask = np.isfinite(vals)
                    if mask.any():
                        bin_sums[col] += np.bincount(bin_idx[mask], weights=vals[mask], minlength=target_bins)
                        bin_counts[col] += np.bincount(bin_idx[mask], minlength=target_bins)
            current_index += n
    agg = {c: np.divide(bin_sums[c], bin_counts[c],
                        out=np.full(target_bins, np.nan), where=bin_counts[c] != 0)
           for c in numeric_cols}
    return pd.DataFrame(agg)

# -------------------- Fuse --------------------
def fuse(nb_agg, swat_df):
    T = len(swat_df)
    if len(nb_agg) != T:
        raise ValueError(f"Mismatch: nb_agg={len(nb_agg)} vs swat={T}")
    t0, t1 = swat_df["timestamp"].min(), swat_df["timestamp"].max()
    nb_agg = nb_agg.copy()
    nb_agg["timestamp"] = pd.date_range(start=t0, end=t1, periods=T)
    nb_agg = nb_agg.add_suffix("_nbaiot")
    sw = swat_df.add_suffix("_swat")
    fused = strip_cols(pd.concat([nb_agg.reset_index(drop=True),
                                  sw.reset_index(drop=True)], axis=1))
    fused = fused.rename(columns={"timestamp_swat": "timestamp"})
    if "timestamp_nbaiot" in fused.columns:
        fused.drop(columns=["timestamp_nbaiot"], inplace=True)
    if "label_swat" in fused.columns:
        fused["label"] = fused["label_swat"].astype(int)
        fused.drop(columns=["label_swat"], inplace=True)

    feat_cols = [c for c in fused.columns if c not in ("timestamp", "label")]
    for c in feat_cols:
        fused[c] = pd.to_numeric(fused[c], errors="coerce")

    if POST_FUSION_MISSING_POLICY == "median":
        print("[WARN] Global median fill leaks val/test statistics - "
              "exporting NaNs instead is recommended.")
        med = fused[feat_cols].median(numeric_only=True)
        fused[feat_cols] = fused[feat_cols].fillna(med)

    nb_cols = [c for c in fused.columns if str(c).lower().endswith("_nbaiot")]
    sw_cols = [c for c in fused.columns if str(c).lower().endswith("_swat")]
    if len(nb_cols) == 0 or len(sw_cols) == 0:
        raise RuntimeError(f"Hybrid FAIL: nbaiot={len(nb_cols)}, swat={len(sw_cols)}")
    return fused, nb_cols, sw_cols

# -------------------- Main --------------------
def main():
    print("=== 1) Load SWaT (v5: self-diagnosing loader) ===")
    swat_df = load_swat_clean_numeric()
    swat_df.to_csv(SWAT_CLEAN_CSV, index=False)
    T = len(swat_df)
    print("[INFO] T =", T, "| range:", swat_df["timestamp"].min(), "->", swat_df["timestamp"].max())

    print("\n=== 2) Count N-BaIoT rows ===")
    total_nb = count_nbaiot_rows()

    print("\n=== 3) Aggregate N-BaIoT into T bins ===")
    nb_agg = aggregate_nbaiot_to_bins(total_nb, T)
    nb_agg.to_csv(NB_AGG_CSV, index=False)

    print("\n=== 4) Fuse ===")
    fused, nb_cols, sw_cols = fuse(nb_agg, swat_df)

    eps, gaps = episode_audit(fused["label"].values)
    n_anom = int(fused["label"].sum())
    print(f"[AUDIT] anomalous rows: {n_anom} | episodes: {len(eps)} | "
          f"median gap: {int(np.median(gaps)) if gaps else 'n/a'} rows")
    if len(eps) < MIN_ANOMALY_EPISODES:
        raise RuntimeError(
            f"Only {len(eps)} anomaly episodes in the fused corpus (need >= "
            f"{MIN_ANOMALY_EPISODES}; SWaT contains 36 attacks). DO NOT train "
            f"on this export.")

    fused.to_csv(FUSED_CSV, index=False)
    print("[OK] Saved fused hybrid:", FUSED_CSV)

    with open(REPORT_TXT, "w") as f:
        f.write("=== CPIoT-XAD2025 FUSION REPORT v5 ===\n")
        f.write(f"SWaT clean shape: {swat_df.shape}\n")
        f.write(f"Downsample seconds: {DOWNSAMPLE_SECONDS}\n")
        f.write(f"Fused shape: {fused.shape}\n")
        f.write(f"NBaiot cols: {len(nb_cols)} | SWaT cols: {len(sw_cols)}\n")
        f.write(f"Anomalous rows: {n_anom}\n")
        f.write(f"Anomaly episodes: {len(eps)}\n")
        f.write(f"Episode lengths (rows): {[e - s for s, e in eps]}\n")
        f.write(f"Label counts: {fused['label'].value_counts(dropna=False).to_dict()}\n")
        f.write(f"Time range: {fused['timestamp'].min()} -> {fused['timestamp'].max()}\n")
    print("[OK] Report saved:", REPORT_TXT)
    print("\nDONE - expected audit: ~30+ episodes, tens of thousands of anomalous rows")
    gc.collect()

if __name__ == "__main__":
    main()
