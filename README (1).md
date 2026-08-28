# Corpus construction

CPIoT-XAD2025 is built from two public corpora that are **not** redistributed
here:

| Source | Contents | Access |
|---|---|---|
| N-BaIoT | IoT botnet network traffic | UCI Machine Learning Repository, public download |
| SWaT | Water-treatment process telemetry | iTrust, Singapore University of Technology and Design — **request-based research agreement** |

SWaT cannot be redistributed by us under the terms of that agreement. This
directory therefore holds the code needed to rebuild the fused corpus once you
have obtained both sources yourself.

## Contents

| File | Purpose |
|---|---|
| `build_corpus.py` | Builds the fused CSV from the two source corpora. Released exactly as run. |
| `verify_alignment.py` | Checks a fused CSV against the structural properties reported in the paper. Does not build anything. |

## Before you run it

`build_corpus.py` is the script that produced the corpus that was evaluated,
released unmodified rather than tidied, so that a rebuild cannot silently
diverge from what the paper reports. Two consequences follow.

First, the input and output locations are hard-coded constants at the top of
the file (`NB_IOT_FOLDER`, `SWAT_FOLDER`, `OUT_DIR`). Edit those three lines to
point at your own copies before running; nothing else needs to change.

Second, the script was run under Google Colab against Drive-mounted folders and
requires `numpy`, `pandas` and `tqdm` only.

```bash
python fusion/build_corpus.py
```

## What the script does

**SWaT ingest.** Every CSV in `SWAT_FOLDER` is examined independently: the
loader tries each combination of delimiter (`,`, `;`, tab, `|`) and leading
junk-row count (0, 1, 2) on a 30-row sample and keeps whichever combination
parses the largest fraction of timestamps, preferring a named timestamp column
over the positional first column. A file whose best combination parses under
80% of its sample (`MIN_FILE_PARSE_FRAC`) aborts the run with a diagnostic
rather than being silently dropped. Timestamps are then parsed by a cascade of
twenty explicit formats, followed by an ISO-8601 pass, a day-first fallback and
an Excel-serial fallback; blank and placeholder rows are counted separately from
genuine failures and do not count toward the gate. Rows whose timestamps do not
parse are dropped. If more than 1% of non-empty timestamps fail
(`MAX_TS_PARSE_FAIL_FRAC`) the run aborts and a census of the offending raw
values is written to `unparsed_timestamps.csv`.

Labels are coerced to binary: numeric label columns are thresholded at zero,
string columns are whitespace-stripped and matched on the substrings `attack`
or `anomaly` (this is what handles the `A ttack` token in the SWaT attack file).
A SWaT file with no recognised label column is treated as a normal-operation run
and assigned `label = 0` throughout. Feature columns are salvaged to float, with
`Bad Input`, `NA`, `inf` and similar tokens mapped to NaN; columns that are
entirely NaN are dropped. Files are concatenated on the union of their columns
and sorted by timestamp.

`DOWNSAMPLE_SECONDS = 1` in the released configuration, which is a no-op — every
SWaT row is kept. Temporal resampling runs only if that constant is raised
above 1.

**N-BaIoT aggregation.** The N-BaIoT files are counted, then re-read and
aggregated into `T = |SWaT|` bins, where a row's bin is
`clip(int(idx_global / total_rows * T), 0, T-1)` — that is, bins are equal-count
partitions of the concatenated file order, assigned by **row index and not by
any timestamp**, since N-BaIoT carries no usable clock. Each bin holds the mean
of the numeric columns over the rows falling in it; a bin with no finite value
for a column receives NaN. The numeric column set is determined from the first
50 rows of the first file.

**Fusion.** The aggregated network frame is given a timestamp axis of `T`
equally spaced instants spanning the SWaT timestamp range
(`pd.date_range(t0, t1, periods=T)`), columns are suffixed `_nbaiot` and
`_swat`, and the two frames are concatenated positionally. The SWaT timestamp
becomes the canonical `timestamp` column and the SWaT label becomes `label`;
the synthetic network timestamp is dropped.

**No feature scaling is applied.** The fused CSV holds raw units. All
normalisation — robust per-feature scaling, `log1p`, and the scalar
standardisation of stream scores — is performed inside the training pipeline
and fitted on training-split statistics only.

**Missing values are exported as NaN.** `POST_FUSION_MISSING_POLICY = "none"`;
the alternative median-fill branch is deliberately not used, because a global
median computed over the whole corpus would leak validation and test statistics
into the training split. Imputation is the pipeline's responsibility and is
fitted train-only.

**Audit gates.** Before writing, the fused label vector is audited for
contiguous anomaly episodes. The run aborts if fewer than 10 episodes are found
(`MIN_ANOMALY_EPISODES`), or earlier if fewer than 1000 anomalous rows survive
SWaT cleaning (`MIN_ANOMALY_ROWS`).

## Outputs

All are written to `OUT_DIR`:

| File | Contents |
|---|---|
| `CPIoT-XAD2025_fused_hybrid_v5.csv` | The fused corpus. This is the pipeline input. |
| `SWaT_clean_numeric.csv` | The cleaned, timestamp-sorted SWaT frame before fusion. |
| `NBaiot_agg_to_T.csv` | The binned network features before fusion. |
| `fusion_report_v5.txt` | Shapes, per-stream column counts, anomalous-row and episode counts, episode lengths, label counts, time range. |
| `unparsed_timestamps.csv` | Written only when some timestamps failed to parse: a census of the offending raw values by file. |

## Fused schema

Each row is one aligned timestep on the SWaT clock:

- network feature columns with the suffix `_nbaiot`
- process feature columns with the suffix `_swat`
- one binary label column (`label`), taken from SWaT
- one timestamp column (`timestamp`), taken from SWaT

This is exactly the input the training pipeline expects, so the output of this
directory feeds directly into `python -m cpiot_xad.cli --data <fused.csv>`.

The fused CSV retains every numeric column that survived ingest. The reduction
to **115 network and 45 process features** reported in the paper is applied by
the training pipeline's low-variance filter, not here.

## A note on the temporal pairing

N-BaIoT and SWaT were recorded independently, so the alignment this script
produces is positional rather than causal: network bin *t* has no physical
relationship to sensor row *t*. This is stated plainly in the paper and is the
reason the modality ablation is reported. The script is released partly so that
this construction can be inspected directly rather than inferred.

## Verifying a rebuild

```bash
python fusion/verify_alignment.py --data CPIoT-XAD2025_fused_hybrid_v5.csv
```

It reports feature counts per stream, label prevalence, timestamp regularity,
anomaly-episode structure after burst merging, and whether the corpus supports
the episode-aware split configuration used in the paper. Exit status is 0 when
every check passes. Add `--json diagnostics.json` to save a machine-readable
record.

A rebuild that matches the evaluated corpus should report 115 network and 45
process features after the low-variance filter, and a row-level anomaly
prevalence consistent with the window-level prevalence of 9.68% quoted in the
paper. The "no missing feature values" check is expected to emit a **warning**
rather than a pass, since NaN export is intentional.
