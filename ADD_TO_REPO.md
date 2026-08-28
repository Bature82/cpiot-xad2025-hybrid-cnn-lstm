# Adding the missing parts to `cpiot-xad2025-hybrid-cnn-lstm`

The repository currently contains the training and evaluation pipeline only.
The paper's data-availability statement also promises the fusion pipeline, the
alignment diagnostics and the per-run result tables. This package now supplies
all three, so once it is committed every noun in that statement maps to
something a reader can click.

## What is in this package

| Path | Status | Notes |
|---|---|---|
| `fusion/build_corpus.py` | **Your v5 script, verbatim** | 449 lines, unmodified. Compiles cleanly; scanned for credentials, local usernames and manuscript cross-references — none present. See step 2. |
| `fusion/verify_alignment.py` | **New, written for release** | Checks a rebuilt corpus against the paper's claims. Tested; exits 0 on pass, 1 on failure. |
| `fusion/README.md` | **New** | Documents what `build_corpus.py` actually does, the fused schema, the outputs and the quality gates. |
| `results/results_v8/` | **From the actual run** | The 12 CSVs, 2 JSON files and 18 figures produced by the reported execution. 1.2 MB. |
| `results/README.md` | **New** | Explains every file and the column conventions. |

## Step 1 — Clone and create a branch

```bash
git clone https://github.com/Bature82/cpiot-xad2025-hybrid-cnn-lstm.git
cd cpiot-xad2025-hybrid-cnn-lstm
git checkout -b add-fusion-and-results
```

## Step 2 — The fusion script

`fusion/build_corpus.py` in this package is your v5 script **exactly as run**,
byte for byte. It has deliberately not been tidied: a released script that
differs from the one that produced the corpus is worse than no script at all,
because a rebuild would silently diverge from what was evaluated.

It has been checked for the three things that matter before publication:

1. **No credentials and no local usernames.** The only absolute paths are the
   three Colab Drive constants at the top of the file (`NB_IOT_FOLDER`,
   `SWAT_FOLDER`, `OUT_DIR`). They point at a `MyDrive` folder and expose
   nothing personal. `fusion/README.md` tells a reader to edit those three
   lines; leave them as they are rather than converting them to command-line
   arguments, since changing the script now would break the "exactly as run"
   guarantee for the sake of cosmetics.
2. **No copy of the SWaT data.** The script reads SWaT from a local folder and
   embeds none of it.
3. **No manuscript cross-references.** Nothing in the file mentions reviewers,
   sections, tables, figures or equations.

One thing to reconcile before you commit, because it is visible to any reader
(see the note at the end of this document): the `SWAT_FOLDER` path names the
`Dec 2015` collection while the comment on `DOWNSAMPLE_SECONDS` describes the
Jul-2019 collection. Only one of these can be right for the corpus you built.

## Step 3 — Copy in the rest of this package

```bash
mkdir -p fusion
cp    /path/to/addon/fusion/build_corpus.py     fusion/
cp    /path/to/addon/fusion/verify_alignment.py fusion/
cp    /path/to/addon/fusion/README.md           fusion/
mkdir -p results
cp -r /path/to/addon/results/results_v8         results/
cp    /path/to/addon/results/README.md          results/
```

## Step 4 — Check nothing large or private is being committed

```bash
du -sh results/results_v8          # expect about 1.2 MB
git status --short
git check-ignore -v results/results_v8/*.csv   # should print nothing
```

The trained model weights (`best_*_seed42.keras`, 3.5 MB) are deliberately
excluded — they are reproducible from the code and add bulk without adding
evidence. If you want them, attach them to a GitHub release rather than
committing them.

## Step 5 — Point the top-level README at the new directories

Add to the "Repository layout" section:

```
fusion/     corpus construction and alignment diagnostics
results/    committed output of the reported run
```

And under "Input data", replace the note that the corpus is not redistributed
with:

> The corpus itself is not redistributed here. `fusion/` contains the script
> that builds it from the two public sources, and `fusion/verify_alignment.py`
> checks that a rebuild matches the corpus that was evaluated. SWaT requires a
> request to iTrust.

## Step 6 — Commit and push

```bash
git add fusion results README.md
git commit -m "Add corpus construction, alignment diagnostics and reported results

- fusion/build_corpus.py: builds CPIoT-XAD2025 from N-BaIoT and SWaT
- fusion/verify_alignment.py: checks a rebuilt corpus against the reported structure
- results/results_v8/: complete output of the 45-run protocol
"
git push -u origin add-fusion-and-results
```

Then open a pull request and merge, or push straight to `main` if you would
rather not branch.

## Step 7 — Tag a release and cite the tag

A moving `main` branch is not a citable artefact. Tag the state that
corresponds to the submission:

```bash
git tag -a v1.0-submission -m "State at journal submission"
git push origin v1.0-submission
```

If you want a DOI, connect the repository to Zenodo before tagging; Zenodo
mints a DOI per release, and a DOI in the paper is stronger than a bare
GitHub URL.

## Step 8 — Verify the claim now holds

Run the diagnostics against your own fused CSV and confirm it passes:

```bash
python fusion/verify_alignment.py --data CPIoT-XAD2025_fused_hybrid_v5.csv
```

Then re-read the paper's data-availability statement against the repository
contents, item by item. Every noun in that sentence should map to something a
reader can click.

---

## Two things in the paper the script now contradicts

Releasing the script means a reader can check the paper's description of the
corpus against the code. Two descriptions currently do not survive that check.

**1. Figure 3 claims a normalisation stage that does not exist.** The fusion
script applies no feature scaling of any kind — no min–max, no z-score, no
interpolation of gaps. It exports raw units with NaNs
(`POST_FUSION_MISSING_POLICY = "none"`), precisely so that all scaling can be
fitted train-only inside the pipeline. The figure's normalisation box and its
mention of linear interpolation must therefore be corrected; the replacement
wording is in the accompanying note.

**2. The SWaT collection is ambiguous.** `SWAT_FOLDER` names
`SWaT.A1 & A2_Dec 2015` while the comment beside `DOWNSAMPLE_SECONDS` says the
corpus is "~31k rows (Jul-2019 collection)". The paper should state which
collection was used, and the number should agree with `fusion_report_v5.txt`.

Both are cheap to fix and expensive to leave: the whole point of publishing the
construction code is that a sceptical reader can verify the synthetic-pairing
argument, and the first thing such a reader does is compare the figure to the
script.
