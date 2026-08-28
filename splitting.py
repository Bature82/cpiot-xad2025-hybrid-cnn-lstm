"""Leakage-safe, anomaly-mass-stratified splitting of the row timeline.

Anomaly episodes closer than ``MIN_EPISODE_GAP`` rows are merged into one
physical attack burst; each burst plus ``EPISODE_MARGIN`` rows of surrounding
context is an ATOMIC segment that goes to exactly one split. Windows are built
INSIDE each split afterwards, so no window ever spans a split boundary and no
attack is ever partly in train and partly in test.
"""

import numpy as np

from .config import CFG


def find_episodes(y):
    """Contiguous runs of label 1 -> list of (start, end) half-open intervals."""
    y = np.asarray(y).astype(int)
    d = np.flatnonzero(np.diff(np.r_[0, y, 0]))
    return list(zip(d[0::2], d[1::2]))


def build_atomic_segments(y):
    """Merge nearby anomaly episodes into physical bursts and wrap each burst
    in a context margin, clipped at the midpoint toward the neighbouring
    burst so segments never overlap. Each returned segment is indivisible:
    it is assigned to exactly one split."""
    N = len(y)
    margin = int(CFG["EPISODE_MARGIN"])
    min_gap = int(CFG["MIN_EPISODE_GAP"])
    eps = find_episodes(y)
    if not eps:
        raise SystemExit("[ERROR] Dataset contains no anomalous rows.")

    clusters = [list(eps[0])]
    for s, e in eps[1:]:
        if s - clusters[-1][1] < min_gap:
            clusters[-1][1] = e
        else:
            clusters.append([s, e])

    segs = []
    for k, (s, e) in enumerate(clusters):
        lo, hi = max(0, s - margin), min(N, e + margin)
        if k > 0:
            lo = max(lo, (clusters[k - 1][1] + s) // 2)
        if k < len(clusters) - 1:
            hi = min(hi, (e + clusters[k + 1][0]) // 2)
        segs.append((lo, hi))
    return segs, len(eps)


def assign_blocks(y, n_blocks, ratios, seed):
    """Allocate atomic anomaly segments to splits so that each split receives
    close to its target share of the total anomalous ROW MASS, then fill the
    remaining normal-only timeline by row-count deficit.

    Allocation is greedy on mass: segments are visited largest-first and each
    goes to whichever split is furthest below its anomaly-mass target. This
    removes the dominant source of seed-to-seed variance in a round-robin
    scheme, where a single large burst could hand one split most of the
    anomalies and leave another with almost none.

    With fewer than three separated bursts a balanced allocation is
    impossible; the pipeline then falls back to the classical
    train-on-normal / test-on-attack protocol, which is still leakage-free
    but statistically weak, and says so loudly."""
    N = len(y)
    y = np.asarray(y).astype(int)
    segs, n_eps = build_atomic_segments(y)
    rng = np.random.default_rng(seed)

    seg_mass = np.array([int(y[lo:hi].sum()) for lo, hi in segs], dtype=np.int64)
    total_mass = int(seg_mass.sum())
    anom_ratios = CFG.get("ANOM_RATIOS", ratios)

    fallback = len(segs) < 3
    if fallback:
        print(f"[WARN] {n_eps} anomaly episodes form only {len(segs)} separated "
              f"burst(s) -> FALLBACK: normal-train / anomaly-test protocol. "
              f"All anomaly segments go to TEST; train and validation are "
              f"normal-only. Leakage-free but statistically weak: regenerate "
              f"the fusion export with more attack episodes before drawing "
              f"quantitative conclusions.")
        seg_role = {si: "test" for si in range(len(segs))}
    else:
        # Largest-first visit order, perturbed by a small multiplicative
        # jitter so that segments of comparable size can swap places between
        # seeds. The greedy deficit rule below still keeps the resulting mass
        # shares close to target, so the partition varies without any split
        # ever being starved of anomalies.
        jit = np.exp(rng.normal(0.0, float(CFG.get("SPLIT_JITTER", 0.0)), len(segs)))
        order = sorted(range(len(segs)), key=lambda i: -seg_mass[i] * jit[i])
        mass_target = {k: anom_ratios[k] * total_mass for k in ratios}
        mass_now = {k: 0.0 for k in ratios}
        seg_role = {}
        for si in order:
            sp = max(mass_target, key=lambda k: mass_target[k] - mass_now[k])
            seg_role[int(si)] = sp
            mass_now[sp] += float(seg_mass[si])
        # guarantee that val and test are never left without anomalies
        for need in ("test", "val"):
            if mass_now[need] <= 0 and total_mass > 0:
                donor = max(mass_now, key=mass_now.get)
                cand = [si for si, r in seg_role.items()
                        if r == donor and seg_mass[si] > 0]
                if cand:
                    move = min(cand, key=lambda si: seg_mass[si])
                    seg_role[move] = need
                    mass_now[donor] -= float(seg_mass[move])
                    mass_now[need] += float(seg_mass[move])

    split_per_row = np.empty(N, dtype=object)
    covered = np.zeros(N, dtype=bool)
    for si, (lo, hi) in enumerate(segs):
        split_per_row[lo:hi] = seg_role[si]
        covered[lo:hi] = True

    # chop the normal-only remainder into blocks; allocate by row-count deficit
    target = {k: ratios[k] * N for k in ratios}
    counts = {k: 0 for k in ratios}
    for si, (lo, hi) in enumerate(segs):
        counts[seg_role[si]] += hi - lo

    blk = max(50, N // max(n_blocks, 1))
    i = 0
    while i < N:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < N and not covered[j]:
            j += 1
        for a in range(i, j, blk):
            b = min(a + blk, j)
            deficit = {k: target[k] - counts[k] for k in ratios}
            sp = max(deficit, key=deficit.get)
            split_per_row[a:b] = sp
            counts[sp] += b - a
        i = j

    mass_per_split = {k: int(y[split_per_row == k].sum()) for k in ratios}
    seg_per_split = {k: sum(1 for v in seg_role.values() if v == k) for k in ratios}
    print(f"[SPLIT] anomaly bursts per split: {seg_per_split} | "
          f"anomaly rows per split: {mass_per_split} | "
          f"rows: { {k: int(v) for k, v in counts.items()} }")
    return split_per_row, segs, seg_role
