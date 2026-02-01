#!/usr/bin/env python3
"""
Inspect FirstBlood result file formats and units (minimal, no assumptions).

What it does:
- scans results/<model>/arterial/*.txt
- reports column counts distribution (3 vs 5 vs 7 etc.)
- prints sample stats to infer:
  * pressure column (Pa absolute vs gauge)
  * velocity column existence
- checks a few specific vessel IDs if provided
"""

import numpy as np
from pathlib import Path
from collections import Counter

P_ATMO = 1.0e5
PA_TO_MMHG = 133.322

BASE = Path.home() / "first_blood/projects/simple_run/results"

MODELS = ["Abel_ref2", "patient025_CoW_v2"]

# Optional: focus vessels you care about (add/change as needed)
FOCUS_VESSELS = ["A1", "A12", "A16", "A59", "A60", "A61", "A70", "A73"]


def load_txt(fp: Path):
    data = np.loadtxt(fp, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def summarize_file(fp: Path, max_rows=5):
    data = load_txt(fp)
    nrow, ncol = data.shape
    t = data[:, 0]
    dt = t[1] - t[0] if len(t) > 1 else float("nan")

    # column summaries (min/max/mean) for first few columns
    stats = []
    for c in range(min(ncol, 5)):
        col = data[:, c]
        stats.append((c, float(np.min(col)), float(np.mean(col)), float(np.max(col))))

    return {
        "path": fp,
        "nrow": nrow,
        "ncol": ncol,
        "dt": dt,
        "stats": stats,
        "data": data,
    }


def infer_pressure_velocity_columns(data: np.ndarray):
    """
    Heuristic inference:
    - time is column 0
    - pressure is often col 1 (start) and maybe col 2 (end)
    - velocity often follows pressure columns
    We'll just return candidate indices and show ranges.
    """
    ncol = data.shape[1]
    candidates = {}

    # Common patterns:
    # 3 cols: t, p, v
    if ncol == 3:
        candidates["p_cols"] = [1]
        candidates["v_cols"] = [2]

    # 5 cols: t, p_s, p_e, v_s, v_e
    elif ncol == 5:
        candidates["p_cols"] = [1, 2]
        candidates["v_cols"] = [3, 4]

    # 7+ cols: t, p_s, p_e, v_s, v_e, q_s, q_e ...
    elif ncol >= 7:
        candidates["p_cols"] = [1, 2]
        candidates["v_cols"] = [3, 4]
        candidates["q_cols"] = [5, 6]

    else:
        # Unknown, but still guess first numeric columns
        candidates["p_cols"] = [1] if ncol > 1 else []
        candidates["v_cols"] = [2] if ncol > 2 else []

    return candidates


def pressure_unit_guess(p_values: np.ndarray):
    """
    Guess if p is:
    - absolute pressure in Pa (~1e5)
    - gauge pressure in Pa (~1e4)
    - mmHg (~100)
    """
    pmin = float(np.min(p_values))
    pmax = float(np.max(p_values))
    pm = float(np.mean(p_values))

    if 8e4 < pm < 2e5:
        return f"looks like ABSOLUTE Pa (mean ~ {pm:.2e})"
    if 1e3 < pm < 5e4:
        return f"looks like GAUGE Pa (mean ~ {pm:.2e})"
    if 20 < pm < 200:
        return f"looks like mmHg (mean ~ {pm:.1f})"
    return f"unknown units (mean ~ {pm:.2e}, min {pmin:.2e}, max {pmax:.2e})"


def print_focus_summary(model: str, vessel_id: str):
    fp = BASE / model / "arterial" / f"{vessel_id}.txt"
    if not fp.exists():
        print(f"  {model}/{vessel_id}: MISSING")
        return

    info = summarize_file(fp)
    data = info["data"]
    cand = infer_pressure_velocity_columns(data)

    print(f"\n--- {model} / {vessel_id} ---")
    print(f"file: {fp}")
    print(f"shape: rows={info['nrow']}, cols={info['ncol']}, dt={info['dt']:.6f}s")

    # Print first-row sample
    print("first row:", ", ".join([f"{x:.6g}" for x in data[0, :min(data.shape[1], 8)]]))

    # Pressure candidates
    for pc in cand.get("p_cols", []):
        p = data[:, pc]
        print(f"p col {pc}: min={np.min(p):.3e} mean={np.mean(p):.3e} max={np.max(p):.3e} -> {pressure_unit_guess(p)}")

        # Also show gauge mmHg conversion if absolute Pa
        if 8e4 < np.mean(p) < 2e5:
            p_mmhg = (p - P_ATMO) / PA_TO_MMHG
            print(f"      gauge mmHg if subtract 1e5 Pa: min={np.min(p_mmhg):.1f}, max={np.max(p_mmhg):.1f}")

    # Velocity candidates
    for vc in cand.get("v_cols", []):
        v = data[:, vc]
        print(f"v col {vc}: min={np.min(v):.3e} mean={np.mean(v):.3e} max={np.max(v):.3e}")

    # Flow candidates (if present)
    for qc in cand.get("q_cols", []):
        q = data[:, qc]
        print(f"q col {qc}: min={np.min(q):.3e} mean={np.mean(q):.3e} max={np.max(q):.3e}")


def main():
    for model in MODELS:
        arterial_dir = BASE / model / "arterial"
        if not arterial_dir.exists():
            print(f"\nMODEL {model}: arterial dir missing -> {arterial_dir}")
            continue

        files = sorted(arterial_dir.glob("A*.txt"))
        if not files:
            # some builds use p#.txt naming; include those too
            files = sorted(arterial_dir.glob("*.txt"))

        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"arterial dir: {arterial_dir}")
        print(f"num files: {len(files)}")

        # column count distribution
        col_counts = Counter()
        sample_paths = []
        for fp in files[:200]:  # keep it minimal
            try:
                data = load_txt(fp)
                col_counts[data.shape[1]] += 1
                if len(sample_paths) < 5:
                    sample_paths.append(fp)
            except Exception as e:
                print(f"  read error {fp.name}: {e}")

        print("column count distribution:", dict(sorted(col_counts.items())))

        # show a couple random samples
        print("\nSample files:")
        for fp in sample_paths:
            info = summarize_file(fp)
            print(f"  {fp.name:20s} cols={info['ncol']} dt={info['dt']:.6f}s")

        # focus vessels
        print("\nFOCUS VESSEL CHECKS:")
        for vid in FOCUS_VESSELS:
            print_focus_summary(model, vid)

    print(f"\nDone. If pressure looks like ~1e5 Pa, your Figure-10 plot must do (p-1e5)/133.322.")
    print("If some files have 3 columns and others 5/7, your plotting script must auto-detect per file.")


if __name__ == "__main__":
    main()
