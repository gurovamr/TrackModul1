#!/usr/bin/env python3
"""
Minimal IO contract for FirstBlood results:
- validates expected column structure
- extracts p, v (start or end)
- converts pressure to gauge mmHg (absolute Pa -> mmHg via subtract 1e5)
"""

import numpy as np
from pathlib import Path

P_ATMO = 1.0e5
PA_TO_MMHG = 133.322

def load_arterial(model: str, vessel_id: str, side: str = "start",
                 base: Path = Path.home() / "first_blood/projects/simple_run/results"):
    """
    Returns (t, p_mmhg, v) from results/<model>/arterial/<vessel_id>.txt

    side: "start" uses cols (1,3), "end" uses cols (2,4)
    """
    fp = base / model / "arterial" / f"{vessel_id}.txt"
    if not fp.exists():
        raise FileNotFoundError(fp)

    data = np.loadtxt(fp, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Validate expected structure
    if data.shape[1] != 13:
        raise ValueError(f"{fp} has {data.shape[1]} cols; expected 13 (your current solver output).")

    t = data[:, 0]

    if side == "start":
        p_pa = data[:, 1]
        v = data[:, 3]
    elif side == "end":
        p_pa = data[:, 2]
        v = data[:, 4]
    else:
        raise ValueError("side must be 'start' or 'end'")

    # Convert ABS Pa -> gauge mmHg (as in the paper plots)
    p_mmhg = (p_pa - P_ATMO) / PA_TO_MMHG
    return t, p_mmhg, v


def quick_check():
    base = Path.home() / "first_blood/projects/simple_run/results"
    for model in ["Abel_ref2", "patient025_CoW_v2"]:
        for vid in ["A1", "A12", "A16"]:
            t, p, v = load_arterial(model, vid, side="start", base=base)
            print(f"{model:18s} {vid:4s}  p=[{p.min():6.1f},{p.max():6.1f}] mmHg  "
                  f"v=[{v.min():7.3f},{v.max():7.3f}] m/s  dt={t[1]-t[0]:.3f}s")

if __name__ == "__main__":
    quick_check()
