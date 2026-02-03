#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

P_ATMO = 1.0e5
PA_TO_MMHG = 133.322
M3S_TO_LMIN = 60.0 * 1000.0  # m^3/s -> L/min

BASE = Path.home() / "first_blood/projects/simple_run/results"

# Use IDs you have confirmed from arterial.csv
VESSELS = [
    ("Aorta",   "A1",  "end"),    # often better at end for A1
    ("R-ICA",   "A12", "start"),
    ("L-ICA",   "A16", "start"),
    ("Basilar", "A59", "start"),
    ("R-MCA",   "A70", "start"),
    ("L-MCA",   "A73", "start"),
    # If you want PCA/ACA, use the IDs you grepped:
    # ("R-PCA", "A64", "start"),
    # ("L-PCA", "A65", "start"),
    # ("R-ACA", "A76", "start"),
    # ("L-ACA", "A78", "start"),
]

def load_arterial(model: str, vid: str):
    fp = BASE / model / "arterial" / f"{vid}.txt"
    if not fp.exists():
        raise FileNotFoundError(fp)
    data = np.loadtxt(fp, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 13:
        raise ValueError(f"{fp} has {data.shape[1]} cols; expected 13.")
    return data

def extract_last_cycle(t, p_mmhg, min_dist_s=0.4):
    dt = t[1] - t[0]
    min_dist = max(1, int(min_dist_s / dt))
    peaks, _ = find_peaks(p_mmhg, height=np.max(p_mmhg) * 0.75, distance=min_dist)
    if len(peaks) < 2:
        # fallback: last 1 second
        mask = t >= (t[-1] - 1.0)
        return (t[mask] - t[mask][0]), p_mmhg[mask], None

    i0, i1 = peaks[-2], peaks[-1]
    return (t[i0:i1+1] - t[i0]), p_mmhg[i0:i1+1], (t[i1] - t[i0])

def pick_cols(side: str):
    # 0:t, 1:P_start, 2:P_end, 3:V_start, 4:V_end, 5:Q_start, 6:Q_end
    if side == "start":
        return dict(p=1, v=3, q=5)
    elif side == "end":
        return dict(p=2, v=4, q=6)
    else:
        raise ValueError("side must be 'start' or 'end'")

def summarize_model(model: str):
    print(f"\n{'='*80}\nMODEL: {model}\n{'='*80}")

    for name, vid, side in VESSELS:
        data = load_arterial(model, vid)
        t = data[:, 0]
        cols = pick_cols(side)

        p_pa = data[:, cols["p"]]
        v = data[:, cols["v"]]          # m/s
        q = data[:, cols["q"]]          # m^3/s

        p_mmhg = (p_pa - P_ATMO) / PA_TO_MMHG
        q_lmin = q * M3S_TO_LMIN

        tc, pc, T = extract_last_cycle(t, p_mmhg)

        # also crop v and q to same indices by reconstructing mask from tc length:
        # simplest: re-extract indices using last-cycle start/end if T exists
        if T is not None:
            # find cycle indices again (safe, cheap)
            dt = t[1] - t[0]
            min_dist = max(1, int(0.4 / dt))
            peaks, _ = find_peaks(p_mmhg, height=np.max(p_mmhg) * 0.75, distance=min_dist)
            i0, i1 = peaks[-2], peaks[-1]
            vc = v[i0:i1+1]
            qc = q_lmin[i0:i1+1]
        else:
            # fallback: last second
            mask = t >= (t[-1] - 1.0)
            vc = v[mask]
            qc = q_lmin[mask]

        sys = float(np.max(pc))
        dia = float(np.min(pc))
        mean_p = float(np.mean(pc))

        peak_v = float(np.max(vc))
        mean_q = float(np.mean(qc))

        print(f"\n{name:8s} ({vid}, {side})")
        print(f"  dt = {t[1]-t[0]:.6f} s, T_cycle = {T if T else float('nan'):.3f} s")
        print(f"  P_gauge [mmHg]: min={dia:.1f}  mean={mean_p:.1f}  max={sys:.1f}")
        print(f"  V [m/s]:        min={np.min(vc):.3f}  mean={np.mean(vc):.3f}  max={peak_v:.3f}")
        print(f"  Q [L/min]:      min={np.min(qc):.3f}  mean={mean_q:.3f}  max={np.max(qc):.3f}")

def main():
    for model in ["Abel_ref2", "patient025_CoW_v2"]:
        summarize_model(model)

if __name__ == "__main__":
    main()
