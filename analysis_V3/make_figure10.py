#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")  # headless backend: avoids Qt/Wayland issues

import matplotlib.pyplot as plt
from pathlib import Path
from io_contract import load_arterial

BASE = Path.home() / "first_blood/projects/simple_run/results"
MODEL_PAT = "patient025_CoW_v2"
MODEL_REF = "Abel_ref2"

# Replace the ??? IDs with the correct ones from arterial.csv
LOCATIONS = [
    ("Carotid", "A5"),           # Common carotid
    ("Aorta", "A1"),
    ("Radial", "A8"),
    ("Femoral", "A46"),
    ("Anterior Tibial", "A49"),
]

SIDE = "start"   # "start" or "end"
WINDOW = 0.8     # seconds to show (roughly one cycle in your sim)

def crop_last_window(t, *ys, window=0.8):
    """
    Crop arrays to the last <window> seconds using a single mask.
    Returns (t0, y0, y1, ...) where t0 is shifted to start at 0.
    """
    t_end = t[-1]
    t0 = max(t[0], t_end - window)
    mask = (t >= t0) & (t <= t_end)
    tt = t[mask] - t[mask][0]
    out = [tt]
    for y in ys:
        out.append(y[mask])
    return out

def main():
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10), sharex="col")
    fig.suptitle("Figure 10 style: Patient025 (red) vs Abel_ref2 (blue)", fontsize=14, fontweight="bold")

    for i, (label, vid) in enumerate(LOCATIONS):
        # Load full time series
        tP, pP, vP = load_arterial(MODEL_PAT, vid, side=SIDE, base=BASE)
        tR, pR, vR = load_arterial(MODEL_REF, vid, side=SIDE, base=BASE)

        # Crop consistently
        tPc, pPc, vPc = crop_last_window(tP, pP, vP, window=WINDOW)
        tRc, pRc, vRc = crop_last_window(tR, pR, vR, window=WINDOW)

        # Pressure
        axp = axes[i, 0]
        axp.plot(tPc, pPc, color="red", lw=2)
        axp.plot(tRc, pRc, color="blue", lw=2)
        axp.set_ylabel(label)
        axp.grid(True, alpha=0.3)
        if i == 0:
            axp.set_title("p [mmHg]")

        # Velocity
        axv = axes[i, 1]
        axv.plot(tPc, vPc, color="red", lw=2)
        axv.plot(tRc, vRc, color="blue", lw=2)
        axv.grid(True, alpha=0.3)
        if i == 0:
            axv.set_title("v [m/s]")

    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")

    fig.text(0.92, 0.5, "Patient Simulation (red)\nAbel_ref2 Reference (blue)",
             rotation=90, va="center", ha="center", fontsize=10)

    out = Path.home() / "first_blood/analysis_V3/validation_results/figure10_patient_vs_abel.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
