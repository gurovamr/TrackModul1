#!/usr/bin/env python3
"""Quick diagnostic: show what the peak detector is actually finding."""
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

RESULTS_DIR = Path.home() / "first_blood/projects/simple_run/results/patient025_CoW_v2/arterial"
P_ATMO = 1.0e5
PA_TO_MMHG = 133.322

for vid in ['A1', 'A12', 'A16', 'A70', 'A59']:
    data = np.loadtxt(RESULTS_DIR / f"{vid}.txt", delimiter=',')
    t = data[:, 0]
    # A1: use end (col2), others: use start (col1)
    P_col = 2 if vid == 'A1' else 1
    P = (data[:, P_col] - P_ATMO) / PA_TO_MMHG
    V_col = 4 if vid == 'A1' else 3
    V = data[:, V_col]

    dt = t[1] - t[0]
    min_dist = int(0.4 / dt)
    peaks, _ = find_peaks(P, height=np.max(P)*0.75, distance=min_dist)

    print(f"\n=== {vid} ===")
    print(f"  Time range: {t[0]:.3f} – {t[-1]:.3f} s  ({len(t)} points, dt={dt:.4f})")
    print(f"  P range: {np.min(P):.1f} – {np.max(P):.1f} mmHg")
    print(f"  V range: {np.min(V):.4f} – {np.max(V):.4f} m/s")
    print(f"  Peaks found: {len(peaks)}")
    if len(peaks) >= 2:
        print(f"  Peak times: {t[peaks]}")
        cycle_durs = np.diff(t[peaks])
        print(f"  Cycle durations: {cycle_durs}")
        print(f"  Last cycle: t={t[peaks[-2]]:.3f} – {t[peaks[-1]]:.3f} s  "
              f"(duration={t[peaks[-1]]-t[peaks[-2]]:.3f} s)")
        # Show the actual last-cycle waveform stats
        i0, i1 = peaks[-2], peaks[-1]
        print(f"  Last cycle P: {np.min(P[i0:i1+1]):.1f} – {np.max(P[i0:i1+1]):.1f} mmHg")
        print(f"  Last cycle V: {np.min(V[i0:i1+1]):.4f} – {np.max(V[i0:i1+1]):.4f} m/s")
        # Also check second-to-last cycle for comparison (convergence check)
        if len(peaks) >= 3:
            i0b, i1b = peaks[-3], peaks[-2]
            print(f"  Prev cycle P: {np.min(P[i0b:i1b+1]):.1f} – {np.max(P[i0b:i1b+1]):.1f} mmHg  "
                  f"(cycle dur={t[i1b]-t[i0b]:.3f} s)")
    else:
        print(f"  WARNING: fewer than 2 peaks detected!")
        print(f"  Threshold used: {np.max(P)*0.75:.1f} mmHg")