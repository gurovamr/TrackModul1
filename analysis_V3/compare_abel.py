#!/usr/bin/env python3
"""
Compare Patient025 CoW v2 waveform shape against Abel_ref2 (the reference model).
If Abel_ref2 also shows the steep-rise / long-decay shape, it's a solver characteristic.
If Abel_ref2 has normal dicrotic morphology, something is wrong with patient025 setup.
"""
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import matplotlib.pyplot as plt

P_ATMO     = 1.0e5
PA_TO_MMHG = 133.322
BASE       = Path.home() / "first_blood/projects/simple_run/results"

def load_and_extract(model, vessel_id, use_end=False):
    """Load arterial file, find peaks, return last cycle."""
    fp = BASE / model / "arterial" / f"{vessel_id}.txt"
    if not fp.exists():
        return None, None, None, None
    data = np.loadtxt(fp, delimiter=',')
    t = data[:, 0]
    dt = t[1] - t[0]
    P_col = 2 if use_end else 1
    V_col = 4 if use_end else 3
    P = (data[:, P_col] - P_ATMO) / PA_TO_MMHG
    V = data[:, V_col]

    min_dist = int(0.4 / dt)
    peaks, _ = find_peaks(P, height=np.max(P)*0.75, distance=min_dist)
    if len(peaks) < 2:
        return t, P, V, None

    i0, i1 = peaks[-2], peaks[-1]
    t_cyc = t[i0:i1+1] - t[i0]
    return t_cyc, P[i0:i1+1], V[i0:i1+1], (t[peaks[-1]] - t[peaks[-2]])

# Vessels to compare (use same IDs; Abel_ref2 has the same arterial tree)
vessels = {
    'A1':  {'use_end': True,  'label': 'Aorta'},
    'A12': {'use_end': False, 'label': 'R. ICA'},
    'A16': {'use_end': False, 'label': 'L. ICA'},
}

models = ['Abel_ref2', 'patient025_CoW_v2']
colors = {'Abel_ref2': 'blue', 'patient025_CoW_v2': 'red'}

fig, axes = plt.subplots(len(vessels), 2, figsize=(13, 3.5*len(vessels)))
fig.suptitle("Abel_ref2 (blue) vs Patient025 CoW v2 (red) — Last Cycle Comparison",
             fontsize=13, fontweight='bold')

for row, (vid, cfg) in enumerate(vessels.items()):
    for model in models:
        t_c, P_c, V_c, dur = load_and_extract(model, vid, use_end=cfg['use_end'])
        if t_c is None:
            print(f"  {model}/{vid}: FILE NOT FOUND")
            continue
        c = colors[model]
        lw = 2.0 if model == 'patient025_CoW_v2' else 1.5
        ls = '-' if model == 'patient025_CoW_v2' else '--'
        label = f"{model} (T={dur:.3f}s)" if dur else model

        axes[row, 0].plot(t_c, P_c, color=c, lw=lw, ls=ls, label=label)
        axes[row, 1].plot(t_c, V_c, color=c, lw=lw, ls=ls, label=label)

        print(f"  {model:>25s} / {vid}: P=[{np.min(P_c):.1f}, {np.max(P_c):.1f}] mmHg, "
              f"V=[{np.min(V_c):.3f}, {np.max(V_c):.3f}] m/s, T={dur:.3f}s")

    axes[row, 0].set_ylabel('Pressure (mmHg)', fontsize=9)
    axes[row, 0].set_title(f"{cfg['label']} ({vid})", fontsize=10, fontweight='bold', loc='left')
    axes[row, 0].legend(fontsize=8)
    axes[row, 0].grid(True, alpha=0.25)

    axes[row, 1].set_ylabel('Velocity (m/s)', fontsize=9)
    axes[row, 1].axhline(0, color='k', lw=0.5, alpha=0.4)
    axes[row, 1].grid(True, alpha=0.25)

axes[-1, 0].set_xlabel('Time (s)', fontsize=10)
axes[-1, 1].set_xlabel('Time (s)', fontsize=10)
plt.tight_layout()

out = Path.home() / "first_blood/analysis_V3/validation_results"
out.mkdir(exist_ok=True)
save = out / "abel_vs_patient025_comparison.png"
fig.savefig(save, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {save}")