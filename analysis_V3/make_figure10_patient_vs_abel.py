#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# CONFIG: paths
# ----------------------------
RESULTS_BASE = Path.home() / "first_blood/projects/simple_run/results"
MODEL_PATIENT = "patient025_CoW_v2"
MODEL_REF = "Abel_ref2"

# Choose last-cycle window (seconds) if you can’t detect peaks reliably.
# If peak detection works, we’ll override this automatically.
FALLBACK_LAST_SECONDS = 1.2

# ----------------------------
# Robust loader for arterial output
# ----------------------------
def load_arterial_timeseries(results_dir: Path, vessel_file_stem: str):
    """
    Loads a vessel file from results/<model>/arterial/<stem>.txt

    Handles common formats:
    - 3 cols: t, p, v   (or t,p,0)
    - 5 cols: t, p_start, p_end, v_start, v_end
    - 7 cols: t, p_s, p_e, v_s, v_e, q_s, q_e  (your validator expects this)

    Returns dict with:
      time, p (Pa), v (m/s)
    """
    f = results_dir / "arterial" / f"{vessel_file_stem}.txt"
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

    data = np.loadtxt(f, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncol = data.shape[1]
    t = data[:, 0]

    if ncol == 3:
        p = data[:, 1]
        v = data[:, 2]
    elif ncol == 5:
        # assume: t, p_start, p_end, v_start, v_end
        p = data[:, 1]
        v = data[:, 3]
    elif ncol >= 7:
        # assume: t, p_start, p_end, v_start, v_end, q_start, q_end
        p = data[:, 1]
        v = data[:, 3]
    else:
        raise ValueError(f"Unsupported column count {ncol} in {f}")

    return {"time": t, "p": p, "v": v, "path": f, "ncol": ncol}


def pa_to_mmhg(p_pa):
    return p_pa / 133.322


def extract_last_cycle(time, signal):
    """
    Try to estimate cycle length from peaks in pressure.
    If it fails, return last FALLBACK_LAST_SECONDS.
    """
    # Simple peak detection without scipy
    # (works ok for pressure waveforms if smooth)
    y = signal
    if len(y) < 10:
        return time, signal

    # crude peak indices: local maxima above 80% of max
    thr = 0.8 * np.max(y)
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > thr and y[i] > y[i-1] and y[i] > y[i+1]:
            peaks.append(i)

    # If we have enough peaks, use last two to get cycle duration
    if len(peaks) >= 2:
        T = time[peaks[-1]] - time[peaks[-2]]
        if T > 0.2 and T < 2.0:
            t_end = time[-1]
            t_start = t_end - T
            mask = (time >= t_start) & (time <= t_end)
            return time[mask] - time[mask][0], signal[mask]

    # Fallback: last N seconds
    t_end = time[-1]
    t_start = max(time[0], t_end - FALLBACK_LAST_SECONDS)
    mask = (time >= t_start) & (time <= t_end)
    return time[mask] - time[mask][0], signal[mask]


# ----------------------------
# 1) Find vessel IDs by name (recommended)
# ----------------------------
def find_vessel_id_by_keywords(model_dir: Path, keywords):
    """
    Reads models/<model>/arterial.csv and finds first ID whose 'name'
    contains all keywords (case-insensitive).
    """
    arterial_csv = model_dir / "arterial.csv"
    df = pd.read_csv(arterial_csv)
    name_col = "name"
    if name_col not in df.columns:
        raise ValueError(f"'name' column not found in {arterial_csv}")

    mask = np.ones(len(df), dtype=bool)
    for kw in keywords:
        mask &= df[name_col].astype(str).str.lower().str.contains(kw.lower(), na=False)

    hits = df[mask]
    if hits.empty:
        return None
    return str(hits.iloc[0]["ID"])


def main():
    # Point to model folders (for name lookup)
    model_dir_patient = Path.home() / "first_blood/models" / MODEL_PATIENT
    model_dir_ref = Path.home() / "first_blood/models" / MODEL_REF

    # -----------------------------------------
    # 2) Choose locations: either auto-find or hardcode
    # -----------------------------------------
    # Try to auto-find IDs by name keywords in arterial.csv.
    # If your naming differs, this may return None -> then hardcode IDs.
    locations = [
        ("Carotid", ["carotid"]),                    # might match "carotid"
        ("Aorta", ["aorta"]),                        # might match "ascending aorta"
        ("Radial", ["radial"]),
        ("Femoral", ["femoral"]),
        ("Anterior Tibial", ["anterior", "tibial"]),
    ]

    chosen = []
    for label, kws in locations:
        vid = find_vessel_id_by_keywords(model_dir_ref, kws)
        chosen.append((label, vid))

    print("\nAuto-selected vessel IDs (from Abel_ref2 arterial.csv):")
    for label, vid in chosen:
        print(f"  {label:15s} -> {vid}")

    # If any are None, you must hardcode them:
    # Example:
    # chosen = [
    #   ("Carotid", "A12"),
    #   ("Aorta", "A1"),
    #   ("Radial", "A??"),
    #   ("Femoral", "A??"),
    #   ("Anterior Tibial", "A??"),
    # ]

    if any(v is None for _, v in chosen):
        print("\nERROR: Some IDs could not be auto-detected.")
        print("Open models/Abel_ref2/arterial.csv and search for carotid/radial/femoral/tibial.")
        print("Then hardcode the IDs in this script under the 'chosen' list.")
        return

    # -----------------------------------------
    # 3) Load series for patient and reference
    # -----------------------------------------
    results_patient = RESULTS_BASE / MODEL_PATIENT
    results_ref = RESULTS_BASE / MODEL_REF

    # Build figure: 5 rows x 2 cols
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10), sharex="col")
    fig.suptitle("Figure 10 style: Patient025 (red) vs Abel_ref2 (blue)", fontsize=14, fontweight="bold")

    for i, (label, vessel_id) in enumerate(chosen):
        # Load
        ts_pat = load_arterial_timeseries(results_patient, vessel_id)
        ts_ref = load_arterial_timeseries(results_ref, vessel_id)

        # Extract one cycle (use patient pressure for cycle detection)
        t_pat, p_pat = extract_last_cycle(ts_pat["time"], ts_pat["p"])
        t_ref, p_ref = extract_last_cycle(ts_ref["time"], ts_ref["p"])

        # Velocity
        t_pat_v, v_pat = extract_last_cycle(ts_pat["time"], ts_pat["v"])
        t_ref_v, v_ref = extract_last_cycle(ts_ref["time"], ts_ref["v"])

        # Pressure plot (mmHg)
        axp = axes[i, 0]
        axp.plot(t_pat, pa_to_mmhg(p_pat), color="red", linewidth=2)
        axp.plot(t_ref, pa_to_mmhg(p_ref), color="blue", linewidth=2)
        axp.set_ylabel(label)
        if i == 0:
            axp.set_title("p [mmHg]")
        axp.grid(True, alpha=0.3)

        # Velocity plot
        axv = axes[i, 1]
        axv.plot(t_pat_v, v_pat, color="red", linewidth=2)
        axv.plot(t_ref_v, v_ref, color="blue", linewidth=2)
        if i == 0:
            axv.set_title("v [m/s]")
        axv.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")

    # Add legend-like text on the right
    fig.text(0.92, 0.5, "Patient Simulation (red)\nAbel_ref2 Reference (blue)", rotation=90,
             va="center", ha="center", fontsize=10)

    out = Path.home() / "first_blood/analysis_V3" / "figure10_patient_vs_abel.png"
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out, dpi=200)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
