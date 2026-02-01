#!/usr/bin/env python3
import numpy as np
from pathlib import Path

BASE = Path.home() / "first_blood/projects/simple_run/results"
M3S_TO_LMIN = 60.0 * 1000.0

INFLOW = {"R-ICA": "A12", "L-ICA": "A16", "Basilar": "A59"}
OUTFLOW = {"R-MCA": "A70", "L-MCA": "A73", "R-ACA": "A76", "L-ACA": "A78", "R-PCA": "A64", "L-PCA": "A65"}

def mean_q_lmin(model, vid, side="start"):
    fp = BASE / model / "arterial" / f"{vid}.txt"
    data = np.loadtxt(fp, delimiter=",")
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] != 13:
        raise ValueError(f"{fp} has {data.shape[1]} cols, expected 13")
    qcol = 5 if side == "start" else 6
    return float(np.mean(data[:, qcol]) * M3S_TO_LMIN)

def run(model):
    print(f"\n{'='*70}\n{model}\n{'='*70}")
    Qin = {k: mean_q_lmin(model, v) for k, v in INFLOW.items()}
    Qout = {k: mean_q_lmin(model, v) for k, v in OUTFLOW.items()}

    print("INFLOW (L/min):")
    for k, q in Qin.items():
        print(f"  {k:8s}: {q:.4f}")
    print(f"  TOTAL  : {sum(Qin.values()):.4f}")

    print("\nOUTFLOW (L/min):")
    for k, q in Qout.items():
        print(f"  {k:8s}: {q:.4f}")
    print(f"  TOTAL  : {sum(Qout.values()):.4f}")

    leak = sum(Qin.values()) - sum(Qout.values())
    rel = leak / max(1e-9, sum(Qin.values())) * 100.0
    print(f"\nBALANCE: Qin - Qout = {leak:.6f} L/min ({rel:.3f} %)")

if __name__ == "__main__":
    run("Abel_ref2")
    run("patient025_CoW_v2")
