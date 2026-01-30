#!/usr/bin/env python3
"""
Compare Circle of Willis flow distribution between patient and reference
"""
import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path.home() / "first_blood/projects/simple_run/results"

# CoW vessel IDs in FirstBlood (from arterial.csv inspection)
cow_vessels = {
    'R-ICA': 'A12',  # Right Internal Carotid
    'L-ICA': 'A16',  # Left Internal Carotid
    'R-VA': 'A6',    # Right Vertebral
    'L-VA': 'A20',   # Left Vertebral
    # Add more as identified in injection script output
}

print("="*70)
print("Circle of Willis Flow Analysis: Patient vs Reference")
print("="*70)

results = []

for vessel_name, vessel_id in cow_vessels.items():
    # Load patient data
    try:
        patient_file = results_dir / f"patient025_CoW/arterial/{vessel_id}.txt"
        data_p = np.loadtxt(patient_file, delimiter=',')
        flow_p = np.mean((data_p[:, 5] + data_p[:, 6]) / 2.0) * 1000 * 60  # L/min
        
        # Load reference data
        ref_file = results_dir / f"Abel_ref2/arterial/{vessel_id}.txt"
        data_r = np.loadtxt(ref_file, delimiter=',')
        flow_r = np.mean((data_r[:, 5] + data_r[:, 6]) / 2.0) * 1000 * 60  # L/min
        
        results.append({
            'Vessel': vessel_name,
            'ID': vessel_id,
            'Reference (L/min)': flow_r,
            'Patient (L/min)': flow_p,
            'Ratio': flow_p / flow_r if flow_r > 0 else 0,
            'Difference (%)': ((flow_p - flow_r) / flow_r * 100) if flow_r > 0 else 0
        })
        
    except FileNotFoundError:
        print(f"Warning: Could not find data for {vessel_name} ({vessel_id})")

# Create DataFrame
df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

print("\n" + "="*70)
print("Key Findings:")
print(f"  - Vessels analyzed: {len(results)}")
if len(results) > 0:
    print(f"  - Average flow change: {df['Difference (%)'].mean():.1f}%")
    print(f"  - Max increase: {df['Difference (%)'].max():.1f}% in {df.loc[df['Difference (%)'].idxmax(), 'Vessel']}")
    print(f"  - Max decrease: {df['Difference (%)'].min():.1f}% in {df.loc[df['Difference (%)'].idxmin(), 'Vessel']}")

