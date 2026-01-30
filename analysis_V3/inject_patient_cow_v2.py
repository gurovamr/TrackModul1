#!/usr/bin/env python3
"""
Inject patient025 Circle of Willis geometry into FirstBlood Abel_ref2 model
Version 2: Correct data extraction
"""
import json
import pandas as pd
import shutil
from pathlib import Path

# Paths
patient_data_dir = Path.home() / "Simulation-of-Circle-of-Willis/data_patient025"
abel_ref2_dir = Path.home() / "first_blood/models/Abel_ref2"
patient_model_dir = Path.home() / "first_blood/models/patient025_CoW"

print("="*70)
print("Patient-Specific CoW Geometry Injection - Version 2")
print("="*70)

# Load patient data
print("\n1. Loading patient data...")
with open(patient_data_dir / "feature_mr_025.json") as f:
    features = json.load(f)

# Extract vessels from nested structure
patient_vessels = {}
for region_id, region_data in features.items():
    for vessel_name, vessel_list in region_data.items():
        if 'bifurcation' in vessel_name.lower():
            continue
            
        if isinstance(vessel_list, list) and len(vessel_list) > 0:
            vessel_info = vessel_list[0]
            
            if 'length' in vessel_info and 'radius' in vessel_info:
                length_mm = vessel_info['length']
                radius_mm = vessel_info['radius']['mean']
                
                key = f"{region_id}_{vessel_name}"
                patient_vessels[key] = {
                    'name': vessel_name,
                    'length_mm': length_mm,
                    'diameter_mm': radius_mm * 2.0,
                    'length_m': length_mm / 1000.0,
                    'diameter_m': (radius_mm * 2.0) / 1000.0
                }

print(f"   Extracted {len(patient_vessels)} vessel segments")

# Create patient model directory
print("\n2. Creating patient model directory...")
patient_model_dir.mkdir(exist_ok=True)

# Copy Abel_ref2 as baseline
for file in abel_ref2_dir.glob("*.csv"):
    shutil.copy(file, patient_model_dir / file.name)

# Load arterial parameters
arterial_file = patient_model_dir / "arterial.csv"
df_arterial = pd.read_csv(arterial_file)

print(f"\n3. Mapping patient vessels to FirstBlood model...")

# Mapping based on Abel_ref2 inspection
mapping = {
    'BA': ['A31'],
    'ICA': ['A12', 'A16'],
    'MCA': ['A73', 'A76'],
    'ACA': ['A47', 'A48'],
    'A1': ['A47', 'A56'],
    'A2': ['A48', 'A57'],
    'PCA': ['A36', 'A41'],
    'P1': ['A35', 'A40'],
    'P2': ['A36', 'A41'],
    'Pcom': ['A34', 'A39'],
    'Acom': ['A55'],
}

modifications = []

for patient_key, patient_data in patient_vessels.items():
    vessel_name = patient_data['name']
    
    if vessel_name in mapping:
        firstblood_ids = mapping[vessel_name]
        
        for fb_id in firstblood_ids:
            idx = df_arterial[df_arterial['ID'] == fb_id].index
            
            if len(idx) > 0:
                idx = idx[0]
                
                # Convert to float explicitly
                old_length = float(df_arterial.loc[idx, 'length[SI]'])
                old_diameter = float(df_arterial.loc[idx, 'start_diameter[SI]'])
                fb_name = str(df_arterial.loc[idx, 'name'])
                
                # Update geometry
                df_arterial.loc[idx, 'length[SI]'] = patient_data['length_m']
                df_arterial.loc[idx, 'start_diameter[SI]'] = patient_data['diameter_m']
                df_arterial.loc[idx, 'end_diameter[SI]'] = patient_data['diameter_m']
                df_arterial.loc[idx, 'start_thickness[SI]'] = patient_data['diameter_m'] * 0.1
                df_arterial.loc[idx, 'end_thickness[SI]'] = patient_data['diameter_m'] * 0.1
                
                old_l_mm = old_length * 1000
                new_l_mm = patient_data['length_mm']
                old_d_mm = old_diameter * 1000
                new_d_mm = patient_data['diameter_mm']
                
                modifications.append({
                    'ID': fb_id,
                    'Name': fb_name,
                    'Patient_vessel': vessel_name,
                    'Old_length_mm': old_l_mm,
                    'New_length_mm': new_l_mm,
                    'Old_diameter_mm': old_d_mm,
                    'New_diameter_mm': new_d_mm
                })
                
                print(f"   ✓ {fb_id} ({fb_name[:30]}): L {old_l_mm:.1f}→{new_l_mm:.1f}mm, D {old_d_mm:.1f}→{new_d_mm:.1f}mm")
                
                break

# Save modified arterial file
df_arterial.to_csv(arterial_file, index=False)

print(f"\n{'='*70}")
print(f"✓ SUCCESS: Modified {len(modifications)} CoW vessels")
print(f"✓ Patient model saved to: {patient_model_dir}")
print(f"{'='*70}")

# Save modification log
if modifications:
    mod_df = pd.DataFrame(modifications)
    mod_df.to_csv(patient_model_dir / "modifications_log.csv", index=False)
    print(f"\n✓ Modification log saved")
else:
    print("\n⚠ WARNING: No vessels were modified!")

print("\nNext step:")
print("  cd ~/first_blood/projects/simple_run")
print("  ./simple_run.out patient025_CoW")

