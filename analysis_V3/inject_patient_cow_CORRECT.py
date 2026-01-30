#!/usr/bin/env python3
"""
Inject patient025 Circle of Willis geometry into FirstBlood Abel_ref2 model
CORRECT VERSION with proper vessel mapping
"""
import json
import pandas as pd
import shutil
from pathlib import Path

# Paths
patient_data_dir = Path.home() / "Simulation-of-Circle-of-Willis/data_patient025"
abel_ref2_dir = Path.home() / "first_blood/models/Abel_ref2"
patient_model_dir = Path.home() / "first_blood/models/patient025_CoW_v2"

print("="*70)
print("Patient-Specific CoW Geometry Injection - CORRECT MAPPING")
print("="*70)

# Load patient data
print("\n1. Loading patient data...")
with open(patient_data_dir / "feature_mr_025.json") as f:
    features = json.load(f)

# Extract vessels from nested structure
patient_vessels = {}
vessel_counter = {}  # Track R/L sides

for region_id, region_data in features.items():
    for vessel_name, vessel_list in region_data.items():
        if 'bifurcation' in vessel_name.lower():
            continue
            
        if isinstance(vessel_list, list) and len(vessel_list) > 0:
            vessel_info = vessel_list[0]
            
            if 'length' in vessel_info and 'radius' in vessel_info:
                length_mm = vessel_info['length']
                radius_mm = vessel_info['radius']['mean']
                
                # Track which side (R/L) this is
                if vessel_name not in vessel_counter:
                    vessel_counter[vessel_name] = 0
                side = 'R' if vessel_counter[vessel_name] == 0 else 'L'
                vessel_counter[vessel_name] += 1
                
                key = f"{side}_{vessel_name}"
                patient_vessels[key] = {
                    'name': vessel_name,
                    'side': side,
                    'length_mm': length_mm,
                    'diameter_mm': radius_mm * 2.0,
                    'length_m': length_mm / 1000.0,
                    'diameter_m': (radius_mm * 2.0) / 1000.0
                }

print(f"   Extracted {len(patient_vessels)} vessel segments")
for key, data in patient_vessels.items():
    print(f"     {key}: L={data['length_mm']:.1f}mm, D={data['diameter_mm']:.1f}mm")

# Create patient model directory
print("\n2. Creating patient model directory...")
patient_model_dir.mkdir(exist_ok=True)

# Copy Abel_ref2 as baseline
for file in abel_ref2_dir.glob("*.csv"):
    shutil.copy(file, patient_model_dir / file.name)

# Load arterial parameters
arterial_file = patient_model_dir / "arterial.csv"
df_arterial = pd.read_csv(arterial_file)

print(f"\n3. CORRECT mapping patient vessels to FirstBlood CoW...")

# CORRECT MAPPING based on grep output
mapping = {
    'R_BA': 'A59',           # Basilar artery 1
    'L_BA': 'A56',           # Basilar artery 2
    'R_ICA': 'A12',          # Right Internal carotid
    'L_ICA': 'A16',          # Left Internal carotid
    'R_MCA': 'A70',          # Right Middle cerebral M1
    'L_MCA': 'A73',          # Left Middle cerebral M1
    'R_ACA': 'A68',          # Right Ant. cerebral 1
    'L_ACA': 'A69',          # Left Ant. cerebral 1
    'R_A1': 'A68',           # Right ACA A1
    'L_A1': 'A69',           # Left ACA A1
    'R_A2': 'A76',           # Right Ant. cerebral A2
    'L_A2': 'A78',           # Left Ant. cerebral A2
    'R_PCA': 'A60',          # Right Post. cerebral 1
    'L_PCA': 'A61',          # Left Post. cerebral 1
    'R_P1': 'A60',           # Right PCA P1
    'L_P1': 'A61',           # Left PCA P1
    'R_P2': 'A64',           # Right Post. cerebral 2
    'L_P2': 'A65',           # Left Post. cerebral 2
    'R_Pcom': 'A62',         # Right Post. communicating
    'L_Pcom': 'A63',         # Left Post. communicating
    'R_Acom': 'A77',         # Ant. communicating (no side)
    'L_Acom': 'A77',         # Same vessel
}

modifications = []

for patient_key, patient_data in patient_vessels.items():
    if patient_key in mapping:
        fb_id = mapping[patient_key]
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
                'Patient_ID': patient_key,
                'FirstBlood_ID': fb_id,
                'Name': fb_name,
                'Old_length_mm': old_l_mm,
                'New_length_mm': new_l_mm,
                'Old_diameter_mm': old_d_mm,
                'New_diameter_mm': new_d_mm,
                'Length_change_%': ((new_l_mm - old_l_mm) / old_l_mm * 100),
                'Diameter_change_%': ((new_d_mm - old_d_mm) / old_d_mm * 100)
            })
            
            print(f"   ✓ {patient_key:10s} → {fb_id} ({fb_name[:25]:25s}): L {old_l_mm:5.1f}→{new_l_mm:5.1f}mm ({((new_l_mm-old_l_mm)/old_l_mm*100):+6.1f}%), D {old_d_mm:4.1f}→{new_d_mm:4.1f}mm ({((new_d_mm-old_d_mm)/old_d_mm*100):+6.1f}%)")

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
    print(f"\nSummary of changes:")
    print(f"  Average length change: {mod_df['Length_change_%'].mean():+.1f}%")
    print(f"  Average diameter change: {mod_df['Diameter_change_%'].mean():+.1f}%")
    print(f"  Max diameter increase: {mod_df['Diameter_change_%'].max():+.1f}% ({mod_df.loc[mod_df['Diameter_change_%'].idxmax(), 'Patient_ID']})")

print("\nNext step:")
print("  cd ~/first_blood/projects/simple_run")
print("  ./simple_run.out patient025_CoW_v2")

