#!/usr/bin/env python3
"""
Inject patient025 Circle of Willis geometry into FirstBlood Abel_ref2 model
"""
import json
import pandas as pd
import shutil
from pathlib import Path

# Paths
patient_data_dir = Path.home() / "Simulation-of-Circle-of-Willis/data_patient025"
abel_ref2_dir = Path.home() / "first_blood/models/Abel_ref2"
patient_model_dir = Path.home() / "first_blood/models/patient025_CoW"

print("="*60)
print("Patient-Specific CoW Geometry Injection for FirstBlood")
print("="*60)

# Load patient data
print("\n1. Loading patient data...")
with open(patient_data_dir / "feature_mr_025.json") as f:
    features = json.load(f)
    
with open(patient_data_dir / "nodes_mr_025.json") as f:
    nodes = json.load(f)
    
with open(patient_data_dir / "variant_mr_025.json") as f:
    variant = json.load(f)

print(f"   - Loaded {len(features)} vessel features")
print(f"   - Loaded {len(nodes)} nodes")
print(f"   - Variant type: {variant}")

# Create patient model directory
print("\n2. Creating patient model directory...")
patient_model_dir.mkdir(exist_ok=True)

# Copy all Abel_ref2 files as baseline
print("   - Copying Abel_ref2 as baseline...")
for file in abel_ref2_dir.glob("*.csv"):
    shutil.copy(file, patient_model_dir / file.name)

# Load arterial parameters
arterial_file = patient_model_dir / "arterial.csv"
df_arterial = pd.read_csv(arterial_file)

print(f"\n3. Abel_ref2 has {len(df_arterial)} vessels")
print("   Circle of Willis vessels in Abel_ref2:")

# Map patient vessel names to FirstBlood names
cow_mapping = {
    # Internal Carotid Arteries
    'R-ICA': ['Internal carotid'],  # Right ICA - will match A12
    'L-ICA': ['Internal carotid'],  # Left ICA - will match A16
    
    # Middle Cerebral Arteries  
    'R-MCA-M1': ['Middle cerebral'],
    'L-MCA-M1': ['Middle cerebral'],
    
    # Anterior Cerebral Arteries
    'R-ACA-A1': ['Anterior cerebral A'],
    'L-ACA-A1': ['Anterior cerebral A'],
    'R-ACA-A2': ['Anterior cerebral B'],
    'L-ACA-A2': ['Anterior cerebral B'],
    
    # Posterior Circulation
    'R-PCA-P1': ['Posterior cerebral A'],
    'L-PCA-P1': ['Posterior cerebral A'],
    'R-PCA-P2': ['Posterior cerebral B'],
    'L-PCA-P2': ['Posterior cerebral B'],
    
    # Basilar and Vertebrals
    'Basilar': ['Basilar'],
    'R-VA': ['Vertebral'],
    'L-VA': ['Vertebral'],
    
    # Communicating arteries
    'ACoA': ['Anterior communicating'],
    'R-PCoA': ['Posterior communicating'],
    'L-PCoA': ['Posterior communicating'],
}

# Find CoW vessels in Abel_ref2
cow_vessels = []
for idx, row in df_arterial.iterrows():
    vessel_name = row['name']
    for cow_key, patterns in cow_mapping.items():
        for pattern in patterns:
            if pattern.lower() in vessel_name.lower():
                cow_vessels.append({
                    'index': idx,
                    'firstblood_id': row['ID'],
                    'firstblood_name': vessel_name,
                    'patient_name': cow_key
                })
                print(f"   - Found: {row['ID']} ({vessel_name})")
                break

print(f"\n4. Found {len(cow_vessels)} CoW vessels to modify")

# Now extract patient geometry and inject
print("\n5. Extracting patient vessel geometry...")
patient_vessels = {}

for vessel_id, vessel_data in features.items():
    # Extract vessel name from ID (e.g., "R-ICA" from ID)
    vessel_name = vessel_id.split('_')[0] if '_' in vessel_id else vessel_id
    
    if 'length' in vessel_data and 'diameter' in vessel_data:
        # Convert from mm to m (FirstBlood uses SI units)
        length_m = vessel_data['length'] / 1000.0
        diameter_m = vessel_data['diameter'] / 1000.0
        
        patient_vessels[vessel_name] = {
            'length': length_m,
            'diameter': diameter_m
        }
        print(f"   - {vessel_name}: L={vessel_data['length']:.1f}mm, D={vessel_data['diameter']:.1f}mm")

print(f"\n6. Injecting patient geometry into FirstBlood model...")
modifications = 0

for cow_vessel in cow_vessels:
    patient_name = cow_vessel['patient_name']
    
    if patient_name in patient_vessels:
        idx = cow_vessel['index']
        patient_data = patient_vessels[patient_name]
        
        # Update geometry
        old_length = df_arterial.loc[idx, 'length[SI]']
        old_diam = df_arterial.loc[idx, 'start_diameter[SI]']
        
        df_arterial.loc[idx, 'length[SI]'] = patient_data['length']
        df_arterial.loc[idx, 'start_diameter[SI]'] = patient_data['diameter']
        df_arterial.loc[idx, 'end_diameter[SI]'] = patient_data['diameter']
        df_arterial.loc[idx, 'start_thickness[SI]'] = patient_data['diameter'] * 0.1
        df_arterial.loc[idx, 'end_thickness[SI]'] = patient_data['diameter'] * 0.1
        
        print(f"   ✓ {cow_vessel['firstblood_id']}: L {old_length*1000:.1f}→{patient_data['length']*1000:.1f}mm, D {old_diam*1000:.1f}→{patient_data['diameter']*1000:.1f}mm")
        modifications += 1

# Save modified arterial file
df_arterial.to_csv(arterial_file, index=False)

print(f"\n{'='*60}")
print(f"✓ SUCCESS: Modified {modifications} CoW vessels")
print(f"✓ Patient model saved to: {patient_model_dir}")
print(f"{'='*60}")

print("\nNext steps:")
print("  cd ~/first_blood/projects/simple_run")
print("  ./simple_run.out patient025_CoW")

