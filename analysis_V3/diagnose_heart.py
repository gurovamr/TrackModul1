import numpy as np
from pathlib import Path

print("="*60)
print("HEART MODEL DIAGNOSTICS")
print("="*60)

# Check heart output over time
heart_dir = Path.home() / "first_blood/projects/simple_run/results/Abel_ref2/heart_kim_lit"

if heart_dir.exists():
    print(f"\nHeart output files:")
    for f in sorted(heart_dir.glob("*.txt")):
        print(f"  {f.name}")
    
    # Check aorta pressure/flow
    aorta_file = heart_dir / "aorta.txt"
    if aorta_file.exists():
        data = np.loadtxt(aorta_file, delimiter=',')
        print(f"\nAorta output:")
        print(f"  Columns: {data.shape[1]}")
        print(f"  Time points: {data.shape[0]}")
        print(f"\n  First few rows:")
        print(data[:5, :])
        
        if data.shape[1] >= 3:
            # Column 1: pressure, Column 2: flow
            pressure_cgs = data[:, 1]
            flow_cm3s = data[:, 2] if data.shape[1] > 2 else None
            
            pressure_mmhg = pressure_cgs / 1333.22
            
            print(f"\n  Pressure range: {np.min(pressure_mmhg):.1f} - {np.max(pressure_mmhg):.1f} mmHg")
            if flow_cm3s is not None:
                print(f"  Flow range: {np.min(flow_cm3s):.6e} - {np.max(flow_cm3s):.6e} cm3/s")
                print(f"  Mean flow: {np.mean(flow_cm3s):.6e} cm3/s = {np.mean(flow_cm3s)*60:.3f} mL/min")
    
    # Check ventricular pressure
    p_lv_file = heart_dir / "p_LV1.txt"
    if p_lv_file.exists():
        data = np.loadtxt(p_lv_file, delimiter=',')
        pressure_cgs = data[:, 1]
        pressure_mmhg = pressure_cgs / 1333.22
        print(f"\n  LV Pressure range: {np.min(pressure_mmhg):.1f} - {np.max(pressure_mmhg):.1f} mmHg")

else:
    print("\nNo heart output directory found!")

print("\n" + "="*60)
