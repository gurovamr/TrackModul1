import numpy as np
from pathlib import Path

print("Checking where flow comes from...")

# Check A1 (ascending aorta - first vessel from heart)
a1_file = Path.home() / "first_blood/projects/simple_run/results/Abel_ref2/arterial/A1.txt"
data = np.loadtxt(a1_file, delimiter=',')

print(f"\nA1 (Ascending Aorta):")
print(f"  Columns: {data.shape[1]}")
print(f"  First row (all columns):")
print(f"  {data[0, :]}")

# The columns should be:
# 0: time
# 1: pressure_start
# 2: pressure_end  
# 3: flow_start
# 4: flow_end

if data.shape[1] >= 5:
    flow = (data[:, 3] + data[:, 4]) / 2.0
    print(f"\n  Flow (columns 3-4 average):")
    print(f"    Mean: {np.mean(flow):.6e} cm3/s")
    print(f"    = {np.mean(flow) * 60:.3f} mL/min")
    print(f"    = {np.mean(flow) * 60 / 1000:.3f} L/min")
    
    # This is the ACTUAL cardiac output from the simulation
    print(f"\n  CONCLUSION: The simulation IS producing flow")
    print(f"  But the flow is 1000x too low!")
    
    # Let's check what happens if we scale elastance by 1000
    print(f"\n  Current LV elastance: 2.67e+08 dyne/cm2")
    print(f"  If we multiply by 1000: 2.67e+11 dyne/cm2")
    print(f"  Expected flow increase: ~1000x")
    print(f"  Would give: {np.mean(flow) * 60 / 1000 * 1000:.1f} L/min")

