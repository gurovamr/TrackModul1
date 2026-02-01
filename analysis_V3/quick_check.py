import numpy as np
from pathlib import Path

results_dir = Path.home() / "first_blood/projects/simple_run/results/Abel_ref2"
a1_file = results_dir / "arterial/A1.txt"

data = np.loadtxt(a1_file, delimiter=',')

# Flow in columns 3 and 4 (cm3/s)
flow_cm3s = (data[:, 3] + data[:, 4]) / 2.0
flow_mlmin = flow_cm3s * 60
cardiac_output = np.mean(flow_mlmin) / 1000

print(f"Cardiac Output: {cardiac_output:.3f} L/min")
print(f"Mean flow: {np.mean(flow_mlmin):.1f} mL/min")

if 4.0 <= cardiac_output <= 7.0:
    print("SUCCESS: Baseline Abel_ref2 works correctly!")
else:
    print("PROBLEM: Cardiac output is wrong")
