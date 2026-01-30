import numpy as np
from pathlib import Path

a1_file = Path.home() / "first_blood/projects/simple_run/results/Abel_ref1/arterial/A1.txt"
data = np.loadtxt(a1_file, delimiter=',')

flow_cm3s = (data[:, 3] + data[:, 4]) / 2.0
cardiac_output = np.mean(flow_cm3s) * 60 / 1000

print(f"Abel_ref1 Cardiac Output: {cardiac_output:.3f} L/min")
