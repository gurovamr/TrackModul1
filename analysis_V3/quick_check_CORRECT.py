import numpy as np
from pathlib import Path

def check_cardiac_output(model_name='Abel_ref2'):
    """Check cardiac output with CORRECT columns"""
    results_dir = Path.home() / f"first_blood/projects/simple_run/results/{model_name}"
    a1_file = results_dir / "arterial/A1.txt"
    
    data = np.loadtxt(a1_file, delimiter=',')
    
    # CORRECT: Use columns 5-6 (volume_flow_rate in m³/s)
    flow_m3s = (data[:, 5] + data[:, 6]) / 2.0
    cardiac_output = np.mean(flow_m3s) * 1000 * 60  # m³/s -> L/min
    
    print(f"Model: {model_name}")
    print(f"Cardiac Output: {cardiac_output:.3f} L/min")
    
    if 4.0 <= cardiac_output <= 7.0:
        print("✓ Physiological range!")
    else:
        print("⚠ Outside normal range")
    
    return cardiac_output

if __name__ == '__main__':
    check_cardiac_output()
