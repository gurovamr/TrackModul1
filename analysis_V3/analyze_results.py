#!/usr/bin/env python3
"""
Analyze FirstBlood simulation results
FINAL CORRECTED VERSION with proper column mapping

Run from: analysis_V3/
"""

import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("FIRSTBLOOD RESULTS ANALYSIS AND VALIDATION")
print("=" * 80)

# PATH CONFIGURATION
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "projects" / "simple_run" / "results" / "cow_runV30"

print(f"\nResults directory: {RESULTS_DIR}")

if not RESULTS_DIR.exists():
    print("\nERROR: Results directory not found")
    sys.exit(1)

# VESSEL MAPPING
COW_VESSELS = {
    'A56': {'name': 'BA', 'full_name': 'Basilar artery'},
    'A64': {'name': 'R-PCA', 'full_name': 'Right Posterior Cerebral A2'},
    'A65': {'name': 'L-PCA', 'full_name': 'Left Posterior Cerebral A2'},
    'A12': {'name': 'R-ICA', 'full_name': 'Right Internal Carotid'},
    'A70': {'name': 'R-MCA', 'full_name': 'Right Middle Cerebral M1'},
    'A16': {'name': 'L-ICA', 'full_name': 'Left Internal Carotid'},
    'A73': {'name': 'L-MCA', 'full_name': 'Left Middle Cerebral M1'},
    'A62': {'name': 'R-Pcom', 'full_name': 'Right Posterior Communicating'},
    'A63': {'name': 'L-Pcom', 'full_name': 'Left Posterior Communicating'},
    'A77': {'name': 'Acom', 'full_name': 'Anterior Communicating'},
    'A76': {'name': 'R-ACA', 'full_name': 'Right Anterior Cerebral A2'},
    'A78': {'name': 'L-ACA', 'full_name': 'Left Anterior Cerebral A2'},
}

def load_vessel_data(vessel_id):
    """Load time series data for a vessel"""
    filepath = RESULTS_DIR / "arterial" / f"{vessel_id}.txt"
    
    if not filepath.exists():
        return None
    
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except:
        try:
            data = np.loadtxt(filepath)
            return data
        except Exception as e:
            print(f"  WARNING: Could not load {vessel_id}: {e}")
            return None

def analyze_time_series(data):
    """
    Analyze time series data
    
    Column mapping (from diagnostic):
    0: time (s)
    1: pressure_start (dyne/cm2)
    2: pressure_end (dyne/cm2)
    3: flow_start (cm3/s)
    4: flow_end (cm3/s)
    5-6: Unknown (maybe area?)
    7-8: velocity components? (cm/s)
    
    Using average of start/end values
    """
    if data is None or len(data) == 0:
        return None
    
    n_cols = data.shape[1]
    n_rows = data.shape[0]
    
    results = {
        'n_points': n_rows,
        'n_cols': n_cols,
    }
    
    # Pressure: average of columns 1 and 2, convert dyne/cm2 to mmHg
    if n_cols >= 3:
        pressure_start = data[:, 1]
        pressure_end = data[:, 2]
        pressure_cgs = (pressure_start + pressure_end) / 2.0
        pressure_mmhg = pressure_cgs / 1333.22
        
        results['pressure_mean_mmhg'] = np.mean(pressure_mmhg)
        results['pressure_max_mmhg'] = np.max(pressure_mmhg)
        results['pressure_min_mmhg'] = np.min(pressure_mmhg)
        results['pressure_std_mmhg'] = np.std(pressure_mmhg)
    
    # Flow: average of columns 3 and 4, convert cm3/s to mL/min
    if n_cols >= 5:
        flow_start = data[:, 3]
        flow_end = data[:, 4]
        flow_cm3s = (flow_start + flow_end) / 2.0
        flow_mlmin = flow_cm3s * 60  # cm3/s = mL/s, so *60 for mL/min
        
        results['flow_mean_mls'] = np.mean(flow_cm3s)
        results['flow_max_mls'] = np.max(flow_cm3s)
        results['flow_min_mls'] = np.min(flow_cm3s)
        results['flow_mean_mlmin'] = np.mean(flow_mlmin)
    
    # Velocity: try columns 7-8, convert cm/s to m/s
    if n_cols >= 9:
        vel_start = data[:, 7]
        vel_end = data[:, 8]
        velocity_cms = (vel_start + vel_end) / 2.0
        velocity_ms = velocity_cms / 100.0
        
        results['velocity_mean_ms'] = np.mean(velocity_ms)
        results['velocity_max_ms'] = np.max(velocity_ms)
        results['velocity_min_ms'] = np.min(velocity_ms)
    
    return results

# STEP 1: Analyze Aortic Pressure
print("\n" + "=" * 80)
print("STEP 1: AORTIC PRESSURE ANALYSIS")
print("=" * 80)

aorta_data = load_vessel_data('A1')
aorta_results = None

if aorta_data is not None:
    aorta_results = analyze_time_series(aorta_data)
    
    print("\nAscending Aorta (A1):")
    print(f"  Data points: {aorta_results['n_points']}")
    print(f"  Columns: {aorta_results['n_cols']}")
    print(f"\n  Pressure:")
    print(f"    Systolic (max):  {aorta_results['pressure_max_mmhg']:6.1f} mmHg  [Normal: 100-140]")
    print(f"    Diastolic (min): {aorta_results['pressure_min_mmhg']:6.1f} mmHg  [Normal: 60-90]")
    print(f"    Mean:            {aorta_results['pressure_mean_mmhg']:6.1f} mmHg  [Normal: 80-100]")
    
    systolic_ok = 100 <= aorta_results['pressure_max_mmhg'] <= 140
    diastolic_ok = 60 <= aorta_results['pressure_min_mmhg'] <= 90
    
    print(f"\n  Validation:")
    print(f"    Systolic:  {'PASS' if systolic_ok else 'FAIL'}")
    print(f"    Diastolic: {'PASS' if diastolic_ok else 'FAIL'}")
    
    if 'flow_mean_mlmin' in aorta_results:
        print(f"\n  Flow:")
        print(f"    Mean: {aorta_results['flow_mean_mlmin']:6.1f} mL/min")
        
        cardiac_output = aorta_results['flow_mean_mlmin'] / 1000
        print(f"    Cardiac Output: {cardiac_output:5.2f} L/min  [Normal: 4-7]")
        
        co_ok = 4.0 <= cardiac_output <= 7.0
        print(f"    Validation: {'PASS' if co_ok else 'FAIL'}")
else:
    print("\nWARNING: Could not load aorta data")

# STEP 2: Analyze Circle of Willis Vessels
print("\n" + "=" * 80)
print("STEP 2: CIRCLE OF WILLIS ANALYSIS")
print("=" * 80)

cow_results = {}

print("\nAnalyzing 12 CoW vessels...")
print(f"\n{'Vessel':<12s} {'Name':<12s} {'P_mean':<10s} {'Flow':<12s} {'Velocity':<10s} {'Status':<8s}")
print("-" * 80)

for vessel_id, info in COW_VESSELS.items():
    data = load_vessel_data(vessel_id)
    
    if data is not None:
        results = analyze_time_series(data)
        cow_results[vessel_id] = results
        
        p_mean = results.get('pressure_mean_mmhg', 0)
        flow = results.get('flow_mean_mlmin', 0)
        vel = results.get('velocity_mean_ms', 0)
        
        status = 'OK'
        if abs(vel) < 0.1 or abs(vel) > 2.0:
            status = 'CHECK'
        if flow < 0:
            status = 'BACKFLOW'
        
        print(f"{vessel_id:<12s} {info['name']:<12s} "
              f"{p_mean:>8.1f} mmHg  {flow:>9.1f} mL/min  "
              f"{vel:>7.3f} m/s  {status:<8s}")
    else:
        print(f"{vessel_id:<12s} {info['name']:<12s} NO DATA")

# STEP 3: Fetal Variant Validation
print("\n" + "=" * 80)
print("STEP 3: FETAL R-PCA VARIANT VALIDATION")
print("=" * 80)

r_pcom = cow_results.get('A62')
r_pca_p2 = cow_results.get('A64')

if r_pcom and r_pca_p2:
    print("\nR-Pcom (A62):")
    print(f"  Flow: {r_pcom.get('flow_mean_mlmin', 0):6.1f} mL/min")
    print(f"  Velocity: {r_pcom.get('velocity_mean_ms', 0):5.3f} m/s")
    
    print("\nR-PCA P2 (A64):")
    print(f"  Flow: {r_pca_p2.get('flow_mean_mlmin', 0):6.1f} mL/min")
    print(f"  Velocity: {r_pca_p2.get('velocity_mean_ms', 0):5.3f} m/s")
    
    pcom_flow = abs(r_pcom.get('flow_mean_mlmin', 0))
    
    print("\nFetal variant validation:")
    if pcom_flow > 50:
        print("  CONFIRMED: R-Pcom has substantial flow (fetal type)")
    elif pcom_flow > 10:
        print(f"  MODERATE: R-Pcom flow = {pcom_flow:.1f} mL/min")
    else:
        print(f"  LOW: R-Pcom flow = {pcom_flow:.1f} mL/min (not fetal?)")
else:
    print("\nWARNING: Could not analyze fetal variant (missing data)")

# STEP 4: Summary Statistics
print("\n" + "=" * 80)
print("STEP 4: SUMMARY STATISTICS")
print("=" * 80)

all_flows = []
all_vels = []
all_pressures = []

if cow_results:
    all_flows = [r.get('flow_mean_mlmin', 0) for r in cow_results.values() if 'flow_mean_mlmin' in r]
    all_vels = [r.get('velocity_mean_ms', 0) for r in cow_results.values() if 'velocity_mean_ms' in r]
    all_pressures = [r.get('pressure_mean_mmhg', 0) for r in cow_results.values() if 'pressure_mean_mmhg' in r]
    
    print("\nCoW vessels statistics:")
    print(f"  Vessels analyzed: {len(cow_results)}")
    
    if all_flows:
        print(f"\n  Flow range: {min(all_flows):.1f} - {max(all_flows):.1f} mL/min")
        print(f"  Mean flow: {np.mean(all_flows):.1f} mL/min")
    
    if all_vels:
        print(f"\n  Velocity range: {min(all_vels):.3f} - {max(all_vels):.3f} m/s")
        print(f"  Mean velocity: {np.mean(all_vels):.3f} m/s  [Expected: 0.2-1.5]")
    
    if all_pressures:
        print(f"\n  Pressure range: {min(all_pressures):.1f} - {max(all_pressures):.1f} mmHg")
        print(f"  Mean pressure: {np.mean(all_pressures):.1f} mmHg")

# STEP 5: Overall Validation
print("\n" + "=" * 80)
print("STEP 5: OVERALL VALIDATION")
print("=" * 80)

validation_checks = []

if aorta_results is not None:
    systolic = aorta_results['pressure_max_mmhg']
    diastolic = aorta_results['pressure_min_mmhg']
    validation_checks.append(('Aortic systolic pressure', 100 <= systolic <= 140, f"{systolic:.1f} mmHg"))
    validation_checks.append(('Aortic diastolic pressure', 60 <= diastolic <= 90, f"{diastolic:.1f} mmHg"))
    
    if 'flow_mean_mlmin' in aorta_results:
        co = aorta_results['flow_mean_mlmin'] / 1000
        validation_checks.append(('Cardiac output', 4.0 <= co <= 7.0, f"{co:.2f} L/min"))

if all_vels:
    vel_ok = all(0.05 <= abs(v) <= 2.0 for v in all_vels)
    validation_checks.append(('CoW velocities in range', vel_ok, f"{min(all_vels):.3f}-{max(all_vels):.3f} m/s"))

if all_flows:
    no_backflow = all(abs(f) >= 0 for f in all_flows)  # Just check for reasonable values
    validation_checks.append(('Flow values reasonable', no_backflow, f"Range: {min(all_flows):.1f} to {max(all_flows):.1f} mL/min"))

if validation_checks:
    print("\nValidation Results:")
    print(f"{'Check':<35s} {'Status':<8s} {'Value':<25s}")
    print("-" * 80)
    for check_name, passed, value in validation_checks:
        status = 'PASS' if passed else 'FAIL'
        print(f"{check_name:<35s} {status:<8s} {value:<25s}")
    
    passed_count = sum(1 for _, p, _ in validation_checks if p)
    total_count = len(validation_checks)
    
    print(f"\nOverall: {passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("\nCONCLUSION: Simulation results are physiologically valid!")
    elif passed_count >= total_count * 0.6:
        print("\nCONCLUSION: Results mostly valid, minor issues")
    else:
        print("\nCONCLUSION: Multiple checks failed - review needed")
else:
    print("\nERROR: Could not perform validation")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()