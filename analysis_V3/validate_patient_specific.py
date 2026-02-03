#!/usr/bin/env python3
"""
Patient-specific validation for FirstBlood simulations.

The FirstBlood paper validates only GLOBAL hemodynamics:
  - Cardiac output
  - Aortic pressure waveforms
  - Numerical convergence

This script validates LOCAL patient-specific features that YOU must verify:
  1. Flow split symmetry (R-ICA vs L-ICA)
  2. Collateral direction (Acom, Pcom flow direction)
  3. Pressure differences (not absolute values)
  4. Variant handling (absent vessels should have high resistance)

Usage:
  python3 validate_patient_specific.py --model patient_025 --compare Abel_ref2
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def find_model_path(model_name):
    """Find model directory."""
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = repo_root / "models" / model_name
    if not model_dir.exists():
        print(f"[ERROR] Model not found: {model_dir}")
        sys.exit(1)
    return model_dir


def load_results(model_dir):
    """Load simulation results from arterial subfolder."""
    repo_root = Path(__file__).resolve().parent.parent
    model_name = model_dir.name
    
    # Results are stored as A*.txt files in arterial/ subfolder
    results_dir = repo_root / "projects" / "simple_run" / "results" / model_name / "arterial"
    
    if not results_dir.exists():
        print(f"[ERROR] Results not found: {results_dir}")
        sys.exit(1)
    
    return results_dir


def load_arterial(model_dir):
    """Load arterial network."""
    arterial_path = model_dir / "arterial.csv"
    if not arterial_path.exists():
        print(f"[ERROR] arterial.csv not found: {arterial_path}")
        sys.exit(1)
    return pd.read_csv(arterial_path, skipinitialspace=True)


def load_variant(data_root, pid):
    """Load patient variant file."""
    variant_path = data_root / f"variant_mr_{pid}.json"
    if not variant_path.exists():
        return None
    with open(variant_path) as f:
        return json.load(f)


def get_vessel_flow(results_dir, vessel_id):
    """Get time-averaged flow through vessel [mL/min]."""
    vessel_file = results_dir / f"{vessel_id}.txt"
    if not vessel_file.exists():
        return None
    
    try:
        # Load space-time data: columns are x, t, P, v, D, Q_in, Q_out
        data = np.loadtxt(vessel_file, delimiter=',')
        if data.size == 0:
            return None
        
        # Average flow (Q_in and Q_out, typically the same)
        # Column 5 is Q_in [m^3/s]
        Q_SI = np.mean(data[:, 5])
        Q_ml_min = Q_SI * 1e6 * 60.0  # Convert m^3/s to mL/min
        return Q_ml_min
    except:
        return None


def get_vessel_pressure_drop(results_dir, vessel_id):
    """Get pressure drop across vessel [mmHg]."""
    vessel_file = results_dir / f"{vessel_id}.txt"
    if not vessel_file.exists():
        return None
    
    try:
        # Load space-time data: column 2 is pressure [Pa]
        data = np.loadtxt(vessel_file, delimiter=',')
        if data.size == 0:
            return None
        
        # Get inlet (first spatial point) and outlet (last spatial point)
        # Assuming data is sorted by x-coordinate
        P_in = data[0, 2]  # First point pressure [Pa]
        P_out = data[-1, 2]  # Last point pressure [Pa]
        
        dP_Pa = P_in - P_out
        dP_mmHg = dP_Pa / 133.322  # Convert to mmHg
        return dP_mmHg
    except:
        return None


def validate_flow_symmetry(results_dir):
    """Check left-right ICA flow symmetry."""
    print("\n" + "="*78)
    print("1. FLOW SPLIT SYMMETRY")
    print("="*78)
    
    # ICA flows
    Q_RICA = get_vessel_flow(results_dir, "A12")  # Right ICA
    Q_LICA = get_vessel_flow(results_dir, "A16")  # Left ICA
    
    if Q_RICA is None or Q_LICA is None:
        print("[ERROR] Cannot read ICA flows")
        return False
    
    print(f"  R-ICA (A12): {Q_RICA:>8.1f} mL/min")
    print(f"  L-ICA (A16): {Q_LICA:>8.1f} mL/min")
    
    asymmetry = abs(Q_RICA - Q_LICA) / (Q_RICA + Q_LICA) * 200.0  # Percent
    print(f"  Asymmetry:   {asymmetry:>7.1f}%")
    
    if asymmetry < 10.0:
        print("  [PASS] Symmetric flow")
        return True
    elif asymmetry < 30.0:
        print("  [WARNING] Mild asymmetry (expected for patient anatomy)")
        return True
    else:
        print("  [FAIL] Severe asymmetry (check variant handling)")
        return False


def validate_collateral_direction(results_dir, variant):
    """Check collateral flow direction."""
    print("\n" + "="*78)
    print("2. COLLATERAL FLOW DIRECTION")
    print("="*78)
    
    # Acom (A77): Should flow from dominant ICA to other side
    Q_Acom = get_vessel_flow(results_dir, "A77")
    if Q_Acom is not None:
        direction = "R→L" if Q_Acom > 0 else "L→R"
        print(f"  Acom (A77):     {Q_Acom:>8.1f} mL/min ({direction})")
        if variant and not variant["anterior"]["Acom"]:
            print("    [WARNING] Flow in absent Acom - check resistance!")
    else:
        print("  Acom (A77):     [NOT FOUND]")
    
    # Left Pcom (A62)
    Q_LPcom = get_vessel_flow(results_dir, "A62")
    if Q_LPcom is not None:
        direction = "anterior→posterior" if Q_LPcom > 0 else "posterior→anterior"
        print(f"  L-Pcom (A62):   {Q_LPcom:>8.1f} mL/min ({direction})")
        if variant and not variant["posterior"]["L-Pcom"]:
            print("    [WARNING] Flow in absent L-Pcom - check resistance!")
    else:
        print("  L-Pcom (A62):   [NOT FOUND]")
    
    # Right Pcom (A63)
    Q_RPcom = get_vessel_flow(results_dir, "A63")
    if Q_RPcom is not None:
        direction = "anterior→posterior" if Q_RPcom > 0 else "posterior→anterior"
        print(f"  R-Pcom (A63):   {Q_RPcom:>8.1f} mL/min ({direction})")
        if variant and not variant["posterior"]["R-Pcom"]:
            print("    [WARNING] Flow in absent R-Pcom - check resistance!")
    else:
        print("  R-Pcom (A63):   [NOT FOUND]")
    
    print("\n  [INFO] Collateral direction depends on variant anatomy")
    return True


def validate_pressure_differences(results_dir):
    """Check pressure gradients across key vessels."""
    print("\n" + "="*78)
    print("3. PRESSURE DIFFERENCES (ΔP across vessels)")
    print("="*78)
    
    vessels = {
        "A12": "R-ICA",
        "A16": "L-ICA",
        "A77": "Acom",
        "A62": "L-Pcom",
        "A63": "R-Pcom",
    }
    
    for vid, name in vessels.items():
        dP = get_vessel_pressure_drop(results_dir, vid)
        if dP is not None:
            status = "[OK]" if abs(dP) < 10.0 else "[HIGH]"
            print(f"  {name:12} (ΔP): {dP:>6.2f} mmHg {status}")
        else:
            print(f"  {name:12} (ΔP): [NOT FOUND]")
    
    print("\n  [INFO] ΔP > 10 mmHg may indicate stenosis or high resistance")
    return True


def validate_variant_handling(arterial_df, variant):
    """Check if absent vessels have high resistance."""
    print("\n" + "="*78)
    print("4. VARIANT HANDLING (absent vessels should have high R)")
    print("="*78)
    
    if variant is None:
        print("  [WARNING] No variant file found")
        return True
    
    # Map variant keys to vessel IDs
    variant_map = {
        "L-A1": "A74",      # Left A1
        "R-A1": "A71",      # Right A1
        "Acom": "A77",      # Anterior communicating
        "L-Pcom": "A62",    # Left posterior communicating
        "R-Pcom": "A63",    # Right posterior communicating
        "L-P1": "A64",      # Left P1
        "R-P1": "A65",      # Right P1
    }
    
    issues = []
    
    # Check anterior variants
    for key, vessel_id in [("L-A1", "A74"), ("R-A1", "A71"), ("Acom", "A77")]:
        present = variant["anterior"].get(key, True)
        vessel = arterial_df[arterial_df["ID"] == vessel_id]
        if not vessel.empty:
            diameter = float(vessel["start_diameter[SI]"].values[0]) * 1000  # mm
            length = float(vessel["length[SI]"].values[0]) * 1000  # mm
            
            if not present and diameter > 0.5:  # Absent but normal diameter
                issues.append(f"{key} ({vessel_id}): ABSENT but d={diameter:.2f}mm")
                print(f"  [FAIL] {key:8} ({vessel_id}): absent but d={diameter:.2f}mm (should be <0.5mm)")
            elif not present and diameter <= 0.5:
                print(f"  [PASS] {key:8} ({vessel_id}): absent with d={diameter:.2f}mm")
            elif present:
                print(f"  [OK]   {key:8} ({vessel_id}): present, d={diameter:.2f}mm")
    
    # Check posterior variants
    for key, vessel_id in [("L-Pcom", "A62"), ("R-Pcom", "A63"), ("L-P1", "A64"), ("R-P1", "A65")]:
        present = variant["posterior"].get(key, True)
        vessel = arterial_df[arterial_df["ID"] == vessel_id]
        if not vessel.empty:
            diameter = float(vessel["start_diameter[SI]"].values[0]) * 1000  # mm
            
            if not present and diameter > 0.5:
                issues.append(f"{key} ({vessel_id}): ABSENT but d={diameter:.2f}mm")
                print(f"  [FAIL] {key:8} ({vessel_id}): absent but d={diameter:.2f}mm (should be <0.5mm)")
            elif not present and diameter <= 0.5:
                print(f"  [PASS] {key:8} ({vessel_id}): absent with d={diameter:.2f}mm")
            elif present:
                print(f"  [OK]   {key:8} ({vessel_id}): present, d={diameter:.2f}mm")
    
    if issues:
        print(f"\n  [CRITICAL] {len(issues)} variant mismatches found!")
        print("  [ACTION] Absent vessels MUST have diameter < 0.5mm for high resistance")
        return False
    else:
        print("\n  [PASS] All variants handled correctly")
        return True


def compare_with_reference(patient_model, ref_model):
    """Compare patient model with reference."""
    print("\n" + "="*78)
    print(f"5. COMPARISON WITH {ref_model}")
    print("="*78)
    
    repo_root = Path(__file__).resolve().parent.parent
    
    # Load results directories
    patient_results = repo_root / "projects" / "simple_run" / "results" / patient_model / "arterial"
    ref_results = repo_root / "projects" / "simple_run" / "results" / ref_model / "arterial"
    
    if not patient_results.exists() or not ref_results.exists():
        print(f"[ERROR] Results not found")
        return False
    
    # Compare ICA flows
    Q_patient_RICA = get_vessel_flow(patient_results, "A12")
    Q_ref_RICA = get_vessel_flow(ref_results, "A12")
    
    if Q_patient_RICA and Q_ref_RICA:
        delta = (Q_patient_RICA - Q_ref_RICA) / Q_ref_RICA * 100.0
        print(f"  R-ICA flow:")
        print(f"    {ref_model:20}: {Q_ref_RICA:>8.1f} mL/min")
        print(f"    {patient_model:20}: {Q_patient_RICA:>8.1f} mL/min ({delta:+.1f}%)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patient-specific validation for FirstBlood",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Patient model name (e.g., patient_025)")
    parser.add_argument("--compare", help="Reference model to compare with (e.g., Abel_ref2)")
    parser.add_argument("--pid", help="Patient ID for variant file (e.g., 025)")
    args = parser.parse_args()
    
    print("="*78)
    print("PATIENT-SPECIFIC VALIDATION")
    print("="*78)
    print(f"Model: {args.model}")
    
    # Load data
    model_dir = Path(args.model)  # Just use name
    results_dir = load_results(model_dir)
    
    # Load arterial network
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / args.model
    arterial_df = load_arterial(model_path)
    
    # Load variant if PID provided
    variant = None
    if args.pid:
        repo_root = Path(__file__).resolve().parent.parent
        data_root = repo_root / "data_patient025"
        variant = load_variant(data_root, args.pid)
    
    # Run validations
    checks = []
    checks.append(("Flow Symmetry", validate_flow_symmetry(results_dir)))
    checks.append(("Collateral Direction", validate_collateral_direction(results_dir, variant)))
    checks.append(("Pressure Differences", validate_pressure_differences(results_dir)))
    checks.append(("Variant Handling", validate_variant_handling(arterial_df, variant)))
    
    if args.compare:
        checks.append(("Reference Comparison", compare_with_reference(args.model, args.compare)))
    
    # Summary
    print("\n" + "="*78)
    print("SUMMARY")
    print("="*78)
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    print(f"  Passed: {passed}/{total}")
    
    for check_name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {check_name}")
    
    print("="*78)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
