#!/usr/bin/env python3
"""
Complete validation workflow for patient models.
Runs both global (paper-scope) and patient-specific (your responsibility) checks.

Usage:
  python3 validate_complete.py --model patient_025 --pid 025
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_global_validation(model_name):
    """Run paper-scope global validation."""
    print("\n" + "="*80)
    print("STEP 1: GLOBAL VALIDATION (Paper Scope)")
    print("="*80)
    
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        "python3",
        str(repo_root / "pipeline" / "validation.py"),
        "--model", model_name
    ]
    
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode == 0


def run_patient_validation(model_name, pid=None, compare=None):
    """Run patient-specific validation."""
    print("\n" + "="*80)
    print("STEP 2: PATIENT-SPECIFIC VALIDATION (Your Responsibility)")
    print("="*80)
    
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        "python3",
        str(repo_root / "analysis_V3" / "validate_patient_specific.py"),
        "--model", model_name
    ]
    
    if pid:
        cmd.extend(["--pid", pid])
    if compare:
        cmd.extend(["--compare", compare])
    
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode == 0


def print_summary(global_ok, patient_ok):
    """Print validation summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    status_global = "[PASS]" if global_ok else "[WARNING]"
    status_patient = "[PASS]" if patient_ok else "[FAIL]"
    
    print(f"\n{status_global} Global validation (Cardiac output, pressures, stability)")
    print(f"  → The paper validates global hemodynamics only (CO, waveform shape)")
    print(f"  → Absolute local pressures are model-dependent, not validated by paper\n")
    
    print(f"{status_patient} Patient-specific validation (Flow symmetry, collaterals, variants)")
    print(f"  → YOUR responsibility to check: flow splits, collateral direction")
    print(f"  → Ensures patient anatomy is correctly represented\n")
    
    if patient_ok:
        print("[SUCCESS] Patient model is valid for CoW analysis ✓\n")
        print("Next steps:")
        print("  1. Analyze flow distribution: analysis_V3/analyze_cow_flows.py")
        print("  2. Compare waveforms: analysis_V3/compare_abel.py")
        print("  3. Study collateral pathways: Use pressure differences from above\n")
        return 0
    else:
        print("[ACTION REQUIRED] Fix patient-specific checks before proceeding\n")
        print("See PATIENT_VALIDATION_GUIDE.md for debugging steps\n")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Complete validation workflow for patient models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validate_complete.py --model patient_025 --pid 025
  python3 validate_complete.py --model patient_025 --pid 025 --compare Abel_ref2
  python3 validate_complete.py --model Abel_ref2

See PATIENT_VALIDATION_GUIDE.md for interpretation guide.
        """
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., patient_025)")
    parser.add_argument("--pid", help="Patient ID for variant file (e.g., 025)")
    parser.add_argument("--compare", help="Reference model (e.g., Abel_ref2)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPLETE VALIDATION WORKFLOW")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Variant PID: {args.pid if args.pid else 'Not specified'}")
    print(f"Comparison: {args.compare if args.compare else 'No reference'}")
    
    # Run both validations
    global_ok = run_global_validation(args.model)
    patient_ok = run_patient_validation(args.model, args.pid, args.compare)
    
    # Print summary
    return print_summary(global_ok, patient_ok)


if __name__ == "__main__":
    sys.exit(main())
