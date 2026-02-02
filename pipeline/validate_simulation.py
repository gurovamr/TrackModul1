#!/usr/bin/env python3
"""
Modular Validation Workflow for FirstBlood Simulations
======================================================
Runs essential validation checks for any simulation model:
1. Quick cardiac output check
2. Mass conservation (CoW balance)
3. Full simulation validation
4. Comparison with Abel_ref2

Usage:
    cd ~/first_blood/pipeline
    python3 validate_simulation.py --model MODEL_NAME

Example:
    python3 validate_simulation.py --model patient_025
    python3 validate_simulation.py --model Abel_ref2
"""

import sys
import argparse
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

def get_repo_root():
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent

def load_vessel_flow(results_path, vessel_id, use_start=True):
    """Load mean flow for a vessel"""
    file_path = results_path / f"{vessel_id}.txt"
    if not file_path.exists():
        return None
    
    data = np.loadtxt(file_path, delimiter=',')
    flow = data[:, 5] if use_start else data[:, 6]  # start or end
    mean_flow_ml_min = np.mean(flow) * 60 * 1000
    return mean_flow_ml_min

def check_cardiac_output(model_name, results_base, output_dir):
    """Check cardiac output and save results"""
    print("\n" + "="*80)
    print("STEP 1: CARDIAC OUTPUT CHECK")
    print("="*80)
    
    results_path = results_base / model_name / "arterial"
    a1_file = results_path / "A1.txt"
    
    if not a1_file.exists():
        print(f"[ERROR] A1.txt not found: {a1_file}")
        return False
    
    data = np.loadtxt(a1_file, delimiter=',')
    flow_m3s = (data[:, 5] + data[:, 6]) / 2.0
    cardiac_output = np.mean(flow_m3s) * 1000 * 60  # m³/s -> L/min
    
    print(f"\nModel: {model_name}")
    print(f"Cardiac Output: {cardiac_output:.3f} L/min")
    
    # Check physiological range
    if 4.0 <= cardiac_output <= 7.0:
        status = "PASS"
        print("[PASS] Within physiological range (4-7 L/min)")
    else:
        status = "WARNING"
        print(f"[WARNING] Outside normal range (4-7 L/min)")
    
    # Save results
    results = {
        'model': model_name,
        'cardiac_output_l_min': cardiac_output,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = output_dir / f"{model_name}_cardiac_output.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Results saved: {output_file}")
    return True

def check_cow_balance(model_name, results_base, output_dir):
    """Check Circle of Willis mass balance"""
    print("\n" + "="*80)
    print("STEP 2: CIRCLE OF WILLIS MASS BALANCE")
    print("="*80)
    
    results_path = results_base / model_name / "arterial"
    
    # Define vessels
    inflow_vessels = {
        'R-ICA': 'A12',
        'L-ICA': 'A16', 
        'Basilar': 'A59'
    }
    
    outflow_vessels = {
        'R-MCA': 'A70',
        'L-MCA': 'A73',
        'R-ACA': 'A76',
        'L-ACA': 'A78',
        'R-PCA': 'A64',
        'L-PCA': 'A65'
    }
    
    # Calculate inflows
    print("\nINFLOW (entering CoW):")
    total_in = 0
    inflow_data = {}
    
    for name, vid in inflow_vessels.items():
        flow = load_vessel_flow(results_path, vid, use_start=False)  # END of vessel
        if flow is not None:
            total_in += flow
            inflow_data[name] = flow
            print(f"  {name:12s} ({vid}): {flow:7.2f} mL/min")
        else:
            print(f"  {name:12s} ({vid}): MISSING")
    
    print(f"  {'TOTAL':12s}      : {total_in:7.2f} mL/min")
    
    # Calculate outflows  
    print(f"\nOUTFLOW (leaving CoW):")
    total_out = 0
    outflow_data = {}
    
    for name, vid in outflow_vessels.items():
        flow = load_vessel_flow(results_path, vid, use_start=True)  # START of vessel
        if flow is not None:
            total_out += flow
            outflow_data[name] = flow
            print(f"  {name:12s} ({vid}): {flow:7.2f} mL/min")
        else:
            print(f"  {name:12s} ({vid}): MISSING")
    
    print(f"  {'TOTAL':12s}      : {total_out:7.2f} mL/min")
    
    # Calculate balance
    imbalance = total_in - total_out
    imbalance_pct = abs(imbalance) / total_in * 100 if total_in > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"BALANCE ANALYSIS:")
    print(f"  Total Inflow:    {total_in:7.2f} mL/min")
    print(f"  Total Outflow:   {total_out:7.2f} mL/min")
    print(f"  Difference:      {imbalance:+7.2f} mL/min")
    print(f"  Imbalance:       {imbalance_pct:6.1f}%")
    
    # Interpret
    if imbalance_pct < 5:
        status = "EXCELLENT"
        explanation = "Perfect conservation"
    elif imbalance_pct < 15:
        status = "GOOD"
        explanation = "Acceptable - minor peripheral leakage"
    elif imbalance_pct < 30:
        status = "MODERATE"
        explanation = "Peripheral resistances may need adjustment"
    else:
        status = "LARGE"
        explanation = "Significant peripheral leakage"
    
    print(f"  Status:          {status}")
    print(f"  Explanation:     {explanation}")
    
    # Save results
    results = {
        'model': model_name,
        'total_inflow_ml_min': total_in,
        'total_outflow_ml_min': total_out,
        'imbalance_ml_min': imbalance,
        'imbalance_percent': imbalance_pct,
        'status': status,
        'explanation': explanation,
        'inflow_vessels': inflow_data,
        'outflow_vessels': outflow_data,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = output_dir / f"{model_name}_cow_balance.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Results saved: {output_file}")
    return True

def run_full_validation(model_name, output_dir):
    """Run the full validation suite"""
    print("\n" + "="*80)
    print("STEP 3: FULL VALIDATION SUITE")
    print("="*80)
    
    # Import and run validate_simulation.py
    try:
        sys.path.insert(0, str(get_repo_root() / "analysis_V3"))
        from validate_simulation import SimulationValidator
        
        validator = SimulationValidator(model_name, get_repo_root() / "projects/simple_run/results")
        
        # Run key validation checks
        validator.check_periodicity()
        validator.check_pressure_ranges()
        validator.check_negative_values()
        validator.check_oscillations()
        validator.check_mass_conservation()
        
        # Save validation results
        results = {
            'model': model_name,
            'issues': validator.issues,
            'validation_results': validator.validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = output_dir / f"{model_name}_full_validation.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[OK] Full validation results saved: {output_file}")
        return True
        
    except ImportError as e:
        print(f"[WARNING] Could not import validate_simulation.py: {e}")
        print("  Run manually: cd ~/first_blood/analysis_V3 && python3 validate_simulation.py")
        return False

def run_comparison(model_name, output_dir):
    """Run comparison with Abel_ref2"""
    print("\n" + "="*80)
    print("STEP 4: COMPARISON WITH ABEL_REF2 REFERENCE")
    print("="*80)
    
    try:
        # Import and run compare_abel.py
        sys.path.insert(0, str(get_repo_root() / "analysis_V3"))
        import compare_abel
        
        print("[OK] Comparison plot generated (check analysis_V3/validation_results/)")
        return True
        
    except ImportError as e:
        print(f"[WARNING] Could not import compare_abel.py: {e}")
        print("  Run manually: cd ~/first_blood/analysis_V3 && python3 compare_abel.py")
        return False

def main():
    """Run validation workflow"""
    parser = argparse.ArgumentParser(description="Validate FirstBlood simulation results")
    parser.add_argument('--model', required=True, help='Model name (e.g., patient_025, Abel_ref2)')
    parser.add_argument('--results-base', default=None, 
                       help='Results base directory (default: ~/first_blood/projects/simple_run/results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ~/first_blood/pipeline/output/validation)')
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = get_repo_root()
    results_base = Path(args.results_base) if args.results_base else repo_root / "projects/simple_run/results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "pipeline/output/validation"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model
    model_results = results_base / model_name / "arterial"
    
    # Check if results exist
    if not model_results.exists():
        print(f"[ERROR] Results not found: {model_results}")
        print(f"\nPlease run simulation first:")
        print(f"  cd ~/first_blood/projects/simple_run")
        print(f"  ./simple_run.out {model_name}")
        return
    
    print("="*80)
    print(f"VALIDATION WORKFLOW FOR: {model_name}")
    print("="*80)
    print(f"Results directory: {model_results}")
    print(f"Output directory:  {output_dir}")
    
    # Run validation steps
    success = True
    
    success &= check_cardiac_output(model_name, results_base, output_dir)
    success &= check_cow_balance(model_name, results_base, output_dir)
    success &= run_full_validation(model_name, output_dir)
    success &= run_comparison(model_name, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if success:
        print(f"\n[PASS] All validation checks completed for {model_name}")
    else:
        print(f"\n[WARNING] Some validation checks had issues for {model_name}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review JSON files in {output_dir}")
    print(f"  2. Check plots in ~/first_blood/analysis_V3/validation_results/")
    print(f"  3. Run individual scripts if needed")

if __name__ == '__main__':
    main()
