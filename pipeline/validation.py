#!/usr/bin/env python3
"""
Unified Validation Script for FirstBlood Simulations
=====================================================
Self-contained validation for any simulation model.

Checks:
1. Cardiac output (4-7 L/min physiological range)
2. Circle of Willis mass balance
3. Cardiac cycle periodicity (convergence)
4. Physiological pressure ranges
5. Negative values check
6. Numerical stability (oscillations)

Usage:
    python3 validation.py --model patient_025
    python3 validation.py --model Abel_ref2
    python3 validation.py --model patient_025 --output-dir ./my_output

Output:
    JSON files saved to pipeline/output/validation/
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Constants
PA_TO_MMHG = 133.322
P_ATMO = 1.0e5


def get_repo_root():
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent


class SimulationValidator:
    """Self-contained simulation validator"""
    
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.validation_results = {}
        self.issues = []
        
    def load_vessel_data(self, vessel_id):
        """Load arterial data for a vessel"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        
        data = np.loadtxt(file_path, delimiter=',')
        return {
            'time': data[:, 0],
            'pressure_start': data[:, 1],
            'pressure_end': data[:, 2],
            'velocity_start': data[:, 3],
            'velocity_end': data[:, 4],
            'flow_start': data[:, 5],
            'flow_end': data[:, 6]
        }
    
    def load_vessel_flow(self, vessel_id, use_start=True):
        """Load mean flow for a vessel in mL/min"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        
        data = np.loadtxt(file_path, delimiter=',')
        flow = data[:, 5] if use_start else data[:, 6]
        mean_flow_ml_min = np.mean(flow) * 60 * 1000
        return mean_flow_ml_min

    def check_cardiac_output(self):
        """Check cardiac output from ascending aorta"""
        print("\n" + "="*70)
        print("1. CARDIAC OUTPUT CHECK")
        print("="*70)
        
        a1_file = self.results_path / "A1.txt"
        if not a1_file.exists():
            self.issues.append("[ERROR] A1.txt not found")
            print("[ERROR] A1.txt not found")
            return None
        
        data = np.loadtxt(a1_file, delimiter=',')
        flow_m3s = (data[:, 5] + data[:, 6]) / 2.0
        cardiac_output = np.mean(flow_m3s) * 1000 * 60  # m3/s -> L/min
        
        print(f"\nModel: {self.model_name}")
        print(f"Cardiac Output: {cardiac_output:.3f} L/min")
        
        if 4.0 <= cardiac_output <= 7.0:
            status = "PASS"
            print("[PASS] Within physiological range (4-7 L/min)")
        else:
            status = "WARNING"
            print("[WARNING] Outside normal range (4-7 L/min)")
            self.issues.append(f"Cardiac output {cardiac_output:.2f} L/min outside range")
        
        self.validation_results['cardiac_output'] = {
            'value_l_min': cardiac_output,
            'status': status
        }
        return cardiac_output

    def check_cow_balance(self):
        """Check Circle of Willis mass balance"""
        print("\n" + "="*70)
        print("2. CIRCLE OF WILLIS MASS BALANCE")
        print("="*70)
        
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
            flow = self.load_vessel_flow(vid, use_start=False)  # END of vessel
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
            flow = self.load_vessel_flow(vid, use_start=True)  # START of vessel
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
        
        print(f"\n" + "-"*50)
        print(f"BALANCE ANALYSIS:")
        print(f"  Total Inflow:    {total_in:7.2f} mL/min")
        print(f"  Total Outflow:   {total_out:7.2f} mL/min")
        print(f"  Difference:      {imbalance:+7.2f} mL/min")
        print(f"  Imbalance:       {imbalance_pct:6.1f}%")
        
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
            self.issues.append(f"CoW imbalance {imbalance_pct:.1f}%")
        
        print(f"  Status:          {status}")
        print(f"  Explanation:     {explanation}")
        
        self.validation_results['cow_balance'] = {
            'total_inflow_ml_min': total_in,
            'total_outflow_ml_min': total_out,
            'imbalance_ml_min': imbalance,
            'imbalance_percent': imbalance_pct,
            'status': status,
            'inflow_vessels': inflow_data,
            'outflow_vessels': outflow_data
        }
        return imbalance_pct

    def check_periodicity(self, vessel_id='A1'):
        """Check if cardiac cycles are periodic (converged)"""
        print("\n" + "="*70)
        print(f"3. PERIODICITY CHECK: {vessel_id} (Ascending Aorta)")
        print("="*70)
        
        data = self.load_vessel_data(vessel_id)
        if data is None:
            self.issues.append(f"[ERROR] Could not load {vessel_id}")
            return False
        
        time = data['time']
        pressure = data['pressure_start']
        
        # Find peaks in pressure
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(pressure, height=np.max(pressure)*0.8)
        except ImportError:
            # Fallback without scipy
            print("[WARNING] scipy not available, using simple peak detection")
            peaks = []
            for i in range(1, len(pressure)-1):
                if pressure[i] > pressure[i-1] and pressure[i] > pressure[i+1]:
                    if pressure[i] > np.max(pressure)*0.8:
                        peaks.append(i)
            peaks = np.array(peaks)
        
        if len(peaks) < 2:
            self.issues.append(f"[ERROR] Could not detect cardiac cycles in {vessel_id}")
            return False
        
        # Calculate cycle duration
        cycle_duration = np.mean(np.diff(time[peaks]))
        heart_rate = 60.0 / cycle_duration
        n_total_cycles = int(time[-1] / cycle_duration)
        
        print(f"\nCycle Duration: {cycle_duration:.3f} s")
        print(f"Heart Rate: {heart_rate:.1f} bpm")
        print(f"Total Cycles Simulated: {n_total_cycles}")
        
        if n_total_cycles < 3:
            self.issues.append(f"[WARNING] Only {n_total_cycles} cycles - need at least 3")
            print(f"[WARNING] Only {n_total_cycles} cycles simulated")
            self.validation_results['periodicity'] = 'INSUFFICIENT_DATA'
            return True
        
        # Compare last 3 cycles
        n_cycles = 3
        cycle_data = []
        
        for i in range(n_cycles):
            start_idx = peaks[-(n_cycles-i)]
            if i == n_cycles - 1:
                end_idx = len(time)
            else:
                end_idx = peaks[-(n_cycles-i-1)]
            
            cycle_pressure = pressure[start_idx:end_idx]
            cycle_data.append(cycle_pressure)
        
        # Calculate RMS differences between consecutive cycles
        rms_diffs = []
        for i in range(len(cycle_data) - 1):
            len_min = min(len(cycle_data[i]), len(cycle_data[i+1]))
            p1 = cycle_data[i][:len_min]
            p2 = cycle_data[i+1][:len_min]
            
            rms_diff = np.sqrt(np.mean((p1 - p2)**2))
            rms_diff_percent = rms_diff / np.mean(p1) * 100
            rms_diffs.append(rms_diff_percent)
            
            print(f"\nCycle {i+1} vs Cycle {i+2}:")
            print(f"  RMS difference: {rms_diff:.1f} Pa ({rms_diff_percent:.3f}%)")
        
        avg_rms = np.mean(rms_diffs)
        if avg_rms < 0.1:
            print(f"\n[PASS] Excellent periodicity (RMS < 0.1%)")
            self.validation_results['periodicity'] = 'PASS'
        elif avg_rms < 1.0:
            print(f"\n[PASS] Good periodicity (RMS < 1%)")
            self.validation_results['periodicity'] = 'PASS'
        else:
            print(f"\n[WARNING] Poor periodicity (RMS = {avg_rms:.2f}%)")
            self.validation_results['periodicity'] = 'WARNING'
            self.issues.append(f"Poor periodicity: RMS={avg_rms:.2f}%")
        
        return True

    def check_pressure_ranges(self):
        """Check physiological pressure ranges in key vessels"""
        print("\n" + "="*70)
        print("4. PRESSURE RANGE CHECK")
        print("="*70)
        
        vessels_to_check = {
            'A1': 'Ascending Aorta',
            'A12': 'R Internal Carotid',
            'A16': 'L Internal Carotid',
            'A70': 'R Middle Cerebral',
            'A73': 'L Middle Cerebral',
        }
        
        all_valid = True
        pressure_summary = []
        
        for vessel_id, vessel_name in vessels_to_check.items():
            data = self.load_vessel_data(vessel_id)
            if data is None:
                continue
            
            # Convert Pa to mmHg (subtract atmospheric pressure first)
            pressure_mmhg = (data['pressure_start'] - P_ATMO) / PA_TO_MMHG
            
            systolic = np.max(pressure_mmhg)
            diastolic = np.min(pressure_mmhg)
            mean_pressure = np.mean(pressure_mmhg)
            pulse_pressure = systolic - diastolic
            
            # Check ranges (allowing wider range)
            systolic_ok = 80 <= systolic <= 160
            diastolic_ok = 40 <= diastolic <= 100
            
            status_sym = "[OK]" if (systolic_ok and diastolic_ok) else "[FAIL]"
            
            print(f"\n{vessel_name} ({vessel_id}):")
            print(f"  Systolic:  {systolic:6.1f} mmHg {'[OK]' if systolic_ok else '[OUT OF RANGE]'}")
            print(f"  Diastolic: {diastolic:6.1f} mmHg {'[OK]' if diastolic_ok else '[OUT OF RANGE]'}")
            print(f"  Mean:      {mean_pressure:6.1f} mmHg")
            print(f"  Pulse:     {pulse_pressure:6.1f} mmHg")
            
            pressure_summary.append({
                'Vessel': vessel_name,
                'ID': vessel_id,
                'Systolic_mmHg': systolic,
                'Diastolic_mmHg': diastolic,
                'Mean_mmHg': mean_pressure,
                'Pulse_mmHg': pulse_pressure,
                'Valid': systolic_ok and diastolic_ok
            })
            
            if not (systolic_ok and diastolic_ok):
                all_valid = False
                self.issues.append(f"{vessel_name}: Pressure out of range")
        
        self.validation_results['pressure_ranges'] = 'PASS' if all_valid else 'FAIL'
        self.validation_results['pressure_summary'] = pressure_summary
        
        if all_valid:
            print(f"\n[PASS] All vessels within physiological range")
        else:
            print(f"\n[WARNING] Some vessels have pressures outside normal range")
        
        return all_valid

    def check_negative_values(self):
        """Check for negative pressures or flows"""
        print("\n" + "="*70)
        print("5. NEGATIVE VALUE CHECK")
        print("="*70)
        
        vessel_ids = ['A1', 'A12', 'A16', 'A70', 'A73', 'A60', 'A61', 'A59']
        
        negative_pressure_found = False
        unexpected_negative_flow = []
        
        for vessel_id in vessel_ids:
            data = self.load_vessel_data(vessel_id)
            if data is None:
                continue
            
            pressure = data['pressure_start']
            flow = data['flow_start']
            
            # Check for negative pressures
            if np.any(pressure < 0):
                negative_pressure_found = True
                min_pressure = np.min(pressure) / PA_TO_MMHG
                self.issues.append(f"{vessel_id}: Negative pressure ({min_pressure:.1f} mmHg)")
                print(f"[FAIL] {vessel_id}: Negative pressure detected ({min_pressure:.1f} mmHg)")
            
            # Check flow direction
            mean_flow = np.mean(flow)
            min_flow = np.min(flow)
            
            if mean_flow < -1e-6:  # Persistent negative mean flow
                unexpected_negative_flow.append(vessel_id)
                print(f"[WARNING] {vessel_id}: Mean flow is negative ({mean_flow*60*1000:.2f} mL/min)")
            elif min_flow < -1e-7:  # Diastolic backflow (normal)
                print(f"[INFO] {vessel_id}: Minor diastolic backflow detected (normal)")
        
        if not negative_pressure_found and len(unexpected_negative_flow) == 0:
            print(f"\n[PASS] No unexpected negative pressures or flows")
            self.validation_results['negative_values'] = 'PASS'
        else:
            print(f"\n[WARNING] Issues found with negative values")
            self.validation_results['negative_values'] = 'WARNING'
        
        return True

    def check_oscillations(self):
        """Check for numerical oscillations/artifacts"""
        print("\n" + "="*70)
        print("6. NUMERICAL STABILITY CHECK")
        print("="*70)
        
        vessel_id = 'A1'
        data = self.load_vessel_data(vessel_id)
        if data is None:
            return False
        
        pressure = data['pressure_start']
        flow = data['flow_start']
        time = data['time']
        
        # Check for high-frequency oscillations
        dt = time[1] - time[0]
        pressure_accel = np.diff(np.diff(pressure)) / dt**2
        
        accel_threshold = 1e9  # Pa/s^2
        max_accel = np.max(np.abs(pressure_accel))
        
        print(f"\nPressure acceleration analysis:")
        print(f"  Max acceleration: {max_accel:.2e} Pa/s^2")
        
        if max_accel > accel_threshold:
            print(f"  [WARNING] High-frequency oscillations detected")
            self.validation_results['oscillations'] = 'WARNING'
            self.issues.append("Possible numerical oscillations")
        else:
            print(f"  [OK] No significant oscillations")
            self.validation_results['oscillations'] = 'PASS'
        
        # Check flow smoothness
        flow_gradient = np.abs(np.diff(flow))
        mean_gradient = np.mean(flow_gradient)
        max_gradient = np.max(flow_gradient)
        
        print(f"\nFlow gradient analysis:")
        print(f"  Mean gradient: {mean_gradient:.2e} m^3/s per timestep")
        print(f"  Max gradient:  {max_gradient:.2e} m^3/s per timestep")
        
        if max_gradient > 100 * mean_gradient:
            print(f"  [WARNING] Possible flow discontinuities")
            self.validation_results['oscillations'] = 'WARNING'
        else:
            print(f"  [OK] Flow is smooth")
        
        return True

    def run_all(self):
        """Run all validation checks"""
        self.check_cardiac_output()
        self.check_cow_balance()
        self.check_periodicity()
        self.check_pressure_ranges()
        self.check_negative_values()
        self.check_oscillations()
        
        return self.validation_results, self.issues

    def get_summary(self):
        """Get validation summary"""
        summary = {
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'issues': self.issues,
            'overall_status': 'PASS' if len(self.issues) == 0 else 'WARNING'
        }
        return summary


def main():
    """Run validation workflow"""
    parser = argparse.ArgumentParser(
        description="Validate FirstBlood simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 validation.py --model patient_025
    python3 validation.py --models patient_025,patient_026
    python3 validation.py --models-file ./patients.txt
    python3 validation.py --all
    python3 validation.py --model patient_025 --output-dir ./my_output
        """
    )
    parser.add_argument('--model',
                       help='Single model name (e.g., patient_025, Abel_ref2)')
    parser.add_argument('--models',
                       help='Comma-separated model names (e.g., patient_025,patient_026)')
    parser.add_argument('--models-file',
                       help='Path to text file with one model name per line')
    parser.add_argument('--all', action='store_true',
                       help='Validate all models found in results directory')
    parser.add_argument('--results-base', default=None, 
                       help='Results base directory (default: projects/simple_run/results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: pipeline/output/validation)')
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = get_repo_root()
    results_base = Path(args.results_base) if args.results_base else repo_root / "projects/simple_run/results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "pipeline/output/validation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve model list (interactive prompt if none provided)
    model_names = []
    if args.model:
        model_names.append(args.model)
    if args.models:
        model_names.extend([m.strip() for m in args.models.split(',') if m.strip()])
    if args.models_file:
        models_file = Path(args.models_file)
        if not models_file.exists():
            print(f"[ERROR] models-file not found: {models_file}")
            sys.exit(1)
        with open(models_file, 'r') as f:
            model_names.extend([line.strip() for line in f if line.strip()])
    if args.all:
        if results_base.exists():
            model_names.extend([p.name for p in results_base.iterdir() if p.is_dir()])

    # De-duplicate while preserving order
    seen = set()
    model_names = [m for m in model_names if not (m in seen or seen.add(m))]

    if not model_names:
        try:
            entered = input("Enter model name (e.g., patient_025): ").strip()
        except EOFError:
            entered = ""
        if entered:
            model_names.append(entered)
        else:
            print("[ERROR] No models specified. Use --model, --models, --models-file, or --all.")
            sys.exit(1)

    overall_exit = 0

    for model_name in model_names:
        model_results = results_base / model_name / "arterial"

        # Check if results exist
        if not model_results.exists():
            print("="*70)
            print(f"VALIDATION FOR: {model_name}")
            print("="*70)
            print(f"[ERROR] Results not found: {model_results}")
            print(f"\nPlease run simulation first:")
            print(f"  cd ~/first_blood/projects/simple_run")
            print(f"  ./simple_run.out {model_name}")
            overall_exit = 1
            continue

        print("="*70)
        print(f"VALIDATION FOR: {model_name}")
        print("="*70)
        print(f"Results directory: {model_results}")
        print(f"Output directory:  {output_dir}")

        # Run validation
        validator = SimulationValidator(model_name, results_base)
        validator.run_all()

        # Get summary
        summary = validator.get_summary()

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        if summary['overall_status'] == 'PASS':
            print(f"\n[PASS] All validation checks passed for {model_name}")
        else:
            print(f"\n[WARNING] Issues found for {model_name}:")
            for issue in summary['issues']:
                print(f"  - {issue}")
            overall_exit = 1

        # Save results
        output_file = output_dir / f"{model_name}_validation.json"

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        summary_json = convert_numpy(summary)

        with open(output_file, 'w') as f:
            json.dump(summary_json, f, indent=2)

        print(f"\n[OK] Results saved: {output_file}")

    return overall_exit


if __name__ == '__main__':
    sys.exit(main())
