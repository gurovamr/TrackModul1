#!/usr/bin/env python3
"""
FirstBlood Simulation Validation Script
Validates patient025_CoW_v2 and Abel_ref2 simulation quality

Checks:
1. Cardiac cycle periodicity (convergence)
2. Physiological pressure ranges
3. No negative flows or pressures
4. Numerical artifacts/oscillations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

class SimulationValidator:
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
    
    def check_periodicity(self, vessel_id='A1', n_cycles=3):
        """Check if cardiac cycles are periodic (converged)"""
        print(f"\n{'='*70}")
        print(f"1. PERIODICITY CHECK: {vessel_id} (Ascending Aorta)")
        print(f"{'='*70}")
        
        data = self.load_vessel_data(vessel_id)
        if data is None:
            self.issues.append(f"❌ Could not load {vessel_id}")
            return False
        
        time = data['time']
        pressure = data['pressure_start']
        
        # Estimate heart rate and cycle duration
        # Find peaks in pressure
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pressure, height=np.max(pressure)*0.8)
        
        if len(peaks) < 2:
            self.issues.append(f"❌ Could not detect cardiac cycles in {vessel_id}")
            return False
        
        # Calculate cycle duration
        cycle_duration = np.mean(np.diff(time[peaks]))
        heart_rate = 60.0 / cycle_duration
        n_total_cycles = int(time[-1] / cycle_duration)
        
        print(f"\nCycle Duration: {cycle_duration:.3f} s")
        print(f"Heart Rate: {heart_rate:.1f} bpm")
        print(f"Total Cycles Simulated: {n_total_cycles}")
        
        if n_total_cycles < 3:
            self.issues.append(f"⚠️  Only {n_total_cycles} cycles - need at least 3 for validation")
            print(f"⚠️  WARNING: Only {n_total_cycles} cycles simulated")
        
        # Extract last n_cycles for comparison
        if n_total_cycles >= n_cycles:
            cycle_starts = time[peaks[-n_cycles:]]
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
                # Interpolate to same length
                len_min = min(len(cycle_data[i]), len(cycle_data[i+1]))
                p1 = cycle_data[i][:len_min]
                p2 = cycle_data[i+1][:len_min]
                
                rms_diff = np.sqrt(np.mean((p1 - p2)**2))
                rms_diff_percent = rms_diff / np.mean(p1) * 100
                rms_diffs.append(rms_diff_percent)
                
                print(f"\nCycle {i+1} vs Cycle {i+2}:")
                print(f"  RMS difference: {rms_diff:.1f} Pa ({rms_diff_percent:.3f}%)")
            
            # Check convergence criterion
            avg_rms = np.mean(rms_diffs)
            if avg_rms < 0.1:
                print(f"\n✓ PASSED: Excellent periodicity (RMS < 0.1%)")
                self.validation_results['periodicity'] = 'PASS'
            elif avg_rms < 1.0:
                print(f"\n✓ PASSED: Good periodicity (RMS < 1%)")
                self.validation_results['periodicity'] = 'PASS'
            else:
                print(f"\n⚠️  WARNING: Poor periodicity (RMS = {avg_rms:.2f}%)")
                print(f"    Simulation may not be fully converged")
                self.validation_results['periodicity'] = 'WARNING'
                self.issues.append(f"⚠️  Poor periodicity: RMS={avg_rms:.2f}%")
        
        else:
            print(f"\n⚠️  Not enough cycles for periodicity check")
            self.validation_results['periodicity'] = 'INSUFFICIENT_DATA'
        
        return True
    
    def check_pressure_ranges(self):
        """Check physiological pressure ranges in key vessels"""
        print(f"\n{'='*70}")
        print(f"2. PRESSURE RANGE CHECK")
        print(f"{'='*70}")
        
        # Check multiple locations
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
            
            # Convert Pa to mmHg (1 mmHg = 133.322 Pa)
            pressure_mmhg = data['pressure_start'] / 133.322
            
            systolic = np.max(pressure_mmhg)
            diastolic = np.min(pressure_mmhg)
            mean_pressure = np.mean(pressure_mmhg)
            pulse_pressure = systolic - diastolic
            
            # Check ranges
            systolic_ok = 80 <= systolic <= 160  # Allowing wider range
            diastolic_ok = 40 <= diastolic <= 100
            
            status = "✓" if (systolic_ok and diastolic_ok) else "✗"
            
            print(f"\n{vessel_name} ({vessel_id}):")
            print(f"  Systolic:  {systolic:6.1f} mmHg {' ✓' if systolic_ok else ' ✗ OUT OF RANGE'}")
            print(f"  Diastolic: {diastolic:6.1f} mmHg {' ✓' if diastolic_ok else ' ✗ OUT OF RANGE'}")
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
                self.issues.append(f"✗ {vessel_name}: Pressure out of range")
        
        self.validation_results['pressure_ranges'] = 'PASS' if all_valid else 'FAIL'
        self.validation_results['pressure_summary'] = pd.DataFrame(pressure_summary)
        
        if all_valid:
            print(f"\n{'='*70}")
            print("✓ ALL VESSELS: Pressures within physiological range")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print("⚠️  WARNING: Some vessels have pressures outside normal range")
            print(f"{'='*70}")
        
        return all_valid
    
    def check_negative_values(self):
        """Check for negative pressures or flows (except backflow)"""
        print(f"\n{'='*70}")
        print(f"3. NEGATIVE VALUE CHECK")
        print(f"{'='*70}")
        
        # Check several vessels
        vessel_ids = ['A1', 'A12', 'A16', 'A70', 'A73', 'A60', 'A61', 'A59']
        
        negative_pressure_found = False
        unexpected_negative_flow = []
        
        for vessel_id in vessel_ids:
            data = self.load_vessel_data(vessel_id)
            if data is None:
                continue
            
            pressure = data['pressure_start']
            flow = data['flow_start']
            
            # Check for negative pressures (absolute pressure)
            if np.any(pressure < 0):
                negative_pressure_found = True
                min_pressure = np.min(pressure) / 133.322
                self.issues.append(f"✗ {vessel_id}: Negative pressure ({min_pressure:.1f} mmHg)")
                print(f"✗ {vessel_id}: Negative pressure detected ({min_pressure:.1f} mmHg)")
            
            # Check flow direction
            mean_flow = np.mean(flow)
            min_flow = np.min(flow)
            
            # Some backflow during diastole is normal
            if mean_flow < -1e-6:  # Persistent negative mean flow
                unexpected_negative_flow.append(vessel_id)
                print(f"⚠️  {vessel_id}: Mean flow is negative ({mean_flow*60*1000:.2f} mL/min)")
            elif min_flow < -1e-7:  # Diastolic backflow
                print(f"ℹ️  {vessel_id}: Minor diastolic backflow detected (normal)")
        
        if not negative_pressure_found and len(unexpected_negative_flow) == 0:
            print(f"\n✓ PASSED: No unexpected negative pressures or flows")
            self.validation_results['negative_values'] = 'PASS'
        else:
            print(f"\n⚠️  Issues found with negative values")
            self.validation_results['negative_values'] = 'WARNING'
        
        return True
    
    def check_oscillations(self):
        """Check for numerical oscillations/artifacts"""
        print(f"\n{'='*70}")
        print(f"4. NUMERICAL STABILITY CHECK")
        print(f"{'='*70}")
        
        vessel_id = 'A1'
        data = self.load_vessel_data(vessel_id)
        if data is None:
            return False
        
        pressure = data['pressure_start']
        flow = data['flow_start']
        time = data['time']
        
        # Check for high-frequency oscillations
        # Calculate second derivative (acceleration)
        dt = time[1] - time[0]
        pressure_accel = np.diff(np.diff(pressure)) / dt**2
        
        # Look for abnormally large accelerations
        accel_threshold = 1e9  # Pa/s^2 (adjust based on typical values)
        max_accel = np.max(np.abs(pressure_accel))
        
        print(f"\nPressure acceleration analysis:")
        print(f"  Max acceleration: {max_accel:.2e} Pa/s²")
        
        if max_accel > accel_threshold:
            print(f"  ⚠️  High-frequency oscillations detected")
            self.validation_results['oscillations'] = 'WARNING'
            self.issues.append("⚠️  Possible numerical oscillations")
        else:
            print(f"  ✓ No significant oscillations")
            self.validation_results['oscillations'] = 'PASS'
        
        # Check flow smoothness
        flow_gradient = np.abs(np.diff(flow))
        mean_gradient = np.mean(flow_gradient)
        max_gradient = np.max(flow_gradient)
        
        print(f"\nFlow gradient analysis:")
        print(f"  Mean gradient: {mean_gradient:.2e} m³/s per timestep")
        print(f"  Max gradient:  {max_gradient:.2e} m³/s per timestep")
        
        if max_gradient > 100 * mean_gradient:
            print(f"  ⚠️  Possible flow discontinuities")
            self.validation_results['oscillations'] = 'WARNING'
        else:
            print(f"  ✓ Flow is smooth")
        
        return True
    
    def check_mass_conservation(self):
        """Check mass conservation at major bifurcations"""
        print(f"\n{'='*70}")
        print(f"5. MASS CONSERVATION CHECK")
        print(f"{'='*70}")
        
        # Check a few key junctions
        # Aortic arch: A1 should split into A2 (arch) + A3 (brachiocephalic)
        
        print(f"\nChecking major bifurcations...")
        print(f"(Note: Some error expected due to peripheral leakage)")
        
        # Simple check: Compare inflow to CoW with outflow
        cow_inflow_ids = ['A12', 'A16', 'A59']  # R-ICA, L-ICA, Basilar
        cow_outflow_ids = ['A70', 'A73', 'A76', 'A78', 'A64', 'A65']  # MCAs, ACAs, PCAs
        
        inflow_total = 0
        outflow_total = 0
        
        for vessel_id in cow_inflow_ids:
            data = self.load_vessel_data(vessel_id)
            if data:
                flow = np.mean(data['flow_start']) * 60 * 1000  # mL/min
                inflow_total += flow
        
        for vessel_id in cow_outflow_ids:
            data = self.load_vessel_data(vessel_id)
            if data:
                flow = np.mean(data['flow_start']) * 60 * 1000  # mL/min
                outflow_total += flow
        
        balance_error = abs(inflow_total - outflow_total) / inflow_total * 100
        
        print(f"\nCircle of Willis balance:")
        print(f"  Total inflow:  {inflow_total:.2f} mL/min")
        print(f"  Total outflow: {outflow_total:.2f} mL/min")
        print(f"  Imbalance:     {balance_error:.2f}%")
        
        if balance_error < 5:
            print(f"  ✓ Excellent mass conservation")
        elif balance_error < 15:
            print(f"  ✓ Acceptable mass conservation")
        else:
            print(f"  ⚠️  Large imbalance - check peripheral resistances")
        
        return True
    
    def generate_summary_plot(self, save_path=None):
        """Generate summary validation plot"""
        print(f"\n{'='*70}")
        print(f"6. GENERATING VALIDATION PLOTS")
        print(f"{'='*70}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Simulation Validation: {self.model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Pressure waveform with last 3 cycles
        ax1 = axes[0, 0]
        data = self.load_vessel_data('A1')
        if data:
            time = data['time']
            pressure_mmhg = data['pressure_start'] / 133.322
            
            # Plot last 3 seconds
            last_3s = time >= (time[-1] - 3.0)
            ax1.plot(time[last_3s], pressure_mmhg[last_3s], 'b-', linewidth=1.5)
            ax1.set_xlabel('Time (s)', fontsize=11)
            ax1.set_ylabel('Pressure (mmHg)', fontsize=11)
            ax1.set_title('Aortic Pressure - Last 3 Cycles', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=120, color='r', linestyle='--', alpha=0.5, label='Normal systolic')
            ax1.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Normal diastolic')
            ax1.legend(fontsize=9)
        
        # Plot 2: Flow waveform
        ax2 = axes[0, 1]
        if data:
            flow_ml_min = data['flow_start'] * 60 * 1000
            ax2.plot(time[last_3s], flow_ml_min[last_3s], 'r-', linewidth=1.5)
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Flow (mL/min)', fontsize=11)
            ax2.set_title('Aortic Flow - Last 3 Cycles', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot 3: Pressure distribution across vessels
        ax3 = axes[1, 0]
        if 'pressure_summary' in self.validation_results:
            df = self.validation_results['pressure_summary']
            x = range(len(df))
            ax3.bar(x, df['Systolic_mmHg'], alpha=0.7, label='Systolic', color='red')
            ax3.bar(x, df['Diastolic_mmHg'], alpha=0.7, label='Diastolic', color='blue')
            ax3.set_xticks(x)
            ax3.set_xticklabels(df['ID'], rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Pressure (mmHg)', fontsize=11)
            ax3.set_title('Pressure Ranges Across Vessels', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhspan(100, 140, alpha=0.1, color='red', label='Normal systolic range')
            ax3.axhspan(60, 90, alpha=0.1, color='blue', label='Normal diastolic range')
        
        # Plot 4: Validation summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"VALIDATION SUMMARY\n{'='*40}\n\n"
        
        for key, value in self.validation_results.items():
            if key != 'pressure_summary':
                icon = "✓" if value == "PASS" else "⚠️" if value == "WARNING" else "ℹ️"
                summary_text += f"{icon} {key.replace('_', ' ').title()}: {value}\n"
        
        summary_text += f"\n{'='*40}\n"
        summary_text += f"Issues Found: {len(self.issues)}\n\n"
        
        if self.issues:
            summary_text += "Issues:\n"
            for issue in self.issues[:5]:  # Show first 5 issues
                summary_text += f"  • {issue}\n"
        else:
            summary_text += "✓ No major issues detected\n"
            summary_text += "✓ Simulation is valid\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Validation plot saved: {save_path}")
        
        return fig
    
    def run_all_checks(self):
        """Run all validation checks"""
        print(f"\n{'#'*70}")
        print(f"# FIRSTBLOOD SIMULATION VALIDATION")
        print(f"# Model: {self.model_name}")
        print(f"# Results path: {self.results_path}")
        print(f"{'#'*70}")
        
        self.check_periodicity()
        self.check_pressure_ranges()
        self.check_negative_values()
        self.check_oscillations()
        self.check_mass_conservation()
        
        # Generate summary
        print(f"\n{'='*70}")
        print(f"FINAL VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        all_pass = all(v == 'PASS' for k, v in self.validation_results.items() 
                      if k != 'pressure_summary')
        
        if all_pass:
            print(f"\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
            print(f"Simulation is valid and ready for analysis")
        else:
            print(f"\n⚠️  SOME CHECKS FAILED OR HAVE WARNINGS")
            print(f"Review issues before proceeding with analysis")
            print(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        
        return all_pass


def main():
    """Main validation function"""
    
    # Set paths
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/validation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Validate both models
    models = ['Abel_ref2', 'patient025_CoW_v2']
    
    for model_name in models:
        print(f"\n\n{'#'*70}")
        print(f"# VALIDATING: {model_name}")
        print(f"{'#'*70}\n")
        
        validator = SimulationValidator(model_name, results_base)
        validator.run_all_checks()
        
        # Generate plots
        plot_path = output_dir / f"validation_{model_name}.png"
        validator.generate_summary_plot(save_path=plot_path)
        
        # Save validation results
        results_file = output_dir / f"validation_{model_name}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Validation Results: {model_name}\n")
            f.write(f"{'='*70}\n\n")
            for key, value in validator.validation_results.items():
                if key != 'pressure_summary':
                    f.write(f"{key}: {value}\n")
            f.write(f"\nIssues ({len(validator.issues)}):\n")
            for issue in validator.issues:
                f.write(f"  {issue}\n")
        
        print(f"\n✓ Validation results saved: {results_file}")
    
    print(f"\n\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()