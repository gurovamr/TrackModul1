#!/usr/bin/env python3
"""
Analysis Script for FirstBlood Simulations
===========================================
Performs detailed analysis aligned with FirstBlood paper validation requirements.

Required Analysis (from paper):
1. Temporal history plots (pressure & velocity waveforms)
2. Heart model dynamics (atrium, ventricle, aorta pressures; valve flows)
3. Waveform amplitude and shape verification
4. Grid convergence analysis (optional - requires coarse/normal/fine runs)
5. Runtime efficiency measurement

Usage:
    python3 analysis.py                          # Interactive prompt
    python3 analysis.py --model patient_025
    python3 analysis.py --models patient_025,patient_026
    python3 analysis.py --all

Output:
    Plots and data saved to pipeline/output/analysis/<model_name>/
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Constants
PA_TO_MMHG = 133.322
P_ATMO = 1.0e5
MS_TO_MMS = 1000.0


def get_repo_root():
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent


class SimulationAnalyzer:
    """Analyze FirstBlood simulation results"""
    
    def __init__(self, model_name: str, results_base: Path, output_base: Path):
        self.model_name = model_name
        self.results_path = results_base / model_name / "arterial"
        self.heart_path = results_base / model_name / "heart_kim_lit"
        self.output_path = output_base / model_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.analysis_results = {}
        
    def load_vessel_timeseries(self, vessel_id: str) -> Optional[pd.DataFrame]:
        """Load full time-series data for a vessel"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        
        data = np.loadtxt(file_path, delimiter=',')
        # Pressure at inlet (column 1) and outlet (column 2), average them
        pressure_pa = (data[:, 1] + data[:, 2]) / 2.0
        pressure_gauge = (pressure_pa - P_ATMO) / PA_TO_MMHG  # Convert to gauge mmHg
        
        df = pd.DataFrame({
            'time': data[:, 0],
            'pressure': pressure_gauge,
            'velocity': data[:, 3] * MS_TO_MMS,    # Convert to mm/s
            'diameter': data[:, 9] * 1000,         # Convert to mm (column 9)
            'flow_in': data[:, 5] * 1e6,           # Convert to mL/s
            'flow_out': data[:, 6] * 1e6           # Convert to mL/s
        })
        return df
    
    def load_heart_timeseries(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load heart model data"""
        heart_data = {}
        
        # Heart chamber pressure files (actual names: E_la.txt, E_lv.txt, etc.)
        chamber_map = {
            'atrium': 'E_la.txt',     # Left atrium pressure
            'ventricle': 'E_lv.txt',  # Left ventricle pressure
        }
        
        for chamber, filename in chamber_map.items():
            file_path = self.heart_path / filename
            if file_path.exists():
                data = np.loadtxt(file_path, delimiter=',')
                pressure_gauge = (data[:, 1] - P_ATMO) / PA_TO_MMHG
                heart_data[chamber] = pd.DataFrame({
                    'time': data[:, 0],
                    'pressure': pressure_gauge
                })
        
        # Load aorta pressure from arterial/A1.txt instead
        aorta_file = self.results_path / 'A1.txt'
        if aorta_file.exists():
            data = np.loadtxt(aorta_file, delimiter=',')
            pressure_pa = (data[:, 1] + data[:, 2]) / 2.0
            pressure_gauge = (pressure_pa - P_ATMO) / PA_TO_MMHG
            heart_data['aorta'] = pd.DataFrame({
                'time': data[:, 0],
                'pressure': pressure_gauge
            })
        
        # Valve flow files
        valve_map = {
            'mitral_valve': 'L_la.txt',      # Mitral valve flow
            'aortic_valve': 'L_lv_aorta.txt' # Aortic valve flow
        }
        
        for valve, filename in valve_map.items():
            file_path = self.heart_path / filename
            if file_path.exists():
                data = np.loadtxt(file_path, delimiter=',')
                heart_data[valve] = pd.DataFrame({
                    'time': data[:, 0],
                    'flow': data[:, 1] * 60 * 1000  # Convert to mL/min
                })
        
        return heart_data if heart_data else None
    
    def extract_cardiac_cycle(self, time: np.ndarray, signal: np.ndarray, 
                            cycle_duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract last complete cardiac cycle from signal"""
        # Find last complete cycle
        max_time = time[-1]
        cycle_start = max_time - cycle_duration
        
        mask = time >= cycle_start
        return time[mask] - cycle_start, signal[mask]
    
    def calculate_periodicity(self, time: np.ndarray, signal: np.ndarray,
                            cycle_duration: float = 1.0) -> float:
        """Calculate RMS difference between last two cardiac cycles"""
        max_time = time[-1]
        
        # Extract last two cycles
        cycle2_mask = (time >= max_time - cycle_duration) & (time <= max_time)
        cycle1_mask = (time >= max_time - 2*cycle_duration) & (time < max_time - cycle_duration)
        
        cycle2 = signal[cycle2_mask]
        cycle1 = signal[cycle1_mask]
        
        # Ensure same length
        min_len = min(len(cycle1), len(cycle2))
        if min_len == 0:
            return np.nan
        
        cycle1 = cycle1[:min_len]
        cycle2 = cycle2[:min_len]
        
        # RMS difference
        rms_diff = np.sqrt(np.mean((cycle2 - cycle1)**2))
        rms_signal = np.sqrt(np.mean(cycle2**2))
        
        return (rms_diff / rms_signal * 100) if rms_signal > 0 else np.nan
    
    def plot_temporal_history(self):
        """Create temporal history plots for key arterial locations"""
        print("\n  → Generating temporal history plots...")
        
        # Key vessels to plot (from paper)
        vessels = {
            'A1': 'Ascending Aorta',
            'A12': 'R Internal Carotid',
            'A8': 'R Radial',
            'A48': 'R Femoral',
            'A53': 'R Anterior Tibial'
        }
        
        fig, axes = plt.subplots(len(vessels), 2, figsize=(14, 3*len(vessels)))
        fig.suptitle(f'Temporal History - {self.model_name}', fontsize=14, fontweight='bold')
        
        for idx, (vessel_id, vessel_name) in enumerate(vessels.items()):
            df = self.load_vessel_timeseries(vessel_id)
            if df is None:
                axes[idx, 0].text(0.5, 0.5, f'No data for {vessel_name}', 
                                ha='center', va='center')
                axes[idx, 1].text(0.5, 0.5, f'No data for {vessel_name}',
                                ha='center', va='center')
                continue
            
            # Get last cardiac cycle (assume 1 second cycle)
            t_cycle, p_cycle = self.extract_cardiac_cycle(df['time'].values, 
                                                         df['pressure'].values)
            _, v_cycle = self.extract_cardiac_cycle(df['time'].values,
                                                   df['velocity'].values)
            
            # Pressure plot
            axes[idx, 0].plot(t_cycle, p_cycle, 'b-', linewidth=1.5)
            axes[idx, 0].set_ylabel('Pressure (mmHg)', fontsize=10)
            axes[idx, 0].set_title(f'{vessel_name} - Pressure', fontsize=10)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Velocity plot
            axes[idx, 1].plot(t_cycle, v_cycle, 'r-', linewidth=1.5)
            axes[idx, 1].set_ylabel('Velocity (mm/s)', fontsize=10)
            axes[idx, 1].set_title(f'{vessel_name} - Velocity', fontsize=10)
            axes[idx, 1].grid(True, alpha=0.3)
            
            if idx == len(vessels) - 1:
                axes[idx, 0].set_xlabel('Time (s)', fontsize=10)
                axes[idx, 1].set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_path / 'temporal_history.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved: {output_file}")
        self.analysis_results['temporal_history'] = str(output_file)
    
    def plot_heart_dynamics(self):
        """Plot heart model dynamics (pressures and valve flows)"""
        print("\n  → Generating heart dynamics plots...")
        
        heart_data = self.load_heart_timeseries()
        if heart_data is None:
            print("     [WARNING] No heart data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Heart Model Dynamics - {self.model_name}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Chamber pressures
        if 'atrium' in heart_data:
            df = heart_data['atrium']
            t, p = self.extract_cardiac_cycle(df['time'].values, df['pressure'].values)
            ax1.plot(t, p, 'b-', linewidth=2, label='Atrium')
        
        if 'ventricle' in heart_data:
            df = heart_data['ventricle']
            t, p = self.extract_cardiac_cycle(df['time'].values, df['pressure'].values)
            ax1.plot(t, p, 'r-', linewidth=2, label='Ventricle')
        
        if 'aorta' in heart_data:
            df = heart_data['aorta']
            t, p = self.extract_cardiac_cycle(df['time'].values, df['pressure'].values)
            ax1.plot(t, p, 'g-', linewidth=2, label='Aorta')
        
        ax1.set_ylabel('Pressure (mmHg)', fontsize=11)
        ax1.set_title('Chamber Pressures', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Valve flows
        if 'mitral_valve' in heart_data:
            df = heart_data['mitral_valve']
            t, f = self.extract_cardiac_cycle(df['time'].values, df['flow'].values)
            ax2.plot(t, f, 'b-', linewidth=2, label='Mitral Valve')
        
        if 'aortic_valve' in heart_data:
            df = heart_data['aortic_valve']
            t, f = self.extract_cardiac_cycle(df['time'].values, df['flow'].values)
            ax2.plot(t, f, 'r-', linewidth=2, label='Aortic Valve')
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Flow (mL/min)', fontsize=11)
        ax2.set_title('Valve Flows', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_path / 'heart_dynamics.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved: {output_file}")
        self.analysis_results['heart_dynamics'] = str(output_file)
    
    def analyze_waveform_characteristics(self):
        """Analyze waveform amplitude and shape characteristics"""
        print("\n  → Analyzing waveform characteristics...")
        
        # Analyze ascending aorta (A1)
        df = self.load_vessel_timeseries('A1')
        if df is None:
            print("     [WARNING] No aorta data found")
            return
        
        # Extract last cycle
        t_cycle, p_cycle = self.extract_cardiac_cycle(df['time'].values,
                                                     df['pressure'].values)
        
        # Calculate characteristics
        systolic = np.max(p_cycle)
        diastolic = np.min(p_cycle)
        pulse_pressure = systolic - diastolic
        mean_pressure = np.mean(p_cycle)
        
        # Periodicity
        periodicity_rms = self.calculate_periodicity(df['time'].values,
                                                     df['pressure'].values)
        
        results = {
            'systolic_mmHg': float(systolic),
            'diastolic_mmHg': float(diastolic),
            'pulse_pressure_mmHg': float(pulse_pressure),
            'mean_pressure_mmHg': float(mean_pressure),
            'periodicity_rms_percent': float(periodicity_rms)
        }
        
        print(f"     Systolic:        {systolic:.1f} mmHg")
        print(f"     Diastolic:       {diastolic:.1f} mmHg")
        print(f"     Pulse Pressure:  {pulse_pressure:.1f} mmHg")
        print(f"     Mean Pressure:   {mean_pressure:.1f} mmHg")
        print(f"     Periodicity RMS: {periodicity_rms:.3f}%")
        
        self.analysis_results['waveform_characteristics'] = results
    
    def measure_runtime(self):
        """Extract runtime information if available"""
        print("\n  → Checking runtime data...")
        
        # Look for timing files (implementation-specific)
        # This is a placeholder - adjust based on actual output
        runtime_file = self.results_path.parent / 'runtime.txt'
        if runtime_file.exists():
            with open(runtime_file, 'r') as f:
                runtime_data = f.read()
            print(f"     Runtime data found: {runtime_file}")
            self.analysis_results['runtime_file'] = str(runtime_file)
        else:
            print("     [INFO] No runtime file found")
    
    def run_all(self):
        """Run all analysis steps"""
        print("="*70)
        print(f"ANALYSIS FOR: {self.model_name}")
        print("="*70)
        
        self.plot_temporal_history()
        self.plot_heart_dynamics()
        self.analyze_waveform_characteristics()
        self.measure_runtime()
        
        # Save summary
        summary_file = self.output_path / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"\n[OK] Analysis complete. Results saved to: {self.output_path}")


def main():
    """Run analysis workflow"""
    parser = argparse.ArgumentParser(
        description="Analyze FirstBlood simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 analysis.py                          # Interactive prompt
    python3 analysis.py --model patient_025
    python3 analysis.py --models patient_025,patient_026
    python3 analysis.py --all
        """
    )
    parser.add_argument('--model',
                       help='Single model name (e.g., patient_025, Abel_ref2)')
    parser.add_argument('--models',
                       help='Comma-separated model names')
    parser.add_argument('--models-file',
                       help='Path to text file with one model name per line')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all models found in results directory')
    parser.add_argument('--results-base', default=None,
                       help='Results base directory (default: projects/simple_run/results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: pipeline/output/analysis)')
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = get_repo_root()
    results_base = Path(args.results_base) if args.results_base else repo_root / "projects/simple_run/results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "pipeline/output/analysis"
    
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
            entered = input("Enter model name for analysis (e.g., patient_025): ").strip()
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
            print(f"ANALYSIS FOR: {model_name}")
            print("="*70)
            print(f"[ERROR] Results not found: {model_results}")
            print(f"\nPlease run simulation first:")
            print(f"  cd ~/first_blood/projects/simple_run")
            print(f"  ./simple_run.out {model_name}")
            overall_exit = 1
            continue
        
        # Run analysis
        analyzer = SimulationAnalyzer(model_name, results_base, output_dir)
        try:
            analyzer.run_all()
        except Exception as e:
            print(f"[ERROR] Analysis failed for {model_name}: {e}")
            overall_exit = 1
        
        print("\n")
    
    return overall_exit


if __name__ == '__main__':
    sys.exit(main())
