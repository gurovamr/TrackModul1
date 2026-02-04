#!/usr/bin/env python3
"""
Visualization Script for FirstBlood Simulations
================================================
Creates publication-quality figures aligned with FirstBlood paper.

Figures generated (based on paper):
1. Temporal History Plots - Pressure & Velocity waveforms for key vessels
2. Heart Model Dynamics - Chamber pressures and valve flows
3. Grid Convergence Plots - Relative error vs division points (requires grid study)
4. Grid Sensitivity - Waveform overlay for coarse/normal/fine (requires grid study)
5. Runtime Analysis - Simulation time vs grid size (requires timing data)

Usage:
    python3 visualization.py                        # Interactive prompt
    python3 visualization.py --model patient_025
    python3 visualization.py --models patient_025,Abel_ref2
    python3 visualization.py --all

Output:
    Figures saved to pipeline/output/visualization/<model_name>/
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Tuple

# Publication-quality plot settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['grid.alpha'] = 0.3

# Constants
PA_TO_MMHG = 133.322
P_ATMO = 1.0e5
MS_TO_MMS = 1000.0


def get_repo_root():
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent


class SimulationVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, model_name: str, results_base: Path, output_base: Path):
        self.model_name = model_name
        self.results_path = results_base / model_name / "arterial"
        self.heart_path = results_base / model_name / "heart_kim_lit"
        self.output_path = output_base / model_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results not found: {self.results_path}")
    
    def load_vessel_timeseries(self, vessel_id: str) -> Optional[pd.DataFrame]:
        """Load vessel time-series data with gauge pressure"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        
        data = np.loadtxt(file_path, delimiter=',')
        pressure_pa = (data[:, 1] + data[:, 2]) / 2.0
        pressure_gauge = (pressure_pa - P_ATMO) / PA_TO_MMHG
        
        df = pd.DataFrame({
            'time': data[:, 0],
            'pressure': pressure_gauge,
            'velocity': data[:, 3] * MS_TO_MMS,
            'flow': (data[:, 5] + data[:, 6]) / 2.0 * 1e6 * 60  # mL/min
        })
        return df
    
    def load_heart_data(self) -> Dict[str, pd.DataFrame]:
        """Load heart model data"""
        heart_data = {}
        
        # Chamber pressures
        chamber_map = {
            'Atrium': 'E_la.txt',
            'Ventricle': 'E_lv.txt'
        }
        
        for name, filename in chamber_map.items():
            file_path = self.heart_path / filename
            if file_path.exists():
                data = np.loadtxt(file_path, delimiter=',')
                pressure_gauge = (data[:, 1] - P_ATMO) / PA_TO_MMHG
                heart_data[name] = pd.DataFrame({
                    'time': data[:, 0],
                    'pressure': pressure_gauge
                })
        
        # Aorta from arterial results
        aorta_df = self.load_vessel_timeseries('A1')
        if aorta_df is not None:
            heart_data['Aorta'] = pd.DataFrame({
                'time': aorta_df['time'],
                'pressure': aorta_df['pressure']
            })
        
        # Valve flows
        valve_map = {
            'Mitral': 'L_la.txt',
            'Aortic': 'L_lv_aorta.txt'
        }
        
        for name, filename in valve_map.items():
            file_path = self.heart_path / filename
            if file_path.exists():
                data = np.loadtxt(file_path, delimiter=',')
                heart_data[name] = pd.DataFrame({
                    'time': data[:, 0],
                    'flow': data[:, 1] * 60 * 1000  # mL/min
                })
        
        return heart_data
    
    def extract_last_cycle(self, df: pd.DataFrame, 
                          cycle_duration: float = 1.0) -> pd.DataFrame:
        """Extract last complete cardiac cycle"""
        max_time = df['time'].max()
        cycle_start = max_time - cycle_duration
        return df[df['time'] >= cycle_start].copy()
    
    def figure1_temporal_history(self):
        """
        Figure 1: Temporal History - Pressure & Velocity Waveforms
        Shows one cardiac cycle for key arterial locations from paper.
        """
        print("\n  → Creating Figure 1: Temporal History...")
        
        vessels = {
            'A1': 'Ascending Aorta',
            'A12': 'R Internal Carotid',
            'A8': 'R Radial',
            'A48': 'R Femoral',
            'A53': 'R Anterior Tibial'
        }
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
        
        for idx, (vessel_id, vessel_name) in enumerate(vessels.items()):
            df = self.load_vessel_timeseries(vessel_id)
            if df is None:
                continue
            
            df_cycle = self.extract_last_cycle(df)
            t = df_cycle['time'].values - df_cycle['time'].values[0]
            
            # Pressure subplot
            ax_p = fig.add_subplot(gs[idx, 0])
            ax_p.plot(t, df_cycle['pressure'].values, 'b-', linewidth=1.8)
            ax_p.set_ylabel('Pressure (mmHg)', fontsize=10, fontweight='bold')
            ax_p.set_title(vessel_name, fontsize=11, fontweight='bold')
            ax_p.grid(True, alpha=0.3)
            if idx == len(vessels) - 1:
                ax_p.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            
            # Velocity subplot
            ax_v = fig.add_subplot(gs[idx, 1])
            ax_v.plot(t, df_cycle['velocity'].values, 'r-', linewidth=1.8)
            ax_v.set_ylabel('Velocity (mm/s)', fontsize=10, fontweight='bold')
            ax_v.set_title(vessel_name, fontsize=11, fontweight='bold')
            ax_v.grid(True, alpha=0.3)
            if idx == len(vessels) - 1:
                ax_v.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Temporal History - {self.model_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        output_file = self.output_path / 'figure1_temporal_history.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved: {output_file.name}")
        return str(output_file)
    
    def figure2_heart_dynamics(self):
        """
        Figure 2: Heart Model Dynamics
        Top: Chamber pressures (atrium, ventricle, aorta)
        Bottom: Valve flows (mitral, aortic) with open/close markers
        """
        print("\n  → Creating Figure 2: Heart Dynamics...")
        
        heart_data = self.load_heart_data()
        if not heart_data:
            print("     [WARNING] No heart data available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
        
        # Extract last cycle for all data
        cycle_data = {}
        for key, df in heart_data.items():
            cycle_df = self.extract_last_cycle(df)
            cycle_df['time'] = cycle_df['time'] - cycle_df['time'].iloc[0]
            cycle_data[key] = cycle_df
        
        # Top panel: Chamber pressures
        colors = {'Atrium': '#2E86AB', 'Ventricle': '#A23B72', 'Aorta': '#F18F01'}
        for chamber in ['Atrium', 'Ventricle', 'Aorta']:
            if chamber in cycle_data:
                df = cycle_data[chamber]
                ax1.plot(df['time'], df['pressure'], 
                        color=colors[chamber], linewidth=2.5, label=chamber)
        
        ax1.set_ylabel('Pressure (mmHg)', fontsize=12, fontweight='bold')
        ax1.set_title('Chamber Pressures', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 1.0)
        
        # Bottom panel: Valve flows
        colors_valve = {'Mitral': '#2E86AB', 'Aortic': '#A23B72'}
        for valve in ['Mitral', 'Aortic']:
            if valve in cycle_data:
                df = cycle_data[valve]
                ax2.plot(df['time'], df['flow'], 
                        color=colors_valve[valve], linewidth=2.5, label=f'{valve} Valve')
                
                # Mark valve opening/closing (flow threshold)
                flow = df['flow'].values
                time = df['time'].values
                
                # Find transitions
                open_close = np.diff((flow > 1.0).astype(int))
                opens = time[1:][open_close > 0]
                closes = time[1:][open_close < 0]
                
                for t_open in opens:
                    ax2.axvline(t_open, color=colors_valve[valve], 
                              linestyle='--', alpha=0.5, linewidth=1.5)
                for t_close in closes:
                    ax2.axvline(t_close, color=colors_valve[valve], 
                              linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax2.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Flow (mL/min)', fontsize=12, fontweight='bold')
        ax2.set_title('Valve Flows', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 1.0)
        
        fig.suptitle(f'Heart Model Dynamics - {self.model_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_path / 'figure2_heart_dynamics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved: {output_file.name}")
        return str(output_file)
    
    def figure3_grid_convergence(self, grid_models: Optional[List[str]] = None):
        """
        Figure 3: Grid Convergence Analysis
        Requires coarse/normal/fine grid runs.
        Plots relative error vs number of division points.
        """
        if grid_models is None:
            print("\n  → Skipping Figure 3: Grid convergence (requires grid study)")
            print("     To generate: run simulations with different grid densities")
            print("     Example: patient_025_coarse, patient_025, patient_025_fine")
            return None
        
        print("\n  → Creating Figure 3: Grid Convergence...")
        print("     [NOT IMPLEMENTED] Requires grid sensitivity data")
        return None
    
    def figure4_grid_sensitivity(self, grid_models: Optional[List[str]] = None):
        """
        Figure 4: Grid Sensitivity - Waveform Overlay
        Shows coarse/normal/fine pressure waveforms overlaid.
        """
        if grid_models is None:
            print("\n  → Skipping Figure 4: Grid sensitivity (requires grid study)")
            return None
        
        print("\n  → Creating Figure 4: Grid Sensitivity...")
        print("     [NOT IMPLEMENTED] Requires grid sensitivity data")
        return None
    
    def figure5_runtime_analysis(self):
        """
        Figure 5: Runtime vs Grid Size
        Requires timing data from multiple grid densities.
        """
        print("\n  → Skipping Figure 5: Runtime analysis (requires timing data)")
        print("     To generate: save runtime.txt in each model's results folder")
        return None
    
    def generate_all(self):
        """Generate all available figures"""
        print("="*70)
        print(f"VISUALIZATION FOR: {self.model_name}")
        print("="*70)
        
        figures = {}
        
        # Generate available figures
        figures['figure1'] = self.figure1_temporal_history()
        figures['figure2'] = self.figure2_heart_dynamics()
        figures['figure3'] = self.figure3_grid_convergence()
        figures['figure4'] = self.figure4_grid_sensitivity()
        figures['figure5'] = self.figure5_runtime_analysis()
        
        # Save summary
        generated = [k for k, v in figures.items() if v is not None]
        summary = {
            'model': self.model_name,
            'output_directory': str(self.output_path),
            'figures_generated': generated,
            'files': {k: v for k, v in figures.items() if v is not None}
        }
        
        import json
        summary_file = self.output_path / 'visualization_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Generated {len(generated)} figures")
        print(f"     Output: {self.output_path}")
        print(f"     Summary: {summary_file.name}")


def main():
    """Run visualization workflow"""
    parser = argparse.ArgumentParser(
        description="Create publication-quality figures for FirstBlood",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 visualization.py                        # Interactive prompt
    python3 visualization.py --model patient_025
    python3 visualization.py --models patient_025,Abel_ref2
    python3 visualization.py --all

Output:
    Figures saved to pipeline/output/visualization/<model_name>/
        """
    )
    parser.add_argument('--model',
                       help='Single model name (e.g., patient_025, Abel_ref2)')
    parser.add_argument('--models',
                       help='Comma-separated model names')
    parser.add_argument('--models-file',
                       help='Path to text file with one model name per line')
    parser.add_argument('--all', action='store_true',
                       help='Visualize all models found in results directory')
    parser.add_argument('--results-base', default=None,
                       help='Results base directory (default: projects/simple_run/results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: pipeline/output/visualization)')
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = get_repo_root()
    results_base = Path(args.results_base) if args.results_base else repo_root / "projects/simple_run/results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "pipeline/output/visualization"
    
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
            entered = input("Enter model name for visualization (e.g., patient_025): ").strip()
        except EOFError:
            entered = ""
        if entered:
            model_names.append(entered)
        else:
            print("[ERROR] No models specified. Use --model, --models, --models-file, or --all.")
            sys.exit(1)
    
    overall_exit = 0
    
    for model_name in model_names:
        try:
            visualizer = SimulationVisualizer(model_name, results_base, output_dir)
            visualizer.generate_all()
        except FileNotFoundError as e:
            print("="*70)
            print(f"VISUALIZATION FOR: {model_name}")
            print("="*70)
            print(f"[ERROR] {e}")
            print(f"\nPlease run simulation first:")
            print(f"  cd ~/first_blood/projects/simple_run")
            print(f"  ./simple_run.out {model_name}")
            overall_exit = 1
        except Exception as e:
            print(f"[ERROR] Visualization failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            overall_exit = 1
        
        print("\n")
    
    return overall_exit


if __name__ == '__main__':
    sys.exit(main())
