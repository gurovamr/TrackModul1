#!/usr/bin/env python3
"""
Comprehensive Waveform Analysis - Phase 2.1 & 2.2
Pressure and flow waveforms with hemodynamic indices
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

class WaveformAnalyzer:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.waveform_data = {}
        self.PA_TO_MMHG = 1.0 / 1333.22
        self.M3S_TO_MLMIN = 60 * 1000
    
    def load_vessel_data(self, vessel_id):
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        data = np.loadtxt(file_path, delimiter=',')
        return {
            'time': data[:, 0],
            'pressure': data[:, 1] * self.PA_TO_MMHG,
            'flow': data[:, 5] * self.M3S_TO_MLMIN
        }
    
    def extract_last_cycle(self, time, signal):
        peaks, _ = find_peaks(signal, height=np.max(signal)*0.7)
        if len(peaks) < 2:
            return time[-1000:], signal[-1000:]
        start_idx = peaks[-2]
        return time[start_idx:], signal[start_idx:]
    
    def analyze_pressure(self, vessel_id, name):
        data = self.load_vessel_data(vessel_id)
        if not data:
            return None
        t_cycle, p_cycle = self.extract_last_cycle(data['time'], data['pressure'])
        return {
            'vessel_id': vessel_id, 'vessel_name': name,
            'systolic': np.max(p_cycle), 'diastolic': np.min(p_cycle),
            'mean': np.mean(p_cycle), 'pulse': np.max(p_cycle) - np.min(p_cycle),
            'MAP': np.min(p_cycle) + (np.max(p_cycle) - np.min(p_cycle))/3,
            't_cycle': t_cycle, 'p_cycle': p_cycle
        }
    
    def analyze_flow(self, vessel_id, name):
        data = self.load_vessel_data(vessel_id)
        if not data:
            return None
        t_cycle, f_cycle = self.extract_last_cycle(data['time'], data['flow'])
        peak = np.max(f_cycle)
        mean_f = np.mean(f_cycle)
        end_dias = f_cycle[-1]
        PI = (peak - np.min(f_cycle)) / abs(mean_f) if abs(mean_f) > 1e-6 else 0
        RI = (peak - end_dias) / abs(peak) if abs(peak) > 1e-6 else 0
        return {
            'vessel_id': vessel_id, 'vessel_name': name,
            'mean_flow': mean_f, 'peak': peak, 'min': np.min(f_cycle),
            'PI': PI, 'RI': RI, 'reversal': np.any(f_cycle < 0),
            't_cycle': t_cycle, 'f_cycle': f_cycle
        }
    
    def analyze_all(self):
        vessels = {
            'Aorta': 'A1', 'R-ICA': 'A12', 'L-ICA': 'A16',
            'R-MCA': 'A70', 'L-MCA': 'A73', 'Basilar': 'A59',
            'R-PCA': 'A60', 'L-PCA': 'A61', 'R-ACA': 'A68', 'L-ACA': 'A69'
        }
        print(f"\n{'='*70}\nWAVEFORM ANALYSIS: {self.model_name}\n{'='*70}\n")
        p_results, f_results = [], []
        for name, vid in vessels.items():
            p = self.analyze_pressure(vid, name)
            f = self.analyze_flow(vid, name)
            if p:
                p_results.append(p)
                print(f"{name:15s}: Pressure {p['systolic']:.1f}/{p['diastolic']:.1f} mmHg")
            if f:
                f_results.append(f)
                print(f"{'':15s}  Flow {f['mean_flow']:.2f} mL/min, PI={f['PI']:.2f}, RI={f['RI']:.2f}")
        self.waveform_data = {'pressure': p_results, 'flow': f_results}
    
    def plot_waveforms(self, save_path):
        fig, axes = plt.subplots(6, 2, figsize=(14, 14))
        fig.suptitle(f'Waveforms: {self.model_name}', fontsize=16, fontweight='bold')
        vessels = ['Aorta', 'R-ICA', 'L-ICA', 'R-MCA', 'L-MCA', 'Basilar']
        for idx, vname in enumerate(vessels):
            p = next((x for x in self.waveform_data['pressure'] if x['vessel_name'] == vname), None)
            f = next((x for x in self.waveform_data['flow'] if x['vessel_name'] == vname), None)
            if p:
                axes[idx,0].plot(p['t_cycle'], p['p_cycle'], 'b-', lw=1.5)
                axes[idx,0].set_ylabel('Pressure (mmHg)', fontsize=9)
                axes[idx,0].set_title(f"{vname} - Pressure", fontsize=10, fontweight='bold')
                axes[idx,0].grid(True, alpha=0.3)
                axes[idx,0].text(0.02,0.98,f"Sys:{p['systolic']:.1f}\nDia:{p['diastolic']:.1f}",
                               transform=axes[idx,0].transAxes, fontsize=8, va='top',
                               bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
            if f:
                axes[idx,1].plot(f['t_cycle'], f['f_cycle'], 'r-', lw=1.5)
                axes[idx,1].set_ylabel('Flow (mL/min)', fontsize=9)
                axes[idx,1].set_title(f"{vname} - Flow", fontsize=10, fontweight='bold')
                axes[idx,1].grid(True, alpha=0.3)
                axes[idx,1].axhline(0, color='k', ls='-', alpha=0.3, lw=0.5)
                axes[idx,1].text(0.02,0.98,f"Mean:{f['mean_flow']:.2f}\nPI:{f['PI']:.2f}\nRI:{f['RI']:.2f}",
                               transform=axes[idx,1].transAxes, fontsize=8, va='top',
                               bbox=dict(boxstyle='round', fc='lightblue', alpha=0.5))
            if idx == 5:
                axes[idx,0].set_xlabel('Time (s)', fontsize=9)
                axes[idx,1].set_xlabel('Time (s)', fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Waveform plot saved: {save_path}")
    
    def plot_indices(self, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Hemodynamic Indices: {self.model_name}', fontsize=16, fontweight='bold')
        names = [x['vessel_name'] for x in self.waveform_data['flow']]
        pi_vals = [x['PI'] for x in self.waveform_data['flow']]
        ri_vals = [x['RI'] for x in self.waveform_data['flow']]
        axes[0].bar(range(len(names)), pi_vals, color='steelblue', alpha=0.7)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0].set_ylabel('Pulsatility Index', fontsize=11)
        axes[0].set_title('Pulsatility Index (PI)', fontsize=12)
        axes[0].axhline(1.2, color='orange', ls='--', alpha=0.5, label='High threshold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].legend(fontsize=9)
        axes[1].bar(range(len(names)), ri_vals, color='coral', alpha=0.7)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel('Resistive Index', fontsize=11)
        axes[1].set_title('Resistive Index (RI)', fontsize=12)
        axes[1].axhline(0.8, color='orange', ls='--', alpha=0.5, label='High threshold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Hemodynamic indices plot saved: {save_path}")
    
    def export_csv(self, output_dir):
        output_dir = Path(output_dir)
        # Pressure
        pd.DataFrame([{
            'Vessel': x['vessel_name'], 'Systolic': x['systolic'],
            'Diastolic': x['diastolic'], 'Mean': x['mean'],
            'Pulse': x['pulse'], 'MAP': x['MAP']
        } for x in self.waveform_data['pressure']]).to_csv(
            output_dir / f"pressure_{self.model_name}.csv", index=False)
        # Flow
        pd.DataFrame([{
            'Vessel': x['vessel_name'], 'Mean_Flow': x['mean_flow'],
            'Peak': x['peak'], 'PI': x['PI'], 'RI': x['RI'],
            'Reversal': x['reversal']
        } for x in self.waveform_data['flow']]).to_csv(
            output_dir / f"flow_{self.model_name}.csv", index=False)
        print(f"✓ CSV files exported to {output_dir}")

def compare_models(ref, pat, output_dir):
    print(f"\n{'='*70}\nCOMPARATIVE ANALYSIS\n{'='*70}\n")
    comps = []
    for pf in pat.waveform_data['flow']:
        rf = next((x for x in ref.waveform_data['flow'] if x['vessel_name']==pf['vessel_name']), None)
        if rf:
            change = (pf['mean_flow'] - rf['mean_flow']) / abs(rf['mean_flow']) * 100 if abs(rf['mean_flow']) > 1e-6 else 0
            comps.append({
                'Vessel': pf['vessel_name'], 'Ref_Flow': rf['mean_flow'],
                'Pat_Flow': pf['mean_flow'], 'Change_%': change,
                'Ref_PI': rf['PI'], 'Pat_PI': pf['PI']
            })
    df = pd.DataFrame(comps)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    df.to_csv(Path(output_dir) / "comparison_ref_vs_patient.csv", index=False)
    print(f"\n✓ Comparison saved")

def main():
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/waveform_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "#"*70 + "\n# REFERENCE MODEL\n" + "#"*70)
    ref = WaveformAnalyzer('Abel_ref2', results_base)
    ref.analyze_all()
    ref.plot_waveforms(output_dir / "waveforms_Abel_ref2.png")
    ref.plot_indices(output_dir / "indices_Abel_ref2.png")
    ref.export_csv(output_dir)
    
    print("\n" + "#"*70 + "\n# PATIENT MODEL\n" + "#"*70)
    pat = WaveformAnalyzer('patient025_CoW_v2', results_base)
    pat.analyze_all()
    pat.plot_waveforms(output_dir / "waveforms_patient025.png")
    pat.plot_indices(output_dir / "indices_patient025.png")
    pat.export_csv(output_dir)
    
    compare_models(ref, pat, output_dir)
    print(f"\n{'='*70}\nANALYSIS COMPLETE\n{'='*70}\n")

if __name__ == "__main__":
    main()