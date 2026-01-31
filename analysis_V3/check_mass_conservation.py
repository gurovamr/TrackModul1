#!/usr/bin/env python3
"""
Mass Conservation Check for FirstBlood Simulation
Verifies flow continuity at bifurcations in Circle of Willis

Checks:
1. Flow continuity at each bifurcation (inflow = outflow)
2. Total CoW balance (3 inflows vs 6 outflows)
3. Numerical leakage identification
4. Temporal consistency (conservation over cardiac cycle)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class MassConservationChecker:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.conservation_results = []
        
    def load_vessel_data(self, vessel_id):
        """Load arterial data for a vessel"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        
        data = np.loadtxt(file_path, delimiter=',')
        return {
            'time': data[:, 0],
            'flow_start': data[:, 5],  # m³/s
            'flow_end': data[:, 6]      # m³/s
        }
    
    def check_bifurcation(self, parent_id, child_ids, bif_name, at_start=True):
        """
        Check mass conservation at a bifurcation
        
        Parameters:
        - parent_id: vessel ID of parent (inflow)
        - child_ids: list of vessel IDs for children (outflow)
        - bif_name: name of bifurcation
        - at_start: True if checking at parent's start, False for end
        """
        # Load parent flow
        parent_data = self.load_vessel_data(parent_id)
        if parent_data is None:
            return None
        
        parent_flow = parent_data['flow_start'] if at_start else parent_data['flow_end']
        
        # Load children flows
        child_flows = []
        child_names = []
        for child_id in child_ids:
            child_data = self.load_vessel_data(child_id)
            if child_data is not None:
                # At bifurcation, we check the START of child vessels
                child_flows.append(child_data['flow_start'])
                child_names.append(child_id)
        
        if len(child_flows) == 0:
            return None
        
        # Calculate conservation over time
        time = parent_data['time']
        
        # Sum child flows (handle different lengths)
        min_len = min(len(parent_flow), min(len(cf) for cf in child_flows))
        parent_flow = parent_flow[:min_len]
        child_flows = [cf[:min_len] for cf in child_flows]
        total_child_flow = np.sum(child_flows, axis=0)
        
        # Calculate error
        flow_error = parent_flow - total_child_flow
        abs_error = np.abs(flow_error)
        rel_error = abs_error / (np.abs(parent_flow) + 1e-12) * 100  # Percent
        
        # Statistics
        mean_parent = np.mean(parent_flow) * 60 * 1000  # mL/min
        mean_children = np.mean(total_child_flow) * 60 * 1000
        mean_error = np.mean(abs_error) * 60 * 1000
        max_error = np.max(abs_error) * 60 * 1000
        mean_rel_error = np.mean(rel_error)
        max_rel_error = np.max(rel_error)
        
        result = {
            'bifurcation': bif_name,
            'parent_id': parent_id,
            'children_ids': ', '.join(child_names),
            'n_children': len(child_names),
            'parent_flow_ml_min': mean_parent,
            'children_flow_ml_min': mean_children,
            'abs_error_ml_min': mean_error,
            'max_error_ml_min': max_error,
            'rel_error_pct': mean_rel_error,
            'max_rel_error_pct': max_rel_error,
            'time': time,
            'parent_flow': parent_flow,
            'child_flow': total_child_flow,
            'error': flow_error
        }
        
        return result
    
    def check_all_bifurcations(self):
        """Check all major bifurcations in Circle of Willis"""
        print(f"\n{'='*70}")
        print(f"MASS CONSERVATION AT BIFURCATIONS: {self.model_name}")
        print(f"{'='*70}\n")
        
        # Define Circle of Willis bifurcations based on Abel_ref2 topology
        bifurcations = [
            # Format: (parent_id, [child_ids], name, check_at_start)
            
            # 1. Basilar bifurcation → R-PCA + L-PCA
            ('A59', ['A60', 'A61'], 'Basilar → R-PCA + L-PCA', False),
            
            # 2. R-ICA → R-MCA + R-ACA + R-PCoA (complex)
            # In Abel_ref2: A12 (R-ICA) connects to node n46
            # From n46: need to check what connects
            
            # 3. L-ICA → L-MCA + L-ACA + L-PCoA
            # Similar structure on left
            
            # Simplified checks for major vessels:
            # R-ICA terminal (approximately)
            ('A12', ['A70'], 'R-ICA → R-MCA (simplified)', False),
            
            # L-ICA terminal
            ('A16', ['A73'], 'L-ICA → L-MCA (simplified)', False),
            
            # Note: Full CoW topology is complex with communicating arteries
            # These are the main bifurcations we can check directly
        ]
        
        results = []
        
        for parent_id, child_ids, bif_name, at_start in bifurcations:
            result = self.check_bifurcation(parent_id, child_ids, bif_name, at_start)
            
            if result is not None:
                results.append(result)
                
                # Print result
                status = "✓" if result['rel_error_pct'] < 5.0 else "⚠️" if result['rel_error_pct'] < 10.0 else "✗"
                
                print(f"{status} {bif_name}")
                print(f"   Parent ({parent_id}):  {result['parent_flow_ml_min']:7.2f} mL/min")
                print(f"   Children ({result['children_ids']}): {result['children_flow_ml_min']:7.2f} mL/min")
                print(f"   Error:      {result['abs_error_ml_min']:7.2f} mL/min ({result['rel_error_pct']:.2f}%)")
                print(f"   Max error:  {result['max_error_ml_min']:7.2f} mL/min ({result['max_rel_error_pct']:.2f}%)")
                print()
        
        self.conservation_results = results
        return results
    
    def check_total_cow_balance(self):
        """Check total Circle of Willis mass balance"""
        print(f"\n{'='*70}")
        print(f"TOTAL CIRCLE OF WILLIS MASS BALANCE")
        print(f"{'='*70}\n")
        
        # Inflow vessels
        inflow_vessels = {
            'R-ICA': 'A12',
            'L-ICA': 'A16',
            'Basilar': 'A59'
        }
        
        # Outflow vessels (terminal branches)
        outflow_vessels = {
            'R-MCA': 'A70',
            'L-MCA': 'A73',
            'R-ACA-A2': 'A76',
            'L-ACA-A2': 'A78',
            'R-PCA-P2': 'A64',
            'L-PCA-P2': 'A65'
        }
        
        # Calculate total inflow
        print("INFLOW (to CoW):")
        total_inflow = 0
        inflow_data = {}
        
        for name, vessel_id in inflow_vessels.items():
            data = self.load_vessel_data(vessel_id)
            if data is not None:
                # Use flow at END of vessel (entering CoW)
                mean_flow = np.mean(data['flow_end']) * 60 * 1000  # mL/min
                total_inflow += mean_flow
                inflow_data[name] = {
                    'flow': mean_flow,
                    'time_series': data['flow_end']
                }
                print(f"  {name:15s} ({vessel_id}): {mean_flow:7.2f} mL/min")
        
        print(f"  {'TOTAL INFLOW':15s}        : {total_inflow:7.2f} mL/min")
        
        # Calculate total outflow
        print(f"\nOUTFLOW (from CoW):")
        total_outflow = 0
        outflow_data = {}
        
        for name, vessel_id in outflow_vessels.items():
            data = self.load_vessel_data(vessel_id)
            if data is not None:
                # Use flow at START of vessel (leaving CoW)
                mean_flow = np.mean(data['flow_start']) * 60 * 1000  # mL/min
                total_outflow += mean_flow
                outflow_data[name] = {
                    'flow': mean_flow,
                    'time_series': data['flow_start']
                }
                print(f"  {name:15s} ({vessel_id}): {mean_flow:7.2f} mL/min")
        
        print(f"  {'TOTAL OUTFLOW':15s}        : {total_outflow:7.2f} mL/min")
        
        # Calculate imbalance
        imbalance = total_inflow - total_outflow
        imbalance_pct = abs(imbalance) / total_inflow * 100
        
        print(f"\n{'='*70}")
        print(f"BALANCE:")
        print(f"  Inflow:     {total_inflow:7.2f} mL/min")
        print(f"  Outflow:    {total_outflow:7.2f} mL/min")
        print(f"  Imbalance:  {imbalance:+7.2f} mL/min ({imbalance_pct:.2f}%)")
        
        # Interpret results
        if imbalance_pct < 5:
            print(f"  Status: ✓ EXCELLENT mass conservation")
        elif imbalance_pct < 15:
            print(f"  Status: ✓ ACCEPTABLE (peripheral leakage expected)")
        elif imbalance_pct < 30:
            print(f"  Status: ⚠️  MODERATE imbalance (check peripherals)")
        else:
            print(f"  Status: ✗ LARGE imbalance (check model)")
        
        print(f"{'='*70}")
        
        # Check where the leakage is
        self.identify_leakage_location(inflow_data, outflow_data)
        
        return {
            'total_inflow': total_inflow,
            'total_outflow': total_outflow,
            'imbalance': imbalance,
            'imbalance_pct': imbalance_pct
        }
    
    def identify_leakage_location(self, inflow_data, outflow_data):
        """Identify where numerical leakage occurs"""
        print(f"\n{'='*70}")
        print(f"LEAKAGE ANALYSIS")
        print(f"{'='*70}\n")
        
        # Check anterior vs posterior balance
        anterior_in = inflow_data.get('R-ICA', {}).get('flow', 0) + \
                     inflow_data.get('L-ICA', {}).get('flow', 0)
        
        posterior_in = inflow_data.get('Basilar', {}).get('flow', 0)
        
        print(f"Anterior circulation inflow: {anterior_in:.2f} mL/min")
        print(f"Posterior circulation inflow: {posterior_in:.2f} mL/min")
        print(f"Ratio (Ant/Post): {anterior_in/posterior_in:.2f}")
        
        # Expected distribution
        print(f"\nExpected distribution:")
        print(f"  Anterior: ~70-80% of total")
        print(f"  Posterior: ~20-30% of total")
        
        actual_ant_pct = anterior_in / (anterior_in + posterior_in) * 100
        actual_post_pct = posterior_in / (anterior_in + posterior_in) * 100
        
        print(f"\nActual distribution:")
        print(f"  Anterior: {actual_ant_pct:.1f}%")
        print(f"  Posterior: {actual_post_pct:.1f}%")
        
        if 70 <= actual_ant_pct <= 80:
            print(f"  ✓ Distribution is physiological")
        else:
            print(f"  ⚠️  Distribution differs from typical")
    
    def check_temporal_conservation(self):
        """Check if mass is conserved throughout the cardiac cycle"""
        print(f"\n{'='*70}")
        print(f"TEMPORAL MASS CONSERVATION")
        print(f"{'='*70}\n")
        
        # Load CoW inflow vessels
        inflow_ids = ['A12', 'A16', 'A59']
        outflow_ids = ['A70', 'A73', 'A76', 'A78', 'A64', 'A65']
        
        # Get time series
        inflow_series = []
        for vid in inflow_ids:
            data = self.load_vessel_data(vid)
            if data is not None:
                inflow_series.append(data['flow_end'])
        
        outflow_series = []
        for vid in outflow_ids:
            data = self.load_vessel_data(vid)
            if data is not None:
                outflow_series.append(data['flow_start'])
        
        if len(inflow_series) == 0 or len(outflow_series) == 0:
            print("⚠️  Insufficient data for temporal analysis")
            return
        
        # Sum over vessels
        total_inflow_t = np.sum(inflow_series, axis=0)
        total_outflow_t = np.sum(outflow_series, axis=0)
        
        # Calculate error over time
        error_t = total_inflow_t - total_outflow_t
        rel_error_t = np.abs(error_t) / (np.abs(total_inflow_t) + 1e-12) * 100
        
        mean_rel_error = np.mean(rel_error_t)
        max_rel_error = np.max(rel_error_t)
        min_rel_error = np.min(rel_error_t)
        
        print(f"Conservation over cardiac cycle:")
        print(f"  Mean error:    {mean_rel_error:.2f}%")
        print(f"  Max error:     {max_rel_error:.2f}%")
        print(f"  Min error:     {min_rel_error:.2f}%")
        
        # Check if error is constant or varies
        error_std = np.std(rel_error_t)
        print(f"  Error std dev: {error_std:.2f}%")
        
        if error_std < 2.0:
            print(f"  ✓ Error is constant (systematic peripheral leakage)")
        else:
            print(f"  ⚠️  Error varies (possible numerical issues)")
        
        return {
            'time': data['time'],
            'error_series': rel_error_t,
            'inflow_series': total_inflow_t,
            'outflow_series': total_outflow_t
        }
    
    def generate_conservation_plots(self, save_path=None):
        """Generate mass conservation visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Mass Conservation Analysis: {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Bifurcation errors
        ax1 = axes[0, 0]
        if self.conservation_results:
            bif_names = [r['bifurcation'].split('→')[0].strip() for r in self.conservation_results]
            rel_errors = [r['rel_error_pct'] for r in self.conservation_results]
            
            bars = ax1.bar(range(len(bif_names)), rel_errors)
            ax1.set_xticks(range(len(bif_names)))
            ax1.set_xticklabels(bif_names, rotation=45, ha='right')
            ax1.set_ylabel('Relative Error (%)', fontsize=11)
            ax1.set_title('Mass Conservation at Bifurcations', fontsize=12)
            ax1.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Excellent (<5%)')
            ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Good (<10%)')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Color bars
            for i, bar in enumerate(bars):
                if rel_errors[i] < 5:
                    bar.set_color('green')
                elif rel_errors[i] < 10:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # Plot 2: Inflow vs Outflow balance
        ax2 = axes[0, 1]
        inflow_vessels = ['R-ICA', 'L-ICA', 'Basilar']
        outflow_vessels = ['R-MCA', 'L-MCA', 'R-ACA', 'L-ACA', 'R-PCA', 'L-PCA']
        
        # Load actual flows
        inflows = []
        for vid in ['A12', 'A16', 'A59']:
            data = self.load_vessel_data(vid)
            if data:
                inflows.append(np.mean(data['flow_end']) * 60 * 1000)
        
        outflows = []
        for vid in ['A70', 'A73', 'A76', 'A78', 'A64', 'A65']:
            data = self.load_vessel_data(vid)
            if data:
                outflows.append(np.mean(data['flow_start']) * 60 * 1000)
        
        x_in = range(len(inflows))
        x_out = range(len(outflows))
        
        ax2.bar([i - 0.2 for i in x_in], inflows, width=0.4, label='Inflow', alpha=0.7, color='blue')
        ax2.bar([i + 0.2 for i in x_out], outflows, width=0.4, label='Outflow', alpha=0.7, color='red')
        
        ax2.set_ylabel('Flow (mL/min)', fontsize=11)
        ax2.set_title('CoW Inflow vs Outflow Distribution', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Temporal error (if available)
        ax3 = axes[1, 0]
        temporal_data = self.check_temporal_conservation()
        if temporal_data:
            time = temporal_data['time']
            error = temporal_data['error_series']
            
            # Plot last 2 cardiac cycles
            last_2s = time >= (time[-1] - 2.0)
            ax3.plot(time[last_2s], error[last_2s], 'b-', linewidth=1.5)
            ax3.set_xlabel('Time (s)', fontsize=11)
            ax3.set_ylabel('Conservation Error (%)', fontsize=11)
            ax3.set_title('Temporal Mass Conservation', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=5, color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary
        balance = self.check_total_cow_balance()
        
        summary_text = f"MASS CONSERVATION SUMMARY\n{'='*35}\n\n"
        summary_text += f"Total CoW Balance:\n"
        summary_text += f"  Inflow:  {balance['total_inflow']:.1f} mL/min\n"
        summary_text += f"  Outflow: {balance['total_outflow']:.1f} mL/min\n"
        summary_text += f"  Error:   {balance['imbalance_pct']:.1f}%\n\n"
        
        if balance['imbalance_pct'] < 5:
            summary_text += "✓ Excellent conservation\n"
        elif balance['imbalance_pct'] < 15:
            summary_text += "✓ Acceptable (peripheral leakage)\n"
        else:
            summary_text += "⚠️ Check peripheral resistances\n"
        
        summary_text += f"\nBifurcation Checks:\n"
        if self.conservation_results:
            n_good = sum(1 for r in self.conservation_results if r['rel_error_pct'] < 5)
            n_total = len(self.conservation_results)
            summary_text += f"  {n_good}/{n_total} bifurcations < 5% error\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Conservation plot saved: {save_path}")
        
        return fig
    
    def run_all_checks(self):
        """Run all mass conservation checks"""
        print(f"\n{'#'*70}")
        print(f"# MASS CONSERVATION ANALYSIS")
        print(f"# Model: {self.model_name}")
        print(f"{'#'*70}")
        
        self.check_all_bifurcations()
        balance = self.check_total_cow_balance()
        self.check_temporal_conservation()
        
        return balance


def main():
    """Main function"""
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/conservation_results"
    output_dir.mkdir(exist_ok=True)
    
    models = ['Abel_ref2', 'patient025_CoW_v2']
    
    for model_name in models:
        print(f"\n\n{'#'*70}")
        print(f"# ANALYZING: {model_name}")
        print(f"{'#'*70}\n")
        
        checker = MassConservationChecker(model_name, results_base)
        checker.run_all_checks()
        
        # Generate plots
        plot_path = output_dir / f"conservation_{model_name}.png"
        checker.generate_conservation_plots(save_path=plot_path)
    
    print(f"\n\n{'='*70}")
    print(f"MASS CONSERVATION ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()