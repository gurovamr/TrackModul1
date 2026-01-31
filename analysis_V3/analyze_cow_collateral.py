#!/usr/bin/env python3
"""
Circle of Willis Collateral Flow Analysis - Phase 3
Analyzes inflow, outflow, collateral vessels, and L/R asymmetry
PATIENT-SPECIFIC ANALYSIS ONLY (patient025_CoW_v2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class CoWAnalyzer:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.M3S_TO_MLMIN = 60 * 1000
        self.M3S_TO_LMIN = 60
        
    def load_mean_flow(self, vessel_id):
        """Load mean flow for a vessel in mL/min"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return None
        data = np.loadtxt(file_path, delimiter=',')
        flow = data[:, 5]  # Start flow in m³/s
        return np.mean(flow) * self.M3S_TO_MLMIN
    
    def analyze_inflow(self):
        """3.1 INFLOW ANALYSIS"""
        print(f"\n{'='*70}")
        print(f"3.1 INFLOW ANALYSIS: {self.model_name}")
        print(f"{'='*70}\n")
        
        inflow_vessels = {
            'Right Internal Carotid (R-ICA)': 'A12',
            'Left Internal Carotid (L-ICA)': 'A16',
            'Basilar Artery': 'A59'
        }
        
        total_inflow = 0
        inflow_data = {}
        
        print("Sources of blood flow to Circle of Willis:\n")
        
        for name, vessel_id in inflow_vessels.items():
            flow = self.load_mean_flow(vessel_id)
            if flow is not None:
                total_inflow += flow
                inflow_data[name] = flow
                print(f"  {name:35s}: {flow:7.2f} mL/min")
        
        print(f"\n  {'TOTAL INFLOW':35s}: {total_inflow:7.2f} mL/min")
        
        # Calculate percentages
        print(f"\n{'='*70}")
        print("Contribution from each source:")
        print(f"{'='*70}\n")
        
        for name, flow in inflow_data.items():
            pct = (flow / total_inflow) * 100
            bar = '█' * int(pct / 2)
            print(f"  {name:35s}: {pct:5.1f}% {bar}")
        
        # Clinical interpretation
        print(f"\n{'='*70}")
        print("CLINICAL INTERPRETATION:")
        print(f"{'='*70}\n")
        
        ant_flow = inflow_data.get('Right Internal Carotid (R-ICA)', 0) + \
                   inflow_data.get('Left Internal Carotid (L-ICA)', 0)
        post_flow = inflow_data.get('Basilar Artery', 0)
        
        ant_pct = (ant_flow / total_inflow) * 100
        post_pct = (post_flow / total_inflow) * 100
        
        print(f"  Anterior circulation (ICAs): {ant_pct:.1f}%")
        print(f"  Posterior circulation (BA):  {post_pct:.1f}%")
        print()
        print(f"  Expected normal: 70-80% anterior, 20-30% posterior")
        
        if post_pct < 10:
            print(f"  ⚠️  SEVERE posterior circulation reduction!")
            print(f"      → Suggests fetal posterior variant")
            print(f"      → PCAs likely supplied by ICAs via PCoAs")
        elif post_pct < 20:
            print(f"  ⚠️  Reduced posterior circulation")
        else:
            print(f"  ✓ Normal anterior/posterior distribution")
        
        return inflow_data, total_inflow
    
    def analyze_outflow(self):
        """3.2 OUTFLOW ANALYSIS"""
        print(f"\n{'='*70}")
        print(f"3.2 OUTFLOW ANALYSIS: Brain Territory Distribution")
        print(f"{'='*70}\n")
        
        outflow_vessels = {
            'R-MCA Territory (motor/sensory)': 'A70',
            'L-MCA Territory (motor/sensory)': 'A73',
            'R-ACA Territory (frontal/medial)': 'A76',
            'L-ACA Territory (frontal/medial)': 'A78',
            'R-PCA Territory (occipital/vision)': 'A64',
            'L-PCA Territory (occipital/vision)': 'A65'
        }
        
        total_outflow = 0
        outflow_data = {}
        
        print("Blood flow to brain territories:\n")
        
        for name, vessel_id in outflow_vessels.items():
            flow = self.load_mean_flow(vessel_id)
            if flow is not None:
                total_outflow += flow
                outflow_data[name] = flow
                
                # Determine status
                if abs(flow) < 0.01:
                    status = "⚠️  MINIMAL"
                elif flow < 0:
                    status = "🔄 REVERSED"
                else:
                    status = "✓"
                
                print(f"  {status} {name:40s}: {flow:7.2f} mL/min")
        
        print(f"\n  {'TOTAL OUTFLOW':45s}: {total_outflow:7.2f} mL/min")
        
        # Analyze territory distribution
        print(f"\n{'='*70}")
        print("Territory Distribution:")
        print(f"{'='*70}\n")
        
        # Group by territory
        territories = {
            'MCA (Middle Cerebral)': ['R-MCA Territory (motor/sensory)', 
                                     'L-MCA Territory (motor/sensory)'],
            'ACA (Anterior Cerebral)': ['R-ACA Territory (frontal/medial)', 
                                       'L-ACA Territory (frontal/medial)'],
            'PCA (Posterior Cerebral)': ['R-PCA Territory (occipital/vision)', 
                                        'L-PCA Territory (occipital/vision)']
        }
        
        for territory_name, vessel_names in territories.items():
            territory_flow = sum(outflow_data.get(v, 0) for v in vessel_names)
            pct = (territory_flow / total_outflow) * 100 if total_outflow > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {territory_name:30s}: {pct:5.1f}% {bar}")
        
        return outflow_data, total_outflow
    
    def analyze_collaterals(self):
        """3.3 COLLATERAL FLOW ASSESSMENT"""
        print(f"\n{'='*70}")
        print(f"3.3 COLLATERAL VESSEL ANALYSIS")
        print(f"{'='*70}\n")
        
        # Communicating arteries
        collateral_vessels = {
            'Anterior Communicating (ACoA)': 'A77',
            'Right Posterior Communicating (R-PCoA)': 'A62',
            'Left Posterior Communicating (L-PCoA)': 'A63'
        }
        
        # Also check PCA P1 segments (can show reversal)
        p1_vessels = {
            'Right PCA P1': 'A60',
            'Left PCA P1': 'A61'
        }
        
        print("Communicating Artery Flow:\n")
        
        collateral_data = {}
        
        for name, vessel_id in collateral_vessels.items():
            flow = self.load_mean_flow(vessel_id)
            if flow is not None:
                collateral_data[name] = flow
                flow_lmin = flow / 1000  # Convert to L/min for threshold check
                
                # Determine status
                if abs(flow_lmin) < 0.01:
                    status = "○ INACTIVE"
                    activity = "(< 0.01 L/min threshold)"
                elif flow < 0:
                    status = "🔄 ACTIVE (REVERSED)"
                    activity = f"(flow from posterior → anterior)"
                else:
                    status = "✓ ACTIVE (FORWARD)"
                    activity = f"(flow from anterior → posterior)"
                
                print(f"  {status} {name:40s}: {flow:+7.2f} mL/min")
                print(f"       {activity}")
                print()
        
        # Check P1 segments for reversal
        print("PCA P1 Segment Flow (reversal indicates fetal variant):\n")
        
        p1_data = {}
        for name, vessel_id in p1_vessels.items():
            flow = self.load_mean_flow(vessel_id)
            if flow is not None:
                p1_data[name] = flow
                
                if flow < -0.01:
                    status = "🔴 FLOW REVERSAL"
                    interpretation = "→ FETAL VARIANT CONFIRMED"
                elif flow < 0.01:
                    status = "⚠️  MINIMAL FLOW"
                    interpretation = "→ Possible hypoplasia"
                else:
                    status = "✓ NORMAL FLOW"
                    interpretation = "→ Typical anatomy"
                
                print(f"  {status} {name:30s}: {flow:+7.2f} mL/min")
                print(f"       {interpretation}")
                print()
        
        # Clinical interpretation
        print(f"{'='*70}")
        print("COLLATERAL FLOW INTERPRETATION:")
        print(f"{'='*70}\n")
        
        # ACoA assessment
        acoa_flow = collateral_data.get('Anterior Communicating (ACoA)', 0)
        if abs(acoa_flow / 1000) > 0.01:
            print(f"  ✓ ACoA is ACTIVE ({acoa_flow:.2f} mL/min)")
            if acoa_flow > 0:
                print(f"    → Right-to-left flow: Right hemisphere supplying left")
            else:
                print(f"    → Left-to-right flow: Left hemisphere supplying right")
        else:
            print(f"  ○ ACoA is INACTIVE (minimal cross-flow)")
            print(f"    → L/R hemispheres independently perfused")
        
        print()
        
        # PCoA assessment
        r_pcoa = collateral_data.get('Right Posterior Communicating (R-PCoA)', 0)
        l_pcoa = collateral_data.get('Left Posterior Communicating (L-PCoA)', 0)
        
        if abs(r_pcoa / 1000) > 0.01 or abs(l_pcoa / 1000) > 0.01:
            print(f"  ✓ Posterior communicating arteries are ACTIVE")
            
            if r_pcoa > 0:
                print(f"    → R-PCoA: Anterior → Posterior flow ({r_pcoa:.2f} mL/min)")
                print(f"       ⚠️  FETAL VARIANT: R-PCA supplied by R-ICA")
            
            if l_pcoa > 0:
                print(f"    → L-PCoA: Anterior → Posterior flow ({l_pcoa:.2f} mL/min)")
                print(f"       ⚠️  FETAL VARIANT: L-PCA supplied by L-ICA")
        else:
            print(f"  ○ PCoAs are INACTIVE")
            print(f"    → Normal anatomy: PCAs supplied by basilar")
        
        print()
        
        # P1 reversal assessment
        r_p1 = p1_data.get('Right PCA P1', 0)
        l_p1 = p1_data.get('Left PCA P1', 0)
        
        if r_p1 < -0.01 or l_p1 < -0.01:
            print(f"  🔴 PCA P1 FLOW REVERSAL DETECTED")
            if r_p1 < -0.01:
                print(f"     → Right P1 reversed ({r_p1:.2f} mL/min)")
                print(f"     → Confirms RIGHT FETAL POSTERIOR CIRCULATION")
            if l_p1 < -0.01:
                print(f"     → Left P1 reversed ({l_p1:.2f} mL/min)")
                print(f"     → Confirms LEFT FETAL POSTERIOR CIRCULATION")
            
            print(f"\n  STROKE RISK IMPLICATION:")
            print(f"     • ICA now supplies BOTH anterior AND posterior territories")
            print(f"     • ICA stenosis → larger stroke territory")
            print(f"     • Patient requires careful ICA monitoring")
        
        return collateral_data, p1_data
    
    def analyze_asymmetry(self):
        """3.4 ASYMMETRY ANALYSIS"""
        print(f"\n{'='*70}")
        print(f"3.4 LEFT/RIGHT ASYMMETRY ANALYSIS")
        print(f"{'='*70}\n")
        
        vessel_pairs = {
            'Internal Carotid (ICA)': ('A12', 'A16'),
            'Middle Cerebral (MCA)': ('A70', 'A73'),
            'Anterior Cerebral A1': ('A68', 'A69'),
            'Anterior Cerebral A2': ('A76', 'A78'),
            'Posterior Cerebral P1': ('A60', 'A61'),
            'Posterior Cerebral P2': ('A64', 'A65'),
            'Posterior Communicating': ('A62', 'A63')
        }
        
        print("L/R Flow Ratios (Normal range: 0.8 - 1.2):\n")
        
        asymmetry_results = []
        
        for vessel_name, (r_id, l_id) in vessel_pairs.items():
            r_flow = self.load_mean_flow(r_id)
            l_flow = self.load_mean_flow(l_id)
            
            if r_flow is not None and l_flow is not None and abs(r_flow) > 0.001:
                ratio = l_flow / r_flow
                
                # Determine asymmetry status
                if 0.8 <= ratio <= 1.2:
                    status = "✓ SYMMETRIC"
                    interpretation = "Normal"
                elif ratio < 0.8:
                    status = "⚠️  R > L"
                    interpretation = "Right-sided dominance"
                elif ratio > 1.2:
                    status = "⚠️  L > R"
                    interpretation = "Left-sided dominance"
                else:
                    status = "?"
                    interpretation = "Unusual"
                
                asymmetry_results.append({
                    'Vessel': vessel_name,
                    'R_Flow': r_flow,
                    'L_Flow': l_flow,
                    'L/R_Ratio': ratio,
                    'Status': status,
                    'Interpretation': interpretation
                })
                
                print(f"  {status} {vessel_name:30s}: L/R = {ratio:5.2f}  ({interpretation})")
                print(f"       Right: {r_flow:+7.2f} mL/min | Left: {l_flow:+7.2f} mL/min")
                print()
        
        # Summary
        print(f"{'='*70}")
        print("ASYMMETRY SUMMARY:")
        print(f"{'='*70}\n")
        
        n_asymmetric = sum(1 for r in asymmetry_results if '⚠️' in r['Status'])
        n_total = len(asymmetry_results)
        
        print(f"  Vessels analyzed: {n_total}")
        print(f"  Symmetric: {n_total - n_asymmetric}")
        print(f"  Asymmetric: {n_asymmetric}")
        
        if n_asymmetric > 0:
            print(f"\n  SIGNIFICANT ASYMMETRIES:")
            for r in asymmetry_results:
                if '⚠️' in r['Status']:
                    print(f"    • {r['Vessel']}: {r['Interpretation']} (ratio={r['L/R_Ratio']:.2f})")
            
            print(f"\n  CLINICAL SIGNIFICANCE:")
            print(f"    Asymmetry may indicate:")
            print(f"    • Anatomical variant (e.g., fetal PCA)")
            print(f"    • Stenosis or hypoplasia")
            print(f"    • Compensatory flow redistribution")
            print(f"    • Increased stroke risk if >20% asymmetry")
        else:
            print(f"\n  ✓ All vessel pairs show symmetric flow")
        
        return pd.DataFrame(asymmetry_results)
    
    def create_flow_diagram(self, inflow_data, outflow_data, save_path):
        """Create visual flow distribution diagram"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(f'Circle of Willis Flow Distribution: {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Inflow pie chart
        ax1 = axes[0]
        inflow_labels = [k.split('(')[0].strip() for k in inflow_data.keys()]
        inflow_values = list(inflow_data.values())
        colors1 = ['#ff9999', '#66b3ff', '#99ff99']
        
        wedges, texts, autotexts = ax1.pie(inflow_values, labels=inflow_labels, autopct='%1.1f%%',
                                           colors=colors1, startangle=90)
        ax1.set_title('Inflow Sources', fontsize=12, fontweight='bold')
        
        # Plot 2: Outflow bar chart
        ax2 = axes[1]
        outflow_labels = [k.split('(')[0].strip() for k in outflow_data.keys()]
        outflow_values = list(outflow_data.values())
        colors2 = ['red' if v < 0 else 'steelblue' for v in outflow_values]
        
        bars = ax2.barh(range(len(outflow_labels)), outflow_values, color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(outflow_labels)))
        ax2.set_yticklabels(outflow_labels, fontsize=9)
        ax2.set_xlabel('Flow (mL/min)', fontsize=10)
        ax2.set_title('Outflow Distribution', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Summary text
        ax3 = axes[2]
        ax3.axis('off')
        
        summary_text = "FLOW ANALYSIS SUMMARY\n"
        summary_text += "="*35 + "\n\n"
        
        total_in = sum(inflow_data.values())
        total_out = sum(outflow_data.values())
        
        summary_text += f"Total Inflow:  {total_in:.1f} mL/min\n"
        summary_text += f"Total Outflow: {total_out:.1f} mL/min\n\n"
        
        # Calculate circulation split
        ant_flow = inflow_data.get('Right Internal Carotid (R-ICA)', 0) + \
                   inflow_data.get('Left Internal Carotid (L-ICA)', 0)
        post_flow = inflow_data.get('Basilar Artery', 0)
        
        ant_pct = (ant_flow / total_in) * 100
        post_pct = (post_flow / total_in) * 100
        
        summary_text += f"Anterior: {ant_pct:.1f}%\n"
        summary_text += f"Posterior: {post_pct:.1f}%\n\n"
        
        # Key findings
        if post_pct < 10:
            summary_text += "KEY FINDINGS:\n"
            summary_text += "⚠️ Severe posterior\n   reduction\n"
            summary_text += "→ Fetal variant likely\n\n"
        
        # Check for reversals
        reversed_vessels = [k for k, v in outflow_data.items() if v < 0]
        if reversed_vessels:
            summary_text += "🔄 Flow Reversals:\n"
            for vessel in reversed_vessels:
                short_name = vessel.split('(')[0][:15]
                summary_text += f"  • {short_name}\n"
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Flow diagram saved: {save_path}")
    
    def create_asymmetry_plot(self, asymmetry_df, save_path):
        """Create L/R asymmetry visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        vessels = asymmetry_df['Vessel'].tolist()
        ratios = asymmetry_df['L/R_Ratio'].tolist()
        
        colors = ['green' if 0.8 <= r <= 1.2 else 'orange' for r in ratios]
        
        bars = ax.barh(range(len(vessels)), ratios, color=colors, alpha=0.7)
        ax.set_yticks(range(len(vessels)))
        ax.set_yticklabels(vessels, fontsize=10)
        ax.set_xlabel('L/R Flow Ratio', fontsize=11)
        ax.set_title(f'Left/Right Flow Asymmetry: {self.model_name}', 
                    fontsize=14, fontweight='bold')
        
        # Add reference lines
        ax.axvline(x=1.0, color='blue', linestyle='-', linewidth=2, alpha=0.5, label='Perfect symmetry')
        ax.axvline(x=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Normal range')
        ax.axvline(x=1.2, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        # Shade normal range
        ax.axvspan(0.8, 1.2, alpha=0.1, color='green')
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Asymmetry plot saved: {save_path}")


def main():
    """Main analysis function"""
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/cow_collateral_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "#"*70)
    print("# CIRCLE OF WILLIS COLLATERAL FLOW ANALYSIS")
    print("# Patient: patient025_CoW_v2")
    print("#"*70)
    
    analyzer = CoWAnalyzer('patient025_CoW_v2', results_base)
    
    # 3.1 Inflow Analysis
    inflow_data, total_inflow = analyzer.analyze_inflow()
    
    # 3.2 Outflow Analysis
    outflow_data, total_outflow = analyzer.analyze_outflow()
    
    # 3.3 Collateral Assessment
    collateral_data, p1_data = analyzer.analyze_collaterals()
    
    # 3.4 Asymmetry Analysis
    asymmetry_df = analyzer.analyze_asymmetry()
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    flow_diagram_path = output_dir / "cow_flow_distribution.png"
    analyzer.create_flow_diagram(inflow_data, outflow_data, flow_diagram_path)
    
    asymmetry_plot_path = output_dir / "cow_asymmetry_analysis.png"
    analyzer.create_asymmetry_plot(asymmetry_df, asymmetry_plot_path)
    
    # Export data
    asymmetry_csv = output_dir / "asymmetry_analysis.csv"
    asymmetry_df.to_csv(asymmetry_csv, index=False)
    print(f"✓ Asymmetry data saved: {asymmetry_csv}")
    
    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE: Circle of Willis Flow Analysis")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()