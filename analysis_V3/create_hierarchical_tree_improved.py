#!/usr/bin/env python3
"""
Improved Hierarchical Arterial Tree
Proper tree layout showing blood flow from aorta to periphery
Similar to FirstBlood paper Figure 3
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, Circle

class ImprovedArterialTree:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.model_path = Path.home() / "first_blood/models" / model_name
        self.M3S_TO_MLMIN = 60 * 1000
        
    def load_topology(self):
        """Load arterial network topology"""
        arterial_file = self.model_path / "arterial.csv"
        df = pd.read_csv(arterial_file)
        return df
    
    def load_mean_flow(self, vessel_id):
        """Load mean flow for a vessel"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return 0.0
        
        try:
            data = np.loadtxt(file_path, delimiter=',')
            if data.ndim == 1 or data.shape[1] < 6:
                return 0.0
            flow = data[:, 5]
            return np.mean(flow) * self.M3S_TO_MLMIN
        except:
            return 0.0
    
    def draw_vessel(self, ax, x1, y1, x2, y2, vessel_id, vessel_name, flow):
        """Draw a vessel with flow-based styling"""
        # Determine color and style
        if flow < -0.01:
            color = 'blue'
            linestyle = '-'
            linewidth = 2
        elif abs(flow) <= 0.01:
            color = 'gray'
            linestyle = '--'
            linewidth = 0.5
        else:
            color = 'red'
            linestyle = '-'
            linewidth = 1.5
        
        # Draw line
        ax.plot([x1, x2], [y1, y2], color=color, 
               linestyle=linestyle, linewidth=linewidth, alpha=0.7)
        
        # Add vessel label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Show ID and abbreviated name
        label = f"{vessel_id}"
        if abs(flow) > 0.01:  # Only label significant vessels
            ax.text(mid_x + 0.1, mid_y, label, fontsize=6,
                   bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='white', alpha=0.7))
    
    def draw_node(self, ax, x, y, label=''):
        """Draw junction node"""
        circle = Circle((x, y), 0.15, color='black', zorder=10)
        ax.add_patch(circle)
        if label:
            ax.text(x, y-0.4, label, fontsize=7, ha='center', 
                   fontweight='bold')
    
    def create_hierarchical_tree(self, save_path=None):
        """Create proper hierarchical tree layout"""
        
        df = self.load_topology()
        
        # Create figure
        fig, (ax_main, ax_stats) = plt.subplots(1, 2, figsize=(20, 14),
                                                gridspec_kw={'width_ratios': [4, 1]})
        
        fig.suptitle(f'Arterial Tree (Hierarchical): {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        ax_main.set_xlim(-2, 18)
        ax_main.set_ylim(0, 20)
        ax_main.axis('off')
        
        # LEVEL 0: Heart/Aortic Root
        y_level = 19
        x_center = 8
        
        ax_main.text(x_center, y_level, 'HEART', fontsize=12, 
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # LEVEL 1: Ascending Aorta
        y_level = 17
        vessels_L1 = ['A1']
        for vid in vessels_L1:
            flow = self.load_mean_flow(vid)
            self.draw_vessel(ax_main, x_center, 19, x_center, y_level, 
                           vid, 'Ascending Aorta', flow)
        self.draw_node(ax_main, x_center, y_level, 'Aortic Arch')
        
        # LEVEL 2: Major branches from aortic arch
        y_level = 15
        
        # Define branch points
        branches_L2 = [
            ('A5', 'R-Carotid', 3, 3),      # Right to x=11
            ('A15', 'L-Carotid', -3, -3),   # Left to x=5
            ('A7', 'R-Subclavian', 2, 5),   # Right arm to x=13
            ('A19', 'L-Subclavian', -2, -5),# Left arm to x=3
            ('A25', 'Descending', 0, 0),    # Down center to x=8
        ]
        
        for vid, name, dx_start, dx_end in branches_L2:
            flow = self.load_mean_flow(vid)
            x_start = x_center + dx_start
            x_end = x_center + dx_end
            self.draw_vessel(ax_main, x_center, 17, x_end, y_level, 
                           vid, name, flow)
            self.draw_node(ax_main, x_end, y_level)
        
        # LEVEL 3: HEAD/NECK branches
        y_level_head = 13
        
        # Right side
        x_r_carotid = x_center + 3
        vid = 'A12'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_r_carotid, 15, x_r_carotid, y_level_head,
                       vid, 'R-ICA', flow)
        self.draw_node(ax_main, x_r_carotid, y_level_head, 'R-ICA')
        
        # Left side
        x_l_carotid = x_center - 3
        vid = 'A16'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_l_carotid, 15, x_l_carotid, y_level_head,
                       vid, 'L-ICA', flow)
        self.draw_node(ax_main, x_l_carotid, y_level_head, 'L-ICA')
        
        # LEVEL 4: CIRCLE OF WILLIS
        y_cow = 11
        
        # Draw CoW highlight box
        ax_main.add_patch(plt.Rectangle((x_center-4, y_cow-1), 8, 3,
                                       facecolor='yellow', alpha=0.2,
                                       edgecolor='orange', linewidth=2))
        ax_main.text(x_center, y_cow+2.2, 'CIRCLE OF WILLIS', 
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Basilar from below (posterior circulation)
        x_basilar = x_center
        y_basilar_bottom = 8
        vid = 'A59'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_basilar, y_basilar_bottom, 
                       x_basilar, y_cow-0.5, vid, 'Basilar', flow)
        self.draw_node(ax_main, x_basilar, y_cow-0.5, 'BA')
        
        # R-PCA P1 (may be reversed!)
        vid = 'A60'
        flow = self.load_mean_flow(vid)
        x_r_pca = x_center + 1.5
        self.draw_vessel(ax_main, x_basilar, y_cow-0.5, 
                       x_r_pca, y_cow, vid, 'R-PCA-P1', flow)
        self.draw_node(ax_main, x_r_pca, y_cow, 'R-P1')
        
        # Add REVERSAL annotation if flow is negative
        if flow < 0:
            ax_main.text(x_r_pca+0.5, y_cow, 'REVERSED!', 
                       fontsize=9, color='blue', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
        
        # L-PCA P1
        vid = 'A61'
        flow = self.load_mean_flow(vid)
        x_l_pca = x_center - 1.5
        self.draw_vessel(ax_main, x_basilar, y_cow-0.5, 
                       x_l_pca, y_cow, vid, 'L-PCA-P1', flow)
        self.draw_node(ax_main, x_l_pca, y_cow, 'L-P1')
        
        # R-MCA from R-ICA
        vid = 'A70'
        flow = self.load_mean_flow(vid)
        x_r_mca = x_r_carotid + 1.5
        self.draw_vessel(ax_main, x_r_carotid, y_level_head, 
                       x_r_mca, y_cow+0.5, vid, 'R-MCA', flow)
        self.draw_node(ax_main, x_r_mca, y_cow+0.5, 'R-MCA')
        
        # L-MCA from L-ICA
        vid = 'A73'
        flow = self.load_mean_flow(vid)
        x_l_mca = x_l_carotid - 1.5
        self.draw_vessel(ax_main, x_l_carotid, y_level_head, 
                       x_l_mca, y_cow+0.5, vid, 'L-MCA', flow)
        self.draw_node(ax_main, x_l_mca, y_cow+0.5, 'L-MCA')
        
        # R-ACA from R-ICA
        vid = 'A68'
        flow = self.load_mean_flow(vid)
        x_r_aca = x_r_carotid - 0.5
        self.draw_vessel(ax_main, x_r_carotid, y_level_head, 
                       x_r_aca, y_cow+1, vid, 'R-ACA', flow)
        self.draw_node(ax_main, x_r_aca, y_cow+1, 'R-ACA')
        
        # L-ACA from L-ICA
        vid = 'A69'
        flow = self.load_mean_flow(vid)
        x_l_aca = x_l_carotid + 0.5
        self.draw_vessel(ax_main, x_l_carotid, y_level_head, 
                       x_l_aca, y_cow+1, vid, 'L-ACA', flow)
        self.draw_node(ax_main, x_l_aca, y_cow+1, 'L-ACA')
        
        # ACoA (connecting ACAs)
        vid = 'A77'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_r_aca, y_cow+1, 
                       x_l_aca, y_cow+1, vid, 'ACoA', flow)
        
        # R-PCoA (connecting ICA to PCA)
        vid = 'A62'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_r_carotid, y_level_head-0.5, 
                       x_r_pca, y_cow, vid, 'R-PCoA', flow)
        
        # L-PCoA
        vid = 'A63'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_l_carotid, y_level_head-0.5, 
                       x_l_pca, y_cow, vid, 'L-PCoA', flow)
        
        # LEVEL 5: TRUNK (descending aorta branches)
        y_trunk = 12
        x_trunk = x_center
        
        # Show some major trunk vessels
        trunk_vessels = [
            ('A31', 'Hepatic', 1, 10),
            ('A34', 'Sup.Mesenteric', 0, 9),
            ('A35', 'Renal', -1, 8),
        ]
        
        for vid, name, dx, y_end in trunk_vessels:
            flow = self.load_mean_flow(vid)
            x_end = x_trunk + dx
            self.draw_vessel(ax_main, x_trunk, y_trunk, x_end, y_end,
                           vid, name, flow)
            self.draw_node(ax_main, x_end, y_end)
        
        # LEVEL 6: LEGS (iliac branches)
        y_legs = 6
        
        # Right leg
        x_r_leg = x_center + 2
        vid = 'A42'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_trunk, 8, x_r_leg, y_legs,
                       vid, 'R-Leg', flow)
        self.draw_node(ax_main, x_r_leg, y_legs, 'R-Iliac')
        
        # Left leg
        x_l_leg = x_center - 2
        vid = 'A49'
        flow = self.load_mean_flow(vid)
        self.draw_vessel(ax_main, x_trunk, 8, x_l_leg, y_legs,
                       vid, 'L-Leg', flow)
        self.draw_node(ax_main, x_l_leg, y_legs, 'L-Iliac')
        
        # Add region labels
        ax_main.text(1, 18, 'HEAD/NECK', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax_main.text(1, 10, 'TRUNK', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax_main.text(1, 5, 'LOWER LIMBS', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add flow direction arrows
        ax_main.annotate('', xy=(x_center-0.5, 3), xytext=(x_center-0.5, 5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax_main.text(x_center-0.8, 4, 'FLOW', fontsize=9, rotation=90, va='center')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
            Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
            Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Minimal flow')
        ]
        ax_main.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # STATISTICS PANEL
        ax_stats.axis('off')
        ax_stats.text(0.5, 0.97, 'Flow Analysis', 
                     fontsize=14, fontweight='bold', ha='center',
                     transform=ax_stats.transAxes)
        
        # Calculate stats
        cow_vessels = {
            'A59': 'Basilar',
            'A60': 'R-PCA P1', 
            'A61': 'L-PCA P1',
            'A62': 'R-PCoA',
            'A63': 'L-PCoA',
            'A68': 'R-ACA',
            'A69': 'L-ACA',
            'A70': 'R-MCA',
            'A73': 'L-MCA',
            'A77': 'ACoA'
        }
        
        stats_text = "Circle of Willis:\n"
        stats_text += "="*30 + "\n\n"
        
        for vid, vname in cow_vessels.items():
            flow = self.load_mean_flow(vid)
            if flow < -0.01:
                status = "REV"
                color_mark = "[BLUE]"
            elif abs(flow) <= 0.01:
                status = "MIN"
                color_mark = "[GRAY]"
            else:
                status = "FWD"
                color_mark = "[RED]"
            
            stats_text += f"{vid} {vname:12s}\n"
            stats_text += f"  {color_mark} {status}\n"
            stats_text += f"  {flow:+6.2f} mL/min\n\n"
        
        # Add key findings
        stats_text += "="*30 + "\n"
        stats_text += "KEY FINDINGS:\n"
        stats_text += "="*30 + "\n\n"
        
        r_pca_flow = self.load_mean_flow('A60')
        if r_pca_flow < 0:
            stats_text += "FLOW REVERSAL!\n"
            stats_text += "R-PCA P1 reversed\n\n"
            stats_text += "Indicates:\n"
            stats_text += "- Fetal variant\n"
            stats_text += "- R-PCA fed by ICA\n"
            stats_text += "- Increased stroke\n"
            stats_text += "  risk if ICA\n"
            stats_text += "  stenosis occurs\n"
        
        ax_stats.text(0.05, 0.92, stats_text, 
                     fontsize=8, va='top', family='monospace',
                     transform=ax_stats.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"[SUCCESS] Hierarchical tree saved: {save_path}")
        
        return fig


def main():
    """Main function"""
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING IMPROVED HIERARCHICAL ARTERIAL TREE")
    print("="*70 + "\n")
    
    visualizer = ImprovedArterialTree('patient025_CoW_v2', results_base)
    
    tree_path = output_dir / "hierarchical_tree_improved.png"
    visualizer.create_hierarchical_tree(save_path=tree_path)
    
    print("\n" + "="*70)
    print("COMPLETE - Hierarchical tree shows:")
    print("  - Proper flow hierarchy from heart to periphery")
    print("  - Circle of Willis detail with flow reversal")
    print("  - Major anatomical regions clearly separated")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()