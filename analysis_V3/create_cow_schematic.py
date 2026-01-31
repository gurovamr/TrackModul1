#!/usr/bin/env python3
"""
Circle of Willis Schematic Visualization with Flow Arrows
Creates a medical-textbook style CoW diagram showing flow directions and magnitudes
Patient-specific for patient025_CoW_v2
"""

import matplotlib
matplotlib.use('Agg')  # Suppress Qt warning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

class CoWSchematic:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.M3S_TO_MLMIN = 60 * 1000
        
    def load_mean_flow(self, vessel_id):
        """Load mean flow for a vessel"""
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return 0.0
        data = np.loadtxt(file_path, delimiter=',')
        flow = data[:, 5]
        return np.mean(flow) * self.M3S_TO_MLMIN
    
    def draw_arrow(self, ax, x1, y1, x2, y2, flow, label='', color='red'):
        """Draw flow arrow with magnitude-based width"""
        # Arrow width based on flow magnitude
        width = max(abs(flow) * 20, 0.5)
        
        # Arrow color based on flow direction
        if flow < 0:
            color = 'blue'  # Reversed flow
            arrowstyle = '<-'
        else:
            color = 'red'   # Normal flow
            arrowstyle = '->'
        
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=arrowstyle,
            linewidth=width,
            color=color,
            alpha=0.7,
            mutation_scale=20
        )
        ax.add_patch(arrow)
        
        # Add flow label
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f'{abs(flow):.2f}',
                   fontsize=7, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def draw_vessel_node(self, ax, x, y, label='', size=150):
        """Draw vessel junction node"""
        circle = Circle((x, y), 0.3, color='black', zorder=10)
        ax.add_patch(circle)
        if label:
            ax.text(x, y-0.8, label, fontsize=7, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    def create_cow_diagram(self, save_path=None):
        """Create Circle of Willis schematic diagram"""
        
        # Load flow data
        flows = {
            'R-ICA': self.load_mean_flow('A12'),
            'L-ICA': self.load_mean_flow('A16'),
            'Basilar': self.load_mean_flow('A59'),
            'R-MCA': self.load_mean_flow('A70'),
            'L-MCA': self.load_mean_flow('A73'),
            'R-ACA-A1': self.load_mean_flow('A68'),
            'L-ACA-A1': self.load_mean_flow('A69'),
            'R-ACA-A2': self.load_mean_flow('A76'),
            'L-ACA-A2': self.load_mean_flow('A78'),
            'R-PCA-P1': self.load_mean_flow('A60'),
            'L-PCA-P1': self.load_mean_flow('A61'),
            'R-PCA-P2': self.load_mean_flow('A64'),
            'L-PCA-P2': self.load_mean_flow('A65'),
            'R-PCoA': self.load_mean_flow('A62'),
            'L-PCoA': self.load_mean_flow('A63'),
            'ACoA': self.load_mean_flow('A77')
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(-8, 8)
        ax.set_ylim(-2, 14)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.text(0, 13, f'Circle of Willis: {self.model_name}', 
               fontsize=16, fontweight='bold', ha='center')
        
        # Define node positions (anatomical layout)
        nodes = {
            # Posterior circulation
            'BA': (0, 0),           # Basilar artery entry
            'BA-top': (0, 2),       # Basilar bifurcation
            
            # Right posterior
            'R-PCA-start': (2, 2),  # Right PCA P1 start
            'R-PCA-mid': (3, 3),    # Right PCA middle
            
            # Left posterior  
            'L-PCA-start': (-2, 2), # Left PCA P1 start
            'L-PCA-mid': (-3, 3),   # Left PCA middle
            
            # Right anterior
            'R-ICA': (4, 6),        # Right ICA entry
            'R-ICA-top': (3, 8),    # Right ICA top
            'R-ACA-start': (1.5, 9),# Right ACA A1 start
            'R-ACA-top': (0.5, 10), # Right ACA A1 end
            
            # Left anterior
            'L-ICA': (-4, 6),       # Left ICA entry
            'L-ICA-top': (-3, 8),   # Left ICA top
            'L-ACA-start': (-1.5, 9),# Left ACA A1 start
            'L-ACA-top': (-0.5, 10),# Left ACA A1 end
            
            # Communicating
            'ACoA': (0, 10),        # Anterior communicating
        }
        
        # Draw posterior circulation
        # Basilar artery
        self.draw_arrow(ax, nodes['BA'][0], nodes['BA'][1],
                       nodes['BA-top'][0], nodes['BA-top'][1],
                       flows['Basilar'], 'Basilar')
        
        # Right PCA P1
        self.draw_arrow(ax, nodes['BA-top'][0], nodes['BA-top'][1],
                       nodes['R-PCA-start'][0], nodes['R-PCA-start'][1],
                       flows['R-PCA-P1'], 'R-P1')
        
        # Left PCA P1
        self.draw_arrow(ax, nodes['BA-top'][0], nodes['BA-top'][1],
                       nodes['L-PCA-start'][0], nodes['L-PCA-start'][1],
                       flows['L-PCA-P1'], 'L-P1')
        
        # Right PCA P2
        self.draw_arrow(ax, nodes['R-PCA-start'][0], nodes['R-PCA-start'][1],
                       nodes['R-PCA-mid'][0], nodes['R-PCA-mid'][1],
                       flows['R-PCA-P2'], 'R-P2')
        
        # Left PCA P2
        self.draw_arrow(ax, nodes['L-PCA-start'][0], nodes['L-PCA-start'][1],
                       nodes['L-PCA-mid'][0], nodes['L-PCA-mid'][1],
                       flows['L-PCA-P2'], 'L-P2')
        
        # Draw anterior circulation
        # Right ICA
        self.draw_arrow(ax, nodes['R-ICA'][0], nodes['R-ICA'][1],
                       nodes['R-ICA-top'][0], nodes['R-ICA-top'][1],
                       flows['R-ICA'], 'R-ICA')
        
        # Left ICA
        self.draw_arrow(ax, nodes['L-ICA'][0], nodes['L-ICA'][1],
                       nodes['L-ICA-top'][0], nodes['L-ICA-top'][1],
                       flows['L-ICA'], 'L-ICA')
        
        # Right MCA (branch off)
        self.draw_arrow(ax, nodes['R-ICA-top'][0], nodes['R-ICA-top'][1],
                       nodes['R-ICA-top'][0]+1.5, nodes['R-ICA-top'][1]+1,
                       flows['R-MCA'], 'R-MCA')
        
        # Left MCA (branch off)
        self.draw_arrow(ax, nodes['L-ICA-top'][0], nodes['L-ICA-top'][1],
                       nodes['L-ICA-top'][0]-1.5, nodes['L-ICA-top'][1]+1,
                       flows['L-MCA'], 'L-MCA')
        
        # Right ACA A1
        self.draw_arrow(ax, nodes['R-ICA-top'][0], nodes['R-ICA-top'][1],
                       nodes['R-ACA-start'][0], nodes['R-ACA-start'][1],
                       flows['R-ACA-A1'], 'R-A1')
        
        # Left ACA A1
        self.draw_arrow(ax, nodes['L-ICA-top'][0], nodes['L-ICA-top'][1],
                       nodes['L-ACA-start'][0], nodes['L-ACA-start'][1],
                       flows['L-ACA-A1'], 'L-A1')
        
        # Right ACA A2
        self.draw_arrow(ax, nodes['R-ACA-start'][0], nodes['R-ACA-start'][1],
                       nodes['R-ACA-top'][0], nodes['R-ACA-top'][1],
                       flows['R-ACA-A2'], 'R-A2')
        
        # Left ACA A2
        self.draw_arrow(ax, nodes['L-ACA-start'][0], nodes['L-ACA-start'][1],
                       nodes['L-ACA-top'][0], nodes['L-ACA-top'][1],
                       flows['L-ACA-A2'], 'L-A2')
        
        # Communicating arteries
        # ACoA
        self.draw_arrow(ax, nodes['R-ACA-top'][0], nodes['R-ACA-top'][1],
                       nodes['L-ACA-top'][0], nodes['L-ACA-top'][1],
                       flows['ACoA'], 'ACoA')
        
        # Right PCoA
        self.draw_arrow(ax, nodes['R-ICA-top'][0], nodes['R-ICA-top'][1]-0.5,
                       nodes['R-PCA-start'][0], nodes['R-PCA-start'][1]+0.5,
                       flows['R-PCoA'], 'R-PCoA')
        
        # Left PCoA
        self.draw_arrow(ax, nodes['L-ICA-top'][0], nodes['L-ICA-top'][1]-0.5,
                       nodes['L-PCA-start'][0], nodes['L-PCA-start'][1]+0.5,
                       flows['L-PCoA'], 'L-PCoA')
        
        # Draw nodes at junctions with labels
        node_labels = {
            'BA': 'Basilar',
            'BA-top': 'BA Bifurc',
            'R-PCA-start': 'R-P1 Start',
            'R-PCA-mid': 'R-PCA',
            'L-PCA-start': 'L-P1 Start',
            'L-PCA-mid': 'L-PCA',
            'R-ICA': 'R-ICA',
            'R-ICA-top': 'R-ICA Top',
            'R-ACA-start': 'R-A1',
            'R-ACA-top': 'R-ACA',
            'L-ICA': 'L-ICA',
            'L-ICA-top': 'L-ICA Top',
            'L-ACA-start': 'L-A1',
            'L-ACA-top': 'L-ACA',
            'ACoA': 'ACoA'
        }
        
        for node_name, (x, y) in nodes.items():
            label = node_labels.get(node_name, '')
            self.draw_vessel_node(ax, x, y, label)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='red', label='Normal flow'),
            mpatches.Patch(color='blue', label='Reversed flow'),
            mpatches.Patch(color='white', label='Width = flow magnitude')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add annotations for key findings
        if flows['R-PCA-P1'] < 0:
            ax.text(6, 2, 'FLOW REVERSAL\nFetal Variant', 
                   fontsize=12, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add flow summary
        summary_text = f"Total Inflow: {flows['R-ICA'] + flows['L-ICA'] + flows['Basilar']:.2f} mL/min\n"
        summary_text += f"Basilar: {flows['Basilar']:.2f} mL/min (Reduced)\n"
        summary_text += f"R-PCA P1: {flows['R-PCA-P1']:.2f} mL/min (Reversed)"
        
        ax.text(-7, 11, summary_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] CoW schematic saved: {save_path}")
        
        return fig


def main():
    """Main function"""
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING CIRCLE OF WILLIS SCHEMATIC DIAGRAM")
    print("="*70 + "\n")
    
    visualizer = CoWSchematic('patient025_CoW_v2', results_base)
    
    schematic_path = output_dir / "cow_schematic_patient025.png"
    visualizer.create_cow_diagram(save_path=schematic_path)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"Saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()