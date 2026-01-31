#!/usr/bin/env python3
"""
Hierarchical Arterial Tree with Flow Connections
Shows vascular network topology from aorta to periphery with connecting lines
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import networkx as nx

class ArterialTreeNetwork:
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
    
    def build_graph(self, df):
        """Build network graph from topology"""
        G = nx.DiGraph()
        
        # Add edges from start_node to end_node
        for _, row in df.iterrows():
            vessel_id = row['ID']
            start_node = row['start_node']
            end_node = row['end_node']
            vessel_name = row['name']
            
            flow = self.load_mean_flow(vessel_id)
            
            # Add edge with attributes
            G.add_edge(start_node, end_node, 
                      vessel_id=vessel_id,
                      name=vessel_name,
                      flow=flow)
        
        return G
    
    def create_tree_layout(self, save_path=None):
        """Create hierarchical tree layout"""
        
        # Load data
        df = self.load_topology()
        G = self.build_graph(df)
        
        # Create figure
        fig, (ax_tree, ax_stats) = plt.subplots(1, 2, figsize=(18, 12),
                                                gridspec_kw={'width_ratios': [3, 1]})
        
        fig.suptitle(f'Arterial Network Tree: {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.shell_layout(G)
        
        # Separate nodes by flow direction
        forward_edges = []
        reversed_edges = []
        minimal_edges = []
        
        for u, v, data in G.edges(data=True):
            flow = data.get('flow', 0)
            if flow < -0.01:
                reversed_edges.append((u, v))
            elif abs(flow) <= 0.01:
                minimal_edges.append((u, v))
            else:
                forward_edges.append((u, v))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=forward_edges,
                              edge_color='red', alpha=0.6, width=1.5,
                              arrows=True, arrowsize=10, ax=ax_tree)
        
        nx.draw_networkx_edges(G, pos, edgelist=reversed_edges,
                              edge_color='blue', alpha=0.6, width=2,
                              arrows=True, arrowsize=10, ax=ax_tree)
        
        nx.draw_networkx_edges(G, pos, edgelist=minimal_edges,
                              edge_color='gray', alpha=0.3, width=0.5,
                              arrows=True, arrowsize=5, ax=ax_tree)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='black', 
                              node_size=100, ax=ax_tree)
        
        # Add labels for important nodes
        important_nodes = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 
                          'n31', 'n32', 'n46', 'n37']  # Key junctions
        
        labels = {node: node for node in G.nodes() if node in important_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax_tree)
        
        ax_tree.set_title('Vascular Network Topology\n(Red=Forward, Blue=Reversed, Gray=Minimal)', 
                         fontsize=12)
        ax_tree.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
            Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
            Line2D([0], [0], color='gray', linewidth=1, label='Minimal flow')
        ]
        ax_tree.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Statistics panel
        ax_stats.axis('off')
        ax_stats.text(0.5, 0.95, 'Network Statistics', 
                     fontsize=14, fontweight='bold', ha='center',
                     transform=ax_stats.transAxes)
        
        # Calculate statistics
        n_vessels = len(df)
        n_forward = len(forward_edges)
        n_reversed = len(reversed_edges)
        n_minimal = len(minimal_edges)
        
        stats_text = f"Total vessels: {n_vessels}\n"
        stats_text += f"Total connections: {G.number_of_edges()}\n"
        stats_text += f"Junction nodes: {G.number_of_nodes()}\n\n"
        stats_text += f"Forward flow: {n_forward}\n"
        stats_text += f"Reversed flow: {n_reversed}\n"
        stats_text += f"Minimal flow: {n_minimal}\n\n"
        
        # Circle of Willis vessels
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
        
        stats_text += "Circle of Willis:\n"
        stats_text += "-" * 30 + "\n"
        
        for vid, vname in cow_vessels.items():
            flow = self.load_mean_flow(vid)
            if flow < -0.01:
                status = "REV"
            elif abs(flow) <= 0.01:
                status = "MIN"
            else:
                status = "FWD"
            stats_text += f"{vid} {vname:12s}: {status}\n"
        
        ax_stats.text(0.1, 0.85, stats_text, 
                     fontsize=9, va='top', family='monospace',
                     transform=ax_stats.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] Network tree saved: {save_path}")
        
        return fig
    
    def create_simplified_tree(self, save_path=None):
        """Create simplified tree showing major pathways only"""
        
        df = self.load_topology()
        
        # Define major pathways
        major_vessels = {
            'Aorta': ['A1', 'A2', 'A3', 'A4'],
            'R-Head': ['A5', 'A12'],
            'L-Head': ['A15', 'A16'],
            'R-Brain': ['A68', 'A70', 'A76'],
            'L-Brain': ['A69', 'A73', 'A78'],
            'Posterior': ['A59', 'A60', 'A61'],
            'Trunk': ['A31', 'A34', 'A35'],
            'R-Leg': ['A42', 'A47'],
            'L-Leg': ['A49', 'A54']
        }
        
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle(f'Simplified Arterial Tree: {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Create hierarchical layout manually
        y_start = 10
        x_center = 5
        
        # Aorta (vertical)
        y = y_start
        ax.plot([x_center, x_center], [y, y-2], 'r-', linewidth=3, label='Aorta')
        ax.text(x_center+0.2, y-1, 'Aorta', fontsize=10, fontweight='bold')
        y -= 2
        
        # Main branches
        branches = [
            ('R-Head', 2, -2),
            ('L-Head', -2, -2),
            ('Trunk', 0, -3),
        ]
        
        for branch_name, dx, dy in branches:
            x_branch = x_center + dx
            y_branch = y + dy
            
            # Draw connection
            ax.plot([x_center, x_branch], [y, y_branch], 'r-', linewidth=2)
            
            # Draw branch vessels
            if branch_name in major_vessels:
                vessel_ids = major_vessels[branch_name]
                for i, vid in enumerate(vessel_ids):
                    flow = self.load_mean_flow(vid)
                    color = 'blue' if flow < 0 else 'red'
                    ax.plot(x_branch, y_branch - i*0.5, 'o', 
                           color=color, markersize=8)
                    
                    vessel_data = df[df['ID'] == vid]
                    if len(vessel_data) > 0:
                        vname = vessel_data['name'].values[0][:15]
                        ax.text(x_branch+0.3, y_branch - i*0.5, 
                               f"{vid}: {vname}", fontsize=7)
        
        # Add Circle of Willis detail
        y_cow = y - 3
        ax.text(x_center, y_cow-1, 'Circle of Willis', 
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Draw CoW vessels
        cow_y = y_cow - 2
        for i, (vid, vname) in enumerate([('A59', 'Basilar'), 
                                         ('A60', 'R-PCA'), 
                                         ('A61', 'L-PCA'),
                                         ('A70', 'R-MCA'),
                                         ('A73', 'L-MCA')]):
            flow = self.load_mean_flow(vid)
            color = 'blue' if flow < 0 else 'red'
            marker = '<' if flow < 0 else '>'
            
            x_cow = x_center - 2 + i * 1
            ax.plot(x_cow, cow_y, marker, color=color, markersize=10)
            ax.text(x_cow, cow_y-0.3, f"{vid}\n{vname}", 
                   fontsize=7, ha='center')
        
        ax.set_xlim(-2, 12)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', marker='>', linewidth=0, 
                  markersize=10, label='Forward flow'),
            Line2D([0], [0], color='blue', marker='<', linewidth=0, 
                  markersize=10, label='Reversed flow')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] Simplified tree saved: {save_path}")
        
        return fig


def main():
    """Main function"""
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING ARTERIAL NETWORK TREE DIAGRAMS")
    print("="*70 + "\n")
    
    visualizer = ArterialTreeNetwork('patient025_CoW_v2', results_base)
    
    # Create network graph
    print("Creating network topology graph...")
    network_path = output_dir / "network_topology_patient025.png"
    visualizer.create_tree_layout(save_path=network_path)
    
    # Create simplified tree
    print("Creating simplified pathway tree...")
    simplified_path = output_dir / "simplified_tree_patient025.png"
    visualizer.create_simplified_tree(save_path=simplified_path)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"Saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()