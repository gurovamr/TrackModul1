#!/usr/bin/env python3
"""
Hierarchical Arterial Tree v2
Uses ACTUAL topology from arterial.csv
Brain at TOP, heart at BOTTOM (anatomical orientation)
Blood flows UPWARD - matches FirstBlood paper style
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class ArterialTreeV2:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.M3S_TO_MLMIN = 60 * 1000

    def load_mean_flow(self, vessel_id):
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return 0.0
        try:
            data = np.loadtxt(file_path, delimiter=',')
            if data.ndim == 1 or data.shape[1] < 6:
                return 0.0
            return np.mean(data[:, 5]) * self.M3S_TO_MLMIN
        except:
            return 0.0

    def get_color_and_width(self, flow):
        if flow < -0.01:
            return 'blue', 2.5
        elif abs(flow) <= 0.01:
            return 'gray', 0.8
        else:
            return 'red', 1.8

    def draw_vessel(self, ax, x1, y1, x2, y2, vessel_id, flow):
        color, width = self.get_color_and_width(flow)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.75, zorder=2)
        # vessel ID label at midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.08, my, vessel_id, fontsize=5.5, color='dimgray',
                va='center', zorder=3)

    def draw_node(self, ax, x, y, label='', highlight=False):
        fc = 'black'
        sz = 40
        if highlight:
            fc = 'darkred'
            sz = 60
        ax.plot(x, y, 'o', color=fc, markersize=sz**0.5, zorder=4)
        if label:
            ax.text(x, y - 0.28, label, fontsize=6.5, ha='center',
                    fontweight='bold', zorder=5)

    def create_tree(self, save_path=None):
        # ---------------------------------------------------------------
        # POSITIONS  (x, y)   y=0 bottom (heart), y=20 top (brain tips)
        # Anatomical: brain TOP, heart BOTTOM, blood flows UP
        # ---------------------------------------------------------------
        pos = {
            # --- HEART (bottom) ---
            'H':   (8, 0),

            # --- AORTA ---
            'n1':  (8, 1),      # Asc aorta end / coronary branch
            'n53': (7.2, 0.5),  # Left main coronary branch

            'n2':  (8, 2.5),    # Aortic arch start

            # --- ARCH BRANCHES (spread left-right) ---
            'n6':  (11, 3.5),   # Brachiocephalic end (R side)
            'n3':  (5, 3.5),    # Aortic arch A end (L side)
            'n4':  (3.5, 4.5),  # Aortic arch B end (L side)

            # --- R ARM (far right) ---
            'n7':  (13, 4.5),   # R subclavian end
            'n8':  (15, 5.5),   # R arm brachial
            'n9':  (15.5, 6.5), # R ulnar A

            # --- L ARM (far left) ---
            'n10': (1.5, 5.5),  # L subclavian end
            'n11': (0, 6.5),    # L arm brachial
            'n12': (-0.5, 7.5), # L ulnar A

            # --- R CAROTID ---
            'n32': (12, 6),     # R common carotid end
            'n46': (12, 8),     # R ICA end (ophthalmic branch)
            'n45': (11.5, 9.5), # R ICA sinus (PCoA junction)
            'n44': (11, 11),    # R ICA distal (ant chroid)
            'n43': (10.5, 12.5),# R M1 segment (MCA/ACA split)

            # --- L CAROTID ---
            'n26': (4, 6),      # L common carotid end
            'n37': (4, 8),      # L ICA end (ophthalmic branch)
            'n38': (4.5, 9.5),  # L ICA sinus (PCoA junction)
            'n39': (5, 11),     # L ICA distal (ant chroid)
            'n40': (5.5, 12.5), # L M1 segment (MCA/ACA split)

            # --- POSTERIOR (vertebral / basilar) ---
            'n31': (8, 6.5),    # Vertebral confluence
            'n50': (8, 8),      # Basilar 2 end (cerebellar)
            'n49': (8, 9.5),    # Basilar 1 end (PCA bifurcation)

            # --- PCA P1 ---
            'n47': (9.8, 10.8), # R PCA P1 end  (PCoA + P2)
            'n48': (6.2, 10.8), # L PCA P1 end  (PCoA + P2)

            # --- BRAIN TERMINALS (top) ---
            'R-MCA':  (12.5, 14),
            'L-MCA':  (3.5, 14),
            'R-ACA':  (9.5, 15.5),
            'L-ACA':  (6.5, 15.5),
            'R-PCA':  (10.5, 12.2),
            'L-PCA':  (5.5, 12.2),
            'ACoA':   (8, 14.5),

            # --- TRUNK (descending, below aortic arch) ---
            'n51': (2.8, 5.5),  # Thoracic aorta A end
            'n52': (2, 6.5),    # Thoracic aorta B end
            'n22': (1.5, 7.5),  # Abdominal aorta A end
            'n23': (1.2, 8.5),
            'n24': (1.0, 9.5),
            'n25': (0.8, 10.5),
            'n13': (0.6, 11.5), # Iliac bifurcation

            # --- LEGS ---
            'n14': (-0.5, 12.5),# R iliac
            'n17': (1.7, 12.5), # L iliac
            'n18': (-1, 13.5),
            'n15': (2.2, 13.5),
            'n19': (-1.5, 14.5),
            'n16': (2.7, 14.5),
        }

        # ---------------------------------------------------------------
        # VESSEL CONNECTIONS  (vessel_id, start_node, end_node)
        # Drawn as lines between node positions
        # ---------------------------------------------------------------
        vessels = [
            # Aorta
            ('A1',  'H',   'n1'),
            ('A95', 'n1',  'n2'),
            ('A96', 'n1',  'n53'),   # R coronary (short stub)
            ('A97', 'n1',  'n53'),   # L coronary

            # Arch
            ('A3',  'n2',  'n6'),    # Brachiocephalic
            ('A2',  'n2',  'n3'),    # Arch A
            ('A14', 'n3',  'n4'),    # Arch B

            # R arm
            ('A4',  'n6',  'n7'),
            ('A7',  'n7',  'n8'),
            ('A9',  'n8',  'n9'),

            # L arm
            ('A19', 'n4',  'n10'),
            ('A21', 'n10', 'n11'),
            ('A23', 'n11', 'n12'),

            # R carotid -> ICA -> brain
            ('A5',  'n6',  'n32'),
            ('A12', 'n32', 'n46'),
            ('A79', 'n46', 'n45'),
            ('A66', 'n45', 'n44'),
            ('A101','n44', 'n43'),

            # L carotid -> ICA -> brain
            ('A15', 'n3',  'n26'),
            ('A16', 'n26', 'n37'),
            ('A81', 'n37', 'n38'),
            ('A67', 'n38', 'n39'),
            ('A103','n39', 'n40'),

            # Posterior: vertebrals -> basilar
            ('A6',  'n7',  'n31'),   # R vertebral
            ('A20', 'n10', 'n31'),   # L vertebral
            ('A56', 'n31', 'n50'),   # Basilar 2
            ('A59', 'n50', 'n49'),   # Basilar 1

            # PCA P1
            ('A60', 'n49', 'n47'),   # R PCA P1  <-- REVERSAL
            ('A61', 'n49', 'n48'),   # L PCA P1

            # Communicating arteries
            ('A62', 'n47', 'n45'),   # R PCoA
            ('A63', 'n48', 'n38'),   # L PCoA

            # Trunk (descending)
            ('A18', 'n4',  'n51'),
            ('A27', 'n51', 'n52'),
            ('A28', 'n52', 'n22'),
            ('A35', 'n22', 'n23'),
            ('A37', 'n23', 'n24'),
            ('A39', 'n24', 'n25'),
            ('A41', 'n25', 'n13'),

            # Legs
            ('A42', 'n13', 'n14'),
            ('A43', 'n13', 'n17'),
            ('A50', 'n14', 'n15'),
            ('A44', 'n17', 'n18'),
            ('A52', 'n15', 'n16'),
            ('A46', 'n18', 'n19'),
        ]

        # ---------------------------------------------------------------
        # BRAIN TERMINAL VESSELS (drawn separately with named endpoints)
        # ---------------------------------------------------------------
        brain_terminals = [
            ('A70', 'n43', 'R-MCA'),
            ('A73', 'n40', 'L-MCA'),
            ('A68', 'n43', 'R-ACA'),
            ('A69', 'n40', 'L-ACA'),
            ('A64', 'n47', 'R-PCA'),
            ('A65', 'n48', 'L-PCA'),
            ('A77', 'n40', 'ACoA'),   # ACoA connects L-ACA side to R-ACA side
            ('A76', 'R-ACA', 'ACoA'), # A2 segment stub
        ]

        # ---------------------------------------------------------------
        # DRAW
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.set_xlim(-3, 17)
        ax.set_ylim(-0.8, 16.5)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.suptitle(f'Arterial Network: {self.model_name}',
                     fontsize=17, fontweight='bold')

        # --- Region background boxes ---
        # Brain
        ax.add_patch(plt.Rectangle((3, 9), 10.5, 7, fc='#e6f0ff', ec='steelblue',
                                   lw=1.5, alpha=0.4, zorder=0))
        ax.text(8.25, 15.7, 'BRAIN', fontsize=11, ha='center',
                fontweight='bold', color='steelblue')

        # Head/neck
        ax.add_patch(plt.Rectangle((2.5, 5.5), 11.5, 3.8, fc='#fff0e6', ec='orange',
                                   lw=1.2, alpha=0.3, zorder=0))
        ax.text(8.25, 9.1, 'HEAD / NECK', fontsize=9, ha='center',
                fontweight='bold', color='darkorange')

        # Trunk
        ax.add_patch(plt.Rectangle((-1.8, 4.2), 3.8, 8, fc='#e6ffe6', ec='green',
                                   lw=1.2, alpha=0.3, zorder=0))
        ax.text(-0.2, 12.5, 'TRUNK', fontsize=9, ha='center',
                fontweight='bold', color='green', rotation=90)

        # Arms
        ax.add_patch(plt.Rectangle((14.2, 4.8), 2.2, 2.5, fc='#f0e6ff', ec='purple',
                                   lw=1, alpha=0.3, zorder=0))
        ax.text(15.3, 7.6, 'R ARM', fontsize=7, ha='center', color='purple', fontweight='bold')

        ax.add_patch(plt.Rectangle((-1.2, 6), 1.8, 2.2, fc='#f0e6ff', ec='purple',
                                   lw=1, alpha=0.3, zorder=0))
        ax.text(-0.3, 8.5, 'L ARM', fontsize=7, ha='center', color='purple', fontweight='bold')

        # --- Draw all vessels ---
        for vid, start, end in vessels:
            if start in pos and end in pos:
                flow = self.load_mean_flow(vid)
                self.draw_vessel(ax, pos[start][0], pos[start][1],
                                 pos[end][0], pos[end][1], vid, flow)

        for vid, start, end in brain_terminals:
            if start in pos and end in pos:
                flow = self.load_mean_flow(vid)
                self.draw_vessel(ax, pos[start][0], pos[start][1],
                                 pos[end][0], pos[end][1], vid, flow)

        # --- Draw nodes ---
        # All junction nodes (unlabeled)
        for node, (x, y) in pos.items():
            if node.startswith('n') or node == 'H':
                self.draw_node(ax, x, y)

        # Key labeled nodes
        labels = {
            'H':   'Heart',
            'n2':  'Aortic Arch',
            'n6':  'Brachicep.',
            'n32': 'R-CCA',
            'n26': 'L-CCA',
            'n46': 'R-ICA',
            'n37': 'L-ICA',
            'n45': 'R-PCoA jxn',
            'n38': 'L-PCoA jxn',
            'n43': 'R-M1',
            'n40': 'L-M1',
            'n31': 'Vertebral\nconfluence',
            'n49': 'BA bifurc',
            'n47': 'R-P1',
            'n48': 'L-P1',
            'n13': 'Iliac bifurc',
        }
        for node, label in labels.items():
            if node in pos:
                x, y = pos[node]
                highlight = (node == 'H')
                self.draw_node(ax, x, y, label, highlight)

        # Brain terminal labeled nodes
        for name in ['R-MCA', 'L-MCA', 'R-ACA', 'L-ACA', 'R-PCA', 'L-PCA', 'ACoA']:
            if name in pos:
                x, y = pos[name]
                ax.plot(x, y, 'o', color='black', markersize=5, zorder=4)
                ax.text(x, y + 0.25, name, fontsize=7, ha='center',
                        fontweight='bold', color='navy', zorder=5)

        # --- CoW highlight box ---
        ax.add_patch(plt.Rectangle((5.5, 9.2), 5, 2.2, fc='yellow', ec='darkorange',
                                   lw=2, alpha=0.35, zorder=1))
        ax.text(8, 11.6, 'Circle of Willis', fontsize=9, ha='center',
                fontweight='bold', color='darkorange')

        # --- Flow reversal annotation ---
        r_pca_flow = self.load_mean_flow('A60')
        if r_pca_flow < 0:
            ax.annotate('FLOW REVERSAL\n(Fetal variant)',
                        xy=(9.8, 10.8), xytext=(12.5, 11.5),
                        fontsize=8, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round', fc='cyan', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))

        # --- Flow direction arrow (global) ---
        ax.annotate('', xy=(16, 14), xytext=(16, 1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        ax.text(16.4, 7.5, 'Blood\nflow\ndirection', fontsize=8,
                ha='left', va='center', style='italic')

        # --- Legend ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red',  linewidth=2.5, label='Forward flow'),
            Line2D([0], [0], color='blue', linewidth=2.5, label='Reversed flow'),
            Line2D([0], [0], color='gray', linewidth=1,   linestyle='--', label='Minimal flow'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                  framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"[SUCCESS] Arterial tree saved: {save_path}")

        return fig


def main():
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING HIERARCHICAL ARTERIAL TREE v2")
    print("  - Brain at TOP (anatomical orientation)")
    print("  - Heart at BOTTOM")
    print("  - Using actual topology from arterial.csv")
    print("=" * 70 + "\n")

    viz = ArterialTreeV2('patient025_CoW_v2', results_base)

    tree_path = output_dir / "hierarchical_tree_v2.png"
    viz.create_tree(save_path=tree_path)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()