#!/usr/bin/env python3
"""
Arterial Tree - 3 Subplot Layout (matches paper Figure 3)

SUBPLOT 1 (top):    HEAD only - brain fan, CoW, carotids, ext carotid branches
SUBPLOT 2 (middle): TORSO + ARMS - heart, aorta down, long arms down sides
SUBPLOT 3 (bottom): LEGS only - iliac split, femoral, tibial

Each subplot has its own coordinate space.
White open circles = junctions, gray filled = terminals, red dot = heart.
Blue = reversed flow. No background boxes.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ThreePartTree:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.M3S_TO_MLMIN = 60 * 1000

    def load_mean_flow(self, vid):
        fp = self.results_path / f"{vid}.txt"
        if not fp.exists():
            return 0.0
        try:
            data = np.loadtxt(fp, delimiter=',')
            if data.ndim == 1 or data.shape[1] < 6:
                return 0.0
            return np.mean(data[:, 5]) * self.M3S_TO_MLMIN
        except:
            return 0.0

    def _style(self, flow):
        if flow < -0.01:
            return 'blue', 2.2
        elif abs(flow) <= 0.01:
            return '#bbbbbb', 0.9
        return 'red', 1.6

    def line(self, ax, pts, vid):
        flow = self.load_mean_flow(vid)
        color, lw = self._style(flow)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=lw,
                solid_capstyle='round', zorder=2, alpha=0.85)
        return flow

    def junc(self, ax, x, y):
        ax.plot(x, y, 'o', color='white', markersize=6,
                markeredgecolor='black', markeredgewidth=0.9, zorder=4)

    def term(self, ax, x, y):
        ax.plot(x, y, 'o', color='#777777', markersize=4.5, zorder=4)

    # ------------------------------------------------------------------
    # SUBPLOT 1: HEAD
    # Coordinate space: x in [-6, 6], y in [0, 12]
    # Layout: carotids enter at bottom-center, CoW ring in middle,
    #         brain terminals fan out at top, ext carotid fans on sides
    # ------------------------------------------------------------------
    def draw_head(self, ax):
        # --- Entry points (bottom of head subplot) ---
        r_cca  = (1.0, 1.0)    # R common carotid top
        l_cca  = (-1.0, 1.0)   # L common carotid top

        # --- External carotid fans (spread wide on each side) ---
        r_eca  = (2.0, 1.5)
        l_eca  = (-2.0, 1.5)

        # R ext carotid sub-branches
        r_eca2 = (3.0, 1.2)
        r_eca3 = (3.8, 2.0)
        self.line(ax, [r_cca, r_eca], 'A13');  self.junc(ax, *r_eca)
        self.line(ax, [r_eca, r_eca2], 'A83'); self.junc(ax, *r_eca2)
        self.line(ax, [r_eca, (2.2, 0.6)], 'A84'); self.term(ax, 2.2, 0.6)
        self.line(ax, [r_eca2, r_eca3], 'A87'); self.junc(ax, *r_eca3)
        self.line(ax, [r_eca3, (4.5, 2.5)], 'A91'); self.term(ax, 4.5, 2.5)
        self.line(ax, [r_eca3, (4.2, 1.2)], 'A92'); self.term(ax, 4.2, 1.2)
        self.line(ax, [r_eca2, (3.5, 0.4)], 'A88'); self.term(ax, 3.5, 0.4)

        # L ext carotid (mirror)
        l_eca2 = (-3.0, 1.2)
        l_eca3 = (-3.8, 2.0)
        self.line(ax, [l_cca, l_eca], 'A17');  self.junc(ax, *l_eca)
        self.line(ax, [l_eca, l_eca2], 'A85'); self.junc(ax, *l_eca2)
        self.line(ax, [l_eca, (-2.2, 0.6)], 'A86'); self.term(ax, -2.2, 0.6)
        self.line(ax, [l_eca2, l_eca3], 'A89'); self.junc(ax, *l_eca3)
        self.line(ax, [l_eca3, (-4.5, 2.5)], 'A93'); self.term(ax, -4.5, 2.5)
        self.line(ax, [l_eca3, (-4.2, 1.2)], 'A94'); self.term(ax, -4.2, 1.2)
        self.line(ax, [l_eca2, (-3.5, 0.4)], 'A90'); self.term(ax, -3.5, 0.4)

        # --- ICA path upward into CoW ---
        r_ica     = (0.9, 2.5)     # n46
        l_ica     = (-0.9, 2.5)    # n37
        r_pcoa_jn = (0.75, 3.8)    # n45  PCoA junction
        l_pcoa_jn = (-0.75, 3.8)   # n38
        r_ant_ch  = (0.7, 4.8)     # n44  anterior choroidal
        l_ant_ch  = (-0.7, 4.8)    # n39
        r_m1      = (0.65, 5.8)    # n43  MCA/ACA split
        l_m1      = (-0.65, 5.8)   # n40

        self.line(ax, [r_cca, r_ica], 'A12');  self.junc(ax, *r_cca); self.junc(ax, *r_ica)
        self.line(ax, [l_cca, l_ica], 'A16');  self.junc(ax, *l_cca); self.junc(ax, *l_ica)

        # Ophthalmic stubs
        self.line(ax, [r_ica, (1.6, 2.8)], 'A80'); self.term(ax, 1.6, 2.8)
        self.line(ax, [l_ica, (-1.6, 2.8)], 'A82'); self.term(ax, -1.6, 2.8)

        self.line(ax, [r_ica, r_pcoa_jn], 'A79');  self.junc(ax, *r_pcoa_jn)
        self.line(ax, [l_ica, l_pcoa_jn], 'A81');  self.junc(ax, *l_pcoa_jn)
        self.line(ax, [r_pcoa_jn, r_ant_ch], 'A66');  self.junc(ax, *r_ant_ch)
        self.line(ax, [l_pcoa_jn, l_ant_ch], 'A67');  self.junc(ax, *l_ant_ch)
        self.line(ax, [r_ant_ch, r_m1], 'A101'); self.junc(ax, *r_m1)
        self.line(ax, [l_ant_ch, l_m1], 'A103'); self.junc(ax, *l_m1)

        # Anterior choroidal stubs
        self.line(ax, [r_ant_ch, (1.3, 5.2)], 'A100'); self.term(ax, 1.3, 5.2)
        self.line(ax, [l_ant_ch, (-1.3, 5.2)], 'A102'); self.term(ax, -1.3, 5.2)

        # --- POSTERIOR: vertebrals -> basilar -> PCA ---
        vert_conf = (0, 2.2)   # n31
        ba_mid    = (0, 3.2)   # n50
        ba_bif    = (0, 4.2)   # n49
        r_p1      = (0.4, 4.8) # n47
        l_p1      = (-0.4, 4.8)# n48

        # Vertebrals enter from bottom (stubs representing entry from torso)
        self.line(ax, [(0.6, 0.2), vert_conf], 'A6');
        self.line(ax, [(-0.6, 0.2), vert_conf], 'A20');
        self.junc(ax, *vert_conf)

        self.line(ax, [vert_conf, ba_mid], 'A56'); self.junc(ax, *ba_mid)
        # Superior cerebellar stubs
        self.line(ax, [ba_mid, (0.5, 2.8)], 'A57'); self.term(ax, 0.5, 2.8)
        self.line(ax, [ba_mid, (-0.5, 2.8)], 'A58'); self.term(ax, -0.5, 2.8)

        self.line(ax, [ba_mid, ba_bif], 'A59'); self.junc(ax, *ba_bif)

        # PCA P1 --- R may be REVERSED
        r_pca_flow = self.line(ax, [ba_bif, r_p1], 'A60'); self.junc(ax, *r_p1)
        self.line(ax, [ba_bif, l_p1], 'A61');              self.junc(ax, *l_p1)

        # PCoA connections
        self.line(ax, [r_p1, r_pcoa_jn], 'A62')
        self.line(ax, [l_p1, l_pcoa_jn], 'A63')

        # PCA P2 terminals
        self.line(ax, [r_p1, (1.0, 5.3)], 'A64'); self.term(ax, 1.0, 5.3)
        self.line(ax, [l_p1, (-1.0, 5.3)], 'A65'); self.term(ax, -1.0, 5.3)

        # --- BRAIN TERMINALS: MCA, ACA fan outward ---
        # R MCA
        r_mca  = (2.2, 6.8)
        r_m2a  = (3.2, 7.8)
        r_m2b  = (2.0, 8.2)
        self.line(ax, [r_m1, r_mca], 'A70');  self.junc(ax, *r_mca)
        self.line(ax, [r_mca, r_m2a], 'A71'); self.term(ax, *r_m2a)
        self.line(ax, [r_mca, r_m2b], 'A72'); self.term(ax, *r_m2b)

        # L MCA (mirror)
        l_mca  = (-2.2, 6.8)
        l_m2a  = (-3.2, 7.8)
        l_m2b  = (-2.0, 8.2)
        self.line(ax, [l_m1, l_mca], 'A73');  self.junc(ax, *l_mca)
        self.line(ax, [l_mca, l_m2a], 'A74'); self.term(ax, *l_m2a)
        self.line(ax, [l_mca, l_m2b], 'A75'); self.term(ax, *l_m2b)

        # R ACA
        r_aca_a1 = (0.8, 7.0)   # n42
        r_aca_a2 = (1.2, 8.5)
        self.line(ax, [r_m1, r_aca_a1], 'A68'); self.junc(ax, *r_aca_a1)
        self.line(ax, [r_aca_a1, r_aca_a2], 'A76'); self.term(ax, *r_aca_a2)

        # L ACA (mirror)
        l_aca_a1 = (-0.8, 7.0)  # n41
        l_aca_a2 = (-1.2, 8.5)
        self.line(ax, [l_m1, l_aca_a1], 'A69'); self.junc(ax, *l_aca_a1)
        self.line(ax, [l_aca_a1, l_aca_a2], 'A78'); self.term(ax, *l_aca_a2)

        # ACoA connects the two ACA A1 nodes
        self.line(ax, [l_aca_a1, r_aca_a1], 'A77')

        # --- Annotation for flow reversal ---
        if r_pca_flow < 0:
            ax.annotate('FLOW REVERSAL\n(Fetal variant)',
                        xy=(0.4, 4.8), xytext=(2.5, 4.0),
                        fontsize=7, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round', fc='cyan', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                        zorder=6)

    # ------------------------------------------------------------------
    # SUBPLOT 2: TORSO + ARMS
    # Coordinate space: x in [-7, 7], y in [0, 12]
    # Heart at top-center. Aorta straight down.
    # Arms start at shoulders and curve down the full height on each side.
    # ------------------------------------------------------------------
    def draw_torso(self, ax):
        heart = (0, 11.0)
        ax.plot(*heart, 'o', color='red', markersize=9, zorder=5)

        # Ascending aorta
        n1 = (0, 11.5)   # coronary branch point
        n2 = (0, 11.8)   # aortic arch
        self.line(ax, [heart, n1], 'A1');  self.junc(ax, *n1)
        self.line(ax, [n1, n2], 'A95');    self.junc(ax, *n2)

        # Coronaries (short stubs)
        self.line(ax, [n1, (-0.5, 11.2)], 'A96'); self.term(ax, -0.5, 11.2)
        n53 = (-0.7, 11.4)
        self.line(ax, [n1, n53], 'A97'); self.junc(ax, *n53)
        self.line(ax, [n53, (-1.1, 11.1)], 'A98'); self.term(ax, -1.1, 11.1)
        self.line(ax, [n53, (-0.9, 11.6)], 'A99'); self.term(ax, -0.9, 11.6)

        # Arch branches
        n6  = (1.2, 11.5)    # brachiocephalic end
        n3  = (-0.8, 11.8)   # arch A end
        n4  = (-1.8, 11.5)   # arch B end
        self.line(ax, [n2, n6], 'A3');  self.junc(ax, *n6)
        self.line(ax, [n2, n3], 'A2');  self.junc(ax, *n3)
        self.line(ax, [n3, n4], 'A14'); self.junc(ax, *n4)

        # --- R CAROTID exits top (stub upward) ---
        r_cca = (1.0, 11.9)
        self.line(ax, [n6, r_cca], 'A5'); self.junc(ax, *r_cca)
        # stub going up (connects to head subplot)
        ax.plot([r_cca[0], r_cca[0]], [r_cca[1], 12.0],
                color='red', linewidth=1.6, alpha=0.85, zorder=2)
        self.term(ax, r_cca[0], 12.0)

        # --- L CAROTID exits top ---
        l_cca = (-1.0, 11.9)
        self.line(ax, [n3, l_cca], 'A15'); self.junc(ax, *l_cca)
        ax.plot([l_cca[0], l_cca[0]], [l_cca[1], 12.0],
                color='red', linewidth=1.6, alpha=0.85, zorder=2)
        self.term(ax, l_cca[0], 12.0)

        # --- VERTEBRALS exit top (stubs) ---
        n7  = (2.2, 11.2)   # R subclavian end
        n10 = (-2.5, 11.2)  # L subclavian end
        self.line(ax, [n6, n7], 'A4');  self.junc(ax, *n7)
        self.line(ax, [n4, n10], 'A19'); self.junc(ax, *n10)
        # vertebral stubs going up
        self.line(ax, [n7, (1.5, 11.9)], 'A6');  self.term(ax, 1.5, 11.9)
        self.line(ax, [n10, (-1.5, 11.9)], 'A20'); self.term(ax, -1.5, 11.9)

        # --- R ARM: long curve down right side ---
        # n7 (2.2, 11.2) -> shoulder -> elbow -> wrist
        r_shoulder = (3.0, 10.8)
        r_elbow    = (4.5, 8.5)
        r_wrist    = (5.0, 6.0)
        r_n9       = (5.2, 5.2)   # ulnar split

        self.line(ax, [n7, r_shoulder, r_elbow, r_wrist], 'A7')
        self.junc(ax, *r_wrist)  # n8

        # Radial
        self.line(ax, [r_wrist, (5.8, 5.2)], 'A8'); self.term(ax, 5.8, 5.2)
        # Ulnar A
        self.line(ax, [r_wrist, r_n9], 'A9'); self.junc(ax, *r_n9)
        # Interosseous + Ulnar B
        self.line(ax, [r_n9, (5.6, 4.2)], 'A10'); self.term(ax, 5.6, 4.2)
        self.line(ax, [r_n9, (4.8, 4.0)], 'A11'); self.term(ax, 4.8, 4.0)

        # --- L ARM: mirror ---
        l_shoulder = (-3.0, 10.8)
        l_elbow    = (-4.5, 8.5)
        l_wrist    = (-5.0, 6.0)
        l_n12      = (-5.2, 5.2)

        self.line(ax, [n10, l_shoulder, l_elbow, l_wrist], 'A21')
        self.junc(ax, *l_wrist)  # n11

        self.line(ax, [l_wrist, (-5.8, 5.2)], 'A22'); self.term(ax, -5.8, 5.2)
        self.line(ax, [l_wrist, l_n12], 'A23'); self.junc(ax, *l_n12)
        self.line(ax, [l_n12, (-5.6, 4.2)], 'A24'); self.term(ax, -5.6, 4.2)
        self.line(ax, [l_n12, (-4.8, 4.0)], 'A25'); self.term(ax, -4.8, 4.0)

        # --- DESCENDING AORTA: straight down center ---
        # Thoracic A
        n51 = (0, 10.2)
        self.line(ax, [n4, (-1.2, 10.8), (-0.3, 10.5), n51], 'A18')
        self.junc(ax, *n51)
        # Intercostals stub
        self.line(ax, [n51, (0.7, 10.0)], 'A26'); self.term(ax, 0.7, 10.0)

        # Thoracic B
        n52 = (0, 9.0)
        self.line(ax, [n51, n52], 'A27'); self.junc(ax, *n52)

        # Celiac branch
        n20 = (-0.7, 8.8)
        self.line(ax, [n52, n20], 'A29'); self.junc(ax, *n20)
        n21 = (-1.3, 8.5)
        self.line(ax, [n20, n21], 'A30'); self.junc(ax, *n21)
        self.line(ax, [n21, (-1.8, 8.2)], 'A31'); self.term(ax, -1.8, 8.2)  # hepatic
        self.line(ax, [n21, (-1.5, 7.8)], 'A33'); self.term(ax, -1.5, 7.8)  # splenic
        self.line(ax, [n20, (-1.0, 8.0)], 'A32'); self.term(ax, -1.0, 8.0)  # gastric

        # Abdominal A
        n22 = (0, 7.8)
        self.line(ax, [n52, n22], 'A28'); self.junc(ax, *n22)
        # Superior mesenteric
        self.line(ax, [n22, (-0.8, 7.6)], 'A34'); self.term(ax, -0.8, 7.6)

        # Abdominal B
        n23 = (0, 6.8)
        self.line(ax, [n22, n23], 'A35'); self.junc(ax, *n23)
        # Renal R
        self.line(ax, [n23, (0.8, 6.6)], 'A36'); self.term(ax, 0.8, 6.6)
        # Renal L
        self.line(ax, [n23, (-0.8, 6.6)], 'A38'); self.term(ax, -0.8, 6.6)

        # Abdominal C
        n24 = (0, 5.8)
        self.line(ax, [n23, n24], 'A37'); self.junc(ax, *n24)

        # Abdominal D
        n25 = (0, 4.8)
        self.line(ax, [n24, n25], 'A39'); self.junc(ax, *n25)
        # Inferior mesenteric
        self.line(ax, [n25, (-0.8, 4.6)], 'A40'); self.term(ax, -0.8, 4.6)

        # Abdominal E -> iliac bifurcation (exits bottom)
        n13 = (0, 3.8)
        self.line(ax, [n25, n13], 'A41'); self.junc(ax, *n13)
        # stub downward (connects to leg subplot)
        ax.plot([n13[0], n13[0]], [n13[1], 3.0],
                color='red', linewidth=1.6, alpha=0.85, zorder=2)
        self.term(ax, n13[0], 3.0)

    # ------------------------------------------------------------------
    # SUBPLOT 3: LEGS
    # Coordinate space: x in [-5, 5], y in [0, 12]
    # Iliac bifurcation at top center, legs go straight down
    # ------------------------------------------------------------------
    def draw_legs(self, ax):
        n13 = (0, 11.5)   # iliac bifurcation (top)
        self.junc(ax, *n13)

        # --- R LEG ---
        n17 = (1.2, 10.2)   # R common iliac end
        self.line(ax, [n13, n17], 'A42'); self.junc(ax, *n17)
        # R inner iliac stub
        self.line(ax, [n17, (0.6, 9.5)], 'A45'); self.term(ax, 0.6, 9.5)

        n18 = (1.5, 8.8)    # R external iliac end
        self.line(ax, [n17, n18], 'A44'); self.junc(ax, *n18)
        # R deep femoral stub
        self.line(ax, [n18, (2.2, 8.3)], 'A47'); self.term(ax, 2.2, 8.3)

        n19 = (1.7, 6.5)    # R femoral end (knee)
        self.line(ax, [n18, n19], 'A46'); self.junc(ax, *n19)

        # R posterior tibial
        self.line(ax, [n19, (1.5, 2.0)], 'A48'); self.term(ax, 1.5, 2.0)
        # R anterior tibial
        self.line(ax, [n19, (2.2, 2.0)], 'A49'); self.term(ax, 2.2, 2.0)

        # --- L LEG (mirror) ---
        n14 = (-1.2, 10.2)
        self.line(ax, [n13, n14], 'A43'); self.junc(ax, *n14)
        self.line(ax, [n14, (-0.6, 9.5)], 'A51'); self.term(ax, -0.6, 9.5)

        n15 = (-1.5, 8.8)
        self.line(ax, [n14, n15], 'A50'); self.junc(ax, *n15)
        self.line(ax, [n15, (-2.2, 8.3)], 'A53'); self.term(ax, -2.2, 8.3)

        n16 = (-1.7, 6.5)
        self.line(ax, [n15, n16], 'A52'); self.junc(ax, *n16)

        self.line(ax, [n16, (-1.5, 2.0)], 'A54'); self.term(ax, -1.5, 2.0)
        self.line(ax, [n16, (-2.2, 2.0)], 'A55'); self.term(ax, -2.2, 2.0)

    # ------------------------------------------------------------------
    # MAIN: assemble 3 subplots
    # ------------------------------------------------------------------
    def create(self, save_path=None):
        fig, axes = plt.subplots(3, 1, figsize=(8, 20),
                                 gridspec_kw={'height_ratios': [1.0, 1.0, 0.85],
                                              'hspace': 0.06})

        fig.suptitle(f'Arterial Network: {self.model_name}',
                     fontsize=15, fontweight='bold', y=0.995)

        # --- Subplot 1: HEAD ---
        ax1 = axes[0]
        ax1.set_xlim(-5.5, 5.5)
        ax1.set_ylim(-0.5, 9.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        self.draw_head(ax1)

        # --- Subplot 2: TORSO ---
        ax2 = axes[1]
        ax2.set_xlim(-6.5, 6.5)
        ax2.set_ylim(2.5, 12.2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        self.draw_torso(ax2)

        # --- Subplot 3: LEGS ---
        ax3 = axes[2]
        ax3.set_xlim(-3.5, 3.5)
        ax3.set_ylim(0.5, 12.0)
        ax3.set_aspect('equal')
        ax3.axis('off')
        self.draw_legs(ax3)

        # --- Shared legend at bottom ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red',  linewidth=2,   label='Forward flow'),
            Line2D([0], [0], color='blue', linewidth=2,   label='Reversed flow'),
            Line2D([0], [0], color='#bbbbbb', linewidth=1, label='Minimal flow'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Junction'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
                   markersize=5, linewidth=0, label='Terminal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, linewidth=0, label='Heart'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6,
                   fontsize=7.5, framealpha=0.95, bbox_to_anchor=(0.5, 0.005))

        plt.savefig(save_path, dpi=200, bbox_inches='tight') if save_path else None
        if save_path:
            print(f"[SUCCESS] 3-part tree saved: {save_path}")
        return fig


def main():
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir   = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING 3-SUBPLOT ARTERIAL TREE (paper style)")
    print("  Subplot 1: HEAD   | Subplot 2: TORSO+ARMS | Subplot 3: LEGS")
    print("=" * 70 + "\n")

    viz  = ThreePartTree('patient025_CoW_v2', results_base)
    path = output_dir / "arterial_tree_3part.png"
    viz.create(save_path=path)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()