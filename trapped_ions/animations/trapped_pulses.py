"""
Manim animation: pi-pulse on a trapped ion
===========================================
Left   - Bloch sphere (2-D projection) with state vector
Right  - Fock-state bar chart  +  Phase-space Wigner function

Run with:
    manim -pql trapped_pulses.py PiPulseScene
    manim -pqh trapped_pulses.py PiPulseScene   (higher quality)
"""

from manim import *
import numpy as np
from PIL import Image
import tempfile, os

# ---------- QuTiP imports (pre-compute physics) ----------
from qutip import (
    basis, destroy, sigmax, sigmay, sigmaz, sigmap, sigmam,
    tensor, qeye, mesolve, expect, wigner, displace,
)

# =====================================================================
#  Physics parameters
# =====================================================================
N_FOCK = 15          # Fock-space truncation
ETA = 0.15           # Lamb-Dicke parameter
OMEGA = 1.0          # Rabi frequency (sets the time-scale)
NU = 1.0             # Motional frequency
N_THERMAL = 2        # initial mean phonon number

# pi-pulse duration for a carrier transition
T_PI = np.pi / OMEGA

# Number of time-steps for smooth animation
N_STEPS = 100

# =====================================================================
#  Pre-compute the time evolution with QuTiP
# =====================================================================
a   = destroy(N_FOCK)
ad  = a.dag()
n_op = ad * a

sx = sigmax()
sy = sigmay()
sz = sigmaz()
sp = sigmap()
sm = sigmam()

# full (spin x motion) operators
sz_full = tensor(sz, qeye(N_FOCK))
sx_full = tensor(sx, qeye(N_FOCK))
sy_full = tensor(sy, qeye(N_FOCK))
sp_full = tensor(sp, qeye(N_FOCK))
sm_full = tensor(sm, qeye(N_FOCK))
a_full  = tensor(qeye(2), a)
ad_full = tensor(qeye(2), ad)

# Displacement operator
D_op = tensor(qeye(2), displace(N_FOCK, 1j * ETA))

# Hamiltonian:  H = (Omega/2)[ sigma+ D(i*eta) + sigma- D^dag(i*eta) ]
H = 0.5 * OMEGA * (sp_full * D_op + sm_full * D_op.dag())

# Initial state: spin-down (|1>) tensor coherent motional state |alpha>
alpha0 = 1.0 + 0.0j
coh_state = displace(N_FOCK, alpha0) * basis(N_FOCK, 0)
spin_down = basis(2, 1)  # |down>
psi0 = tensor(spin_down, coh_state)

# Solve
tlist = np.linspace(0, T_PI, N_STEPS)
result = mesolve(H, psi0, tlist, [], [])

# Extract observables
bloch_xyz = np.zeros((N_STEPS, 3))
fock_pops = np.zeros((N_STEPS, N_FOCK))
wigner_data = []
xvec = np.linspace(-4, 4, 80)

for i in range(N_STEPS):
    psi_t = result.states[i]
    rho_spin = psi_t.ptrace(0)
    rho_mot  = psi_t.ptrace(1)

    bloch_xyz[i, 0] = float(expect(sx, rho_spin).real)
    bloch_xyz[i, 1] = float(expect(sy, rho_spin).real)
    bloch_xyz[i, 2] = float(expect(sz, rho_spin).real)

    fock_pops[i, :] = np.real(np.array(rho_mot.diag()).flatten())

    W = wigner(rho_mot, xvec, xvec)
    wigner_data.append(np.array(W))

wigner_data = np.array(wigner_data)
W_max = np.max(np.abs(wigner_data)) * 1.05

print(f"Pre-computation done.  Bloch z: {bloch_xyz[0,2]:.2f} -> {bloch_xyz[-1,2]:.2f}")


# =====================================================================
#  Helper: save Wigner as temp PNG and return ImageMobject
# =====================================================================
_tmp_dir = tempfile.mkdtemp()

def wigner_to_mobject(W, W_max, size=2.4):
    """Create an ImageMobject from a 2D Wigner array via a temp PNG."""
    W_norm = np.clip(W / W_max, -1, 1)
    pos = np.clip(W_norm, 0, 1)
    neg = np.clip(-W_norm, 0, 1)

    r = (255 * (1 - neg)).astype(np.uint8)
    g = np.clip(255 * (1 - pos - neg), 0, 255).astype(np.uint8)
    b = (255 * (1 - pos)).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb[::-1, :, :]  # flip so p increases upward

    path = os.path.join(_tmp_dir, "wig.png")
    Image.fromarray(rgb).save(path)
    mob = ImageMobject(path)
    mob.set_width(size).set_height(size)
    return mob


# =====================================================================
#  Manim Scene  (pure 2-D — no ThreeDScene)
# =====================================================================
class PiPulseScene(Scene):
    """
    Three-panel animation of a pi-pulse on a trapped ion.
    Left:   Bloch sphere (2-D projection with x-z view)
    Top-R:  Fock-state bar chart
    Bot-R:  Phase-space Wigner function
    """

    def construct(self):
        # ----------------------------------------------------------
        #  Title
        # ----------------------------------------------------------
        title = Text("Pi-pulse on a trapped ion", font_size=36)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=1)

        # ----------------------------------------------------------
        #  LEFT PANEL — Bloch sphere (2-D projected view: x horizontal, z vertical)
        # ----------------------------------------------------------
        bloch_center = np.array([-3.8, -0.5, 0.0])
        R = 1.6  # sphere radius on screen

        # Sphere outline
        sphere_outline = Circle(radius=R, color=BLUE_C, stroke_width=1.5,
                                stroke_opacity=0.6)
        sphere_outline.move_to(bloch_center)

        # Equator ellipse (tilted perspective hint)
        equator = Ellipse(width=2*R, height=0.5*R, color=BLUE_D,
                          stroke_width=1, stroke_opacity=0.35)
        equator.move_to(bloch_center)

        # Meridian ellipse (x-z plane — front view, so it's a full circle outline)
        meridian = Ellipse(width=0.5*R, height=2*R, color=BLUE_D,
                           stroke_width=1, stroke_opacity=0.35)
        meridian.move_to(bloch_center)

        # Axes: x right, z up (y goes into screen — we don't draw it)
        ax_len = R * 1.2
        x_axis = Arrow(bloch_center - ax_len * RIGHT, bloch_center + ax_len * RIGHT,
                        buff=0, stroke_width=1.5, color=RED_B, max_tip_length_to_length_ratio=0.08)
        z_axis = Arrow(bloch_center - ax_len * UP, bloch_center + ax_len * UP,
                        buff=0, stroke_width=1.5, color=GREEN_B, max_tip_length_to_length_ratio=0.08)

        x_lbl = Text("x", font_size=18, color=RED_B).next_to(x_axis.get_end(), RIGHT, buff=0.08)
        z_lbl = Text("z", font_size=18, color=GREEN_B).next_to(z_axis.get_end(), UP, buff=0.08)

        # Pole labels
        up_lbl = Text("|up>", font_size=16, color=GREEN_A).next_to(
            bloch_center + R * UP, UP, buff=0.12)
        dn_lbl = Text("|dn>", font_size=16, color=GREEN_A).next_to(
            bloch_center + R * DOWN, DOWN, buff=0.12)

        # State arrow — 2-D projection: x -> right, z -> up
        bx0, by0, bz0 = bloch_xyz[0]
        arrow_tip = bloch_center + R * np.array([bx0, bz0, 0])
        state_arrow = Arrow(bloch_center, arrow_tip, buff=0,
                             color=YELLOW, stroke_width=4,
                             max_tip_length_to_length_ratio=0.15)

        # Small dot at tip for trace
        tip_dot = Dot(point=arrow_tip, radius=0.04, color=YELLOW)

        # Traced path that follows the dot
        trace = TracedPath(tip_dot.get_center, stroke_color=YELLOW_A,
                           stroke_width=2, stroke_opacity=0.5)

        bloch_label = Text("Bloch sphere", font_size=20).move_to(
            bloch_center + (R + 0.55) * UP)

        bloch_group = VGroup(sphere_outline, equator, meridian,
                              x_axis, z_axis, x_lbl, z_lbl,
                              up_lbl, dn_lbl, bloch_label)

        # ----------------------------------------------------------
        #  TOP-RIGHT PANEL — Fock-state bar chart
        # ----------------------------------------------------------
        N_BARS = min(8, N_FOCK)
        bar_w = 0.25
        bar_sp = 0.36
        chart_left = 0.6
        chart_bot = 0.0
        max_bar_h = 2.0

        # Axes
        bar_x_axis = Line([chart_left - 0.1, chart_bot, 0],
                           [chart_left + N_BARS * bar_sp + 0.1, chart_bot, 0],
                           stroke_width=1.5)
        bar_y_axis = Line([chart_left - 0.1, chart_bot, 0],
                           [chart_left - 0.1, chart_bot + max_bar_h + 0.3, 0],
                           stroke_width=1.5)
        bar_x_lbl = Text("n", font_size=18).next_to(bar_x_axis, DOWN, buff=0.1)
        bar_y_lbl = Text("P(n)", font_size=18).next_to(bar_y_axis, LEFT, buff=0.1)
        bar_title = Text("Motional state", font_size=20).move_to(
            [chart_left + N_BARS * bar_sp / 2, chart_bot + max_bar_h + 0.6, 0])

        # Number labels
        bar_num_labels = VGroup()
        for n in range(N_BARS):
            cx = chart_left + n * bar_sp + bar_sp / 2
            lbl = Text(str(n), font_size=16).move_to([cx, chart_bot - 0.2, 0])
            bar_num_labels.add(lbl)

        # Bars (initial)
        bars = VGroup()
        for n in range(N_BARS):
            h = max(float(fock_pops[0, n]) * max_bar_h, 0.01)
            cx = chart_left + n * bar_sp + bar_sp / 2
            bar = Rectangle(width=bar_w, height=h,
                             fill_color=interpolate_color(TEAL_B, GREEN_B, n / N_BARS),
                             fill_opacity=0.85, stroke_width=0.5)
            bar.move_to([cx, chart_bot + h / 2, 0])
            bars.add(bar)

        bar_chart_group = VGroup(bar_x_axis, bar_y_axis, bar_x_lbl, bar_y_lbl,
                                  bar_title, bar_num_labels, bars)

        # ----------------------------------------------------------
        #  BOTTOM-RIGHT PANEL — Wigner function
        # ----------------------------------------------------------
        wig_size = 2.4
        wig_center = np.array([chart_left + N_BARS * bar_sp / 2, -2.2, 0])

        wig_mob = wigner_to_mobject(wigner_data[0], W_max, wig_size)
        wig_mob.move_to(wig_center)

        wig_border = SurroundingRectangle(wig_mob, buff=0, color=WHITE, stroke_width=1)
        wig_title = Text("Phase space (Wigner)", font_size=20).next_to(
            wig_mob, UP, buff=0.2)
        wig_x_lbl = Text("x", font_size=16).next_to(wig_mob, DOWN, buff=0.1)
        wig_p_lbl = Text("p", font_size=16).next_to(wig_mob, LEFT, buff=0.1)

        wig_group = VGroup(wig_border, wig_title, wig_x_lbl, wig_p_lbl)

        # ----------------------------------------------------------
        #  Appear everything
        # ----------------------------------------------------------
        self.play(
            FadeIn(bloch_group),
            Create(state_arrow),
            FadeIn(tip_dot),
            FadeIn(bar_chart_group),
            FadeIn(wig_mob), FadeIn(wig_group),
            run_time=1.5,
        )
        self.add(trace)

        # Time counter
        time_text = Text(f"t = 0.00 / {T_PI:.2f}", font_size=20).to_edge(DOWN, buff=0.2)
        self.add(time_text)

        # ----------------------------------------------------------
        #  Animate the pi-pulse
        # ----------------------------------------------------------
        n_anim = N_STEPS - 1
        progress = ValueTracker(0)

        def updater(mob, dt=None):
            idx = int(np.clip(progress.get_value(), 0, n_anim))

            # --- Bloch arrow ---
            bx, by, bz = bloch_xyz[idx]
            new_tip = bloch_center + R * np.array([bx, bz, 0])
            new_arrow = Arrow(bloch_center, new_tip, buff=0,
                               color=YELLOW, stroke_width=4,
                               max_tip_length_to_length_ratio=0.15)
            state_arrow.become(new_arrow)
            tip_dot.move_to(new_tip)

            # --- Bars ---
            for n in range(N_BARS):
                h = max(float(fock_pops[idx, n]) * max_bar_h, 0.005)
                cx = chart_left + n * bar_sp + bar_sp / 2
                new_bar = Rectangle(width=bar_w, height=h,
                                     fill_color=interpolate_color(TEAL_B, GREEN_B,
                                                                   n / N_BARS),
                                     fill_opacity=0.85, stroke_width=0.5)
                new_bar.move_to([cx, chart_bot + h / 2, 0])
                bars[n].become(new_bar)

            # --- Wigner ---
            new_wig = wigner_to_mobject(wigner_data[idx], W_max, wig_size)
            new_wig.move_to(wig_center)
            wig_mob.become(new_wig)

            # --- Time ---
            t_val = tlist[idx]
            new_tt = Text(f"t = {t_val:.2f} / {T_PI:.2f}",
                          font_size=20).to_edge(DOWN, buff=0.2)
            time_text.become(new_tt)

        state_arrow.add_updater(updater)

        self.play(
            progress.animate.set_value(n_anim),
            run_time=6,
            rate_func=linear,
        )
        state_arrow.remove_updater(updater)

        # Hold final frame
        self.wait(2)
