"""
Manim animation: single application of the pulse sequence (Bloch sphere)
======================================================================

Goal
----
Animate one *single* application of the pulse sequence used in the trapped-ion
cooling protocol, focusing on the spin dynamics on a Bloch sphere.

In particular, it makes the X-pulse segment explicit as a full 2π rotation by
showing a marker completing one full loop on the Bloch sphere during the X step.

Run with:
    manim -pql single_sequence_bloch.py SingleSequenceBlochScene
    manim -pqh single_sequence_bloch.py SingleSequenceBlochScene

Notes
-----
This is a geometric Bloch-sphere animation (no QuTiP solve). It is designed to
be slow/clear and pedagogical.
"""

from __future__ import annotations

from manim import *
import numpy as np

from qutip import (
    basis,
    destroy,
    displace,
    expect,
    liouvillian,
    operator_to_vector,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    sigmap,
    sigmam,
    tensor,
    vector_to_operator,
)


VIEW_AZIMUTH_DEG = 45.0
# Lower elevation makes the (s_x, s_y) plane read more "horizontal" in the 2-D projection.
VIEW_ELEVATION_DEG = 18.0

# Hilbert space / display
N_FOCK = 16
N_BARS = 8

# Physical parameters (match your provided Hamiltonians)
eta = 0.01
nu = 1.0
Delta = 0.0
Omega = nu
Omega_y = 100 * nu
n0 = 1
RF_strength = 4.0
gamma = 1000.0
beta = 0.1    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature

# Choose beta via a target mean phonon number to keep dynamics visible.
# This still uses your exact thermal-state formula: exp(-beta*nu*a^\dag a/kT).
N_THERMAL = 2.0

# Precompute resolution (more frames = slower + smoother)
N_RF_FRAMES = 80
N_Y_FRAMES = 80
N_X_FRAMES = 240
N_Y_UNDO_FRAMES = 80
N_DISS_FRAMES = 120
N_RF_UNDO_FRAMES = 80

# Segment run times in the final Manim animation
T_RF = 4.0
T_Y = 4.0
T_X = 12.0
T_Y_UNDO = 4.0
T_DISS = 6.0
T_RF_UNDO = 4.0

# Screen-space placement of the Bloch sphere (edit here)
BLOCH_CENTER = np.array([-3.6, -0.35, 0.0])


def build_c_ops(eta_val, xop, s_minus_full, gamma_val, N):
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
    W = W / np.sum(W)

    G = gamma_val / 2
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta_val * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))

    c_ops.append(np.sqrt(gamma_val) * s_minus_full)
    return c_ops


def precompute_single_cycle():
    a = destroy(N_FOCK)
    a_dag = a.dag()
    xop = a + a_dag

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    s_x_full = tensor(sx, qeye(N_FOCK))
    s_y_full = tensor(sy, qeye(N_FOCK))
    s_z_full = tensor(sz, qeye(N_FOCK))
    s_plus_full = tensor(sp, qeye(N_FOCK))
    s_minus_full = tensor(sm, qeye(N_FOCK))
    a_full = tensor(qeye(2), a)
    a_dag_full = tensor(qeye(2), a_dag)

    D_op = tensor(qeye(2), displace(N_FOCK, 1j * eta))

    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))

    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    H_rf = (RF_strength / 2) * (a_full + a_dag_full)
    rf_duration = 10 * eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf_step = (L_rf * (rf_duration / N_RF_FRAMES)).expm()
    prop_rf_undo_step = (-L_rf * (rf_duration / N_RF_UNDO_FRAMES)).expm()

    H_y = (
        0.5 * Delta * s_z_full
        + nu * a_dag_full * a_full
        + 0.5
        * Omega_y
        * (-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag())
    )
    y_duration = np.pi / (2 * Omega_y)
    prop_y_step = (liouvillian(H_y, []) * (y_duration / N_Y_FRAMES)).expm()
    prop_y_undo_step = (liouvillian(-H_y, []) * (y_duration / N_Y_UNDO_FRAMES)).expm()

    H_x = (
        0.5 * nu * s_x_full
        + nu * a_dag_full * a_full
        + 0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full)
    )
    prop_x_step = (liouvillian(H_x, []) * (tau / N_X_FRAMES)).expm()

    c_ops = build_c_ops(eta, xop, s_minus_full, gamma, N_FOCK)
    G = gamma / 2
    tg = 2 / G
    prop_diss_step = (liouvillian(H_diss, c_ops) * (tg / N_DISS_FRAMES)).expm()

    # Initial state: spin ground (|↓>) tensor motional thermal
    thermal_state = (-beta * nu * a_dag * a / kT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    ground_state = basis(2, 1)
    initial_state = tensor(ground_state * ground_state.dag(), thermal_state)
    rho_vec = operator_to_vector(initial_state)

    frames_bloch = []
    frames_fock = []
    frames_purity = []

    def append_frame(rho_v):
        rho = vector_to_operator(rho_v)
        rho_spin = rho.ptrace(0)
        rho_mot = rho.ptrace(1)
        bx = float(expect(sx, rho_spin).real)
        by = float(expect(sy, rho_spin).real)
        bz = float(expect(sz, rho_spin).real)
        purity = float((rho_spin * rho_spin).tr().real)
        pops = np.real(np.array(rho_mot.diag()).flatten())
        frames_bloch.append([bx, by, bz])
        frames_fock.append(pops)
        frames_purity.append(purity)

    append_frame(rho_vec)

    for _ in range(N_RF_FRAMES):
        rho_vec = prop_rf_step * rho_vec
        append_frame(rho_vec)
    for _ in range(N_Y_FRAMES):
        rho_vec = prop_y_step * rho_vec
        append_frame(rho_vec)
    for _ in range(N_X_FRAMES):
        rho_vec = prop_x_step * rho_vec
        append_frame(rho_vec)
    for _ in range(N_Y_UNDO_FRAMES):
        rho_vec = prop_y_undo_step * rho_vec
        append_frame(rho_vec)
    for _ in range(N_DISS_FRAMES):
        rho_vec = prop_diss_step * rho_vec
        append_frame(rho_vec)
    for _ in range(N_RF_UNDO_FRAMES):
        rho_vec = prop_rf_undo_step * rho_vec
        append_frame(rho_vec)

    return np.array(frames_bloch), np.array(frames_fock), np.array(frames_purity)


def rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


class SingleSequenceBlochScene(Scene):
    def construct(self):
        title = Text("Single pulse-sequence application", font_size=34)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=1.0)

        bloch_xyz, fock_pops, spin_purity = precompute_single_cycle()
        n_frames = len(bloch_xyz)

        # Layout: keep Bloch sphere left so we can place the motional bar chart on the right.
        bloch_center = np.array(BLOCH_CENTER, dtype=float)
        R = 2.0

        az = np.deg2rad(VIEW_AZIMUTH_DEG)
        el = np.deg2rad(VIEW_ELEVATION_DEG)

        def proj(vec3: np.ndarray, scale: float = 1.0) -> np.ndarray:
            x, y, z = [float(v) for v in vec3]
            x_cam = np.cos(az) * x - np.sin(az) * y
            y_cam = np.sin(el) * (np.sin(az) * x + np.cos(az) * y) + np.cos(el) * z
            return bloch_center + scale * (x_cam * RIGHT + y_cam * UP)

        # Sphere outline + guide ellipses
        sphere = Circle(radius=R, color=BLUE_C, stroke_opacity=0.65, stroke_width=1.6).move_to(
            bloch_center
        )
        # (Removed) guide ellipses (equator/meridian) for a cleaner look.

        axis_len = 1.18 * R
        x_axis = Arrow(
            proj(np.array([1.0, 0.0, 0.0]), axis_len),
            proj(np.array([-1.0, 0.0, 0.0]), axis_len),
            buff=0,
            stroke_width=1.6,
            color=RED_B,
            max_tip_length_to_length_ratio=0.08,
        )
        y_axis = Arrow(
            proj(np.array([0.0, 1.0, 0.0]), axis_len),
            proj(np.array([0.0, -1.0, 0.0]), axis_len),
            buff=0,
            stroke_width=1.6,
            color=PURPLE_B,
            max_tip_length_to_length_ratio=0.08,
        )
        z_axis = Arrow(
            proj(np.array([0.0, 0.0, -1.0]), axis_len),
            proj(np.array([0.0, 0.0, 1.0]), axis_len),
            buff=0,
            stroke_width=1.6,
            color=GREEN_B,
            max_tip_length_to_length_ratio=0.08,
        )
        x_lbl = MathTex(r"s_x", font_size=42, color=RED_B).next_to(x_axis.get_end(), LEFT, buff=0.22)
        y_lbl = MathTex(r"s_y", font_size=42, color=PURPLE_B).next_to(y_axis.get_end(), RIGHT, buff=0.22)
        z_lbl = MathTex(r"s_z", font_size=42, color=GREEN_B).next_to(z_axis.get_end(), RIGHT, buff=0.22)

        bloch_title = MathTex(
            r"\text{Reduced\ spin\ state}\; \rho_{\mathrm{spin}}= Tr_{mot} \{\rho\}",
            font_size=28,
        ).move_to(bloch_center + (R + 1.35) * UP)

        self.play(
            FadeIn(sphere),
            FadeIn(VGroup(x_axis, y_axis, z_axis, x_lbl, y_lbl, z_lbl, bloch_title)),
            run_time=1.0,
        )

        # -----------------------------------------------------------------
        # RIGHT PANEL — Motional state bar chart (Fock populations)
        # -----------------------------------------------------------------
        chart_left = 1.05
        chart_bot = -1.0
        bar_w = 0.26
        bar_sp = 0.40
        max_bar_h = 2.2

        bar_x_axis = Line(
            [chart_left - 0.12, chart_bot, 0],
            [chart_left + N_BARS * bar_sp + 0.12, chart_bot, 0],
            stroke_width=1.6,
        )
        bar_y_axis = Line(
            [chart_left - 0.12, chart_bot, 0],
            [chart_left - 0.12, chart_bot + max_bar_h + 0.35, 0],
            stroke_width=1.6,
        )
        bar_x_lbl = Text("n", font_size=18).next_to(bar_x_axis, DOWN, buff=0.36)
        bar_y_lbl = Text("P(n)", font_size=18).next_to(bar_y_axis, LEFT, buff=0.1)
        bar_title = Text("Motional state", font_size=20).move_to(
            [chart_left + N_BARS * bar_sp / 2, chart_bot + max_bar_h + 0.75, 0]
        )

        purity_label = Text("Spin purity:", font_size=24, color=YELLOW_A)
        purity_value = DecimalNumber(1.00, num_decimal_places=3, font_size=24, color=YELLOW_A)
        purity_value.add_updater(
            lambda m: m.set_value(float(spin_purity[int(np.clip(progress.get_value(), 0, n_frames - 1))]))
        )
        purity_group = VGroup(purity_label, purity_value).arrange(RIGHT, buff=0.18)
        purity_group.next_to(bloch_title, DOWN, buff=0.14)

        bar_num_labels = VGroup()
        for n in range(N_BARS):
            cx = chart_left + n * bar_sp + bar_sp / 2
            bar_num_labels.add(Text(str(n), font_size=16).move_to([cx, chart_bot - 0.22, 0]))

        # Progress tracker (frame index)
        progress = ValueTracker(0)

        # Initialize bars from frame 0
        init_pops = fock_pops[0, :N_BARS].astype(float)
        s0 = float(np.sum(init_pops))
        if s0 > 0:
            init_pops /= s0
        bars = VGroup()
        for n in range(N_BARS):
            h = max(float(init_pops[n]) * max_bar_h, 0.01)
            cx = chart_left + n * bar_sp + bar_sp / 2
            bar = Rectangle(
                width=bar_w,
                height=h,
                fill_color=interpolate_color(TEAL_B, GREEN_B, n / max(1, N_BARS - 1)),
                fill_opacity=0.85,
                stroke_width=0.6,
            )
            bar.move_to([cx, chart_bot + h / 2, 0])
            bars.add(bar)

        def bars_updater(mob: VGroup, dt: float):
            idx = int(np.clip(progress.get_value(), 0, n_frames - 1))
            pops = fock_pops[idx, :N_BARS].astype(float)
            s = float(np.sum(pops))
            if s > 0:
                pops /= s
            for n in range(N_BARS):
                h = max(float(pops[n]) * max_bar_h, 0.005)
                cx = chart_left + n * bar_sp + bar_sp / 2
                new_bar = Rectangle(
                    width=bar_w,
                    height=h,
                    fill_color=interpolate_color(TEAL_B, GREEN_B, n / max(1, N_BARS - 1)),
                    fill_opacity=0.85,
                    stroke_width=0.6,
                )
                new_bar.move_to([cx, chart_bot + h / 2, 0])
                mob[n].become(new_bar)

        bars.add_updater(bars_updater)

        bar_chart_group = VGroup(
            bar_x_axis,
            bar_y_axis,
            bar_x_lbl,
            bar_y_lbl,
            bar_title,
            bar_num_labels,
            bars,
        )

        self.play(FadeIn(bar_chart_group), FadeIn(purity_group), run_time=1.0)

        # -----------------------------------------------------------------
        # State arrow driven by the precomputed Bloch vector.
        # -----------------------------------------------------------------
        def current_bloch_vec() -> np.ndarray:
            idx = int(np.clip(progress.get_value(), 0, n_frames - 1))
            return np.array(bloch_xyz[idx], dtype=float)

        state_arrow = always_redraw(
            lambda: Arrow(
                bloch_center,
                proj(current_bloch_vec(), R),
                buff=0,
                color=YELLOW,
                stroke_width=5,
                max_tip_length_to_length_ratio=0.15,
            )
        )
        tip_dot = always_redraw(lambda: Dot(point=proj(current_bloch_vec(), R), radius=0.04, color=YELLOW))
        trace = TracedPath(tip_dot.get_center, stroke_color=YELLOW_A, stroke_width=2.2, stroke_opacity=0.45)

        self.add(trace)
        self.play(Create(state_arrow), FadeIn(tip_dot), run_time=0.8)

        # -----------------------------------------------------------------
        # Step label (bottom) and X-pulse 2π marker on a rotation ring.
        # -----------------------------------------------------------------
        step_text = Text("Step: RF displacement (motion only)", font_size=22)
        step_text.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(step_text), run_time=0.6)

        # Ring showing the path of a point under rotation about the x-axis.
        # We draw the x=0 great circle in the y-z plane and animate a marker.
        theta = ValueTracker(0.0)

        def ring_points(num: int = 160) -> list[np.ndarray]:
            pts = []
            for th in np.linspace(0, 2 * np.pi, num=num):
                v = np.array([0.0, np.cos(th), np.sin(th)], dtype=float)
                pts.append(proj(v, R))
            return pts

        x_ring = VMobject(stroke_color=TEAL_A, stroke_opacity=0.45, stroke_width=3)
        x_ring.set_points_smoothly(ring_points())

        x_marker = always_redraw(
            lambda: Dot(
                point=proj(
                    np.array([0.0, np.cos(theta.get_value()), np.sin(theta.get_value())], dtype=float),
                    R,
                ),
                radius=0.045,
                color=TEAL_B,
            )
        )

        # Progress label for the X segment: θ/(2π) from 0 → 1
        frac_label = Text("X pulse progress: θ/(2π) =", font_size=20, color=TEAL_A)
        frac_value = DecimalNumber(0.00, num_decimal_places=2, font_size=20, color=TEAL_A)
        frac_value.add_updater(lambda m: m.set_value(theta.get_value() / (2 * np.pi)))
        frac_group = VGroup(frac_label, frac_value).arrange(RIGHT, buff=0.15)
        frac_group.next_to(step_text, UP, buff=0.2)

        # ----------------------------------------------------------
        # 1) RF displacement (spin unchanged)
        # ----------------------------------------------------------
        rf_end = N_RF_FRAMES
        self.play(progress.animate.set_value(rf_end), run_time=T_RF, rate_func=linear)

        # ----------------------------------------------------------
        # 2) Y(π/2) pulse — rotate about y axis
        # ----------------------------------------------------------
        new_step = Text("Step: Y(π/2) pulse", font_size=22).move_to(step_text)
        self.play(Transform(step_text, new_step), run_time=0.45)
        y_end = rf_end + N_Y_FRAMES
        self.play(progress.animate.set_value(y_end), run_time=T_Y, rate_func=linear)

        # ----------------------------------------------------------
        # 3) X pulse — show an explicit 2π rotation marker
        # ----------------------------------------------------------
        new_step = Text("Step: X pulse (2π rotation)", font_size=22).move_to(step_text)
        self.play(Transform(step_text, new_step), FadeIn(x_ring), FadeIn(x_marker), FadeIn(frac_group), run_time=0.7)
        x_end = y_end + N_X_FRAMES
        self.play(progress.animate.set_value(x_end), theta.animate.set_value(2 * np.pi), run_time=T_X, rate_func=linear)
        self.play(FadeOut(frac_group, run_time=0.4), FadeOut(x_marker, run_time=0.4), FadeOut(x_ring, run_time=0.4))

        # ----------------------------------------------------------
        # 4) Undo Y(π/2)
        # ----------------------------------------------------------
        new_step = Text("Step: Undo Y(π/2)", font_size=22).move_to(step_text)
        self.play(Transform(step_text, new_step), run_time=0.45)
        y_undo_end = x_end + N_Y_UNDO_FRAMES
        self.play(progress.animate.set_value(y_undo_end), run_time=T_Y_UNDO, rate_func=linear)

        # ----------------------------------------------------------
        # 5) Relaxation to |dn> (already there in this cartoon)
        # ----------------------------------------------------------
        new_step = Text("Step: Relaxation (pump to |dn>)", font_size=22).move_to(step_text)
        self.play(Transform(step_text, new_step), run_time=0.45)
        diss_end = y_undo_end + N_DISS_FRAMES
        self.play(progress.animate.set_value(diss_end), run_time=T_DISS, rate_func=linear)

        # ----------------------------------------------------------
        # 6) Undo RF displacement (motion only)
        # ----------------------------------------------------------
        new_step = Text("Step: Undo RF displacement (motion only)", font_size=22).move_to(step_text)
        self.play(Transform(step_text, new_step), run_time=0.45)
        end_all = diss_end + N_RF_UNDO_FRAMES
        self.play(progress.animate.set_value(end_all), run_time=T_RF_UNDO, rate_func=linear)

        bars.remove_updater(bars_updater)

        self.wait(1.0)
