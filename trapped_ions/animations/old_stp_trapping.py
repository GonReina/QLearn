"""
Manim animation: old STP cooling protocol (trapping)
====================================================
Left   - Bloch sphere projection (spin state)
Right  - Fock-state populations (motional state)
Bottom - Cycle/phase indicator

Run with:
    manim -pql old_stp_trapping.py OldSTPTrappingScene
    manim -pqh old_stp_trapping.py OldSTPTrappingScene
"""

from manim import *
import numpy as np

from qutip import (
    basis,
    destroy,
    sigmax,
    sigmay,
    sigmaz,
    sigmap,
    sigmam,
    tensor,
    qeye,
    liouvillian,
    operator_to_vector,
    vector_to_operator,
    expect,
    displace,
)

N_FOCK = 16
ETA = 0.05
OMEGA = 0.001
NU = 1.0
DELTA = 1.0
RF_STRENGTH = 1.0
GAMMA = 1000.0
N0 = 1
KT = 1.0
BETA = 0.01
N_CYCLES = 26
N_PULSE_FRAMES = 10
N_DISS_FRAMES = 4


def precompute_old_stp():
    a = destroy(N_FOCK)
    ad = a.dag()
    xop = a + ad

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    sx_full = tensor(sx, qeye(N_FOCK))
    sz_full = tensor(sz, qeye(N_FOCK))
    sp_full = tensor(sp, qeye(N_FOCK))
    sm_full = tensor(sm, qeye(N_FOCK))
    a_full = tensor(qeye(2), a)
    ad_full = tensor(qeye(2), ad)

    D_op = tensor(qeye(2), displace(N_FOCK, 1j * ETA))
    tau = 2 * np.pi / (ETA * OMEGA * np.sqrt(N0))

    H_pulse = (
        0.5 * DELTA * sz_full
        + NU * ad_full * a_full
        + 0.5 * OMEGA * (sp_full * D_op + sm_full * D_op.dag())
    )
    H_diss = DELTA * sz_full / 2 + NU * ad_full * a_full

    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
    W = W / np.sum(W)

    G = GAMMA / 2
    tg = 2 / G
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * ETA * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (sm_full * U_full))
    c_ops.append(np.sqrt(GAMMA) * sm_full)

    L_pulse = liouvillian(H_pulse, [])
    L_diss = liouvillian(H_diss, c_ops)

    prop_pulse_step = (L_pulse * (tau / N_PULSE_FRAMES)).expm()
    prop_diss_step = (L_diss * (tg / N_DISS_FRAMES)).expm()

    thermal_state = (-BETA * NU * ad * a / KT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    spin_down = basis(2, 1)
    rho0 = tensor(spin_down * spin_down.dag(), thermal_state)

    bloch = []
    fock_pops = []
    cycle_idx = []
    phase_label = []
    phase_frac = []

    rho_vec = operator_to_vector(rho0)

    def append_state_frame(rho, cyc, phase, frac):
        rho_spin = rho.ptrace(0)
        rho_mot = rho.ptrace(1)

        bx = float(expect(sx, rho_spin).real)
        by = float(expect(sy, rho_spin).real)
        bz = float(expect(sz, rho_spin).real)
        pops = np.real(np.array(rho_mot.diag()).flatten())

        bloch.append([bx, by, bz])
        fock_pops.append(pops)
        cycle_idx.append(cyc)
        phase_label.append(phase)
        phase_frac.append(frac)

    append_state_frame(vector_to_operator(rho_vec), 0, "start", 0.0)

    for k in range(1, N_CYCLES + 1):
        for j in range(1, N_PULSE_FRAMES + 1):
            rho_vec = prop_pulse_step * rho_vec
            append_state_frame(vector_to_operator(rho_vec), k, "pulse", j / N_PULSE_FRAMES)

        for j in range(1, N_DISS_FRAMES + 1):
            rho_vec = prop_diss_step * rho_vec
            append_state_frame(vector_to_operator(rho_vec), k, "diss", j / N_DISS_FRAMES)

    return (
        np.array(bloch),
        np.array(fock_pops),
        np.array(cycle_idx),
        np.array(phase_label),
        np.array(phase_frac),
    )


BLOCH_XYZ, FOCK_POPS, CYCLE_IDX, PHASE_LABEL, PHASE_FRAC = precompute_old_stp()
print(f"Old STP precompute done. Frames: {len(CYCLE_IDX)}")


class OldSTPTrappingScene(Scene):
    def construct(self):
        title = Text("Old STP protocol: repeated cooling cycles", font_size=34)
        subtitle = Text("Slow cycle-by-cycle trapping", font_size=24, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.15)
        head = VGroup(title, subtitle).to_edge(UP, buff=0.2)
        self.play(Write(title), FadeIn(subtitle, shift=0.1 * DOWN), run_time=1.3)

        bloch_center = np.array([-4.1, -0.4, 0.0])
        R = 1.5

        az = np.deg2rad(-35)
        el = np.deg2rad(22)

        def proj(vec3, scale=1.0):
            x, y, z = vec3
            x_cam = np.cos(az) * x - np.sin(az) * y
            y_cam = np.sin(el) * (np.sin(az) * x + np.cos(az) * y) + np.cos(el) * z
            return bloch_center + scale * (x_cam * RIGHT + y_cam * UP)

        sphere = Circle(radius=R, color=BLUE_C, stroke_opacity=0.65, stroke_width=1.5).move_to(bloch_center)
        equator = Ellipse(width=2 * R, height=0.60 * R, color=BLUE_D, stroke_opacity=0.35, stroke_width=1).rotate(-12 * DEGREES).move_to(bloch_center)
        meridian = Ellipse(width=0.90 * R, height=2 * R, color=BLUE_D, stroke_opacity=0.3, stroke_width=1).rotate(28 * DEGREES).move_to(bloch_center)

        axis_len = 1.2 * R
        x_axis = Arrow(
            proj([-1, 0, 0], axis_len),
            proj([1, 0, 0], axis_len),
            buff=0,
            stroke_width=1.5,
            color=RED_B,
            max_tip_length_to_length_ratio=0.08,
        )
        y_axis = Arrow(
            proj([0, -1, 0], axis_len),
            proj([0, 1, 0], axis_len),
            buff=0,
            stroke_width=1.5,
            color=PURPLE_B,
            max_tip_length_to_length_ratio=0.08,
        )
        y_axis.set_stroke(opacity=0.8)
        z_axis = Arrow(
            proj([0, 0, -1], axis_len),
            proj([0, 0, 1], axis_len),
            buff=0,
            stroke_width=1.5,
            color=GREEN_B,
            max_tip_length_to_length_ratio=0.08,
        )
        x_lbl = Text("x", font_size=18, color=RED_B).next_to(x_axis.get_end(), RIGHT, buff=0.05)
        y_lbl = Text("y", font_size=18, color=PURPLE_B).next_to(y_axis.get_end(), LEFT, buff=0.05)
        z_lbl = Text("z", font_size=18, color=GREEN_B).next_to(z_axis.get_end(), UP, buff=0.05)
        up_lbl = Text("|up>", font_size=16).next_to(bloch_center + R * UP, UP, buff=0.1)
        dn_lbl = Text("|dn>", font_size=16).next_to(bloch_center + R * DOWN, DOWN, buff=0.1)
        bloch_title = Text("Spin state", font_size=20).move_to(bloch_center + (R + 0.5) * UP)

        b0x, b0y, b0z = BLOCH_XYZ[0]
        tip0 = proj([b0x, b0y, b0z], R)
        spin_arrow = Arrow(
            bloch_center,
            tip0,
            buff=0,
            color=YELLOW,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15,
        )
        spin_tip = Dot(tip0, color=YELLOW, radius=0.04)
        spin_trace = TracedPath(spin_tip.get_center, stroke_color=YELLOW_A, stroke_width=2, stroke_opacity=0.55)

        left_group = VGroup(sphere, equator, meridian, x_axis, y_axis, z_axis, x_lbl, y_lbl, z_lbl, up_lbl, dn_lbl, bloch_title)

        n_plot = min(12, N_FOCK)
        bar_w = 0.23
        bar_sp = 0.33
        chart_left = 0.35
        chart_bot = -1.35
        max_bar_h = 2.5
        p_scale_max = max(0.12, float(np.max(FOCK_POPS[:, :n_plot])) * 1.08)

        x_line = Line([chart_left - 0.1, chart_bot, 0], [chart_left + n_plot * bar_sp + 0.1, chart_bot, 0], stroke_width=1.5)
        y_line = Line([chart_left - 0.1, chart_bot, 0], [chart_left - 0.1, chart_bot + max_bar_h + 0.3, 0], stroke_width=1.5)
        x_label = Text("n", font_size=18).next_to(x_line, DOWN, buff=0.1)
        y_label = Text("P(n)", font_size=18).next_to(y_line, LEFT, buff=0.1)
        chart_title = Text("Motional populations", font_size=20).move_to([chart_left + n_plot * bar_sp / 2, chart_bot + max_bar_h + 0.6, 0])

        y_tick_0 = Text("0", font_size=14).next_to([chart_left - 0.1, chart_bot, 0], LEFT, buff=0.05)
        y_tick_max = Text(f"{p_scale_max:.2f}", font_size=14).next_to(
            [chart_left - 0.1, chart_bot + max_bar_h, 0], LEFT, buff=0.05
        )

        ticks = VGroup()
        for n in range(0, n_plot, 2):
            cx = chart_left + n * bar_sp + bar_sp / 2
            ticks.add(Text(str(n), font_size=16).move_to([cx, chart_bot - 0.2, 0]))

        bars = VGroup()
        for n in range(n_plot):
            h = max(float(FOCK_POPS[0, n]) / p_scale_max * max_bar_h, 0.008)
            cx = chart_left + n * bar_sp + bar_sp / 2
            bar = Rectangle(
                width=bar_w,
                height=h,
                fill_color=interpolate_color(TEAL_B, GREEN_B, n / max(1, n_plot - 1)),
                fill_opacity=0.88,
                stroke_width=0.5,
            )
            bar.move_to([cx, chart_bot + h / 2, 0])
            bars.add(bar)

        right_top = VGroup(x_line, y_line, x_label, y_label, y_tick_0, y_tick_max, chart_title, ticks, bars)

        cycle_text = Text(f"Cycle 0 / {N_CYCLES}   |   start", font_size=24)
        cycle_text.to_edge(DOWN, buff=0.15)

        self.play(
            FadeIn(left_group),
            Create(spin_arrow),
            FadeIn(spin_tip),
            FadeIn(right_top),
            FadeIn(cycle_text),
            run_time=1.6,
        )
        self.add(spin_trace)

        progress = ValueTracker(0)
        n_frames = len(CYCLE_IDX) - 1

        def updater(mob, dt=None):
            idx = int(np.clip(progress.get_value(), 0, n_frames))

            bx, by, bz = BLOCH_XYZ[idx]
            tip = proj([bx, by, bz], R)
            new_arrow = Arrow(
                bloch_center,
                tip,
                buff=0,
                color=YELLOW,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.15,
            )
            spin_arrow.become(new_arrow)
            spin_tip.move_to(tip)

            for n in range(n_plot):
                h = max(float(FOCK_POPS[idx, n]) / p_scale_max * max_bar_h, 0.008)
                cx = chart_left + n * bar_sp + bar_sp / 2
                new_bar = Rectangle(
                    width=bar_w,
                    height=h,
                    fill_color=interpolate_color(TEAL_B, GREEN_B, n / max(1, n_plot - 1)),
                    fill_opacity=0.88,
                    stroke_width=0.5,
                )
                new_bar.move_to([cx, chart_bot + h / 2, 0])
                bars[n].become(new_bar)

            cyc = int(CYCLE_IDX[idx])
            phase = PHASE_LABEL[idx]
            frac = PHASE_FRAC[idx]
            if phase == "pulse":
                phase_str = f"pulse {int(100 * frac):02d}%"
            elif phase == "diss":
                phase_str = f"diss {int(100 * frac):02d}%"
            else:
                phase_str = "start"
            cycle_text.become(Text(f"Cycle {cyc} / {N_CYCLES}   |   {phase_str}", font_size=24).to_edge(DOWN, buff=0.15))

        spin_arrow.add_updater(updater)

        self.play(progress.animate.set_value(n_frames), run_time=34, rate_func=linear)
        spin_arrow.remove_updater(updater)

        end_note = Text("Repeated STP cycles cool and trap population in low n", font_size=24, color=YELLOW_B)
        end_note.to_edge(DOWN, buff=0.2)
        self.play(Transform(cycle_text, end_note), run_time=1.2)
        self.wait(2)
