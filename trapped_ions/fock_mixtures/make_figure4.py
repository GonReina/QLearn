# -*- coding: utf-8 -*-
"""
Rebuilds figure_3_plus_trace.pdf keeping only the two trace-distance panels
(previously panels A and B), stacked as a single column.

Reproduces the trace-distance computation of Fock_mixtures_sim.ipynb
(legacy STP vs new resonant sequence, with recoil dissipation).
Results are checkpointed in figure4_cache.pkl so the script can resume.

Run:  python3 make_figure4.py
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qutip import (destroy, sigmaz, sigmax, sigmap, sigmam, qeye, tensor,
                   basis, fock_dm, displace, expect, Qobj, num, liouvillian,
                   operator_to_vector, vector_to_operator, tracedist)

plt.rcParams.update({
    "text.usetex": False, "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "lines.linewidth": 1.4, "lines.markersize": 4,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
CACHEFILE = os.path.join(HERE, "figure4_cache.pkl")
try:
    with open(CACHEFILE, "rb") as fh:
        _CACHE = pickle.load(fh)
except Exception:
    _CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
        tmp = CACHEFILE + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(_CACHE, fh)
        os.replace(tmp, CACHEFILE)
    return _CACHE[key]


# --- Shared parameters (identical to Fock_mixtures_sim.ipynb) ---
N = 14
n0 = 1
eta_values = [0.02, 0.05, 0.08]
nu = 1.0
gamma = 1000
kT = 1.0
beta = 0.01
num_pulses = 30
RF_strength = 1
Omega_y = 1000
Delta_old = 1.0
Delta_new = 0.0
Omega_values = np.array([0.001, 0.005, 0.01, 0.1, 1.0, 2.0, 4.0, 8.0])
fock_projectors = [fock_dm(N, n) for n in range(N)]


def build_ops(N, eta_val):
    a = destroy(N)
    a_dag = a.dag()
    return {
        'xop': a + a_dag,
        's_z_full': tensor(sigmaz(), qeye(N)),
        's_plus_full': tensor(sigmap(), qeye(N)),
        's_minus_full': tensor(sigmam(), qeye(N)),
        's_x_full': tensor(sigmax(), qeye(N)),
        'a_full': tensor(qeye(2), a),
        'a_dag_full': tensor(qeye(2), a_dag),
        'D_op': tensor(qeye(2), displace(N, 1j * eta_val)),
    }


def build_c_ops(eta_val, xop, s_minus_full, gamma):
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal ** 2 + 1) / 4 / (2 * cosmax)
    W = W / np.sum(W)
    G = gamma / 2
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta_val * xop).expm()
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * tensor(qeye(2), U_mot)))
    c_ops.append(np.sqrt(gamma) * s_minus_full)
    return c_ops, 2 / G


def build_thermal_state(N, nu, beta, kT):
    a_th = destroy(N)
    rho_th = (-beta * nu * a_th.dag() * a_th / kT).expm()
    return rho_th / rho_th.tr()


thermal_state_ref = build_thermal_state(N, nu, beta, kT)
initial_probs = np.real(np.array(thermal_state_ref.diag())).flatten()

ideal_probs = np.zeros(N)
m = 0
while True:
    trap_idx = n0 * m ** 2
    if trap_idx >= N:
        break
    ideal_probs[trap_idx] = np.sum(initial_probs[trap_idx:min(n0 * (m + 1) ** 2, N)])
    m += 1


def to_diag_dm(probs):
    probs = np.clip(np.real(np.array(probs, dtype=float)).flatten(), 0.0, None)
    return Qobj(np.diag(probs / probs.sum()), dims=[[N], [N]])


rho_ideal = to_diag_dm(ideal_probs)


def final_pops_old_stp(Omega, eta_val):
    ops = build_ops(N, eta_val)
    tau = 2 * np.pi / (eta_val * Omega * np.sqrt(n0))
    H_pulse = (0.5 * Delta_old * ops['s_z_full'] + nu * ops['a_dag_full'] * ops['a_full']
               + 0.5 * Omega * (ops['s_plus_full'] * ops['D_op']
                                + ops['s_minus_full'] * ops['D_op'].dag()))
    H_diss = 0.5 * Delta_old * ops['s_z_full'] + nu * ops['a_dag_full'] * ops['a_full']
    c_ops, tg = build_c_ops(eta_val, ops['xop'], ops['s_minus_full'], gamma)
    prop_cycle = ((liouvillian(H_diss, c_ops) * tg).expm()
                  * (liouvillian(H_pulse, []) * tau).expm())
    spin_g = basis(2, 1)
    rho_vec = operator_to_vector(
        tensor(spin_g * spin_g.dag(), build_thermal_state(N, nu, beta, kT)))
    for _ in range(num_pulses):
        rho_vec = prop_cycle * rho_vec
    rho_motional = vector_to_operator(rho_vec).ptrace(1)
    return np.real(np.array(expect(fock_projectors, rho_motional))).flatten()


def final_pops_new_sequence(Omega, eta_val):
    ops = build_ops(N, eta_val)
    tau = 2 * np.pi / (eta_val * Omega * np.sqrt(n0))
    H_diss = 0.5 * Delta_new * ops['s_z_full'] + nu * ops['a_dag_full'] * ops['a_full']
    H_rf = 0.5 * RF_strength * (ops['a_full'] + ops['a_dag_full'])
    L_rf = liouvillian(H_rf, [])
    rf_duration = eta_val / RF_strength
    prop_rf = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()
    H_y = (0.5 * Delta_new * ops['s_z_full'] + nu * ops['a_dag_full'] * ops['a_full']
           + 0.5 * Omega_y * (-1j * ops['s_plus_full'] * ops['D_op']
                              + 1j * ops['s_minus_full'] * ops['D_op'].dag()))
    y_duration = np.pi / (2 * Omega_y)
    prop_y = (liouvillian(H_y, []) * y_duration).expm()
    prop_y_undo = (liouvillian(-H_y, []) * y_duration).expm()
    H_x = (0.5 * Delta_new * ops['s_x_full'] + nu * ops['a_dag_full'] * ops['a_full']
           + 0.5 * Omega * (ops['s_plus_full'] * ops['D_op']
                            + ops['s_minus_full'] * ops['D_op'].dag()))
    prop_x = (liouvillian(H_x, []) * tau).expm()
    c_ops, tg = build_c_ops(eta_val, ops['xop'], ops['s_minus_full'], gamma)
    prop_diss = (liouvillian(H_diss, c_ops) * tg).expm()
    prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf
    spin_g = basis(2, 1)
    rho_vec = operator_to_vector(
        tensor(spin_g * spin_g.dag(), build_thermal_state(N, nu, beta, kT)))
    for _ in range(num_pulses):
        rho_vec = prop_cycle * rho_vec
    rho_motional = vector_to_operator(rho_vec).ptrace(1)
    return np.real(np.array(expect(fock_projectors, rho_motional))).flatten()


def td_old(Om, eta_val):
    return float(tracedist(to_diag_dm(final_pops_old_stp(Om, eta_val)), rho_ideal))


def td_new(Om, eta_val):
    return float(tracedist(to_diag_dm(final_pops_new_sequence(Om, eta_val)), rho_ideal))


# --- Panel A data: old vs Omega for each eta; new at Omega = nu ---
trace_results = {}
for eta_val in eta_values:
    old = [cached(("old", float(Om), float(eta_val)), lambda: td_old(Om, eta_val))
           for Om in Omega_values]
    new_at_nu = cached(("new", 1.0, float(eta_val)), lambda: td_new(1.0, eta_val))
    trace_results[eta_val] = {'old': np.array(old), 'new_at_nu': new_at_nu}
    print(f"[fig 4] panel A eta={eta_val}: done")

# --- Panel B data: eta sweep at fixed Omega ---
omega_used_new = nu
omega_used_old = float(nu / 1000)
eta_sweep = np.linspace(0.02, 0.4, 16)
td_new_sweep = np.array(
    [cached(("new", omega_used_new, float(e)), lambda: td_new(omega_used_new, float(e)))
     for e in eta_sweep])
td_old_sweep = np.array(
    [cached(("old", omega_used_old, float(e)), lambda: td_old(omega_used_old, float(e)))
     for e in eta_sweep])
print("[fig 4] panel B sweep: done")

# --- Plot: two stacked panels ---
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.8))

eta_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(eta_values)))
omega_arr = np.asarray(Omega_values, dtype=float)
for c, eta_val in zip(eta_colors, eta_values):
    a1.plot(omega_arr, trace_results[eta_val]['old'], marker='o', linestyle='--',
            color=c, ms=4, label=rf"$\eta={eta_val:.2f}$")
    a1.plot([1.0], [trace_results[eta_val]['new_at_nu']], marker='x',
            linestyle='None', color=c, ms=8, mew=2.0)
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xlabel(r'$\Omega$ $[\nu]$')
a1.set_ylabel('trace distance to ideal trap')
a1.grid(True, which='both', alpha=0.3)
a1.set_ylim(8e-4, 8)
leg_eta = a1.legend(loc='upper left', frameon=False, ncol=3,
                    columnspacing=0.7, handlelength=1.5, handletextpad=0.4)
a1.add_artist(leg_eta)
a1.legend(handles=[
    Line2D([0], [0], color='k', lw=1.4, linestyle='--', marker='o', ms=4,
           label='old protocol'),
    Line2D([0], [0], color='k', lw=0, marker='x', ms=7, mew=2.0,
           label=r'new protocol at $\Omega=\nu$')],
    loc='lower left', frameon=False)
a1.text(0.0, 1.02, "A)", transform=a1.transAxes, va="bottom", fontweight="bold")

a2.plot(eta_sweep, td_old_sweep, marker='o', linestyle='--', color='tab:blue',
        ms=4, label=rf'old protocol ($\Omega={omega_used_old:g}\nu$)')
a2.plot(eta_sweep, td_new_sweep, marker='s', linestyle='-', color='tab:red',
        ms=4, label=r'new protocol ($\Omega=\nu$)')
a2.set_xlabel(r'Lamb-Dicke parameter $\eta$')
a2.set_ylabel('trace distance to ideal trap')
a2.set_yscale('log'); a2.grid(True, which='both', alpha=0.3)
a2.set_xlim(float(eta_sweep.min()), float(eta_sweep.max()))
a2.legend(loc='lower right', frameon=False)
a2.text(0.0, 1.02, "B)", transform=a2.transAxes, va="bottom", fontweight="bold")

fig.tight_layout()
out = os.path.join(FIGDIR, "figure_3_plus_trace.pdf")
fig.savefig(out)
plt.close(fig)
print(f"[fig 4] saved -> {out}")
