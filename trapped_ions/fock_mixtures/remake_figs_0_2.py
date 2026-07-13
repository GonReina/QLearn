# -*- coding: utf-8 -*-
"""
Regenerates figure_0.pdf (paper Fig. 1, Wigner comparison) and
figure_2.pdf (paper Fig. 3, population scan) with fonts enlarged by 2 pt,
reproducing cells 16-17 of Fock_mixtures_sim.ipynb.
Simulation results are checkpointed in figs02_cache.pkl.

Run:  python3 remake_figs_0_2.py
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qutip import (destroy, sigmaz, sigmax, sigmay, sigmap, sigmam, qeye,
                   tensor, basis, displace, liouvillian, operator_to_vector,
                   vector_to_operator, Qobj, squeeze, ket2dm, expect, wigner)

# Computer-Modern mathtext (stands in for the notebook's text.usetex=True)
plt.rcParams.update({
    'text.usetex': False, 'mathtext.fontset': 'cm', 'font.family': 'serif',
    'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22,
    'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 20,
    'savefig.bbox': 'tight',
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE)
CACHEFILE = os.path.join(HERE, "figs02_cache.pkl")
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


# --- parameters (identical to notebook cell) ---
Delta, RF_strength, nu = 0, 1, 1.0
Omega, Omega_y = 1.0, 100
n0, gamma, kT, N, num_pulses = 1, 1000, 100, 14, 30
eta_values = [0.05, 0.008, 0.1, 0.2]


def run_occupancies(eta):
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))
    a = destroy(N); a_dag = a.dag(); xop = a + a_dag
    s_z = tensor(sigmaz(), qeye(N))
    s_p = tensor(sigmap(), qeye(N)); s_m = tensor(sigmam(), qeye(N))
    s_x = tensor(sigmax(), qeye(N))
    A = tensor(qeye(2), a); Ad = tensor(qeye(2), a_dag)
    D_op = tensor(qeye(2), displace(N, 1j * eta))

    H_diss = Delta * s_z / 2 + nu * Ad * A
    H_rf = (RF_strength / 2) * (A + Ad)
    rf_duration = eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()

    H_y = 0.5 * Delta * s_z + nu * Ad * A + \
        0.5 * Omega_y * (-1j * s_p * D_op + 1j * s_m * D_op.dag())
    y_duration = np.pi / (2 * Omega_y)
    prop_y = (liouvillian(H_y, []) * y_duration).expm()
    prop_y_undo = (liouvillian(-H_y, []) * y_duration).expm()

    H_x = 0.5 * Delta * s_x + nu * Ad * A + \
        0.5 * Omega * (s_p * D_op + s_m * D_op.dag())
    prop_x = (liouvillian(H_x, []) * tau).expm()

    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal ** 2 + 1) / 4 / (2 * cosmax)
    W = W / np.sum(W)
    G = gamma / 2
    tg = 2 / G
    c_ops = [np.sqrt(G * W[i]) * (s_m * tensor(qeye(2), (1j * c * eta * xop).expm()))
             for i, c in enumerate(cosal)]
    c_ops.append(np.sqrt(gamma) * s_m)
    prop_diss = (liouvillian(H_diss, c_ops) * tg).expm()
    prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf

    thermal = (-nu * a_dag * a / kT).expm()
    thermal = thermal / thermal.tr()
    g = basis(2, 1)
    occ = np.zeros((num_pulses + 1, N))
    v = operator_to_vector(tensor(g * g.dag(), thermal))
    occ[0, :] = vector_to_operator(v).ptrace(1).diag().real
    for k in range(1, num_pulses + 1):
        v = prop_cycle * v
        occ[k, :] = vector_to_operator(v).ptrace(1).diag().real
    return occ


def trap_bin(p_initial):
    expected = np.zeros(N)
    m = 0
    while True:
        trap_idx = int(n0 * m ** 2)
        if trap_idx >= N:
            break
        expected[trap_idx] = np.sum(p_initial[trap_idx:min(int(n0 * (m + 1) ** 2), N)])
        m += 1
    return expected


# ===================== figure_2.pdf (paper Fig. 3) =====================
occs = {eta: cached(("occ", eta), lambda: run_occupancies(eta)) for eta in eta_values}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
axes = axes.flatten()
legend_handles = legend_labels = None
n_states = np.arange(N)
for idx, eta in enumerate(eta_values):
    occ = occs[eta]
    p_initial, p_final = occ[0, :], occ[-1, :]
    expected = trap_bin(p_initial)
    trap_idx = np.nonzero(expected)[0]
    ax = axes[idx]
    ax.plot(n_states, p_initial, 'o-', color='tab:red', markersize=4, label='Initial')
    ax.bar(n_states, p_final, width=0.5, alpha=0.7, color='tab:blue', label='Final')
    ax.plot(trap_idx, expected[trap_idx], linestyle='None', marker='x', color='k',
            markersize=8, markeredgewidth=2, label='Analytical final')
    if idx == 0:
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.set_title(rf'$\eta = {eta}$')
    ax.set_xlim(-0.5, 14)
    ax.set_xticks(np.arange(0, N, 2))
    ax.grid(True, alpha=0.3)

fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.96))
fig.supxlabel('Occupation number $n$')
fig.supylabel('Probability')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGDIR, 'figure_2.pdf'), format='pdf',
            bbox_inches='tight', dpi=600)
plt.close(fig)
print('saved figures/figure_2.pdf')

# ===================== figure_0.pdf (paper Fig. 1) =====================
# Fock mixture = final populations of the LAST eta in the loop (as in notebook)
p_mix = np.clip(occs[eta_values[-1]][-1, :].astype(float), 0, None)
p_mix = p_mix / p_mix.sum()

N_mix = len(p_mix)
N_wig = max(40, N_mix)
xvec = np.linspace(-5, 5, 251)
r, squeeze_phase = 0.8, np.pi / 2
squeezed_axis_angle = squeeze_phase / 2
alpha = 1.6 * np.exp(1j * (squeezed_axis_angle + np.pi / 2))

psi_sq = squeeze(N_wig, r * np.exp(1j * squeeze_phase)) * basis(N_wig, 0)
rho_sq = ket2dm(psi_sq)
D = displace(N_wig, alpha)
rho_sq_disp = D * rho_sq * D.dag()
p_pad = np.zeros(N_wig); p_pad[:N_mix] = p_mix
rho_mix = Qobj(np.diag(p_pad), dims=[[N_wig], [N_wig]])
rho_mix_disp = D * rho_mix * D.dag()

a_wig = destroy(N_wig)
xop_wig = (a_wig + a_wig.dag()) / np.sqrt(2)
pop_wig = (-1j) * (a_wig - a_wig.dag()) / np.sqrt(2)
center = lambda rho: (float(expect(xop_wig, rho)), float(expect(pop_wig, rho)))

def wig(rho):
    return wigner(rho, xvec, xvec)

W_sq, W_sq_disp = cached(("wig", "sq"), lambda: (wig(rho_sq), wig(rho_sq_disp)))
W_mix, W_mix_disp = cached(("wig", "mix"), lambda: (wig(rho_mix), wig(rho_mix_disp)))

vmin = min(W_sq.min(), W_sq_disp.min(), W_mix.min(), W_mix_disp.min())
vmax = max(W_sq.max(), W_sq_disp.max(), W_mix.max(), W_mix_disp.max())

fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
panels = [(W_sq, 'Squeezed state (before)'), (W_sq_disp, 'Squeezed state (after D)'),
          (W_mix, 'Fock mixture (before)'), (W_mix_disp, 'Fock mixture (after D)')]
im0 = None
for ax, (Wf, title) in zip(axs.flat, panels):
    im = ax.contourf(xvec, xvec, Wf, 120, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    if im0 is None:
        im0 = im
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('p')
    dx = 4.5 * np.cos(squeezed_axis_angle); dp = 4.5 * np.sin(squeezed_axis_angle)
    ax.plot([-dx, dx], [-dp, dp], '--', color='gold', lw=1.6, alpha=0.9)
    ax.set_aspect('equal')

def add_arrow(ax, before, after):
    (x0, p0), (x1, p1) = before, after
    dx, dp = x1 - x0, p1 - p0
    ax.annotate("", xy=(x1, p1), xytext=(x0, p0),
                arrowprops=dict(arrowstyle="<->", lw=3, color="yellow", mutation_scale=18))
    ax.text(0.02, 0.98,
            rf"$\Delta x={dx:.2f},\ \Delta p={dp:.2f},\ |\Delta|={np.hypot(dx, dp):.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            fontsize=17)

add_arrow(axs[0, 1], center(rho_sq), center(rho_sq_disp))
add_arrow(axs[1, 1], center(rho_mix), center(rho_mix_disp))

cbar = fig.colorbar(im0, ax=axs, shrink=0.92, pad=0.02)
cbar.set_label('Wigner value')
fig.suptitle('Phase space response to displacement', y=1.04)
fig.savefig(os.path.join(FIGDIR, 'figure_0.pdf'), format='pdf',
            bbox_inches='tight', dpi=600)
plt.close(fig)
print('saved figures/figure_0.pdf')
