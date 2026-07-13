# -*- coding: utf-8 -*-
"""
Comparison of the full pulse sequence WITH and WITHOUT the RF kick
(figure-5 setup with the full dissipative reset with photon recoil,
N = 16, nbar = 1, 40 cycles).
'refined' means (f, delta) re-optimized per eta for EACH variant separately.

Saves figures/figure_rf_comparison.pdf.  Cached in rfcomp_cache.pkl.
Run:  python3 compare_rf.py
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qutip import (destroy, sigmax, sigmap, sigmam, qeye, tensor, basis,
                   fock_dm, displace, expect, Qobj)
import make_paper_figures as mpf

plt.rcParams.update({
    "text.usetex": False, "font.size": 11, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
CACHEFILE = os.path.join(HERE, "rfcomp_diss_cache.pkl")
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


nu, n0, Omega, N, nbar, R = 1.0, 1, 1.0, 16, 1.0, 40
a = destroy(N); ad = a.dag()
sx = tensor(sigmax(), qeye(N)); sp = tensor(sigmap(), qeye(N)); sm = tensor(sigmam(), qeye(N))
A = tensor(qeye(2), a); Ad = tensor(qeye(2), ad); num = tensor(qeye(2), ad * a)
gket = basis(2, 1); projs = [fock_dm(N, n) for n in range(N)]
p0 = (nbar / (nbar + 1.0)) ** np.arange(N); p0 /= p0.sum()
pid = np.zeros(N); m = 0
while n0 * m ** 2 < N:
    lo, hi = n0 * m ** 2, min(n0 * (m + 1) ** 2, N)
    pid[lo] = p0[lo:hi].sum(); m += 1
pid /= pid.sum()


def tdist(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def prep(eta, f=1.0, de=0.0, rf=True, Omega_y=100.0, RF=1.0):
    D = tensor(qeye(2), displace(N, 1j * eta))
    tau = f * 2 * np.pi / (eta * Omega * np.sqrt(n0))
    H_y = 0.5 * Omega_y * (-1j * sp * D + 1j * sm * D.dag())
    ty = np.pi / (2 * Omega_y)
    U_y = (-1j * H_y * ty).expm(); U_y_u = (1j * H_y * ty).expm()
    H_x = 0.5 * de * sx + nu * num + 0.5 * Omega * (sp * D + sm * D.dag())
    U_x = (-1j * H_x * tau).expm()
    if rf:
        U_rf = (-1j * 0.5 * RF * (A + Ad) * (eta / RF)).expm()
        U = U_rf.dag() * U_y_u * U_x * U_y * U_rf
    else:
        U = U_y_u * U_x * U_y
    return mpf._iterate(U, eta, p_init=p0, R_=R, N_=N)


def refine(eta, rf):
    best = (1e9, 1.0, 0.0)
    for f in np.linspace(0.85, 1.15, 31):
        d = tdist(prep(eta, f, rf=rf), pid)
        if d < best[0]:
            best = (d, f, 0.0)
    for de in np.linspace(-0.4, 0.4, 17):
        for f in np.linspace(best[1] - 0.04, best[1] + 0.04, 9):
            d = tdist(prep(eta, f, de, rf=rf), pid)
            if d < best[0]:
                best = (d, f, de)
    return best


etas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
rows = {}
for eta in etas:
    nom_rf = cached(("nom", True, eta), lambda: tdist(prep(eta, rf=True), pid))
    nom_no = cached(("nom", False, eta), lambda: tdist(prep(eta, rf=False), pid))
    ref_rf = cached(("ref", True, eta), lambda: refine(eta, True))
    ref_no = cached(("ref", False, eta), lambda: refine(eta, False))
    rows[eta] = (nom_rf, nom_no, ref_rf, ref_no)
    print(f"eta={eta}: nominal {nom_rf:.4f}/{nom_no:.4f} (RF/noRF), "
          f"refined {ref_rf[0]:.4f}/{ref_no[0]:.4f}")

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(etas, [rows[e][0] for e in etas], '^-', color='tab:red',
            label='nominal, with RF')
    ax.plot(etas, [rows[e][1] for e in etas], '^--', color='tab:red', alpha=0.55,
            label='nominal, no RF')
    ax.plot(etas, [rows[e][2][0] for e in etas], 's-', color='tab:green',
            label=r'refined ($f,\delta$), with RF')
    ax.plot(etas, [rows[e][3][0] for e in etas], 's--', color='tab:green', alpha=0.55,
            label=r'refined ($f,\delta$), no RF')
    ax.set_xlabel(r'Lamb-Dicke parameter $\eta$')
    ax.set_ylabel('trace distance to ideal trap')
    ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'figure_rf_comparison.pdf')
    fig.savefig(out)
    print('saved', out)
