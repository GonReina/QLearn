# -*- coding: utf-8 -*-
"""
Robustness check for the refocusing + Bloch-Siegert detuning correction:
replace the idealized instantaneous spin reset of Figs. 5-7 by the full
dissipative reset of the main text (spontaneous emission with recoil,
gamma = 1000, as in Fock_mixtures_sim.ipynb), keeping everything else
identical to the figure-5 setup (N = 16, thermal nbar = 1, 40 cycles).

The (f, delta) values are the ones calibrated on the ideal-reset model,
i.e. NOT re-optimized for the dissipative cycle (conservative test).

Run:  python3 check_dissipative_reset.py
"""
import os
import pickle
import numpy as np
from qutip import (destroy, sigmaz, sigmax, sigmap, sigmam, qeye, tensor,
                   basis, fock_dm, displace, expect, Qobj, liouvillian,
                   operator_to_vector, vector_to_operator, spre, spost)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHEFILE = os.path.join(HERE, "dissreset_cache.pkl")
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


# --- figure-5 setup, but with the dissipative reset of the main text ---
N, n0, nu, Omega, nbar, R = 16, 1, 1.0, 1.0, 1.0, 40
gamma, RF, Omega_y = 1000.0, 1.0, 100.0

a = destroy(N); ad = a.dag()
sz = tensor(sigmaz(), qeye(N)); sx = tensor(sigmax(), qeye(N))
sp = tensor(sigmap(), qeye(N)); sm = tensor(sigmam(), qeye(N))
A = tensor(qeye(2), a); Ad = tensor(qeye(2), ad)
num = tensor(qeye(2), ad * a)
xop = a + ad


def thermal_p(N, nbar):
    p = (nbar / (nbar + 1.0)) ** np.arange(N)
    return p / p.sum()


def ideal_trapped(p0, n0=1):
    Nl = len(p0); out = np.zeros(Nl); m = 0
    while n0 * m ** 2 < Nl:
        lo, hi = n0 * m ** 2, min(n0 * (m + 1) ** 2, Nl)
        out[lo] = p0[lo:hi].sum(); m += 1
    return out / out.sum()


def tdist(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


p0 = thermal_p(N, nbar)
pid = ideal_trapped(p0, n0)
projs = [fock_dm(N, n) for n in range(N)]
gket = basis(2, 1)

# dissipative reset with recoil (as in Fock_mixtures_sim.ipynb)
cosmax = 100
cosal = np.arange(-cosmax, cosmax + 1) / cosmax
W = 3 * (cosal ** 2 + 1) / 4 / (2 * cosmax)
W = W / np.sum(W)
G = gamma / 2
tg = 2 / G
H_diss = nu * num


def prep_dissipative(eta, tau_factor=1.0, delta_x=0.0):
    D = tensor(qeye(2), displace(N, 1j * eta))
    tau = tau_factor * 2 * np.pi / (eta * Omega * np.sqrt(n0))
    c_ops = [np.sqrt(G * W[i]) * (sm * tensor(qeye(2), (1j * c * eta * xop).expm()))
             for i, c in enumerate(cosal)]
    c_ops.append(np.sqrt(gamma) * sm)
    prop_diss = (liouvillian(H_diss, c_ops) * tg).expm()

    U_rf = (-1j * 0.5 * RF * (A + Ad) * (eta / RF)).expm()
    H_y = 0.5 * Omega_y * (-1j * sp * D + 1j * sm * D.dag())
    ty = np.pi / (2 * Omega_y)
    U_y = (-1j * H_y * ty).expm()
    H_x = 0.5 * delta_x * sx + nu * num + 0.5 * Omega * (sp * D + sm * D.dag())
    U_x = (-1j * H_x * tau).expm()
    U = U_rf.dag() * U_y.dag() * U_x * U_y * U_rf
    prop_coh = spre(U) * spost(U.dag())
    prop_cycle = prop_diss * prop_coh

    rho = tensor(gket * gket.dag(), Qobj(np.diag(p0), dims=[[N], [N]]))
    rho_vec = operator_to_vector(rho)
    for _ in range(R):
        rho_vec = prop_cycle * rho_vec
    rho_m = vector_to_operator(rho_vec).ptrace(1)
    return np.real(np.array(expect(projs, rho_m)))


# (f*, delta*) calibrated on the IDEAL-reset model (figure 5), not re-optimized
calib = {0.2: (1.020, 0.0), 0.3: (1.020, 0.0), 0.4: (0.960, -0.05), 0.5: (0.950, -0.15)}

if __name__ == "__main__":
    print(f"{'eta':>5} {'nominal':>9} {'refined':>9}   (dissipative reset with recoil)")
    for eta, (f, de) in calib.items():
        d_nom = cached(("nom", eta), lambda: tdist(prep_dissipative(eta), pid))
        d_fix = cached(("fix", eta), lambda: tdist(
            prep_dissipative(eta, tau_factor=f, delta_x=de), pid))
        print(f"{eta:>5} {d_nom:>9.4f} {d_fix:>9.4f}")
