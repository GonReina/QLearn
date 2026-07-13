# -*- coding: utf-8 -*-
"""
Publication figures for the section
"Blue-sideband refocusing with Bloch-Siegert detuning".

Reproduces the analyses of Improved_pulse_sequence.ipynb and
Pulse_shaping.ipynb, with the idealized instantaneous spin reset replaced
by the full dissipative reset with photon recoil of the main text
(gamma = 1000), and saves three PDF figures into figures/:

    figure_5.pdf : refocusing + Bloch-Siegert detuning on the full sequence
                   (a) trace distance vs eta, (b) (f, delta) calibration map
    figure_6.pdf : metrological benchmark (displacement Fisher information)
                   (a) F(alpha) at eta = 0.5, (b) gain over SQL vs eta
    figure_7.pdf : pulse shaping comparison
                   (a) amplitude windows vs refocusing + detuning
                   (b) time-dependent detuning profiles

Run:  python3 make_paper_figures.py
"""
import os
import pickle
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from qutip import (destroy, sigmaz, sigmax, sigmap, sigmam, qeye, tensor,
                   basis, fock_dm, displace, expect, Qobj, liouvillian)

plt.rcParams.update({
    "text.usetex": False, "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "lines.linewidth": 1.4, "lines.markersize": 4,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------- simple checkpoint cache (results are expensive; allow resuming)
CACHEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figgen_diss_cache.pkl")
try:
    with open(CACHEFILE, "rb") as fh:
        _CACHE = pickle.load(fh)
except Exception:
    _CACHE = {}


def cached(key, fn):
    """Return _CACHE[key], computing and checkpointing it if absent."""
    if key not in _CACHE:
        _CACHE[key] = fn()
        tmp = CACHEFILE + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(_CACHE, fh)
        os.replace(tmp, CACHEFILE)
    return _CACHE[key]

# ---------------- shared setup (full dissipative spin reset with photon recoil)
nu, n0, Omega, N, nbar, R = 1.0, 1, 1.0, 16, 1.0, 40

a = destroy(N); ad = a.dag()
sz = tensor(sigmaz(), qeye(N)); sx = tensor(sigmax(), qeye(N))
sp = tensor(sigmap(), qeye(N)); sm = tensor(sigmam(), qeye(N))
A = tensor(qeye(2), a); Ad = tensor(qeye(2), ad); num = tensor(qeye(2), ad * a)
gket = basis(2, 1); projs = [fock_dm(N, n) for n in range(N)]


def thermal_p(N, nbar):
    p = (nbar / (nbar + 1.0)) ** np.arange(N)
    return p / p.sum()


def ideal_trapped(p0, n0=1):
    N = len(p0); out = np.zeros(N); m = 0
    while n0 * m ** 2 < N:
        lo, hi = n0 * m ** 2, min(n0 * (m + 1) ** 2, N)
        out[lo] = p0[lo:hi].sum(); m += 1
    return out / out.sum()


def tdist(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


p0 = thermal_p(N, nbar)
pid = ideal_trapped(p0, n0)


def Dop(eta):
    return tensor(qeye(2), displace(N, 1j * eta))


# --- dissipative spin reset with photon recoil (main-text model, gamma = 1000)
gamma_reset = 1000.0
_RESET_CACHE = {}


def reset_super(N_, eta):
    """Dense column-stacked superoperator of the dissipative reset (duration
    tg = 4/gamma) with recoil, as in Eq. (9) of the manuscript."""
    key = (N_, round(float(eta), 6))
    if key not in _RESET_CACHE:
        fname = os.path.join(tempfile.gettempdir(), f"reset_{N_}_{key[1]}.npy")
        if os.path.exists(fname):
            _RESET_CACHE[key] = np.load(fname)
        else:
            aa = destroy(N_); xop = aa + aa.dag()
            smf = tensor(sigmam(), qeye(N_))
            numf = tensor(qeye(2), aa.dag() * aa)
            cosmax = 100
            cosal = np.arange(-cosmax, cosmax + 1) / cosmax
            Wd = 3 * (cosal ** 2 + 1) / 4 / (2 * cosmax)
            Wd = Wd / np.sum(Wd)
            G = gamma_reset / 2
            tg = 2 / G
            c_ops = [np.sqrt(G * Wd[i]) * (smf * tensor(qeye(2), (1j * c * eta * xop).expm()))
                     for i, c in enumerate(cosal)]
            c_ops.append(np.sqrt(gamma_reset) * smf)
            Ld = (liouvillian(nu * numf, c_ops) * tg).expm().full()
            _RESET_CACHE[key] = np.ascontiguousarray(Ld)
            np.save(fname, _RESET_CACHE[key])
    return _RESET_CACHE[key]


def _iterate(U, eta, p_init=None, R_=R, N_=N):
    """Cycle: coherent pulse-sequence unitary U, then dissipative reset."""
    Ud = U.full()
    Ld = reset_super(N_, eta)
    dim = 2 * N_
    rho = np.zeros((dim, dim), dtype=complex)
    rho[N_:, N_:] = np.diag(p0 if p_init is None else p_init)  # |g><g| x rho_m
    for _ in range(R_):
        rho = Ud @ rho @ Ud.conj().T
        rho = (Ld @ rho.flatten(order="F")).reshape((dim, dim), order="F")
    d = np.real(np.diag(rho))
    return d[:N_] + d[N_:]


def prep_square(eta, tau_factor=1.0, delta_x=0.0, Omega_y=100.0, RF=1.0):
    """Full pulse sequence (RF kick, Y(pi/2), X pulse, undo) with square X-pulse."""
    D = Dop(eta); tau = tau_factor * 2 * np.pi / (eta * Omega * np.sqrt(n0))
    U_rf = (-1j * 0.5 * RF * (A + Ad) * (eta / RF)).expm(); U_rf_u = U_rf.dag()
    H_y = 0.5 * Omega_y * (-1j * sp * D + 1j * sm * D.dag()); ty = np.pi / (2 * Omega_y)
    U_y = (-1j * H_y * ty).expm(); U_y_u = (1j * H_y * ty).expm()
    H_x = 0.5 * delta_x * sx + nu * num + 0.5 * Omega * (sp * D + sm * D.dag())
    U_x = (-1j * H_x * tau).expm()
    return _iterate(U_rf_u * U_y_u * U_x * U_y * U_rf, eta)


# =====================================================================
# Figure 5: refocusing + Bloch-Siegert detuning (full sequence)
# =====================================================================
def figure_refocusing(fname="figure_5.pdf"):
    print("[fig 5] trace distance vs eta ...")
    etas = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    def _one_eta(eta):
        dn = tdist(prep_square(eta), pid)
        best = (1e9, 1.0)
        for f in np.linspace(0.85, 1.15, 31):
            d = tdist(prep_square(eta, tau_factor=f), pid)
            if d < best[0]:
                best = (d, f)
        best2 = (best[0], best[1], 0.0)
        for de in np.linspace(-0.4, 0.4, 17):
            for f in np.linspace(best[1] - 0.04, best[1] + 0.04, 9):
                d = tdist(prep_square(eta, tau_factor=f, delta_x=de), pid)
                if d < best2[0]:
                    best2 = (d, f, de)
        return dn, best[0], best[1], best2[0], best2[2]

    d_nom, d_f, d_fd, f_opt, de_opt = [], [], [], [], []
    for eta in etas:
        dn, df, fo, dfd, deo = cached(("fig5a", float(eta)), lambda: _one_eta(eta))
        d_nom.append(dn); d_f.append(df); f_opt.append(fo)
        d_fd.append(dfd); de_opt.append(deo)
        print(f"   eta={eta:.2f}  nominal={dn:.4f}  +tau={df:.4f} "
              f" +tau,delta={dfd:.4f}  f*={fo:.3f}  d*={deo:+.3f}")

    print("[fig 5] (f, delta) calibration map at eta=0.4 ...")
    eta_map = 0.4
    f_grid = np.linspace(0.70, 1.30, 41)
    delta_grid = np.linspace(-0.5, 0.5, 33)
    TD = np.zeros((delta_grid.size, f_grid.size))
    for i, de in enumerate(delta_grid):
        TD[i, :] = cached(("fig5b", float(de)), lambda: np.array(
            [tdist(prep_square(eta_map, tau_factor=f, delta_x=de), pid)
             for f in f_grid]))
    i_b, j_b = np.unravel_index(np.argmin(TD), TD.shape)
    td_nom_map = tdist(prep_square(eta_map), pid)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.8))
    a1.plot(etas, d_nom, "^-", color="tab:red", label=r"nominal $\tau$")
    a1.plot(etas, d_f, "o-", color="tab:orange", label=r"refocused $f\tau$")
    a1.plot(etas, d_fd, "s-", color="tab:green", label=r"refocused $+$ detuning $\delta$")
    a1.set_xlabel(r"Lamb-Dicke parameter $\eta$")
    a1.set_ylabel("trace distance to ideal trap")
    a1.set_yscale("log"); a1.grid(True, which="both", alpha=0.3)
    a1.set_ylim(1.2e-3, None)
    a1.legend(frameon=False, loc="lower left")
    a1.text(0.0, 1.02, "A)", transform=a1.transAxes, va="bottom", fontweight="bold")

    pcm = a2.pcolormesh(f_grid, delta_grid, TD, shading="auto", cmap="viridis",
                        norm=LogNorm(vmin=max(TD.min(), 1e-3), vmax=TD.max()))
    cb = fig.colorbar(pcm, ax=a2, pad=0.02)
    cb.set_label("trace distance", fontsize=10)
    a2.plot(1.0, 0.0, "o", ms=7, mfc="white", mec="k", label="nominal")
    a2.plot(f_grid[j_b], delta_grid[i_b], "*", ms=12, mfc="red", mec="k", label="optimum")
    a2.set_xlabel(r"pulse-length factor $f$")
    a2.set_ylabel(r"detuning $\delta$ $[\nu]$")
    a2.legend(loc="upper right", framealpha=0.9)
    a2.text(0.0, 1.02, "B)", transform=a2.transAxes, va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print(f"[fig 5] saved -> figures/{fname}  "
          f"(map optimum f={f_grid[j_b]:.3f}, delta={delta_grid[i_b]:+.3f}, "
          f"TD={TD[i_b, j_b]:.4f}; nominal TD={td_nom_map:.4f})")
    return dict(etas=etas, d_nom=d_nom, d_f=d_f, d_fd=d_fd,
                f_opt=f_opt, de_opt=de_opt)


# =====================================================================
# Figure 6: metrological benchmark (Fisher information)
# =====================================================================
def figure_metrology(fname="figure_6.pdf"):
    print("[fig 6] metrology benchmark ...")
    N_met, nbar_met, R_met, n_meas, F_SQL = 24, 5.0, 40, 4, 4.0
    aa = destroy(N_met); ada = aa.dag()
    Am = tensor(qeye(2), aa); Adm = tensor(qeye(2), ada)
    numm = tensor(qeye(2), ada * aa)
    sxm = tensor(sigmax(), qeye(N_met))
    spm = tensor(sigmap(), qeye(N_met)); smm = tensor(sigmam(), qeye(N_met))
    g_ = basis(2, 1); projs_m = [fock_dm(N_met, n) for n in range(N_met)]
    p0_met = thermal_p(N_met, nbar_met)
    pid_met = ideal_trapped(p0_met, n0)

    def prep_met(eta, tau_factor=1.0, delta_x=0.0, Omega_y=100.0, RF=1.0):
        D = tensor(qeye(2), displace(N_met, 1j * eta))
        tau = tau_factor * 2 * np.pi / (eta * Omega * np.sqrt(n0))
        H_rf = 0.5 * RF * (Am + Adm)
        U_rf = (-1j * H_rf * (eta / RF)).expm(); U_rf_u = U_rf.dag()
        H_y = 0.5 * Omega_y * (-1j * spm * D + 1j * smm * D.dag())
        ty = np.pi / (2 * Omega_y)
        U_y = (-1j * H_y * ty).expm(); U_y_u = (1j * H_y * ty).expm()
        H_x = 0.5 * delta_x * sxm + nu * numm + 0.5 * Omega * (spm * D + smm * D.dag())
        U_x = (-1j * H_x * tau).expm()
        U = U_rf_u * U_y_u * U_x * U_y * U_rf
        return _iterate(U, eta, p_init=p0_met, R_=R_met, N_=N_met)

    alphas = np.linspace(1e-3, 1.2, 241)
    Dmats = [displace(N_met, al).full() for al in alphas]

    def fisher_curve(p, n):
        xi = np.array([np.sum(p * np.abs(Dm[n, :]) ** 2) for Dm in Dmats])
        dxi = np.gradient(xi, alphas)
        return dxi ** 2 / np.clip(xi * (1 - xi), 1e-9, None)

    def FQ(p, n):
        return fisher_curve(p, n).max()

    etas_met = [0.3, 0.4, 0.5]

    def _one_eta_met(eta):
        best = (1e9, 1.0, 0.0)
        for f in np.linspace(0.92, 1.08, 9):
            for de in np.linspace(-0.3, 0.3, 9):
                d = tdist(prep_met(eta, tau_factor=f, delta_x=de), pid_met)
                if d < best[0]:
                    best = (d, f, de)
        return prep_met(eta), prep_met(eta, tau_factor=best[1], delta_x=best[2]), best

    rows = {}
    for eta in etas_met:
        p_nom, p_fix, best = cached(("fig6", float(eta)), lambda: _one_eta_met(eta))
        rows[eta] = dict(p_nom=p_nom, p_fix=p_fix,
                         FQ_nom=FQ(p_nom, n_meas), FQ_fix=FQ(p_fix, n_meas))
        print(f"   eta={eta}  f*={best[1]:.3f} d*={best[2]:+.3f}  "
              f"FQ_nom={rows[eta]['FQ_nom']:.2f} FQ_fix={rows[eta]['FQ_fix']:.2f}")
    FQ_ideal = FQ(pid_met, n_meas)
    gdB = lambda F: 10 * np.log10(F / F_SQL)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.8))
    eta_show = 0.5
    a1.plot(alphas, fisher_curve(pid_met, n_meas), "k-", label="ideal trap")
    a1.plot(alphas, fisher_curve(rows[eta_show]["p_fix"], n_meas), "-",
            color="tab:green", label=r"refocused $+$ $\delta$")
    a1.plot(alphas, fisher_curve(rows[eta_show]["p_nom"], n_meas), "-",
            color="tab:red", label="nominal")
    a1.axhline(F_SQL, color="gray", ls="--", lw=1.2, label=r"SQL ($\mathcal{F}=4$)")
    a1.set_xlabel(r"displacement $|\alpha|$")
    a1.set_ylabel(r"Fisher information $\mathcal{F}(\alpha)$")
    a1.grid(True, alpha=0.3); a1.legend(frameon=False)
    a1.text(0.0, 1.02, "A)", transform=a1.transAxes, va="bottom", fontweight="bold")

    g_id = [gdB(FQ_ideal)] * len(etas_met)
    g_no = [gdB(rows[e]["FQ_nom"]) for e in etas_met]
    g_fx = [gdB(rows[e]["FQ_fix"]) for e in etas_met]
    a2.axhline(0.0, color="gray", ls="--", lw=1.2, label="SQL")
    a2.plot(etas_met, g_id, "k:", marker="d", label="ideal trap")
    a2.plot(etas_met, g_fx, "s-", color="tab:green", label=r"refocused $+$ $\delta$")
    a2.plot(etas_met, g_no, "^-", color="tab:red", label="nominal")
    a2.set_xlabel(r"Lamb-Dicke parameter $\eta$")
    a2.set_ylabel("gain over SQL [dB]")
    a2.grid(True, alpha=0.3); a2.legend(frameon=False, loc="center right")
    a2.text(0.0, 1.02, "B)", transform=a2.transAxes, va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print(f"[fig 6] saved -> figures/{fname}  (ideal gain {g_id[0]:+.2f} dB)")


# =====================================================================
# Figure 7: pulse shaping comparison
# =====================================================================
def window(name, M):
    t = np.linspace(0, 1, M)
    if name == "square":
        return np.ones(M)
    if name == "cos":
        return 0.5 * (1 - np.cos(2 * np.pi * t))          # Hann
    if name == "black":
        return 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    if name == "flattop":
        w = np.ones(M); r = max(1, M // 4)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
        w[:r] = ramp; w[-r:] = ramp[::-1]
        return w
    raise ValueError(name)


def prep_shaped(eta, shape="cos", kappa=1.0, M=50, Omega_y=100.0, RF=1.0):
    D = Dop(eta); tau = 2 * np.pi / (eta * Omega * np.sqrt(n0)); T = kappa * tau
    w = window(shape, M); dt = T / M
    Om_t = w * (Omega * tau) / (np.sum(w) * dt)           # area-matched
    U_x = tensor(qeye(2), qeye(N))
    for k in range(M):
        Hk = nu * num + 0.5 * Om_t[k] * (sp * D + sm * D.dag())
        U_x = (-1j * Hk * dt).expm() * U_x
    U_rf = (-1j * 0.5 * RF * (A + Ad) * (eta / RF)).expm(); U_rf_u = U_rf.dag()
    H_y = 0.5 * Omega_y * (-1j * sp * D + 1j * sm * D.dag()); ty = np.pi / (2 * Omega_y)
    U_y = (-1j * H_y * ty).expm(); U_y_u = (1j * H_y * ty).expm()
    return _iterate(U_rf_u * U_y_u * U_x * U_y * U_rf, eta)


def detuning_profile(name, M, amp):
    t = np.linspace(0, 1, M)
    if name == "const":
        return amp * np.ones(M)
    if name == "bump":
        return amp * np.exp(-((t - 0.5) / 0.2) ** 2)
    if name == "edges":
        return amp * (np.exp(-(t / 0.2) ** 2) + np.exp(-((t - 1) / 0.2) ** 2))
    if name == "ramp":
        return amp * (2 * t - 1)
    raise ValueError(name)


def prep_chirp(eta, profile="const", amp=0.0, f=1.0, M=40, Omega_y=100.0, RF=1.0):
    D = Dop(eta); tau = f * 2 * np.pi / (eta * Omega * np.sqrt(n0)); dt = tau / M
    dprof = detuning_profile(profile, M, amp)
    U_x = tensor(qeye(2), qeye(N))
    for k in range(M):                     # Omega fixed at nu; only detuning varies
        Hk = 0.5 * dprof[k] * sx + nu * num + 0.5 * Omega * (sp * D + sm * D.dag())
        U_x = (-1j * Hk * dt).expm() * U_x
    U_rf = (-1j * 0.5 * RF * (A + Ad) * (eta / RF)).expm(); U_rf_u = U_rf.dag()
    H_y = 0.5 * Omega_y * (-1j * sp * D + 1j * sm * D.dag()); ty = np.pi / (2 * Omega_y)
    U_y = (-1j * H_y * ty).expm(); U_y_u = (1j * H_y * ty).expm()
    return _iterate(U_rf_u * U_y_u * U_x * U_y * U_rf, eta)


def figure_pulse_shaping(fname="figure_7.pdf"):
    print("[fig 7] amplitude windows ...")
    shape_map = {"square": "square", "Hann": "cos",
                 "Blackman": "black", "flat-top": "flattop"}
    shape_col = {"square": "tab:red", "Hann": "tab:purple",
                 "Blackman": "tab:brown", "flat-top": "tab:orange"}
    etas2 = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    td_shapes = {name: [cached(("fig7a", key, float(e)),
                               lambda: tdist(prep_shaped(e, key, 1.0), pid))
                        for e in etas2]
                 for name, key in shape_map.items()}

    def _one_ref(e):
        best = 1e9
        for f in np.linspace(0.90, 1.10, 11):
            for de in np.linspace(-0.3, 0.3, 7):
                best = min(best, tdist(prep_square(e, tau_factor=f, delta_x=de), pid))
        return best

    td_ref = [cached(("fig7ref", float(e)), lambda: _one_ref(e)) for e in etas2]

    print("[fig 7] detuning profiles ...")
    amps = np.linspace(-0.4, 0.4, 17)
    profiles = ["const", "bump", "edges", "ramp"]
    prof_col = {"const": "tab:green", "bump": "tab:blue",
                "edges": "tab:purple", "ramp": "tab:brown"}
    prof_lbl = {"const": r"constant $\delta$", "bump": r"$\delta(t)$ bump",
                "edges": r"$\delta(t)$ at edges", "ramp": r"$\delta(t)$ ramp"}
    etas3 = [0.2, 0.3, 0.4, 0.5]
    td_chirp = {pr: [] for pr in profiles}
    td_chirp_nom = []
    for e in etas3:
        td_chirp_nom.append(
            cached(("fig7nom", float(e)), lambda: tdist(prep_chirp(e), pid)))
        for pr in profiles:
            td_chirp[pr].append(
                cached(("fig7b", pr, float(e)),
                       lambda: min(tdist(prep_chirp(e, pr, A_), pid)
                                   for A_ in amps)))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.8))
    for name in shape_map:
        a1.plot(etas2, td_shapes[name], "o-", color=shape_col[name], label=name)
    a1.plot(etas2, td_ref, "s-", color="tab:green", label=r"refocusing $+$ $\delta$")
    a1.set_xlabel(r"Lamb-Dicke parameter $\eta$")
    a1.set_ylabel("trace distance to ideal trap")
    a1.set_yscale("log"); a1.grid(True, which="both", alpha=0.3)
    a1.set_ylim(1.2e-3, None)
    a1.legend(frameon=False, ncol=2, loc="lower center")
    a1.text(0.0, 1.02, "A)", transform=a1.transAxes, va="bottom", fontweight="bold")

    a2.plot(etas3, td_chirp_nom, "^-", color="tab:red", label=r"nominal ($\delta=0$)")
    for pr in profiles:
        a2.plot(etas3, td_chirp[pr], "o-", color=prof_col[pr], label=prof_lbl[pr])
    a2.set_xlabel(r"Lamb-Dicke parameter $\eta$")
    a2.set_ylabel("trace distance to ideal trap")
    a2.set_yscale("log"); a2.grid(True, which="both", alpha=0.3)
    a2.set_ylim(5.0e-3, None)
    a2.legend(frameon=False, ncol=2, loc="lower center")
    a2.text(0.0, 1.02, "B)", transform=a2.transAxes, va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    print(f"[fig 7] saved -> figures/{fname}")


if __name__ == "__main__":
    figure_refocusing()
    figure_metrology()
    figure_pulse_shaping()
    print("All figures done.")
