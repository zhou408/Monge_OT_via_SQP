# -*- coding: utf-8 -*-
"""Gaussian -> Exponential notebook section.

Extracted from original notebook export on split date 2026-02-20.
This file intentionally keeps original logic, defaults, and analytic comparisons.
"""

import sys

# Entry policy for notebook-derived scripts:
# run heavy demo blocks only when explicitly requested.
_RUN_FULL_DEMO = ('--demo' in sys.argv and 'full' in sys.argv)
if __name__ == '__main__' and not _RUN_FULL_DEMO:
    print('This is a notebook-style experiment script.')
    print('Run with: python {} --demo full'.format(__file__))
    sys.exit(0)

# Ensure notebook-style unicode prints work on Windows terminals.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


"""# Gaussian to Exponential

## Old Ver
"""

# ============================================================
# Gaussian → Exponential OT (1D), monotone PL map + SQP
# - Map: s(x) piecewise-affine with slopes a_i = exp(gamma_i) > 0
# - Objective: J = 1/2 E[(s(X) - X)^2], X ~ N(mu0, s0^2)
# - Constraints: pushforward density on a d-grid matches Exp(lam1)
# - Step-size modes: "trust_clip", "fixed", "linesearch"
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# -----------------------------
# Densities & helper utilities
# -----------------------------
def normal_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

def exp_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

# -----------------------------
# Problem + grid specs
# -----------------------------
@dataclass
class G2EParams:
    mu0: float    # source Gaussian mean
    s0:  float    # source Gaussian std
    lam1: float   # target Exponential rate

@dataclass
class XGrid:
    L: float
    R: float
    m: int
    x: np.ndarray  # shape (m,)
    dx: float

@dataclass
class DGrid:
    L: float
    R: float
    m: int
    d: np.ndarray  # shape (m,)

def domain_gauss95(params: G2EParams, k: float = 2.8) -> Tuple[float, float]:
    """A bit wider than 95% mass to reduce boundary effects."""
    L = params.mu0 - k * params.s0
    R = params.mu0 + k * params.s0
    return L, R

def make_xgrid(params: G2EParams, m: int, L: Optional[float]=None, R: Optional[float]=None) -> XGrid:
    if L is None or R is None:
        L2, R2 = domain_gauss95(params, k=2.8)
        if L is None: L = L2
        if R is None: R = R2
    x = np.linspace(L, R, m)
    dx = (R - L) / (m - 1)
    return XGrid(L, R, m, x, dx)

def make_dgrid(params: G2EParams, m: int, p: float = 0.999, L: float = 0.0) -> DGrid:
    """Right-quantile coverage for Exp(lam1)."""
    R = -np.log(1.0 - p) / params.lam1
    d = np.linspace(L, R, m)
    return DGrid(L, R, m, d)

# --------------------------------------------
# Monotone piecewise-linear map specification
# --------------------------------------------
@dataclass
class PLSpec:
    knots: np.ndarray     # x_0 < ... < x_K (K+1 knots)
    K: int                # number of pieces (K)

def build_plspec(xgrid: XGrid, K: int) -> PLSpec:
    knots = np.linspace(xgrid.L, xgrid.R, K+1)
    return PLSpec(knots=knots, K=K)

def unpack_params(theta: np.ndarray, K: int):
    """From theta=[gamma_0..gamma_{K-1}, y0] -> (a (slopes), y0)."""
    gammas = theta[:K]
    y0     = theta[K]
    a = np.exp(gammas)  # slopes > 0 (monotone)
    return a, y0

def pl_continuity_y(a: np.ndarray, y0: float, knots: np.ndarray) -> np.ndarray:
    """Return y_i = s(x_i) for i=0..K, given slopes a_i and y0=s(x0)."""
    K = len(a)
    y = np.zeros(K+1)
    y[0] = y0
    for i in range(K):
        y[i+1] = y[i] + a[i] * (knots[i+1] - knots[i])
    return y  # length K+1

def piece_affine_params(a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """Return (a_i, b_i) for each piece so s(x)=a_i x + b_i on [x_i,x_{i+1}]."""
    K = len(a)
    b = np.zeros(K)
    for i in range(K):
        b[i] = y[i] - a[i] * knots[i]
    return a, b

def eval_s_and_jac_s(x: np.ndarray, a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """
    Evaluate s(x) and ∂s/∂θ for θ=(gamma[0:K], y0).
    Returns:
      svals: shape (m,)
      Js:    shape (m, K+1)  (columns: d/d gamma_0..gamma_{K-1}, d/d y0)
    """
    K = len(a)
    m = len(x)
    svals = np.zeros(m)
    Js = np.zeros((m, K+1))

    # which piece each x lies in
    idx = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, K-1)

    for t in range(m):
        i = idx[t]         # active piece index
        xi = knots[i]
        # s(x) on piece i
        svals[t] = y[i] + a[i] * (x[t] - xi)

        # d/d y0: every y_i shifts by +1, so ∂s/∂y0 = 1
        Js[t, K] = 1.0

        # d/d gamma_k:
        # for k < i: y[i] depends on a_k via Δx_k
        for k in range(i):
            Js[t, k] += a[k] * (knots[k+1] - knots[k])  # ∂y[i]/∂gamma_k = a_k Δx_k
        # for k = i: slope term
        Js[t, i] += a[i] * (x[t] - xi)
        # for k > i: 0
    return svals, Js

# --------------------------------------------
# Objective J(s) and Gauss–Newton (g, H)
# --------------------------------------------
def objective_J(theta: np.ndarray, spec: PLSpec, xgrid: XGrid, params: G2EParams):
    """
    J = 1/2 E[(s(X) - X)^2] via quadrature on x-grid.
    Return (J, grad, H) with Gauss–Newton approximation.
    """
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    svals, Js = eval_s_and_jac_s(xgrid.x, a, y, spec.knots)

    err = (svals - xgrid.x)
    w = normal_pdf(xgrid.x, params.mu0, params.s0) * xgrid.dx  # quadrature weights

    Jval = 0.5 * np.sum((err**2) * w)

    # Gauss–Newton: g ≈ Σ err * (∂s/∂θ) * w ; H ≈ Σ (∂s/∂θ)(∂s/∂θ)^T * w
    g = (Js.T @ (err * w))
    H = (Js.T * w) @ Js + 1e-10 * np.eye(spec.K+1)  # tiny damping
    return Jval, g, H

# -------------------------------------------------
# Constraint zeta(d) = Pf(s)(d) - g(d) and Jacobian
# -------------------------------------------------
def pushforward_and_jac(theta: np.ndarray, spec: PLSpec, dgrid: DGrid, params: G2EParams):
    """
    Pf(s)(d) = sum_i (1/a_i) f((d-b_i)/a_i) * 1_{u in [x_i,x_{i+1}]},
    with f Gaussian(mu0, s0). Return Pf (m,), A (m, K+1) = d Pf / d theta.
    """
    mu0, s0, lam1 = params.mu0, params.s0, params.lam1
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, y, spec.knots)

    m = dgrid.m
    Pf = np.zeros(m)
    A  = np.zeros((m, spec.K+1))

    # Precompute derivatives of b_i wrt gammas and y0
    K = spec.K
    dx_segs = np.diff(spec.knots)
    db_dgamma = np.zeros((K, K))  # rows i, cols k
    db_dy0    = np.ones(K)        # ∂b_i/∂y0 = 1
    for i in range(K):
        # via y_i for k < i
        for k in range(i):
            db_dgamma[i, k] += a[k] * dx_segs[k]
        # minus a_i * x_i for k=i (since b_i = y_i - a_i x_i)
        db_dgamma[i, i] += -a[i] * spec.knots[i]
        # k>i: no effect

    d = dgrid.d
    inv_s02 = 1.0 / (s0 * s0)

    for i in range(K):
        ai, bi = a_i[i], b_i[i]
        xi, xip1 = spec.knots[i], spec.knots[i+1]

        u = (d - bi) / ai
        valid = (u >= xi) & (u <= xip1)
        if not np.any(valid):
            continue

        du = u[valid]
        f_u = normal_pdf(du, mu0, s0)

        # Pf add
        Pf[valid] += (1.0 / ai) * f_u

        # Local partials (for valid entries)
        # z = (1/a) f(u), u=(d-b)/a
        # ∂z/∂a = (f(u)/a^2) * ( -1 + u(u-μ0)/s0^2 )
        # ∂z/∂b =  f(u) * (u-μ0)/(a^2 s0^2)
        dz_da = (f_u / (ai * ai)) * (-1.0 + du * (du - mu0) * inv_s02)
        dz_db = f_u * ((du - mu0) * inv_s02) / (ai * ai)

        # Chain to parameters gamma_k and y0:
        # ∂a_i/∂γ_k = a_i if k=i else 0
        # ∂z/∂γ_k = dz_da * ∂a_i/∂γ_k + dz_db * ∂b_i/∂γ_k
        for k in range(K):
            contrib = np.zeros_like(dz_db)
            if k == i:
                contrib = contrib + dz_da * ai
            if db_dgamma[i, k] != 0.0:
                contrib = contrib + dz_db * db_dgamma[i, k]
            A[valid, k] += contrib

        # y0 column
        A[valid, K] += dz_db * db_dy0[i]

    # subtract target pdf
    g_d = exp_pdf(d, lam1)
    zeta = Pf - g_d
    return zeta, A

# -----------------------------
# KKT direction (no update)
# -----------------------------
def sqp_direction_g2e(
    theta: np.ndarray, lam: np.ndarray,
    spec: PLSpec, xgrid: XGrid, dgrid: DGrid, params: G2EParams,
    reg_H: float = 1e-10, reg_dual: float = 1e-8
):
    """Single KKT solve: return (dtheta, w, g, zeta, A, Jval)."""
    Jval, gJ, HJ = objective_J(theta, spec, xgrid, params)
    zeta, A = pushforward_and_jac(theta, spec, dgrid, params)
    g = gJ + A.T @ lam
    H_reg = HJ + reg_H * np.eye(HJ.shape[0])

    KKT = np.block([
        [H_reg,              A.T],
        [A,       -reg_dual * np.eye(A.shape[0])]
    ])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    dtheta = sol[:theta.size]
    w = sol[theta.size:]
    return dtheta, w, g, zeta, A, Jval

# -----------------------------
# Merit function (Armijo LS)
# -----------------------------
def merit_phi_g2e(theta, spec, xgrid, dgrid, params, mu: float):
    Jval, _, _ = objective_J(theta, spec, xgrid, params)
    zeta, _ = pushforward_and_jac(theta, spec, dgrid, params)
    return float(Jval + 0.5 * mu * (zeta @ zeta))

def merit_dirderiv_g2e(theta, spec, xgrid, dgrid, params, zeta, A, dtheta, mu: float):
    # ∇φ = ∇J + μ A^T ζ
    _, gJ, _ = objective_J(theta, spec, xgrid, params)
    grad_phi = gJ + mu * (A.T @ zeta)
    return float(grad_phi @ dtheta)

# -----------------------------
# Full SQP driver (3 step modes)
# -----------------------------
def sqp_solve_g2e(
    params: G2EParams,
    K_pieces: int = 6,
    x_m: int = 501, d_m: int = 101,
    max_iter: int = 200,
    tol_opt: float = 1e-7, tol_feas: float = 1e-6,
    step_mode: str = "trust_clip",
    trust_clip: Optional[float] = 0.5,
    fixed_step_eta: float = 0.5,
    ls_mu: float = 10.0, ls_c1: float = 1e-4, ls_shrink: float = 0.5,
    ls_min_step: float = 1e-6, ls_max_backtracks: int = 25,
    reg_H: float = 1e-10, reg_dual: float = 1e-8,
    dual_step_mode: str = "scaled",
    # NEW:
    verbose: bool = False,
    print_every: int = 1,
):
    xgrid = make_xgrid(params, m=x_m)
    dgrid = make_dgrid(params, m=d_m, L=0.0)
    spec  = build_plspec(xgrid, K_pieces)

    gammas0 = np.log(np.ones(spec.K) * (params.s0 / (params.s0 + 1e-3)))
    theta = np.concatenate([gammas0, np.array([0.0])])
    lam   = np.zeros(dgrid.m)

    hist = {"J": [], "opt": [], "feas": [], "t": [], "d_norm_inf": [], "w_norm_inf": []}

    for it in range(max_iter):
        Jval, gJ_now, _ = objective_J(theta, spec, xgrid, params)
        zeta_now, A_now = pushforward_and_jac(theta, spec, dgrid, params)
        g_now = gJ_now + A_now.T @ lam
        hist["J"].append(Jval)
        hist["opt"].append(np.linalg.norm(g_now, np.inf))
        hist["feas"].append(np.linalg.norm(zeta_now, np.inf))
        if hist["opt"][-1] <= tol_opt and hist["feas"][-1] <= tol_feas:
            break

        dtheta, w, g, zeta, A, _ = sqp_direction_g2e(
            theta, lam, spec, xgrid, dgrid, params, reg_H=reg_H, reg_dual=reg_dual
        )

        t = 1.0
        if step_mode == "trust_clip":
            ninf = np.linalg.norm(dtheta, np.inf)
            if (trust_clip is not None) and (ninf > max(trust_clip, 0.0)):
                t = trust_clip / ninf
        elif step_mode == "fixed":
            t = float(fixed_step_eta)
        elif step_mode == "linesearch":
            phi0 = merit_phi_g2e(theta, spec, xgrid, dgrid, params, ls_mu)
            dir_deriv = merit_dirderiv_g2e(theta, spec, xgrid, dgrid, params, zeta, A, dtheta, ls_mu)
            t = 1.0; ok = False
            for _ in range(ls_max_backtracks):
                theta_try = theta + t * dtheta
                if merit_phi_g2e(theta_try, spec, xgrid, dgrid, params, ls_mu) <= phi0 + ls_c1 * t * dir_deriv:
                    ok = True; break
                t *= ls_shrink
                if t < ls_min_step: break
            if not ok and t < ls_min_step: t = ls_min_step
        else:
            raise ValueError("step_mode must be one of {'trust_clip','fixed','linesearch'}")

        # --- LOG + PRINT ---
        hist["t"].append(t)
        hist["d_norm_inf"].append(np.linalg.norm(dtheta, np.inf))
        hist["w_norm_inf"].append(np.linalg.norm(w, np.inf))
        if verbose and (it % print_every == 0):
            print(f"it={it:03d}  mode={step_mode:<10}  t={t:.3e}  |dθ|_inf={hist['d_norm_inf'][-1]:.3e}  "
                  f"J={Jval:.3e}  opt={hist['opt'][-1]:.3e}  feas={hist['feas'][-1]:.3e}")

        # update (optionally scale dual)
        theta = theta + t * dtheta
        lam   = lam + (t * w if (dual_step_mode == "scaled" or step_mode == "linesearch") else w)

    # pack results …
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, y, spec.knots)
    return {
        "theta": theta, "lambda": lam, "spec": spec,
        "a": a_i, "b": b_i, "y_knots": y,
        "xgrid": xgrid, "dgrid": dgrid, "history": hist,
        "params": params,
    }

# -----------------------
# Utilities for plotting / evaluation
# -----------------------
def learned_pushforward(sol, d: np.ndarray) -> np.ndarray:
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    mu0, s0 = sol["params"].mu0, sol["params"].s0
    K = len(a)
    Pf = np.zeros_like(d)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (d - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1])
        if np.any(valid):
            Pf[valid] += (1.0/ai) * normal_pdf(u[valid], mu0, s0)
    return Pf

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Problem
    params = G2EParams(mu0=0.0, s0=1.25, lam1=1.0)

    # A) Trust-clip
    # sol = sqp_solve_g2e(params, K_pieces=12, x_m=501, d_m=2000, step_mode="trust_clip", trust_clip=0.25, reg_H=1e-8, reg_dual=1e-8,
    #                     verbose=True, print_every=20)
    # print("Trust-clip: final J, opt, feas:", sol["history"]["J"][-1],
    #       sol["history"]["opt"][-1], sol["history"]["feas"][-1])

    # B) Fixed step
    # sol = sqp_solve_g2e(params, K_pieces=6,x_m=501, d_m=1000, step_mode="fixed", fixed_step_eta=0.25, reg_H=1e-8, reg_dual=1e-8)
    # print("Fixed step: final J, opt, feas:", sol["history"]["J"][-1],
    #       sol["history"]["opt"][-1], sol["history"]["feas"][-1])

    # C) Line search (robust)
    sol = sqp_solve_g2e(params, K_pieces=12, x_m=501, d_m=2000, step_mode="linesearch", ls_mu=50.0, ls_c1=1e-4, ls_shrink=0.5, ls_max_backtracks=25, reg_H=1e-8, reg_dual=1e-8)
    print("Line search: final J, opt, feas:", sol["history"]["J"][-1],
          sol["history"]["opt"][-1], sol["history"]["feas"][-1])

import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt

# --- Analytic map: Gaussian(mu0,s0) -> Exponential(lam1)
from scipy.special import erf  # <-- vectorized erf

def analytic_map_g2e(x: np.ndarray, mu0: float, s0: float, lam1: float) -> np.ndarray:
    """
    Analytic OT map: Gaussian(mu0, s0^2) → Exponential(lam1).
    """
    z = (x - mu0) / s0
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))  # vectorized erf
    Phi = np.clip(Phi, 1e-12, 1.0 - 1e-12)     # avoid log(0)
    return -(1.0 / lam1) * np.log(1.0 - Phi)

# --- Evaluate learned piecewise-linear map s(x) from your solver output
def eval_learned_map_pl(x: np.ndarray, sol: dict) -> np.ndarray:
    a = sol["a"]                 # slopes per piece (length K)
    b = sol["b"]                 # intercepts per piece (length K)
    knots = sol["spec"].knots    # x_0 < ... < x_K  (length K+1)
    K = len(a)

    # which piece for each x
    idx = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, K-1)
    xi  = knots[idx]
    ai  = a[idx]
    bi  = b[idx]
    return ai * x + bi  # equals y_i + a_i (x - x_i); both forms are equivalent

# --- L2 (w.r.t. source density) and Linf errors on a grid
def map_error_metrics_g2e(sol: dict, num_pts: int = 1201):
    mu0, s0, lam1 = sol["params"].mu0, sol["params"].s0, sol["params"].lam1
    xL, xR = sol["xgrid"].L, sol["xgrid"].R
    x = np.linspace(xL, xR, num_pts)

    s_analytic = analytic_map_g2e(x, mu0, s0, lam1)
    s_learned  = eval_learned_map_pl(x, sol)
    err = s_learned - s_analytic

    # weight by source density for an E[·] style metric
    from math import pi
    w = np.exp(-0.5 * ((x - mu0)/s0)**2) / (s0 * np.sqrt(2.0*pi))
    dx = (xR - xL) / (num_pts - 1)

    L2_mu  = np.sqrt(np.sum((err**2) * w) * dx)   # L2 under μ
    Linf   = np.max(np.abs(err))
    L1_mu  = np.sum(np.abs(err) * w) * dx
    return {"L2_mu": L2_mu, "Linf": Linf, "L1_mu": L1_mu}

# --- Plot: analytic vs learned map
def plot_maps_g2e(sol: dict,
                  X_RANGE=None,
                  SAVE_PATH: str = "g2e_maps.png",
                  num_pts_plot: int = 800,
                  title: str = "Gaussian→Exponential Transport Map: Analytic vs SQP"):
    mu0, s0, lam1 = sol["params"].mu0, sol["params"].s0, sol["params"].lam1
    if X_RANGE is None:
        xmin, xmax = sol["xgrid"].L, sol["xgrid"].R
    else:
        xmin, xmax = X_RANGE

    xs = np.linspace(xmin, xmax, num_pts_plot)
    ys_analytic = analytic_map_g2e(xs, mu0, s0, lam1)
    ys_learned  = eval_learned_map_pl(xs, sol)

    # y-limits
    ymin = min(ys_analytic.min(), ys_learned.min())
    ymax = max(ys_analytic.max(), ys_learned.max())
    pad = 0.05 * (ymax - ymin + 1e-12)
    ylo, yhi = ymin - pad, ymax + pad

    plt.figure(figsize=(8,6), dpi=120)
    plt.plot(xs, ys_analytic, linewidth=2, label="Analytic map T(x)")
    plt.plot(xs, ys_learned,  "--", linewidth=2, label="SQP piecewise-linear s(x)")
    plt.xlim(xmin, xmax); plt.ylim(ylo, yhi)
    plt.xlabel("x"); plt.ylabel("y = s(x)")
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    plt.show()
    return SAVE_PATH

# --- Convenience wrapper: compute + print metrics, then plot
def compare_maps_g2e(sol: dict,
                     X_RANGE=None,
                     SAVE_PATH: str = "g2e_maps.png",
                     num_pts_err: int = 1201,
                     num_pts_plot: int = 800):
    metrics = map_error_metrics_g2e(sol, num_pts=num_pts_err)
    print("=== Map error (learned vs analytic) ===")
    print(f"L2 under μ(x): {metrics['L2_mu']:.6e}")
    print(f"L1 under μ(x): {metrics['L1_mu']:.6e}")
    print(f"L∞ (sup over grid): {metrics['Linf']:.6e}")
    out = plot_maps_g2e(sol, X_RANGE=X_RANGE, SAVE_PATH=SAVE_PATH, num_pts_plot=num_pts_plot)
    return out, metrics
out_path, metrics = compare_maps_g2e(sol, X_RANGE=None, SAVE_PATH="g2e_maps.png")
print("Saved figure:", out_path)
print(metrics)

def pushforward_pdf_g2e(d: np.ndarray, sol: dict) -> np.ndarray:
    """
    Compute Pf(s)(d) for learned piecewise-linear map from Gaussian(mu0,s0) to Exp(lam1).
    Uses the piecewise (a_i,b_i) representation from sol.
    """
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    mu0, s0 = sol["params"].mu0, sol["params"].s0
    K = len(a)

    Pf = np.full_like(d, np.nan)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (d - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1])
        if np.any(valid):
            Pf[valid] = (1.0/ai) * normal_pdf(u[valid], mu0, s0)
    return Pf

def plot_source_and_targets_g2e(sol: dict, num_pts: int = 2001):
    """
    Two-panel figure:
      (1) Source mu(x) ~ Gaussian(mu0,s0)
      (2) Pushforward Pf(s)(d) vs true target nu(d) ~ Exp(lam1)
    """
    import matplotlib.pyplot as plt

    params = sol["params"]

    # grids (exclude extreme tails for clarity)
    x = np.linspace(sol["xgrid"].L, sol["xgrid"].R, num_pts)
    d = np.linspace(sol["dgrid"].L, sol["dgrid"].R, num_pts)

    mu_x = normal_pdf(x, params.mu0, params.s0)
    nu_d = exp_pdf(d, params.lam1)
    pf_d = pushforward_pdf_g2e(d, sol)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source Gaussian
    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title(r"Source density $\mu(x)$ (Gaussian)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: pushforward vs true exponential
    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label=r"Pushforward $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label=r"True target $\nu(d)$")
    ax.set_title("Target densities on $d$")
    ax.set_xlabel("d")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Gaussian→Exponential: Source vs. Learned Target vs. True Target",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    plt.show()
    return

plot_source_and_targets_g2e(sol, num_pts=2001)

"""## New Ver

### Quantile Grids
When `use_quantile_grids=True`, the solver switches to **quantile-based grids** for both source and target, plus quantile-based PL knots.

- **Source grid (`xgrid`)** → `make_xgrid_quantile()`  
  Uses Gaussian inverse CDF (`Φ⁻¹`) nodes → equal-mass samples with `dx = 1/m`.

- **Target grid (`dgrid`)** → `make_dgrid_quantile()`  
  Uses exponential inverse CDF (`F⁻¹(u) = -ln(1-u)/λ`) up to `pmax=0.999`.

- **PL knots (`spec`)** → `build_plspec_from_quantiles()`  
  Knot locations = Gaussian quantiles (`norm_ppf(u)`).

- **Objective weighting:**  
  `weighted_by_pdf=False` → uniform weights over quantile points (equal CDF intervals).

If `use_quantile_grids=False`, it instead uses uniform linspace grids with pdf-weighted integration.
"""

# Gaussian → Exponential OT (1D), monotone PL map + SQP
# Experiment runner comparing: (A) uniform grids vs (B) quantile-based grids,
# with optional adaptive knot refinement once mid-training.
#
# Notes for this run cell:
# - Uses matplotlib (no seaborn), single-plot figures, no explicit colors.
# - Saves result figures under /mnt/data and displays them inline.
# - Prints summary metrics (objective, feasibility, optimality) for quick comparison.

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional

# -----------------------------
# Densities & helper utilities
# -----------------------------
def normal_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

def exp_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

# -----------------------------
# Problem + grid specs
# -----------------------------
@dataclass
class G2EParams:
    mu0: float    # source Gaussian mean
    s0:  float    # source Gaussian std
    lam1: float   # target Exponential rate

@dataclass
class XGrid:
    L: float
    R: float
    m: int
    x: np.ndarray  # shape (m,)
    dx: float      # integration weight (can be 1/m for quantile grids)

@dataclass
class DGrid:
    L: float
    R: float
    m: int
    d: np.ndarray  # shape (m,)

def domain_gauss95(params: G2EParams, k: float = 2.8) -> Tuple[float, float]:
    """A bit wider than 95% mass to reduce boundary effects."""
    L = params.mu0 - k * params.s0
    R = params.mu0 + k * params.s0
    return L, R

def make_xgrid(params: G2EParams, m: int, L: Optional[float]=None, R: Optional[float]=None) -> XGrid:
    if L is None or R is None:
        L2, R2 = domain_gauss95(params, k=2.8)
        if L is None: L = L2
        if R is None: R = R2
    x = np.linspace(L, R, m)
    dx = (R - L) / (m - 1)
    return XGrid(L, R, m, x, dx)

# ---- Quantile-based X grid (under source Gaussian) ----
def make_xgrid_quantile(params: G2EParams, m: int, clip_sigma: float = 4.0) -> XGrid:
    # inverse-CDF nodes (open interval to avoid infinities)
    # We'll implement a simple normal PPF using numpy + approximation
    # Since internet is off, we implement a stable erf^-1 based ppf.
    # However, to avoid complexity, we'll approximate with grid search on CDF.
    # For accuracy and simplicity here, we approximate via scipy-like formula using erfinv.
    import math
    from math import sqrt
    # erfinv approximation (Winitzki) for good-enough accuracy
    def erfinv(y):
        a = 0.147  # Winitzki constant
        s = np.sign(y)
        ln = np.log(1.0 - y*y)
        first = 2.0/(np.pi*a) + ln/2.0
        val = s * np.sqrt( np.sqrt(first*first - ln/a) - first )
        return val

    def norm_ppf(u, mu, sigma):
        # PPF via inverse error function
        # Φ^{-1}(u) = sqrt(2) * erfinv(2u-1)
        u = np.asarray(u)
        u = np.clip(u, 1e-12, 1-1e-12)
        return mu + sigma * np.sqrt(2.0) * erfinv(2.0*u - 1.0)

    u = (np.arange(m) + 0.5) / m
    x = norm_ppf(u, params.mu0, params.s0)
    # optional clipping to finite window for safety
    L = params.mu0 - clip_sigma * params.s0
    R = params.mu0 + clip_sigma * params.s0
    x = np.clip(x, L, R)
    # Since expectation is under f, use equal-CDF weights: E_f[h] ≈ (1/m) Σ h(x_j)
    return XGrid(L=L, R=R, m=m, x=x, dx=1.0/m)

def make_dgrid(params: G2EParams, m: int, p: float = 0.999, L: float = 0.0) -> DGrid:
    """Right-quantile coverage for Exp(lam1)."""
    R = -np.log(1.0 - p) / params.lam1
    d = np.linspace(L, R, m)
    return DGrid(L, R, m, d)

# ---- Quantile-based D grid (under target exponential) ----
def make_dgrid_quantile(params: G2EParams, m: int, pmax: float = 0.999):
    # exponential PPF: F^{-1}(u) = -ln(1-u)/lam, but we avoid u close to 1
    u = np.linspace(1.0/(m+1), pmax, m)
    d = -np.log(1.0 - u) / params.lam1
    return DGrid(L=float(d.min()), R=float(d.max()), m=m, d=d)

# --------------------------------------------
# Monotone piecewise-linear map specification
# --------------------------------------------
@dataclass
class PLSpec:
    knots: np.ndarray     # x_0 < ... < x_K (K+1 knots)
    K: int                # number of pieces (K)

def build_plspec(xgrid: XGrid, K: int) -> PLSpec:
    knots = np.linspace(xgrid.L, xgrid.R, K+1)
    return PLSpec(knots=knots, K=K)

def build_plspec_from_quantiles(params: G2EParams, K: int, clip_sigma: float = 4.0) -> PLSpec:
    eps = 1e-6
    u_knots = np.linspace(0.0, 1.0, K+1)
    u_knots = np.clip(u_knots, eps, 1-eps)
    # use the same norm_ppf as above
    from math import sqrt
    def erfinv(y):
        a = 0.147
        s = np.sign(y)
        ln = np.log(1.0 - y*y)
        first = 2.0/(np.pi*a) + ln/2.0
        val = s * np.sqrt( np.sqrt(first*first - ln/a) - first )
        return val
    def norm_ppf(u, mu, sigma):
        u = np.asarray(u)
        u = np.clip(u, 1e-12, 1-1e-12)
        return mu + sigma * np.sqrt(2.0) * erfinv(2.0*u - 1.0)
    knots = norm_ppf(u_knots, params.mu0, params.s0)
    knots = np.clip(knots, params.mu0 - clip_sigma*params.s0, params.mu0 + clip_sigma*params.s0)
    return PLSpec(knots=knots, K=K)

def unpack_params(theta: np.ndarray, K: int):
    """From theta=[gamma_0..gamma_{K-1}, y0] -> (a (slopes), y0)."""
    gammas = theta[:K]
    y0     = theta[K]
    a = np.exp(gammas)  # slopes > 0 (monotone)
    return a, y0

def pl_continuity_y(a: np.ndarray, y0: float, knots: np.ndarray) -> np.ndarray:
    """Return y_i = s(x_i) for i=0..K, given slopes a_i and y0=s(x0)."""
    K = len(a)
    y = np.zeros(K+1)
    y[0] = y0
    for i in range(K):
        y[i+1] = y[i] + a[i] * (knots[i+1] - knots[i])
    return y  # length K+1

def piece_affine_params(a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """Return (a_i, b_i) for each piece so s(x)=a_i x + b_i on [x_i,x_{i+1}]."""
    K = len(a)
    b = np.zeros(K)
    for i in range(K):
        b[i] = y[i] - a[i] * knots[i]
    return a, b

def eval_s_and_jac_s(x: np.ndarray, a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """
    Evaluate s(x) and ∂s/∂θ for θ=(gamma[0:K], y0).
    Returns:
      svals: shape (m,)
      Js:    shape (m, K+1)  (columns: d/d gamma_0..gamma_{K-1}, d/d y0)
    """
    K = len(a)
    m = len(x)
    svals = np.zeros(m)
    Js = np.zeros((m, K+1))

    # which piece each x lies in
    idx = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, K-1)

    for t in range(m):
        i = idx[t]         # active piece index
        xi = knots[i]
        # s(x) on piece i
        svals[t] = y[i] + a[i] * (x[t] - xi)

        # d/d y0: every y_i shifts by +1, so ∂s/∂y0 = 1
        Js[t, K] = 1.0

        # d/d gamma_k:
        # for k < i: y[i] depends on a_k via Δx_k
        for k in range(i):
            Js[t, k] += a[k] * (knots[k+1] - knots[k])  # ∂y[i]/∂gamma_k = a_k Δx_k
        # for k = i: slope term
        Js[t, i] += a[i] * (x[t] - xi)
        # for k > i: 0
    return svals, Js

# --------------------------------------------
# Objective J(s) and Gauss–Newton (g, H)
# --------------------------------------------
def objective_J(theta: np.ndarray, spec: PLSpec, xgrid: XGrid, params: G2EParams, weighted_by_pdf: bool):
    """
    J = 1/2 E[(s(X) - X)^2].
    If weighted_by_pdf=True: interprets xgrid.dx as a spatial step and weights by f(x)dx (uniform x-grid).
    If weighted_by_pdf=False: interprets xgrid.dx as equal-CDF weight (1/m) (quantile grid).
    Return (J, grad, H) with Gauss–Newton approximation.
    """
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    svals, Js = eval_s_and_jac_s(xgrid.x, a, y, spec.knots)

    err = (svals - xgrid.x)
    if weighted_by_pdf:
        w = normal_pdf(xgrid.x, params.mu0, params.s0) * xgrid.dx  # quadrature weights on uniform x-grid
    else:
        w = np.ones_like(xgrid.x) * xgrid.dx                       # equal-CDF weights (dx = 1/m)

    Jval = 0.5 * np.sum((err**2) * w)

    # Gauss–Newton: g ≈ Σ err * (∂s/∂θ) * w ; H ≈ Σ (∂s/∂θ)(∂s/∂θ)^T * w
    g = (Js.T @ (err * w))
    H = (Js.T * w) @ Js + 1e-10 * np.eye(spec.K+1)  # tiny damping
    return Jval, g, H

# -------------------------------------------------
# Constraint zeta(d) = Pf(s)(d) - g(d) and Jacobian
# -------------------------------------------------
def pushforward_and_jac(theta: np.ndarray, spec: PLSpec, dgrid: DGrid, params: G2EParams):
    """
    Pf(s)(d) = sum_i (1/a_i) f((d-b_i)/a_i) * 1_{u in [x_i,x_{i+1}]},
    with f Gaussian(mu0, s0). Return Pf (m,), A (m, K+1) = d Pf / d theta.
    """
    mu0, s0, lam1 = params.mu0, params.s0, params.lam1
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, y, spec.knots)

    m = dgrid.m
    Pf = np.zeros(m)
    A  = np.zeros((m, spec.K+1))

    # Precompute derivatives of b_i wrt gammas and y0
    K = spec.K
    dx_segs = np.diff(spec.knots)
    db_dgamma = np.zeros((K, K))  # rows i, cols k
    db_dy0    = np.ones(K)        # ∂b_i/∂y0 = 1
    for i in range(K):
        # via y_i for k < i
        for k in range(i):
            db_dgamma[i, k] += a[k] * dx_segs[k]
        # minus a_i * x_i for k=i (since b_i = y_i - a_i x_i)
        db_dgamma[i, i] += -a[i] * spec.knots[i]
        # k>i: no effect

    d = dgrid.d
    inv_s02 = 1.0 / (s0 * s0)

    for i in range(K):
        ai, bi = a_i[i], b_i[i]
        xi, xip1 = spec.knots[i], spec.knots[i+1]

        u = (d - bi) / ai
        valid = (u >= xi) & (u <= xip1)
        if not np.any(valid):
            continue

        du = u[valid]
        f_u = normal_pdf(du, mu0, s0)

        # Pf add
        Pf[valid] += (1.0 / ai) * f_u

        # Local partials (for valid entries)
        # z = (1/a) f(u), u=(d-b)/a
        # ∂z/∂a = (f(u)/a^2) * ( -1 + u(u-μ0)/s0^2 )
        # ∂z/∂b =  f(u) * (u-μ0)/(a^2 s0^2)
        dz_da = (f_u / (ai * ai)) * (-1.0 + du * (du - mu0) * inv_s02)
        dz_db = f_u * ((du - mu0) * inv_s02) / (ai * ai)

        # Chain to parameters gamma_k and y0:
        for k in range(K):
            contrib = 0.0
            if k == i:
                contrib += np.mean(dz_da * ai * 0 + 1)  # placeholder to keep shape
            # accumulate vectorized
        # vectorized update (avoid the loop above for speed/clarity):
        # build per-k contributions
        # For k == i:
        A[valid, i] += dz_da * ai
        # For k < i (via db dependency):
        if i > 0:
            for k in range(i):
                if db_dgamma[i, k] != 0.0:
                    A[valid, k] += dz_db * db_dgamma[i, k]
        # y0 column
        A[valid, K] += dz_db * db_dy0[i]

    # subtract target pdf
    g_d = exp_pdf(d, lam1)
    zeta = Pf - g_d
    return zeta, A

# -----------------------------
# KKT direction (no update)
# -----------------------------
def sqp_direction_g2e(
    theta: np.ndarray, lam: np.ndarray,
    spec: PLSpec, xgrid: XGrid, dgrid: DGrid, params: G2EParams,
    weighted_by_pdf: bool,
    reg_H: float = 1e-10, reg_dual: float = 1e-8
):
    """Single KKT solve: return (dtheta, w, g, zeta, A, Jval)."""
    Jval, gJ, HJ = objective_J(theta, spec, xgrid, params, weighted_by_pdf=weighted_by_pdf)
    zeta, A = pushforward_and_jac(theta, spec, dgrid, params)
    g = gJ + A.T @ lam
    H_reg = HJ + reg_H * np.eye(HJ.shape[0])

    KKT = np.block([
        [H_reg,              A.T],
        [A,       -reg_dual * np.eye(A.shape[0])]
    ])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    dtheta = sol[:theta.size]
    w = sol[theta.size:]
    return dtheta, w, g, zeta, A, Jval

# -----------------------------
# Merit function (Armijo LS)
# -----------------------------
def merit_phi_g2e(theta, spec, xgrid, dgrid, params, mu: float, weighted_by_pdf: bool):
    Jval, _, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf=weighted_by_pdf)
    zeta, _ = pushforward_and_jac(theta, spec, dgrid, params)
    return float(Jval + 0.5 * mu * (zeta @ zeta))

def merit_dirderiv_g2e(theta, spec, xgrid, dgrid, params, zeta, A, dtheta, mu: float, weighted_by_pdf: bool):
    # ∇φ = ∇J + μ A^T ζ
    _, gJ, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf=weighted_by_pdf)
    grad_phi = gJ + mu * (A.T @ zeta)
    return float(grad_phi @ dtheta)

# -----------------------------
# Full SQP driver (3 step modes) + grid options
# -----------------------------
def sqp_solve_g2e(
    params: G2EParams,
    K_pieces: int = 8,
    x_m: int = 501, d_m: int = 151,
    max_iter: int = 120,
    tol_opt: float = 1e-7, tol_feas: float = 5e-5,
    step_mode: str = "linesearch",
    trust_clip: Optional[float] = 0.5,
    fixed_step_eta: float = 0.5,
    ls_mu: float = 10.0, ls_c1: float = 1e-4, ls_shrink: float = 0.5,
    ls_min_step: float = 1e-6, ls_max_backtracks: int = 25,
    reg_H: float = 1e-10, reg_dual: float = 1e-8,
    dual_step_mode: str = "scaled",
    verbose: bool = False,
    print_every: int = 10,
    use_quantile_grids: bool = True,
):
    if use_quantile_grids:
        xgrid = make_xgrid_quantile(params, m=x_m)
        dgrid = make_dgrid_quantile(params, m=d_m, pmax=0.999)
        spec  = build_plspec_from_quantiles(params, K_pieces)
        weighted_by_pdf = False
    else:
        xgrid = make_xgrid(params, m=x_m)
        dgrid = make_dgrid(params, m=d_m, L=0.0)
        spec  = build_plspec(xgrid, K_pieces)
        weighted_by_pdf = True

    gammas0 = np.log(np.ones(spec.K) * (params.s0 / (params.s0 + 1e-3)))
    theta = np.concatenate([gammas0, np.array([0.0])])
    lam   = np.zeros(dgrid.m)

    hist = {"J": [], "opt": [], "feas": [], "t": [], "d_norm_inf": [], "w_norm_inf": []}

    for it in range(max_iter):
        Jval, gJ_now, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf=weighted_by_pdf)
        zeta_now, A_now = pushforward_and_jac(theta, spec, dgrid, params)
        g_now = gJ_now + A_now.T @ lam
        hist["J"].append(Jval)
        hist["opt"].append(np.linalg.norm(g_now, np.inf))
        hist["feas"].append(np.linalg.norm(zeta_now, np.inf))
        if hist["opt"][-1] <= tol_opt and hist["feas"][-1] <= tol_feas:
            break

        dtheta, w, g, zeta, A, _ = sqp_direction_g2e(
            theta, lam, spec, xgrid, dgrid, params, weighted_by_pdf=weighted_by_pdf,
            reg_H=reg_H, reg_dual=reg_dual
        )

        t = 1.0
        if step_mode == "trust_clip":
            ninf = np.linalg.norm(dtheta, np.inf)
            if (trust_clip is not None) and (ninf > max(trust_clip, 0.0)):
                t = trust_clip / ninf
        elif step_mode == "fixed":
            t = float(fixed_step_eta)
        elif step_mode == "linesearch":
            phi0 = merit_phi_g2e(theta, spec, xgrid, dgrid, params, ls_mu, weighted_by_pdf=weighted_by_pdf)
            dir_deriv = merit_dirderiv_g2e(theta, spec, xgrid, dgrid, params, zeta, A, dtheta, ls_mu, weighted_by_pdf=weighted_by_pdf)
            t = 1.0; ok = False
            for _ in range(ls_max_backtracks):
                theta_try = theta + t * dtheta
                if merit_phi_g2e(theta_try, spec, xgrid, dgrid, params, ls_mu, weighted_by_pdf=weighted_by_pdf) <= phi0 + ls_c1 * t * dir_deriv:
                    ok = True; break
                t *= ls_shrink
                if t < ls_min_step: break
            if not ok and t < ls_min_step: t = ls_min_step
        else:
            raise ValueError("step_mode must be one of {'trust_clip','fixed','linesearch'}")

        # --- LOG + PRINT ---
        hist["t"].append(t)
        hist["d_norm_inf"].append(np.linalg.norm(dtheta, np.inf))
        hist["w_norm_inf"].append(np.linalg.norm(w, np.inf))
        if verbose and (it % print_every == 0):
            print(f"it={it:03d}  mode={step_mode:<10}  t={t:.3e}  |dθ|_inf={hist['d_norm_inf'][-1]:.3e}  "
                  f"J={Jval:.3e}  opt={hist['opt'][-1]:.3e}  feas={hist['feas'][-1]:.3e}")

        # update (optionally scale dual)
        theta = theta + t * dtheta
        lam   = lam + (t * w if (dual_step_mode == "scaled" or step_mode == "linesearch") else w)

    # pack results …
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, y, spec.knots)
    return {
        "theta": theta, "lambda": lam, "spec": spec,
        "a": a_i, "b": b_i, "y_knots": y,
        "xgrid": xgrid, "dgrid": dgrid, "history": hist,
        "params": params, "weighted_by_pdf": weighted_by_pdf
    }

# -----------------------
# Utilities for plotting / evaluation
# -----------------------
def learned_pushforward(sol, d: np.ndarray) -> np.ndarray:
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    mu0, s0 = sol["params"].mu0, sol["params"].s0
    K = len(a)
    Pf = np.zeros_like(d)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (d - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1])
        if np.any(valid):
            Pf[valid] += (1.0/ai) * normal_pdf(u[valid], mu0, s0)
    return Pf

# -----------------------
# Run two experiments
# -----------------------
params = G2EParams(mu0=0.0, s0=1.0, lam1=1.0)

# sol_uniform = sqp_solve_g2e(
#     params, K_pieces=8, x_m=501, d_m=201, max_iter=150,
#     use_quantile_grids=False, step_mode="linesearch", verbose=False
# )
sol   = sqp_solve_g2e(
    params, K_pieces=8, x_m=501, d_m=201, max_iter=150,
    use_quantile_grids=True, step_mode="linesearch", verbose=False
)

# Retry with math.erf (numpy.erf may be unavailable in this environment)
import numpy as np
import matplotlib.pyplot as plt
import math

def eval_map_on_dense(sol, xs):
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    idx = np.clip(np.searchsorted(knots, xs, side='right') - 1, 0, len(a)-1)
    return a[idx]*xs + b[idx]

def analytic_map_gauss_to_exp(xs, mu, sigma, lam):
    z = (xs - mu) / (sigma + 1e-12)
    # vectorize math.erf
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    Phi = np.clip(Phi, 1e-12, 1.0 - 1e-12)
    return -np.log(1.0 - Phi) / lam

# choose a solution
chosen = None
for name in ["sol", "sol_quant", "sol_uniform"]:
    if name in globals():
        chosen = globals()[name]
        break
if chosen is None:
    raise RuntimeError("No solution dict found. Please run your SQP solver to create `sol` (or `sol_quant` / `sol_uniform`).")

params = chosen["params"]
mu0, s0, lam1 = params.mu0, params.s0, params.lam1

xmin = mu0 - 3.5*s0
xmax = mu0 + 3.5*s0
xs = np.linspace(xmin, xmax, 600)

ys_analytic = analytic_map_gauss_to_exp(xs, mu0, s0, lam1)
ys_sqp      = eval_map_on_dense(chosen, xs)

ymin = float(min(ys_analytic.min(), ys_sqp.min()))
ymax = float(max(ys_analytic.max(), ys_sqp.max()))
pad = 0.05 * (ymax - ymin + 1e-12)
ylo, yhi = ymin - pad, ymax + pad

plt.figure(figsize=(7, 5))
plt.plot(xs, ys_analytic, linewidth=2, label="Analytic (quantile map)")
plt.plot(xs, ys_sqp, linestyle="--", linewidth=2, label="Computed (SQP, PL)")
plt.xlim(xmin, xmax)
plt.ylim(ylo, yhi)
plt.xlabel("x")
plt.ylabel("y = s(x)")
plt.title("Gaussian → Exponential Transport Map: Analytic vs Computed")
plt.legend()
plt.tight_layout()

# SAVE_PATH = "/mnt/data/transport_maps_gauss_to_exp.png"
# plt.savefig(SAVE_PATH, dpi=200)
plt.show()

# print("Saved to:", SAVE_PATH)
# SAVE_PATH

import numpy as np
import matplotlib.pyplot as plt

def pushforward_pdf_g2e(d: np.ndarray, sol: dict) -> np.ndarray:
    """
    Compute Pf(s)(d) for learned piecewise-linear map from Gaussian(mu0,s0) to Exp(lam1).
    Uses the piecewise (a_i,b_i) representation from sol. Outside the image of s,
    we set Pf=0 for a continuous-looking curve.
    """
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    mu0, s0 = sol["params"].mu0, sol["params"].s0
    K = len(a)
    # start at zero (nicer than NaNs for plotting)
    Pf = np.zeros_like(d, dtype=float)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (d - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1])
        if np.any(valid):
            z = (u[valid] - mu0) / s0
            Pf[valid] = np.exp(-0.5 * z * z) / (s0 * np.sqrt(2.0 * np.pi) * ai)
    return Pf

def plot_source_and_targets_g2e_side_by_side(sol: dict, num_pts: int = 2001):
    """
    One figure with two panels:
      (1) Source μ(x) ~ Gaussian(μ0, s0)
      (2) Pushforward Pf(s)(d) vs true target ν(d) ~ Exp(λ1)
    """
    # --- helpers (inline to keep it self-contained) ---
    def normal_pdf(x, mu, s):
        z = (x - mu) / s
        return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

    def exp_pdf(x, lam):
        out = lam * np.exp(-lam * np.maximum(x, 0.0))
        out[x < 0] = 0.0
        return out

    params = sol["params"]

    # grids (match your ranges)
    x = np.linspace(sol["xgrid"].L, sol["xgrid"].R, num_pts)
    d = np.linspace(sol["dgrid"].L, sol["dgrid"].R, num_pts)

    mu_x = normal_pdf(x, params.mu0, params.s0)
    nu_d = exp_pdf(d, params.lam1)
    pf_d = pushforward_pdf_g2e(d, sol)

    # --- figure with 1×2 subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source Gaussian
    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title(r"Source density $\mu(x)$ (Gaussian)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: pushforward vs true exponential
    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label=r"Pushforward $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label=r"True target $\nu(d)$")
    ax.set_title("Target densities on $d$")
    ax.set_xlabel("d")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Gaussian→Exponential: Source vs. Learned Target vs. True Target",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    return fig, axes

# Usage:
fig, axes = plot_source_and_targets_g2e_side_by_side(sol, num_pts=2001)
# fig.savefig("g2e_side_by_side.png", dpi=200)
plt.show()

