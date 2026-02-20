# -*- coding: utf-8 -*-
"""Exponential -> Gaussian notebook section.

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


"""# Exponential to Gaussian"""

# ============================================================
# Exponential → Gaussian OT (1D), monotone PL map + SQP
# - Source:  X ~ Exp(lam0)             (support x ≥ 0)
# - Target:  Y ~ N(mu1, s1^2)
# - Map:     s(x) piecewise-affine with slopes a_i = exp(gamma_i) > 0
# - Objective: J = 1/2 E[(s(X) - X)^2]
# - Constraints: pushforward density on y-grid matches N(mu1, s1^2)
# - Step-size modes: "trust_clip", "fixed", "linesearch"
# - Improvements: deep-tail quantile grids, analytic init, optional adapt split
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# -----------------------------
# PDFs
# -----------------------------
def exp_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

def normal_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

# -----------------------------
# Problem + grid specs
# -----------------------------
@dataclass
class E2GParams:
    lam0: float   # source Exp rate
    mu1:  float   # target Gaussian mean
    s1:   float   # target Gaussian std

@dataclass
class XGrid:
    L: float
    R: float
    m: int
    x: np.ndarray  # (m,)
    dx: float      # quadrature weight per node (use 1/m for quantile grids)

@dataclass
class YGrid:
    L: float
    R: float
    m: int
    y: np.ndarray  # (m,)

# ---- Default domains (for uniform grids) ----
def domain_exp(params: E2GParams, p: float = 0.999) -> Tuple[float, float]:
    L = 0.0
    R = -np.log(1.0 - p) / params.lam0
    return L, R

def domain_gauss(params: E2GParams, k: float = 4.0) -> Tuple[float, float]:
    L = params.mu1 - k * params.s1
    R = params.mu1 + k * params.s1
    return L, R

# ---- Uniform grids ----
def make_xgrid_uniform(params: E2GParams, m: int, L: Optional[float]=None, R: Optional[float]=None) -> XGrid:
    if L is None or R is None:
        L2, R2 = domain_exp(params)
        if L is None: L = L2
        if R is None: R = R2
    x = np.linspace(L, R, m)
    dx = (R - L) / (m - 1)
    return XGrid(L, R, m, x, dx)

def make_ygrid_uniform(params: E2GParams, m: int, L: Optional[float]=None, R: Optional[float]=None) -> YGrid:
    if L is None or R is None:
        L2, R2 = domain_gauss(params)
        if L is None: L = L2
        if R is None: R = R2
    y = np.linspace(L, R, m)
    return YGrid(L, R, m, y)

# ---- Quantile helpers (no SciPy) ----
def _erfinv(y):
    # Winitzki approximation — good enough for grids/initialization
    a = 0.147
    s = np.sign(y)
    ln = np.log(1.0 - y*y)
    first = 2.0/(np.pi*a) + ln/2.0
    return s * np.sqrt(np.sqrt(first*first - ln/a) - first)

def _norm_ppf(u, mu, s):
    u = np.asarray(u)
    u = np.clip(u, 1e-300, 1-1e-16)
    return mu + s * np.sqrt(2.0) * _erfinv(2.0*u - 1.0)

# ---- Deep-tail quantile grids ----
def make_xgrid_quantile(params: E2GParams, m: int, p_lo: float = 1e-10, p_hi: float = 1-1e-10) -> XGrid:
    # Exp PPF: F^{-1}(u) = -ln(1-u)/lam0
    u = np.linspace(p_lo, p_hi, m)
    x = -np.log(1.0 - u) / params.lam0
    return XGrid(float(x.min()), float(x.max()), m, x, dx=1.0/m)  # equal-CDF weights

def make_ygrid_quantile(params: E2GParams, m: int, p_lo: float = 1e-10, p_hi: float = 1-1e-10) -> YGrid:
    u = np.linspace(p_lo, p_hi, m)
    y = _norm_ppf(u, params.mu1, params.s1)
    return YGrid(float(y.min()), float(y.max()), m, y)

# --------------------------------------------
# PL map specification
# --------------------------------------------
@dataclass
class PLSpec:
    knots: np.ndarray     # x_0 < ... < x_K (K+1 knots on source domain)
    K: int

def build_plspec_uniform(xgrid: XGrid, K: int) -> PLSpec:
    knots = np.linspace(xgrid.L, xgrid.R, K+1)
    return PLSpec(knots=knots, K=K)

def build_plspec_from_exp_quantiles(params: E2GParams, K: int, p_lo: float=1e-10, p_hi: float=1-1e-10) -> PLSpec:
    u = np.linspace(p_lo, p_hi, K+1)
    knots = -np.log(1.0 - u) / params.lam0
    return PLSpec(knots=knots, K=K)

# --------------------------------------------
# Parameter unpacking and map evaluation
# --------------------------------------------
def unpack_params(theta: np.ndarray, K: int):
    gammas = theta[:K]
    y0     = theta[K]
    a = np.exp(gammas)  # positive slopes
    return a, y0

def pl_continuity_y(a: np.ndarray, y0: float, knots: np.ndarray) -> np.ndarray:
    K = len(a)
    y = np.zeros(K+1)
    y[0] = y0
    for i in range(K):
        y[i+1] = y[i] + a[i] * (knots[i+1] - knots[i])
    return y

def piece_affine_params(a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    K = len(a)
    b = np.zeros(K)
    for i in range(K):
        b[i] = y[i] - a[i] * knots[i]
    return a, b

def eval_s_and_jac_s(x: np.ndarray, a: np.ndarray, y: np.ndarray, knots: np.ndarray):
    """
    Evaluate s(x) and ∂s/∂θ for θ=(gamma[0:K], y0).
    Returns:
      svals: (m,)
      Js:    (m, K+1), columns = d/d gamma_0..gamma_{K-1}, d/d y0
    """
    K = len(a)
    m = len(x)
    svals = np.zeros(m)
    Js = np.zeros((m, K+1))

    idx = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, K-1)

    for t in range(m):
        i = idx[t]
        xi = knots[i]
        svals[t] = y[i] + a[i] * (x[t] - xi)
        Js[t, K] = 1.0  # d/d y0 passes through all y_i

        # d/d gamma_k contributions
        for k in range(i):
            Js[t, k] += a[k] * (knots[k+1] - knots[k])
        Js[t, i] += a[i] * (x[t] - xi)

    return svals, Js

# --------------------------------------------
# Analytic-quantile initialization on knots
# --------------------------------------------
def init_theta_from_analytic(params: E2GParams, spec: PLSpec):
    # Analytic optimal: s*(x) = mu1 + s1 * Φ^{-1}(1 - e^{-lam0 x})
    u_knots = 1.0 - np.exp(-params.lam0 * np.maximum(spec.knots, 0.0))
    y_star = _norm_ppf(u_knots, params.mu1, params.s1)
    dx = np.diff(spec.knots)
    dy = np.diff(y_star)
    a_init = np.maximum(dy / dx, 1e-12)  # guard positivity/underflow
    gammas0 = np.log(a_init)
    y0 = y_star[0]
    return np.concatenate([gammas0, np.array([y0])]), y_star

# --------------------------------------------
# Objective and Gauss–Newton pieces
# --------------------------------------------
def objective_J(theta: np.ndarray, spec: PLSpec, xgrid: XGrid, params: E2GParams, weighted_by_pdf: bool):
    a, y0 = unpack_params(theta, spec.K)
    y = pl_continuity_y(a, y0, spec.knots)
    svals, Js = eval_s_and_jac_s(xgrid.x, a, y, spec.knots)

    err = (svals - xgrid.x)
    if weighted_by_pdf:
        w = exp_pdf(xgrid.x, params.lam0) * xgrid.dx    # uniform x-grid: f_X dx
    else:
        w = np.ones_like(xgrid.x) * xgrid.dx            # quantile x-grid: equal-CDF weights

    Jval = 0.5 * np.sum((err**2) * w)
    g = (Js.T @ (err * w))
    H = (Js.T * w) @ Js + 1e-10 * np.eye(spec.K+1)
    return Jval, g, H

# -------------------------------------------------
# Constraint zeta(y) = Pf(s)(y) - g(y) and Jacobian
# -------------------------------------------------
def pushforward_and_jac(theta: np.ndarray, spec: PLSpec, ygrid: YGrid, params: E2GParams):
    """
    Source f_X = Exp(lam0). On piece i: s(x)=a_i x + b_i, x∈[x_i,x_{i+1}], a_i>0.
    Pf(y) = sum_i (1/a_i) f_X((y - b_i)/a_i) * 1_{ (y-b_i)/a_i ∈ [x_i,x_{i+1}] }.
    Returns Pf (m,), A (m, K+1) = d Pf / d theta.
    """
    lam0 = params.lam0
    a, y0 = unpack_params(theta, spec.K)
    yk = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, yk, spec.knots)

    m = ygrid.m
    Pf = np.zeros(m)
    A  = np.zeros((m, spec.K+1))

    K = spec.K
    dx_segs = np.diff(spec.knots)
    db_dgamma = np.zeros((K, K))  # rows i, cols k
    db_dy0    = np.ones(K)
    for i in range(K):
        for k in range(i):
            db_dgamma[i, k] += a[k] * dx_segs[k]
        db_dgamma[i, i] += -a[i] * spec.knots[i]

    y = ygrid.y

    for i in range(K):
        ai, bi = a_i[i], b_i[i]
        xi, xip1 = spec.knots[i], spec.knots[i+1]

        u = (y - bi) / ai
        valid = (u >= xi) & (u <= xip1) & (u >= 0.0)   # source support x≥0
        if not np.any(valid):
            continue

        du = u[valid]
        f_u = exp_pdf(du, lam0)

        Pf[valid] += (1.0 / ai) * f_u

        # z = (1/a) f(u), u=(y-b)/a ; f(u)=lam0 exp(-lam0 u) 1_{u≥0}
        # ∂z/∂a = (f(u)/a^2) * ( -1 + lam0*u )
        # ∂z/∂b =  f(u) * ( lam0 / a^2 )
        dz_da = (f_u / (ai * ai)) * (-1.0 + lam0 * du)
        dz_db =  f_u * (lam0 / (ai * ai))

        # chain to gamma_k
        A[valid, i] += dz_da * ai  # ∂a_i/∂gamma_i = a_i
        if i > 0:
            for k in range(i):
                if db_dgamma[i, k] != 0.0:
                    A[valid, k] += dz_db * db_dgamma[i, k]
        # y0 column
        A[valid, K] += dz_db * db_dy0[i]

    # subtract target pdf g(y) = N(mu1, s1^2)
    g_y = normal_pdf(y, params.mu1, params.s1)
    zeta = Pf - g_y
    return zeta, A

# -----------------------------
# KKT direction
# -----------------------------
def sqp_direction_e2g(
    theta: np.ndarray, lam: np.ndarray,
    spec: PLSpec, xgrid: XGrid, ygrid: YGrid, params: E2GParams,
    weighted_by_pdf: bool,
    reg_H: float = 1e-10, reg_dual: float = 1e-8
):
    Jval, gJ, HJ = objective_J(theta, spec, xgrid, params, weighted_by_pdf)
    zeta, A = pushforward_and_jac(theta, spec, ygrid, params)
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
def merit_phi_e2g(theta, spec, xgrid, ygrid, params, mu: float, weighted_by_pdf: bool):
    Jval, _, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf)
    zeta, _ = pushforward_and_jac(theta, spec, ygrid, params)
    return float(Jval + 0.5 * mu * (zeta @ zeta))

def merit_dirderiv_e2g(theta, spec, xgrid, ygrid, params, zeta, A, dtheta, mu: float, weighted_by_pdf: bool):
    _, gJ, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf)
    grad_phi = gJ + mu * (A.T @ zeta)
    return float(grad_phi @ dtheta)

# -----------------------------
# Optional: simple adaptive split (split first piece once)
# -----------------------------
def split_piece(spec: PLSpec, piece_idx: int) -> PLSpec:
    k = spec.knots
    if piece_idx < 0 or piece_idx >= len(k)-1:
        return spec
    mid = 0.5*(k[piece_idx] + k[piece_idx+1])
    new_knots = np.sort(np.concatenate([k, [mid]]))
    return PLSpec(knots=new_knots, K=len(new_knots)-1)

def project_theta_to_new_knots(theta_old, spec_old: PLSpec, spec_new: PLSpec):
    # Evaluate old s on new knots, re-difference to get new slopes
    a_old, y0_old = unpack_params(theta_old, spec_old.K)
    y_old = pl_continuity_y(a_old, y0_old, spec_old.knots)
    a_i_old, b_i_old = piece_affine_params(a_old, y_old, spec_old.knots)

    def s_eval(x):
        idx = np.clip(np.searchsorted(spec_old.knots, x, side='right') - 1, 0, len(a_i_old)-1)
        return a_i_old[idx]*x + b_i_old[idx]

    y_new = np.array([s_eval(xk) for xk in spec_new.knots])
    a_new = np.maximum(np.diff(y_new) / np.diff(spec_new.knots), 1e-12)
    gammas_new = np.log(a_new)
    theta_new = np.concatenate([gammas_new, np.array([y_new[0]])])
    return theta_new

# -----------------------------
# Full SQP driver
# -----------------------------
def sqp_solve_e2g(
    params: E2GParams,
    K_pieces: int = 12,
    x_m: int = 601, y_m: int = 251,
    max_iter: int = 200,
    tol_opt: float = 1e-7, tol_feas: float = 1e-6,
    step_mode: str = "linesearch",           # "trust_clip", "fixed", "linesearch"
    trust_clip: Optional[float] = 0.5,
    fixed_step_eta: float = 0.5,
    ls_mu: float = 10.0, ls_c1: float = 1e-4, ls_shrink: float = 0.5,
    ls_min_step: float = 1e-6, ls_max_backtracks: int = 25,
    reg_H: float = 1e-10, reg_dual: float = 1e-8,
    dual_step_mode: str = "scaled",
    use_quantile_grids: bool = True,
    p_lo: float = 1e-10, p_hi: float = 1-1e-10,  # deep-tail coverage for quantile grids
    verbose: bool = False, print_every: int = 10,
    adapt_after: Optional[int] = None, adapt_piece_idx: int = 0
):
    # Grids + PL spec
    if use_quantile_grids:
        xgrid = make_xgrid_quantile(params, m=x_m, p_lo=p_lo, p_hi=p_hi)
        ygrid = make_ygrid_quantile(params, m=y_m, p_lo=p_lo, p_hi=p_hi)
        spec  = build_plspec_from_exp_quantiles(params, K_pieces, p_lo=p_lo, p_hi=p_hi)
        weighted_by_pdf = False
    else:
        xgrid = make_xgrid_uniform(params, m=x_m)
        ygrid = make_ygrid_uniform(params, m=y_m)
        spec  = build_plspec_uniform(xgrid, K_pieces)
        weighted_by_pdf = True

    # Init θ to analytic quantile map on knots (key improvement)
    theta, y_star_knots = init_theta_from_analytic(params, spec)
    lam   = np.zeros(ygrid.m)

    hist = {"J": [], "opt": [], "feas": [], "t": [], "d_norm_inf": [], "w_norm_inf": []}

    for it in range(max_iter):
        # Optional one-time adaptive split
        if (adapt_after is not None) and (it == adapt_after):
            spec_new = split_piece(spec, adapt_piece_idx)
            theta = project_theta_to_new_knots(theta, spec, spec_new)
            spec = spec_new  # swap in
            # (lam stays same dimension y_m; constraints unchanged)

        Jval, gJ_now, _ = objective_J(theta, spec, xgrid, params, weighted_by_pdf)
        zeta_now, A_now = pushforward_and_jac(theta, spec, ygrid, params)
        g_now = gJ_now + A_now.T @ lam

        hist["J"].append(Jval)
        hist["opt"].append(np.linalg.norm(g_now, np.inf))
        hist["feas"].append(np.linalg.norm(zeta_now, np.inf))

        if hist["opt"][-1] <= tol_opt and hist["feas"][-1] <= tol_feas:
            break

        dtheta, w, g, zeta, A, _ = sqp_direction_e2g(
            theta, lam, spec, xgrid, ygrid, params, weighted_by_pdf,
            reg_H=reg_H, reg_dual=reg_dual
        )

        # step-size policy
        t = 1.0
        if step_mode == "trust_clip":
            ninf = np.linalg.norm(dtheta, np.inf)
            if (trust_clip is not None) and (ninf > max(trust_clip, 0.0)):
                t = trust_clip / ninf
        elif step_mode == "fixed":
            t = float(fixed_step_eta)
        elif step_mode == "linesearch":
            phi0 = merit_phi_e2g(theta, spec, xgrid, ygrid, params, ls_mu, weighted_by_pdf)
            dir_deriv = merit_dirderiv_e2g(theta, spec, xgrid, ygrid, params, zeta, A, dtheta, ls_mu, weighted_by_pdf)
            ok = False; t = 1.0
            for _ in range(ls_max_backtracks):
                theta_try = theta + t * dtheta
                if merit_phi_e2g(theta_try, spec, xgrid, ygrid, params, ls_mu, weighted_by_pdf) <= phi0 + ls_c1 * t * dir_deriv:
                    ok = True; break
                t *= ls_shrink
                if t < ls_min_step: break
            if not ok and t < ls_min_step: t = ls_min_step
        else:
            raise ValueError("step_mode must be one of {'trust_clip','fixed','linesearch'}")

        # log
        hist["t"].append(t)
        hist["d_norm_inf"].append(np.linalg.norm(dtheta, np.inf))
        hist["w_norm_inf"].append(np.linalg.norm(w, np.inf))
        if verbose and (it % print_every == 0):
            print(f"it={it:03d}  t={t:.2e}  J={Jval:.3e}  opt={hist['opt'][-1]:.3e}  feas={hist['feas'][-1]:.3e}")

        # updates
        theta = theta + t * dtheta
        lam   = lam + (t * w if (dual_step_mode == "scaled" or step_mode == "linesearch") else w)

    # Pack results
    a, y0 = unpack_params(theta, spec.K)
    yk = pl_continuity_y(a, y0, spec.knots)
    a_i, b_i = piece_affine_params(a, yk, spec.knots)
    return {
        "theta": theta, "lambda": lam, "spec": spec,
        "a": a_i, "b": b_i, "y_knots": yk,
        "xgrid": xgrid, "ygrid": ygrid, "history": hist,
        "params": params, "weighted_by_pdf": weighted_by_pdf
    }

# -----------------------
# Utilities (optional)
# -----------------------
def learned_pushforward_e2g(sol, y: np.ndarray) -> np.ndarray:
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    lam0 = sol["params"].lam0
    K = len(a)
    Pf = np.zeros_like(y, dtype=float)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (y - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1]) & (u >= 0.0)
        if np.any(valid):
            Pf[valid] += (1.0/ai) * exp_pdf(u[valid], lam0)
    return Pf

def print_pieces(sol, name=""):
    a = sol["a"]; b = sol["b"]; xk = sol["spec"].knots; yk = sol.get("y_knots")
    if yk is None:
        yk = pl_continuity_y(a, (a[0]*xk[0]+b[0]), xk)
    print(f"\n--- Piece summary {name} ---  K={len(a)}")
    for i in range(len(a)):
        print(f"piece {i:02d}: x∈[{xk[i]:.6g},{xk[i+1]:.6g}]  s(x)={a[i]:.6g}*x + {b[i]:.6g}  "
              f"image y∈[{min(yk[i],yk[i+1]):.6g},{max(yk[i],yk[i+1]):.6g}]")

# Example params
params = E2GParams(lam0=1.0, mu1=0.0, s1=1.0)

sol_e2g = sqp_solve_e2g(
    params,
    K_pieces=200,
    x_m=601, y_m=401, max_iter=500,
    use_quantile_grids=True,
    p_lo=1e-12, p_hi=1-1e-12,   # very deep tails
    step_mode="linesearch",
    verbose=True, print_every=10,
    adapt_after=20, adapt_piece_idx=0   # optional: split first piece after 20 iters
)

# Access map pieces: sol_e2g["a"], sol_e2g["b"], sol_e2g["spec"].knots
# Pushforward check on y-grid:
# Pf = learned_pushforward_e2g(sol_e2g, sol_e2g["ygrid"].y)

import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- PDFs ----------
def exp_pdf(x, lam):
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

def normal_pdf(x, mu, s):
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

# ---------- erfinv-based Normal PPF (no SciPy) ----------
def _erfinv(y):
    # Winitzki approximation (good enough for plotting)
    a = 0.147
    s = np.sign(y)
    ln = np.log(1.0 - y*y)
    first = 2.0/(np.pi*a) + ln/2.0
    return s * np.sqrt(np.sqrt(first*first - ln/a) - first)

def _norm_ppf(u, mu, s):
    u = np.asarray(u)
    u = np.clip(u, 1e-12, 1-1e-12)
    return mu + s * np.sqrt(2.0) * _erfinv(2.0*u - 1.0)

# ---------- Analytic quantile map: Exp(λ0) -> N(μ1, s1) ----------
# s*(x) = Φ_{μ1,s1}^{-1}( F_Exp(x) ) = μ1 + s1 * Φ^{-1}(1 - e^{-λ0 x})
def analytic_map_exp_to_gauss(xs, lam0, mu1, s1):
    u = 1.0 - np.exp(-lam0 * np.maximum(xs, 0.0))
    return _norm_ppf(u, mu1, s1)

# ---------- Evaluate learned PL map on dense grid ----------
def eval_map_pl(sol, xs):
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    idx = np.clip(np.searchsorted(knots, xs, side='right') - 1, 0, len(a)-1)
    return a[idx]*xs + b[idx]

# ---------- Plot: analytic vs computed (side-by-side style) ----------
def plot_e2g_analytic_vs_computed(sol, X_MAX=8.0, N=600, title="Exponential → Gaussian Transport Map"):
    # Expect sol["params"] with lam0, mu1, s1
    lam0, mu1, s1 = sol["params"].lam0, sol["params"].mu1, sol["params"].s1

    xs = np.linspace(0.0, X_MAX, N)  # Exp support starts at 0
    ys_analytic = analytic_map_exp_to_gauss(xs, lam0, mu1, s1)
    ys_sqp      = eval_map_pl(sol, xs)

    ymin = float(min(ys_analytic.min(), ys_sqp.min()))
    ymax = float(max(ys_analytic.max(), ys_sqp.max()))
    pad  = 0.05 * (ymax - ymin + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_analytic, linewidth=2, label="Analytic (quantile map)")
    plt.plot(xs, ys_sqp, linestyle="--", linewidth=2, label="Computed (SQP, PL)")
    plt.xlim(0.0, X_MAX)
    plt.ylim(ymin - pad, ymax + pad)
    plt.xlabel("x"); plt.ylabel("y = s(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

fig_map = plot_e2g_analytic_vs_computed(sol_e2g, X_MAX=8.0, N=600)

import numpy as np
import matplotlib.pyplot as plt

def exp_pdf(x, lam):
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

def normal_pdf(x, mu, s):
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

# Pf(y) for Exp→Gaussian with PL map s(x)=a_i x + b_i on x∈[x_i, x_{i+1}]
def pushforward_pdf_e2g(y: np.ndarray, sol: dict) -> np.ndarray:
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    lam0 = sol["params"].lam0
    K = len(a)
    Pf = np.zeros_like(y, dtype=float)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (y - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1]) & (u >= 0.0)  # source support
        if np.any(valid):
            Pf[valid] += (1.0/ai) * exp_pdf(u[valid], lam0)
    return Pf

def plot_source_and_targets_e2g_side_by_side(sol: dict, num_pts: int = 2001):
    """
    One figure, two panels:
      (left)  source ν(x) = Exp(λ0) on x≥0
      (right) pushforward Pf(s)(y) vs target μ(y) = N(μ1, s1^2)
    """
    lam0, mu1, s1 = sol["params"].lam0, sol["params"].mu1, sol["params"].s1

    # Left panel grid (source domain)
    xL = max(0.0, sol["xgrid"].L)
    xR = max(0.0, sol["xgrid"].R)
    x = np.linspace(xL, xR, num_pts)
    nu_x = exp_pdf(x, lam0)

    # Right panel grid (target domain)
    y_span = 4.0 * s1
    y = np.linspace(mu1 - y_span, mu1 + y_span, num_pts)
    mu_y = normal_pdf(y, mu1, s1)
    pf_y = pushforward_pdf_e2g(y, sol)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source exponential
    ax = axes[0]
    ax.plot(x, nu_x, lw=2)
    ax.set_title(r"Source density $\nu(x)$ (Exponential)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: pushforward vs true Gaussian
    ax = axes[1]
    ax.plot(y, pf_y, lw=2, label=r"Pushforward $P_f(s)(y)$")
    ax.plot(y, mu_y, "k--", lw=2, label=r"True target $\mu(y)$")
    ax.set_title("Target densities on $y$")
    ax.set_xlabel("y")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Exponential→Gaussian: Source vs. Learned Target vs. True Target",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    return fig, axes

fig_side, axes_side = plot_source_and_targets_e2g_side_by_side(sol_e2g, num_pts=2001)

"""### Grid Check"""

def print_pieces(sol, name=""):
    a = sol["a"]                  # slopes per piece
    b = sol["b"]                  # intercepts per piece
    xk = sol["spec"].knots        # source-domain knots (x_0 < ... < x_K)
    yk = sol.get("y_knots", None) # values s(x_k); present in your sol dicts

    K = len(a)
    print(f"\n--- Piece summary {name} ---")
    print(f"K = {K}")
    print(f"x-knots (source): {xk}\n")

    if yk is None:
        # compute y_k = s(x_k) if not already stored
        yk = a * 0.0  # dummy to keep shape; we'll fill below
        yk = [a[i]*xk[i] + b[i] for i in range(K)] + [a[-1]*xk[-1] + b[-1]]
        yk = np.asarray(yk)

    for i in range(K):
        xL, xR = xk[i], xk[i+1]
        yL, yR = yk[i], yk[i+1]
        print(f"piece {i:02d}:")
        print(f"  x ∈ [{xL:.6g}, {xR:.6g}]")
        print(f"  s(x) = {a[i]:.6g} * x + {b[i]:.6g}")
        print(f"  image in y: [{min(yL,yR):.6g}, {max(yL,yR):.6g}]\n")

    # Useful quick checks
    print("s(0) =",
          (a[0]*0.0 + b[0]) if xk[0] <= 0.0 <= xk[1] else "0 not in first source interval")
print_pieces(sol_e2g, name="E→G")

# Plot the x-grid nodes used by the solver (works for G→E or E→G `sol` dicts)
# - Shows grid nodes as vertical tick markers along y=0
# - Overlays the source pdf (scaled to [0,1]) so you can see why a quantile grid clusters

import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

def exp_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    out = lam * np.exp(-lam * np.maximum(x, 0.0))
    out[x < 0] = 0.0
    return out

def plot_xgrid(sol: dict, save_path="/mnt/data/xgrid_plot.png"):
    x = sol["xgrid"].x
    m = sol["xgrid"].m
    L, R = sol["xgrid"].L, sol["xgrid"].R

    # detect source pdf from params
    params = sol["params"]
    xs_dense = np.linspace(L, R, 2000)
    if hasattr(params, "mu0") and hasattr(params, "s0"):        # Gaussian source (G→E)
        pdf = normal_pdf(xs_dense, params.mu0, params.s0)
        src_name = "Gaussian source"
    elif hasattr(params, "lam0"):                               # Exponential source (E→G)
        pdf = exp_pdf(xs_dense, params.lam0)
        src_name = "Exponential source"
    else:
        pdf = np.zeros_like(xs_dense)
        src_name = "source"

    # scale pdf to [0,1] for overlay near y=0
    if np.max(pdf) > 0:
        pdf = pdf / np.max(pdf)

    plt.figure(figsize=(10, 2.8))
    # plot markers for grid nodes
    y0 = np.zeros_like(x)
    plt.plot(x, y0, marker="|", linestyle="None", markersize=18, label=f"x-grid nodes (m={m})")
    # overlay scaled source pdf
    plt.plot(xs_dense, pdf, linewidth=1.5, label=f"{src_name} (scaled)")

    # aesthetics
    plt.xlim(L, R)
    plt.ylim(-0.2, 1.2)
    plt.xlabel("x")
    plt.ylabel("grid / scaled pdf")
    # try to show what kind of grid we used
    grid_type = "quantile grid" if sol.get("weighted_by_pdf") is False else "uniform grid"
    plt.title(f"x-grid nodes ({grid_type})")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(save_path, dpi=200)
    plt.show()
    return save_path

# Choose a solution dict from common names
chosen = None
for name in ["sol_e2g", "sol_quant", "sol_uniform", "sol"]:
    if name in globals():
        chosen = globals()[name]
        break

if chosen is None:
    raise RuntimeError("No solution dict found in this session. Run your solver to create `sol_e2g`, `sol`, `sol_quant`, or `sol_uniform`.")

path = plot_xgrid(chosen)
path