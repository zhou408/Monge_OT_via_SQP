# -*- coding: utf-8 -*-
"""Uniform -> Beta notebook section.

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


"""# Uniform to Beta"""

import numpy as np
from dataclasses import dataclass

# =========================
# Beta helpers (no SciPy)
# =========================
def _beta_pdf(y, a, b):
    import math
    y = np.asarray(y)
    out = np.zeros_like(y, dtype=float)
    mask = (y >= 0) & (y <= 1)
    if not np.any(mask): return out
    B = math.gamma(a)*math.gamma(b)/math.gamma(a+b)
    ym = y[mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        outm = np.exp((a-1)*np.log(np.clip(ym, 1e-300, 1.0)) +
                      (b-1)*np.log(np.clip(1-ym, 1e-300, 1.0))) / B
    out[mask] = outm
    return out

def _build_beta_tables(alpha, beta, N=40000):
    y = np.linspace(0.0, 1.0, N)
    pdf = _beta_pdf(y, alpha, beta)
    dy = y[1]-y[0]
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * dy)
    cdf = np.concatenate([[0.0], cdf])
    area = max(cdf[-1], 1e-300)
    cdf /= area
    return y, pdf, cdf

def beta_cdf_numeric(y, table):
    y_tab, _, cdf_tab = table
    y = np.clip(y, 0.0, 1.0)
    return np.interp(y, y_tab, cdf_tab)

def beta_ppf_numeric(u, table):
    y_tab, _, cdf_tab = table
    u = np.clip(u, 0.0, 1.0)
    return np.interp(u, cdf_tab, y_tab)

# =========================
# Problem / grids / PL map
# =========================
@dataclass
class U2BParams:
    alpha: float
    beta:  float

@dataclass
class XGrid:
    x: np.ndarray  # uniform in [0,1] for source sampling (not critical here)

@dataclass
class YGrid:
    y: np.ndarray  # constraint points; we’ll use Beta percentiles (quantiles)

@dataclass
class PLSpec:
    knots: np.ndarray  # x_0<...<x_K
    K: int

def make_xgrid(m=401):
    x = np.linspace(0.0, 1.0, m)
    return XGrid(x=x)

def make_ygrid_beta_quantiles(params: U2BParams, m=301, p_lo=1e-6, p_hi=1-1e-6):
    table = _build_beta_tables(params.alpha, params.beta)
    u = np.linspace(p_lo, p_hi, m)
    y = beta_ppf_numeric(u, table)
    return YGrid(y=y), table

def build_plspec(K):
    knots = np.linspace(0.0, 1.0, K+1)
    return PLSpec(knots=knots, K=K)

# =========================
# PL utilities
# =========================
def unpack(theta, K):
    gammas = theta[:K]
    y0     = theta[K]
    a = np.exp(gammas)           # slopes > 0
    return a, y0

def y_knots_from_theta(theta, spec: PLSpec):
    a, y0 = unpack(theta, spec.K)
    yk = np.empty(spec.K+1)
    yk[0] = y0
    dx = np.diff(spec.knots)
    for i in range(spec.K):
        yk[i+1] = yk[i] + a[i]*dx[i]
    return yk

def piece_affine_params(theta, spec: PLSpec):
    a, y0 = unpack(theta, spec.K)
    yk = y_knots_from_theta(theta, spec)
    b = np.empty(spec.K)
    for i in range(spec.K):
        b[i] = yk[i] - a[i]*spec.knots[i]
    return a, b, yk

def eval_map(theta, spec: PLSpec, xs: np.ndarray):
    a, b, _ = piece_affine_params(theta, spec)
    K = spec.K
    idx = np.clip(np.searchsorted(spec.knots, xs, side='right')-1, 0, K-1)
    return a[idx]*xs + b[idx]

# =========================
# Soft-CDF model & Jacobian
# =========================
def _sigmoid_safe(z):
    # tanh-based, overflow-safe; z can be vector
    z = np.clip(z, -80.0, 80.0)
    return 0.5*(1.0 + np.tanh(0.5*z))

def model_cdf_and_jac(theta, spec: PLSpec, ygrid: YGrid, eps=5e-3):
    """
    F_f(y) ≈ sum_i Δx_i * (σ((u-x_i)/eps) - σ((u-x_{i+1})/eps)), u=(y-b_i)/a_i
    Return Ff (m,), J (m × (K+1)) where θ=[γ_0..γ_{K-1}, y0].
    """
    K = spec.K
    y = ygrid.y
    a, b, yk = piece_affine_params(theta, spec)
    xk = spec.knots
    dx = np.diff(xk)

    Ff = np.zeros_like(y)
    J  = np.zeros((y.size, K+1))

    # precompute ∂b_i/∂γ_j and ∂b_i/∂y0
    db_dgamma = np.zeros((K, K))
    db_dy0    = np.ones(K)
    for i in range(K):
        # b_i = y_i - a_i x_i; y_i depends on previous a_j
        for j in range(i):
            db_dgamma[i, j] += a[j]*dx[j]
        db_dgamma[i, i] += -a[i]*xk[i]

    inv_eps = 1.0/max(eps, 1e-6)

    for i in range(K):
        ai = float(max(a[i], 1e-8))  # guard tiny slopes
        bi = b[i]
        xi, xip1 = xk[i], xk[i+1]
        dxi = dx[i]

        u   = (y - bi)/ai
        zL  = (u - xi)*inv_eps
        zR  = (u - xip1)*inv_eps
        sL  = _sigmoid_safe(zL)
        sR  = _sigmoid_safe(zR)

        Li = dxi*(sL - sR)
        Ff += Li

        # dLi/du = Δx * [σ'(zL)*inv_eps - σ'(zR)*inv_eps]
        # with σ'(z) = 0.25*(1 - tanh(0.5z)^2)
        tL = np.tanh(0.5*np.clip(zL, -80, 80))
        tR = np.tanh(0.5*np.clip(zR, -80, 80))
        sLp = 0.25*(1 - tL*tL)*inv_eps
        sRp = 0.25*(1 - tR*tR)*inv_eps
        dLi_du = dxi*(sLp - sRp)

        # u wrt params: u=(y-b)/a
        du_da = -u/ai
        du_db = -1.0/ai

        # column γ_i (local slope)
        J[:, i] += dLi_du * (du_da * ai)   # ∂a/∂γ_i = a_i

        # columns γ_j (j<i) via b_i
        if i > 0:
            for j in range(i):
                if db_dgamma[i, j] != 0.0:
                    J[:, j] += dLi_du * (du_db * db_dgamma[i, j])

        # column y0 via b_i
        J[:, K] += dLi_du * (du_db * db_dy0[i])

    return Ff, J

# =========================
# Objective (penalized CDF LS)
# =========================
def objective(theta, spec, ygrid, Gy, lam_end=10.0, lam_smooth=1e-2, eps=5e-3):
    # residuals: r = [ Ff(y) - Gy ; sqrt(lam_end)*c_end ; sqrt(lam_smooth)*Dγ ]
    Ff, J = model_cdf_and_jac(theta, spec, ygrid, eps=eps)
    r_main = Ff - Gy

    # endpoints: c0 = s(0), c1 = s(1)-1
    a, b, yk = piece_affine_params(theta, spec)
    c0 = yk[0]
    c1 = yk[-1] - 1.0
    # Jacobian rows for endpoints
    K = spec.K
    xk = spec.knots
    dx = np.diff(xk)
    g0 = np.zeros(K+1); g0[-1] = 1.0
    g1 = np.zeros(K+1); g1[:K] = a*dx; g1[-1] = 1.0

    # smoothness on γ (second difference)
    gammas = theta[:K]
    D = np.zeros((max(K-2,0), K))
    for i in range(K-2):
        D[i, i:i+3] = np.array([1, -2, 1])
    r_smooth = (D @ gammas) if K >= 3 else np.zeros(0)
    J_smooth = np.zeros((r_smooth.size, K+1))
    if r_smooth.size:
        J_smooth[:, :K] = D

    # stack residuals and Jacobian
    r = np.concatenate([r_main, np.sqrt(lam_end)*np.array([c0, c1]), np.sqrt(lam_smooth)*r_smooth])
    Jall = np.vstack([
        J,
        np.sqrt(lam_end)*g0[None,:],
        np.sqrt(lam_end)*g1[None,:],
        np.sqrt(lam_smooth)*J_smooth
    ])
    return r, Jall

# =========================
# Solver: damped Gauss–Newton
# =========================
def solve_u2b(params: U2BParams,
              K_pieces=16,
              y_m=301,
              eps=5e-3,
              lam_end=10.0,
              lam_smooth=1e-2,
              max_iter=200,
              tol=1e-6,
              # NEW: step control (match your other solvers)
              step_mode: str = "linesearch",         # {"linesearch","fixed","trust_clip"}
              fixed_step_eta: float = 0.5,
              trust_clip: float = 0.5,
              ls_c1: float = 1e-4,
              ls_max_backtracks: int = 30,
              ls_shrink: float = 0.5,
              ls_min_step: float = 1e-6,
              # LM damping
              damping_init: float = 1e-2,
              verbose: bool = True,
              print_every: int = 10):
    """
    Penalized Gauss–Newton with configurable step policy.
    Returns dict with 'history_t' (step sizes) and everything else as before.
    """
    # y-grid & target CDF
    ygrid, table = make_ygrid_beta_quantiles(params, m=y_m, p_lo=1e-6, p_hi=1-1e-6)
    Gy = beta_cdf_numeric(ygrid.y, table)

    # PL spec + quantile init on knots
    spec = build_plspec(K_pieces)
    y_star_knots = beta_ppf_numeric(spec.knots, table)
    dy = np.diff(y_star_knots); dx = np.diff(spec.knots)
    a0 = np.clip(dy/np.maximum(dx,1e-12), 1e-6, 1e6)
    gammas0 = np.log(a0)
    y0 = float(y_star_knots[0])
    theta = np.concatenate([gammas0, np.array([y0])])

    damping = float(damping_init)
    history_t = []

    for it in range(max_iter):
        r, J = objective(theta, spec, ygrid, Gy,
                         lam_end=lam_end, lam_smooth=lam_smooth, eps=eps)
        H = J.T @ J + damping*np.eye(J.shape[1])
        g = J.T @ r
        grad_inf = np.linalg.norm(g, np.inf)
        cost = 0.5*np.dot(r, r)

        if verbose and (it % print_every == 0):
            print(f"[U→B] it={it:03d}  cost={cost:.3e}  |grad|_inf={grad_inf:.3e}  damp={damping:.1e}")

        if grad_inf < tol:
            break

        # Proposed GN step
        try:
            dtheta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        # ------------------------------
        # Step-size policy
        # ------------------------------
        if step_mode == "trust_clip":
            ninf = np.linalg.norm(dtheta, np.inf)
            t = 1.0
            if (trust_clip is not None) and (ninf > max(trust_clip, 0.0)):
                t = trust_clip / max(ninf, 1e-30)
            theta_try = theta + t*dtheta
            # accept unconditionally (trust region is the guard)
            theta = theta_try

        elif step_mode == "fixed":
            t = float(fixed_step_eta)
            theta = theta + t*dtheta

        elif step_mode == "linesearch":
            # Armijo backtracking on 0.5||r||^2
            t = 1.0
            phi0 = cost
            armijo_rhs_slope = 1e-0  # optional scale (kept at 1 for pure LS on cost)
            # NOTE: Classical Armijo uses directional derivative of the cost:
            # dphi = g.T dtheta when cost=0.5||r||^2 and J fixed; we use exact recomputation.
            success = False
            for _ in range(ls_max_backtracks):
                theta_try = theta + t*dtheta
                r_try, _ = objective(theta_try, spec, ygrid, Gy,
                                     lam_end=lam_end, lam_smooth=lam_smooth, eps=eps)
                phi_try = 0.5*np.dot(r_try, r_try)
                # Sufficient decrease: phi(θ+tΔ) ≤ phi(θ) + c1 * t * ∇φ·Δ
                # We approximate with -c1 * t * ||g||^2 scaled by armijo_rhs_slope (simple, robust)
                if phi_try <= phi0 - ls_c1 * t * armijo_rhs_slope * np.dot(g, dtheta):
                    theta = theta_try
                    # light LM damping schedule
                    damping = max(damping/1.5, 1e-6)
                    success = True
                    break
                t *= ls_shrink
                if t < ls_min_step:
                    t = ls_min_step
                    theta = theta + t*dtheta
                    break
            if not success and t == ls_min_step:
                # on failure we can increase damping to stabilize next iteration
                damping *= 5.0
        else:
            raise ValueError("step_mode must be one of {'linesearch','fixed','trust_clip'}")

        history_t.append(float(t))

    # pack solution
    a, b, yk = piece_affine_params(theta, spec)
    return {
        "theta": theta,
        "a": a,
        "b": b,
        "y_knots": yk,
        "spec": spec,
        "ygrid": ygrid,
        "params": params,
        "history_t": history_t,
    }

# =========================
# Pushforward PDF (for plots)
# =========================
def pushforward_pdf_uniform_source(y, sol, eps=5e-3):
    """
    Differentiate the soft CDF numerically to visualize Pf(y).
    For Uniform source, this is adequate for plots.
    """
    y = np.asarray(y)
    dy = (y[-1]-y[0])/(len(y)-1) if len(y)>1 else 1e-3
    # reuse model CDF
    Ff, _ = model_cdf_and_jac(sol["theta"], sol["spec"], YGrid(y=y), eps=eps)
    pdf = np.gradient(Ff, dy)
    return np.maximum(pdf, 0.0)

# solve for Beta(1,3)
params = U2BParams(alpha=1.0, beta=3.0)
sol = solve_u2b(params, K_pieces=8, y_m=401, eps=5e-3, step_mode="linesearch",ls_shrink=0.5, ls_max_backtracks=30, lam_end=20.0, lam_smooth=5e-3, verbose=True, print_every=10)

# quick visual checks (optional)
# import matplotlib.pyplot as plt
# y = np.linspace(0,1,1001)
# pf = pushforward_pdf_uniform_source(y, sol, eps=5e-3)
# tar = _beta_pdf(y, params.alpha, params.beta)
# plt.figure(); plt.plot(y, pf, label="pushforward"); plt.plot(y, tar, "--", label="target"); plt.legend(); plt.show()

# xs = np.linspace(0,1,600)
# ys = eval_map(sol["theta"], sol["spec"], xs)
# y_tab,_,cdf_tab = _build_beta_tables(params.alpha, params.beta)
# ys_star = np.interp(xs, cdf_tab, y_tab)
# plt.figure(); plt.plot(xs, ys_star, "--", label="analytic quantile"); plt.plot(xs, ys, label="learned PL"); plt.legend(); plt.show()

print(sol)

import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- Beta pdf ----------
def beta_pdf(y: np.ndarray, a: float, b: float) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    mask = (y >= 0.0) & (y <= 1.0)
    if not np.any(mask): return out
    B = math.gamma(a)*math.gamma(b)/math.gamma(a+b)
    ym = y[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        outm = np.exp((a-1.0)*np.log(np.clip(ym, 1e-300, 1.0))
                      + (b-1.0)*np.log(np.clip(1.0-ym, 1e-300, 1.0))) / B
    out[mask] = outm
    return out

# ---------- Numeric Beta PPF (no SciPy): invert CDF on a fine grid ----------
# Build once per (alpha,beta) and reuse.
def _build_beta_ppf(alpha: float, beta: float, N_grid: int = 20000):
    y = np.linspace(0.0, 1.0, N_grid)
    pdf = beta_pdf(y, alpha, beta)
    # cumulative via trapezoid rule; normalize for safety
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * (y[1] - y[0]))
    cdf = np.concatenate([[0.0], cdf])
    if cdf[-1] <= 0:  # degenerate guard
        cdf[-1] = 1.0
    cdf /= cdf[-1]
    return y, cdf

def beta_ppf_numeric(u: np.ndarray, alpha: float, beta: float, table=None):
    if table is None:
        table = _build_beta_ppf(alpha, beta)
    y_grid, cdf_grid = table
    u = np.clip(u, 0.0, 1.0)
    # invert by 1D interpolation (monotone)
    return np.interp(u, cdf_grid, y_grid)

# ---------- Evaluate learned PL map s(x) on dense xs ----------
def _eval_map_pl(sol: dict, xs: np.ndarray):
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    idx = np.clip(np.searchsorted(knots, xs, side="right") - 1, 0, len(a)-1)
    return a[idx]*xs + b[idx]

# ---------- Pushforward Pf(s)(y) for Uniform(0,1) source ----------
def pushforward_pdf_u2b(y: np.ndarray, sol: dict) -> np.ndarray:
    a = sol["a"]; b = sol["b"]; knots = sol["spec"].knots
    K = len(a)
    Pf = np.zeros_like(y, dtype=float)
    for i in range(K):
        ai, bi = a[i], b[i]
        u = (y - bi) / ai
        valid = (u >= knots[i]) & (u <= knots[i+1]) & (u >= 0.0) & (u <= 1.0)
        if np.any(valid):
            Pf[valid] += (1.0 / ai)   # f_X(u)=1 on [0,1]
    return Pf

# =========================================================
# (A) Analytic optimal map vs computed PL map (Uniform→Beta)
# =========================================================
def plot_u2b_analytic_vs_computed(sol: dict, N: int = 600, title: str = "Uniform → Beta Transport Map"):
    alpha, beta = sol["params"].alpha, sol["params"].beta
    xs = np.linspace(0.0, 1.0, N)

    # Analytic quantile map: s*(x) = F_Beta^{-1}(x)
    ppf_table = _build_beta_ppf(alpha, beta)
    ys_analytic = beta_ppf_numeric(xs, alpha, beta, table=ppf_table)

    # Computed PL map
    ys_sqp = _eval_map_pl(sol, xs)

    ylo = float(min(ys_analytic.min(), ys_sqp.min()))
    yhi = float(max(ys_analytic.max(), ys_sqp.max()))
    pad = 0.05 * (yhi - ylo + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_analytic, linewidth=2, label="Analytic (quantile map)")
    plt.plot(xs, ys_sqp, linestyle="--", linewidth=2, label="Computed (SQP, PL)")
    plt.xlim(0.0, 1.0)
    plt.ylim(ylo - pad, yhi + pad)
    plt.xlabel("x"); plt.ylabel("y = s(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

# =========================================================
# (B) Side-by-side densities: source vs (pushforward vs target)
# =========================================================
def plot_source_and_targets_u2b_side_by_side(sol: dict, num_pts: int = 2001):
    """
    One figure with 1×2 panels:
      (left)  source μ(x)=Uniform(0,1)
      (right) pushforward Pf(s)(y) vs target ν(y)=Beta(α,β)
    """
    alpha, beta = sol["params"].alpha, sol["params"].beta

    # Grids for panels
    x = np.linspace(0.0, 1.0, num_pts)
    y = np.linspace(1e-4, 1-1e-4, num_pts)

    # Densities
    mu_x = np.ones_like(x)  # Uniform(0,1)
    nu_y = beta_pdf(y, alpha, beta)
    pf_y = pushforward_pdf_u2b(y, sol)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source uniform
    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_ylim(-0.05, max(1.1, 1.1*np.max(mu_x)))
    ax.set_title(r"Source density $\mu(x)$ (Uniform[0,1])")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: pushforward vs Beta target
    ax = axes[1]
    ax.plot(y, pf_y, lw=2, label=r"Pushforward $P_f(s)(y)$")
    ax.plot(y, nu_y, "k--", lw=2, label=r"True target $\nu(y)$")
    ax.set_title("Target densities on $y$")
    ax.set_xlabel("y")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Uniform→Beta: Source vs. Learned Target vs. True Target",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    return fig, axes

# After solving:
# sol_u2b = sqp_solve_u2b(params, ...)

# (A) Analytic vs computed map
fig_map = plot_u2b_analytic_vs_computed(sol, N=600)

# (B) Side-by-side densities
fig_den, axes_den = plot_source_and_targets_u2b_side_by_side(sol, num_pts=2001)