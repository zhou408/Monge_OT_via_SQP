# -*- coding: utf-8 -*-
"""Concave / McCann-style sections (piecewise + polynomial).

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


"""# Concave

## Optimal Partition
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =========================
# Measures for Example 1.1
# =========================
# Domain is [-10, 10], rho(x) = sin((pi/5) x).
# We use normalized densities mu = rho_+ / Z, nu = rho_- / Z, Z = ∫ rho_+ = ∫ rho_- = 20/pi.
PI = np.pi
C = PI / 5.0          # frequency
Z = 20.0 / PI         # exact positive/negative mass on [-10, 10]
Z_INV = 1.0 / Z       # = PI / 20

def rho_signed(x: np.ndarray) -> np.ndarray:
    return np.sin(C * x)

def mu_pdf(x: np.ndarray) -> np.ndarray:
    # mu(x) = (pi/20) * max(sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    # nu(x) = (pi/20) * max(-sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    # derivative of mu: (pi/20) * 1_{sin>0} * C * cos(C x)
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# ============
# Grids
# ============
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray # shape (m,)
    weights: np.ndarray # trapezoid weights for integrations

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    if m >= 2:
        w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

# ====================================
# Piecewise linear map specification
# ====================================
@dataclass
class PWLinSpec:
    # knots define K pieces on [knots[i], knots[i+1]]
    knots: np.ndarray      # shape (K+1,)
    signs: np.ndarray      # shape (K,), each in {+1, -1}; slope a_i = signs[i]*exp(gamma_i)
    # variable vector is theta = [gamma_0..gamma_{K-1}, beta_0..beta_{K-1}] -> size n=2K

def n_pieces(spec: PWLinSpec) -> int:
    return len(spec.knots) - 1

# Evaluate s(x) and piece indices
def s_eval_piecewise(x: np.ndarray, gammas: np.ndarray, betas: np.ndarray, spec: PWLinSpec) -> np.ndarray:
    K = n_pieces(spec)
    y = np.empty_like(x)
    # map x to its piece via digitize
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a = spec.signs[idx] * np.exp(gammas[idx])
    b = betas[idx]
    y[:] = a * x + b
    return y

# ===========================
# Objective  c(x,y)=sqrt(2|x-y|)
# with analytic gradient
# ===========================
# def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
#                    xgrid: GridSpec, spec: PWLinSpec) -> Tuple[float, np.ndarray, np.ndarray]:
#     """
#     J = ∫ mu(x) * sqrt(2 * | x - s(x) |) dx   (integrated on xgrid)
#     Returns J, grad (size 2K), and a small diagonal Hessian (for stability).
#     """
#     K = n_pieces(spec)
#     x = xgrid.points
#     w = xgrid.weights
#     mu = mu_pdf(x)

#     # which piece each x falls into
#     idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)

#     # slopes/intercepts
#     a_all = spec.signs * np.exp(gammas)   # (K,)
#     a_x   = a_all[idx]
#     b_x   = betas[idx]

#     # map and residual
#     s_x  = a_x * x + b_x
#     r_x  = x - s_x
#     abs_r = np.abs(r_x)

#     # cost = sqrt(2 * |r|)
#     cost = np.sqrt(2.0 * abs_r)
#     J = np.sum(mu * cost * w)

#     # ----- gradient -----
#     # For r != 0: d/ds sqrt(2|r|) = - sign(r) / sqrt(2|r|)
#     # Chain rule through s(x)=a_i x + b_i on the active piece.
#     eps = 1e-12  # avoid division by 0
#     inv_sqrt_2absr = 1.0 / np.sqrt(2.0 * np.maximum(abs_r, eps))
#     sgn_r = np.sign(r_x)  # subgradient at 0; OK in practice

#     grad = np.zeros(2 * K)
#     for i in range(K):
#         mask = (idx == i)
#         if not np.any(mask):
#             continue
#         a_i = a_all[i]
#         xi, wi = x[mask], w[mask]
#         mui = mu[mask]
#         sgn_i = sgn_r[mask]
#         inv_i = inv_sqrt_2absr[mask]

#         # dJ/dgamma_i: ∂s/∂gamma_i = a_i * x  (since a_i = sign_i * exp(gamma_i))
#         grad[i] = -np.sum(mui * sgn_i * inv_i * (a_i * xi) * wi)

#         # dJ/dbeta_i: ∂s/∂beta_i = 1
#         grad[K + i] = -np.sum(mui * sgn_i * inv_i * 1.0 * wi)

#     # small positive-definite Hessian for numerical robustness (objective is non-smooth)
#     H = 1e-10 * np.eye(2 * K)
#     return J, grad, H
def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
                   xgrid: GridSpec, spec: PWLinSpec,
                   eps: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Smoothed objective for c(r) = sqrt(2|r|) via
        c_eps(r) = sqrt(2) * (r^2 + eps^2)^(1/4).
    Returns J, grad (size 2K), and a PSD Gauss-Newton-like Hessian.
    """
    K = n_pieces(spec)
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    # piece membership and piece parameters per-sample
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a_all = spec.signs * np.exp(gammas)     # (K,)
    a_x   = a_all[idx]
    b_x   = betas[idx]

    # residual r = x - s(x)
    s_x  = a_x * x + b_x
    r    = x - s_x
    q    = r*r + eps*eps                   # r^2 + eps^2

    # c_eps(r) = sqrt(2) * q^(1/4)
    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr = (sqrt(2)/2) * r * q^(-3/4)
    dc_dr = (np.sqrt(2.0)/2.0) * r * np.power(q, -0.75)

    # grad wrt theta on each piece
    grad = np.zeros(2*K)
    for i in range(K):
        msk = (idx == i)
        if not np.any(msk):
            continue
        a_i = a_all[i]
        xi, wi = x[msk], w[msk]
        mui    = mu[msk]
        dci    = dc_dr[msk]

        # ∂s/∂gamma_i = a_i * x, ∂s/∂beta_i = 1
        # ∂J/∂theta = ∫ mu * (dc/dr) * (-∂s/∂theta) dx
        grad[i]      = -np.sum(mui * dci * (a_i * xi) * wi)
        grad[K + i]  = -np.sum(mui * dci * 1.0        * wi)

    # ----- PSD Hessian (Gauss–Newton style with curvature clipping) -----
    # d2c/dr2 = (sqrt(2)/2) * (q)^(-7/4) * (eps^2 - 0.5*r^2)
    d2c = (np.sqrt(2.0)/2.0) * np.power(q, -1.75) * (eps*eps - 0.5 * r*r)
    w_curv = np.maximum(d2c, 0.0)   # clip to keep PSD

    H = np.zeros((2*K, 2*K))
    for i in range(K):
        msk_i = (idx == i)
        if not np.any(msk_i):
            continue
        a_i = a_all[i]
        xi, wi = x[msk_i], w[msk_i]
        mui    = mu[msk_i]
        wi_curv = w_curv[msk_i]

        # features for this piece: [a_i * x, 1]
        f1 = a_i * xi
        f2 = np.ones_like(xi)

        # weight = mu * w * w_curv
        ww = mui * wi * wi_curv

        # accumulate 2x2 block for (gamma_i, beta_i)
        H[i, i]             += np.sum(ww * f1 * f1)
        H[i, K + i]         += np.sum(ww * f1 * f2)
        H[K + i, i]         += np.sum(ww * f2 * f1)
        H[K + i, K + i]     += np.sum(ww * f2 * f2)

    # mild diagonal regularization for numerical stability
    H += 1e-10 * np.eye(2*K)
    return J, grad, H


# ==========================================================
# PF constraint zeta(d) = P_f(s)(d) - g(d) on a d-grid
# and analytic Jacobian A = d zeta / d theta
# ==========================================================
def constraint_zeta_pw(gammas: np.ndarray, betas: np.ndarray, dgrid: GridSpec, spec: PWLinSpec) -> Dict[str, np.ndarray]:
    """
    Returns:
      zeta : shape (m_d,)
      A    : shape (m_d, 2K) for theta=[gammas..., betas...]
    Uses formula:
      For piece i with slope a_i = sign_i * exp(gamma_i),
        u_i(d) = (d - beta_i) / a_i = sign_i * exp(-gamma_i) * (d - beta_i)
        contrib_i(d) = mu(u_i) / |a_i| = exp(-gamma_i) * mu(u_i)
      zeta(d) = sum_i contrib_i(d) * 1_{u_i in [x_i, x_{i+1}]} - nu(d)
    Jacobians:
      ∂/∂gamma_i contrib = -exp(-gamma_i) * [mu(u) + u * mu'(u)]
      ∂/∂beta_i  contrib = -sign_i * exp(-2*gamma_i) * mu'(u)
    """
    K = n_pieces(spec)
    d = dgrid.points
    m = dgrid.m
    zeta = np.zeros(m)
    A = np.zeros((m, 2*K))

    for i in range(K):
        sgn = spec.signs[i]
        eg = np.exp(gammas[i])
        emg = np.exp(-gammas[i])      # 1/|a_i|
        a_i = sgn * eg

        # preimage u(d) on this piece
        u = (d - betas[i]) / a_i      # = sgn * exp(-gamma) * (d - beta)
        # only valid if u in the segment
        L, R = spec.knots[i], spec.knots[i+1]
        valid = (u >= L) & (u <= R)
        if not np.any(valid):
            continue

        mu_u  = mu_pdf(u)
        mu_u_p = mu_pdf_prime(u)

        contrib = emg * mu_u
        zeta += contrib * valid

        # Jacobians (only where valid)
        d_contrib_dgamma = -emg * (mu_u + u * mu_u_p)
        d_contrib_dbeta  = -sgn * (emg**2) * mu_u_p

        A[valid, i]     += d_contrib_dgamma[valid]
        A[valid, K + i] += d_contrib_dbeta[valid]

    # subtract target density
    zeta -= nu_pdf(d)
    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian pieces & single SQP step (PW)
# ===========================================
def build_lagrangian_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                        xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J, gJ, HJ = objective_J_pw(gammas, betas, xgrid, spec)
    parts = constraint_zeta_pw(gammas, betas, dgrid, spec)
    zeta, A = parts["zeta"], parts["A"]
    # gradient of L
    g = gJ + A.T @ lam
    # (we keep H = HJ; constraint second-derivatives omitted for stability)
    H = HJ
    return g, H, A, J

def sqp_step_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec,
                reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
    zeta = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

    n = 2 * n_pieces(spec)
    m = dgrid.m

    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    new_gammas = gammas + dtheta[:n_pieces(spec)]
    new_betas  = betas  + dtheta[n_pieces(spec):]
    new_lam = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res

# =====================
# Full solver (PW-SQP)
# =====================
def sqp_solve_piecewise(
    spec: PWLinSpec,
    xgrid_m: int = 1001,  # integration grid for objective on x
    dgrid_m: int = 401,   # PF constraint grid on d
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    init_gammas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.5,
) -> Dict[str, np.ndarray]:

    K = n_pieces(spec)
    xgrid = make_uniform_grid(-10.0, 10.0, xgrid_m)
    dgrid = make_uniform_grid(-10.0, 10.0, dgrid_m)

    gammas = np.zeros(K) if init_gammas is None else init_gammas.copy()
    betas  = np.zeros(K) if init_betas  is None else init_betas.copy()
    lam = np.zeros(dgrid.m)

    hist = {"J": [], "opt_res": [], "feas_res": [], "gammas": [], "betas": []}

    for it in range(max_iter):
        g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
        zeta_now = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["gammas"].append(gammas.copy())
        hist["betas"].append(betas.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res = sqp_step_pw(
            gammas, betas, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        # simple trust-region clip in infinity norm on the parameter update
        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_gammas = gammas + dtheta[:K]
                new_betas  = betas  + dtheta[K:]

        gammas, betas, lam = new_gammas, new_betas, new_lam

    return {
        "gammas": gammas, "betas": betas,
        "slopes": spec.signs * np.exp(gammas),
        "lambda": lam, "history": hist, "spec": spec,
        "xgrid": xgrid, "dgrid": dgrid
    }

# =============================
# Example usage for Fig. 1 map
# =============================
if __name__ == "__main__":
    # Knots chosen to match the example's pieces:
    # (-10,-9), (-9,-1), (-1,1), (1,9), (9,10)
    knots = np.array([-10., -9., -1., 1., 9., 10.])
    signs = np.array([-1, -1, -1, -1, -1])  # slope is -exp(gamma_i)

    spec = PWLinSpec(knots=knots, signs=signs)

    # A near-analytic initialization (optional, speeds convergence):
    # desired s(x) ~ -x on |x|<1 or |x|>9  => beta=0
    # desired s(x) ~ -x - 10 on (-9,-1)    => beta=-10
    # desired s(x) ~ -x + 10 on (1,9)      => beta=+10
    init_gammas = np.zeros(len(signs))  # slope ~ -1
    init_betas  = np.array([0., -10., 0., +10., 0.])

    sol = sqp_solve_piecewise(
        spec,
        xgrid_m=1001, dgrid_m=1001,
        max_iter=100, tol_opt=1e-1, tol_feas=1e-3,
        init_gammas=init_gammas, init_betas=init_betas,
        trust_clip=0.25
    )
    # init_gammas=np.random.uniform(-1, 1, 5), init_betas=np.random.uniform(-10, 10, 5)
    print("Slopes a_i:", sol["slopes"])
    print("Betas  b_i:", sol["betas"])
    print("Final opt_res, feas_res:", sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =========================
# Measures for Example 1.1
# =========================
# Domain is [-10, 10], rho(x) = sin((pi/5) x).
# We use normalized densities mu = rho_+ / Z, nu = rho_- / Z, Z = ∫ rho_+ = ∫ rho_- = 20/pi.
PI = np.pi
C = PI / 5.0          # frequency
Z = 20.0 / PI         # exact positive/negative mass on [-10, 10]
Z_INV = 1.0 / Z       # = PI / 20

def rho_signed(x: np.ndarray) -> np.ndarray:
    return np.sin(C * x)

def mu_pdf(x: np.ndarray) -> np.ndarray:
    # mu(x) = (pi/20) * max(sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    # nu(x) = (pi/20) * max(-sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    # derivative of mu: (pi/20) * 1_{sin>0} * C * cos(C x)
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# ============
# Grids
# ============
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray # shape (m,)
    weights: np.ndarray # trapezoid weights for integrations

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    if m >= 2:
        w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

# ====================================
# Piecewise linear map specification
# ====================================
@dataclass
class PWLinSpec:
    # knots define K pieces on [knots[i], knots[i+1]]
    knots: np.ndarray      # shape (K+1,)
    signs: np.ndarray      # shape (K,), each in {+1, -1}; slope a_i = signs[i]*exp(gamma_i)
    # The fixed signs array is used to enforce monotonicity of the piecewise linear map.
    # variable vector is theta = [gamma_0..gamma_{K-1}, beta_0..beta_{K-1}] -> size n=2K

def n_pieces(spec: PWLinSpec) -> int:
    return len(spec.knots) - 1

# Evaluate s(x) and piece indices
def s_eval_piecewise(x: np.ndarray, gammas: np.ndarray, betas: np.ndarray, spec: PWLinSpec) -> np.ndarray:
    K = n_pieces(spec)
    y = np.empty_like(x)
    # map x to its piece via digitize
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a = spec.signs[idx] * np.exp(gammas[idx])
    b = betas[idx]
    y[:] = a * x + b
    return y

# ===========================
# Objective  c(x,y)=sqrt(2|x-y|)
# with analytic gradient
# ===========================
# def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
#                    xgrid: GridSpec, spec: PWLinSpec) -> Tuple[float, np.ndarray, np.ndarray]:
#     """
#     J = ∫ mu(x) * sqrt(2 * | x - s(x) |) dx   (integrated on xgrid)
#     Returns J, grad (size 2K), and a small diagonal Hessian (for stability).
#     """
#     K = n_pieces(spec)
#     x = xgrid.points
#     w = xgrid.weights
#     mu = mu_pdf(x)

#     # which piece each x falls into
#     idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)

#     # slopes/intercepts
#     a_all = spec.signs * np.exp(gammas)   # (K,)
#     a_x   = a_all[idx]
#     b_x   = betas[idx]

#     # map and residual
#     s_x  = a_x * x + b_x
#     r_x  = x - s_x
#     abs_r = np.abs(r_x)

#     # cost = sqrt(2 * |r|)
#     cost = np.sqrt(2.0 * abs_r)
#     J = np.sum(mu * cost * w)

#     # ----- gradient -----
#     # For r != 0: d/ds sqrt(2|r|) = - sign(r) / sqrt(2|r|)
#     # Chain rule through s(x)=a_i x + b_i on the active piece.
#     eps = 1e-12  # avoid division by 0
#     inv_sqrt_2absr = 1.0 / np.sqrt(2.0 * np.maximum(abs_r, eps))
#     sgn_r = np.sign(r_x)  # subgradient at 0; OK in practice

#     grad = np.zeros(2 * K)
#     for i in range(K):
#         mask = (idx == i)
#         if not np.any(mask):
#             continue
#         a_i = a_all[i]
#         xi, wi = x[mask], w[mask]
#         mui = mu[mask]
#         sgn_i = sgn_r[mask]
#         inv_i = inv_sqrt_2absr[mask]

#         # dJ/dgamma_i: ∂s/∂gamma_i = a_i * x  (since a_i = sign_i * exp(gamma_i))
#         grad[i] = -np.sum(mui * sgn_i * inv_i * (a_i * xi) * wi)

#         # dJ/dbeta_i: ∂s/∂beta_i = 1
#         grad[K + i] = -np.sum(mui * sgn_i * inv_i * 1.0 * wi)

#     # small positive-definite Hessian for numerical robustness (objective is non-smooth)
#     H = 1e-10 * np.eye(2 * K)
#     return J, grad, H
def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
                   xgrid: GridSpec, spec: PWLinSpec,
                   eps: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Smoothed objective for c(r) = sqrt(2|r|) via
        c_eps(r) = sqrt(2) * (r^2 + eps^2)^(1/4).
    Returns J, grad (size 2K), and a PSD Gauss-Newton-like Hessian.
    """
    K = n_pieces(spec)
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    # piece membership and piece parameters per-sample
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a_all = spec.signs * np.exp(gammas)     # (K,)
    a_x   = a_all[idx]
    b_x   = betas[idx]

    # residual r = x - s(x)
    s_x  = a_x * x + b_x
    r    = x - s_x
    q    = r*r + eps*eps                   # r^2 + eps^2

    # c_eps(r) = sqrt(2) * q^(1/4)
    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr = (sqrt(2)/2) * r * q^(-3/4)
    dc_dr = (np.sqrt(2.0)/2.0) * r * np.power(q, -0.75)

    # grad wrt theta on each piece
    grad = np.zeros(2*K)
    for i in range(K):
        msk = (idx == i)
        if not np.any(msk):
            continue
        a_i = a_all[i]
        xi, wi = x[msk], w[msk]
        mui    = mu[msk]
        dci    = dc_dr[msk]

        # ∂s/∂gamma_i = a_i * x, ∂s/∂beta_i = 1
        # ∂J/∂theta = ∫ mu * (dc/dr) * (-∂s/∂theta) dx
        grad[i]      = -np.sum(mui * dci * (a_i * xi) * wi)
        grad[K + i]  = -np.sum(mui * dci * 1.0        * wi)

    # ----- PSD Hessian (Gauss–Newton style with curvature clipping) -----
    # d2c/dr2 = (sqrt(2)/2) * (q)^(-7/4) * (eps^2 - 0.5*r^2)
    d2c = (np.sqrt(2.0)/2.0) * np.power(q, -1.75) * (eps*eps - 0.5 * r*r)
    w_curv = np.maximum(d2c, 0.0)   # clip to keep PSD

    H = np.zeros((2*K, 2*K))
    for i in range(K):
        msk_i = (idx == i)
        if not np.any(msk_i):
            continue
        a_i = a_all[i]
        xi, wi = x[msk_i], w[msk_i]
        mui    = mu[msk_i]
        wi_curv = w_curv[msk_i]

        # features for this piece: [a_i * x, 1]
        f1 = a_i * xi
        f2 = np.ones_like(xi)

        # weight = mu * w * w_curv
        ww = mui * wi * wi_curv

        # accumulate 2x2 block for (gamma_i, beta_i)
        H[i, i]             += np.sum(ww * f1 * f1)
        H[i, K + i]         += np.sum(ww * f1 * f2)
        H[K + i, i]         += np.sum(ww * f2 * f1)
        H[K + i, K + i]     += np.sum(ww * f2 * f2)

    # mild diagonal regularization for numerical stability
    H += 1e-10 * np.eye(2*K)
    return J, grad, H


# ==========================================================
# PF constraint zeta(d) = P_f(s)(d) - g(d) on a d-grid
# and analytic Jacobian A = d zeta / d theta
# ==========================================================
def constraint_zeta_pw(gammas: np.ndarray, betas: np.ndarray, dgrid: GridSpec, spec: PWLinSpec) -> Dict[str, np.ndarray]:
    """
    Returns:
      zeta : shape (m_d,)
      A    : shape (m_d, 2K) for theta=[gammas..., betas...]
    Uses formula:
      For piece i with slope a_i = sign_i * exp(gamma_i),
        u_i(d) = (d - beta_i) / a_i = sign_i * exp(-gamma_i) * (d - beta_i)
        contrib_i(d) = mu(u_i) / |a_i| = exp(-gamma_i) * mu(u_i)
      zeta(d) = sum_i contrib_i(d) * 1_{u_i in [x_i, x_{i+1}]} - nu(d)
    Jacobians:
      ∂/∂gamma_i contrib = -exp(-gamma_i) * [mu(u) + u * mu'(u)]
      ∂/∂beta_i  contrib = -sign_i * exp(-2*gamma_i) * mu'(u)
    """
    K = n_pieces(spec)
    d = dgrid.points
    m = dgrid.m
    zeta = np.zeros(m)
    A = np.zeros((m, 2*K))

    for i in range(K):
        sgn = spec.signs[i]
        eg = np.exp(gammas[i])
        emg = np.exp(-gammas[i])      # 1/|a_i|
        a_i = sgn * eg

        # preimage u(d) on this piece
        u = (d - betas[i]) / a_i      # = sgn * exp(-gamma) * (d - beta)
        # only valid if u in the segment
        L, R = spec.knots[i], spec.knots[i+1]
        valid = (u >= L) & (u <= R)
        if not np.any(valid):
            continue

        mu_u  = mu_pdf(u)
        mu_u_p = mu_pdf_prime(u)

        contrib = emg * mu_u
        zeta += contrib * valid

        # Jacobians (only where valid)
        d_contrib_dgamma = -emg * (mu_u + u * mu_u_p)
        d_contrib_dbeta  = -sgn * (emg**2) * mu_u_p

        A[valid, i]     += d_contrib_dgamma[valid]
        A[valid, K + i] += d_contrib_dbeta[valid]

    # subtract target density
    zeta -= nu_pdf(d)
    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian pieces & single SQP step (PW)
# ===========================================
def build_lagrangian_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                        xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J, gJ, HJ = objective_J_pw(gammas, betas, xgrid, spec)
    parts = constraint_zeta_pw(gammas, betas, dgrid, spec)
    zeta, A = parts["zeta"], parts["A"]
    # gradient of L
    g = gJ + A.T @ lam
    # (we keep H = HJ; constraint second-derivatives omitted for stability)
    H = HJ
    return g, H, A, J

def sqp_step_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec,
                reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
    zeta = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

    n = 2 * n_pieces(spec)
    m = dgrid.m

    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    new_gammas = gammas + dtheta[:n_pieces(spec)]
    new_betas  = betas  + dtheta[n_pieces(spec):]
    new_lam = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res

# =====================
# Full solver (PW-SQP)
# =====================
def sqp_solve_piecewise(
    spec: PWLinSpec,
    xgrid_m: int = 1001,  # integration grid for objective on x
    dgrid_m: int = 401,   # PF constraint grid on d
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    init_gammas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.5,
) -> Dict[str, np.ndarray]:

    K = n_pieces(spec)
    xgrid = make_uniform_grid(-10.0, 10.0, xgrid_m)
    dgrid = make_uniform_grid(-10.0, 10.0, dgrid_m)

    gammas = np.zeros(K) if init_gammas is None else init_gammas.copy()
    betas  = np.zeros(K) if init_betas  is None else init_betas.copy()
    lam = np.zeros(dgrid.m)

    hist = {"J": [], "opt_res": [], "feas_res": [], "gammas": [], "betas": []}

    for it in range(max_iter):
        g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
        zeta_now = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["gammas"].append(gammas.copy())
        hist["betas"].append(betas.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res = sqp_step_pw(
            gammas, betas, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        # simple trust-region clip in infinity norm on the parameter update
        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_gammas = gammas + dtheta[:K]
                new_betas  = betas  + dtheta[K:]

        gammas, betas, lam = new_gammas, new_betas, new_lam

    return {
        "gammas": gammas, "betas": betas,
        "slopes": spec.signs * np.exp(gammas),
        "lambda": lam, "history": hist, "spec": spec,
        "xgrid": xgrid, "dgrid": dgrid
    }

# =============================
# Example usage for Fig. 1 map
# =============================
if __name__ == "__main__":
    # Knots chosen to match the example's pieces:
    # (-10,-9), (-9,-1), (-1,1), (1,9), (9,10)
    knots = np.array([-10., -9., -1., 1., 9., 10.])
    signs = np.array([-1, -1, -1, -1, -1])  # slope is -exp(gamma_i)

    spec = PWLinSpec(knots=knots, signs=signs)

    # A near-analytic initialization (optional, speeds convergence):
    # desired s(x) ~ -x on |x|<1 or |x|>9  => beta=0
    # desired s(x) ~ -x - 10 on (-9,-1)    => beta=-10
    # desired s(x) ~ -x + 10 on (1,9)      => beta=+10
    init_gammas = np.zeros(len(signs))  # slope ~ -1
    init_betas  = np.array([0., -10., 0., +10., 0.])

    sol = sqp_solve_piecewise(
        spec,
        xgrid_m=1001, dgrid_m=1001,
        max_iter=100, tol_opt=1e-1, tol_feas=1e-3,
        init_gammas=init_gammas, init_betas=init_betas,
        trust_clip=0.25
    )
    # init_gammas=np.random.uniform(-1, 1, 5), init_betas=np.random.uniform(-10, 10, 5)
    print("Slopes a_i:", sol["slopes"])
    print("Betas  b_i:", sol["betas"])
    print("Final opt_res, feas_res:", sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])

"""In cell `Tv5_5V1lXnWu`, the `objective_J_pw(gammas, betas, xgrid, spec, eps)` function calculates the value of the objective function for the piecewise linear transport map.

The objective being minimized is a measure of the "cost" of transporting mass from the source distribution ($\mu$) to the target distribution ($\nu$) using the map $s(x)$. The specific cost function used here is based on the squared Euclidean distance, but with a square root: $c(x, y) = \sqrt{2|x-y|}$. The objective is the expected value of this cost under the source distribution:
$$ J = \mathbb{E}_{\mu}[\sqrt{2|X - s(X)|}] = \int \mu(x) \sqrt{2|x - s(x)|} \, dx $$

The function implements a **smoothed version** of this objective to make it differentiable, as the absolute value and square root can cause issues for optimization. It uses the smoothing:
$$ c_\epsilon(r) = \sqrt{2} (r^2 + \epsilon^2)^{1/4} $$
where $r = x - s(x)$ is the residual, and $\epsilon$ is a small positive value (`eps`).

The function calculates:

1.  The value of this smoothed objective function `J` by integrating $\mu(x) \cdot c_\epsilon(x - s(x))$ over the `xgrid` using trapezoidal integration.
2.  The **gradient** of this smoothed objective with respect to the optimization variables (`gammas` and `betas`) using an analytic formula derived from the smoothed cost.
3.  A **PSD (Positive Semi-Definite) Gauss-Newton-like Hessian** of the objective. This is an approximation of the true Hessian that is more robust for non-linear least squares problems like this, especially with the smoothed cost. It includes a curvature clipping step (`w_curv = np.maximum(d2c, 0.0)`) to ensure positive semi-definiteness.

It returns the calculated objective value `J`, the `grad`ient, and the approximated `H`essian. This function is a core component used by the SQP solver to evaluate the objective and its derivatives during the optimization process.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =========================
# Measures for Example 1.1
# =========================
# Domain is [-10, 10], rho(x) = sin((pi/5) x).
# We use normalized densities mu = rho_+ / Z, nu = rho_- / Z, Z = ∫ rho_+ = ∫ rho_- = 20/pi.
PI = np.pi
C = PI / 5.0          # frequency
Z = 20.0 / PI         # exact positive/negative mass on [-10, 10]
Z_INV = 1.0 / Z       # = PI / 20

def rho_signed(x: np.ndarray) -> np.ndarray:
    return np.sin(C * x)

def mu_pdf(x: np.ndarray) -> np.ndarray:
    # mu(x) = (pi/20) * max(sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    # nu(x) = (pi/20) * max(-sin(C x), 0)
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    # derivative of mu: (pi/20) * 1_{sin>0} * C * cos(C x)
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# ============
# Grids
# ============
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray # shape (m,)
    weights: np.ndarray # trapezoid weights for integrations

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    if m >= 2:
        w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

# ====================================
# Piecewise linear map specification
# ====================================
@dataclass
class PWLinSpec:
    # knots define K pieces on [knots[i], knots[i+1]]
    knots: np.ndarray      # shape (K+1,)
    signs: np.ndarray      # shape (K,), each in {+1, -1}; slope a_i = signs[i]*exp(gamma_i)
    # The fixed signs array is used to enforce monotonicity of the piecewise linear map.
    # variable vector is theta = [gamma_0..gamma_{K-1}, beta_0..beta_{K-1}] -> size n=2K

def n_pieces(spec: PWLinSpec) -> int:
    return len(spec.knots) - 1

# Evaluate s(x) and piece indices
def s_eval_piecewise(x: np.ndarray, gammas: np.ndarray, betas: np.ndarray, spec: PWLinSpec) -> np.ndarray:
    K = n_pieces(spec)
    y = np.empty_like(x)
    # map x to its piece via digitize
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a = spec.signs[idx] * np.exp(gammas[idx])
    b = betas[idx]
    y[:] = a * x + b
    return y

# ===========================
# Objective  c(x,y)=sqrt(2|x-y|)
# with analytic gradient
# ===========================
# def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
#                    xgrid: GridSpec, spec: PWLinSpec) -> Tuple[float, np.ndarray, np.ndarray]:
#     """
#     J = ∫ mu(x) * sqrt(2 * | x - s(x) |) dx   (integrated on xgrid)
#     Returns J, grad (size 2K), and a small diagonal Hessian (for stability).
#     """
#     K = n_pieces(spec)
#     x = xgrid.points
#     w = xgrid.weights
#     mu = mu_pdf(x)

#     # which piece each x falls into
#     idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)

#     # slopes/intercepts
#     a_all = spec.signs * np.exp(gammas)   # (K,)
#     a_x   = a_all[idx]
#     b_x   = betas[idx]

#     # map and residual
#     s_x  = a_x * x + b_x
#     r_x  = x - s_x
#     abs_r = np.abs(r_x)

#     # cost = sqrt(2 * |r|)
#     cost = np.sqrt(2.0 * abs_r)
#     J = np.sum(mu * cost * w)

#     # ----- gradient -----
#     # For r != 0: d/ds sqrt(2|r|) = - sign(r) / sqrt(2|r|)
#     # Chain rule through s(x)=a_i x + b_i on the active piece.
#     eps = 1e-12  # avoid division by 0
#     inv_sqrt_2absr = 1.0 / np.sqrt(2.0 * np.maximum(abs_r, eps))
#     sgn_r = np.sign(r_x)  # subgradient at 0; OK in practice

#     grad = np.zeros(2 * K)
#     for i in range(K):
#         mask = (idx == i)
#         if not np.any(mask):
#             continue
#         a_i = a_all[i]
#         xi, wi = x[mask], w[mask]
#         mui = mu[mask]
#         sgn_i = sgn_r[mask]
#         inv_i = inv_sqrt_2absr[mask]

#         # dJ/dgamma_i: ∂s/∂gamma_i = a_i * x  (since a_i = sign_i * exp(gamma_i))
#         grad[i] = -np.sum(mui * sgn_i * inv_i * (a_i * xi) * wi)

#         # dJ/dbeta_i: ∂s/∂beta_i = 1
#         grad[K + i] = -np.sum(mui * sgn_i * inv_i * 1.0 * wi)

#     # small positive-definite Hessian for numerical robustness (objective is non-smooth)
#     H = 1e-10 * np.eye(2 * K)
#     return J, grad, H
def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
                   xgrid: GridSpec, spec: PWLinSpec,
                   eps: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Smoothed objective for c(r) = sqrt(2|r|) via
        c_eps(r) = sqrt(2) * (r^2 + eps^2)^(1/4).
    Returns J, grad (size 2K), and a PSD Gauss-Newton-like Hessian.
    """
    K = n_pieces(spec)
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    # piece membership and piece parameters per-sample
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a_all = spec.signs * np.exp(gammas)     # (K,)
    a_x   = a_all[idx]
    b_x   = betas[idx]

    # residual r = x - s(x)
    s_x  = a_x * x + b_x
    r    = x - s_x
    q    = r*r + eps*eps                   # r^2 + eps^2

    # c_eps(r) = sqrt(2) * q^(1/4)
    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr = (sqrt(2)/2) * r * q^(-3/4)
    dc_dr = (np.sqrt(2.0)/2.0) * r * np.power(q, -0.75)

    # grad wrt theta on each piece
    grad = np.zeros(2*K)
    for i in range(K):
        msk = (idx == i)
        if not np.any(msk):
            continue
        a_i = a_all[i]
        xi, wi = x[msk], w[msk]
        mui    = mu[msk]
        dci    = dc_dr[msk]

        # ∂s/∂gamma_i = a_i * x, ∂s/∂beta_i = 1
        # ∂J/∂theta = ∫ mu * (dc/dr) * (-∂s/∂theta) dx
        grad[i]      = -np.sum(mui * dci * (a_i * xi) * wi)
        grad[K + i]  = -np.sum(mui * dci * 1.0        * wi)

    # ----- PSD Hessian (Gauss–Newton style with curvature clipping) -----
    # d2c/dr2 = (sqrt(2)/2) * (q)^(-7/4) * (eps^2 - 0.5*r^2)
    d2c = (np.sqrt(2.0)/2.0) * np.power(q, -1.75) * (eps*eps - 0.5 * r*r)
    w_curv = np.maximum(d2c, 0.0)   # clip to keep PSD

    H = np.zeros((2*K, 2*K))
    for i in range(K):
        msk_i = (idx == i)
        if not np.any(msk_i):
            continue
        a_i = a_all[i]
        xi, wi = x[msk_i], w[msk_i]
        mui    = mu[msk_i]
        wi_curv = w_curv[msk_i]

        # features for this piece: [a_i * x, 1]
        f1 = a_i * xi
        f2 = np.ones_like(xi)

        # weight = mu * w * w_curv
        ww = mui * wi * wi_curv

        # accumulate 2x2 block for (gamma_i, beta_i)
        H[i, i]             += np.sum(ww * f1 * f1)
        H[i, K + i]         += np.sum(ww * f1 * f2)
        H[K + i, i]         += np.sum(ww * f2 * f1)
        H[K + i, K + i]     += np.sum(ww * f2 * f2)

    # mild diagonal regularization for numerical stability
    H += 1e-10 * np.eye(2*K)
    return J, grad, H


# ==========================================================
# PF constraint zeta(d) = P_f(s)(d) - g(d) on a d-grid
# and analytic Jacobian A = d zeta / d theta
# ==========================================================
def constraint_zeta_pw(gammas: np.ndarray, betas: np.ndarray, dgrid: GridSpec, spec: PWLinSpec) -> Dict[str, np.ndarray]:
    """
    Returns:
      zeta : shape (m_d,)
      A    : shape (m_d, 2K) for theta=[gammas..., betas...]
    Uses formula:
      For piece i with slope a_i = sign_i * exp(gamma_i),
        u_i(d) = (d - beta_i) / a_i = sign_i * exp(-gamma_i) * (d - beta_i)
        contrib_i(d) = mu(u_i) / |a_i| = exp(-gamma_i) * mu(u_i)
      zeta(d) = sum_i contrib_i(d) * 1_{u_i in [x_i, x_{i+1}]} - nu(d)
    Jacobians:
      ∂/∂gamma_i contrib = -exp(-gamma_i) * [mu(u) + u * mu'(u)]
      ∂/∂beta_i  contrib = -sign_i * exp(-2*gamma_i) * mu'(u)
    """
    K = n_pieces(spec)
    d = dgrid.points
    m = dgrid.m
    zeta = np.zeros(m)
    A = np.zeros((m, 2*K))

    for i in range(K):
        sgn = spec.signs[i]
        eg = np.exp(gammas[i])
        emg = np.exp(-gammas[i])      # 1/|a_i|
        a_i = sgn * eg

        # preimage u(d) on this piece
        u = (d - betas[i]) / a_i      # = sgn * exp(-gamma) * (d - beta)
        # only valid if u in the segment
        L, R = spec.knots[i], spec.knots[i+1]
        valid = (u >= L) & (u <= R)
        if not np.any(valid):
            continue

        mu_u  = mu_pdf(u)
        mu_u_p = mu_pdf_prime(u)

        contrib = emg * mu_u
        zeta += contrib * valid

        # Jacobians (only where valid)
        d_contrib_dgamma = -emg * (mu_u + u * mu_u_p)
        d_contrib_dbeta  = -sgn * (emg**2) * mu_u_p

        A[valid, i]     += d_contrib_dgamma[valid]
        A[valid, K + i] += d_contrib_dbeta[valid]

    # subtract target density
    zeta -= nu_pdf(d)
    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian pieces & single SQP step (PW)
# ===========================================
def build_lagrangian_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                        xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J, gJ, HJ = objective_J_pw(gammas, betas, xgrid, spec)
    parts = constraint_zeta_pw(gammas, betas, dgrid, spec)
    zeta, A = parts["zeta"], parts["A"]
    # gradient of L
    g = gJ + A.T @ lam
    # (we keep H = HJ; constraint second-derivatives omitted for stability)
    H = HJ
    return g, H, A, J

def sqp_step_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec,
                reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
    zeta = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

    n = 2 * n_pieces(spec)
    m = dgrid.m

    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    new_gammas = gammas + dtheta[:n_pieces(spec)]
    new_betas  = betas  + dtheta[n_pieces(spec):]
    new_lam = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res

# =====================
# Full solver (PW-SQP)
# =====================
def sqp_solve_piecewise(
    spec: PWLinSpec,
    xgrid_m: int = 1001,  # integration grid for objective on x
    dgrid_m: int = 401,   # PF constraint grid on d
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    init_gammas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.5,
) -> Dict[str, np.ndarray]:

    K = n_pieces(spec)
    xgrid = make_uniform_grid(-10.0, 10.0, xgrid_m)
    dgrid = make_uniform_grid(-10.0, 10.0, dgrid_m)

    gammas = np.zeros(K) if init_gammas is None else init_gammas.copy()
    betas  = np.zeros(K) if init_betas  is None else init_betas.copy()
    lam = np.zeros(dgrid.m)

    hist = {"J": [], "opt_res": [], "feas_res": [], "gammas": [], "betas": []}

    for it in range(max_iter):
        g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
        zeta_now = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["gammas"].append(gammas.copy())
        hist["betas"].append(betas.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res = sqp_step_pw(
            gammas, betas, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        # simple trust-region clip in infinity norm on the parameter update
        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_gammas = gammas + dtheta[:K]
                new_betas  = betas  + dtheta[K:]

        gammas, betas, lam = new_gammas, new_betas, new_lam

    return {
        "gammas": gammas, "betas": betas,
        "slopes": spec.signs * np.exp(gammas),
        "lambda": lam, "history": hist, "spec": spec,
        "xgrid": xgrid, "dgrid": dgrid
    }

# =============================
# Example usage for Fig. 1 map
# =============================
if __name__ == "__main__":
    # Knots chosen to match the example's pieces:
    # (-10,-9), (-9,-1), (-1,1), (1,9), (9,10)
    knots = np.array([-10., -9., -1., 1., 9., 10.])
    signs = np.array([-1, -1, -1, -1, -1])  # slope is -exp(gamma_i)

    spec = PWLinSpec(knots=knots, signs=signs)

    # A near-analytic initialization (optional, speeds convergence):
    # desired s(x) ~ -x on |x|<1 or |x|>9  => beta=0
    # desired s(x) ~ -x - 10 on (-9,-1)    => beta=-10
    # desired s(x) ~ -x + 10 on (1,9)      => beta=+10
    init_gammas = np.zeros(len(signs))  # slope ~ -1
    init_betas  = np.array([0., -10., 0., +10., 0.])

    sol = sqp_solve_piecewise(
        spec,
        xgrid_m=1001, dgrid_m=1001,
        max_iter=100, tol_opt=1e-1, tol_feas=1e-3,
        init_gammas=init_gammas, init_betas=init_betas,
        trust_clip=0.25
    )
    # init_gammas=np.random.uniform(-1, 1, 5), init_betas=np.random.uniform(-10, 10, 5)
    print("Slopes a_i:", sol["slopes"])
    print("Betas  b_i:", sol["betas"])
    print("Final opt_res, feas_res:", sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])

# ==========================
# True solution (Example 1.1)
# ==========================
def true_piece_params_for_example(spec: PWLinSpec):
    """
    Returns (a_true, b_true) per piece, aligned with `spec.knots`.
    For Example 1.1 ([-10,-9],[-9,-1],[-1,1],[1,9],[9,10]):
        s(x) = -x-10 on (-9,-1)
             = -x+10 on (1,9)
             = -x     otherwise
    Assumes each piece matches exactly one of those intervals.
    """
    knots = spec.knots
    K = n_pieces(spec)
    a_true = np.zeros(K)
    b_true = np.zeros(K)

    # all slopes are -1
    a_true[:] = -1.0

    def find_piece(L, R):
        idx = np.where((np.isclose(knots[:-1], L)) & (np.isclose(knots[1:], R)))[0]
        if len(idx) != 1:
            raise ValueError(f"Expected one piece for interval ({L},{R}), found {len(idx)}.")
        return int(idx[0])

    # set the two shifted middle pieces
    i_neg = find_piece(-9.0, -1.0)  # s = -x - 10
    i_pos = find_piece( 1.0,  9.0)  # s = -x + 10
    b_true[:] = 0.0
    b_true[i_neg] = -10.0
    b_true[i_pos] = +10.0

    return a_true, b_true


def true_map_example(x: np.ndarray) -> np.ndarray:
    """
    Vectorized true mapping s(x) for the example.
    """
    y = -x.copy()
    y = np.where((-9.0 < x) & (x < -1.0), -x - 10.0, y)
    y = np.where(( 1.0 < x) & (x <  9.0), -x + 10.0, y)
    return y


# ==========================================
# Per-piece comparison & metrics (report)
# ==========================================
def compare_to_true_per_piece(sol: Dict, return_table: bool = False):
    """
    Prints a neat per-piece report comparing learned vs. true (a,b)
    and computing RMSE / max|err| on the x-grid within each piece.
    If return_table=True, also returns a dict-of-lists with metrics.
    """
    spec: PWLinSpec = sol["spec"]
    xgrid: GridSpec = sol["xgrid"]
    K = n_pieces(spec)

    a_hat = sol["slopes"]   # learned slopes
    b_hat = sol["betas"]    # learned intercepts
    a_true, b_true = true_piece_params_for_example(spec)

    x = xgrid.points
    w = xgrid.weights
    knots = spec.knots

    # headers
    print("\n=== Per-piece comparison (learned vs. true) ===")
    print("piece | interval         | a_hat     a_true   Δa        | b_hat     b_true   Δb        | RMSE(x)   max|err|")
    print("------|-------------------|-------------------------------|-------------------------------|------------------")

    rows = {"piece": [], "L": [], "R": [], "a_hat": [], "a_true": [], "da": [],
            "b_hat": [], "b_true": [], "db": [], "rmse": [], "max_abs": []}

    for i in range(K):
        L, R = knots[i], knots[i+1]
        mask = (x >= L) & (x <= R)
        # predicted and true on this piece
        y_hat = a_hat[i] * x[mask] + b_hat[i]
        y_true = a_true[i] * x[mask] + b_true[i]
        err = y_hat - y_true

        # weighted RMSE on x-grid portion
        w_i = w[mask]
        rmse = np.sqrt(np.sum((err**2) * w_i) / np.sum(w_i)) if np.any(mask) else np.nan
        max_abs = np.max(np.abs(err)) if np.any(mask) else np.nan

        da = a_hat[i] - a_true[i]
        db = b_hat[i] - b_true[i]

        print(f"{i:>5d} | ({L:>5.1f},{R:>5.1f}) | {a_hat[i]:>8.4f} {a_true[i]:>8.4f} {da:>8.4e} |"
              f" {b_hat[i]:>8.4f} {b_true[i]:>8.4f} {db:>8.4e} | {rmse:>8.4e} {max_abs:>9.4e}")

        rows["piece"].append(i); rows["L"].append(L); rows["R"].append(R)
        rows["a_hat"].append(a_hat[i]); rows["a_true"].append(a_true[i]); rows["da"].append(da)
        rows["b_hat"].append(b_hat[i]); rows["b_true"].append(b_true[i]); rows["db"].append(db)
        rows["rmse"].append(rmse); rows["max_abs"].append(max_abs)

    if return_table:
        return rows


# ===========================
# Optional: quick plot overlay
# ===========================
def plot_overlay_solution(sol: Dict, num_pts_per_piece: int = 200):
    """
    Draws learned vs. true mapping piece-by-piece.
    (Requires matplotlib; safe to call in Colab/locally.)
    """
    import matplotlib.pyplot as plt

    spec: PWLinSpec = sol["spec"]
    K = n_pieces(spec)
    a_hat = sol["slopes"]; b_hat = sol["betas"]
    a_true, b_true = true_piece_params_for_example(spec)

    plt.figure(figsize=(8,5))
    for i in range(K):
        L, R = spec.knots[i], spec.knots[i+1]
        xx = np.linspace(L, R, num_pts_per_piece)
        y_hat = a_hat[i]*xx + b_hat[i]
        y_tru = a_true[i]*xx + b_true[i]
        # learned
        plt.plot(xx, y_hat, lw=2)
        # true (dashed)
        plt.plot(xx, y_tru, linestyle="--")
    plt.title("Piecewise transport: learned (solid) vs. true (dashed)")
    plt.xlabel("x")
    plt.ylabel("s(x)")
    plt.grid(True, alpha=0.3)
    plt.show()

# sol = sqp_solve_piecewise(
#     spec,
#     xgrid_m=1001,   # integration grid for x
#     dgrid_m=401,    # PF constraint grid for d
#     max_iter=50,
#     tol_opt=1e-9,
#     tol_feas=1e-9,
#     init_gammas=init_gammas,
#     init_betas=init_betas,
#     trust_clip=0.25
# )
compare_to_true_per_piece(sol)
plot_overlay_solution(sol)

print(sol["spec"], sol["xgrid"],  sol["slopes"], sol["betas"])

# --- half-open validity to avoid boundary double-counting ---
def _valid_on_piece(u: np.ndarray, L: float, R: float, is_last: bool, tol: float = 1e-12):
    if is_last:
        return (u >= L - tol) & (u <= R + tol)   # [L, R]
    else:
        return (u >= L - tol) & (u <  R - tol)   # [L, R)

def learned_target_pdf(sol, d: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Same as learned_target_pdf but uses half-open piece ownership.
    """
    spec = sol["spec"]
    gammas, betas = sol["gammas"], sol["betas"]
    K = len(gammas)

    Pf = np.zeros_like(d, dtype=float)
    for i in range(K):
        sgn = spec.signs[i]
        eg  = np.exp(gammas[i])
        emg = np.exp(-gammas[i])                 # 1/|a_i|
        a_i = sgn * eg

        u = (d - betas[i]) / a_i
        L, R = spec.knots[i], spec.knots[i+1]
        valid = _valid_on_piece(u, L, R, is_last=(i == K-1), tol=tol)
        if not np.any(valid):
            continue
        u_v = u[valid]
        Pf[valid] += emg * mu_pdf(u_v)           # Pf(s)(d) = sum mu(u)/|a|
    return Pf


def _knot_images(spec, slopes, betas):
    imgs = []
    for i in range(len(slopes)):
        L, R = spec.knots[i], spec.knots[i+1]
        a, b = slopes[i], betas[i]
        imgs.extend([a*L + b, a*R + b])
    return np.array(imgs, dtype=float)

def linspace_without(L, R, n, avoid=(), eps=1e-9):
    x = np.linspace(L, R, n)
    for a in avoid:
        x = x[np.abs(x - a) > eps]
    return x

def plot_source_and_targets(sol, num_pts: int = 202, skip_images: bool = True):
    import matplotlib.pyplot as plt
    L, R = sol["spec"].knots[0], sol["spec"].knots[-1]

    avoid = ()
    if skip_images:
        avoid = _knot_images(sol["spec"], sol["slopes"], sol["betas"])

    x = linspace_without(L, R, num_pts, avoid=avoid)   # skip exact images for cleanliness
    d = linspace_without(L, R, num_pts, avoid=avoid)

    mu_x = mu_pdf(x)
    nu_d = nu_pdf(d)
    pf_d = learned_target_pdf(sol, d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title(r"Source density  $\mu(x)$")
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label=r"Learned  $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label=r"True  $\nu(d)$")
    ax.set_title(r"Target densities on $d$")
    ax.set_xlabel("d"); ax.grid(alpha=0.3)
    ax.legend(loc="upper left")


    fig.suptitle("Source vs. Learned Target vs. True Target", y=1.02, fontsize=13)
    fig.tight_layout(); plt.show()
plot_source_and_targets(sol)

"""## Uniform Partition"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ======================================================
# Example 1.1 measures on [-10,10]: mu = rho_+, nu = rho_-
# ======================================================
PI = np.pi
C = PI / 5.0          # frequency for sin((pi/5) x)
Z = 20.0 / PI         # ∫_{-10}^{10} max(sin,0) = ∫ max(-sin,0) = 20/pi
Z_INV = 1.0 / Z       # = pi / 20

def mu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    # derivative of mu: (pi/20) * 1_{sin>0} * C * cos(C x)
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# =========================
# Uniform grids & weights
# =========================
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray
    weights: np.ndarray   # trapezoidal weights

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    assert m >= 2, "m must be >= 2 for trapezoid rule"
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

def interior_linspace(L: float, R: float, n: int) -> np.ndarray:
    """Cell-center grid: avoids endpoints exactly (great for plotting)."""
    h = (R - L) / n
    return L + (np.arange(n) + 0.5) * h

# =====================================
# Piecewise-linear map: uniform knots
# =====================================
@dataclass
class PWLinSpec:
    knots: np.ndarray      # shape (K+1,), uniform here
    signs: np.ndarray      # shape (K,), slope a_i = sign_i * exp(gamma_i)

def make_uniform_knots(L: float, R: float, K: int) -> np.ndarray:
    return np.linspace(L, R, K + 1)

def n_pieces(spec: PWLinSpec) -> int:
    return len(spec.knots) - 1

# Evaluate s(x) given gammas, betas (stable near knots)
def s_eval_piecewise(x: np.ndarray, gammas: np.ndarray, betas: np.ndarray, spec: PWLinSpec) -> np.ndarray:
    K = n_pieces(spec)
    eps_b = 1e-12 * (spec.knots[-1] - spec.knots[0])
    x_shift = np.clip(x, spec.knots[0] + eps_b, spec.knots[-1] - eps_b)
    idx = np.clip(np.digitize(x_shift, spec.knots) - 1, 0, K-1)
    a = spec.signs[idx] * np.exp(gammas[idx])
    b = betas[idx]
    return a * x + b

# =======================================
# Smoothed objective: J = ∫ mu(x) sqrt(2) (r^2+eps^2)^(1/4) dx
# =======================================
def objective_J_pw(gammas: np.ndarray, betas: np.ndarray, xgrid: GridSpec, spec: PWLinSpec,
                   eps: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    K = n_pieces(spec)
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    eps_b = 1e-12 * (xgrid.R - xgrid.L)
    x_shift = np.clip(x, xgrid.L + eps_b, xgrid.R - eps_b)
    idx = np.clip(np.digitize(x_shift, spec.knots) - 1, 0, K-1)

    a_all = spec.signs * np.exp(gammas)
    a_x   = a_all[idx]
    b_x   = betas[idx]

    r = x - (a_x * x + b_x)
    q = r*r + eps*eps

    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr = (sqrt(2)/2) * r * q^(-3/4)
    dc_dr = (np.sqrt(2.0)/2.0) * r * np.power(q, -0.75)

    grad = np.zeros(2*K)
    for i in range(K):
        msk = (idx == i)
        if not np.any(msk): continue
        a_i = a_all[i]
        xi, wi = x[msk], w[msk]
        mui    = mu[msk]
        dci    = dc_dr[msk]
        grad[i]      = -np.sum(mui * dci * (a_i * xi) * wi)  # d/dgamma: a_i * x
        grad[K + i]  = -np.sum(mui * dci * 1.0        * wi)  # d/dbeta: 1

    # Gauss–Newton PSD Hessian with curvature clipping
    d2c = (np.sqrt(2.0)/2.0) * np.power(q, -1.75) * (eps*eps - 0.5 * r*r)
    w_curv = np.maximum(d2c, 0.0)
    H = np.zeros((2*K, 2*K))
    for i in range(K):
        msk_i = (idx == i)
        if not np.any(msk_i): continue
        a_i = a_all[i]
        xi, wi = x[msk_i], w[msk_i]
        mui = mu[msk_i]; ww = mui * wi * w_curv[msk_i]
        f1 = a_i * xi; f2 = np.ones_like(xi)
        H[i, i]             += np.sum(ww * f1 * f1)
        H[i, K + i]         += np.sum(ww * f1 * f2)
        H[K + i, i]         += np.sum(ww * f2 * f1)
        H[K + i, K + i]     += np.sum(ww * f2 * f2)
    H += 1e-10 * np.eye(2*K)
    return J, grad, H

# ==========================================================
# PF constraint on d-grid: zeta(d) = P_f(s)(d) - nu(d)
# With half-open preimage ownership: [L,R) except last piece [L,R]
# ==========================================================
def _own_u_halfopen(u: np.ndarray, L: float, R: float, i: int, K: int, tol: float = 1e-12):
    if i == K - 1:
        return (u >= L - tol) & (u <= R + tol)
    else:
        return (u >= L - tol) & (u <  R - tol)

def constraint_zeta_pw(gammas: np.ndarray, betas: np.ndarray, dgrid: GridSpec, spec: PWLinSpec) -> Dict[str, np.ndarray]:
    K = n_pieces(spec)
    d = dgrid.points
    m = dgrid.m
    zeta = np.zeros(m)
    A = np.zeros((m, 2*K))

    for i in range(K):
        sgn = spec.signs[i]
        eg  = np.exp(gammas[i])
        emg = np.exp(-gammas[i])      # 1/|a_i|
        a_i = sgn * eg

        L, R = spec.knots[i], spec.knots[i+1]
        u = (d - betas[i]) / a_i
        valid = _own_u_halfopen(u, L, R, i, K, tol=1e-12)
        if not np.any(valid):
            continue

        mu_u   = mu_pdf(u)
        mu_u_p = mu_pdf_prime(u)

        zeta[valid] += emg * mu_u[valid]

        d_contrib_dgamma = -emg * (mu_u + u * mu_u_p)
        d_contrib_dbeta  = -sgn * (emg**2) * mu_u_p
        A[valid, i]       += d_contrib_dgamma[valid]
        A[valid, K + i]   += d_contrib_dbeta[valid]

    zeta -= nu_pdf(d)
    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian assembly & single SQP step
# ===========================================
def build_lagrangian_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                        xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec):
    J, gJ, HJ = objective_J_pw(gammas, betas, xgrid, spec)
    parts = constraint_zeta_pw(gammas, betas, dgrid, spec)
    zeta, A = parts["zeta"], parts["A"]
    g = gJ + A.T @ lam
    H = HJ  # keep constraint-second-derivatives off for stability
    return g, H, A, J, zeta

def sqp_step_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec,
                reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval, zeta = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)

    n = 2 * n_pieces(spec)
    m = dgrid.m
    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    K = n_pieces(spec)
    new_gammas = gammas + dtheta[:K]
    new_betas  = betas  + dtheta[K:]
    new_lam = lam + w

    opt_res = np.linalg.norm(g, np.inf)
    feas_res = np.linalg.norm(zeta, np.inf)
    return new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res

# =====================
# Full PW-SQP solver (uniform partition)
# =====================
def sqp_solve_piecewise_uniform(
    K: int,                        # number of uniform pieces
    L: float = -10.0,
    R: float =  10.0,
    signs: Optional[np.ndarray] = None,   # default: all -1
    xgrid_m: int = 1001,          # objective integration grid in x
    dgrid_m: int = 401,           # PF constraint grid in d
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    init_gammas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.5,
) -> Dict[str, np.ndarray]:
    knots = make_uniform_knots(L, R, K)
    if signs is None:
        signs = -np.ones(K, dtype=float)  # monotone decreasing default
    assert len(signs) == K
    spec = PWLinSpec(knots=knots, signs=signs)

    xgrid = make_uniform_grid(L, R, xgrid_m)
    dgrid = make_uniform_grid(L, R, dgrid_m)

    gammas = np.zeros(K) if init_gammas is None else init_gammas.copy()
    betas  = np.zeros(K) if init_betas  is None else init_betas.copy()
    lam = np.zeros(dgrid.m)

    hist = {"J": [], "opt_res": [], "feas_res": [], "gammas": [], "betas": []}

    for it in range(max_iter):
        g, H, A, Jval, zeta_now = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["gammas"].append(gammas.copy())
        hist["betas"].append(betas.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_gammas, new_betas, new_lam, dtheta, _, _ = sqp_step_pw(
            gammas, betas, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_gammas = gammas + dtheta[:K]
                new_betas  = betas  + dtheta[K:]

        gammas, betas, lam = new_gammas, new_betas, new_lam

    return {
        "gammas": gammas,
        "betas": betas,
        "slopes": signs * np.exp(gammas),
        "lambda": lam,
        "history": hist,
        "spec": spec,
        "xgrid": xgrid,
        "dgrid": dgrid
    }

# ==========================
# True solution helpers (for overlay)
# ==========================
def true_piece_params_for_example(spec: PWLinSpec):
    """Robust midpoint classification (no exact-knot assumptions)."""
    K = n_pieces(spec)
    mids = 0.5 * (spec.knots[:-1] + spec.knots[1:])
    a_true = -np.ones(K)
    b_true = np.zeros(K)
    b_true[(mids > -9.0) & (mids < -1.0)] = -10.0
    b_true[(mids >  1.0) & (mids <  9.0)] = +10.0
    return a_true, b_true

def true_map_example(x: np.ndarray) -> np.ndarray:
    y = -x.copy()
    y = np.where((-9.0 < x) & (x < -1.0), -x - 10.0, y)
    y = np.where(( 1.0 < x) & (x <  9.0), -x + 10.0, y)
    return y

# ===========================
# Learned target density for plotting (half-open ownership)
# ===========================
def learned_target_pdf(sol: Dict, d: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    spec = sol["spec"]
    gammas, betas, slopes = sol["gammas"], sol["betas"], sol["slopes"]
    K = len(gammas)
    d = np.asarray(d, float)
    Pf = np.zeros_like(d, dtype=float)
    for i in range(K):
        a_i = slopes[i]; b_i = betas[i]
        L, R = spec.knots[i], spec.knots[i+1]
        u = (d - b_i) / a_i
        valid = _own_u_halfopen(u, L, R, i, K, tol=tol)
        if not np.any(valid): continue
        Pf[valid] += np.exp(-gammas[i]) * mu_pdf(u[valid])  # mu(u)/|a|
    return Pf

# ===========================
# Plots
# ===========================
def plot_overlay_solution_uniform(sol: Dict, num_pts_per_piece: int = 200):
    """Overlay: learned (solid) vs true (dashed) on one interior grid."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    spec: PWLinSpec = sol["spec"]
    K = n_pieces(spec)
    a_hat = sol["slopes"]; b_hat = sol["betas"]
    a_true, b_true = true_piece_params_for_example(spec)

    Ldom, Rdom = float(spec.knots[0]), float(spec.knots[-1])
    eps = 1e-12 * (Rdom - Ldom)

    N = max(2, K * num_pts_per_piece)
    xx_all = interior_linspace(Ldom, Rdom, N)  # interior grid (no endpoints)

    learned_segs, true_segs = [], []

    for i in range(K):
        L, R = float(spec.knots[i]), float(spec.knots[i+1])
        Lp = L + (eps if i > 0     else 0.0)
        Rp = R - (eps if i < K - 1 else 0.0)
        mask = (xx_all > Lp) & (xx_all < Rp)
        if np.count_nonzero(mask) >= 2:
            xpi = xx_all[mask]
            learned_segs.append(np.column_stack([xpi, a_hat[i]*xpi + b_hat[i]]))
            true_segs.append(np.column_stack([xpi, a_true[i]*xpi + b_true[i]]))

    fig, ax = plt.subplots(figsize=(8, 5))
    if learned_segs:
        ax.add_collection(LineCollection(learned_segs, colors="red", linewidths=2, label="Learned map"))
    if true_segs:
        ax.add_collection(LineCollection(true_segs, colors="k", linewidths=2, linestyles="--", label="True map"))
    ax.set_xlim(Ldom, Rdom)
    if learned_segs:
        ymin = min(seg[:,1].min() for seg in (learned_segs + true_segs))
        ymax = max(seg[:,1].max() for seg in (learned_segs + true_segs))
        pad = 0.05 * (ymax - ymin + 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x"); ax.set_ylabel("s(x)")
    ax.set_title("Piecewise transport: learned (solid) vs. true (dashed)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.show()

def _knot_images(spec: PWLinSpec, slopes: np.ndarray, betas: np.ndarray) -> np.ndarray:
    imgs = []
    K = len(slopes)
    for i in range(K):
        L, R = spec.knots[i], spec.knots[i+1]
        a, b = slopes[i], betas[i]
        imgs.extend([a*L + b, a*R + b])
    return np.array(imgs, dtype=float)

def linspace_without(L: float, R: float, n: int, avoid=(), eps: float = 1e-9):
    x = np.linspace(L, R, n)
    if len(avoid):
        for a in avoid:
            x = x[np.abs(x - a) > eps]
    return x

def plot_source_and_targets(sol: Dict, num_pts: int = 2001, skip_images: bool = True):
    """Two-panel plot: μ(x) and Pf(s)(d) vs ν(d) with clean legend & no spikes."""
    import matplotlib.pyplot as plt
    L, R = sol["spec"].knots[0], sol["spec"].knots[-1]

    avoid = ()
    if skip_images:
        avoid = _knot_images(sol["spec"], sol["slopes"], sol["betas"])

    x = linspace_without(L, R, num_pts, avoid=avoid)     # avoid exact images
    d = linspace_without(L, R, num_pts, avoid=avoid)

    mu_x = mu_pdf(x)
    nu_d = nu_pdf(d)
    pf_d = learned_target_pdf(sol, d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title(r"Source density  $\mu(x)$")
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label=r"Learned  $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label=r"True  $\nu(d)$")
    ax.set_title(r"Target densities on $d$")
    ax.set_xlabel("d"); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # place legend outside

    fig.suptitle("Source vs. Learned Target vs. True Target", y=1.02, fontsize=13)
    fig.tight_layout()
    plt.show()

# ===========================
# Example run
# ===========================
# if __name__ == "__main__":
#     K = 100  # uniform partition into 100 pieces
#     sol = sqp_solve_piecewise_uniform(
#         K=K,
#         xgrid_m=1001,
#         dgrid_m=401,
#         max_iter=100,
#         tol_opt=1e-9,
#         tol_feas=1e-9,
#         trust_clip=0.25
#     )

if __name__ == "__main__":
    K = 4  # uniform partition into 100 pieces
    sol = sqp_solve_piecewise_uniform(
        K=K,
        xgrid_m=1001,
        dgrid_m=401,
        max_iter=100,
        tol_opt=1e-9,
        tol_feas=1e-9,
        trust_clip=0.25
    )
print("Final opt_res, feas_res:",
      sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])

# Plots
plot_overlay_solution_uniform(sol, num_pts_per_piece=200)
plot_source_and_targets(sol, num_pts=201, skip_images=True)

d_test = np.linspace(-10, 10, 2001)
print("min/max Pf on [-10,-5]:",
      learned_target_pdf(sol, d_test[(d_test>=-10)&(d_test<=-5)]).min(),
      learned_target_pdf(sol, d_test[(d_test>=-10)&(d_test<=-5)]).max())
print("min/max Pf on [0,5]:",
      learned_target_pdf(sol, d_test[(d_test>=0)&(d_test<=5)]).min(),
      learned_target_pdf(sol, d_test[(d_test>=0)&(d_test<=5)]).max())

"""## 100 random init trials for K = 20"""

def run_random_inits(
    K: int = 20,
    n_trials: int = 100,
    xgrid_m: int = 1001,
    dgrid_m: int = 401,
    max_iter: int = 100,
    tol_opt: float = 1e-9,
    tol_feas: float = 1e-9,
    trust_clip: float = 0.25,
    gamma_scale: float = 1.0,
    beta_scale: float = 10.0,
    seed: int = 42
):
    """
    Run many random initializations for uniform PW-SQP solver.
    Returns dict with history of final objective and constraint violation.
    """
    rng = np.random.default_rng(seed)
    objs, feas_viol = [], []

    for t in range(n_trials):
        # random init: gammas ~ U(-gamma_scale, gamma_scale),
        #              betas  ~ U(-beta_scale, beta_scale)
        init_gammas = rng.uniform(-gamma_scale, gamma_scale, K)
        init_betas  = rng.uniform(-beta_scale, beta_scale, K)

        sol = sqp_solve_piecewise_uniform(
            K=K,
            xgrid_m=xgrid_m,
            dgrid_m=dgrid_m,
            max_iter=max_iter,
            tol_opt=tol_opt,
            tol_feas=tol_feas,
            trust_clip=trust_clip,
            init_gammas=init_gammas,
            init_betas=init_betas
        )

        # final objective and constraint violation
        J_final = sol["history"]["J"][-1]
        feas_final = sol["history"]["feas_res"][-1]

        objs.append(J_final)
        feas_viol.append(feas_final)

    objs = np.array(objs)
    feas_viol = np.array(feas_viol)

    print("=== Monte Carlo SQP runs ===")
    print(f"Trials: {n_trials}, Pieces: {K}")
    print(f"Objective J:   mean={objs.mean():.6f}, std={objs.std():.6f}, min={objs.min():.6f}, max={objs.max():.6f}")
    print(f"Feasibility ∞-norm: mean={feas_viol.mean():.2e}, std={feas_viol.std():.2e}, "
          f"min={feas_viol.min():.2e}, max={feas_viol.max():.2e}")

    return {"objs": objs, "feas_viol": feas_viol}

# Example usage:
if __name__ == "__main__":
    results = run_random_inits(K=20, n_trials=100)

"""## 100 random init trials for K = 10"""

def run_random_inits(
    K: int = 20,
    n_trials: int = 100,
    xgrid_m: int = 1001,
    dgrid_m: int = 401,
    max_iter: int = 100,
    tol_opt: float = 1e-9,
    tol_feas: float = 1e-9,
    trust_clip: float = 0.25,
    gamma_scale: float = 1.0,
    beta_scale: float = 10.0,
    seed: int = 42
):
    """
    Run many random initializations for uniform PW-SQP solver.
    Returns dict with history of final objective and constraint violation.
    """
    rng = np.random.default_rng(seed)
    objs, feas_viol = [], []

    for t in range(n_trials):
        # random init: gammas ~ U(-gamma_scale, gamma_scale),
        #              betas  ~ U(-beta_scale, beta_scale)
        init_gammas = rng.uniform(-gamma_scale, gamma_scale, K)
        init_betas  = rng.uniform(-beta_scale, beta_scale, K)

        sol = sqp_solve_piecewise_uniform(
            K=K,
            xgrid_m=xgrid_m,
            dgrid_m=dgrid_m,
            max_iter=max_iter,
            tol_opt=tol_opt,
            tol_feas=tol_feas,
            trust_clip=trust_clip,
            init_gammas=init_gammas,
            init_betas=init_betas
        )

        # final objective and constraint violation
        J_final = sol["history"]["J"][-1]
        feas_final = sol["history"]["feas_res"][-1]

        objs.append(J_final)
        feas_viol.append(feas_final)

    objs = np.array(objs)
    feas_viol = np.array(feas_viol)

    print("=== Monte Carlo SQP runs ===")
    print(f"Trials: {n_trials}, Pieces: {K}")
    print(f"Objective J:   mean={objs.mean():.6f}, std={objs.std():.6f}, min={objs.min():.6f}, max={objs.max():.6f}")
    print(f"Feasibility ∞-norm: mean={feas_viol.mean():.2e}, std={feas_viol.std():.2e}, "
          f"min={feas_viol.min():.2e}, max={feas_viol.max():.2e}")

    return {"objs": objs, "feas_viol": feas_viol}

# Example usage:
if __name__ == "__main__":
    results = run_random_inits(K=10, n_trials=100)

"""## 100 random init trials for K = 4"""

def run_random_inits(
    K: int = 20,
    n_trials: int = 100,
    xgrid_m: int = 1001,
    dgrid_m: int = 401,
    max_iter: int = 100,
    tol_opt: float = 1e-9,
    tol_feas: float = 1e-9,
    trust_clip: float = 0.25,
    gamma_scale: float = 1.0,
    beta_scale: float = 10.0,
    seed: int = 42
):
    """
    Run many random initializations for uniform PW-SQP solver.
    Returns dict with history of final objective and constraint violation.
    """
    rng = np.random.default_rng(seed)
    objs, feas_viol = [], []

    for t in range(n_trials):
        # random init: gammas ~ U(-gamma_scale, gamma_scale),
        #              betas  ~ U(-beta_scale, beta_scale)
        init_gammas = rng.uniform(-gamma_scale, gamma_scale, K)
        init_betas  = rng.uniform(-beta_scale, beta_scale, K)

        sol = sqp_solve_piecewise_uniform(
            K=K,
            xgrid_m=xgrid_m,
            dgrid_m=dgrid_m,
            max_iter=max_iter,
            tol_opt=tol_opt,
            tol_feas=tol_feas,
            trust_clip=trust_clip,
            init_gammas=init_gammas,
            init_betas=init_betas
        )

        # final objective and constraint violation
        J_final = sol["history"]["J"][-1]
        feas_final = sol["history"]["feas_res"][-1]

        objs.append(J_final)
        feas_viol.append(feas_final)

    objs = np.array(objs)
    feas_viol = np.array(feas_viol)

    print("=== Monte Carlo SQP runs ===")
    print(f"Trials: {n_trials}, Pieces: {K}")
    print(f"Objective J:   mean={objs.mean():.6f}, std={objs.std():.6f}, min={objs.min():.6f}, max={objs.max():.6f}")
    print(f"Feasibility ∞-norm: mean={feas_viol.mean():.2e}, std={feas_viol.std():.2e}, "
          f"min={feas_viol.min():.2e}, max={feas_viol.max():.2e}")

    return {"objs": objs, "feas_viol": feas_viol}

# Example usage:
if __name__ == "__main__":
    results = run_random_inits(K=4, n_trials=100)

"""## Uniform Partition, Histogram Constraint"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =========================
# Measures for Example 1.1
# =========================
PI = np.pi
C = PI / 5.0          # frequency
Z = 20.0 / PI         # exact positive/negative mass on [-10, 10]
Z_INV = 1.0 / Z       # = PI / 20

def rho_signed(x: np.ndarray) -> np.ndarray:
    return np.sin(C * x)

def mu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# ============
# Grids
# ============
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray # shape (m,)
    weights: np.ndarray # trapezoid weights for integrations

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    if m >= 2:
        w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

# ====================================
# Piecewise linear map specification
# ====================================
@dataclass
class PWLinSpec:
    # knots define K pieces on [knots[i], knots[i+1]]
    knots: np.ndarray      # shape (K+1,)
    signs: np.ndarray      # shape (K,), each in {+1, -1}; slope a_i = signs[i]*exp(gamma_i)

def n_pieces(spec: PWLinSpec) -> int:
    return len(spec.knots) - 1

def make_uniform_spec(L: float, R: float, K: int, sign: int = -1) -> PWLinSpec:
    """Uniform K-piece partition on [L, R], with all slopes having the same sign."""
    knots = np.linspace(L, R, K + 1)
    signs = np.full(K, sign, dtype=float)
    return PWLinSpec(knots=knots, signs=signs)

# Evaluate s(x) and piece indices (utility)
def s_eval_piecewise(x: np.ndarray, gammas: np.ndarray, betas: np.ndarray, spec: PWLinSpec) -> np.ndarray:
    K = n_pieces(spec)
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a = spec.signs[idx] * np.exp(gammas[idx])
    b = betas[idx]
    return a * x + b

# ===========================
# Objective  c(x,y)=sqrt(2|x-y|), smoothed
# ===========================
def objective_J_pw(gammas: np.ndarray, betas: np.ndarray,
                   xgrid: GridSpec, spec: PWLinSpec,
                   eps: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Smoothed objective for c(r) = sqrt(2|r|) via
        c_eps(r) = sqrt(2) * (r^2 + eps^2)^(1/4).
    Returns J, grad (size 2K), and a PSD Gauss-Newton-like Hessian.
    """
    K = n_pieces(spec)
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    # piece membership and piece parameters per-sample
    idx = np.clip(np.digitize(x, spec.knots) - 1, 0, K-1)
    a_all = spec.signs * np.exp(gammas)     # (K,)
    a_x   = a_all[idx]
    b_x   = betas[idx]

    # residual r = x - s(x)
    s_x  = a_x * x + b_x
    r    = x - s_x
    q    = r*r + eps*eps                   # r^2 + eps^2

    # c_eps(r) = sqrt(2) * q^(1/4)
    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr = (sqrt(2)/2) * r * q^(-3/4)
    dc_dr = (np.sqrt(2.0)/2.0) * r * np.power(q, -0.75)

    # grad wrt theta on each piece
    grad = np.zeros(2*K)
    for i in range(K):
        msk = (idx == i)
        if not np.any(msk):
            continue
        a_i = a_all[i]
        xi, wi = x[msk], w[msk]
        mui    = mu[msk]
        dci    = dc_dr[msk]

        # ∂s/∂gamma_i = a_i * x, ∂s/∂beta_i = 1
        grad[i]      = -np.sum(mui * dci * (a_i * xi) * wi)
        grad[K + i]  = -np.sum(mui * dci * 1.0        * wi)

    # ----- PSD Hessian (Gauss–Newton style with curvature clipping) -----
    # d2c/dr2 = (sqrt(2)/2) * (q)^(-7/4) * (eps^2 - 0.5*r^2)
    d2c = (np.sqrt(2.0)/2.0) * np.power(q, -1.75) * (eps*eps - 0.5 * r*r)
    w_curv = np.maximum(d2c, 0.0)   # clip to keep PSD

    H = np.zeros((2*K, 2*K))
    for i in range(K):
        msk_i = (idx == i)
        if not np.any(msk_i):
            continue
        a_i = a_all[i]
        xi, wi = x[msk_i], w[msk_i]
        mui    = mu[msk_i]
        wi_curv = w_curv[msk_i]

        # features for this piece: [a_i * x, 1]
        f1 = a_i * xi
        f2 = np.ones_like(xi)

        # weight = mu * w * w_curv
        ww = mui * wi * wi_curv

        # accumulate 2x2 block for (gamma_i, beta_i)
        H[i, i]             += np.sum(ww * f1 * f1)
        H[i, K + i]         += np.sum(ww * f1 * f2)
        H[K + i, i]         += np.sum(ww * f2 * f1)
        H[K + i, K + i]     += np.sum(ww * f2 * f2)

    # mild diagonal regularization for numerical stability
    H += 1e-10 * np.eye(2*K)
    return J, grad, H

# ==========================================================
# PF constraint zeta(y) = P_f(s)(y) - nu(y) on a y-grid
# and analytic Jacobian A = d zeta / d theta
# ==========================================================
def constraint_zeta_pw(gammas: np.ndarray, betas: np.ndarray, dgrid: GridSpec, spec: PWLinSpec) -> Dict[str, np.ndarray]:
    """
    Returns:
      zeta : shape (m_d,)
      A    : shape (m_d, 2K) for theta=[gammas..., betas...]
    Uses formula:
      For piece i with slope a_i = sign_i * exp(gamma_i),
        u_i(y) = (y - beta_i) / a_i = sign_i * exp(-gamma_i) * (y - beta_i)
        contrib_i(y) = mu(u_i) / |a_i| = exp(-gamma_i) * mu(u_i)
      zeta(y) = sum_i contrib_i(y) * 1_{u_i in [x_i, x_{i+1}]} - nu(y)
    Jacobians:
      ∂/∂gamma_i contrib = -exp(-gamma_i) * [mu(u) + u * mu'(u)]
      ∂/∂beta_i  contrib = -sign_i * exp(-2*gamma_i) * mu'(u)
    """
    K = n_pieces(spec)
    y = dgrid.points
    m = dgrid.m
    zeta = np.zeros(m)
    A = np.zeros((m, 2*K))

    for i in range(K):
        sgn = spec.signs[i]
        eg = np.exp(gammas[i])
        emg = np.exp(-gammas[i])      # 1/|a_i|
        a_i = sgn * eg

        # preimage u(y) on this piece
        u = (y - betas[i]) / a_i      # = sgn * exp(-gamma) * (y - beta)
        # only valid if u in the segment
        L, R = spec.knots[i], spec.knots[i+1]
        valid = (u >= L) & (u <= R)
        if not np.any(valid):
            continue

        mu_u  = mu_pdf(u)
        mu_u_p = mu_pdf_prime(u)

        contrib = emg * mu_u
        zeta += contrib * valid

        # Jacobians (only where valid)
        d_contrib_dgamma = -emg * (mu_u + u * mu_u_p)
        d_contrib_dbeta  = -sgn * (emg**2) * mu_u_p

        A[valid, i]     += d_contrib_dgamma[valid]
        A[valid, K + i] += d_contrib_dbeta[valid]

    # subtract target density
    zeta -= nu_pdf(y)
    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian pieces & single SQP step (PW)
# ===========================================
def build_lagrangian_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                        xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J, gJ, HJ = objective_J_pw(gammas, betas, xgrid, spec)
    parts = constraint_zeta_pw(gammas, betas, dgrid, spec)
    zeta, A = parts["zeta"], parts["A"]
    # gradient of L
    g = gJ + A.T @ lam
    # (we keep H = HJ; constraint second-derivatives omitted for stability)
    H = HJ
    return g, H, A, J

def sqp_step_pw(gammas: np.ndarray, betas: np.ndarray, lam: np.ndarray,
                xgrid: GridSpec, dgrid: GridSpec, spec: PWLinSpec,
                reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
    zeta = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

    n = 2 * n_pieces(spec)
    m = dgrid.m

    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    new_gammas = gammas + dtheta[:n_pieces(spec)]
    new_betas  = betas  + dtheta[n_pieces(spec):]
    new_lam = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res

# =====================
# Full solver (PW-SQP) — SAME API + optional uniform partition
# =====================
def sqp_solve_piecewise(
    spec: Optional[PWLinSpec],
    xgrid_m: int = 1001,   # integration grid for objective on x
    dgrid_m: int = 2001,   # PF constraint grid on y
    max_iter: int = 20,
    tol_opt: float = 1e-9,
    tol_feas: float = 1e-9,
    init_gammas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.25,
    *,
    # ---- NEW (optional) uniform partition controls ----
    uniform_K: Optional[int] = None,
    uniform_LR: Tuple[float,float] = (-10.0, 10.0),
    uniform_sign: int = -1
) -> Dict[str, np.ndarray]:
    """
    If `uniform_K` is None: behaves exactly like your original solver and uses `spec`.
    If `uniform_K` is an integer: builds a uniform K-piece spec on `uniform_LR` with slopes' sign = `uniform_sign`,
    and ignores the passed `spec`.
    """
    # Choose spec (original or uniform)
    if uniform_K is not None:
        spec = make_uniform_spec(L=uniform_LR[0], R=uniform_LR[1], K=uniform_K, sign=uniform_sign)
    assert spec is not None, "You must provide a spec or set uniform_K."

    K = n_pieces(spec)
    xgrid = make_uniform_grid(spec.knots[0], spec.knots[-1], xgrid_m)
    dgrid = make_uniform_grid(spec.knots[0], spec.knots[-1], dgrid_m)

    # gammas = np.zeros(K) if init_gammas is None else init_gammas.copy()
    # betas  = np.zeros(K) if init_betas  is None else init_betas.copy()
    gammas = np.random.uniform(-1, 1, K) if init_gammas is None else init_gammas.copy()
    betas  = np.random.uniform(-10, 10, K) if init_betas is None else init_betas.copy()
    lam = np.zeros(dgrid.m)

    # sanity: if user passed init vectors of wrong size (because K changed), resize safely
    if gammas.shape[0] != K:
        gammas = np.zeros(K)
    if betas.shape[0] != K:
        betas = np.zeros(K)

    hist = {"J": [], "opt_res": [], "feas_res": [], "gammas": [], "betas": []}

    for it in range(max_iter):
        g, H, A, Jval = build_lagrangian_pw(gammas, betas, lam, xgrid, dgrid, spec)
        zeta_now = constraint_zeta_pw(gammas, betas, dgrid, spec)["zeta"]

        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["gammas"].append(gammas.copy())
        hist["betas"].append(betas.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_gammas, new_betas, new_lam, dtheta, opt_res, feas_res = sqp_step_pw(
            gammas, betas, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        # simple trust-region clip in infinity norm on the parameter update
        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_gammas = gammas + dtheta[:K]
                new_betas  = betas  + dtheta[K:]

        gammas, betas, lam = new_gammas, new_betas, new_lam

    return {
        "gammas": gammas, "betas": betas,
        "slopes": spec.signs * np.exp(gammas),
        "lambda": lam, "history": hist, "spec": spec,
        "xgrid": xgrid, "dgrid": dgrid
    }

# =============================
# Example usage (uniform K pieces)
# =============================
if __name__ == "__main__":
    # keep API the same; just pass uniform_K to get a uniform partition
    sol = sqp_solve_piecewise(
        spec=None,                 # ignored when uniform_K is set
        xgrid_m=1001, dgrid_m=2001,
        max_iter=20, tol_opt=1e-9, tol_feas=1e-9,
        init_gammas=None, init_betas=None,
        trust_clip=0.25,
        uniform_K=4,               # <-- NEW: 4 uniform pieces on [-10,10]
        uniform_LR=(-10.0, 10.0),
        uniform_sign=-1
    )

    print("Slopes a_i:", sol["slopes"])
    print("Betas  b_i:", sol["betas"])
    J_final, _, _ = objective_J_pw(sol["gammas"], sol["betas"], sol["xgrid"], sol["spec"])
    print("Final opt_res, Final Objective, feas_res:", sol["history"]["opt_res"][-1], J_final, sol["history"]["feas_res"][-1], sol["history"]["J"][-1])

"""### Plot"""

# ==========================
# True solution (Example 1.1)
# ==========================
def true_piece_params_for_example(spec: PWLinSpec):
    """
    Returns (a_true, b_true) per *spec piece* without assuming any special knots.
    We classify each piece by its midpoint m:
        if -9 < m < -1  -> s(x) ~ -x - 10  on that piece
        if  1 < m <  9  -> s(x) ~ -x + 10  on that piece
        else            -> s(x) ~ -x
    """
    K = n_pieces(spec)
    a_true = -np.ones(K)          # slope is -1 everywhere
    b_true = np.zeros(K)

    mids = 0.5 * (spec.knots[:-1] + spec.knots[1:])
    b_true[(mids > -9.0) & (mids < -1.0)] = -10.0
    b_true[(mids >  1.0) & (mids <  9.0)] = +10.0
    return a_true, b_true


def true_map_example(x: np.ndarray) -> np.ndarray:
    """
    Vectorized true mapping s(x) for the example (continuous by parts).
    """
    y = -x.copy()
    y = np.where((-9.0 < x) & (x < -1.0), -x - 10.0, y)
    y = np.where(( 1.0 < x) & (x <  9.0), -x + 10.0, y)
    return y


# ===========================
# Optional: quick plot overlay
# ===========================
from matplotlib.collections import LineCollection

def plot_overlay_solution(sol: Dict, num_pts_per_piece: int = 200):
    """
    Learned map is sampled on ONE global uniform grid over [Ldom, Rdom].
    Learned curve is drawn by spec pieces; true curve is drawn by the actual
    breakpoints [-10,-9],[-9,-1],[-1,1],[1,9],[9,10]. Both use LineCollection
    with tiny endpoint shrink to ensure no vertical connectors.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    spec: PWLinSpec = sol["spec"]
    K = n_pieces(spec)
    a_hat = sol["slopes"]; b_hat = sol["betas"]
    a_true, b_true = true_piece_params_for_example(spec)

    Ldom, Rdom = spec.knots[0], spec.knots[-1]
    eps = 1e-9 * (Rdom - Ldom)

    # global uniform grid
    N = max(2, K * num_pts_per_piece)
    xx_all = np.linspace(Ldom, Rdom, N)

    # ----- learned segments (by spec pieces) -----
    learned_segs = []
    for i in range(K):
        L, R = spec.knots[i], spec.knots[i+1]
        mask = (xx_all > L + (eps if i > 0 else 0.0)) & (xx_all < R - (eps if i < K-1 else 0.0))
        if np.count_nonzero(mask) < 2:
            continue
        xpi = xx_all[mask]
        ypi = a_hat[i] * xpi + b_hat[i]
        learned_segs.append(np.column_stack([xpi, ypi]))

    # ----- true segments (by actual breakpoints, independent of spec) -----
    breaks = [-10.0, -9.0, -1.0, 1.0, 9.0, 10.0]
    true_segs = []
    for j in range(len(breaks)-1):
        L, R = breaks[j], breaks[j+1]
        Lp = L + (eps if j > 0 else 0.0)
        Rp = R - (eps if j < len(breaks)-2 else 0.0)
        mask = (xx_all > Lp) & (xx_all < Rp)
        if np.count_nonzero(mask) < 2:
            continue
        xpi = xx_all[mask]
        # true piecewise rule
        ypi = -xpi
        sel1 = (xpi > -9.0) & (xpi < -1.0)
        ypi[sel1] = -xpi[sel1] - 10.0
        sel2 = (xpi >  1.0) & (xpi <  9.0)
        ypi[sel2] = -xpi[sel2] + 10.0
        true_segs.append(np.column_stack([xpi, ypi]))

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(8, 5))
    if learned_segs:
        ax.add_collection(LineCollection(learned_segs, colors="red", linewidths=2))
    if true_segs:
        ax.add_collection(LineCollection(true_segs, colors="k", linewidths=2, linestyles="--"))

    ax.set_xlim(Ldom, Rdom)
    if learned_segs or true_segs:
        ymin = min(seg[:,1].min() for seg in (learned_segs + true_segs))
        ymax = max(seg[:,1].max() for seg in (learned_segs + true_segs))
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x"); ax.set_ylabel("s(x)")
    ax.set_title("Piecewise transport: learned (solid) vs. true (dashed)")
    import matplotlib.lines as mlines
    ax.legend(handles=[
        mlines.Line2D([], [], color='red', lw=2, label='Learned map'),
        mlines.Line2D([], [], color='k', lw=2, ls='--', label='True map'),
    ], loc="best")
    ax.grid(True, alpha=0.3)
    plt.show()

# K = 40
# init_gammas = np.zeros(K)  # slope ~ -1 because signs=-1 by default
# init_betas  = np.zeros(K)

# solver_kwargs = dict(
#     K=K,
#     xgrid_m=1001,
#     dgrid_m=401,
#     max_iter=50,
#     tol_opt=1e-9,
#     tol_feas=1e-9,
#     init_gammas=init_gammas,
#     init_betas=init_betas,
#     trust_clip=0.25,
# )
# sol = sqp_solve_piecewise_uniform(**solver_kwargs)

# plot comparison
# plot_learned_vs_true(sol)
plot_overlay_solution(sol)

def learned_target_pdf(sol, d: np.ndarray) -> np.ndarray:
    """
    Compute P_f(s)(d) for the learned piecewise-linear map in `sol`.
    Uses the same formula as in the PF constraint but returns only Pf(s).
    """
    spec = sol["spec"]
    gammas, betas = sol["gammas"], sol["betas"]
    K = len(gammas)

    Pf = np.zeros_like(d)
    for i in range(K):
        sgn = spec.signs[i]
        eg  = np.exp(gammas[i])
        emg = np.exp(-gammas[i])        # 1/|a_i|
        a_i = sgn * eg

        u = (d - betas[i]) / a_i        # preimage on piece i
        L, R = spec.knots[i], spec.knots[i+1]
        valid = (u >= L) & (u <= R)
        if not np.any(valid):
            continue

        Pf[valid] += emg * mu_pdf(u[valid])   # Pf(s)(d) = sum_i mu(u_i)/|a_i|
    return Pf


def plot_source_and_targets(sol, num_pts: int = 2001):
    """
    Two-panel figure:
      (1) Source density mu(x)
      (2) Learned Pf(s)(d) vs. true target nu(d)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    L, R = sol["spec"].knots[0], sol["spec"].knots[-1]

    # fine grids
    x = np.linspace(L, R, num_pts)
    d = np.linspace(L, R, num_pts)

    mu_x   = mu_pdf(x)
    nu_d   = nu_pdf(d)
    pf_d   = learned_target_pdf(sol, d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source density
    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title("Source density  $\\mu(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: learned vs. true target
    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label="Learned  $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label="True  $\\nu(d)$")
    ax.set_title("Target densities on $d$")
    ax.set_xlabel("d")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Source vs. Learned Target vs. True Target", y=1.03, fontsize=13)
    fig.tight_layout()
    plt.show()

plot_source_and_targets(sol)

"""## Polynomial"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =========================
# Measures for Example 1.1
# =========================
PI = np.pi
C = PI / 5.0
Z = 20.0 / PI
Z_INV = 1.0 / Z

def rho_signed(x: np.ndarray) -> np.ndarray:
    return np.sin(C * x)

def mu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(s, 0.0)

def nu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * np.maximum(-s, 0.0)

def mu_pdf_prime(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x)
    return Z_INV * C * np.where(s > 0.0, np.cos(C * x), 0.0)

# ============ Grids ============
@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray
    weights: np.ndarray

def make_uniform_grid(L: float, R: float, m: int) -> GridSpec:
    pts = np.linspace(L, R, m)
    w = np.ones_like(pts)
    if m >= 2:
        w[0] = w[-1] = 0.5
    w *= (R - L) / (m - 1)
    return GridSpec(L=L, R=R, m=m, points=pts, weights=w)

# =====================================
# Global Polynomial map s(x) = sum c_k x^k
# =====================================
@dataclass
class PolySpec:
    deg: int                    # degree p (<= 6)
    L: float = -10.0
    R: float =  10.0

def s_eval_poly(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    # c has length p+1, c[0] + c[1] x + ... + c[p] x^p
    # Use Horner for stability
    y = np.zeros_like(x)
    for ck in c[::-1]:
        y = y * x + ck
    return y

def s_prime_poly(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = len(c) - 1
    if p == 0:
        return np.zeros_like(x)
    # derivative coefficients: [c1, 2c2, ..., p c_p]
    coeff = np.array([k * c[k] for k in range(1, p + 1)], dtype=float)
    # Horner derivative evaluation
    y = np.zeros_like(x)
    for ck in coeff[::-1]:
        y = y * x + ck
    return y

def s_second_poly(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = len(c) - 1
    if p <= 1:
        return np.zeros_like(x)
    coeff = np.array([k * (k - 1) * c[k] for k in range(2, p + 1)], dtype=float)
    y = np.zeros_like(x)
    for ck in coeff[::-1]:
        y = y * x + ck
    return y

# =======================================
# Smoothed objective for c(r)=sqrt(2|r|)
# =======================================
def objective_J_poly(c: np.ndarray, xgrid: GridSpec, eps: float = 1e-4
                     ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    c_eps(r) = sqrt(2) * (r^2 + eps^2)^(1/4),  r = x - s(x)
    grad_k = -∫ mu(x) * dc/dr * x^k dx, with ds/dc_k = x^k
    PSD Gauss–Newton-like Hessian with curvature clipping (as before).
    """
    x, w = xgrid.points, xgrid.weights
    mu = mu_pdf(x)

    s_x = s_eval_poly(x, c)
    r = x - s_x
    q = r * r + eps * eps

    # J
    J = np.sum(mu * (np.sqrt(2.0) * np.power(q, 0.25)) * w)

    # dc/dr
    dc_dr = (np.sqrt(2.0) / 2.0) * r * np.power(q, -0.75)

    # Gradient: size p+1
    p1 = len(c)
    grad = np.zeros(p1)
    # features: x^k
    Xpow = np.vstack([x ** k for k in range(p1)])  # shape (p1, m)
    # dJ/dc_k = -∑ mu * dc_dr * x^k * w
    grad = - (Xpow * (mu * dc_dr * w)).sum(axis=1)

    # Hessian (Gauss–Newton clip)
    # d2c/dr2 = (sqrt(2)/2)*(q)^(-7/4)*(eps^2 - 0.5 r^2)
    d2c = (np.sqrt(2.0) / 2.0) * np.power(q, -1.75) * (eps * eps - 0.5 * r * r)
    w_curv = np.maximum(d2c, 0.0)  # keep PSD

    H = np.zeros((p1, p1))
    ww = mu * w * w_curv           # weights per sample
    for i in range(p1):
        fi = Xpow[i]
        for j in range(i, p1):
            fj = Xpow[j]
            hij = np.sum(ww * fi * fj)
            H[i, j] = H[j, i] = hij

    H += 1e-10 * np.eye(p1)
    return J, grad, H

# ==========================================================
# PF constraint zeta(d) = P_f(s)(d) - nu(d)  (poly version)
# ==========================================================
def _poly_roots_for_level(c: np.ndarray, d: float) -> np.ndarray:
    """
    Solve s(u) = d for u (real roots).
    We form polynomial s(u) - d = 0.
    """
    p = len(c) - 1
    coeff = np.array(c, dtype=float)
    coeff[0] -= d
    # numpy.roots expects descending powers
    coeff_desc = coeff[::-1]
    rts = np.roots(coeff_desc)
    # keep real roots (imag part small) and within domain
    rts_real = rts.real[np.isclose(rts.imag, 0.0, atol=1e-10)]
    return rts_real

def constraint_zeta_poly(c: np.ndarray, dgrid: GridSpec,
                         domain: Tuple[float, float] = (-10.0, 10.0),
                         gmin: float = 1e-10
                         ) -> Dict[str, np.ndarray]:
    """
    zeta(d) = sum_{u: s(u)=d, u in [L,R]} mu(u)/|s'(u)| - nu(d)
    A[m, k] = ∂zeta(d_m)/∂c_k using implicit differentiation:
      s(u;c)=d ⇒ du/dc_k = -u^k / s'(u)
      Let g = s'(u), a=|g|, sgn=sign(g), g' = s''(u).
      contrib = mu(u)/a
      ∂contrib/∂c_k = (mu'(u) * du)/a + mu(u) * [ - (1/a^2) * sgn * ∂g ]
      with ∂g = g' * du + k * u^{k-1}
    """
    L, R = domain
    d = dgrid.points
    m = dgrid.m

    zeta = np.zeros(m)
    p1 = len(c)
    A = np.zeros((m, p1))

    for t in range(m):
        dt = float(d[t])
        roots = _poly_roots_for_level(c, dt)
        # filter roots in domain
        roots = roots[(roots >= L - 1e-12) & (roots <= R + 1e-12)]
        if roots.size == 0:
            zeta[t] -= nu_pdf(np.array([dt]))[0]
            continue

        # accumulate contributions per root
        for u in roots:
            u = float(u)
            mu_u = mu_pdf(np.array([u]))[0]
            mu_p = mu_pdf_prime(np.array([u]))[0]
            g = s_prime_poly(np.array([u]), c)[0]
            g = np.clip(g, -1e308, 1e308)  # numeric safety
            # avoid division by zero in |g|
            if np.abs(g) < gmin:
                # skip near-critical roots to avoid instability
                # (optional: add tiny Tikhonov term in denominator)
                continue
            a = np.abs(g)
            sgn = 1.0 if g >= 0.0 else -1.0
            g2 = s_second_poly(np.array([u]), c)[0]

            zeta[t] += mu_u / a

            # Jacobian wrt c_k
            for k in range(p1):
                # du/dc_k = -u^k / g
                u_pow_k = (u ** k)
                du = - u_pow_k / g

                # ∂g = g' * du + k * u^{k-1}
                if k == 0:
                    dg = g2 * du
                else:
                    dg = g2 * du + k * (u ** (k - 1))

                # d(contrib) = (mu'(u) * du)/a + mu(u) * [-(1/a^2) * sgn * dg]
                Jk = (mu_p * du) / a - mu_u * (sgn * dg) / (a * a)
                A[t, k] += Jk

        # subtract target density
        zeta[t] -= nu_pdf(np.array([dt]))[0]

    return {"zeta": zeta, "A": A}

# ===========================================
# Lagrangian pieces & single SQP step (Poly)
# ===========================================
def build_lagrangian_poly(c: np.ndarray, lam: np.ndarray,
                          xgrid: GridSpec, dgrid: GridSpec, spec: PolySpec
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J, gJ, HJ = objective_J_poly(c, xgrid)
    parts = constraint_zeta_poly(c, dgrid, domain=(spec.L, spec.R))
    zeta, A = parts["zeta"], parts["A"]
    g = gJ + A.T @ lam
    H = HJ  # keep GN Hessian
    return g, H, A, J

def sqp_step_poly(c: np.ndarray, lam: np.ndarray,
                  xgrid: GridSpec, dgrid: GridSpec, spec: PolySpec,
                  reg_H: float = 1e-10, reg_KKT_primal: float = 0.0, reg_KKT_dual: float = 1e-8):
    g, H, A, Jval = build_lagrangian_poly(c, lam, xgrid, dgrid, spec)
    zeta = constraint_zeta_poly(c, dgrid, domain=(spec.L, spec.R))["zeta"]

    n = len(c)
    m = dgrid.m

    H_reg = H + reg_H * np.eye(n)
    KKT = np.block([[H_reg + reg_KKT_primal * np.eye(n), A.T],
                    [A, -reg_KKT_dual * np.eye(m)]])
    rhs = -np.concatenate([g, zeta])
    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    dtheta = sol[:n]
    w = sol[n:]

    new_c = c + dtheta
    new_lam = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return new_c, new_lam, dtheta, opt_res, feas_res

# =====================
# Full solver (Poly-SQP)
# =====================
def sqp_solve_poly(
    spec: PolySpec,
    xgrid_m: int = 1001,
    dgrid_m: int = 401,
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    init_c: Optional[np.ndarray] = None,
    trust_clip: Optional[float] = 0.5,
) -> Dict[str, np.ndarray]:

    xgrid = make_uniform_grid(spec.L, spec.R, xgrid_m)
    dgrid = make_uniform_grid(spec.L, spec.R, dgrid_m)

    p1 = spec.deg + 1
    c = np.zeros(p1) if init_c is None else init_c.astype(float).copy()
    lam = np.zeros(dgrid.m)

    hist = {"J": [], "opt_res": [], "feas_res": [], "c": []}

    for it in range(max_iter):
        g, H, A, Jval = build_lagrangian_poly(c, lam, xgrid, dgrid, spec)
        zeta_now = constraint_zeta_poly(c, dgrid, domain=(spec.L, spec.R))["zeta"]

        hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g, np.inf))
        hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))
        hist["c"].append(c.copy())

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        new_c, new_lam, dtheta, opt_res, feas_res = sqp_step_poly(
            c, lam, xgrid, dgrid, spec,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        # simple trust-region clip
        if trust_clip is not None:
            step_norm = np.linalg.norm(dtheta, np.inf)
            if step_norm > trust_clip:
                dtheta *= (trust_clip / step_norm)
                new_c = c + dtheta

        c, lam = new_c, new_lam

    return {
        "c": c, "lambda": lam, "history": hist, "spec": spec,
        "xgrid": xgrid, "dgrid": dgrid
    }

# =============================
# Example usage (degree ≤ 6)
# =============================
if __name__ == "__main__":
    # Heuristic init for McCann-like odd reflection:
    # s(x) ≈ -x + a3 x^3 + a5 x^5  (odd polynomial)
    # Start with linear reflection, small higher-order terms.
    deg = 6
    init_c = np.zeros(deg + 1)
    init_c[1] = -1.0   # linear term: -x
    # small odd terms
    init_c[3] = 0.0
    init_c[5] = 0.0

    spec = PolySpec(deg=deg, L=-10.0, R=10.0)

    sol = sqp_solve_poly(
        spec,
        xgrid_m=1001, dgrid_m=401,
        max_iter=50, tol_opt=1e-9, tol_feas=1e-9,
        init_c=init_c,
        trust_clip=0.25
    )

    print("Coefficients c_k (k=0..p):", sol["c"])
    print("Final opt_res, feas_res:", sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])

import numpy as np
import matplotlib.pyplot as plt

# true analytic McCann map
def true_map_example(x: np.ndarray) -> np.ndarray:
    y = -x.copy()
    y = np.where((-9.0 < x) & (x < -1.0), -x - 10.0, y)
    y = np.where(( 1.0 < x) & (x <  9.0), -x + 10.0, y)
    return y

# safe polynomial evaluator (uses your s_eval_poly if it exists; otherwise Horner here)
def _s_eval_poly_safe(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    try:
        return s_eval_poly(x, c)  # use your existing function if defined
    except NameError:
        y = np.zeros_like(x, dtype=float)
        for ck in c[::-1]:
            y = y * x + ck
        return y

def plot_poly_vs_true(sol_poly: dict, num_pts: int = 2001, save_path: str | None = None):
    """Overlay polynomial SQP solution vs. analytic McCann map."""
    spec = sol_poly["spec"]
    c = sol_poly["c"]
    Ldom, Rdom = float(spec.L), float(spec.R)

    x = np.linspace(Ldom, Rdom, num_pts)
    y_poly = _s_eval_poly_safe(x, c)
    y_true = true_map_example(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_poly, label="Polynomial (learned)", linewidth=2)
    ax.plot(x, y_true, "--", label="True map", linewidth=2)
    ax.set_xlim(Ldom, Rdom)
    ax.set_xlabel("x"); ax.set_ylabel("s(x)")
    ax.set_title("Polynomial transport: learned vs. true")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_poly_error(sol_poly: dict, num_pts: int = 2001, save_path: str | None = None):
    """Absolute error |s_learned(x) - s_true(x)|."""
    spec = sol_poly["spec"]
    c = sol_poly["c"]
    Ldom, Rdom = float(spec.L), float(spec.R)

    x = np.linspace(Ldom, Rdom, num_pts)
    err = np.abs(_s_eval_poly_safe(x, c) - true_map_example(x))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, err, linewidth=1.8)
    ax.set_xlim(Ldom, Rdom)
    ax.set_xlabel("x"); ax.set_ylabel("abs error")
    ax.set_title("Pointwise error: |s_poly(x) - s_true(x)|")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

plot_poly_vs_true(sol)                 # overlay
plot_poly_error(sol)                   # error curve (optional)
# plot_poly_vs_true(sol, save_path="poly_vs_true.png")
# plot_poly_error(sol, save_path="poly_err.png")

import numpy as np
import matplotlib.pyplot as plt

# --- true densities (same as in your notebook) ---
PI = np.pi; C = PI/5.0; Z = 20.0/PI; Z_INV = 1.0/Z
def mu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x); return Z_INV * np.maximum(s, 0.0)
def nu_pdf(x: np.ndarray) -> np.ndarray:
    s = np.sin(C * x); return Z_INV * np.maximum(-s, 0.0)

# --- safe polynomial evaluators (use your existing if present) ---
def _s_eval_poly_safe(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    try:
        return s_eval_poly(x, c)   # your function (if already defined)
    except NameError:
        y = np.zeros_like(x, dtype=float)
        for ck in c[::-1]:
            y = y * x + ck
        return y

def _s_prime_poly_safe(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    try:
        return s_prime_poly(x, c)  # your function (if already defined)
    except NameError:
        p = len(c) - 1
        if p <= 0: return np.zeros_like(x, dtype=float)
        coeff = np.array([k * c[k] for k in range(1, p + 1)], dtype=float)
        y = np.zeros_like(x, dtype=float)
        for ck in coeff[::-1]:
            y = y * x + ck
        return y

def _poly_roots_for_level(c: np.ndarray, d: float) -> np.ndarray:
    """Real roots of s(u)=d (uses numpy.roots)."""
    coeff = np.array(c, dtype=float)
    coeff[0] -= d
    rts = np.roots(coeff[::-1])          # expects descending powers
    rts_real = rts.real[np.isclose(rts.imag, 0.0, atol=1e-10)]
    return rts_real

# -----------------------------------------------------------
# Learned target density for the *polynomial* transport map
# -----------------------------------------------------------
def learned_target_pdf_poly(sol_poly: dict, d: np.ndarray, gmin: float = 1e-10) -> np.ndarray:
    """
    Compute P_f(s)(d) = sum_{u: s(u)=d, u in [L,R]} mu(u)/|s'(u)|
    for the polynomial solution 'sol_poly' from sqp_solve_poly.
    """
    spec = sol_poly["spec"]
    c = sol_poly["c"]
    L, R = float(spec.L), float(spec.R)

    d = np.asarray(d, dtype=float)
    Pf = np.zeros_like(d)

    for i, di in enumerate(d):
        roots = _poly_roots_for_level(c, float(di))
        # keep real roots within domain
        roots = roots[(roots >= L - 1e-12) & (roots <= R + 1e-12)]
        if roots.size == 0:
            continue

        # accumulate contributions
        for u in roots:
            g = float(_s_prime_poly_safe(np.array([u]), c)[0])
            if abs(g) < gmin:
                # near-critical point; skip to avoid blow-up
                # (optional: replace with 1/sqrt(g^2 + tau^2))
                continue
            Pf[i] += mu_pdf(np.array([u]))[0] / abs(g)

    return Pf

# -----------------------------------------------------------
# Two-panel figure: source μ(x) and targets Pf(s)(d) vs ν(d)
# -----------------------------------------------------------
def plot_source_and_targets_poly(sol_poly: dict, num_pts: int = 2001, save_path: str | None = None):
    spec = sol_poly["spec"]
    L, R = float(spec.L), float(spec.R)

    x = np.linspace(L, R, num_pts)
    d = np.linspace(L, R, num_pts)

    mu_x = mu_pdf(x)
    nu_d = nu_pdf(d)
    pf_d = learned_target_pdf_poly(sol_poly, d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # Left: source density
    ax = axes[0]
    ax.plot(x, mu_x, lw=2)
    ax.set_title(r"Source density  $\mu(x)$")
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Right: learned vs. true target
    ax = axes[1]
    ax.plot(d, pf_d, lw=2, label=r"Learned  $P_f(s)(d)$")
    ax.plot(d, nu_d, "k--", lw=2, label=r"True  $\nu(d)$")
    ax.set_title(r"Target densities on $d$")
    ax.set_xlabel("d"); ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Source vs. Learned Target vs. True Target", y=1.02, fontsize=13)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

# sol = sqp_solve_poly(...)
plot_source_and_targets_poly(sol)
# Or just get the learned target density array:
# d = np.linspace(sol["spec"].L, sol["spec"].R, 2001)
# pf = learned_target_pdf_poly(sol, d)

d = np.linspace(sol["spec"].L, sol["spec"].R, 2001)
err_L1 = np.trapz(np.abs(learned_target_pdf_poly(sol, d) - nu_pdf(d)), d)
print("L1 error on target densities:", err_L1)

