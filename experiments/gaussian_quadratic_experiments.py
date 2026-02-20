# -*- coding: utf-8 -*-
"""Gaussian -> Gaussian (Quadratic) notebook section.

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


"""# Quadratic Solution Space

## Code
"""

# SQP for 1D Gaussian OT with quadratic map s(x)=kappa x^2 + alpha x + beta,
# keeping alpha = exp(gamma) for positivity of the linear term.
#
# Changes vs your linear version:
# - Variables are now (kappa, gamma, beta)
# - Objective J uses Gaussian moments up to order 4; grad/Hess are analytic.
# - Constraint uses quadratic inverse u = s^{-1}(d) and s'(u); Jacobians/Hessians
#   of zeta are computed by finite differences for robustness.
#
# Author: (extended from user's original)

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable
import numpy as np


# ==========================
# Module 0: Problem Params
# ==========================

@dataclass
class OTParams:
    mu0: float      # source mean
    s0: float       # source std (sigma_0)
    mu1: float      # target mean
    s1: float       # target std (sigma_1)

@dataclass
class GridSpec:
    L: float
    R: float
    m: int
    points: np.ndarray  # shape (m,)


# ===========================================
# Module 1: Densities & 95% mass domain
# ===========================================

def normal_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

def domain_95(params: OTParams, k: float = 1.96) -> Tuple[float, float]:
    L = min(params.mu0 - k * params.s0, params.mu1 - k * params.s1)
    R = max(params.mu0 + k * params.s0, params.mu1 + k * params.s1)
    return L, R

def make_grid(params: OTParams, m: int, L: Optional[float] = None, R: Optional[float] = None) -> GridSpec:
    if L is None or R is None:
        L2, R2 = domain_95(params)
        if L is None: L = L2
        if R is None: R = R2
    pts = np.linspace(L, R, m)
    return GridSpec(L=L, R=R, m=m, points=pts)


# =====================================================
# Module 2: Objective J in (kappa, gamma, beta)
# =====================================================

def gaussian_raw_moments(mu: float, s: float):
    EX  = mu
    EX2 = s**2 + mu**2
    EX3 = mu**3 + 3*mu*s**2
    EX4 = 3*s**4 + 6*mu**2*s**2 + mu**4
    return EX, EX2, EX3, EX4

def objective_J_kgb(kappa: float, gamma: float, beta: float, params: OTParams) -> float:
    """
    J = 1/2 * E[(X - s(X))^2], with s(X) = kappa X^2 + alpha X + beta, alpha = exp(gamma).
    Expanded using raw moments up to order 4.
    """
    alpha = np.exp(gamma)
    EX, EX2, EX3, EX4 = gaussian_raw_moments(params.mu0, params.s0)

    # r = X - (kappa X^2 + alpha X + beta) = -kappa X^2 + (1-alpha)X - beta
    # E[r^2] = kappa^2 E[X^4] + (1-alpha)^2 E[X^2] + beta^2
    #          -2 kappa (1-alpha) E[X^3] + 2 kappa beta E[X^2] - 2 (1-alpha) beta E[X]
    Er2 = (kappa**2)*EX4 + (1-alpha)**2 * EX2 + beta**2 \
          - 2*kappa*(1-alpha)*EX3 + 2*kappa*beta*EX2 - 2*(1-alpha)*beta*EX
    return 0.5 * Er2

def grad_hess_J_kgb(kappa: float, gamma: float, beta: float, params: OTParams) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    kgb in short for kappa, gamma and beta.
    Analytic grad and Hessian of J wrt (kappa, gamma, beta).
    Uses chain rule through alpha = exp(gamma).
    """
    mu0, s0 = params.mu0, params.s0
    EX, EX2, EX3, EX4 = gaussian_raw_moments(mu0, s0)
    alpha = np.exp(gamma)

    # First: partials wrt (kappa, alpha, beta)
    dJ_dkappa = kappa*EX4 - (1 - alpha)*EX3 + beta*EX2
    dJ_dalpha = -(1 - alpha)*EX2 + kappa*EX3 + beta*EX   # (alpha coordinate)
    dJ_dbeta  = beta - (1 - alpha)*EX + kappa*EX2

    # Chain to gamma
    dJ_dgamma = alpha * dJ_dalpha

    grad = np.array([dJ_dkappa, dJ_dgamma, dJ_dbeta])

    # Hessian in (kappa, alpha, beta)
    d2J_dkappa2 = EX4
    d2J_dalpha2 = EX2
    d2J_dbeta2  = 1.0

    d2J_dkappa_dalpha = EX3
    d2J_dkappa_dbeta  = EX2
    d2J_dalpha_dbeta  = EX

    # Convert to (kappa, gamma, beta):
    # gamma-gamma: alpha^2 * d2J/dalpha^2 + alpha * dJ/dalpha
    d2J_dgamma2 = (alpha**2)*d2J_dalpha2 + alpha*dJ_dalpha
    # kappa-gamma: alpha * d2J/dkappa d alpha
    d2J_dkappa_dgamma = alpha * d2J_dkappa_dalpha
    # beta-gamma: alpha * d2J/dalpha d beta
    d2J_dbeta_dgamma  = alpha * d2J_dalpha_dbeta

    Hess = np.array([
        [d2J_dkappa2,        d2J_dkappa_dgamma, d2J_dkappa_dbeta],
        [d2J_dkappa_dgamma,  d2J_dgamma2,       d2J_dbeta_dgamma],
        [d2J_dkappa_dbeta,   d2J_dbeta_dgamma,  d2J_dbeta2      ],
    ])

    Jval = objective_J_kgb(kappa, gamma, beta, params)
    return grad, Hess, Jval


# ==================================================================
# Module 3: Constraint zeta (pushforward density) on a grid
#           and its Jacobian/Hessian in (kappa, gamma, beta)
# ==================================================================

def invert_quadratic_for_u(d: np.ndarray, kappa: float, alpha: float, beta: float, eps_kappa: float = 1e-10) -> np.ndarray:
    """
    Solve u from kappa u^2 + alpha u + beta = d (vectorized).
    Choose the root continuous in kappa -> 0 that matches affine preimage.
    If |kappa| < eps_kappa, return affine preimage u = (d - beta)/alpha.
    """
    if abs(kappa) < eps_kappa:
        return (d - beta) / alpha

    # Quadratic formula: u = (-alpha ± sqrt(alpha^2 - 4*kappa*(beta - d))) / (2*kappa)
    # We choose the branch that tends to (d - beta)/alpha as kappa -> 0.
    disc = alpha**2 - 4.0*kappa*(beta - d)
    # Guard small negatives due to roundoff
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)

    u_plus  = (-alpha + sqrt_disc) / (2.0*kappa)
    u_minus = (-alpha - sqrt_disc) / (2.0*kappa)

    # Evaluate which branch is closer to affine preimage
    u_aff = (d - beta) / alpha
    choose_plus = np.abs(u_plus - u_aff) <= np.abs(u_minus - u_aff)
    u = np.where(choose_plus, u_plus, u_minus)
    return u

def constraint_zeta_kgb(
    kappa: float,
    gamma: float,
    beta: float,
    grid: GridSpec,
    params: OTParams
) -> np.ndarray:
    """
    Compute zeta(d_j) = f(u_j)/|s'(u_j)| - g(d_j), where u_j = s^{-1}(d_j).
    s(x)=kappa x^2 + alpha x + beta, alpha=exp(gamma).
    """
    mu0, s0 = params.mu0, params.s0
    mu1, s1 = params.mu1, params.s1
    d = grid.points
    alpha = np.exp(gamma)

    u = invert_quadratic_for_u(d, kappa, alpha, beta)
    f_u = normal_pdf(u, mu0, s0)
    s_prime = 2.0*kappa*u + alpha
    # Avoid divide-by-zero if derivative is too small; mild floor for stability
    s_prime = np.where(np.abs(s_prime) < 1e-12, np.sign(s_prime)*1e-12 + (s_prime==0)*1e-12, s_prime)

    g_d = normal_pdf(d, mu1, s1)
    zeta = f_u / np.abs(s_prime) - g_d
    return zeta

def finite_diff_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Central-difference Jacobian of vector-valued fun: R^n -> R^m
    Returns A (m x n).
    """
    m = len(fun(x))
    n = len(x)
    A = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        A[:, i] = (fun(x + dx) - fun(x - dx)) / (2*eps)
    return A

def finite_diff_hessians_of_components(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5):
    """
    Central-difference Hessian for each scalar component of a vector function.
    Returns a list Hlist where Hlist[j] is (n x n).
    """
    y = fun(x)
    m = len(y)
    n = len(x)
    Hlist = []
    for j in range(m):
        def gj(z):
            return fun(z)[j]
        H = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                ea = np.zeros(n); ea[a] = eps
                eb = np.zeros(n); eb[b] = eps
                H[a, b] = (gj(x + ea + eb) - gj(x + ea - eb) - gj(x - ea + eb) + gj(x - ea - eb)) / (4*eps*eps)
        Hlist.append(H)
    return Hlist

def constraint_parts_kgb(
    kappa: float,
    gamma: float,
    beta: float,
    grid: GridSpec,
    params: OTParams,
    compute_hessian: bool = False
) -> Dict[str, np.ndarray]:
    """
    Return {"zeta": (m,), "A": (m,3), optionally "H": list of (3x3) Hessians for each zeta_j}.
    Jacobian/Hessians via finite differences for robustness with quadratic inverse.
    """
    def z_fun(x):
        return constraint_zeta_kgb(x[0], x[1], x[2], grid, params)

    x = np.array([kappa, gamma, beta], dtype=float)
    zeta = z_fun(x)
    A = finite_diff_jacobian(z_fun, x, eps=1e-6)

    out = {"zeta": zeta, "A": A}
    if compute_hessian:
        Hlist = finite_diff_hessians_of_components(z_fun, x, eps=1e-5)
        out["H"] = Hlist
    return out


# ===============================================================
# Module 4: Lagrangian terms & one SQP step (3 vars now)
# ===============================================================

def build_lagrangian_terms(
    kappa: float,
    gamma: float,
    beta: float,
    lam: np.ndarray,
    grid: GridSpec,
    params: OTParams,
    use_exact_constraint_hess: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build grad g (3,), Hessian H(3,3), Jacobian A(m,3) of the Lagrangian and J value.
    """
    # Objective pieces
    gJ, HJ, Jval = grad_hess_J_kgb(kappa, gamma, beta, params)

    # Constraints
    parts = constraint_parts_kgb(kappa, gamma, beta, grid, params, compute_hessian=use_exact_constraint_hess)
    zeta, A = parts["zeta"], parts["A"]

    # Lagrangian gradient
    g = gJ + A.T @ lam

    # Lagrangian Hessian
    if use_exact_constraint_hess:
        H = HJ.copy()
        for j in range(grid.m):
            H += lam[j] * parts["H"][j]
    else:
        H = HJ

    return g, H, A, Jval

def sqp_step_quadratic(
    kappa: float,
    gamma: float,
    beta: float,
    lam: np.ndarray,
    grid: GridSpec,
    params: OTParams,
    use_exact_constraint_hess: bool = False,
    reg_H: float = 1e-10,
    reg_KKT_primal: float = 0.0,
    reg_KKT_dual: float = 1e-8
):
    """
    One SQP step (no line-search); 3 primal vars (kappa, gamma, beta) and m duals.
    """
    g, H, A, Jval = build_lagrangian_terms(
        kappa, gamma, beta, lam, grid, params, use_exact_constraint_hess=use_exact_constraint_hess
    )
    zeta = constraint_parts_kgb(kappa, gamma, beta, grid, params, compute_hessian=False)["zeta"]

    m = grid.m
    H_reg = H + reg_H * np.eye(3)
    K = np.block([
        [H_reg + reg_KKT_primal * np.eye(3), A.T],
        [A, -reg_KKT_dual * np.eye(m)]
    ])
    rhs = -np.concatenate([g, zeta])

    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    d = sol[:3]
    w = sol[3:]

    kappa_new = kappa + d[0]
    gamma_new = gamma + d[1]
    beta_new  = beta  + d[2]
    lam_new   = lam + w

    opt_res = np.linalg.norm(g, ord=np.inf)
    feas_res = np.linalg.norm(zeta, ord=np.inf)
    return np.array([kappa_new, gamma_new, beta_new]), lam_new, d, opt_res, feas_res


# =====================================
# Module 5: Full SQP driver (3D)
# =====================================

def sqp_solve_quadratic(
    params: OTParams,
    m: int = 21,
    max_iter: int = 50,
    tol_opt: float = 1e-8,
    tol_feas: float = 1e-8,
    use_exact_constraint_hess: bool = False,
    init_kappa: float = 0.0,
    init_gamma: float = 0.0,
    init_beta: float = 0.0,
    L: Optional[float] = None,
    R: Optional[float] = None,
    trust_clip: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    grid = make_grid(params, m=m, L=L, R=R)
    kappa, gamma, beta = init_kappa, init_gamma, init_beta
    lam = np.zeros(grid.m)

    hist = {"kappa": [], "gamma": [], "beta": [], "alpha": [], "J": [], "opt_res": [], "feas_res": []}

    for it in range(max_iter):
        alpha = np.exp(gamma)
        _, _, Jval = grad_hess_J_kgb(kappa, gamma, beta, params)
        zeta_now = constraint_parts_kgb(kappa, gamma, beta, grid, params)["zeta"]
        g_now, H_now, A_now, _ = build_lagrangian_terms(kappa, gamma, beta, lam, grid, params, use_exact_constraint_hess)

        hist["kappa"].append(kappa); hist["gamma"].append(gamma); hist["beta"].append(beta)
        hist["alpha"].append(alpha);  hist["J"].append(Jval)
        hist["opt_res"].append(np.linalg.norm(g_now, np.inf)); hist["feas_res"].append(np.linalg.norm(zeta_now, np.inf))

        if hist["opt_res"][-1] <= tol_opt and hist["feas_res"][-1] <= tol_feas:
            break

        x_new, lam_new, d, opt_res, feas_res = sqp_step_quadratic(
            kappa, gamma, beta, lam, grid, params,
            use_exact_constraint_hess=use_exact_constraint_hess,
            reg_H=1e-10, reg_KKT_primal=0.0, reg_KKT_dual=1e-8
        )

        if trust_clip is not None:
            step_norm_inf = np.linalg.norm(d, np.inf)
            if step_norm_inf > trust_clip:
                d = d * (trust_clip / step_norm_inf)
                x_new = np.array([kappa, gamma, beta]) + d

        kappa, gamma, beta = x_new
        lam = lam_new

    return {
        "kappa": kappa, "gamma": gamma, "beta": beta, "alpha": np.exp(gamma),
        "lambda": lam, "grid": grid, "history": hist
    }


# ==================================================
# Module 6: Finite-difference helpers (kept public)
# ==================================================

def finite_diff_grad(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        g[i] = (fun(x + dx) - fun(x - dx)) / (2 * eps)
    return g

def finite_diff_hess(fun, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            H[i, j] = (
                fun(x + ei + ej) - fun(x + ei - ej)
                - fun(x - ei + ej) + fun(x - ei - ej)
            ) / (4 * eps * eps)
    return H


# =====================================
# Demo / Self-tests (safe to run)
# =====================================

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Test parameters: nontrivial Gaussian-to-Gaussian
    params = OTParams(mu0=-1.4, s0=2.2, mu1=0.6, s1=1.25)

    # Ground-truth OT map (affine) for quadratic cost:
    a_star = params.s1 / params.s0
    b_star = params.mu1 - a_star * params.mu0
    print("=== Ground truth (affine OT) ===")
    print(f"alpha* = {a_star:.6f}, beta* = {b_star:.6f} (kappa*=0)\n")

    # Check objective grad/Hess at origin in (kappa, gamma, beta)
    k0, g0, b0 = 0.0, 0.0, 0.0
    gradJ, HJ, Jval = grad_hess_J_kgb(k0, g0, b0, params)
    print("Objective J at (kappa=0, gamma=0, beta=0)")
    print("J =", Jval)
    print("grad J =", gradJ)
    print("Hess J =\n", HJ, "\n")

    # FD check
    def fun_J_kgb(x):
        return objective_J_kgb(x[0], x[1], x[2], params)
    x0 = np.array([k0, g0, b0])
    g_fd = finite_diff_grad(fun_J_kgb, x0)
    H_fd = finite_diff_hess(fun_J_kgb, x0)
    print("FD grad J @ (0,0,0):", g_fd)
    print("FD Hess J @ (0,0,0):\n", H_fd, "\n")

    # Constraint shapes and FD Jacobian/Hess sanity
    grid = make_grid(params, m=17)
    parts = constraint_parts_kgb(k0, g0, b0, grid, params, compute_hessian=True)
    print("Constraint zeta at (kappa=0, gamma=0, beta=0) -- shapes:")
    print("zeta shape:", parts["zeta"].shape, "A shape:", parts["A"].shape, "H list len:", len(parts["H"]), "\n")

    # Run SQP (no line search)
    print("=== Running SQP (no line search) with quadratic map ===")
    sol = sqp_solve_quadratic(
        params,
        m=201,
        max_iter=100,
        tol_opt=1e-8,
        tol_feas=1e-8,
        use_exact_constraint_hess=False,   # (numerical Hessians for zeta are already heavy)
        init_kappa=0.0,                    # start at affine
        init_gamma=0.0,
        init_beta=0.0,
        trust_clip=0.5
    )
    print("Solved (kappa, gamma, beta):", sol["kappa"], sol["gamma"], sol["beta"])
    print("Solved (alpha):", sol["alpha"])
    print("History last opt_res, feas_res:", sol["history"]["opt_res"][-1], sol["history"]["feas_res"][-1])
    print("\nCompare to analytic OT (affine): alpha*={:.6f}, beta*={:.6f}".format(a_star, b_star))

"""## Code Explnation

In summary, the pushforward density is computed by evaluating the source density at the preimage(s) and dividing by the absolute value of the derivative of the map at those points, as per the change of variables formula. The `constraint_zeta_kgb` function then uses this to calculate the residual $\zeta$ by subtracting the target density.

5.  **Form the Constraint (`zeta`):** The constraint $\zeta(d)$ is defined as the difference between the pushforward density at `d` and the target density $g(d)$ at `d$:
    $$ \zeta(d) = P_f(s)(d) - g(d) $$
    The target density $g(d)$ is computed using `normal_pdf(d, mu1, s1)`. So the code calculates `zeta = (1.0 / alpha) * f_u - g_d`, which simplifies from `f_u / np.abs(s_prime) - g_d` when the map is strictly monotone and $s'(u) = \alpha > 0$. In the general quadratic case with variable $\kappa$, the formula is `f_u / np.abs(s_prime) - g_d`. The code actually uses `f_u / alpha` which seems to assume $\kappa=0$ or $\alpha>0$ dominates, which aligns with the SQP aiming for the affine solution.

4.  **Apply Change of Variables Formula:** For each point `d` on the grid, the contribution to the pushforward density from its preimage `u` is calculated as `f_u / np.abs(s_prime)`. Since `invert_quadratic_for_u` is designed to return a single relevant preimage in this context, the sum in the general formula simplifies to this single term for each `d`.

3.  **Compute Derivative of the Map (`s_prime`):** The derivative of the quadratic map is $s'(x) = 2\kappa x + \alpha$. This is calculated at the preimage point(s) `u` as `s_prime = 2.0*kappa*u + alpha`.

2.  **Compute Source Density (`normal_pdf`):** It calculates the density of the source Gaussian $f(u)$ at the preimage point(s) `u` using the `normal_pdf` function with the source parameters (`params.mu0`, `params.s0`).

Here's how the `constraint_zeta_kgb` function implements this for the quadratic map $s(x) = \kappa x^2 + \alpha x + \beta$:

1.  **Find Preimages (`invert_quadratic_for_u`):** For each point `d` on the target grid (`grid.points`), the function first finds the preimage `u` such that `s(u) = d` using the `invert_quadratic_for_u` function. For a quadratic map, there can be up to two real preimages for a given `d`. This function specifically selects the root that is relevant in this context (the one continuous as kappa goes to zero).

The pushforward density, denoted $P_f(s)(d)$, represents the density of the source distribution $f(x)$ after applying the transport map $s(x)$. In this code, the pushforward density is computed implicitly within the `constraint_zeta_kgb` function in cell `dQA45oG_KnKW`.

The core idea comes from the change of variables formula for probability densities. For a 1D map $s(x)$, the pushforward density $P_f(s)$ evaluated at a point $d$ is given by:
$$ P_f(s)(d) = \sum_{u: s(u) = d} \frac{f(u)}{|s'(u)|} $$
where the sum is over all preimages $u$ such that $s(u) = d$, and $s'(u)$ is the derivative of the map $s(x)$ evaluated at $u$.

In the `constraint_parts_kgb` function within cell `dQA45oG_KnKW`, the gradient of the constraint $\zeta$ with respect to the parameters $(\kappa, \gamma, \beta)$ is computed using **finite differences**.

Here's how it works:

1.  The function defines a helper function `z_fun` that takes a single NumPy array representing the parameters `[kappa, gamma, beta]` and returns the vector of constraint residuals `zeta` on the grid.
2.  It then calls the `finite_diff_jacobian` function, passing `z_fun` and the current parameter values `[kappa, gamma, beta]`.
3.  The `finite_diff_jacobian` function estimates the Jacobian matrix (which is the collection of gradients of each constraint component) by evaluating `z_fun` at points slightly perturbed around the current parameter values. Specifically, it uses a central difference method: for each parameter, it perturbs it by a small `eps` in both positive and negative directions, evaluates `z_fun` at these perturbed points, and uses the difference in the results divided by `2*eps` to approximate the partial derivative.

In the code, an `eps` value of `1e-6` is used for this finite difference gradient calculation.

This approach is used here because the analytic derivatives of the constraint $\zeta$ with respect to $(\kappa, \gamma, \beta)$ are complex due to the inverse quadratic function `invert_quadratic_for_u`. Using finite differences provides a robust way to get these derivatives numerically, even if it can be computationally more expensive than analytic differentiation for larger numbers of parameters.

The pushforward density, denoted $P_f(s)(d)$, represents the density of the source distribution $f(x)$ after applying the transport map $s(x)$. In this code, the pushforward density is computed implicitly within the `constraint_zeta_kgb` function in cell `dQA45oG_KnKW`.

The core idea comes from the change of variables formula for probability densities. For a 1D map $s(x)$, the pushforward density $P_f(s)$ evaluated at a point $d$ is given by:
$$ P_f(s)(d) = \sum_{u: s(u) = d} \frac{f(u)}{|s'(u)|} $$
where the sum is over all preimages $u$ such that $s(u) = d$, and $s'(u)$ is the derivative of the map $s(x)$ evaluated at $u$.

Here's how the `constraint_zeta_kgb` function implements this for the quadratic map $s(x) = \kappa x^2 + \alpha x + \beta$:

1.  **Find Preimages (`invert_quadratic_for_u`):** For each point `d` on the target grid (`grid.points`), the function first finds the preimage `u` such that `s(u) = d` using the `invert_quadratic_for_u` function. For a quadratic map, there can be up to two real preimages for a given `d`. This function specifically selects the root that is relevant in this context (the one continuous as kappa goes to zero).

2.  **Compute Source Density (`normal_pdf`):** It calculates the density of the source Gaussian $f(u)$ at the preimage point(s) `u` using the `normal_pdf` function with the source parameters (`params.mu0`, `params.s0`).

3.  **Compute Derivative of the Map (`s_prime`):** The derivative of the quadratic map is $s'(x) = 2\kappa x + \alpha$. This is calculated at the preimage point(s) `u` as `s_prime = 2.0*kappa*u + alpha`.

4.  **Apply Change of Variables Formula:** For each point `d` on the grid, the contribution to the pushforward density from its preimage `u` is calculated as `f_u / np.abs(s_prime)`. Since `invert_quadratic_for_u` is designed to return a single relevant preimage in this context, the sum in the general formula simplifies to this single term for each `d`.

5.  **Form the Constraint (`zeta`):** The constraint $\zeta(d)$ is defined as the difference between the pushforward density at `d` and the target density $g(d)$ at `d$:
    $$ \zeta(d) = P_f(s)(d) - g(d) $$
    The target density $g(d)$ is computed using `normal_pdf(d, mu1, s1)`. So the code calculates `zeta = (1.0 / alpha) * f_u - g_d`, which simplifies from `f_u / np.abs(s_prime) - g_d` when the map is strictly monotone and $s'(u) = \alpha > 0$. In the general quadratic case with variable $\kappa$, the formula is `f_u / np.abs(s_prime) - g_d`. The code actually uses `f_u / alpha` which seems to assume $\kappa=0$ or $\alpha>0$ dominates, which aligns with the SQP aiming for the affine solution.

In summary, the pushforward density is computed by evaluating the source density at the preimage(s) and dividing by the absolute value of the derivative of the map at those points, as per the change of variables formula. The `constraint_zeta_kgb` function then uses this to calculate the residual $\zeta$ by subtracting the target density.

In the SQP algorithm implemented in the `sqp_step_quadratic` function (within cell `dQA45oG_KnKW`), the descent direction (or more generally, the search direction) is chosen by solving a **Karush-Kuhn-Tucker (KKT) system**.

This system is a linear approximation of the optimality conditions for the constrained optimization problem at the current iterate. The KKT system is formed as follows:

$$
\begin{bmatrix}
H+\rho_p I & A^\top \\
A & -\rho_d I
\end{bmatrix}
\begin{bmatrix}
d \\ w
\end{bmatrix}
\;=\;
-\begin{bmatrix}
g \\ \zeta
\end{bmatrix}
$$

Where:

*   $H$ is the Hessian of the Lagrangian (or an approximation, like the objective Hessian `HJ` in this code).
*   $A$ is the Jacobian of the constraints $\zeta$.
*   $\rho_p$ and $\rho_d$ are small regularization parameters (primal and dual damping) used to improve the numerical stability of the system.
*   $g$ is the gradient of the Lagrangian.
*   $\zeta$ is the vector of constraint residuals.
*   $d$ is the vector of search directions for the primal variables $(\kappa, \gamma, \beta)$.
*   $w$ is the vector of search directions for the dual variables (Lagrange multipliers, $\lambda$).

The `sqp_step_quadratic` function builds this KKT matrix and the right-hand side vector using the current parameter values, dual variables, and the computed gradients, Hessians, and constraint values. It then solves this linear system using `np.linalg.lstsq` (a robust least-squares solver) to obtain the search directions $d$ and $w$. The primal variables $(\kappa, \gamma, \beta)$ and dual variables $\lambda$ are then updated by adding these search directions.

So, the descent direction `d` for the parameters is effectively the solution to this linearized system that aims to satisfy the first-order optimality conditions and reduce the constraint violation.

In the `sqp_solve_quadratic` function in cell `dQA45oG_KnKW`, the step size is not chosen through a traditional line search (like backtracking or Armijo conditions) which is common in some optimization algorithms.

Instead, the algorithm takes a **full step** determined directly by the solution `d` of the KKT system computed in `sqp_step_quadratic`. The update is simply:
$$ x_{new} = x + d $$
where $x$ represents the vector of primal variables $(\kappa, \gamma, \beta)$.

However, there is an optional **trust clipping** mechanism controlled by the `trust_clip` parameter. If the infinity norm of the computed step `d` is larger than `trust_clip`, the step `d` is uniformly scaled down so that its infinity norm equals `trust_clip`. This acts as a safeguard to prevent excessively large steps that might lead to instability, but it's not a dynamic step size selection based on function value decrease like a line search.

So, to summarize: there's no adaptive step size selection via line search; it's a full step from the KKT solution, potentially limited by the `trust_clip` parameter.

The `trust_clip` parameter serves as a **safeguard** for the step size determined by solving the KKT system.

Here's its role:

*   After solving the KKT system, the algorithm computes a proposed step `d` for the primal variables $(\kappa, \gamma, \beta)$.
*   The infinity norm of this step (`np.linalg.norm(d, np.inf)`) is checked against the `trust_clip` value.
*   If the step's infinity norm is **larger** than `trust_clip`, the step `d` is uniformly scaled down so that its infinity norm is equal to `trust_clip`.

This mechanism helps to prevent the optimization from taking overly large steps, which could lead to instability or divergence, especially when the local quadratic approximation (used in the KKT system) is not accurate far from the current point. It acts as a simple form of a trust region, limiting the maximum allowable step size in any single parameter dimension.

## Plot
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# from your_3d_solver import OTParams, make_grid, normal_pdf, sqp_solve_quadratic

def plot_ot_composite_quadratic(
    use_sqp: bool = True,
    save_path: str = "ot_composite_quadratic.png",
    params: OTParams = OTParams(mu0=-0.4, s0=0.82, mu1=1.6, s1=0.55),
    fixed_domain = (-1.0, 1.0),
    heatmap_xrange = None,
    heatmap_yrange = None,
    bins: int = 200,
    N_samples: int = 30_000,
    seed: int = 123,
    sqp_kwargs: dict = None
):
    """
    Center panel uses Y = kappa*X^2 + alpha*X + beta.
    If use_sqp=False, falls back to analytic affine OT (kappa=0).
    """
    # ----- choose the map for the center -----
    if use_sqp:
        if sqp_kwargs is None:
            sqp_kwargs = {}
        sol = sqp_solve_quadratic(params, **sqp_kwargs)   # <---- FORWARD kwargs
        kappa_c = float(sol["kappa"])
        alpha_c = float(sol["alpha"])
        beta_c  = float(sol["beta"])
    else:
        # Analytic affine OT for 1D Gaussian with quadratic cost
        kappa_c = 0.0
        alpha_c = params.s1 / params.s0
        beta_c  = params.mu1 - alpha_c * params.mu0

    # ----- domains -----
    grid = make_grid(params, m=401)
    Lc, Rc = grid.L, grid.R

    x_min_fix, x_max_fix = fixed_domain
    y_min_fix, y_max_fix = fixed_domain

    hx_min, hx_max = (heatmap_xrange if heatmap_xrange is not None else (Lc, Rc))
    hy_min, hy_max = (heatmap_yrange if heatmap_yrange is not None else (Lc, Rc))

    # ----- sample for center plan -----
    rng = np.random.default_rng(seed)
    X = rng.normal(params.mu0, params.s0, size=N_samples)
    Y = kappa_c * X**2 + alpha_c * X + beta_c

    # ----- MAIN HEATMAP -----
    H, xe, ye = np.histogram2d(
        X, Y,
        bins=bins,
        range=[[hx_min, hx_max], [hy_min, hy_max]]
    )

    main_w, main_h = 800, 800
    plt.figure(figsize=(main_w/100, main_h/100), dpi=100)
    plt.imshow(
        H.T, origin="lower",
        extent=[hx_min, hx_max, hy_min, hy_max],
        aspect="auto"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout(pad=0.6)
    main_path = "_panel_main_quad.png"
    plt.savefig(main_path, dpi=100)
    plt.close()

    # ----- TOP SOURCE p(x) with FIXED LIMITS -----
    x = np.linspace(x_min_fix, x_max_fix, 600)
    p = normal_pdf(x, params.mu0, params.s0)
    top_w, top_h = main_w, 240
    plt.figure(figsize=(top_w/100, top_h/100), dpi=100)
    plt.plot(x, p)
    plt.xlim(x_min_fix, x_max_fix)
    plt.title("source density p(x)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout(pad=0.6)
    top_path = "_panel_top_quad.png"
    plt.savefig(top_path, dpi=100)
    plt.close()

    # ----- LEFT TARGET q(y) with FIXED LIMITS -----
    y = np.linspace(y_min_fix, y_max_fix, 600)
    q = normal_pdf(y, params.mu1, params.s1)
    left_w, left_h = 240, main_h
    fig = plt.figure(figsize=(left_w/100, left_h/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(q, y)
    ax.invert_xaxis()
    ax.set_ylim(y_min_fix, y_max_fix)
    ax.set_title("target density q(y)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    fig.tight_layout(pad=0.6)
    left_path = "_panel_left_quad.png"
    fig.savefig(left_path, dpi=100)
    plt.close(fig)

    # ----- STITCH PANELS -----
    img_main = Image.open(main_path)
    img_top  = Image.open(top_path)
    img_left = Image.open(left_path)

    canvas_w = left_w + main_w
    canvas_h = top_h + main_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    pos_top  = (left_w, 0)
    pos_left = (0, top_h)
    pos_main = (left_w, top_h)

    canvas.paste(img_top,  pos_top)
    canvas.paste(img_left, pos_left)
    canvas.paste(img_main, pos_main)

    canvas.save(save_path)

    # ----- DISPLAY INLINE -----
    plt.figure(figsize=(canvas_w/100, canvas_h/100), dpi=100)
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout(pad=0.0)
    plt.show()

    return save_path

# Example:
plot_ot_composite_quadratic(
    use_sqp=True,
    save_path="ot_composite_quadratic.png",
    params=OTParams(mu0=-1.4, s0=2.2, mu1=0.6, s1=1.25),
    fixed_domain=(-8, 8),
    heatmap_xrange=(-8, 8),
    heatmap_yrange=(-8, 8),
    bins=200,
    sqp_kwargs=dict(
        m=101,
        max_iter=80,
        tol_opt=1e-8,
        tol_feas=1e-8,
        trust_clip=0.25,
        use_exact_constraint_hess=False,
        # init_kappa=0.0,
        init_gamma=np.log(1.15/1.0),
        init_beta=0.5 - (1.15/1.0)*(-0.2),
    )
)

import numpy as np
import matplotlib.pyplot as plt

# from your_3d_solver import OTParams, make_grid, sqp_solve_quadratic

def plot_transport_maps_quadratic(
    params: OTParams,
    save_path: str = "transport_maps_quadratic.png",
    X_RANGE = None,     # e.g. (-1.0, 1.0) or None for 95% domain
    Y_RANGE = None,     # e.g. (-1.0, 1.0) or None to auto
    m_grid: int = 401,
    sqp_kwargs: dict = None,
):
    # Analytic affine OT for 1D Gaussians
    alpha_star = params.s1 / params.s0
    beta_star  = params.mu1 - alpha_star * params.mu0

    # SQP (quadratic map)
    if sqp_kwargs is None:
            sqp_kwargs = {}
    sol = sqp_solve_quadratic(params, **sqp_kwargs)
    kappa_hat = float(sol["kappa"])
    alpha_hat = float(sol["alpha"])
    beta_hat  = float(sol["beta"])

    # x-range
    if X_RANGE is None:
        grid = make_grid(params, m=m_grid)
        xmin, xmax = grid.L, grid.R
    else:
        xmin, xmax = X_RANGE

    xs = np.linspace(xmin, xmax, 600)
    ys_affine   = alpha_star * xs + beta_star
    ys_quadratic = kappa_hat * xs**2 + alpha_hat * xs + beta_hat

    # y-range
    if Y_RANGE is None:
        ymin = min(ys_affine.min(), ys_quadratic.min())
        ymax = max(ys_affine.max(), ys_quadratic.max())
        pad = 0.05 * (ymax - ymin + 1e-12)
        ylo, yhi = ymin - pad, ymax + pad
    else:
        ylo, yhi = Y_RANGE

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(xs, ys_affine, linewidth=2, label="Analytic (affine): y = α* x + β*")
    plt.plot(xs, ys_quadratic, linestyle="--", linewidth=2,
             label="SQP (quadratic): y = κ x² + α x + β")
    plt.xlim(xmin, xmax)
    plt.ylim(ylo, yhi)
    plt.xlabel("x")
    plt.ylabel("y = s(x)")
    plt.title("Transport maps: Analytic affine vs SQP quadratic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved to:", save_path)

# Example:
plot_transport_maps_quadratic(
    params=OTParams(mu0=-1.4, s0=2.2, mu1=0.6, s1=1.25),
    save_path="transport_maps_quadratic.png",
    X_RANGE=(-8, 8),
    sqp_kwargs=dict(
        m=101,
        max_iter=80,
        tol_opt=1e-8,
        tol_feas=1e-8,
        trust_clip=0.25,
        use_exact_constraint_hess=False,
        init_kappa=0.0,
        init_gamma=np.log(1.15/1.0),
        init_beta=0.5 - (1.15/1.0)*(-0.2),
    )
)

