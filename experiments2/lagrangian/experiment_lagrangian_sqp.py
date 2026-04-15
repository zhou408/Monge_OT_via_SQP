"""
Beyond Brenier: Lagrangian Obstacle-Cost Optimal Transport
==========================================================
Experiment: 1D OT where the cost is itself defined through a variational problem.

PHYSICS:
  The cost of transporting mass from x to y is the minimum-action path
  through a potential landscape V(z):

    c(x,y) = inf_{gamma: gamma(0)=x, gamma(1)=y}
              int_0^1 [ lambda |gamma_dot(t)|^2 + V(gamma(t)) ] dt

  This is a classical mechanics action functional. The minimizing path
  satisfies Newton's equation gamma_ddot = V'(gamma) / (2 lambda).

  When V = 0, c(x,y) = lambda * (x-y)^2  (recovers quadratic cost).
  When V has a barrier, the cost becomes *non-quadratic* and *non-symmetric
  about the diagonal*: paths must detour around the obstacle, and the cost
  depends on the landscape between x and y, not just on |x-y|.

IMPLEMENTATION:
  1. Precompute c(x,y) on a fine grid by solving the inner path optimization
     for each (x,y) pair.
  2. Build a bicubic spline interpolant for c(x,y) and its partial derivatives.
  3. Run the discrete SQP solver (Section 8.4 of the paper):
     - At each iteration, solve the KKT system of a QP subproblem
       using the Hessian of the Lagrangian and the linearized constraint.
     - Armijo line search on the merit function M = L + (sigma/2)||tau||^2.

  Barriers tested:
    - Single Gaussian barrier at origin  (height=5, width=0.15)
    - Double barrier at +/-0.3           (creates two forbidden zones)

  Source:  Uniform[-1,1]
  Targets: ring (bimodal), concentrated (unimodal), asymmetric

Dependencies: numpy, scipy, matplotlib
Usage: python experiment_lagrangian_sqp.py
"""

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.interpolate import RectBivariateSpline, interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, warnings, sys
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration — tune these for your machine
# =============================================================================

# Cost precomputation
N_COST_GRID = 80        # Grid for c(x,y) lookup. 80 is good; 120+ is better.
N_PATH_PTS  = 25        # Interior points for path discretization.
LAM_KINETIC = 0.5       # Kinetic energy weight in Lagrangian.

# Transport solver
N_PIECES    = 40        # PL map pieces.
N_BINS      = 20        # Histogram bins for pushforward constraint.
N_QUAD      = 4000      # Quadrature points for objective.
N_FINE_PF   = 4000      # Fine grid for pushforward evaluation.
MAX_OUTER   = 80        # Max SQP iterations.
TOL_CV      = 5e-3      # Feasibility tolerance (L-inf of tau).

# Kantorovich LP
N_KANT      = 120       # Discretization for LP lower bound.

# Targets to test
TARGETS     = ['ring', 'concentrated', 'asymmetric']

# Barriers to test
BARRIERS    = ['single', 'double']


# =============================================================================
# Potential barriers
# =============================================================================

def V_single(z, h=5.0, w=0.15):
    """Single Gaussian barrier at origin."""
    return h * np.exp(-0.5 * (z / w)**2)

def V_single_prime(z, h=5.0, w=0.15):
    return -h * z / w**2 * np.exp(-0.5 * (z / w)**2)

def V_double(z):
    """Double Gaussian barrier at +/-0.3."""
    return (4.0 * np.exp(-0.5 * ((z - 0.3) / 0.12)**2) +
            4.0 * np.exp(-0.5 * ((z + 0.3) / 0.12)**2))

def V_double_prime(z):
    return (-4.0 * (z - 0.3) / 0.12**2 * np.exp(-0.5 * ((z - 0.3) / 0.12)**2)
            -4.0 * (z + 0.3) / 0.12**2 * np.exp(-0.5 * ((z + 0.3) / 0.12)**2))

BARRIER_FUNCS = {
    'single': (V_single, V_single_prime),
    'double': (V_double, V_double_prime),
}

BARRIER_COLORS = {
    'single': '#E91E63',
    'double': '#9C27B0',
}


# =============================================================================
# Inner path optimization: compute c(x,y) for one (x,y) pair
# =============================================================================

def solve_path(x, y, V, Vp, n_path=N_PATH_PTS, lam=LAM_KINETIC):
    """
    Solve  min_gamma int_0^1 [lam |gamma_dot|^2 + V(gamma)] dt
    with gamma(0)=x, gamma(1)=y.

    Returns (cost, optimal_path).
    """
    if abs(x - y) < 1e-12:
        val = V(np.atleast_1d(x))
        return float(val.ravel()[0]), np.array([x])

    dt = 1.0 / (n_path + 1)
    g0 = np.linspace(x, y, n_path + 2)[1:-1]  # interior points

    def action(gi):
        g = np.concatenate(([x], gi, [y]))
        dg = np.diff(g) / dt
        mid = 0.5 * (g[:-1] + g[1:])
        return lam * np.sum(dg**2) * dt + np.sum(V(mid)) * dt

    def action_grad(gi):
        g = np.concatenate(([x], gi, [y]))
        mid = 0.5 * (g[:-1] + g[1:])
        vp = Vp(mid)
        grad = np.zeros(n_path)
        for k in range(n_path):
            kk = k + 1
            # Kinetic: d/dg_kk of lam * sum((dg/dt)^2) * dt
            grad[k] = 2.0 * lam * (2*g[kk] - g[kk-1] - g[kk+1]) / dt
            # Potential: d/dg_kk of sum(V(mid)) * dt
            grad[k] += 0.5 * dt * (vp[kk - 1] + vp[kk])
        return grad

    res = minimize(action, g0, jac=action_grad, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})

    full_path = np.concatenate(([x], res.x, [y]))
    return res.fun, full_path


# =============================================================================
# Precompute cost table and build spline interpolant
# =============================================================================

def build_cost_spline(barrier_name, ng=N_COST_GRID):
    """
    Precompute c(x,y) on ng x ng grid and return:
      - C_table: the cost matrix
      - cost_spline: RectBivariateSpline interpolant
      - some_paths: dict of optimal paths for visualization
    """
    V, Vp = BARRIER_FUNCS[barrier_name]
    xg = np.linspace(-1, 1, ng)
    yg = np.linspace(-1, 1, ng)
    C = np.zeros((ng, ng))
    some_paths = {}

    t0 = time.time()
    for i in range(ng):
        for j in range(ng):
            c_val, path = solve_path(xg[i], yg[j], V, Vp)
            C[i, j] = c_val
            # Store a subset of paths for visualization
            if i % max(1, ng // 8) == 0 and j % max(1, ng // 8) == 0:
                some_paths[(i, j)] = path
        if (i + 1) % max(1, ng // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (ng - i - 1)
            print(f"    row {i+1}/{ng}  ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)

    spline = RectBivariateSpline(xg, yg, C, kx=3, ky=3)
    print(f"  Cost table built in {time.time()-t0:.1f}s  "
          f"(min={C.min():.4f}, max={C.max():.4f})")

    return C, spline, some_paths, xg, yg


# =============================================================================
# Cost function wrappers using spline interpolant
# =============================================================================

def make_cost_func(spline):
    """Return a vectorized cost function c(x,y) from the spline."""
    def cost_func(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        out = np.empty(len(x))
        for i in range(len(x)):
            out[i] = spline(x[i], y[i], grid=False).ravel()[0]
        return out
    return cost_func

def make_cost_grad_y(spline):
    """Return dc/dy using the spline's analytic partial derivative."""
    def grad_y(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        out = np.empty(len(x))
        for i in range(len(x)):
            out[i] = spline(x[i], y[i], dy=1, grid=False).ravel()[0]
        return out
    return grad_y


# =============================================================================
# Piecewise linear map
# =============================================================================

class PLMap:
    def __init__(self, n, x_range=(-1, 1)):
        self.n = n
        self.x_grid = np.linspace(x_range[0], x_range[1], n + 1)

    def eval(self, a, x):
        return np.interp(x, self.x_grid, a)


# =============================================================================
# Source and target distributions
# =============================================================================

def make_source_target(name='ring'):
    x_range = (-1.0, 1.0)
    if name == 'ring':
        target = lambda y: (0.4 * np.exp(-((y - 0.5)/0.15)**2) +
                            0.4 * np.exp(-((y + 0.5)/0.15)**2) + 0.05)
    elif name == 'concentrated':
        target = lambda y: np.exp(-y**2 / 0.04) + 0.02
    elif name == 'asymmetric':
        target = lambda y: 0.6 * np.exp(-((y - 0.4)/0.2)**2) + 0.15
    else:
        raise ValueError(f"Unknown target: {name}")

    y_fine = np.linspace(-1, 1, 10000)
    dy = 2.0 / 9999
    raw = target(y_fine)
    Z = np.sum(raw) * dy
    target_normed = lambda y, t=target, z=Z: t(y) / z
    return x_range, target_normed


def target_cdf_inv(target_density, n_inv=10000):
    y = np.linspace(-1, 1, n_inv)
    dy = 2.0 / (n_inv - 1)
    pdf = target_density(y)
    cdf = np.cumsum(pdf) * dy
    cdf /= cdf[-1]
    cdf[0] = 0.0
    return interp1d(cdf, y, kind='linear', bounds_error=False,
                    fill_value=(-1, 1))


# =============================================================================
# Pushforward constraint
# =============================================================================

def pushforward_hist(a, plm, target_density, n_bins=N_BINS, y_range=(-1, 1)):
    x_pts = np.linspace(plm.x_grid[0], plm.x_grid[-1], N_FINE_PF)
    dx = (plm.x_grid[-1] - plm.x_grid[0]) / (N_FINE_PF - 1)
    s_vals = plm.eval(a, x_pts)
    bin_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
    tau = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (s_vals >= bin_edges[b]) & (s_vals < bin_edges[b + 1])
        source_mass = np.sum(mask) * dx * 0.5
        y_mid = np.linspace(bin_edges[b], bin_edges[b + 1], 100)
        dy_mid = (bin_edges[b + 1] - bin_edges[b]) / 99
        target_mass = np.sum(target_density(y_mid)) * dy_mid
        tau[b] = source_mass - target_mass
    return tau


def jac_hist(a, plm, target_density, n_bins=N_BINS, fd_eps=5e-7):
    tau0 = pushforward_hist(a, plm, target_density, n_bins)
    J = np.zeros((n_bins, len(a)))
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        J[:, j] = (pushforward_hist(ap, plm, target_density, n_bins) - tau0) / fd_eps
    return J


# =============================================================================
# Objective and gradient (using spline analytic derivatives)
# =============================================================================

def objective(a, plm, cost_func):
    x_pts = np.linspace(plm.x_grid[0], plm.x_grid[-1], N_QUAD)
    dx = (plm.x_grid[-1] - plm.x_grid[0]) / (N_QUAD - 1)
    s_vals = plm.eval(a, x_pts)
    c_vals = cost_func(x_pts, s_vals)
    return np.sum(dx * 0.5 * c_vals)


def grad_obj_analytic(a, plm, cost_grad_y):
    """
    Analytic gradient of J(a) = int c(x, s(x)) * 0.5 dx w.r.t. PL knots a.

    dJ/da_k = int c_y(x, s(x)) * (ds/da_k)(x) * 0.5 dx

    For PL map, ds/da_k is the hat function centered at x_k.
    """
    x_pts = np.linspace(plm.x_grid[0], plm.x_grid[-1], N_QUAD)
    dx = (plm.x_grid[-1] - plm.x_grid[0]) / (N_QUAD - 1)
    s_vals = plm.eval(a, x_pts)
    cy = cost_grad_y(x_pts, s_vals)

    n1 = len(a)
    grad = np.zeros(n1)
    xg = plm.x_grid

    for k in range(n1):
        phi = np.zeros_like(x_pts)
        if k > 0:
            mask_l = (x_pts >= xg[k-1]) & (x_pts <= xg[k])
            h_l = xg[k] - xg[k-1]
            if h_l > 0:
                phi[mask_l] = (x_pts[mask_l] - xg[k-1]) / h_l
        if k < n1 - 1:
            mask_r = (x_pts > xg[k]) & (x_pts <= xg[k+1])
            h_r = xg[k+1] - xg[k]
            if h_r > 0:
                phi[mask_r] = (xg[k+1] - x_pts[mask_r]) / h_r
        grad[k] = np.sum(dx * 0.5 * cy * phi)

    return grad


def grad_obj_fd(a, plm, cost_func, fd_eps=1e-6):
    """Fallback: finite-difference gradient."""
    g = np.zeros(len(a))
    f0 = objective(a, plm, cost_func)
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        g[j] = (objective(ap, plm, cost_func) - f0) / fd_eps
    return g


# =============================================================================
# Hessian of the Lagrangian (diagonal approximation via D^2_yy c)
# =============================================================================

def hessian_lagrangian_diag(a, plm, cost_spline, omega, target_density):
    """
    Compute the diagonal of the Hessian of the Lagrangian H_L = H_obj + omega-weighted part.

    For the objective Hessian: H_ij = int phi_i(x) phi_j(x) D^2_yy c(x, s(x)) f(x) dx.
    We use a diagonal (Gauss-Newton) approximation: H_ii = int phi_i(x)^2 D^2_yy c(x,s(x)) f(x) dx.

    For the constraint Hessian contribution from the multiplier, we use a BFGS-style
    damped approximation (set to zero initially, i.e., Gauss-Newton on the Lagrangian).
    """
    n1 = len(a)
    xg = plm.x_grid
    x_pts = np.linspace(xg[0], xg[-1], N_QUAD)
    dx = (xg[-1] - xg[0]) / (N_QUAD - 1)
    s_vals = plm.eval(a, x_pts)

    # D^2_yy c(x, s(x)) via spline second derivative
    cyy = np.empty(N_QUAD)
    for i in range(N_QUAD):
        cyy[i] = cost_spline(x_pts[i], s_vals[i], dy=2, grid=False).ravel()[0]

    # Source density weight (uniform on [-1,1] => f(x) = 0.5)
    f_weight = 0.5

    diag_H = np.zeros(n1)
    for k in range(n1):
        phi_sq = np.zeros(N_QUAD)
        if k > 0:
            mask_l = (x_pts >= xg[k-1]) & (x_pts <= xg[k])
            h_l = xg[k] - xg[k-1]
            if h_l > 0:
                phi_sq[mask_l] = ((x_pts[mask_l] - xg[k-1]) / h_l) ** 2
        if k < n1 - 1:
            mask_r = (x_pts > xg[k]) & (x_pts <= xg[k+1])
            h_r = xg[k+1] - xg[k]
            if h_r > 0:
                phi_sq[mask_r] = ((xg[k+1] - x_pts[mask_r]) / h_r) ** 2
        diag_H[k] = np.sum(dx * f_weight * cyy * phi_sq)

    # Ensure positive definiteness with a small regularization
    diag_H = np.maximum(diag_H, 1e-6)
    return diag_H


# =============================================================================
# SQP solver (Section 8.4 of the paper)
# =============================================================================

def solve_sqp(plm, target_density, cost_func, cost_grad_y=None,
              cost_spline=None, inits=None, verbose=True):
    """
    Solve the Monge OT problem via Sequential Quadratic Programming.

    At each iteration k, we solve the KKT system of the QP subproblem (CPQPk):

        [ H_L    J_c^T ] [ delta_a     ]   [ -grad_L ]
        [ J_c    0     ] [ delta_omega  ] = [ -tau    ]

    where:
        H_L      = Hessian of the Lagrangian (diagonal approx)
        J_c      = Jacobian of the pushforward constraint (m x n)
        grad_L   = gradient of the Lagrangian = grad_J + J_c^T @ omega
        tau      = pushforward constraint residual

    Then update:  a_{k+1} = a_k + theta_k * delta_a
                  omega_{k+1} = omega_k + theta_k * delta_omega

    Step size theta_k chosen by Armijo line search on the merit function:
        M(a, omega, sigma) = L(a, omega) + (sigma/2) * ||tau(a)||_2^2
    """
    n1 = plm.n + 1
    m = N_BINS
    x_range = (plm.x_grid[0], plm.x_grid[-1])
    inv_cdf = target_cdf_inv(target_density)

    if inits is None:
        u = (plm.x_grid - x_range[0]) / (x_range[1] - x_range[0])
        inits = {
            'quantile':      inv_cdf(u),
            'anti-quantile': inv_cdf(1.0 - u),
            'tent':          inv_cdf(1.0 - np.abs(2*u - 1)),
            'V-shape':       inv_cdf(np.abs(2*u - 1)),
        }

    # Armijo parameters
    c_armijo = 1e-4
    theta_min = 1e-8
    sigma_merit = 10.0   # penalty parameter for merit function

    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"    {name}...", end='', flush=True)

        a = np.clip(a0.copy(), x_range[0] + 1e-4, x_range[1] - 1e-4)
        omega = np.zeros(m)
        hist = {'obj': [], 'cv_inf': [], 'merit': []}

        t0 = time.time()
        for outer in range(MAX_OUTER):
            # --- Evaluate constraint and objective ---
            tau = pushforward_hist(a, plm, target_density)
            J_val = objective(a, plm, cost_func)

            # --- Gradient of objective ---
            if cost_grad_y is not None:
                grad_J = grad_obj_analytic(a, plm, cost_grad_y)
            else:
                grad_J = grad_obj_fd(a, plm, cost_func)

            # --- Constraint Jacobian (m x n1) ---
            Jc = jac_hist(a, plm, target_density)

            # --- Gradient of Lagrangian ---
            grad_L = grad_J + Jc.T @ omega

            # --- Hessian of Lagrangian (diagonal approximation) ---
            if cost_spline is not None:
                H_diag = hessian_lagrangian_diag(a, plm, cost_spline,
                                                  omega, target_density)
            else:
                # Fallback: identity-scaled Hessian
                H_diag = np.ones(n1) * 1.0

            # --- Assemble and solve KKT system ---
            #  [ diag(H_diag)   Jc^T ] [ da     ]   [ -grad_L ]
            #  [ Jc             0    ] [ domega  ] = [ -tau    ]
            #
            # Eliminate da = H^{-1} (-grad_L - Jc^T domega)
            # => Jc H^{-1} Jc^T domega = -tau + Jc H^{-1} grad_L
            # (Schur complement)

            H_inv = 1.0 / H_diag  # diagonal inverse
            Jc_Hinv = Jc * H_inv[np.newaxis, :]  # m x n1
            S = Jc_Hinv @ Jc.T  # Schur complement, m x m

            # Regularize Schur complement for numerical stability
            S += 1e-8 * np.eye(m)

            rhs_schur = -tau + Jc_Hinv @ grad_L

            try:
                d_omega = np.linalg.solve(S, rhs_schur)
            except np.linalg.LinAlgError:
                d_omega = np.linalg.lstsq(S, rhs_schur, rcond=None)[0]

            d_a = H_inv * (-grad_L - Jc.T @ d_omega)

            # --- Record history ---
            cvinf = np.linalg.norm(tau, np.inf)
            hist['obj'].append(J_val)
            hist['cv_inf'].append(cvinf)

            # Check convergence
            if cvinf < TOL_CV and np.linalg.norm(grad_L, np.inf) < TOL_CV:
                break

            # --- Armijo line search on merit function ---
            # M(a, omega, sigma) = L(a, omega) + (sigma/2) * ||tau||^2
            # where L = J + omega^T tau
            L_val = J_val + omega @ tau
            M_val = L_val + 0.5 * sigma_merit * np.dot(tau, tau)

            # Directional derivative of merit function along (d_a, d_omega)
            dM = grad_L @ d_a + sigma_merit * tau @ (Jc @ d_a)

            theta = 1.0
            a_lo, a_hi = x_range[0] + 1e-4, x_range[1] - 1e-4

            for _ in range(30):
                a_trial = np.clip(a + theta * d_a, a_lo, a_hi)
                omega_trial = omega + theta * d_omega

                tau_trial = pushforward_hist(a_trial, plm, target_density)
                J_trial = objective(a_trial, plm, cost_func)
                L_trial = J_trial + omega_trial @ tau_trial
                M_trial = L_trial + 0.5 * sigma_merit * np.dot(tau_trial, tau_trial)

                if M_trial <= M_val + c_armijo * theta * dM:
                    break
                theta *= 0.5
                if theta < theta_min:
                    break

            hist['merit'].append(M_val)

            # --- Update ---
            a = np.clip(a + theta * d_a, a_lo, a_hi)
            omega = omega + theta * d_omega

            # Adaptively increase sigma if constraint not decreasing
            if outer > 0 and cvinf > 0.9 * hist['cv_inf'][-2]:
                sigma_merit = min(sigma_merit * 2.0, 1e6)

        elapsed = time.time() - t0

        # Final evaluation
        tau = pushforward_hist(a, plm, target_density)
        cvinf = np.linalg.norm(tau, np.inf)
        J_val = objective(a, plm, cost_func)

        ds = np.diff(a)
        n_neg = np.sum(ds < -1e-6); n_pos = np.sum(ds > 1e-6)
        if n_neg == 0:
            mono = 'monotone ↑'
        elif n_pos == 0:
            mono = 'monotone ↓'
        else:
            mono = f'non-monotone ({n_neg}↓ {n_pos}↑)'

        if verbose:
            print(f"  J={J_val:.5f}, cv={cvinf:.2e}, {mono}  [{elapsed:.1f}s]")

        all_results[name] = {
            'a': a.copy(), 'hist': hist,
            'J': J_val, 'cv': cvinf, 'mono': mono,
        }

    best_name = min(all_results,
                    key=lambda k: all_results[k]['J']
                    if all_results[k]['cv'] < 0.05 else 1e10)
    return all_results, best_name


# =============================================================================
# Kantorovich LP lower bound
# =============================================================================

def kant_lp(cost_func, target_density, n_disc=N_KANT, x_range=(-1, 1)):
    x = np.linspace(x_range[0], x_range[1], n_disc)
    y = np.linspace(x_range[0], x_range[1], n_disc)
    dx = (x_range[1] - x_range[0]) / (n_disc - 1)

    p = np.full(n_disc, dx * 0.5)
    p /= p.sum()
    q = target_density(y) * dx
    q /= q.sum()

    # Cost matrix
    C = np.zeros((n_disc, n_disc))
    for i in range(n_disc):
        C[i, :] = cost_func(np.full(n_disc, x[i]), y)
    c_flat = C.flatten()

    # Constraints: row sums = p, col sums = q
    n2 = n_disc * n_disc
    A_eq_rows = []
    for i in range(n_disc):
        row = np.zeros(n2)
        row[i*n_disc:(i+1)*n_disc] = 1.0
        A_eq_rows.append(row)
    for j in range(n_disc):
        row = np.zeros(n2)
        for i in range(n_disc):
            row[i*n_disc + j] = 1.0
        A_eq_rows.append(row)

    A_eq = np.array(A_eq_rows)
    b_eq = np.concatenate([p, q])

    try:
        res = linprog(c_flat, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, None)] * n2,
                      method='highs', options={'presolve': True})
        if res.success:
            return res.fun
    except Exception:
        pass
    return None


# =============================================================================
# Twist condition check
# =============================================================================

def check_twist(C_table, xg, yg):
    """Compute c_xy via finite differences on the cost table."""
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    ng = len(xg)
    cxy = np.zeros((ng - 2, ng - 2))
    for i in range(1, ng - 1):
        for j in range(1, ng - 1):
            cxy[i-1, j-1] = (C_table[i+1,j+1] - C_table[i+1,j-1]
                              - C_table[i-1,j+1] + C_table[i-1,j-1]) / (4*dx*dy)
    frac_neg = np.mean(cxy < 0)
    frac_pos = np.mean(cxy > 0)
    return cxy, frac_neg, frac_pos


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 75)
    print("  BEYOND BRENIER: Lagrangian Obstacle-Cost Optimal Transport")
    print("  c(x,y) = inf_gamma int [lam |gamma_dot|^2 + V(gamma)] dt")
    print("  Source: Uniform[-1,1]")
    print("=" * 75)

    cost_quad = lambda x, y: np.atleast_1d((np.asarray(x) - np.asarray(y))**2).ravel()

    all_data = {}

    for bname in BARRIERS:
        print(f"\n{'='*75}")
        print(f"  Barrier: {bname}")
        print(f"{'='*75}")

        # --- Step 1: Precompute cost table ---
        print(f"\n  Precomputing Lagrangian cost on {N_COST_GRID}x{N_COST_GRID} grid...")
        C_table, spline, paths, xg, yg = build_cost_spline(bname)

        cost_func = make_cost_func(spline)
        cost_gy   = make_cost_grad_y(spline)

        # --- Step 2: Twist check ---
        cxy, frac_neg, frac_pos = check_twist(C_table, xg, yg)
        print(f"  Twist check: c_xy < 0 in {frac_neg:.1%}, > 0 in {frac_pos:.1%}")
        print(f"  Twist holds uniformly: {'YES' if np.all(cxy < 0) else 'NO'}")

        all_data[bname] = {
            'C_table': C_table, 'spline': spline, 'paths': paths,
            'xg': xg, 'yg': yg, 'cxy': cxy,
            'frac_neg': frac_neg, 'frac_pos': frac_pos,
            'targets': {},
        }

        # --- Step 3: Solve for each target ---
        for tname in TARGETS:
            print(f"\n  Target: {tname}")
            x_range, target_density = make_source_target(tname)
            plm = PLMap(N_PIECES, x_range)

            ar, best = solve_sqp(plm, target_density, cost_func,
                                cost_grad_y=cost_gy, cost_spline=spline,
                                verbose=True)

            print(f"  Computing Kantorovich LP bound (n={N_KANT})...", end='', flush=True)
            K = kant_lp(cost_func, target_density, n_disc=N_KANT)
            if K is not None:
                gap = ar[best]['J'] - K
                print(f"  K={K:.5f}, gap={gap:.5f} ({100*abs(gap)/abs(K):.1f}%)")
            else:
                print(f"  LP failed")

            # Also solve with quadratic cost for comparison
            print(f"  Quadratic baseline...", flush=True)
            ar_q, best_q = solve_sqp(plm, target_density, cost_quad,
                                     verbose=False)
            K_q = kant_lp(cost_quad, target_density, n_disc=N_KANT)

            all_data[bname]['targets'][tname] = {
                'all_res': ar, 'best_name': best, 'K': K, 'plm': plm,
                'target_density': target_density,
                'quad_res': ar_q, 'quad_best': best_q, 'K_quad': K_q,
            }

    # =========================================================================
    # FIGURES
    # =========================================================================
    print(f"\n{'='*75}")
    print("  Generating figures...")
    print(f"{'='*75}")

    x_fine = np.linspace(-1, 1, 1000)
    init_colors = {
        'quantile': '#2196F3', 'anti-quantile': '#E91E63',
        'tent': '#9C27B0', 'V-shape': '#4CAF50',
    }

    # ---- Fig 0: Potential and cost landscape ----
    fig, axes = plt.subplots(2, len(BARRIERS) + 1,
                             figsize=(7*(len(BARRIERS)+1), 10))

    z = np.linspace(-1, 1, 500)
    ax = axes[0, 0]
    for bname in BARRIERS:
        V, _ = BARRIER_FUNCS[bname]
        ax.plot(z, V(z), color=BARRIER_COLORS[bname], lw=2.5, label=bname)
        ax.fill_between(z, V(z), alpha=0.1, color=BARRIER_COLORS[bname])
    ax.set_xlabel('$z$', fontsize=12); ax.set_ylabel('$V(z)$', fontsize=12)
    ax.set_title('Obstacle Potentials', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    for bi, bname in enumerate(BARRIERS):
        d = all_data[bname]
        ax = axes[0, bi + 1]
        im = ax.imshow(d['C_table'].T, origin='lower', extent=[-1,1,-1,1],
                       cmap='viridis', aspect='auto')
        ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
        ax.set_title(f'$c(x,y)$: {bname} barrier', fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Cost slices
    ax = axes[1, 0]
    for bname in BARRIERS:
        d = all_data[bname]
        cf = make_cost_func(d['spline'])
        yy = np.linspace(-1, 1, 200)
        xx = np.full_like(yy, -0.5)
        cv = cf(xx, yy)
        ax.plot(yy, cv, color=BARRIER_COLORS[bname], lw=2, label=f'{bname} barrier')
    ax.plot(yy, (yy + 0.5)**2, 'k--', lw=1.5, alpha=0.5, label='quadratic')
    ax.set_xlabel('$y$', fontsize=12); ax.set_ylabel('$c(-0.5, y)$', fontsize=12)
    ax.set_title('Cost Slice at $x = -0.5$', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Twist matrices
    for bi, bname in enumerate(BARRIERS):
        d = all_data[bname]
        ax = axes[1, bi + 1]
        vmax = max(np.percentile(np.abs(d['cxy']), 95), 1e-10)
        im = ax.imshow(d['cxy'].T, origin='lower', extent=[-1,1,-1,1],
                       cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
        ax.set_title(f'$c_{{xy}}$: {bname} ($c_{{xy}}<0$: {d["frac_neg"]:.0%})',
                     fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        'Lagrangian Cost: $c(x,y) = \\inf_\\gamma \\int '
        '[\\lambda|\\dot\\gamma|^2 + V(\\gamma)]\\,dt$',
        fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_lagrangian_0_cost_landscape.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_0_cost_landscape.png")

    # ---- Fig 1: Local optima grid — all inits × barriers × targets ----
    fig, axes = plt.subplots(len(BARRIERS), len(TARGETS),
                             figsize=(5.5*len(TARGETS), 5*len(BARRIERS)),
                             sharex=True, sharey=True)
    if len(BARRIERS) == 1:
        axes = axes[np.newaxis, :]

    for row, bname in enumerate(BARRIERS):
        for col, tname in enumerate(TARGETS):
            ax = axes[row, col]
            td = all_data[bname]['targets'][tname]
            plm = td['plm']
            cell = td['all_res']
            best = td['best_name']

            sorted_inits = sorted(cell.keys(), key=lambda k: k == best)
            for iname in sorted_inits:
                res = cell[iname]
                s_vals = plm.eval(res['a'], x_fine)
                ib = (iname == best)
                lw = 2.8 if ib else 1.0
                alpha = 1.0 if ib else 0.35
                star = ' \u2605' if ib else ''
                feas = '' if res['cv'] < 0.05 else ' [!]'
                label = f"{iname}: J={res['J']:.4f}{star}{feas}"
                ax.plot(x_fine, s_vals, color=init_colors[iname],
                        lw=lw, alpha=alpha, label=label)

            ax.plot([-1,1], [-1,1], ':', color='gray', lw=0.7, alpha=0.4)
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(tname, fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{bname} barrier\n$s(x)$', fontsize=11)
            if row == len(BARRIERS) - 1:
                ax.set_xlabel('$x$', fontsize=11)
            ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85)

    plt.suptitle('Local Optima: Lagrangian Obstacle Cost\n'
                 '4 initializations — best marked with \u2605',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_lagrangian_1_local_optima.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_1_local_optima.png")

    # ---- Fig 2: Best map overlaid on cost matrix ----
    fig, axes = plt.subplots(len(BARRIERS), len(TARGETS),
                             figsize=(5.5*len(TARGETS), 5*len(BARRIERS)))
    if len(BARRIERS) == 1:
        axes = axes[np.newaxis, :]

    for row, bname in enumerate(BARRIERS):
        d = all_data[bname]
        for col, tname in enumerate(TARGETS):
            ax = axes[row, col]
            td = d['targets'][tname]
            plm = td['plm']
            best = td['best_name']
            a_best = td['all_res'][best]['a']
            J_best = td['all_res'][best]['J']
            K = td['K']

            im = ax.imshow(d['C_table'].T, origin='lower', extent=[-1,1,-1,1],
                           cmap='viridis', aspect='auto', alpha=0.8)
            s_best = plm.eval(a_best, x_fine)
            ax.plot(x_fine, s_best, 'w-', lw=3, alpha=0.9)
            ax.plot(x_fine, s_best, color=BARRIER_COLORS[bname], lw=1.5, alpha=0.8)

            gap_str = (f'\nKant={K:.4f}, gap={100*abs(J_best-K)/abs(K):.1f}%'
                       if K else '')
            ax.set_title(f'{bname} \u2192 {tname}\nJ={J_best:.4f}{gap_str}',
                         fontsize=10)
            ax.set_xlabel('$x$', fontsize=10)
            if col == 0:
                ax.set_ylabel('$y = s(x)$', fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Best Maps Overlaid on Lagrangian Cost Matrix',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_lagrangian_2_maps_on_cost.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_2_maps_on_cost.png")

    # ---- Fig 3: Lagrangian vs quadratic comparison ----
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6*len(TARGETS), 5))
    for col, tname in enumerate(TARGETS):
        ax = axes[col]
        td_q = all_data[BARRIERS[0]]['targets'][tname]
        s_q = td_q['plm'].eval(
            td_q['quad_res'][td_q['quad_best']]['a'], x_fine)
        ax.plot(x_fine, s_q, 'k--', lw=2, alpha=0.6, label='quadratic (Brenier)')

        for bname in BARRIERS:
            td = all_data[bname]['targets'][tname]
            plm = td['plm']
            s_best = plm.eval(td['all_res'][td['best_name']]['a'], x_fine)
            ax.plot(x_fine, s_best, color=BARRIER_COLORS[bname], lw=2.5,
                    label=f'{bname} barrier')

        ax.plot([-1,1], [-1,1], ':', color='gray', lw=0.7, alpha=0.4)
        ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$s(x)$', fontsize=12)
        ax.set_title(f'Target: {tname}', fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Obstacle Cost vs Quadratic: How Barriers Reshape the Map',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_lagrangian_3_vs_quadratic.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_3_vs_quadratic.png")

    # ---- Fig 4: Convergence ----
    fig, axes = plt.subplots(len(BARRIERS), len(TARGETS),
                             figsize=(5.5*len(TARGETS), 4.5*len(BARRIERS)))
    if len(BARRIERS) == 1:
        axes = axes[np.newaxis, :]

    for row, bname in enumerate(BARRIERS):
        for col, tname in enumerate(TARGETS):
            ax = axes[row, col]
            td = all_data[bname]['targets'][tname]
            for iname, res in td['all_res'].items():
                h = res['hist']
                ib = (iname == td['best_name'])
                ax.semilogy(h['cv_inf'], color=init_colors[iname],
                            lw=2 if ib else 0.8, alpha=1 if ib else 0.4,
                            marker='o', ms=2 if ib else 1, label=iname)
            ax.axhline(TOL_CV, color='red', ls=':', lw=1, alpha=0.5, label='tol')
            ax.set_xlabel('Outer iteration', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{bname}\n$\\|\\tau\\|_\\infty$', fontsize=10)
            ax.set_title(f'{tname}', fontsize=11)
            ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3)

    plt.suptitle('Convergence: Constraint Violation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_lagrangian_4_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_4_convergence.png")

    # ---- Fig 5: Optimal transport paths through barrier ----
    fig, axes = plt.subplots(1, len(BARRIERS), figsize=(8*len(BARRIERS), 6))
    if len(BARRIERS) == 1:
        axes = [axes]

    for bi, bname in enumerate(BARRIERS):
        ax = axes[bi]
        V, Vp = BARRIER_FUNCS[bname]
        d = all_data[bname]

        # Barrier as shaded background
        zz = np.linspace(-1, 1, 500)
        v_vals = V(zz)
        v_norm = v_vals / max(v_vals.max(), 1e-10)
        for k in range(len(zz) - 1):
            ax.axhspan(zz[k], zz[k+1], alpha=0.18*v_norm[k], color='orange')

        # Optimal paths for ring target
        td = d['targets']['ring']
        plm = td['plm']
        best = td['best_name']
        a_best = td['all_res'][best]['a']

        n_show = 20
        x_show = np.linspace(-0.95, 0.95, n_show)
        y_show = plm.eval(a_best, x_show)
        for k in range(n_show):
            _, path = solve_path(x_show[k], y_show[k], V, Vp)
            t_path = np.linspace(0, 1, len(path))
            color = plt.cm.coolwarm(0.5 * (x_show[k] + 1))
            ax.plot(t_path, path, color=color, lw=1.5, alpha=0.6)
            ax.plot(0, x_show[k], 'o', color=color, ms=4, alpha=0.8)
            ax.plot(1, y_show[k], 's', color=color, ms=4, alpha=0.8)

        ax.set_xlabel('$t$ (path parameter)', fontsize=12)
        ax.set_ylabel('$\\gamma(t)$', fontsize=12)
        ax.set_title(f'{bname.title()} Barrier: Optimal Transport Paths\n'
                     f'(Ring target, {best} init)', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_lagrangian_5_paths.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_lagrangian_5_paths.png")

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  {'Barrier':>10} {'Target':>14} {'Best init':>14} "
          f"{'Monge J':>10} {'Kant LB':>10} {'Gap%':>7} {'Mono':>25}")
    print(f"{'-'*90}")
    for bname in BARRIERS:
        for tname in TARGETS:
            td = all_data[bname]['targets'][tname]
            best = td['best_name']
            res = td['all_res'][best]
            K = td['K']
            gap = 100 * abs(res['J'] - K) / abs(K) if K and K != 0 else 0
            K_str = f"{K:.5f}" if K else "FAILED"
            print(f"  {bname:>10} {tname:>14} {best:>14} "
                  f"{res['J']:10.5f} {K_str:>10} {gap:6.1f}% "
                  f"{res['mono']:>25}")

    print(f"\n  Quadratic (Brenier) baselines:")
    for tname in TARGETS:
        td = all_data[BARRIERS[0]]['targets'][tname]
        best_q = td['quad_best']
        res_q = td['quad_res'][best_q]
        K_q = td['K_quad']
        gap_q = 100*abs(res_q['J']-K_q)/abs(K_q) if K_q and K_q != 0 else 0
        K_q_str = f"{K_q:.5f}" if K_q else "FAILED"
        print(f"  {'quadratic':>10} {tname:>14} {best_q:>14} "
              f"{res_q['J']:10.5f} {K_q_str:>10} {gap_q:6.1f}%")
    print(f"{'='*90}")

    print("\nAll figures saved. Done.")


if __name__ == '__main__':
    main()
