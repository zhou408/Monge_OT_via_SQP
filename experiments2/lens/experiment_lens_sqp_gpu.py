"""
Beyond Brenier: SQP for General-Cost Monge Optimal Transport (GPU)
===================================================================
Refactored lens/reflector experiments using:
  - SQP (Sequential Quadratic Programming) instead of Augmented Lagrangian
  - GPU acceleration via PyTorch CUDA

Two sub-experiments:
  1. Twisted costs on [-1,1]: quadratic c=(x-y)^2 and lens c=sqrt((x-y)^2+D^2)
  2. Reflector antenna on S^1: c=-log(2 sin(|θ_x-θ_y|/2))

Usage: python experiment_lens_sqp_gpu.py [--device gpu|cpu]
"""

import argparse
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# PyTorch + device setup
# ---------------------------------------------------------------------------
import torch

_D = torch.float64

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='Compute device: gpu or cpu')
    return parser.parse_args()

def setup_device(requested):
    if requested == 'gpu' and torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
        if requested == 'gpu':
            print("  [WARNING] CUDA not available, falling back to CPU")
    print(f"  [device] {dev}")
    return dev

DEVICE = 'cpu'  # set in main()

def _to(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=_D, device=DEVICE)

def _np(t):
    return t.detach().cpu().numpy()


# =========================================================================
# Cost functions (numpy for LP / exact, torch for GPU SQP)
# =========================================================================

def cost_quadratic_np(x, y):
    return (x - y) ** 2

def cost_lens_np(x, y, D=1.0):
    return np.sqrt((x - y) ** 2 + D ** 2)

def cost_log_reflector_s1_np(theta_x, theta_y):
    d = np.abs(theta_x - theta_y)
    d = np.minimum(d, 2 * np.pi - d)
    cd = 2.0 * np.sin(d / 2.0)
    return -np.log(np.maximum(cd, 1e-15))

def cost_quadratic_torch(x, y):
    return (x - y) ** 2

def cost_lens_torch(x, y, D=1.0):
    return torch.sqrt((x - y) ** 2 + D ** 2)

def cost_log_reflector_s1_torch(theta_x, theta_y):
    d = torch.abs(theta_x - theta_y)
    d = torch.minimum(d, 2 * np.pi - d)
    cd = 2.0 * torch.sin(d / 2.0)
    return -torch.log(torch.clamp(cd, min=1e-15))

# Cost y-derivatives (analytic, torch)
def cost_quadratic_dy(x, y):
    return -2.0 * (x - y)

def cost_lens_dy(x, y, D=1.0):
    return -(x - y) / torch.sqrt((x - y) ** 2 + D ** 2)

def cost_quadratic_dyy(x, y):
    return torch.full_like(x, 2.0)

def cost_lens_dyy(x, y, D=1.0):
    r2 = (x - y) ** 2 + D ** 2
    return D ** 2 / (r2 * torch.sqrt(r2))

def cost_log_reflector_dy(theta_x, theta_y):
    d = theta_x - theta_y
    abs_d = torch.abs(d)
    abs_d_mod = torch.where(abs_d > np.pi, 2 * np.pi - abs_d, abs_d)
    half_d = abs_d_mod / 2.0
    sin_hd = torch.sin(half_d)
    cos_hd = torch.cos(half_d)
    # d/d(theta_y) of -log(2 sin(|theta_x - theta_y|/2))
    # = cos(d/2) / (2 sin(d/2)) * sign
    sign = torch.sign(d)
    sign = torch.where(abs_d > np.pi, -sign, sign)
    return sign * cos_hd / (2.0 * torch.clamp(sin_hd, min=1e-15))

def cost_log_reflector_dyy(theta_x, theta_y):
    d = torch.abs(theta_x - theta_y)
    d = torch.where(d > np.pi, 2 * np.pi - d, d)
    half_d = d / 2.0
    sin_hd = torch.sin(half_d)
    return 1.0 / (4.0 * torch.clamp(sin_hd ** 2, min=1e-15))


# =========================================================================
# Piecewise linear map (GPU)
# =========================================================================

class PLMap:
    def __init__(self, n, x_range=(-1, 1)):
        self.n = n
        self.x_grid = np.linspace(x_range[0], x_range[1], n + 1)

    def eval_np(self, a, x):
        return np.interp(x, self.x_grid, a)

    def eval_torch(self, a_t, x_t):
        """GPU piecewise-linear interpolation."""
        xg = _to(self.x_grid)
        n1 = len(xg)
        sx = (x_t - xg[0]) / (xg[-1] - xg[0]) * (n1 - 1)
        sx = sx.clamp(0, n1 - 1 - 1e-7)
        ix = sx.long()
        ix1 = (ix + 1).clamp(max=n1 - 1)
        frac = sx - ix.to(_D)
        return a_t[ix] * (1 - frac) + a_t[ix1] * frac


# =========================================================================
# Source/target distributions
# =========================================================================

def make_source_target(name='ring'):
    x_range = (-1.0, 1.0)
    if name == 'ring':
        target = lambda y: (0.4 * np.exp(-((y - 0.5) / 0.15) ** 2) +
                             0.4 * np.exp(-((y + 0.5) / 0.15) ** 2) + 0.05)
    elif name == 'concentrated':
        target = lambda y: np.exp(-y ** 2 / 0.04) + 0.02
    elif name == 'asymmetric':
        target = lambda y: 0.6 * np.exp(-((y - 0.4) / 0.2) ** 2) + 0.15
    else:
        raise ValueError(f"Unknown target: {name}")
    y_fine = np.linspace(-1, 1, 10000)
    dy = 2.0 / 9999
    Z = np.sum(target(y_fine)) * dy
    target_normed = lambda y, t=target, z=Z: t(y) / z
    return x_range, target_normed


def target_cdf_inv(target_density, lo=-1.0, hi=1.0, n_inv=10000):
    y = np.linspace(lo, hi, n_inv)
    dy = (hi - lo) / (n_inv - 1)
    pdf = target_density(y)
    cdf = np.cumsum(pdf) * dy
    cdf = cdf / cdf[-1]
    cdf[0] = 0.0
    return interp1d(cdf, y, kind='linear', bounds_error=False,
                    fill_value=(lo, hi))


# =========================================================================
# GPU pushforward constraint + Jacobian
# =========================================================================

def pushforward_hist_gpu(a_t, plm, target_density, n_bins, y_range,
                         n_fine=4000):
    """Histogram pushforward constraint on GPU."""
    x_lo, x_hi = plm.x_grid[0], plm.x_grid[-1]
    f_density = 1.0 / (x_hi - x_lo)
    x_pts = torch.linspace(x_lo, x_hi, n_fine, dtype=_D, device=DEVICE)
    dx = (x_hi - x_lo) / (n_fine - 1)

    s_vals = plm.eval_torch(a_t, x_pts)

    bin_edges = torch.linspace(y_range[0], y_range[1], n_bins + 1,
                               dtype=_D, device=DEVICE)
    tau = torch.zeros(n_bins, dtype=_D, device=DEVICE)
    for b in range(n_bins):
        mask = (s_vals >= bin_edges[b]) & (s_vals < bin_edges[b + 1])
        source_mass = mask.sum().to(_D) * dx * f_density
        y_mid = np.linspace(float(bin_edges[b]), float(bin_edges[b + 1]), 100)
        dy_mid = (float(bin_edges[b + 1]) - float(bin_edges[b])) / 99
        target_mass = np.sum(target_density(y_mid)) * dy_mid
        tau[b] = source_mass - target_mass
    return tau


def jac_hist_gpu(a_t, plm, target_density, n_bins, y_range, fd_eps=5e-7):
    """Finite-difference Jacobian on GPU."""
    tau0 = pushforward_hist_gpu(a_t, plm, target_density, n_bins, y_range)
    n1 = len(a_t)
    J = torch.zeros(n_bins, n1, dtype=_D, device=DEVICE)
    for j in range(n1):
        ap = a_t.clone()
        ap[j] += fd_eps
        J[:, j] = (pushforward_hist_gpu(ap, plm, target_density, n_bins,
                                         y_range) - tau0) / fd_eps
    return J


# =========================================================================
# GPU objective, gradient, Hessian
# =========================================================================

def objective_gpu(a_t, plm, cost_torch_func, n_quad=4000):
    """J(s) = int c(x, s(x)) f(x) dx on GPU."""
    x_lo, x_hi = plm.x_grid[0], plm.x_grid[-1]
    f_density = 1.0 / (x_hi - x_lo)
    x_pts = torch.linspace(x_lo, x_hi, n_quad, dtype=_D, device=DEVICE)
    dx = (x_hi - x_lo) / (n_quad - 1)
    s_vals = plm.eval_torch(a_t, x_pts)
    return (cost_torch_func(x_pts, s_vals) * f_density * dx).sum()


def grad_obj_gpu(a_t, plm, cost_dy_func, n_quad=4000):
    """Analytic gradient: grad_k = int phi_k(x) c_y(x,s(x)) f(x) dx."""
    x_lo, x_hi = plm.x_grid[0], plm.x_grid[-1]
    f_density = 1.0 / (x_hi - x_lo)
    xg = _to(plm.x_grid)
    n1 = len(xg)
    x_pts = torch.linspace(x_lo, x_hi, n_quad, dtype=_D, device=DEVICE)
    dx = (x_hi - x_lo) / (n_quad - 1)
    s_vals = plm.eval_torch(a_t, x_pts)
    cy = cost_dy_func(x_pts, s_vals)
    w = cy * f_density * dx  # (n_quad,)

    # Hat functions via vectorized computation
    grad = torch.zeros(n1, dtype=_D, device=DEVICE)
    for k in range(n1):
        phi = torch.zeros_like(x_pts)
        if k > 0:
            h_l = xg[k] - xg[k - 1]
            if h_l > 0:
                m = (x_pts >= xg[k - 1]) & (x_pts <= xg[k])
                phi[m] = (x_pts[m] - xg[k - 1]) / h_l
        if k < n1 - 1:
            h_r = xg[k + 1] - xg[k]
            if h_r > 0:
                m = (x_pts > xg[k]) & (x_pts <= xg[k + 1])
                phi[m] = (xg[k + 1] - x_pts[m]) / h_r
        grad[k] = (phi * w).sum()
    return grad


def hessian_diag_gpu(a_t, plm, cost_dyy_func, n_quad=4000):
    """Diagonal Hessian: H_k = int phi_k^2(x) c_yy(x,s(x)) f(x) dx."""
    x_lo, x_hi = plm.x_grid[0], plm.x_grid[-1]
    f_density = 1.0 / (x_hi - x_lo)
    xg = _to(plm.x_grid)
    n1 = len(xg)
    x_pts = torch.linspace(x_lo, x_hi, n_quad, dtype=_D, device=DEVICE)
    dx = (x_hi - x_lo) / (n_quad - 1)
    s_vals = plm.eval_torch(a_t, x_pts)
    cyy = cost_dyy_func(x_pts, s_vals)
    w = cyy * f_density * dx

    H = torch.zeros(n1, dtype=_D, device=DEVICE)
    for k in range(n1):
        phi_sq = torch.zeros_like(x_pts)
        if k > 0:
            h_l = xg[k] - xg[k - 1]
            if h_l > 0:
                m = (x_pts >= xg[k - 1]) & (x_pts <= xg[k])
                phi_sq[m] = ((x_pts[m] - xg[k - 1]) / h_l) ** 2
        if k < n1 - 1:
            h_r = xg[k + 1] - xg[k]
            if h_r > 0:
                m = (x_pts > xg[k]) & (x_pts <= xg[k + 1])
                phi_sq[m] = ((xg[k + 1] - x_pts[m]) / h_r) ** 2
        H[k] = (phi_sq * w).sum()

    return torch.clamp(H, min=1e-2)


# =========================================================================
# Monotone reparametrization: a[k] = d[0] + sum_{j=1}^{k} exp(d[j])
# =========================================================================

def _isotonic_projection(a_np):
    """Project a_np to be monotone non-decreasing (pool adjacent violators)."""
    out = a_np.copy()
    n = len(out)
    i = 0
    while i < n - 1:
        if out[i + 1] < out[i]:
            j = i + 1
            while j < n - 1 and out[j + 1] < out[j]:
                j += 1
            avg = np.mean(out[i:j + 1])
            out[i:j + 1] = avg
        i += 1
    for k in range(1, n):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + 1e-8
    return out


def _a_to_d(a_np):
    """Monotone knots a -> unconstrained log-increment params d.

    d[0] = a[0]
    d[k] = log(a[k] - a[k-1])   for k >= 1
    """
    d = np.zeros_like(a_np)
    d[0] = a_np[0]
    diffs = np.diff(a_np)
    diffs = np.maximum(diffs, 1e-10)
    d[1:] = np.log(diffs)
    return d


def _d_to_a(d_t):
    """Unconstrained params d -> monotone knots a (torch, GPU).

    a[0] = d[0]
    a[k] = d[0] + sum_{j=1}^{k} exp(d[j])

    Clamps exp(d) to avoid overflow for extreme d values.
    """
    increments = torch.exp(d_t[1:].clamp(max=10.0))
    a = torch.empty_like(d_t)
    a[0] = d_t[0]
    a[1:] = d_t[0] + torch.cumsum(increments, dim=0)
    return a


def _reparam_jac(d_t):
    """Jacobian da/dd, lower triangular (n1 x n1).

    J[0, 0] = 1
    J[k, 0] = 1            for k >= 1  (shift from d[0])
    J[k, j] = exp(d[j])    for 1 <= j <= k
    """
    n1 = len(d_t)
    Jr = torch.zeros(n1, n1, dtype=_D, device=d_t.device)
    Jr[:, 0] = 1.0
    exp_d = torch.exp(d_t[1:].clamp(max=10.0))
    for k in range(1, n1):
        Jr[k, 1:k + 1] = exp_d[:k]
    return Jr


# =========================================================================
# SQP solver (GPU) — monotone reparametrization
# =========================================================================

def solve_sqp_gpu(plm, target_density, cost_torch_func, cost_dy_func,
                  cost_dyy_func, n_bins=20, y_range=(-1, 1),
                  inits=None, max_outer=200, tol_cv=1e-3,
                  enforce_monotone=True, verbose=True):
    """
    SQP solver on GPU with KKT/Schur complement.

    When enforce_monotone=True, optimizes in d-space where
        a[0] = d[0],  a[k] = d[0] + sum_{j=1}^{k} exp(d[j])
    so monotonicity holds by construction for any d.  All initializations
    are projected to monotone then converted to d-space, so all 4
    starting points (quantile, anti-quantile, tent, V-shape) are used.

    The KKT system in d-space:
        [ H_d    Jc_d^T ] [ dd     ]   [ -grad_Ld ]
        [ Jc_d   0      ] [ domega ] = [ -tau      ]

    where  H_d  = Jr^T diag(H_a) Jr     (diagonal approx)
           Jc_d = Jc_a Jr               (chain rule)
           grad_Ld = Jr^T grad_La       (chain rule)
    and Jr = da/dd is the lower-triangular reparametrization Jacobian.
    """
    n1 = plm.n + 1
    m = n_bins
    x_lo, x_hi = y_range

    inv_cdf = target_cdf_inv(target_density, lo=x_lo, hi=x_hi)

    if inits is None:
        u = (plm.x_grid - x_lo) / (x_hi - x_lo)
        inits = {
            'quantile':      inv_cdf(u),
            'anti-quantile': inv_cdf(1.0 - u),
            'tent':          inv_cdf(1.0 - np.abs(2 * u - 1)),
            'V-shape':       inv_cdf(np.abs(2 * u - 1)),
        }

    c_armijo = 1e-4
    theta_min = 1e-8

    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"    {name}...", end='', flush=True)

        a_np = np.clip(a0.copy(), x_lo + 1e-4, x_hi - 1e-4)

        # Project every init to monotone, then convert to d-space
        if enforce_monotone:
            a_np = _isotonic_projection(a_np)
            a_np = np.clip(a_np, x_lo + 1e-4, x_hi - 1e-4)
            d_t = _to(_a_to_d(a_np))
        else:
            d_t = None  # unused below; work directly with a_t

        a_t = _to(a_np)
        hist = {'obj': [], 'cv_inf': []}

        # Initialize dual variables via least-squares KKT estimate
        grad_J0 = grad_obj_gpu(a_t, plm, cost_dy_func)
        Jc0 = jac_hist_gpu(a_t, plm, target_density, n_bins, y_range)
        S0 = Jc0 @ Jc0.T + 1e-5 * torch.eye(m, dtype=_D, device=DEVICE)
        try:
            omega = -torch.linalg.solve(S0, Jc0 @ grad_J0)
        except Exception:
            omega = torch.zeros(m, dtype=_D, device=DEVICE)

        sigma_merit = max(1000.0, 2.0 * omega.abs().max().item())

        t0 = time.time()
        for outer in range(max_outer):
            # --- Recover a from d (if monotone reparam active) ---
            if enforce_monotone:
                a_t = _d_to_a(d_t).clamp(x_lo + 1e-4, x_hi - 1e-4)

            # --- Evaluate in a-space ---
            tau = pushforward_hist_gpu(
                a_t, plm, target_density, n_bins, y_range)
            J_val = objective_gpu(a_t, plm, cost_torch_func)
            grad_J_a = grad_obj_gpu(a_t, plm, cost_dy_func)
            Jc_a = jac_hist_gpu(a_t, plm, target_density, n_bins, y_range)
            H_a_diag = hessian_diag_gpu(a_t, plm, cost_dyy_func)
            H_a_diag = torch.clamp(H_a_diag, min=0.1)

            if enforce_monotone:
                # --- Transform to d-space via chain rule ---
                Jr = _reparam_jac(d_t)             # (n1, n1) lower tri
                grad_J_d = Jr.T @ grad_J_a         # (n1,)
                Jc_d = Jc_a @ Jr                    # (m, n1)
                grad_L_d = grad_J_d + Jc_d.T @ omega
                # Diagonal approximation of H_d = Jr^T diag(H_a) Jr
                H_d_diag = (Jr * Jr).T @ H_a_diag  # (n1,)
                H_d_diag = torch.clamp(H_d_diag, min=0.1)

                Jc_work = Jc_d
                grad_L_work = grad_L_d
                H_work = H_d_diag
            else:
                grad_L_work = grad_J_a + Jc_a.T @ omega
                Jc_work = Jc_a
                H_work = H_a_diag

            # --- Solve KKT via Schur complement ---
            H_inv = 1.0 / H_work
            Jc_Hinv = Jc_work * H_inv[None, :]
            S = Jc_Hinv @ Jc_work.T
            S += 1e-5 * torch.eye(m, dtype=_D, device=DEVICE)
            rhs_schur = tau - Jc_Hinv @ grad_L_work

            try:
                d_omega = torch.linalg.solve(S, rhs_schur)
            except Exception:
                d_omega = torch.linalg.lstsq(
                    S, rhs_schur.unsqueeze(1)).solution.squeeze(1)

            step = H_inv * (-grad_L_work - Jc_work.T @ d_omega)

            # --- Record ---
            cvinf = tau.abs().max().item()
            hist['obj'].append(J_val.item())
            hist['cv_inf'].append(cvinf)

            if cvinf < tol_cv and grad_L_work.abs().max().item() < tol_cv:
                break

            # Stagnation detection: if cv hasn't improved in 20 iters, stop
            if len(hist['cv_inf']) > 20:
                cv_20_ago = hist['cv_inf'][-20]
                if cvinf >= 0.95 * cv_20_ago:
                    break

            # --- l1 exact penalty merit ---
            omega_new_max = (omega + d_omega).abs().max().item()
            sigma_merit = max(sigma_merit, 1.5 * omega_new_max)
            M_val = J_val.item() + sigma_merit * tau.abs().sum().item()

            # Directional derivative of l1 merit along SQP step
            if enforce_monotone:
                grad_J_for_merit = Jr.T @ grad_J_a
            else:
                grad_J_for_merit = grad_J_a
            D_merit = (grad_J_for_merit @ step).item() \
                - sigma_merit * tau.abs().sum().item()

            # --- Armijo line search ---
            theta = 1.0
            for _ in range(40):
                if enforce_monotone:
                    d_trial = d_t + theta * step
                    a_trial = _d_to_a(d_trial).clamp(
                        x_lo + 1e-4, x_hi - 1e-4)
                else:
                    a_trial = (a_t + theta * step).clamp(
                        x_lo + 1e-4, x_hi - 1e-4)

                tau_trial = pushforward_hist_gpu(
                    a_trial, plm, target_density, n_bins, y_range)
                J_trial = objective_gpu(
                    a_trial, plm, cost_torch_func).item()
                M_trial = J_trial + sigma_merit * tau_trial.abs().sum().item()

                if M_trial <= M_val + c_armijo * theta * D_merit:
                    break
                theta *= 0.5
                if theta < theta_min:
                    break

            # --- Update ---
            if enforce_monotone:
                d_t = d_t + theta * step
                a_t = _d_to_a(d_t).clamp(x_lo + 1e-4, x_hi - 1e-4)
            else:
                a_t = (a_t + theta * step).clamp(x_lo + 1e-4, x_hi - 1e-4)
            omega = omega + theta * d_omega

        # --- Final eval ---
        elapsed = time.time() - t0
        if enforce_monotone:
            a_t = _d_to_a(d_t).clamp(x_lo + 1e-4, x_hi - 1e-4)
        tau = pushforward_hist_gpu(a_t, plm, target_density, n_bins, y_range)
        cvinf = tau.abs().max().item()
        J_val_f = objective_gpu(a_t, plm, cost_torch_func).item()
        a_final = _np(a_t)

        ds = np.diff(a_final)
        n_neg = np.sum(ds < -1e-6)
        n_pos = np.sum(ds > 1e-6)
        if n_neg == 0:
            mono = 'monotone ↑'
        elif n_pos == 0:
            mono = 'monotone ↓'
        else:
            mono = f'non-monotone ({n_neg}↓ {n_pos}↑)'

        if verbose:
            print(f"  J={J_val_f:.5f}, cv={cvinf:.2e}, {mono}  [{elapsed:.1f}s]")

        all_results[name] = {
            'a': a_final, 'hist': hist,
            'J': J_val_f, 'cv': cvinf, 'mono': mono,
        }

    best_name = min(all_results,
                    key=lambda k: all_results[k]['J']
                    if all_results[k]['cv'] < 1e-3 else 1e10)
    return all_results, best_name


# =========================================================================
# Exact solution (quantile coupling) and Kantorovich LP
# =========================================================================

def exact_monotone_map(target_density, x_range=(-1, 1), n_pts=2000):
    a, b = x_range
    x_grid = np.linspace(a, b, n_pts)
    F_mu = (x_grid - a) / (b - a)
    inv_cdf = target_cdf_inv(target_density, lo=a, hi=b)
    s_exact = inv_cdf(F_mu)
    return x_grid, s_exact, inv_cdf


def exact_objective(cost_np_func, target_density, x_range=(-1, 1),
                    n_pts=4000, plm=None):
    """Compute exact optimal cost.

    If plm is given, evaluate using the PL interpolant of the quantile
    coupling (same discretisation as the SQP solver) so the comparison
    is apples-to-apples.  Otherwise use the smooth quantile map.
    """
    if plm is not None:
        # PL-discretised exact map
        _, _, inv_cdf = exact_monotone_map(target_density, x_range)
        u = (plm.x_grid - x_range[0]) / (x_range[1] - x_range[0])
        a_exact = inv_cdf(u)                          # knot values
        x_fine = np.linspace(x_range[0], x_range[1], n_pts)
        s_fine = plm.eval_np(a_exact, x_fine)         # PL interpolation
        dx = (x_range[1] - x_range[0]) / (n_pts - 1)
        f_x = 1.0 / (x_range[1] - x_range[0])
        return np.sum(cost_np_func(x_fine, s_fine) * f_x * dx)
    else:
        x_grid, s_exact, _ = exact_monotone_map(
            target_density, x_range, n_pts)
        dx = (x_range[1] - x_range[0]) / (n_pts - 1)
        f_x = 1.0 / (x_range[1] - x_range[0])
        return np.sum(cost_np_func(x_grid, s_exact) * f_x * dx)


def kant_lp(cost_np_func, target_density, n_disc=100, x_range=(-1, 1)):
    x = np.linspace(x_range[0], x_range[1], n_disc)
    y = np.linspace(x_range[0], x_range[1], n_disc)
    dx = (x_range[1] - x_range[0]) / (n_disc - 1)
    p = np.full(n_disc, dx / (x_range[1] - x_range[0]))
    p /= p.sum()
    q = target_density(y) * dx
    q /= q.sum()
    C = cost_np_func(x[:, None], y[None, :]).flatten()
    A = lil_matrix((2 * n_disc, n_disc ** 2))
    for i in range(n_disc):
        A[i, i * n_disc:(i + 1) * n_disc] = 1.0
    for j in range(n_disc):
        A[n_disc + j, j::n_disc] = 1.0
    res = linprog(C, A_eq=A.tocsc(), b_eq=np.concatenate([p, q]),
                  bounds=(0, None), method='highs')
    return res.fun if res.success else None


# =========================================================================
# S^1 Reflector Antenna helpers
# =========================================================================

S1_DELTA = 0.15
S1_SRC_LO = S1_DELTA
S1_SRC_HI = np.pi - S1_DELTA
S1_TGT_LO = np.pi + S1_DELTA
S1_TGT_HI = 2 * np.pi - S1_DELTA


def s1_normalize(g_func, lo, hi):
    t = np.linspace(lo, hi, 10000)
    dt = (hi - lo) / 9999
    Z = np.sum(g_func(t)) * dt
    return lambda th, g=g_func, z=Z: g(th) / z


def s1_target_ring(theta):
    mid = (S1_TGT_LO + S1_TGT_HI) / 2
    return (0.5 * np.exp(-((theta - (mid - 0.5)) / 0.25) ** 2) +
            0.5 * np.exp(-((theta - (mid + 0.5)) / 0.25) ** 2) + 0.05)

def s1_target_concentrated(theta):
    mid = (S1_TGT_LO + S1_TGT_HI) / 2
    return np.exp(-((theta - mid) / 0.2) ** 2) + 0.02

def s1_target_asymmetric(theta):
    mid = (S1_TGT_LO + S1_TGT_HI) / 2
    return 0.6 * np.exp(-((theta - (mid + 0.3)) / 0.25) ** 2) + 0.15


S1_TARGETS_RAW = {
    'ring': s1_target_ring,
    'concentrated': s1_target_concentrated,
    'asymmetric': s1_target_asymmetric,
}


def s1_solve_sqp_gpu(g_func, a0_np, n_pieces=40, n_bins=20, max_outer=200,
                     tol_cv=1e-3):
    """SQP solver for reflector antenna on S^1, GPU accelerated."""
    n1 = n_pieces + 1
    m = n_bins
    xg = np.linspace(S1_SRC_LO, S1_SRC_HI, n1)
    src_len = S1_SRC_HI - S1_SRC_LO
    f_density = 1.0 / src_len
    n_quad = 3000
    n_fine = 3000

    plm_s1 = PLMap(n_pieces, x_range=(S1_SRC_LO, S1_SRC_HI))

    plm_s1_local = plm_s1  # for closures

    def s1_obj(a_t_):
        x_pts = torch.linspace(S1_SRC_LO, S1_SRC_HI, n_quad,
                                dtype=_D, device=DEVICE)
        dx = src_len / (n_quad - 1)
        sv = plm_s1_local.eval_torch(a_t_, x_pts)
        return (f_density * cost_log_reflector_s1_torch(x_pts, sv) * dx).sum()

    def s1_grad(a_t_):
        return grad_obj_gpu(a_t_, plm_s1_local, cost_log_reflector_dy,
                            n_quad=n_quad)

    def s1_hess(a_t_):
        H = hessian_diag_gpu(a_t_, plm_s1_local, cost_log_reflector_dyy,
                              n_quad=n_quad)
        return torch.clamp(H, min=1.0)

    def s1_pfhist(a_t_):
        x_pts = torch.linspace(S1_SRC_LO, S1_SRC_HI, n_fine,
                                dtype=_D, device=DEVICE)
        dx = src_len / (n_fine - 1)
        sv = plm_s1_local.eval_torch(a_t_, x_pts)
        be = torch.linspace(S1_TGT_LO, S1_TGT_HI, m + 1,
                             dtype=_D, device=DEVICE)
        tau = torch.zeros(m, dtype=_D, device=DEVICE)
        for b in range(m):
            mask = (sv >= be[b]) & (sv < be[b + 1])
            sm = mask.sum().to(_D) * dx * f_density
            ym = np.linspace(float(be[b]), float(be[b + 1]), 100)
            tm = np.sum(g_func(ym)) * (float(be[b + 1]) - float(be[b])) / 99
            tau[b] = sm - tm
        return tau

    def s1_jac(a_t_, eps=5e-7):
        t0 = s1_pfhist(a_t_)
        J = torch.zeros(m, n1, dtype=_D, device=DEVICE)
        for j in range(n1):
            ap = a_t_.clone()
            ap[j] += eps
            J[:, j] = (s1_pfhist(ap) - t0) / eps
        return J

    # Project init to monotone, convert to d-space
    a_np_init = _isotonic_projection(np.clip(a0_np, S1_TGT_LO, S1_TGT_HI))
    d_t = _to(_a_to_d(a_np_init))
    a_t = _d_to_a(d_t).clamp(S1_TGT_LO, S1_TGT_HI)

    c_armijo = 1e-4
    theta_min = 1e-8
    hist = {'obj': [], 'cv_inf': []}

    # Initialize dual variables via least-squares KKT
    grad_J0 = s1_grad(a_t)
    Jc0 = s1_jac(a_t)
    S0 = Jc0 @ Jc0.T + 1e-5 * torch.eye(m, dtype=_D, device=DEVICE)
    try:
        omega = -torch.linalg.solve(S0, Jc0 @ grad_J0)
    except Exception:
        omega = torch.zeros(m, dtype=_D, device=DEVICE)

    sigma_merit = max(1000.0, 2.0 * omega.abs().max().item())

    t_start = time.time()
    for outer in range(max_outer):
        a_t = _d_to_a(d_t).clamp(S1_TGT_LO, S1_TGT_HI)

        tau = s1_pfhist(a_t)
        J_val = s1_obj(a_t)
        grad_J_a = s1_grad(a_t)
        Jc_a = s1_jac(a_t)
        H_a_diag = s1_hess(a_t)

        # Transform to d-space
        Jr = _reparam_jac(d_t)
        grad_J_d = Jr.T @ grad_J_a
        Jc_d = Jc_a @ Jr
        grad_L_d = grad_J_d + Jc_d.T @ omega
        H_d_diag = (Jr * Jr).T @ H_a_diag
        H_d_diag = torch.clamp(H_d_diag, min=0.1)

        # Schur complement KKT solve in d-space
        H_inv = 1.0 / H_d_diag
        Jc_Hinv = Jc_d * H_inv[None, :]
        S = Jc_Hinv @ Jc_d.T + 1e-5 * torch.eye(m, dtype=_D, device=DEVICE)
        rhs_s = tau - Jc_Hinv @ grad_L_d

        try:
            d_omega = torch.linalg.solve(S, rhs_s)
        except Exception:
            d_omega = torch.linalg.lstsq(
                S, rhs_s.unsqueeze(1)).solution.squeeze(1)

        step = H_inv * (-grad_L_d - Jc_d.T @ d_omega)

        cvinf = tau.abs().max().item()
        hist['obj'].append(J_val.item())
        hist['cv_inf'].append(cvinf)

        if cvinf < tol_cv and grad_L_d.abs().max().item() < tol_cv:
            break

        # Stagnation detection
        if len(hist['cv_inf']) > 20:
            cv_20_ago = hist['cv_inf'][-20]
            if cvinf >= 0.95 * cv_20_ago:
                break

        omega_new_max = (omega + d_omega).abs().max().item()
        sigma_merit = max(sigma_merit, 1.5 * omega_new_max)
        M_val = J_val.item() + sigma_merit * tau.abs().sum().item()

        # Directional derivative of l1 merit along SQP step
        D_merit = (grad_J_d @ step).item() \
            - sigma_merit * tau.abs().sum().item()

        theta = 1.0
        for _ in range(40):
            d_trial = d_t + theta * step
            a_trial = _d_to_a(d_trial).clamp(S1_TGT_LO, S1_TGT_HI)

            tau_trial = s1_pfhist(a_trial)
            J_trial = s1_obj(a_trial).item()
            M_trial = J_trial + sigma_merit * tau_trial.abs().sum().item()

            if M_trial <= M_val + c_armijo * theta * D_merit:
                break
            theta *= 0.5
            if theta < theta_min:
                break

        d_t = d_t + theta * step
        omega = omega + theta * d_omega

    elapsed = time.time() - t_start
    a_t = _d_to_a(d_t).clamp(S1_TGT_LO, S1_TGT_HI)
    a_final = _np(a_t)
    tau_f = s1_pfhist(a_t)
    cvinf_f = tau_f.abs().max().item()
    J_f = s1_obj(a_t).item()
    mono = 'mono ↑' if np.all(np.diff(a_final) >= -1e-6) else 'non-mono'
    return a_final, J_f, cvinf_f, mono, hist, elapsed


# =========================================================================
# MAIN
# =========================================================================

def main():
    args = parse_args()
    global DEVICE
    DEVICE = setup_device(args.device)

    print("=" * 70)
    print("  BEYOND BRENIER: SQP for General-Cost Monge OT (GPU)")
    print("  Source: Uniform[-1,1]  Targets: ring/concentrated/asymmetric")
    print("=" * 70)

    # ---- CONFIGURATION ----
    n_pieces = 80
    n_bins = 40
    D_lens = 1.0
    targets = ['ring', 'concentrated', 'asymmetric']

    costs = {
        'quadratic': {
            'np': cost_quadratic_np,
            'torch': cost_quadratic_torch,
            'dy': cost_quadratic_dy,
            'dyy': cost_quadratic_dyy,
        },
        f'lens D={D_lens}': {
            'np': lambda x, y: cost_lens_np(x, y, D=D_lens),
            'torch': lambda x, y: cost_lens_torch(x, y, D=D_lens),
            'dy': lambda x, y: cost_lens_dy(x, y, D=D_lens),
            'dyy': lambda x, y: cost_lens_dyy(x, y, D=D_lens),
        },
    }
    cost_colors = {
        'quadratic': '#2196F3',
        f'lens D={D_lens}': '#E91E63',
    }

    all_results = {}

    for tname in targets:
        print(f"\n{'=' * 70}")
        print(f"  Target: {tname}")
        print(f"{'=' * 70}")

        x_range, target_density = make_source_target(tname)
        plm = PLMap(n_pieces, x_range)

        all_results[tname] = {}

        # Exact solution (quantile coupling)
        x_exact, s_exact, _ = exact_monotone_map(target_density, x_range)
        all_results[tname]['_exact'] = {'x': x_exact, 's': s_exact}
        print(f"  Exact monotone map computed (quantile coupling)")

        for cname, cfuncs in costs.items():
            print(f"\n  Cost: {cname}")

            ar, best = solve_sqp_gpu(
                plm, target_density,
                cfuncs['torch'], cfuncs['dy'], cfuncs['dyy'],
                n_bins=n_bins, y_range=x_range,
                enforce_monotone=True, verbose=True)

            K = kant_lp(cfuncs['np'], target_density, n_disc=100,
                        x_range=x_range)
            J_exact = exact_objective(cfuncs['np'], target_density, x_range,
                                     plm=plm)

            all_results[tname][cname] = {
                'all_res': ar, 'best_name': best, 'K': K, 'plm': plm,
                'J_exact': J_exact,
            }

            best_J = ar[best]['J']
            print(f"  Best: {best}, J={best_J:.5f}, "
                  f"Kant={K:.5f}" if K else f"  Best: {best}, J={best_J:.5f}")
            print(f"  Exact (quantile coupling): J*={J_exact:.5f}")
            if K:
                gap_exact = best_J - J_exact
                print(f"  Gap to exact: {gap_exact:.5f} "
                      f"({100 * abs(gap_exact) / abs(J_exact):.2f}%)")

    # =====================================================================
    # Fig 1: Best maps for each cost, one column per target
    # =====================================================================
    print("\nGenerating figures...")
    x_fine = np.linspace(-1, 1, 1000)

    fig, axes = plt.subplots(2, len(targets), figsize=(6 * len(targets), 10))

    for col, tname in enumerate(targets):
        ax_map = axes[0, col]
        ax_push = axes[1, col]
        _, target_density = make_source_target(tname)

        for cname in costs:
            r = all_results[tname][cname]
            plm_p = r['plm']
            best = r['best_name']
            a = r['all_res'][best]['a']
            J = r['all_res'][best]['J']
            J_ex = r['J_exact']

            s_vals = plm_p.eval_np(a, x_fine)
            label = f'{cname}: J={J:.4f} (exact={J_ex:.4f})'
            ax_map.plot(x_fine, s_vals, color=cost_colors[cname],
                        lw=2, label=label)

            n_check = 5000
            x_check = np.linspace(-1, 1, n_check)
            s_check = plm_p.eval_np(a, x_check)
            ax_push.hist(s_check, bins=50, density=True, alpha=0.3,
                         color=cost_colors[cname])

        ex = all_results[tname]['_exact']
        ax_map.plot(ex['x'], ex['s'], 'k--', lw=1.5, alpha=0.7,
                    label='exact (quantile coupling)', zorder=0)
        ax_map.plot([-1, 1], [-1, 1], ':', color='gray', lw=0.8, alpha=0.4)

        yd = np.linspace(-1, 1, 500)
        ax_push.plot(yd, target_density(yd), 'k-', lw=2.5,
                     label='Target $g(y)$')

        ax_map.set_title(f'Target: {tname}', fontsize=13)
        ax_map.set_xlabel('Source $x$', fontsize=11)
        ax_map.set_ylabel('$s(x)$', fontsize=11)
        ax_map.legend(fontsize=7, loc='best')
        ax_map.grid(True, alpha=0.3)

        ax_push.set_xlabel('$y$', fontsize=11)
        ax_push.set_ylabel('Density', fontsize=11)
        ax_push.set_title(f'Pushforward check', fontsize=11)
        ax_push.legend(fontsize=8)
        ax_push.grid(True, alpha=0.3)

    plt.suptitle('1D Freeform Lens: Optimal Maps (SQP solver, GPU)\n'
                 'Source: Uniform$[-1,1]$', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig1_lens_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig1_lens_maps.png")

    # =====================================================================
    # S^1 REFLECTOR ANTENNA
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  REFLECTOR ANTENNA ON S^1 (SQP, GPU)")
    print(f"  c(theta_x, theta_y) = -log|x - y|")
    print(f"  Source arc: [{S1_DELTA:.2f}, pi-{S1_DELTA:.2f}], "
          f"Target arc: [pi+{S1_DELTA:.2f}, 2pi-{S1_DELTA:.2f}]")
    print(f"{'=' * 70}")

    s1_targets = {k: s1_normalize(v, S1_TGT_LO, S1_TGT_HI)
                  for k, v in S1_TARGETS_RAW.items()}
    s1_n = 80
    s1_xg = np.linspace(S1_SRC_LO, S1_SRC_HI, s1_n + 1)
    s1_results = {}

    for tname, g_func in s1_targets.items():
        print(f"\n  Target: {tname}")
        inv_cdf = target_cdf_inv(g_func, lo=S1_TGT_LO, hi=S1_TGT_HI,
                                  n_inv=20000)
        u = (s1_xg - S1_SRC_LO) / (S1_SRC_HI - S1_SRC_LO)
        a0 = inv_cdf(u)

        # Exact solution — use PL interpolant for apples-to-apples comparison
        plm_s1 = PLMap(s1_n, x_range=(S1_SRC_LO, S1_SRC_HI))
        a_exact_s1 = inv_cdf(u)                           # knot values
        x_ex = np.linspace(S1_SRC_LO, S1_SRC_HI, 2000)
        s_exact = plm_s1.eval_np(a_exact_s1, x_ex)        # PL interpolation
        f_src = 1.0 / (S1_SRC_HI - S1_SRC_LO)
        J_exact = trapezoid(f_src * cost_log_reflector_s1_np(x_ex, s_exact),
                            x_ex)

        a_sol, J_sol, cv, mono, hist, elapsed = s1_solve_sqp_gpu(
            g_func, a0, n_pieces=s1_n, n_bins=40)
        gap = 100 * abs(J_sol - J_exact) / abs(J_exact) if J_exact != 0 else 0
        print(f"    SQP:   J={J_sol:.6f}, cv={cv:.2e}, {mono}  [{elapsed:.1f}s]")
        print(f"    Exact: J={J_exact:.6f}, gap={gap:.2f}%")

        s1_results[tname] = {
            'a': a_sol, 'J': J_sol, 'cv': cv, 'mono': mono, 'hist': hist,
            'J_exact': J_exact, 'x_exact': x_ex, 's_exact': s_exact,
            'g_func': g_func, 'inv_cdf': inv_cdf,
        }

    # ---- Fig 6: Reflector antenna on S^1 ----
    fig, axes = plt.subplots(1, len(s1_targets) + 1,
                             figsize=(5 * (len(s1_targets) + 1), 5))

    # Panel 0: Circle geometry
    ax = axes[0]
    theta_circle = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', lw=0.5,
            alpha=0.3)
    ts = np.linspace(S1_SRC_LO, S1_SRC_HI, 200)
    ax.plot(np.cos(ts), np.sin(ts), '-', color='#2196F3', lw=5, alpha=0.4,
            label='source')
    tt = np.linspace(S1_TGT_LO, S1_TGT_HI, 200)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='#E91E63', lw=5, alpha=0.4,
            label='target')

    r0 = s1_results['ring']
    plm_s1_plot = PLMap(s1_n, x_range=(S1_SRC_LO, S1_SRC_HI))
    n_show = 15
    xs = np.linspace(S1_SRC_LO, S1_SRC_HI, n_show)
    ys = plm_s1_plot.eval_np(r0['a'], xs)
    for k in range(n_show):
        ax.plot([np.cos(xs[k]), np.cos(ys[k])],
                [np.sin(xs[k]), np.sin(ys[k])],
                '-', color='#059669', alpha=0.35, lw=0.8)
    ax.set_aspect('equal')
    ax.set_title('$S^1$ reflector:\ntransport rays', fontsize=12)
    ax.legend(fontsize=8, loc='center')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.2)

    # Panels 1-3: Maps
    for col, tname in enumerate(s1_targets):
        ax = axes[col + 1]
        r = s1_results[tname]
        ax.plot(r['x_exact'], r['s_exact'], 'k--', lw=1.5, alpha=0.6,
                label=f'exact: J={r["J_exact"]:.4f}')
        ax.plot(s1_xg, r['a'], '-', color='#4CAF50', lw=2.5,
                label=f'SQP: J={r["J"]:.4f}')
        gap = 100 * abs(r['J'] - r['J_exact']) / abs(r['J_exact'])
        ax.set_title(f'Target: {tname}\ngap={gap:.2f}%', fontsize=12)
        ax.set_xlabel('$\\theta_x$ (source)', fontsize=11)
        ax.set_ylabel('$\\theta_y = s(\\theta_x)$', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reflector Antenna on $S^1$: $c(\\theta_x, \\theta_y) = '
                 '-\\log|x - y|$ (SQP, GPU)\n'
                 'Source: upper arc, Target: lower arc',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig('fig6_reflector_s1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig6_reflector_s1.png")

    # =====================================================================
    # Verification tables
    # =====================================================================
    print(f"\n{'=' * 100}")
    print("  tab:lens-verification")
    print(f"  {'Target':>14} {'Cost':>16} {'Best init':>14} "
          f"{'SQP J':>10} {'Exact J*':>10} {'Kant LB':>10} "
          f"{'SQP gap%':>8} {'Mono':>20}")
    print(f"{'-' * 100}")
    for tname in targets:
        for cname in costs:
            r = all_results[tname][cname]
            best = r['best_name']
            res = r['all_res'][best]
            K = r['K']
            J_ex = r['J_exact']
            gap_pct = (100 * abs(res['J'] - J_ex) / abs(J_ex)
                       if J_ex and J_ex != 0 else 0)
            K_str = f"{K:10.5f}" if K else "     N/A  "
            print(f"  {tname:>14} {cname:>16} {best:>14} "
                  f"{res['J']:10.5f} {J_ex:10.5f} {K_str} "
                  f"{gap_pct:7.2f}% {res['mono']:>20}")
    print(f"{'=' * 100}")

    print(f"\n  tab:reflector-s1")
    print(f"  {'Target':>14} {'SQP J':>10} {'Exact J*':>10} {'Gap%':>8}")
    print(f"  {'-' * 50}")
    for tname in s1_results:
        r = s1_results[tname]
        gap = (100 * abs(r['J'] - r['J_exact']) / abs(r['J_exact'])
               if r['J_exact'] != 0 else 0)
        print(f"  {tname:>14} {r['J']:10.5f} {r['J_exact']:10.5f} "
              f"{gap:7.2f}%")

    print("\nDone. Figures saved: fig1_lens_maps.png, fig6_reflector_s1.png")


if __name__ == '__main__':
    main()
