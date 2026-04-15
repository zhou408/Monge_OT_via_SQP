"""
Beyond Brenier: Coulomb Optimal Transport (Strictly Correlated Electrons)
=========================================================================

PHYSICS:
  In the strong-interaction limit of Density Functional Theory (DFT),
  the ground state of N electrons with density rho(x) is described by
  an optimal transport problem with Coulomb repulsive cost:

    min_{f: f#(rho/N) = rho/N}  int (1/|x - f(x)|) * rho(x)/N dx

  For N=2 electrons in 1D, this reduces to finding a map f pushing
  rho/2 to itself (self-transport) that minimizes the Coulomb repulsion.

  The exact solution is the co-motion function (Seidl 1999):
    f(x) = F^{-1}(1 - F(x))
  where F is the CDF of rho. This is the ANTI-QUANTILE coupling:
  monotone decreasing, mapping each electron to its "opposite" in
  the density distribution.

WHY GANGBO-MCCANN DOESN'T APPLY:
  1. The cost 1/|x-y| is singular (blows up at x=y)
  2. It is REPULSIVE (decreasing in distance) — opposite of convex costs
  3. c_xy = 2/|x-y|^3 > 0 (wrong sign for standard twist)
  4. Self-transport (mu = nu) is degenerate

  The SQP framework doesn't need these conditions: it finds the map
  by direct optimization, using the Kantorovich LP as a certificate.

SETUP:
  - Density: bimodal (diatomic molecule) with varying atom separation d
  - This ensures min|x - f(x)| > 0, avoiding the Coulomb singularity
  - Physically: electrons on atom A map to atom B and vice versa

  Three test densities:
    (A) Tight molecule:   d=2.5, sigma=0.4 (harder, singularity closer)
    (B) Medium molecule:  d=3.5, sigma=0.4
    (C) Wide molecule:    d=5.0, sigma=0.3 (easiest, well separated)

Dependencies: numpy, scipy, matplotlib
Usage: python experiment_coulomb_sce.py
"""

import numpy as np
from scipy.optimize import linprog
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.sparse import lil_matrix
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings('ignore')

# GPU / CPU device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_D = torch.float64
print(f"  [device] {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))


# =============================================================================
# Configuration
# =============================================================================

EPS_COULOMB = 1e-3      # Smoothing for 1/|x-y| singularity
N_PIECES    = 400       # PL map pieces
N_BINS      = 200       # Histogram bins for pushforward
N_QUAD      = 3000      # Quadrature for objective
N_FINE_PF   = 3000      # Fine grid for pushforward
MAX_OUTER   = 120       # Max SQP iterations
TOL_CV      = 5e-3      # Feasibility tolerance
N_KANT      = 100       # Kantorovich LP discretization

# Domain (large enough to capture tail of bimodal densities)
X_LO, X_HI = -6.0, 6.0


# =============================================================================
# Densities
# =============================================================================

def rho_bimodal(x, d=3.0, sigma=0.4):
    """Symmetric diatomic molecule: two Gaussian atoms at +/-d/2."""
    raw = np.exp(-((x - d/2) / sigma)**2) + np.exp(-((x + d/2) / sigma)**2)
    return raw

def normalize_density(rho_func, x_range=(X_LO, X_HI), n_pts=20000, **kwargs):
    """Return normalized density function."""
    x = np.linspace(x_range[0], x_range[1], n_pts)
    raw = rho_func(x, **kwargs)
    Z = trapezoid(raw, x)
    return lambda t, f=rho_func, z=Z, **kw: f(t, **kwargs) / z


# =============================================================================
# Exact co-motion function (Seidl 1999)
# =============================================================================

def exact_comotion(rho_normed, x_range=(X_LO, X_HI), n_pts=20000):
    """
    Compute exact SCE co-motion function f(x) = F^{-1}(1 - F(x)).

    Returns: x grid, f(x) values, CDF, inverse CDF, exact objective
    """
    x = np.linspace(x_range[0], x_range[1], n_pts)
    rho_vals = rho_normed(x)

    # CDF
    dx = (x_range[1] - x_range[0]) / (n_pts - 1)
    cdf = np.cumsum(rho_vals) * dx
    cdf /= cdf[-1]
    cdf[0] = 0.0

    # Inverse CDF
    F_inv = interp1d(cdf, x, bounds_error=False,
                     fill_value=(x_range[0], x_range[1]))

    # Anti-quantile coupling
    f_exact = F_inv(1.0 - cdf)

    # Exact objective
    dists = np.abs(x - f_exact)
    cost_vals = 1.0 / (dists + EPS_COULOMB)
    J_exact = trapezoid(0.5 * rho_vals * cost_vals, x)

    # Minimum distance (on support)
    mask = rho_vals > 0.01 * rho_vals.max()
    min_dist = dists[mask].min() if mask.any() else 0.0

    return x, f_exact, cdf, F_inv, J_exact, min_dist


# =============================================================================
# Cost function
# =============================================================================

def cost_coulomb(x, y, eps=EPS_COULOMB):
    """Smoothed Coulomb cost: 1 / (|x-y| + eps)."""
    return 1.0 / (np.abs(x - y) + eps)


def cost_coulomb_dy(x, y, eps=EPS_COULOMB):
    """D_y c = d/dy [1/(|x-y|+eps)] = sign(x-y) / (|x-y|+eps)^2."""
    d = x - y
    return np.sign(d) / (np.abs(d) + eps)**2


def cost_coulomb_dyy(x, y, eps=EPS_COULOMB):
    """D^2_yy c = 2 / (|x-y|+eps)^3.  Always positive."""
    return 2.0 / (np.abs(x - y) + eps)**3


# =============================================================================
# Piecewise linear map
# =============================================================================

class PLMap:
    def __init__(self, n, x_range=(X_LO, X_HI)):
        self.n = n
        self.x_grid = np.linspace(x_range[0], x_range[1], n + 1)

    def eval(self, a, x):
        return np.interp(x, self.x_grid, a)


# =============================================================================
# GPU-accelerated SQP components for Coulomb self-transport
# =============================================================================

class CoulombSQP:
    """
    PyTorch-accelerated computations for the Coulomb SCE experiment.

    Precomputes hat-function matrices on device so that gradient, Hessian,
    pushforward, and Jacobian evaluations are fully vectorized.
    """

    def __init__(self, plm, rho_func, device=DEVICE):
        self.dev = device
        self.plm = plm
        self.n1 = plm.n + 1

        x_lo, x_hi = float(plm.x_grid[0]), float(plm.x_grid[-1])
        self.x_lo, self.x_hi, self.L = x_lo, x_hi, x_hi - x_lo

        # Quadrature grids  (persistent on device)
        self.xq = torch.linspace(x_lo, x_hi, N_QUAD, dtype=_D, device=device)
        self.dxq = self.L / (N_QUAD - 1)
        self.rho_q = torch.as_tensor(
            rho_func(np.linspace(x_lo, x_hi, N_QUAD)),
            dtype=_D, device=device)

        # Pushforward grid
        self.xpf = torch.linspace(x_lo, x_hi, N_FINE_PF, dtype=_D, device=device)
        self.dxpf = self.L / (N_FINE_PF - 1)
        self.rho_pf = torch.as_tensor(
            rho_func(np.linspace(x_lo, x_hi, N_FINE_PF)),
            dtype=_D, device=device)

        # Bin edges and precomputed target mass per bin
        self.bin_edges = torch.linspace(x_lo, x_hi, N_BINS + 1, dtype=_D, device=device)
        target_mass = torch.zeros(N_BINS, dtype=_D, device=device)
        be_np = np.linspace(x_lo, x_hi, N_BINS + 1)
        for b in range(N_BINS):
            ym = np.linspace(be_np[b], be_np[b + 1], 100)
            target_mass[b] = trapezoid(rho_func(ym), ym)
        self.target_mass = target_mass

        # Precompute hat function matrix on quadrature grid: (n1, N_QUAD)
        xg = torch.as_tensor(plm.x_grid, dtype=_D, device=device)
        self.xg = xg
        self.Phi = self._build_hats(xg, self.xq)

    @staticmethod
    def _build_hats(xg, pts):
        """Build all hat functions on pts.  Returns (n1, len(pts)) tensor."""
        n1 = len(xg)
        phi = torch.zeros(n1, len(pts), dtype=_D, device=pts.device)
        for k in range(n1):
            if k > 0:
                h = xg[k] - xg[k - 1]
                if h > 0:
                    m = (pts >= xg[k - 1]) & (pts <= xg[k])
                    phi[k, m] = (pts[m] - xg[k - 1]) / h
            if k < n1 - 1:
                h = xg[k + 1] - xg[k]
                if h > 0:
                    m = (pts > xg[k]) & (pts <= xg[k + 1])
                    phi[k, m] = (xg[k + 1] - pts[m]) / h
        return phi

    # ---------- PL interpolation on GPU ----------
    def _pl_eval(self, a_t, pts):
        """PL map evaluation entirely on device.  a_t: (n1,) tensor."""
        sx = (pts - self.xg[0]) / (self.xg[-1] - self.xg[0]) * (self.n1 - 1)
        sx = sx.clamp(0, self.n1 - 1 - 1e-7)
        ix = sx.long()
        ix1 = (ix + 1).clamp(max=self.n1 - 1)
        frac = sx - ix.to(_D)
        return a_t[ix] * (1 - frac) + a_t[ix1] * frac

    # ---------- Objective ----------
    def objective(self, a_np):
        """J = int (1/2) rho(x) / (|x - s(x)| + eps) dx."""
        a_t = torch.as_tensor(a_np, dtype=_D, device=self.dev)
        sv = self._pl_eval(a_t, self.xq)
        c = 1.0 / (torch.abs(self.xq - sv) + EPS_COULOMB)
        return float((0.5 * self.rho_q * c).sum() * self.dxq)

    # ---------- Analytic gradient (Theorem 2 pattern) ----------
    def gradient(self, a_np):
        """grad[k] = int phi_k(x) * D_y c(x,s(x)) * rho(x)/2 dx."""
        a_t = torch.as_tensor(a_np, dtype=_D, device=self.dev)
        sv = self._pl_eval(a_t, self.xq)
        d = self.xq - sv
        cy = torch.sign(d) / (torch.abs(d) + EPS_COULOMB) ** 2
        w = cy * self.rho_q * 0.5 * self.dxq          # (N_QUAD,)
        return (self.Phi @ w).cpu().numpy()             # (n1,)

    # ---------- Diagonal Hessian (Theorem 3 pattern) ----------
    def hessian_diag(self, a_np):
        """H[k] = int phi_k(x)^2 * D^2_yy c(x,s(x)) * rho(x)/2 dx."""
        a_t = torch.as_tensor(a_np, dtype=_D, device=self.dev)
        sv = self._pl_eval(a_t, self.xq)
        cyy = 2.0 / (torch.abs(self.xq - sv) + EPS_COULOMB) ** 3
        w = cyy * self.rho_q * 0.5 * self.dxq
        H = (self.Phi ** 2) @ w                        # (n1,)
        # Floor at 1.0 to prevent huge steps in density tails where rho≈0
        return torch.clamp(H, min=1.0).cpu().numpy()

    # ---------- Pushforward constraint (vectorized) ----------
    def pushforward(self, a_np):
        """tau[b] = source_mass_in_bin_b - target_mass_in_bin_b."""
        a_t = torch.as_tensor(a_np, dtype=_D, device=self.dev)
        sv = self._pl_eval(a_t, self.xpf)
        # Bin index for each pushed-forward point
        bi = torch.bucketize(sv, self.bin_edges[1:-1])  # 0..N_BINS-1
        bi = bi.clamp(0, N_BINS - 1)
        # Weighted scatter
        src_mass = torch.zeros(N_BINS, dtype=_D, device=self.dev)
        src_mass.scatter_add_(0, bi, self.rho_pf * self.dxpf)
        return (src_mass - self.target_mass).cpu().numpy()

    # ---------- Jacobian via batched finite differences ----------
    def jacobian(self, a_np, fd_eps=5e-7):
        """Batched FD Jacobian: perturb all n1 knots in parallel."""
        a_t = torch.as_tensor(a_np, dtype=_D, device=self.dev)
        tau0 = torch.as_tensor(self.pushforward(a_np), dtype=_D, device=self.dev)

        # Build (n1, n1) perturbation matrix: each row is a + eps*e_j
        A_pert = a_t.unsqueeze(0).expand(self.n1, -1).clone()  # (n1, n1)
        A_pert[torch.arange(self.n1), torch.arange(self.n1)] += fd_eps

        # Evaluate pushforward for all perturbations in one batch
        Jac = torch.zeros(N_BINS, self.n1, dtype=_D, device=self.dev)
        for j in range(self.n1):
            tau_j = torch.as_tensor(
                self.pushforward(A_pert[j].cpu().numpy()),
                dtype=_D, device=self.dev)
            Jac[:, j] = (tau_j - tau0) / fd_eps

        return Jac.cpu().numpy()

    # ---------- KKT Schur complement solve ----------
    def solve_kkt(self, H_diag, Jc, grad_L, tau, reg=1e-6):
        """Solve KKT system via Schur complement on device."""
        H  = torch.as_tensor(H_diag, dtype=_D, device=self.dev)
        Jc_t = torch.as_tensor(Jc,   dtype=_D, device=self.dev)
        gL = torch.as_tensor(grad_L, dtype=_D, device=self.dev)
        t  = torch.as_tensor(tau,    dtype=_D, device=self.dev)

        Hi   = 1.0 / H
        JcHi = Jc_t * Hi[None, :]
        S    = JcHi @ Jc_t.T
        S   += reg * torch.eye(S.shape[0], dtype=_D, device=self.dev)
        rhs  = -t + JcHi @ gL

        try:
            do = torch.linalg.solve(S, rhs)
        except Exception:
            do = torch.linalg.lstsq(S, rhs.unsqueeze(1)).solution.squeeze(1)

        da = Hi * (-gL - Jc_t.T @ do)
        return da.cpu().numpy(), do.cpu().numpy()


# =============================================================================
# SQP solver for Coulomb self-transport
# =============================================================================

def solve_sqp(plm, rho_func, inits, verbose=True):
    """
    Solve the Coulomb SCE problem via Sequential Quadratic Programming.

    At each iteration, solves the KKT system of a QP subproblem:
        [ H_L    Jc' ] [ da ]   [ -grad_L ]
        [ Jc     0   ] [ dw ] = [ -tau    ]

    Line search on merit function M = L + (sigma/2)||tau||^2.
    """
    gpu = CoulombSQP(plm, rho_func)
    n1 = plm.n + 1
    m = N_BINS

    c_armijo = 1e-4
    theta_min = 1e-8
    sigma_merit = 200.0   # high initial penalty for repulsive Coulomb cost

    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"    {name}...", end='', flush=True)

        a = np.clip(a0.copy(), X_LO + 1e-4, X_HI - 1e-4)
        omega = np.zeros(m)
        hist = {'obj': [], 'cv_inf': []}

        t0 = time.time()
        sigma = sigma_merit

        for outer in range(MAX_OUTER):
            # --- Evaluate constraint and objective ---
            tau = gpu.pushforward(a)
            J_val = gpu.objective(a)

            # --- Gradient of objective (analytic) ---
            grad_J = gpu.gradient(a)

            # --- Constraint Jacobian (batched FD) ---
            Jc = gpu.jacobian(a)

            # --- Gradient of Lagrangian ---
            grad_L = grad_J + Jc.T @ omega

            # --- Hessian of Lagrangian (diagonal Gauss-Newton) ---
            H_diag = gpu.hessian_diag(a)

            # --- Record history ---
            cvinf = np.linalg.norm(tau, np.inf)
            hist['obj'].append(J_val)
            hist['cv_inf'].append(cvinf)

            # Check convergence
            if cvinf < TOL_CV and np.linalg.norm(grad_L, np.inf) < TOL_CV:
                break

            # --- Solve KKT system ---
            d_a, d_omega = gpu.solve_kkt(H_diag, Jc, grad_L, tau)

            # --- Armijo line search on merit function ---
            L_val = J_val + omega @ tau
            M_val = L_val + 0.5 * sigma * np.dot(tau, tau)
            dM = grad_L @ d_a + sigma * tau @ (Jc @ d_a)

            theta = 1.0
            a_lo, a_hi = X_LO + 1e-4, X_HI - 1e-4

            for _ in range(30):
                a_trial = np.clip(a + theta * d_a, a_lo, a_hi)
                omega_trial = omega + theta * d_omega

                tau_trial = gpu.pushforward(a_trial)
                J_trial = gpu.objective(a_trial)
                L_trial = J_trial + omega_trial @ tau_trial
                M_trial = L_trial + 0.5 * sigma * np.dot(tau_trial, tau_trial)

                if M_trial <= M_val + c_armijo * theta * dM:
                    break
                theta *= 0.5
                if theta < theta_min:
                    break

            # --- Update ---
            a = np.clip(a + theta * d_a, a_lo, a_hi)
            omega = omega + theta * d_omega

            # Adaptively increase sigma if constraint not decreasing
            if outer > 0 and cvinf > 0.9 * hist['cv_inf'][-2]:
                sigma = min(sigma * 2.0, 1e6)

        elapsed = time.time() - t0

        # Final evaluation
        tau = gpu.pushforward(a)
        cvinf = np.linalg.norm(tau, np.inf)
        J_val = gpu.objective(a)

        ds = np.diff(a)
        n_neg = np.sum(ds < -1e-6)
        n_pos = np.sum(ds > 1e-6)
        if n_neg == 0:
            mono = 'monotone +'
        elif n_pos == 0:
            mono = 'monotone -'
        else:
            mono = f'non-mono ({n_neg}d {n_pos}u)'

        if verbose:
            print(f"  J={J_val:.5f}, cv={cvinf:.2e}, {mono}  [{elapsed:.1f}s]")

        all_results[name] = {
            'a': a.copy(), 'hist': hist,
            'J': J_val, 'cv': cvinf, 'mono': mono,
        }

    best_name = min(all_results,
                    key=lambda k: all_results[k]['J']
                    if all_results[k]['cv'] < 0.1 else 1e10)
    return all_results, best_name


# =============================================================================
# Kantorovich LP lower bound
# =============================================================================

def kant_lp(rho_func, n_disc=N_KANT):
    """Kantorovich LP for self-transport with Coulomb cost."""
    x = np.linspace(X_LO, X_HI, n_disc)
    dx = (X_HI - X_LO) / (n_disc - 1)

    p = rho_func(x) * dx
    p /= p.sum()

    C = cost_coulomb(x[:, None], x[None, :]).flatten()

    n2 = n_disc * n_disc
    A_eq = lil_matrix((2 * n_disc, n2))
    for i in range(n_disc):
        A_eq[i, i*n_disc:(i+1)*n_disc] = 1.0
    for j in range(n_disc):
        A_eq[n_disc + j, j::n_disc] = 1.0

    b_eq = np.concatenate([p, p])

    try:
        res = linprog(C, A_eq=A_eq.tocsc(), b_eq=b_eq,
                      bounds=[(0, None)] * n2, method='highs')
        if res.success:
            return res.fun
    except Exception:
        pass
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 75)
    print("  COULOMB OPTIMAL TRANSPORT: Strictly Correlated Electrons")
    print("  c(x,y) = 1/|x-y|  (repulsive Coulomb cost)")
    print("  Self-transport: rho -> rho")
    print("  Exact solution: f(x) = F^{-1}(1 - F(x)) (anti-quantile)")
    print("=" * 75)

    densities = {
        'tight (d=2.5)':  {'d': 2.5, 'sigma': 0.4},
        'medium (d=3.5)': {'d': 3.5, 'sigma': 0.4},
        'wide (d=5.0)':   {'d': 5.0, 'sigma': 0.3},
    }

    all_data = {}

    for dname, params in densities.items():
        print(f"\n{'='*70}")
        print(f"  Density: {dname}")
        print(f"{'='*70}")

        rho_func = normalize_density(rho_bimodal, **params)

        x_ex, f_ex, cdf_ex, F_inv_ex, J_exact, min_dist = exact_comotion(rho_func)
        print(f"  Exact: J={J_exact:.6f}, min|x-f(x)|={min_dist:.4f}")

        plm = PLMap(N_PIECES)
        u_grid = plm.x_grid
        rho_grid = rho_func(u_grid)
        cdf_grid = np.cumsum(rho_grid) * (X_HI - X_LO) / N_PIECES
        cdf_grid /= cdf_grid[-1]; cdf_grid[0] = 0
        F_inv_grid = interp1d(cdf_grid, u_grid, bounds_error=False,
                              fill_value=(X_LO, X_HI))

        inits = {
            'anti-quantile': F_inv_grid(1.0 - cdf_grid),
            'quantile':      F_inv_grid(cdf_grid),
            'tent':          F_inv_grid(1.0 - np.abs(2*cdf_grid - 1)),
            'V-shape':       F_inv_grid(np.abs(2*cdf_grid - 1)),
        }

        ar, best = solve_sqp(plm, rho_func, inits, verbose=True)

        print(f"  Kantorovich LP...", end='', flush=True)
        K = kant_lp(rho_func)
        if K is not None:
            print(f"  K={K:.5f}")
        else:
            print(f"  failed")

        all_data[dname] = {
            'params': params, 'rho_func': rho_func,
            'x_exact': x_ex, 'f_exact': f_ex, 'J_exact': J_exact,
            'min_dist': min_dist, 'F_inv': F_inv_ex,
            'all_res': ar, 'best_name': best, 'K': K, 'plm': plm,
        }

    # =========================================================================
    # FIGURES
    # =========================================================================
    print(f"\n{'='*70}")
    print("  Generating figures...")
    print(f"{'='*70}")

    x_fine = np.linspace(X_LO, X_HI, 1000)
    init_colors = {
        'anti-quantile': '#E91E63', 'quantile': '#2196F3',
        'tent': '#9C27B0', 'V-shape': '#4CAF50',
    }
    dnames = list(densities.keys())

    # ---- Fig 1: Densities and co-motion functions ----
    fig, axes = plt.subplots(2, len(dnames), figsize=(6*len(dnames), 9))

    for col, dname in enumerate(dnames):
        d = all_data[dname]

        ax = axes[0, col]
        ax.plot(d['x_exact'], d['rho_func'](d['x_exact']), 'k-', lw=2)
        ax.fill_between(d['x_exact'], d['rho_func'](d['x_exact']),
                        alpha=0.15, color='steelblue')
        ax.set_xlabel('$x$', fontsize=11)
        ax.set_ylabel('$\\rho(x)$', fontsize=11)
        ax.set_title(f'{dname}\n$d={d["params"]["d"]}$, '
                     f'$\\sigma={d["params"]["sigma"]}$', fontsize=12)
        ax.set_xlim(-6, 6); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(d['x_exact'], d['f_exact'], 'k--', lw=2, alpha=0.7,
                label=f'exact: J={d["J_exact"]:.4f}')
        for iname, res in d['all_res'].items():
            sv = d['plm'].eval(res['a'], x_fine)
            ib = (iname == d['best_name'])
            star = ' \u2605' if ib else ''
            feas = '' if res['cv'] < 0.1 else ' [!]'
            ax.plot(x_fine, sv, color=init_colors[iname],
                    lw=2.5 if ib else 1.0, alpha=1.0 if ib else 0.35,
                    label=f'{iname}: J={res["J"]:.4f}{star}{feas}')
        ax.plot([-6, 6], [-6, 6], ':', color='gray', lw=0.7, alpha=0.3)
        ax.set_xlabel('$x$ (electron 1)', fontsize=11)
        ax.set_ylabel('$f(x)$ (electron 2)', fontsize=11)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.legend(fontsize=6.5, loc='upper right'); ax.grid(True, alpha=0.2)

    plt.suptitle('Strictly Correlated Electrons: $c(x,y) = 1/|x-y|$\n'
                 'Self-transport $\\rho \\to \\rho$, exact = anti-quantile',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig('fig_coulomb_1_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_coulomb_1_maps.png")

    # ---- Fig 2: Cost matrix + maps ----
    fig, axes = plt.subplots(1, len(dnames), figsize=(6*len(dnames), 5))
    for col, dname in enumerate(dnames):
        ax = axes[col]; d = all_data[dname]
        ng = 80; xg = np.linspace(-6, 6, ng)
        C = cost_coulomb(xg[:, None], xg[None, :])
        C = np.clip(C, 0, np.percentile(C, 95))
        im = ax.imshow(C.T, origin='lower', extent=[-6,6,-6,6],
                       cmap='inferno', aspect='auto', alpha=0.8)
        ax.plot(d['x_exact'], d['f_exact'], 'w--', lw=2, alpha=0.8, label='exact')
        best = d['best_name']
        sv = d['plm'].eval(d['all_res'][best]['a'], x_fine)
        ax.plot(x_fine, sv, 'c-', lw=2, alpha=0.9, label=f'SQP ({best})')
        ax.set_xlabel('$x$'); ax.set_title(f'{dname}', fontsize=11)
        if col == 0: ax.set_ylabel('$f(x)$')
        ax.legend(fontsize=8); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle('Coulomb Cost + Optimal Maps', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_coulomb_2_cost_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_coulomb_2_cost_maps.png")

    # ---- Fig 3: Convergence ----
    fig, axes = plt.subplots(1, len(dnames), figsize=(6*len(dnames), 4.5))
    for col, dname in enumerate(dnames):
        ax = axes[col]; d = all_data[dname]
        for iname, res in d['all_res'].items():
            h = res['hist']; ib = (iname == d['best_name'])
            ax.semilogy(h['cv_inf'], color=init_colors[iname],
                        lw=2 if ib else 0.8, alpha=1 if ib else 0.4,
                        marker='o', ms=2 if ib else 1, label=iname)
        ax.axhline(TOL_CV, color='red', ls=':', lw=1, alpha=0.5, label='tol')
        ax.set_xlabel('Outer iteration'); ax.set_title(dname, fontsize=11)
        if col == 0: ax.set_ylabel('$\\|\\tau\\|_\\infty$')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle('Convergence', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_coulomb_3_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_coulomb_3_convergence.png")

    # ---- Fig 4: Cost analysis ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    dist = np.linspace(0.01, 6, 500)
    ax1.plot(dist, 1/dist, 'r-', lw=2.5, label='Coulomb: $1/|x-y|$')
    ax1.plot(dist, dist**2, 'b--', lw=1.5, alpha=0.5, label='quadratic')
    ax1.plot(dist, -np.sqrt(dist), 'g--', lw=1.5, alpha=0.5, label='concave $-|x-y|^{1/2}$')
    ax1.set_xlabel('$|x-y|$'); ax1.set_ylabel('$c$')
    ax1.set_title('Cost Functions', fontsize=13)
    ax1.set_ylim(-3, 8); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    h = 0.02; x0 = np.linspace(-4, 4, 200)
    for y0 in [0.5, 1.5, 2.5]:
        cxy = (cost_coulomb(x0+h, y0+h) - cost_coulomb(x0+h, y0-h)
               - cost_coulomb(x0-h, y0+h) + cost_coulomb(x0-h, y0-h)) / (4*h**2)
        ax2.plot(x0, cxy, lw=1.5, label=f'$y={y0}$')
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('$x$'); ax2.set_ylabel('$c_{xy}$')
    ax2.set_title('$c_{xy} > 0$: wrong sign for Gangbo--McCann', fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_coulomb_4_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_coulomb_4_analysis.png")

    # ---- Summary ----
    print(f"\n{'='*95}")
    print(f"  {'Density':>18} {'Best init':>16} {'SQP J':>10} {'Exact J':>10} "
          f"{'Kant K':>10} {'Gap%':>8} {'cv':>10} {'Mono':>15}")
    print(f"{'-'*95}")
    for dname in dnames:
        d = all_data[dname]; best = d['best_name']; res = d['all_res'][best]
        K = d['K']; J_ex = d['J_exact']
        gap = 100*abs(res['J'] - J_ex)/abs(J_ex) if J_ex else 0
        K_str = f"{K:.5f}" if K else "FAILED"
        print(f"  {dname:>18} {best:>16} {res['J']:10.5f} {J_ex:10.5f} "
              f"{K_str:>10} {gap:7.2f}% {res['cv']:10.2e} {res['mono']:>15}")
    print(f"{'='*95}")
    print("\nDone.")


if __name__ == '__main__':
    main()
