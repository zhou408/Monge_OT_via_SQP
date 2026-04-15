"""
McCann (1999) Concave Transport Cost — SQP Solver
===================================================
Reproduces Example 1.1 from:
  R. J. McCann, "Exact solutions to the transportation problem on the line,"
  Proc. R. Soc. Lond. A 455, 1341–1380 (1999).

Domain: [-10, 10],  cost c(x,y) = sqrt(2|x-y|)  (positive, strictly concave)
Excess production:  d rho(x) = sin(pi*x/5) dx
Source mu = rho_+ = max(sin(pi*x/5), 0)    [support (-10,-5) ∪ (0,5)]
Target nu = rho_- = max(-sin(pi*x/5), 0)   [support (-5,0) ∪ (5,10)]

Analytical optimal map (McCann 1999, eq. 1.2):
    s(x) = -x - 10,   where -9 < x < -1,
    s(x) = -x,         where |x| < 1 or |x| > 9,       (1.2)
    s(x) = -x + 10,    where 1 < x < 9.

Kink locations {-9, 1} satisfy the optimality condition (eq. 1.3):
    c(1,-1) + c(-9,9) = c(1,9) + c(-9,-1)
    2 + 6 = 4 + 4 = 8.  ✓

Uses the existing monge_ot SQP solver with:
  - Continuous PLMap (double-knots at kink points for optimal partition)
  - Gaussian-smoothed pushforward constraint
  - |c_yy| Gauss-Newton Hessian (since true c_yy < 0 for concave cost)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# Add parent to path for monge_ot imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monge_ot.solver import (PLMap, _SQPCore, _run_sqp, _build_hat_matrix,
                              MongeOTResult, _classify_monotonicity)
from monge_ot.distributions import CustomDensity
from monge_ot.costs import CostFunction, _abs, _sign
from monge_ot.backend import get_backend


# =========================================================================
# Problem setup
# =========================================================================
DOMAIN = (-10.0, 10.0)

def rho_plus(x):
    """Source density (unnormalized): max(sin(pi*x/5), 0)."""
    return np.maximum(np.sin(np.pi * x / 5.0), 0.0)

def rho_minus(x):
    """Target density (unnormalized): max(-sin(pi*x/5), 0)."""
    return np.maximum(-np.sin(np.pi * x / 5.0), 0.0)

def s_analytical(x):
    """Analytical optimal map (McCann 1999, eq. 1.2).

    Defined on full domain [-10, 10]:
      s(x) = -x,         |x| > 9      (far reflection)
      s(x) = -x - 10,   -9 < x < -1   (near transport, left)
      s(x) = -x,         |x| < 1      (far reflection)
      s(x) = -x + 10,    1 < x < 9    (near transport, right)

    Kinks at x = {-9, -1, 1, 9}.
    """
    x = np.asarray(x, dtype=float)
    s = np.full_like(x, np.nan)
    # |x| > 9: s = -x
    m = (x >= -10) & (x < -9);  s[m] = -x[m]
    # -9 < x < -1: s = -x - 10
    m = (x >= -9) & (x < -1);   s[m] = -x[m] - 10.0
    # |x| < 1: s = -x
    m = (x >= -1) & (x < 1);    s[m] = -x[m]
    # 1 < x < 9: s = -x + 10
    m = (x >= 1) & (x <= 9);    s[m] = -x[m] + 10.0
    # |x| > 9: s = -x  (right side)
    m = (x > 9) & (x <= 10);    s[m] = -x[m]
    return s


# ── McCann cost: c(x,y) = sqrt(2|x-y|) ──────────────────────────────
def McCannCost(eps=1e-10):
    """c(x,y) = sqrt(2|x-y|).  Positive, strictly concave in displacement.

    c_y  = -sign(x-y) / sqrt(2|x-y|)   (gradient of c w.r.t. y)
    c_yy = -(2|x-y|)^{-3/2} < 0        (true Hessian is negative)

    We return |c_yy| as a Gauss-Newton positive-definite Hessian
    approximation, since the SQP solver requires H > 0.
    """
    def c(x, y):
        return (2.0 * _abs(x - y) + eps) ** 0.5

    def c_y(x, y):
        d = x - y
        return -_sign(d) / (2.0 * _abs(d) + eps) ** 0.5

    def c_yy(x, y):
        d = _abs(x - y) + eps
        return (2.0 * d) ** (-1.5)   # |c_yy| Gauss-Newton approximation

    return CostFunction(name='mccann_sqrt(2|x-y|)', c=c, c_y=c_y, c_yy=c_yy,
                        is_twisted=False)


# =========================================================================
# Build custom PLMap grids
# =========================================================================

def build_optimal_grid(n_sub=10, eps_kink=0.005):
    """Grid with double-knots at kink points {-9, 1} (within source support).

    Source support: (-10,-5) ∪ (0,5).
    Kinks within source support: x=-9 and x=1.
    Gap regions [-5,0] and [5,10]: sparse nodes (zero source mass).
    """
    parts = [
        # Piece 1: (-10, -9)
        np.linspace(-10, -9 - eps_kink, n_sub + 1),
        # Double knot at -9
        np.array([-9 + eps_kink]),
        # Piece 2: (-9, -5)
        np.linspace(-9 + eps_kink, -5, n_sub + 1)[1:],
        # Gap (-5, 0): sparse
        np.array([-3.0]),
        # Piece 3: (0, 1)
        np.linspace(0, 1 - eps_kink, n_sub + 1),
        # Double knot at 1
        np.array([1 + eps_kink]),
        # Piece 4: (1, 5)
        np.linspace(1 + eps_kink, 5, n_sub + 1)[1:],
        # Gap (5, 10): sparse
        np.array([7.5, 10.0]),
    ]
    grid = np.sort(np.unique(np.concatenate(parts)))
    return grid


def build_uniform_grid(n_pieces=30):
    """Uniform partition of [-10, 10]."""
    return np.linspace(-10, 10, n_pieces + 1)


# =========================================================================
# Build analytical initialization for a given grid
# =========================================================================

def build_analytical_init(x_grid):
    """Set knot values to the analytical map values."""
    a = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        sv = s_analytical(np.array([x]))[0]
        if np.isnan(sv):
            # Outside source support: use reflection (arbitrary, no mass)
            a[i] = -x
        else:
            a[i] = sv
    return a


# =========================================================================
# Run SQP experiment
# =========================================================================

def run_experiment(grid_type, n_pieces=30, n_sub=10, verbose=True):
    """Run SQP for one grid type using monge_ot solver infrastructure."""
    source = CustomDensity(rho_plus, lo=-10, hi=10, n_pts=20000)
    target = CustomDensity(rho_minus, lo=-10, hi=10, n_pts=20000)
    cost = McCannCost()

    if grid_type == 'optimal':
        x_grid = build_optimal_grid(n_sub=n_sub)
    else:
        x_grid = build_uniform_grid(n_pieces=n_pieces)

    n1 = len(x_grid)
    n = n1 - 1
    print(f"  Grid: {grid_type}, {n} pieces, {n1} nodes")

    # Create PLMap with custom grid
    plm = PLMap.__new__(PLMap)
    plm.n = n
    plm.x_grid = x_grid

    # Solver parameters
    n_bins = max(60, 2 * n)
    n_quad = 10000
    n_fine = 12000

    backend = get_backend('numpy', 'cpu')
    core = _SQPCore(plm, source, target, cost, n_bins,
                    n_quad, n_fine, 'smooth', backend)

    # Build initializations
    a_analytical = build_analytical_init(x_grid)
    a_reflection = -x_grid.copy()   # s = -x (anti-monotone)
    a_shift = x_grid.copy() + 5.0   # s = x+5 (monotone rearrangement)
    rng = np.random.default_rng(42)
    a_perturbed = a_analytical.copy() + rng.normal(0, 0.5, n1)

    inits = {
        'analytical': a_analytical,
        'reflection': a_reflection,
        'shift(x+5)': a_shift,
        'perturbed': a_perturbed,
    }

    all_results = {}
    for name, a0 in inits.items():
        if verbose:
            print(f"\n    Init: {name}")

        a0_clipped = np.clip(a0, -10 + 1e-4, 10 - 1e-4)
        t0 = time.time()
        res = _run_sqp(
            core, a0_clipped,
            enforce_monotone=False,
            max_iter=300,
            tol_cv=2e-3,
            tol_grad=1e-3,
            merit_type='l1',
            sigma_init=10.0,
        )
        elapsed = time.time() - t0

        if verbose:
            print(f"      J={res['J']:.6f}, cv={res['cv']:.3e}, "
                  f"{res['mono']}  [{elapsed:.1f}s]")
        all_results[name] = res

    # Select best (lowest J among feasible, tight cv threshold)
    feasible = {k: v for k, v in all_results.items() if v['cv'] < 0.01}
    if feasible:
        best_name = min(feasible, key=lambda k: feasible[k]['J'])
    else:
        best_name = min(all_results, key=lambda k: all_results[k]['cv'])

    best = all_results[best_name]
    print(f"\n  Best: {best_name} (J={best['J']:.6f}, cv={best['cv']:.3e})")
    return plm, best, best_name, all_results


# =========================================================================
# Plotting
# =========================================================================

def plot_map_comparison(plm, result, grid_type, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={'width_ratios': [3, 2]})

    ax = axes[0]
    # Plot SQP on source support intervals
    src_intervals = [
        np.linspace(-10, -5, 500),
        np.linspace(0, 5, 500),
    ]
    for i, x_int in enumerate(src_intervals):
        s_sqp = np.interp(x_int, plm.x_grid, result['a'])
        label = f'SQP ($J$={result["J"]:.4f}, cv={result["cv"]:.2e})' if i == 0 else None
        ax.plot(x_int, s_sqp, '-', color='#2196F3', lw=2.5,
                label=label, zorder=5)

    # Plot analytical map piece by piece ON TOP (red dashed)
    # McCann eq. (1.2): 4 pieces on source support
    pieces = [
        (np.linspace(-10, -9, 200),  lambda x: -x),
        (np.linspace(-9,  -5, 400),  lambda x: -x - 10.0),
        (np.linspace(0,    1, 200),  lambda x: -x),
        (np.linspace(1,    5, 400),  lambda x: -x + 10.0),
    ]
    for i, (xp, sp) in enumerate(pieces):
        label = 'McCann (1.2) $s^*(x)$' if i == 0 else None
        ax.plot(xp, sp(xp), '--', color='#E53935', lw=2.0, alpha=0.9,
                label=label, zorder=10)

    # Mark kink points
    ax.plot([-9, 1], [9, -1], 'ko', ms=6, zorder=15)   # discontinuity points
    for k in [-9, 1]:
        ax.axvline(k, color='gray', ls=':', alpha=0.4, lw=1.5)
    for k in [-10, -5, 0, 5]:
        ax.axvline(k, color='gray', ls='-', alpha=0.15, lw=1)

    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$s(x)$', fontsize=13)
    title = 'Optimal' if grid_type == 'optimal' else 'Uniform'
    ax.set_title(f'Piecewise Transport: Learned vs True ({title} Partition)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)

    # Convergence history
    ax2 = axes[1]
    hist = result['hist']
    iters = np.arange(1, len(hist['obj']) + 1)
    ax2.semilogy(iters, hist['cv_inf'], 'o-', ms=2, lw=1.2,
                 color='#E53935', label='$\\|\\tau\\|_\\infty$ (cv)')
    ax2.axhline(2e-3, color='gray', ls='--', alpha=0.5, label='tol$_{cv}$')
    ax2.set_xlabel('SQP Iteration', fontsize=12)
    ax2.set_ylabel('Constraint Violation', fontsize=12)
    ax2.set_title('Convergence History', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {save_path}")


def plot_densities(save_path):
    """Plot source and target densities."""
    x = np.linspace(-10, 10, 2000)
    Z = 20.0 / np.pi  # normalization constant

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x, rho_plus(x) / Z, '-', color='#1976D2', lw=2)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('density', fontsize=13)
    ax.set_title('Source density $\\mu(x) = \\rho_+/Z$', fontsize=14)
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-0.005, 0.17)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, rho_minus(x) / Z, '-', color='#E91E63', lw=2)
    ax.set_xlabel('$y$', fontsize=13)
    ax.set_ylabel('density', fontsize=13)
    ax.set_title('Target density $\\nu(y) = \\rho_-/Z$', fontsize=14)
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-0.005, 0.17)
    ax.grid(True, alpha=0.3)

    plt.suptitle('McCann (1999) Example 1.1: $d\\rho = \\sin(\\pi x/5)\\,dx$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {save_path}")


def plot_target_density(plm, result, grid_type, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute pushforward via fine sampling on source support
    n_s = 80000
    x_all = np.linspace(-10, 10, n_s)
    f_all = rho_plus(x_all)
    dx_s = 20.0 / (n_s - 1)
    s_all = np.interp(x_all, plm.x_grid, result['a'])
    weights = f_all * dx_s
    total_mass = weights.sum()  # ~20/pi

    y_plot = np.linspace(-10, 10, 1000)
    nu_exact = rho_minus(y_plot)

    # Histogram
    ax = axes[0]
    nbins = 80
    ax.hist(s_all, bins=nbins, range=(-10, 10), weights=weights, density=False,
            alpha=0.5, color='#2196F3', label='Pushforward $s_\\#\\mu$')
    bin_w = 20.0 / nbins
    ax.plot(y_plot, nu_exact * bin_w, 'r-', lw=2, label='Target $\\nu$')
    ax.set_xlabel('$y$', fontsize=12)
    ax.set_ylabel('Mass per bin', fontsize=12)
    ax.set_title('Pushforward vs Target (histogram)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # KDE density
    ax = axes[1]
    from scipy.stats import gaussian_kde
    mask = weights > 1e-12
    if mask.sum() > 10:
        w_pos = weights[mask]; w_pos /= w_pos.sum()
        try:
            kde = gaussian_kde(s_all[mask], weights=w_pos, bw_method=0.15)
            ax.plot(y_plot, kde(y_plot) * total_mass, '-', color='#2196F3',
                    lw=2, label='Learned $P_T(s)(d)$')
        except Exception:
            pass
    ax.plot(y_plot, nu_exact, 'k--', lw=2, label='True $\\nu(d)$')
    ax.set_xlabel('$y$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = 'Optimal' if grid_type == 'optimal' else 'Uniform'
    ax.set_title(f'Target Density Check ({title} Partition)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {save_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("  McCann (1999) Example 1.1 — Concave Transport Cost")
    print("  Domain [-10, 10],  c(x,y) = sqrt(2|x-y|)")
    print("  d rho = sin(pi*x/5) dx,  mu = rho_+,  nu = rho_-")
    print("=" * 70)

    # Verify masses and compute J*
    x_v = np.linspace(-10, 10, 200000)
    dx_v = 20.0 / (len(x_v) - 1)
    raw_mass = np.sum(rho_plus(x_v)) * dx_v
    print(f"  Source mass (raw): {raw_mass:.6f}  (Z = 20/pi = {20/np.pi:.6f})")
    print(f"  Target mass (raw): {np.sum(rho_minus(x_v))*dx_v:.6f}")

    # Verify kink optimality condition (McCann eq. 1.3)
    c = lambda x, y: np.sqrt(2.0 * np.abs(x - y))
    lhs = c(1, -1) + c(-9, 9)
    rhs = c(1, 9) + c(-9, -1)
    print(f"\n  Kink optimality (eq. 1.3):")
    print(f"    c(1,-1) + c(-9,9) = {c(1,-1):.4f} + {c(-9,9):.4f} = {lhs:.4f}")
    print(f"    c(1,9)  + c(-9,-1)= {c(1,9):.4f} + {c(-9,-1):.4f} = {rhs:.4f}")
    print(f"    Match: {np.isclose(lhs, rhs)}")

    # Compute J* (normalized)
    mu_norm = rho_plus(x_v) / raw_mass
    s_a = s_analytical(x_v)
    valid = ~np.isnan(s_a)
    J_analytical = np.sum(
        np.sqrt(2.0 * np.abs(x_v[valid] - s_a[valid])) * mu_norm[valid]) * dx_v
    print(f"\n  Analytical J* = {J_analytical:.6f}  (McCann eq. 1.2)")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
    os.makedirs(out_dir, exist_ok=True)

    # Plot densities
    plot_densities(os.path.join(out_dir, 'source_target_densities.png'))

    # ── Experiment 1: Optimal Partition ──
    print(f"\n{'='*70}")
    print("  Experiment 1: Optimal Partition (double-knots at kinks -9, 1)")
    print(f"{'='*70}")
    plm_opt, res_opt, name_opt, all_opt = run_experiment(
        'optimal', n_sub=12, verbose=True)
    plot_map_comparison(plm_opt, res_opt, 'optimal',
                        os.path.join(out_dir, 'concave optimal partition.png'))
    plot_target_density(plm_opt, res_opt, 'optimal',
                        os.path.join(out_dir,
                                     'concave optimal partition target density.png'))

    # ── Experiment 2: Uniform Partition ──
    print(f"\n{'='*70}")
    print("  Experiment 2: Uniform Partition (100 pieces)")
    print(f"{'='*70}")
    plm_uni, res_uni, name_uni, all_uni = run_experiment(
        'uniform', n_pieces=100, verbose=True)
    plot_map_comparison(plm_uni, res_uni, 'uniform',
                        os.path.join(out_dir, 'concave uniform partition.png'))
    plot_target_density(plm_uni, res_uni, 'uniform',
                        os.path.join(out_dir,
                                     'concave uniform partition target density.png'))

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  {'Partition':>12} {'Best init':>14} {'J':>10} {'J*':>10} "
          f"{'cv':>10} {'gap%':>8}")
    print(f"  {'-'*64}")
    for label, res, nm in [('optimal', res_opt, name_opt),
                            ('uniform', res_uni, name_uni)]:
        gap = 100 * abs(res['J'] - J_analytical) / abs(J_analytical)
        print(f"  {label:>12} {nm:>14} {res['J']:10.6f} "
              f"{J_analytical:10.6f} {res['cv']:10.3e} {gap:7.2f}%")
    print(f"{'='*70}")
    print(f"\n  Figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
