"""
Beyond Brenier: Non-Monotone Optimal Transport
===============================================
c(x,y) = -|x - y|^p   for p in {0.5, 1.0}

Source:  Beta(2, 5)  on [0,1]  (left-skewed, many low types)
Target:  Beta(5, 2)  on [0,1]  (right-skewed, many high types)

Concave cost rewards large displacement. With mu != nu, the optimal
non-monotone map has genuine structure — not just a line.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, linprog
from scipy.sparse import lil_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Cost
# =============================================================================

def cost_concave(x, y, p=0.5, eps=1e-8):
    return -(np.abs(x - y) + eps)**p

def cost_concave_dy_fd(x, y, p=0.5, eps=1e-8):
    h = 1e-7
    return (cost_concave(x, y+h, p, eps) - cost_concave(x, y-h, p, eps)) / (2*h)


# =============================================================================
# Piecewise linear map (general, non-monotone)
# =============================================================================

class PLMap:
    def __init__(self, n, mu):
        self.n = n
        self.mu = mu
        # Quantile-based source grid for better resolution
        eps = 2e-3
        self.x_grid = mu.ppf(np.linspace(eps, 1-eps, n+1))

    def eval(self, a, x):
        return np.interp(x, self.x_grid, a)


# =============================================================================
# Pushforward via histogram (for general f-weighted measure)
# =============================================================================

def pushforward_hist(a, plm, nu, n_bins, n_fine=1000):
    """
    For source mu with density f, pushforward constraint:
    int_{s^{-1}(B_j)} f(x) dx = int_{B_j} g(y) dy  for each bin B_j.

    We approximate via fine quadrature on quantile grid of mu.
    """
    eps = 2e-3
    # Quantile quadrature: x_i = F_mu^{-1}(u_i), weight = du
    u = np.linspace(eps, 1-eps, n_fine)
    x_pts = plm.mu.ppf(u)
    du = (1 - 2*eps) / (n_fine - 1)

    s_vals = plm.eval(a, x_pts)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    tau = np.zeros(n_bins)
    for b in range(n_bins):
        # Source mass in bin: sum of du for points mapping into bin
        mask = (s_vals >= bin_edges[b]) & (s_vals < bin_edges[b+1])
        source_mass = np.sum(mask) * du
        # Target mass in bin: int_{bin} g(y) dy
        target_mass = nu.cdf(bin_edges[b+1]) - nu.cdf(bin_edges[b])
        tau[b] = source_mass - target_mass

    return tau


def jac_hist(a, plm, nu, n_bins, eps=5e-7):
    tau0 = pushforward_hist(a, plm, nu, n_bins)
    n1 = len(a)
    J = np.zeros((n_bins, n1))
    for j in range(n1):
        ap = a.copy(); ap[j] += eps
        J[:, j] = (pushforward_hist(ap, plm, nu, n_bins) - tau0) / eps
    return J


def objective(a, plm, cost_func, n_quad=800):
    """J(s) = int c(x, s(x)) f(x) dx via quantile quadrature."""
    eps = 2e-3
    u = np.linspace(eps, 1-eps, n_quad)
    x_pts = plm.mu.ppf(u)
    du = (1 - 2*eps) / (n_quad - 1)
    s_vals = plm.eval(a, x_pts)
    return np.sum(du * cost_func(x_pts, s_vals))


def grad_obj_fd(a, plm, cost_func, eps=1e-6):
    g = np.zeros(len(a))
    f0 = objective(a, plm, cost_func)
    for j in range(len(a)):
        ap = a.copy(); ap[j] += eps
        g[j] = (objective(ap, plm, cost_func) - f0) / eps
    return g


# =============================================================================
# Solver with multiple initializations
# =============================================================================

def solve(cost_func, plm, nu, n_bins=20,
          inits=None, rho_init=50.0, rho_max=1e5,
          max_outer=40, tol_cv=3e-3, verbose=True):

    n1 = plm.n + 1

    if inits is None:
        # Quantile map (monotone increasing, Brenier-optimal for quadratic)
        q_map = nu.ppf(plm.mu.cdf(plm.x_grid))
        # Reversal-type: send low x to high y
        rev_map = nu.ppf(1.0 - plm.mu.cdf(plm.x_grid))
        # V-shape: low and high x go to extremes, middle goes to middle
        u = plm.mu.cdf(plm.x_grid)
        v_map = nu.ppf(np.abs(2*u - 1))
        # Tent: middle x goes to extremes
        tent_map = nu.ppf(1.0 - np.abs(2*u - 1))

        inits = {
            'quantile': q_map,
            'anti-quantile': rev_map,
            'V-shape': v_map,
            'tent': tent_map,
        }

    best_a = None
    best_J = np.inf
    best_name = None
    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"\n  --- Init: {name} ---")

        a = np.clip(a0.copy(), 1e-4, 1-1e-4)
        lam = np.zeros(n_bins)
        rho = rho_init

        hist = {'obj': [], 'cv_inf': []}
        prev_cv = np.inf

        for outer in range(max_outer):
            def auglag(a_f):
                J = objective(a_f, plm, cost_func)
                tau = pushforward_hist(a_f, plm, nu, n_bins)
                return J + np.dot(lam, tau) + 0.5*rho*np.dot(tau, tau)

            def auglag_grad(a_f):
                gJ = grad_obj_fd(a_f, plm, cost_func)
                tau = pushforward_hist(a_f, plm, nu, n_bins)
                Jc = jac_hist(a_f, plm, nu, n_bins)
                return gJ + Jc.T @ (lam + rho*tau)

            bounds = [(1e-4, 1-1e-4)] * n1
            res = minimize(auglag, a, jac=auglag_grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            a = res.x

            tau = pushforward_hist(a, plm, nu, n_bins)
            cv1 = np.linalg.norm(tau, 1)
            cvinf = np.linalg.norm(tau, np.inf)
            J_val = objective(a, plm, cost_func)

            hist['obj'].append(J_val)
            hist['cv_inf'].append(cvinf)

            if verbose and (outer < 3 or outer % 10 == 0):
                print(f"    {outer:3d}  J={J_val:10.6f}  "
                      f"|tau|_inf={cvinf:.3e}  rho={rho:.0f}")

            if cvinf < tol_cv:
                if verbose:
                    print(f"    Converged: |tau|_inf = {cvinf:.2e}")
                break

            lam = lam + rho * tau
            if cv1 > 0.25 * prev_cv:
                rho = min(rho * 2.0, rho_max)
            prev_cv = cv1

        all_results[name] = {
            'a': a.copy(), 'hist': hist,
            'J': hist['obj'][-1], 'cv': hist['cv_inf'][-1]
        }

        if hist['obj'][-1] < best_J and hist['cv_inf'][-1] < 0.05:
            best_J = hist['obj'][-1]
            best_a = a.copy()
            best_name = name

    return best_a, best_name, all_results


# =============================================================================
# Kantorovich LP
# =============================================================================

def kant_lp(cost_func, mu, nu, n_disc=100):
    eps = 2e-3
    u = np.linspace(eps, 1-eps, n_disc)
    x = mu.ppf(u); y = nu.ppf(u)
    du = (1-2*eps)/(n_disc-1)
    # Weights proportional to uniform in u-space
    p_v = np.full(n_disc, du); p_v /= p_v.sum()
    q_v = p_v.copy()

    C = cost_func(x[:, None], y[None, :]).flatten()
    A = lil_matrix((2*n_disc, n_disc**2))
    for i in range(n_disc):
        A[i, i*n_disc:(i+1)*n_disc] = 1.0
    for j in range(n_disc):
        A[n_disc+j, j::n_disc] = 1.0

    res = linprog(C, A_eq=A.tocsc(), b_eq=np.concatenate([p_v, q_v]),
                  bounds=(0, None), method='highs')
    return res.fun if res.success else None


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 65)
    print("  BEYOND BRENIER: Non-Monotone Transport")
    print("  c(x,y) = -|x-y|^p")
    print("  Beta(2,5) -> Beta(5,2)")
    print("=" * 65)

    mu = stats.beta(2, 5)
    nu = stats.beta(5, 2)
    n_pieces = 30
    n_bins = 15

    p_values = [0.5]
    results = {}

    for p in p_values:
        print(f"\n{'='*65}")
        print(f"  p = {p}:  c(x,y) = -|x-y|^{p}")
        print(f"{'='*65}")

        cf = lambda x, y, pp=p: cost_concave(x, y, pp)
        plm = PLMap(n_pieces, mu)

        best_a, best_name, all_res = solve(
            cf, plm, nu, n_bins=n_bins,
            rho_init=50.0, max_outer=25, tol_cv=5e-3, verbose=True)

        K = kant_lp(cf, mu, nu, n_disc=100)

        # Quantile map objective
        q_map = nu.ppf(mu.cdf(plm.x_grid))
        q_obj = objective(q_map, plm, cf)

        # Anti-quantile objective
        aq_map = nu.ppf(1.0 - mu.cdf(plm.x_grid))
        aq_obj = objective(aq_map, plm, cf)

        results[p] = {
            'best_a': best_a, 'best_name': best_name,
            'all_res': all_res, 'K': K,
            'q_obj': q_obj, 'aq_obj': aq_obj, 'plm': plm,
        }

        print(f"\n  Best init: {best_name}")
        for nm, r in all_res.items():
            marker = ' ★' if nm == best_name else ''
            print(f"    {nm:>15}: J={r['J']:.6f}  cv={r['cv']:.3e}{marker}")
        print(f"  Quantile obj:      {q_obj:.6f}")
        print(f"  Anti-quantile obj: {aq_obj:.6f}")
        print(f"  Kantorovich LB:    {K:.6f}" if K else "  Kantorovich: FAILED")

    # Summary
    print(f"\n{'='*65}")
    print(f"  {'p':>4} {'Best SQP':>10} {'Quantile':>10} {'Anti-Q':>10} "
          f"{'Kant LB':>10} {'Best init':>14}")
    print(f"{'-'*65}")
    for p in p_values:
        r = results[p]
        J = r['all_res'][r['best_name']]['J']
        print(f"  {p:4.1f} {J:10.6f} {r['q_obj']:10.6f} {r['aq_obj']:10.6f} "
              f"{r['K']:10.6f} {r['best_name']:>14}")
    print(f"{'='*65}")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\nGenerating figures...")

    eps = 2e-3
    x_fine_u = np.linspace(eps, 1-eps, 1000)
    x_fine = mu.ppf(x_fine_u)

    colors_p = {0.5: '#2196F3', 0.8: '#4CAF50', 1.0: '#FF9800'}
    init_colors = {'quantile': '#2196F3', 'anti-quantile': '#E91E63',
                   'V-shape': '#4CAF50', 'tent': '#9C27B0'}

    # ---- Fig 1: Best transport maps for each p ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, p in enumerate(p_values):
        ax = axes[idx]
        r = results[p]
        plm_p = r['plm']

        # Plot all initializations as thin lines
        for nm, res in r['all_res'].items():
            s_vals = plm_p.eval(res['a'], x_fine)
            is_best = (nm == r['best_name'])
            lw = 2.5 if is_best else 1.0
            alpha = 1.0 if is_best else 0.4
            label = f'{nm}' + (' ★' if is_best else '')
            ax.plot(x_fine, s_vals, color=init_colors.get(nm, '#888'),
                    lw=lw, alpha=alpha, label=f'{label}: {res["J"]:.4f}')

        ax.plot([0, 1], [0, 1], ':', color='gray', lw=0.8, alpha=0.3)
        ax.set_xlabel('Source $x$', fontsize=12)
        ax.set_ylabel('$s(x)$', fontsize=12)
        K_str = f'{r["K"]:.4f}' if r['K'] else 'N/A'
        ax.set_title(f'$c = -|x-y|^{{{p}}}$\nKant LB: {K_str}', fontsize=13)
        ax.legend(fontsize=8, loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.85)

    plt.suptitle('Non-Monotone Optimal Maps: Beta(2,5) $\\to$ Beta(5,2)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fig1_nonmonotone_maps.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Fig 2: Detailed view of p=0.5, all inits ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    p_show = 0.5
    r = results[p_show]
    plm_p = r['plm']

    for idx, (nm, res) in enumerate(r['all_res'].items()):
        ax = axes[idx//2][idx%2]
        s_vals = plm_p.eval(res['a'], x_fine)
        ax.plot(x_fine, s_vals, color=init_colors.get(nm, '#888'), lw=2)
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, lw=1)

        # Check monotonicity
        ds = np.diff(res['a'])
        n_neg = np.sum(ds < 0)
        n_pos = np.sum(ds > 0)
        mono_str = 'monotone ↑' if n_neg == 0 else \
                   'monotone ↓' if n_pos == 0 else \
                   f'non-monotone ({n_neg}↓, {n_pos}↑)'

        star = ' ★ BEST' if nm == r['best_name'] else ''
        ax.set_title(f'{nm}{star}\n$J = {res["J"]:.5f}$,  '
                     f'$|\\tau|_\\infty = {res["cv"]:.2e}$\n{mono_str}',
                     fontsize=11,
                     color='red' if nm == r['best_name'] else 'black')
        ax.set_xlabel('$x$', fontsize=11); ax.set_ylabel('$s(x)$', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.85)

    plt.suptitle(f'All Local Optima: $c = -|x-y|^{{{p_show}}}$,  '
                 f'Kantorovich LB = {r["K"]:.5f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fig2_local_optima.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Fig 3: Pushforward check + convergence ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for p in p_values:
        r = results[p]
        plm_p = r['plm']
        a = r['best_a']
        if a is not None:
            n_check = 5000
            u_check = np.linspace(eps, 1-eps, n_check)
            x_check = mu.ppf(u_check)
            s_check = plm_p.eval(a, x_check)
            ax1.hist(s_check, bins=40, density=True, alpha=0.35,
                     color=colors_p[p], label=f'$p={p}$')

    yd = np.linspace(0.01, 0.99, 500)
    ax1.plot(yd, nu.pdf(yd), 'k-', lw=2.5, label='Target $g(y)$')
    ax1.set_xlabel('$y$', fontsize=12); ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Pushforward Check', fontsize=13)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    # Convergence
    for p in p_values:
        r = results[p]
        h = r['all_res'][r['best_name']]['hist']
        ax2.semilogy(h['cv_inf'], color=colors_p[p], lw=1.8,
                     label=f'$p={p}$', marker='o', ms=3)
    ax2.set_xlabel('Outer iteration', fontsize=12)
    ax2.set_ylabel('$\\|\\tau\\|_\\infty$', fontsize=12)
    ax2.set_title('Constraint Violation (best init)', fontsize=13)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fig3_pushforward_conv.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Fig 4: Problem setup + optimality gaps ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    xd = np.linspace(0.001, 0.999, 500)
    ax1.fill_between(xd, mu.pdf(xd), alpha=0.25, color='#2196F3')
    ax1.plot(xd, mu.pdf(xd), '#2196F3', lw=2, label='Source $\\mu$: Beta(2,5)')
    ax1.fill_between(xd, nu.pdf(xd), alpha=0.25, color='#E91E63')
    ax1.plot(xd, nu.pdf(xd), '#E91E63', lw=2, label='Target $\\nu$: Beta(5,2)')
    ax1.set_xlabel('Type', fontsize=12); ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Source and Target Distributions', fontsize=13)
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

    # Bar chart
    labels = [f'p={p}' for p in p_values]
    best_v = [results[p]['all_res'][results[p]['best_name']]['J'] for p in p_values]
    q_v = [results[p]['q_obj'] for p in p_values]
    aq_v = [results[p]['aq_obj'] for p in p_values]
    k_v = [results[p]['K'] for p in p_values]

    x_pos = np.arange(len(labels))
    w = 0.2
    ax2.bar(x_pos - 1.5*w, best_v, w, label='Best SQP', color='#2196F3', alpha=0.85)
    ax2.bar(x_pos - 0.5*w, q_v, w, label='Quantile (Brenier)', color='#4CAF50', alpha=0.85)
    ax2.bar(x_pos + 0.5*w, aq_v, w, label='Anti-quantile', color='#E91E63', alpha=0.85)
    ax2.bar(x_pos + 1.5*w, k_v, w, label='Kantorovich LP', color='#FF9800', alpha=0.85)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Objective (lower = better)', fontsize=12)
    ax2.set_title('Comparison Across Methods', fontsize=13)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fig4_setup_gaps.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    print("\nAll figures saved.")


if __name__ == '__main__':
    main()
