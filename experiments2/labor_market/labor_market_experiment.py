"""
Beyond Brenier: SQP for General-Cost Monge Optimal Transport
=============================================================
Experiment: Labor Market Matching with Concave Mismatch Cost

ECONOMIC STORY:
  Workers with skill x ~ Beta(2,5) are matched to jobs requiring skill y ~ Beta(5,2).
  "Skills gap" economy: many low-skill workers, many high-skill jobs.

  Cost = complementarity benefit - concave mismatch friction:
    c(x,y) = -alpha * x * y  +  gamma * |x - y|^p

  The complementarity term -alpha*xy rewards assortative matching (high to high).
  The mismatch term gamma*|x-y|^p penalizes skill gaps.

  KEY INSIGHT: When p < 1, the mismatch penalty is CONCAVE in |x-y|.
  This means moderate and large mismatches cost about the same.
  Economically: once you're mismatched, it doesn't matter by how much.
  This "flat penalty" regime breaks assortative matching and can produce
  non-monotone optimal maps where some low workers go to high jobs.

  When p >= 2 (e.g. quadratic), the penalty is convex and reinforces
  assortative matching. The optimal map barely deviates from quantile coupling.

  We compare p in {0.5, 1.0, 2.0, 4.0} to show the transition from
  non-monotone to monotone optimal maps as p increases.

Dependencies: numpy, scipy, matplotlib
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
# Cost function family
# =============================================================================

def cost_fn(x, y, alpha, gamma, p, eps_smooth=1e-8):
    """c(x,y) = -alpha*x*y + gamma * (|x-y|^2 + eps)^{p/2}."""
    return -alpha * x * y + gamma * (np.abs(x - y)**2 + eps_smooth)**(p / 2)


# =============================================================================
# Piecewise linear map (general, non-monotone)
# =============================================================================

class PLMap:
    def __init__(self, n, mu, eps=2e-3):
        self.n = n
        self.mu = mu
        self.x_grid = mu.ppf(np.linspace(eps, 1-eps, n+1))

    def eval(self, a, x):
        return np.interp(x, self.x_grid, a)


# =============================================================================
# Histogram-based pushforward (handles non-monotone maps)
# =============================================================================

def pushforward_hist(a, plm, nu, n_bins, n_fine=3000):
    eps = 2e-3
    u = np.linspace(eps, 1-eps, n_fine)
    x_pts = plm.mu.ppf(u)
    du = (1 - 2*eps) / (n_fine - 1)
    s_vals = plm.eval(a, x_pts)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    tau = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (s_vals >= bin_edges[b]) & (s_vals < bin_edges[b+1])
        tau[b] = np.sum(mask) * du - (nu.cdf(bin_edges[b+1]) - nu.cdf(bin_edges[b]))
    return tau


def jac_hist(a, plm, nu, n_bins, fd_eps=5e-7):
    tau0 = pushforward_hist(a, plm, nu, n_bins)
    J = np.zeros((n_bins, len(a)))
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        J[:, j] = (pushforward_hist(ap, plm, nu, n_bins) - tau0) / fd_eps
    return J


def objective(a, plm, alpha, gamma, p, n_quad=3000):
    eps = 2e-3
    u = np.linspace(eps, 1-eps, n_quad)
    x_pts = plm.mu.ppf(u)
    du = (1 - 2*eps) / (n_quad - 1)
    s_vals = plm.eval(a, x_pts)
    return np.sum(du * cost_fn(x_pts, s_vals, alpha, gamma, p))


def grad_obj_fd(a, plm, alpha, gamma, p, fd_eps=1e-6):
    g = np.zeros(len(a))
    f0 = objective(a, plm, alpha, gamma, p)
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        g[j] = (objective(ap, plm, alpha, gamma, p) - f0) / fd_eps
    return g


# =============================================================================
# Augmented Lagrangian solver
# =============================================================================

def solve_al(plm, nu, alpha, gamma, p, n_bins=25,
             inits=None, rho_init=50.0, rho_max=1e5,
             max_outer=50, tol_cv=5e-3, verbose=True):
    n1 = plm.n + 1
    u = plm.mu.cdf(plm.x_grid)

    if inits is None:
        inits = {
            'quantile': nu.ppf(u),
            'anti-quantile': nu.ppf(1.0 - u),
            'V-shape': nu.ppf(np.abs(2*u - 1)),
            'tent': nu.ppf(1.0 - np.abs(2*u - 1)),
        }

    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"    {name}...", end='', flush=True)

        a = np.clip(a0.copy(), 1e-4, 1-1e-4)
        lam = np.zeros(n_bins)
        rho = rho_init
        prev_cv = np.inf

        hist = {'obj': [], 'cv_inf': []}

        for outer in range(max_outer):
            def auglag(a_f):
                J = objective(a_f, plm, alpha, gamma, p)
                tau = pushforward_hist(a_f, plm, nu, n_bins)
                return J + np.dot(lam, tau) + 0.5*rho*np.dot(tau, tau)

            def auglag_grad(a_f):
                gJ = grad_obj_fd(a_f, plm, alpha, gamma, p)
                tau = pushforward_hist(a_f, plm, nu, n_bins)
                Jc = jac_hist(a_f, plm, nu, n_bins)
                return gJ + Jc.T @ (lam + rho*tau)

            bounds = [(1e-4, 1-1e-4)] * n1
            res = minimize(auglag, a, jac=auglag_grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-11})
            a = res.x

            tau = pushforward_hist(a, plm, nu, n_bins)
            cv1 = np.linalg.norm(tau, 1)
            cvinf = np.linalg.norm(tau, np.inf)
            J_val = objective(a, plm, alpha, gamma, p)

            hist['obj'].append(J_val)
            hist['cv_inf'].append(cvinf)

            if cvinf < tol_cv:
                break

            lam = lam + rho * tau
            if cv1 > 0.25 * prev_cv:
                rho = min(rho * 2.0, rho_max)
            prev_cv = cv1

        ds = np.diff(a)
        n_neg = np.sum(ds < -1e-6); n_pos = np.sum(ds > 1e-6)
        if n_neg == 0:
            mono = 'monotone ↑'
        elif n_pos == 0:
            mono = 'monotone ↓'
        else:
            mono = f'non-monotone ({n_neg}↓ {n_pos}↑)'

        if verbose:
            print(f"  J={J_val:.5f}, cv={cvinf:.2e}, {mono}")

        all_results[name] = {
            'a': a.copy(), 'hist': hist,
            'J': J_val, 'cv': cvinf, 'mono': mono,
        }

    best_name = min(all_results,
                    key=lambda k: all_results[k]['J']
                    if all_results[k]['cv'] < 0.05 else 1e10)
    return all_results, best_name


def kant_lp(alpha, gamma, p, mu, nu, n_disc=120):
    eps = 2e-3
    u = np.linspace(eps, 1-eps, n_disc)
    x = mu.ppf(u); y = nu.ppf(u)
    du = (1-2*eps)/(n_disc-1)
    pv = np.full(n_disc, du); pv /= pv.sum()
    qv = pv.copy()
    C = cost_fn(x[:, None], y[None, :], alpha, gamma, p).flatten()
    A = lil_matrix((2*n_disc, n_disc**2))
    for i in range(n_disc):
        A[i, i*n_disc:(i+1)*n_disc] = 1.0
    for j in range(n_disc):
        A[n_disc+j, j::n_disc] = 1.0
    res = linprog(C, A_eq=A.tocsc(), b_eq=np.concatenate([pv, qv]),
                  bounds=(0, None), method='highs')
    return res.fun if res.success else None


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  BEYOND BRENIER: Labor Market Matching")
    print("  c(x,y) = -alpha*xy + gamma*|x-y|^p")
    print("  Workers: Beta(2,5)  Jobs: Beta(5,2)")
    print("=" * 70)

    mu = stats.beta(2, 5)
    nu = stats.beta(5, 2)

    # ---- CONFIGURATION ----
    alpha = 1.0         # complementarity strength
    gamma = 2.0         # mismatch penalty strength
    n_pieces = 40       # piecewise linear pieces
    n_bins = 20         # histogram constraint bins
    p_values = [0.5, 1.0, 2.0, 4.0]  # mismatch exponents
    # -----------------------

    results = {}

    for p in p_values:
        print(f"\n{'='*70}")
        print(f"  p = {p}:  c(x,y) = -{alpha}*xy + {gamma}*|x-y|^{p}")
        is_concave = "CONCAVE (breaks assortative matching)" if p < 1 else \
                     "LINEAR" if p == 1 else "CONVEX (reinforces assortative matching)"
        print(f"  Mismatch penalty is {is_concave}")
        print(f"{'='*70}")

        plm = PLMap(n_pieces, mu)
        all_res, best_name = solve_al(
            plm, nu, alpha, gamma, p, n_bins=n_bins,
            rho_init=50.0, max_outer=50, tol_cv=5e-3, verbose=True)

        K = kant_lp(alpha, gamma, p, mu, nu)

        results[p] = {
            'all_res': all_res, 'best_name': best_name,
            'K': K, 'plm': plm,
        }

        print(f"\n  Best: {best_name}")
        for nm, r in all_res.items():
            star = ' ★' if nm == best_name else ''
            print(f"    {nm:>14}: J={r['J']:.5f}  cv={r['cv']:.2e}  "
                  f"{r['mono']}{star}")
        if K is not None:
            print(f"  Kantorovich LB: {K:.5f}")
            gap = all_res[best_name]['J'] - K
            print(f"  Monge-Kant gap: {gap:.5f} "
                  f"({100*abs(gap)/abs(K):.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'p':>4}  {'Best J':>10}  {'Mono?':>22}  "
          f"{'Kant LB':>10}  {'Gap%':>6}  {'Best init':>14}")
    print(f"{'-'*70}")
    for p in p_values:
        r = results[p]
        res = r['all_res'][r['best_name']]
        K = r['K']
        gap_pct = 100*abs(res['J']-K)/abs(K) if K else 0
        print(f"  {p:4.1f}  {res['J']:10.5f}  {res['mono']:>22}  "
              f"{K:10.5f}  {gap_pct:5.1f}%  {r['best_name']:>14}")
    print(f"{'='*70}")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\nGenerating figures...")

    eps = 2e-3
    x_fine = mu.ppf(np.linspace(eps, 1-eps, 1000))

    colors_init = {
        'quantile': '#2196F3', 'anti-quantile': '#E91E63',
        'V-shape': '#4CAF50', 'tent': '#9C27B0'}
    colors_p = {0.5: '#9C27B0', 1.0: '#E91E63', 2.0: '#4CAF50', 4.0: '#2196F3'}

    # ---- Fig 1: The money figure. One panel per p, all inits overlaid. ----
    fig, axes = plt.subplots(1, len(p_values), figsize=(5*len(p_values), 5.5),
                             sharey=True)
    for idx, p in enumerate(p_values):
        ax = axes[idx]
        r = results[p]
        plm_p = r['plm']

        for nm, res in r['all_res'].items():
            s_vals = plm_p.eval(res['a'], x_fine)
            is_best = (nm == r['best_name'])
            lw = 2.5 if is_best else 1.0
            alp = 1.0 if is_best else 0.35
            label = f'{nm}: {res["J"]:.4f}'
            if is_best:
                label += ' ★'
            ax.plot(x_fine, s_vals, color=colors_init.get(nm, '#888'),
                    lw=lw, alpha=alp, label=label)

        ax.plot([0,1],[0,1], ':', color='gray', lw=0.8, alpha=0.3)

        concavity = 'concave' if p < 1 else ('linear' if p == 1 else 'convex')
        ax.set_title(f'$p = {p}$  ({concavity} penalty)\n'
                     f'Kant LB: {r["K"]:.4f}', fontsize=12)
        ax.set_xlabel('Worker skill $x$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Job assignment $y = s(x)$', fontsize=11)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.85)

    plt.suptitle('Labor Market Matching: $c(x,y) = -xy + 2|x{-}y|^p$\n'
                 'Workers: Beta(2,5) $\\to$ Jobs: Beta(5,2)',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig('fig1_labor_market_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig1_labor_market_maps.png")

    # ---- Fig 2: Best map for each p on one axis ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    q_ref = nu.ppf(mu.cdf(x_fine))
    ax1.plot(x_fine, q_ref, 'k--', lw=2, label='Quantile map (Brenier)', zorder=10)
    for p in p_values:
        r = results[p]
        a = r['all_res'][r['best_name']]['a']
        s_vals = r['plm'].eval(a, x_fine)
        ax1.plot(x_fine, s_vals, color=colors_p[p], lw=1.8,
                 label=f'$p={p}$ ({r["best_name"]})')
    ax1.plot([0,1],[0,1], ':', color='gray', lw=0.8, alpha=0.3)
    ax1.set_xlabel('Worker skill $x$', fontsize=12)
    ax1.set_ylabel('Job assignment $s(x)$', fontsize=12)
    ax1.set_title('Best Transport Map by Penalty Exponent', fontsize=13)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.85)

    # Distributions
    xd = np.linspace(0.001, 0.999, 500)
    ax2.fill_between(xd, mu.pdf(xd), alpha=0.25, color='#2196F3')
    ax2.plot(xd, mu.pdf(xd), '#2196F3', lw=2, label='Workers $\\mu$: Beta(2,5)')
    ax2.fill_between(xd, nu.pdf(xd), alpha=0.25, color='#E91E63')
    ax2.plot(xd, nu.pdf(xd), '#E91E63', lw=2, label='Jobs $\\nu$: Beta(5,2)')
    ax2.set_xlabel('Skill / Requirement level', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Skills Gap Economy', fontsize=13)
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig2_best_maps_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig2_best_maps_overlay.png")

    # ---- Fig 3: Bar chart of objectives ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(p_values))
    w = 0.18

    for i, nm in enumerate(['quantile', 'anti-quantile', 'tent', 'V-shape']):
        vals = []
        for p in p_values:
            r = results[p]['all_res']
            vals.append(r[nm]['J'] if nm in r else np.nan)
        ax.bar(x_pos + (i - 1.5)*w, vals, w,
               label=nm, color=colors_init.get(nm, '#888'), alpha=0.85)

    # Kantorovich
    k_vals = [results[p]['K'] for p in p_values]
    ax.bar(x_pos + 2.5*w, k_vals, w, label='Kantorovich LP',
           color='#FF9800', alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$p={p}$' for p in p_values], fontsize=12)
    ax.set_ylabel('Objective $J(s)$ (lower = better)', fontsize=12)
    ax.set_title('Local Optima Across Initializations and Penalty Exponents',
                 fontsize=13)
    ax.legend(fontsize=9, ncol=3); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig3_objectives_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig3_objectives_comparison.png")

    # ---- Fig 4: Pushforward check for best maps ----
    fig, axes = plt.subplots(1, len(p_values), figsize=(5*len(p_values), 4.5),
                             sharey=True)
    yd = np.linspace(0.01, 0.99, 500)
    for idx, p in enumerate(p_values):
        ax = axes[idx]
        r = results[p]
        a = r['all_res'][r['best_name']]['a']
        plm_p = r['plm']

        u_c = np.linspace(eps, 1-eps, 5000)
        x_c = mu.ppf(u_c)
        s_c = plm_p.eval(a, x_c)
        ax.hist(s_c, bins=50, density=True, alpha=0.4, color=colors_p[p],
                label=f'$s_\\#\\mu$ ($p={p}$)')
        ax.plot(yd, nu.pdf(yd), 'k-', lw=2, label='Target $g(y)$')
        ax.set_xlabel('$y$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'$p={p}$: {r["best_name"]}', fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Pushforward Check: $s_\\#\\mu$ vs. $\\nu$', fontsize=13)
    plt.tight_layout()
    plt.savefig('fig4_pushforward_check.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig4_pushforward_check.png")

    print("\nDone. All figures saved to current directory.")


if __name__ == '__main__':
    main()
