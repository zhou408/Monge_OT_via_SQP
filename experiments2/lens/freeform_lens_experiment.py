"""
Beyond Brenier: SQP for General-Cost Monge Optimal Transport
=============================================================
Experiment: 1D Freeform Lens Design via Optimal Transport

PHYSICS:
  A cylindrical (1D) freeform lens redirects a parallel beam with source
  intensity f(x) into a target intensity g(y) on a screen.

  The lens has refractive index n (e.g., n=1.5 for glass). A ray entering
  at position x on the lens is refracted to hit position y on a target screen
  at distance D. By Snell's law in the paraxial-to-nonparaxial regime, the
  optical path length (OPL) through the system determines a cost function.

  For a single plano-freeform lens with flat entry face and curved exit face,
  the cost of sending a ray from source position x to target position y is:

    c(x,y) = n * d(x) + sqrt((x - y)^2 + D^2)

  where d(x) is the lens thickness at x. Since d(x) is a design variable
  (not fixed), the OT formulation absorbs it into the potential, and the
  *effective transport cost* reduces to:

    c(x,y) = sqrt((x - y)^2 + D^2)           ... (Euclidean OPL)

  This is the distance from (x, 0) to (y, D): the free-space propagation
  cost of a ray. It equals |x-y| when D->0 (L1 cost) and approaches
  D + (x-y)^2/(2D) when |x-y| << D (quadratic approximation).

  For the REFLECTOR problem (point source to far field), the cost on the
  sphere is c(x,y) = -log|x - y|, which is genuinely non-quadratic.

  We implement BOTH:
    (A) Euclidean lens cost:     c(x,y) = sqrt((x-y)^2 + D^2)
    (B) Logarithmic reflector:   c(x,y) = -log(|x-y| + eps)

  These are compared against the quadratic cost c(x,y) = (x-y)^2 to show
  how the optimal map changes with the cost structure.

SOURCE:  f(x) = Uniform on [-1, 1]  (collimated beam, uniform intensity)
TARGET:  g(y) chosen to be an interesting non-uniform pattern:
         - "ring": bimodal (two peaks, like a ring projection in 1D)
         - "logo": a prescribed intensity pattern

Dependencies: numpy, scipy, matplotlib
Usage: python freeform_lens_experiment.py
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, linprog
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Cost functions
# =============================================================================

def cost_quadratic(x, y):
    """Standard quadratic cost (Brenier regime)."""
    return (x - y)**2

def cost_lens(x, y, D=1.0):
    """Euclidean lens cost: free-space ray propagation distance.
    c(x,y) = sqrt((x-y)^2 + D^2).
    This is the L2 distance from (x,0) to (y,D).
    Non-quadratic: behaves like D + (x-y)^2/(2D) for small |x-y|,
    but like |x-y| for large |x-y|. The transition happens at |x-y| ~ D.
    """
    return np.sqrt((x - y)**2 + D**2)

def cost_log_reflector_s1(theta_x, theta_y):
    """Reflector antenna cost on S^1: c(theta_x, theta_y) = -log|x-y|
    where x = (cos theta_x, sin theta_x), y = (cos theta_y, sin theta_y)
    are points on the unit circle.

    |x - y| = 2 sin(d/2), where d is the shorter arc distance.

    This is the physically correct cost from Wang (1996, 2004) on its
    natural domain. Well-posed when source and target arcs are disjoint
    (cost is bounded). Satisfies the MTW condition with uniformly positive
    cost-sectional curvature (Loeper, 2009, 2011).
    """
    d = np.abs(theta_x - theta_y)
    d = np.minimum(d, 2*np.pi - d)
    cd = 2.0 * np.sin(d / 2.0)
    return -np.log(np.maximum(cd, 1e-15))


def cost_monge(x, y, eps=1e-4):
    """Original Monge cost: c(x,y) = |x-y|.
    The founding problem of optimal transport (Monge 1781).
    NOT a twist cost: c_xy = 0 a.e., undefined at x=y.
    Monge maps are famously non-unique for this cost.
    We use a smooth approximation sqrt((x-y)^2 + eps^2) to enable
    gradient-based optimization; this converges to |x-y| as eps -> 0.
    """
    return np.sqrt((x - y)**2 + eps**2)


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

def make_source_target(name='ring', n_pts=5000):
    """Create source and target distributions on [-1, 1].

    Source: uniform on [-1, 1] (collimated beam).
    Target: various interesting patterns.
    """
    x_range = (-1.0, 1.0)

    if name == 'ring':
        # Bimodal: two Gaussians (1D cross-section of a ring)
        target = lambda y: (0.4 * np.exp(-((y - 0.5)/0.15)**2) +
                            0.4 * np.exp(-((y + 0.5)/0.15)**2) +
                            0.05)
    elif name == 'concentrated':
        # Strongly concentrated in center (like focusing)
        target = lambda y: np.exp(-y**2 / 0.04) + 0.02
    elif name == 'asymmetric':
        # Asymmetric: shifted Gaussian + uniform floor
        target = lambda y: 0.6 * np.exp(-((y - 0.4)/0.2)**2) + 0.15
    elif name == 'edge':
        # Edge-lit: concentrated at boundaries
        target = lambda y: (np.exp(-((y - 0.85)/0.08)**2) +
                            np.exp(-((y + 0.85)/0.08)**2) + 0.03)
    else:
        raise ValueError(f"Unknown target: {name}")

    # Normalize target to be a proper density on [-1, 1]
    y_fine = np.linspace(-1, 1, 10000)
    dy = 2.0 / 9999
    raw = target(y_fine)
    Z = np.sum(raw) * dy
    target_normed = lambda y, t=target, z=Z: t(y) / z

    return x_range, target_normed


def target_cdf_inv(target_density, n_inv=10000):
    """Compute inverse CDF of target density on [-1, 1] numerically."""
    y = np.linspace(-1, 1, n_inv)
    dy = 2.0 / (n_inv - 1)
    pdf = target_density(y)
    cdf = np.cumsum(pdf) * dy
    cdf = cdf / cdf[-1]  # normalize
    cdf[0] = 0.0
    # Invert
    inv_cdf = interp1d(cdf, y, kind='linear', bounds_error=False,
                       fill_value=(-1, 1))
    return inv_cdf


# =============================================================================
# Pushforward constraint (histogram-based)
# =============================================================================

def pushforward_hist(a, plm, target_density, n_bins, y_range=(-1, 1),
                     n_fine=4000):
    """Histogram pushforward: mu(s^{-1}(B_j)) - nu(B_j) for each bin."""
    x_pts = np.linspace(plm.x_grid[0], plm.x_grid[-1], n_fine)
    dx = (plm.x_grid[-1] - plm.x_grid[0]) / (n_fine - 1)
    # Source is uniform on [-1,1], density = 0.5
    source_density = 0.5

    s_vals = plm.eval(a, x_pts)

    bin_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
    tau = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (s_vals >= bin_edges[b]) & (s_vals < bin_edges[b + 1])
        source_mass = np.sum(mask) * dx * source_density
        # Target mass in bin
        y_mid = np.linspace(bin_edges[b], bin_edges[b + 1], 100)
        dy_mid = (bin_edges[b + 1] - bin_edges[b]) / 99
        target_mass = np.sum(target_density(y_mid)) * dy_mid
        tau[b] = source_mass - target_mass
    return tau


def jac_hist(a, plm, target_density, n_bins, fd_eps=5e-7):
    tau0 = pushforward_hist(a, plm, target_density, n_bins)
    J = np.zeros((n_bins, len(a)))
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        J[:, j] = (pushforward_hist(ap, plm, target_density, n_bins) - tau0) / fd_eps
    return J


def objective(a, plm, cost_func, n_quad=4000):
    """J(s) = int c(x, s(x)) * f(x) dx, f=0.5 on [-1,1]."""
    x_pts = np.linspace(plm.x_grid[0], plm.x_grid[-1], n_quad)
    dx = (plm.x_grid[-1] - plm.x_grid[0]) / (n_quad - 1)
    s_vals = plm.eval(a, x_pts)
    return np.sum(dx * 0.5 * cost_func(x_pts, s_vals))


def grad_obj_fd(a, plm, cost_func, fd_eps=1e-6):
    g = np.zeros(len(a))
    f0 = objective(a, plm, cost_func)
    for j in range(len(a)):
        ap = a.copy(); ap[j] += fd_eps
        g[j] = (objective(ap, plm, cost_func) - f0) / fd_eps
    return g


# =============================================================================
# Augmented Lagrangian solver
# =============================================================================

def solve_al(plm, target_density, cost_func, n_bins=25,
             inits=None, rho_init=50.0, rho_max=1e5,
             max_outer=50, tol_cv=5e-3, y_range=(-1, 1),
             verbose=True, enforce_monotone=False):

    n1 = plm.n + 1
    inv_cdf = target_cdf_inv(target_density)

    if inits is None:
        # Quantile map: source CDF = (x+1)/2 on [-1,1]
        u = (plm.x_grid + 1) / 2.0
        q_map = inv_cdf(u)
        rev_map = inv_cdf(1.0 - u)
        tent_map = inv_cdf(1.0 - np.abs(2*u - 1))
        v_map = inv_cdf(np.abs(2*u - 1))
        inits = {
            'quantile': q_map,
            'anti-quantile': rev_map,
            'tent': tent_map,
            'V-shape': v_map,
        }

    all_results = {}

    for name, a0 in inits.items():
        if verbose:
            print(f"    {name}...", end='', flush=True)

        a = np.clip(a0.copy(), y_range[0] + 1e-4, y_range[1] - 1e-4)

        # For monotone enforcement, only use monotone initializations
        if enforce_monotone:
            ds0 = np.diff(a)
            if np.any(ds0 < -1e-6):
                # Skip non-monotone initializations
                if verbose:
                    print(f"  SKIPPED (non-monotone init)")
                continue

        lam = np.zeros(n_bins)
        rho = rho_init
        prev_cv = np.inf
        hist = {'obj': [], 'cv_inf': []}

        for outer in range(max_outer):
            def auglag(a_f):
                J = objective(a_f, plm, cost_func)
                tau = pushforward_hist(a_f, plm, target_density, n_bins,
                                       y_range=y_range)
                return J + np.dot(lam, tau) + 0.5*rho*np.dot(tau, tau)

            def auglag_grad(a_f):
                gJ = grad_obj_fd(a_f, plm, cost_func)
                tau = pushforward_hist(a_f, plm, target_density, n_bins,
                                       y_range=y_range)
                Jc = jac_hist(a_f, plm, target_density, n_bins)
                return gJ + Jc.T @ (lam + rho*tau)

            bounds = [(y_range[0] + 1e-4, y_range[1] - 1e-4)] * n1
            res = minimize(auglag, a, jac=auglag_grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-11})
            a = res.x

            # Isotonic projection: enforce a[0] <= a[1] <= ... <= a[N]
            if enforce_monotone:
                for k in range(1, n1):
                    if a[k] < a[k-1]:
                        a[k] = a[k-1] + 1e-8

            tau = pushforward_hist(a, plm, target_density, n_bins, y_range=y_range)
            cv1 = np.linalg.norm(tau, 1)
            cvinf = np.linalg.norm(tau, np.inf)
            J_val = objective(a, plm, cost_func)

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


# =============================================================================
# Kantorovich LP
# =============================================================================

def kant_lp(cost_func, target_density, n_disc=100, x_range=(-1, 1)):
    x = np.linspace(x_range[0], x_range[1], n_disc)
    y = np.linspace(x_range[0], x_range[1], n_disc)
    dx = (x_range[1] - x_range[0]) / (n_disc - 1)

    # Source marginal: uniform, p_i = dx * 0.5
    p = np.full(n_disc, dx * 0.5)
    p /= p.sum()

    # Target marginal
    q = target_density(y) * dx
    q /= q.sum()

    C = cost_func(x[:, None], y[None, :]).flatten()
    A = lil_matrix((2*n_disc, n_disc**2))
    for i in range(n_disc):
        A[i, i*n_disc:(i+1)*n_disc] = 1.0
    for j in range(n_disc):
        A[n_disc+j, j::n_disc] = 1.0

    res = linprog(C, A_eq=A.tocsc(), b_eq=np.concatenate([p, q]),
                  bounds=(0, None), method='highs')
    return res.fun if res.success else None


# =============================================================================
# Exact solution for twisted costs in 1D
# =============================================================================

def exact_monotone_map(target_density, x_range=(-1, 1), n_pts=2000):
    """Compute the exact optimal map for ANY twisted cost in 1D.

    KEY INSIGHT: In 1D with absolutely continuous measures, the twist
    condition implies the optimal map is monotone increasing. But there
    is exactly ONE monotone increasing map pushing mu to nu — the
    quantile coupling s(x) = F_nu^{-1}(F_mu(x)). This is INDEPENDENT
    of the cost function.

    Different twisted costs produce the SAME optimal map; only the
    optimal transport VALUE differs.

    Returns:
        x_grid: evaluation points
        s_exact: exact optimal map values s(x)
        inv_cdf: the inverse CDF function (for reuse)
    """
    a, b = x_range
    x_grid = np.linspace(a, b, n_pts)

    # Source CDF: uniform on [a, b]
    F_mu = (x_grid - a) / (b - a)

    # Target inverse CDF
    inv_cdf = target_cdf_inv(target_density)
    s_exact = inv_cdf(F_mu)

    return x_grid, s_exact, inv_cdf


def exact_objective(cost_func, target_density, x_range=(-1, 1), n_pts=4000):
    """Compute the exact Monge objective for a twisted cost using the
    quantile coupling (the unique optimal monotone map in 1D).

    J* = int c(x, F_nu^{-1}(F_mu(x))) * f(x) dx
    """
    x_grid, s_exact, _ = exact_monotone_map(target_density, x_range, n_pts)
    dx = (x_range[1] - x_range[0]) / (n_pts - 1)
    f_x = 1.0 / (x_range[1] - x_range[0])  # uniform density
    c_vals = cost_func(x_grid, s_exact)
    return np.sum(c_vals * f_x * dx)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  BEYOND BRENIER: 1D Optimal Transport with General Costs")
    print("  Including Monge's original |x-y| cost (1781)")
    print("  Source: Uniform[-1,1]  Target: ring/concentrated/asymmetric")
    print("=" * 70)

    # ---- CONFIGURATION ----
    n_pieces = 40
    n_bins = 20
    D_lens = 1.0           # screen distance for lens cost
    targets = ['ring', 'concentrated', 'asymmetric']
    # -----------------------

    costs = {
        'quadratic': lambda x, y: cost_quadratic(x, y),
        f'lens D={D_lens}': lambda x, y: cost_lens(x, y, D=D_lens),
        'Monge $|x{-}y|$': lambda x, y: cost_monge(x, y),
    }
    cost_colors = {
        'quadratic': '#2196F3',
        f'lens D={D_lens}': '#E91E63',
        'Monge $|x{-}y|$': '#FF9800',
    }

    all_results = {}

    for tname in targets:
        print(f"\n{'='*70}")
        print(f"  Target: {tname}")
        print(f"{'='*70}")

        x_range, target_density = make_source_target(tname)
        plm = PLMap(n_pieces, x_range)

        all_results[tname] = {}

        # Compute exact solution (quantile coupling — same for all twisted costs)
        x_exact, s_exact, _ = exact_monotone_map(target_density, x_range)
        all_results[tname]['_exact'] = {'x': x_exact, 's': s_exact}
        print(f"  Exact monotone map computed (quantile coupling)")

        for cname, cfunc in costs.items():
            print(f"\n  Cost: {cname}")

            # Twisted costs (quadratic, lens, log-reflector): enforce monotonicity
            # since c_xy < 0 guarantees the optimal map is monotone increasing.
            # Monge |x-y| cost: NOT twisted, allow non-monotone maps.
            is_twisted = 'Monge' not in cname
            ar, best = solve_al(
                plm, target_density, cfunc, n_bins=n_bins,
                rho_init=50.0, max_outer=50, tol_cv=5e-3,
                y_range=x_range, verbose=True,
                enforce_monotone=is_twisted)

            K = kant_lp(cfunc, target_density, n_disc=100, x_range=x_range)

            # Exact objective for twisted costs
            J_exact = exact_objective(cfunc, target_density, x_range)

            all_results[tname][cname] = {
                'all_res': ar, 'best_name': best, 'K': K, 'plm': plm,
                'J_exact': J_exact,
            }

            best_J = ar[best]['J']
            print(f"  Best: {best}, J={best_J:.5f}, "
                  f"Kant={K:.5f}" if K else f"  Best: {best}, J={best_J:.5f}")
            print(f"  Exact (quantile coupling): J*={J_exact:.5f}")
            if K:
                gap = best_J - K
                print(f"  Gap to Kant: {gap:.5f} ({100*abs(gap)/abs(K):.1f}%)")
                gap_exact = best_J - J_exact
                print(f"  Gap to exact: {gap_exact:.5f} ({100*abs(gap_exact)/abs(J_exact):.1f}%)")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\nGenerating figures...")

    x_fine = np.linspace(-1, 1, 1000)

    # ---- Fig 1: THE MONEY FIGURE. Best maps for each cost, one column per target ----
    fig, axes = plt.subplots(2, len(targets),
                             figsize=(6*len(targets), 10))

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
            K = r['K']
            J_ex = r['J_exact']

            s_vals = plm_p.eval(a, x_fine)
            label = f'{cname}: J={J:.4f} (exact={J_ex:.4f})'
            ax_map.plot(x_fine, s_vals, color=cost_colors[cname],
                        lw=2, label=label)

            # Pushforward histogram
            n_check = 5000
            x_check = np.linspace(-1, 1, n_check)
            s_check = plm_p.eval(a, x_check)
            ax_push.hist(s_check, bins=50, density=True, alpha=0.3,
                         color=cost_colors[cname])

        # Exact quantile coupling (ground truth for all twisted costs)
        ex = all_results[tname]['_exact']
        ax_map.plot(ex['x'], ex['s'], 'k--', lw=1.5, alpha=0.7,
                    label='exact (quantile coupling)', zorder=0)

        # Identity line
        ax_map.plot([-1, 1], [-1, 1], ':', color='gray', lw=0.8, alpha=0.4)

        # Target density on pushforward plot
        yd = np.linspace(-1, 1, 500)
        ax_push.plot(yd, target_density(yd), 'k-', lw=2.5, label='Target $g(y)$')

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

    plt.suptitle('1D Freeform Lens: Optimal Maps Under Different Cost Functions\n'
                 'Source: Uniform$[-1,1]$', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig1_lens_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig1_lens_maps.png")

    # ---- Fig 2: All initializations for Monge |x-y| cost + ring target ----
    tname = 'ring'
    cname = 'Monge $|x{-}y|$'
    if cname in all_results[tname]:
        r = all_results[tname][cname]
        plm_p = r['plm']
        init_colors = {'quantile': '#2196F3', 'anti-quantile': '#E91E63',
                       'tent': '#9C27B0', 'V-shape': '#4CAF50'}

        n_inits = len(r['all_res'])
        fig, axes = plt.subplots(1, n_inits, figsize=(5*n_inits, 5))
        if n_inits == 1:
            axes = [axes]

        for idx, (nm, res) in enumerate(r['all_res'].items()):
            ax = axes[idx]
            s_vals = plm_p.eval(res['a'], x_fine)
            ax.plot(x_fine, s_vals, color=init_colors.get(nm, '#888'), lw=2)
            ax.plot([-1, 1], [-1, 1], 'k:', alpha=0.3)
            star = ' ★ BEST' if nm == r['best_name'] else ''
            ax.set_title(f'{nm}{star}\n$J = {res["J"]:.5f}$  '
                         f'$|\\tau|_\\infty = {res["cv"]:.2e}$\n{res["mono"]}',
                         fontsize=10,
                         color='red' if nm == r['best_name'] else 'black')
            ax.set_xlabel('$x$'); ax.set_ylabel('$s(x)$')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Local Optima: Monge $|x-y|$ Cost, Ring Target\n'
                     f'Kantorovich LB: {r["K"]:.5f}' if r['K'] else
                     'Local Optima: Monge $|x-y|$ Cost, Ring Target',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig('fig2_local_optima_monge.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("  fig2_local_optima_monge.png")

    # ---- Fig 3: Cost function comparison ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    delta = np.linspace(-2, 2, 500)

    # Cost as function of displacement
    ax1.plot(delta, delta**2, color='#2196F3', lw=2, label='Quadratic: $(x-y)^2$')
    ax1.plot(delta, np.sqrt(delta**2 + D_lens**2), color='#E91E63', lw=2,
             label=f'Lens: $\\sqrt{{(x-y)^2 + {D_lens}^2}}$')
    ax1.plot(delta, np.sqrt(delta**2 + 1e-8), color='#FF9800', lw=2,
             label='Monge: $|x-y|$')
    ax1.set_xlabel('Displacement $x - y$', fontsize=12)
    ax1.set_ylabel('$c(x, y)$', fontsize=12)
    ax1.set_title('Cost Functions', fontsize=13)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 5)

    # Mixed second derivative (twist condition)
    h = 0.01
    x0 = np.linspace(-1, 1, 200)
    y0 = 0.0
    for cname_short, cfunc, color in [
        ('quadratic', cost_quadratic, '#2196F3'),
        ('lens', lambda x, y: cost_lens(x, y, D_lens), '#E91E63'),
        ('Monge', lambda x, y: cost_monge(x, y), '#FF9800'),
    ]:
        d2 = (cfunc(x0+h, y0+h) - cfunc(x0+h, y0-h)
              - cfunc(x0-h, y0+h) + cfunc(x0-h, y0-h)) / (4*h**2)
        ax2.plot(x0, d2, color=color, lw=2, label=cname_short)

    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('$x$ (at $y=0$)', fontsize=12)
    ax2.set_ylabel('$\\partial^2 c / \\partial x \\partial y$', fontsize=12)
    ax2.set_title('Twist Condition: $D^2_{xy} c$', fontsize=13)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # Optimality gaps
    gap_data = {}
    for tname in targets:
        gap_data[tname] = {}
        for cname in costs:
            r = all_results[tname][cname]
            J = r['all_res'][r['best_name']]['J']
            K = r['K']
            gap_data[tname][cname] = (J, K)

    x_pos = np.arange(len(targets))
    w = 0.12
    for i, cname in enumerate(costs):
        monge = [gap_data[t][cname][0] for t in targets]
        kant = [gap_data[t][cname][1] for t in targets]
        ax3.bar(x_pos + (2*i - len(costs) + 1)*w/2, monge, w,
                label=f'Monge: {cname}',
                color=cost_colors[cname], alpha=0.85)
        ax3.bar(x_pos + (2*i - len(costs) + 1)*w/2 + w*len(costs),
                kant, w,
                color=cost_colors[cname], alpha=0.4, hatch='//')

    ax3.set_xticks(x_pos + w)
    ax3.set_xticklabels(targets, fontsize=11)
    ax3.set_ylabel('Objective', fontsize=12)
    ax3.set_title('Monge (solid) vs Kantorovich (hatched)', fontsize=13)
    ax3.legend(fontsize=7, ncol=2); ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fig3_cost_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig3_cost_analysis.png")

    # ---- Fig 4: Convergence ----
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5))
    for col, tname in enumerate(targets):
        ax = axes[col]
        for cname in costs:
            r = all_results[tname][cname]
            best = r['best_name']
            h = r['all_res'][best]['hist']
            ax.semilogy(h['cv_inf'], color=cost_colors[cname], lw=1.8,
                        marker='o', ms=3, label=cname)
        ax.set_xlabel('Outer iteration', fontsize=11)
        ax.set_ylabel('$\\|\\tau\\|_\\infty$', fontsize=11)
        ax.set_title(f'Convergence: {tname}', fontsize=12)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig4_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig4_convergence.png")

    # ---- Fig 5: Local optima grid — all initializations × costs × targets ----
    init_colors_full = {
        'quantile': '#2196F3', 'anti-quantile': '#E91E63',
        'tent': '#9C27B0', 'V-shape': '#4CAF50',
    }

    fig5, axes5 = plt.subplots(
        len(targets), len(costs),
        figsize=(5.5 * len(costs), 4.2 * len(targets)),
        sharex=True, sharey=True,
    )
    # Ensure 2D indexing even if only 1 row or column
    if len(targets) == 1:
        axes5 = axes5[np.newaxis, :]
    if len(costs) == 1:
        axes5 = axes5[:, np.newaxis]

    cost_list = list(costs.keys())

    for row, tname in enumerate(targets):
        for col, cname in enumerate(cost_list):
            ax = axes5[row, col]
            r = all_results[tname][cname]
            plm_p = r['plm']
            cell = r['all_res']
            best = r['best_name']

            # Sort so best is drawn last (on top)
            sorted_inits = sorted(cell.keys(), key=lambda k: k == best)

            for iname in sorted_inits:
                res = cell[iname]
                s_vals = plm_p.eval(res['a'], x_fine)
                is_best = (iname == best)
                lw = 2.8 if is_best else 1.0
                alpha = 1.0 if is_best else 0.35
                star = ' \u2605' if is_best else ''
                cv_ok = res['cv'] < 0.05
                feas = '' if cv_ok else ' [infeas]'
                label = f"{iname}: J={res['J']:.4f}{star}{feas}"
                ax.plot(x_fine, s_vals,
                        color=init_colors_full.get(iname, '#888'),
                        lw=lw, alpha=alpha, label=label)

            ax.plot([-1, 1], [-1, 1], ':', color='gray', lw=0.7, alpha=0.4)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(cname, fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{tname}\n$s(x)$', fontsize=11)
            if row == len(targets) - 1:
                ax.set_xlabel('$x$', fontsize=11)

            ax.legend(fontsize=6.5, loc='upper left',
                      framealpha=0.85, handlelength=1.5)

    plt.suptitle(
        'Local Optima Across Costs and Targets\n'
        'Source: Uniform$[-1,1]$  |  4 initializations each',
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig5_local_optima_grid.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig5_local_optima_grid.png")

    # =========================================================================
    # S^1 REFLECTOR ANTENNA EXPERIMENT
    # =========================================================================
    # The reflector cost c(x,y) = -log|x-y| is naturally posed on the sphere
    # (Wang 1996, 2004; Loeper 2009, 2011). On the flat interval [-1,1], the
    # singularity at x=y makes the Monge problem ill-posed when source and
    # target overlap. On S^1 with disjoint source/target arcs, the cost is
    # bounded, the twist condition holds, and Wang/Loeper regularity applies.
    # =========================================================================

    print(f"\n{'='*70}")
    print("  REFLECTOR ANTENNA ON S^1")
    print("  c(theta_x, theta_y) = -log|x - y|, x,y on unit circle")
    print("  Source arc: upper semicircle, Target arc: lower semicircle")
    print(f"{'='*70}")

    S1_SRC_LO, S1_SRC_HI = 0.15, np.pi - 0.15
    S1_TGT_LO, S1_TGT_HI = np.pi + 0.15, 2*np.pi - 0.15

    def s1_target_ring(theta):
        mid = (S1_TGT_LO + S1_TGT_HI) / 2
        raw = (0.5 * np.exp(-((theta - (mid - 0.5)) / 0.25)**2) +
               0.5 * np.exp(-((theta - (mid + 0.5)) / 0.25)**2) + 0.05)
        return raw

    def s1_target_concentrated(theta):
        mid = (S1_TGT_LO + S1_TGT_HI) / 2
        return np.exp(-((theta - mid) / 0.2)**2) + 0.02

    def s1_target_asymmetric(theta):
        mid = (S1_TGT_LO + S1_TGT_HI) / 2
        return 0.6 * np.exp(-((theta - (mid + 0.3)) / 0.25)**2) + 0.15

    def s1_normalize(g_func, lo, hi):
        t = np.linspace(lo, hi, 10000)
        dt = (hi - lo) / 9999
        vals = g_func(t)
        Z = np.sum(vals) * dt
        return lambda th, g=g_func, z=Z: g(th) / z

    def s1_target_cdf_inv(g_normed, lo, hi, n_inv=20000):
        t = np.linspace(lo, hi, n_inv)
        dt = (hi - lo) / (n_inv - 1)
        pdf = g_normed(t)
        cdf = np.cumsum(pdf) * dt
        cdf /= cdf[-1]; cdf[0] = 0
        return interp1d(cdf, t, bounds_error=False, fill_value=(lo, hi))

    s1_targets = {
        'ring': s1_normalize(s1_target_ring, S1_TGT_LO, S1_TGT_HI),
        'concentrated': s1_normalize(s1_target_concentrated, S1_TGT_LO, S1_TGT_HI),
        'asymmetric': s1_normalize(s1_target_asymmetric, S1_TGT_LO, S1_TGT_HI),
    }

    # S^1 solver infrastructure
    s1_n = 40; s1_n1 = s1_n + 1
    s1_xg = np.linspace(S1_SRC_LO, S1_SRC_HI, s1_n1)
    s1_NQ = 3000; s1_NB = 20; s1_NF = 3000
    s1_x_pts = np.linspace(S1_SRC_LO, S1_SRC_HI, s1_NQ)
    s1_dx = (S1_SRC_HI - S1_SRC_LO) / (s1_NQ - 1)
    s1_f = 1.0 / (S1_SRC_HI - S1_SRC_LO)

    def s1_pleval(a, x): return np.interp(x, s1_xg, a)

    def s1_obj(a):
        sv = s1_pleval(a, s1_x_pts)
        return np.sum(s1_f * cost_log_reflector_s1(s1_x_pts, sv) * s1_dx)

    def s1_gobj(a, eps=1e-6):
        g = np.zeros(s1_n1); f0 = s1_obj(a)
        for j in range(s1_n1):
            ap = a.copy(); ap[j] += eps
            g[j] = (s1_obj(ap) - f0) / eps
        return g

    def s1_pfhist(a, g_func, nb=20):
        xp = np.linspace(S1_SRC_LO, S1_SRC_HI, s1_NF)
        dxf = (S1_SRC_HI - S1_SRC_LO) / (s1_NF - 1)
        sv = s1_pleval(a, xp)
        be = np.linspace(S1_TGT_LO, S1_TGT_HI, nb+1)
        tau = np.zeros(nb)
        for b in range(nb):
            mask = (sv >= be[b]) & (sv < be[b+1])
            sm = np.sum(mask) * dxf * s1_f
            ym = np.linspace(be[b], be[b+1], 100)
            tm = np.sum(g_func(ym)) * (be[b+1]-be[b]) / 99
            tau[b] = sm - tm
        return tau

    def s1_jac(a, g_func, nb=20, eps=5e-7):
        t0 = s1_pfhist(a, g_func, nb)
        J = np.zeros((nb, s1_n1))
        for j in range(s1_n1):
            ap = a.copy(); ap[j] += eps
            J[:, j] = (s1_pfhist(ap, g_func, nb) - t0) / eps
        return J

    def s1_solve(g_func, a0, max_outer=50):
        a = a0.copy()
        lam = np.zeros(s1_NB); rho = 50.0; pcv = 1e9
        hist = {'obj': [], 'cv_inf': []}
        for outer in range(max_outer):
            def al(af):
                J = s1_obj(af); tau = s1_pfhist(af, g_func)
                return J + lam@tau + 0.5*rho*(tau@tau)
            def alg(af):
                g = s1_gobj(af); tau = s1_pfhist(af, g_func)
                Jc = s1_jac(af, g_func)
                return g + Jc.T @ (lam + rho*tau)
            bounds = [(S1_TGT_LO, S1_TGT_HI)] * s1_n1
            res = minimize(al, a, jac=alg, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            a = res.x
            # Enforce monotonicity (twist holds for reflector on S^1)
            for k in range(1, s1_n1):
                if a[k] < a[k-1]: a[k] = a[k-1] + 1e-8
            tau = s1_pfhist(a, g_func)
            cvi = np.linalg.norm(tau, np.inf)
            cv1 = np.linalg.norm(tau, 1)
            Jv = s1_obj(a)
            hist['obj'].append(Jv); hist['cv_inf'].append(cvi)
            if cvi < 5e-3: break
            lam += rho * tau
            if cv1 > 0.25 * pcv: rho = min(rho*2, 1e5)
            pcv = cv1
        mono = 'mono ↑' if np.all(np.diff(a) >= -1e-6) else 'non-mono'
        return a, Jv, cvi, mono, hist

    # Run for each S^1 target
    s1_results = {}
    for tname, g_func in s1_targets.items():
        print(f"\n  Target: {tname}")
        inv_cdf = s1_target_cdf_inv(g_func, S1_TGT_LO, S1_TGT_HI)
        u = (s1_xg - S1_SRC_LO) / (S1_SRC_HI - S1_SRC_LO)
        a0 = inv_cdf(u)

        # Exact solution (quantile coupling)
        x_ex = np.linspace(S1_SRC_LO, S1_SRC_HI, 2000)
        F_mu = (x_ex - S1_SRC_LO) / (S1_SRC_HI - S1_SRC_LO)
        s_exact = inv_cdf(F_mu)
        from scipy.integrate import trapezoid
        J_exact = trapezoid(s1_f * cost_log_reflector_s1(x_ex, s_exact), x_ex)

        # SQP solve
        a_sol, J_sol, cv, mono, hist = s1_solve(g_func, a0)
        gap = 100 * abs(J_sol - J_exact) / abs(J_exact) if J_exact != 0 else 0
        print(f"    SQP:   J={J_sol:.6f}, cv={cv:.2e}, {mono}")
        print(f"    Exact: J={J_exact:.6f}, gap={gap:.2f}%")

        s1_results[tname] = {
            'a': a_sol, 'J': J_sol, 'cv': cv, 'mono': mono, 'hist': hist,
            'J_exact': J_exact, 'x_exact': x_ex, 's_exact': s_exact,
            'g_func': g_func, 'inv_cdf': inv_cdf,
        }

    # ---- Fig S1: Reflector antenna on the circle ----
    fig, axes = plt.subplots(1, len(s1_targets) + 1,
                             figsize=(5*(len(s1_targets)+1), 5))

    # Panel 0: Circle geometry with transport rays (ring target)
    ax = axes[0]
    theta_circle = np.linspace(0, 2*np.pi, 500)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', lw=0.5, alpha=0.3)
    ts = np.linspace(S1_SRC_LO, S1_SRC_HI, 200)
    ax.plot(np.cos(ts), np.sin(ts), '-', color='#2196F3', lw=5, alpha=0.4,
            label='source')
    tt = np.linspace(S1_TGT_LO, S1_TGT_HI, 200)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='#E91E63', lw=5, alpha=0.4,
            label='target')
    # Transport rays for ring target
    r0 = s1_results['ring']
    n_show = 15
    xs = np.linspace(S1_SRC_LO, S1_SRC_HI, n_show)
    ys = s1_pleval(r0['a'], xs)
    for k in range(n_show):
        ax.plot([np.cos(xs[k]), np.cos(ys[k])],
                [np.sin(xs[k]), np.sin(ys[k])],
                '-', color='#059669', alpha=0.35, lw=0.8)
    ax.set_aspect('equal')
    ax.set_title('$S^1$ reflector:\ntransport rays', fontsize=12)
    ax.legend(fontsize=8, loc='center')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.2)

    # Panels 1-3: Maps for each target
    s1_x_fine = np.linspace(S1_SRC_LO, S1_SRC_HI, 500)
    for col, tname in enumerate(s1_targets):
        ax = axes[col + 1]
        r = s1_results[tname]
        ax.plot(r['x_exact'], r['s_exact'], 'k--', lw=1.5, alpha=0.6,
                label=f'exact: J={r["J_exact"]:.4f}')
        ax.plot(s1_xg, r['a'], '-', color='#4CAF50', lw=2.5,
                label=f'SQP: J={r["J"]:.4f}')
        gap = 100*abs(r['J'] - r['J_exact'])/abs(r['J_exact'])
        ax.set_title(f'Target: {tname}\ngap={gap:.2f}%', fontsize=12)
        ax.set_xlabel('$\\theta_x$ (source)', fontsize=11)
        ax.set_ylabel('$\\theta_y = s(\\theta_x)$', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reflector Antenna on $S^1$: $c(\\theta_x, \\theta_y) = '
                 '-\\log|x - y|$\n'
                 'Source: upper arc, Target: lower arc (disjoint supports)',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig('fig6_reflector_s1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig6_reflector_s1.png")

    # ---- Summary table ----
    print(f"\n{'='*100}")
    print(f"  {'Target':>14} {'Cost':>16} {'Best init':>14} "
          f"{'SQP J':>10} {'Exact J*':>10} {'Kant LB':>10} {'SQP gap%':>8} {'Mono':>20}")
    print(f"{'-'*100}")
    for tname in targets:
        for cname in costs:
            r = all_results[tname][cname]
            best = r['best_name']
            res = r['all_res'][best]
            K = r['K']
            J_ex = r['J_exact']
            gap_exact = 100*abs(res['J'] - J_ex)/abs(J_ex) if J_ex and J_ex != 0 else 0
            print(f"  {tname:>14} {cname:>16} {best:>14} "
                  f"{res['J']:10.5f} {J_ex:10.5f} {K:10.5f} {gap_exact:7.2f}% "
                  f"{res['mono']:>20}")
    print(f"{'='*100}")
    print("\n  NOTE: For twisted costs (quadratic, lens), the exact optimal map")
    print("  in 1D is the quantile coupling F_nu^{-1}(F_mu(x)), independent")
    print("  of the cost. Only the optimal VALUE differs across costs.")
    print("  The Monge |x-y| cost is NOT twisted, so the optimal map may differ.")
    print("  The reflector cost is run on S^1 with disjoint arcs (see above).")

    # S^1 reflector summary
    print(f"\n  S^1 Reflector Antenna:")
    print(f"  {'Target':>14} {'SQP J':>10} {'Exact J*':>10} {'Gap%':>8}")
    print(f"  {'-'*50}")
    for tname in s1_results:
        r = s1_results[tname]
        gap = 100*abs(r['J'] - r['J_exact'])/abs(r['J_exact']) if r['J_exact'] != 0 else 0
        print(f"  {tname:>14} {r['J']:10.5f} {r['J_exact']:10.5f} {gap:7.2f}%")

    print("\nDone. All figures saved.")


if __name__ == '__main__':
    main()
