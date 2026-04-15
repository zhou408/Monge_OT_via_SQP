"""
Compute the TRUE optimal transport map for the McCann concave cost problem
via Kantorovich LP on a fine grid.

Cost: c(x,y) = sqrt(2|x-y|)
Source: mu = max(sin(pi*x/5), 0) / Z on [-10,10]
Target: nu = max(-sin(pi*x/5), 0) / Z on [-10,10]

This gives us ground truth to compare against any analytical candidate.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# ── Densities ──
def rho_plus(x):
    return np.maximum(np.sin(np.pi * x / 5.0), 0.0)

def rho_minus(x):
    return np.maximum(-np.sin(np.pi * x / 5.0), 0.0)


def solve_kantorovich(n_disc=200, verbose=True):
    """Solve the Kantorovich LP on a fine grid."""

    # Source support: (-10,-5) ∪ (0,5)
    # Target support: (-5,0) ∪ (5,10)
    # Use quantile-like placement for better resolution

    n_src = n_disc
    n_tgt = n_disc

    # Source points on source support
    x_left = np.linspace(-10 + 0.01, -5 - 0.01, n_src // 2)
    x_right = np.linspace(0.01, 5 - 0.01, n_src // 2)
    x = np.concatenate([x_left, x_right])

    # Target points on target support
    y_left = np.linspace(-5 + 0.01, -0.01, n_tgt // 2)
    y_right = np.linspace(5.01, 10 - 0.01, n_tgt // 2)
    y = np.concatenate([y_left, y_right])

    # Source weights (proportional to density)
    f_x = rho_plus(x)
    p = f_x / f_x.sum()  # normalize to probability

    # Target weights
    g_y = rho_minus(y)
    q = g_y / g_y.sum()

    # Cost matrix
    C = np.sqrt(2.0 * np.abs(x[:, None] - y[None, :]))

    if verbose:
        print(f"  LP size: {n_src} sources × {n_tgt} targets = {n_src*n_tgt} variables")

    # Solve LP: min c^T π s.t. π1 = p, π^T1 = q, π >= 0
    c_vec = C.flatten()

    # Build constraint matrix
    n_vars = n_src * n_tgt
    A_eq = lil_matrix((n_src + n_tgt, n_vars))

    # Row marginal constraints
    for i in range(n_src):
        A_eq[i, i*n_tgt:(i+1)*n_tgt] = 1.0

    # Column marginal constraints
    for j in range(n_tgt):
        A_eq[n_src + j, j::n_tgt] = 1.0

    b_eq = np.concatenate([p, q])

    if verbose:
        print(f"  Solving LP...")
    t0 = time.time()

    res = linprog(c_vec, A_eq=A_eq.tocsc(), b_eq=b_eq,
                  bounds=(0, None), method='highs',
                  options={'presolve': True, 'time_limit': 300})

    elapsed = time.time() - t0

    if not res.success:
        print(f"  LP FAILED: {res.message}")
        return None

    if verbose:
        print(f"  LP solved in {elapsed:.1f}s, J_LP = {res.fun:.6f}")

    # Extract transport plan
    pi_mat = res.x.reshape(n_src, n_tgt)

    # Extract approximate transport map: s(x_i) = E[y | x_i] (barycentric projection)
    s_map = np.zeros(n_src)
    for i in range(n_src):
        row = pi_mat[i, :]
        if row.sum() > 1e-15:
            s_map[i] = np.dot(row, y) / row.sum()
        else:
            s_map[i] = np.nan

    # Also find the DETERMINISTIC map: for each x_i, find the y_j with max weight
    s_det = np.zeros(n_src)
    for i in range(n_src):
        row = pi_mat[i, :]
        j_max = np.argmax(row)
        s_det[i] = y[j_max]

    return {
        'x': x, 'y': y, 'p': p, 'q': q,
        'pi': pi_mat,
        's_bary': s_map,  # barycentric map
        's_det': s_det,   # deterministic map
        'J_LP': res.fun,
        'C': C,
    }


def compute_map_cost(x, s, f_x):
    """Compute J = sum c(x_i, s_i) * p_i."""
    p = f_x / f_x.sum()
    c = np.sqrt(2.0 * np.abs(x - s))
    return np.dot(c, p)


def s_analytical_4piece(x):
    """Current 4-piece analytical map (kinks at -9, 1)."""
    x = np.asarray(x, dtype=float)
    s = np.full_like(x, np.nan)
    m = (x >= -10) & (x < -9);  s[m] = -x[m]
    m = (x >= -9)  & (x < -1); s[m] = -x[m] - 10.0
    m = (x >= -1)   & (x < 1);   s[m] = -x[m]
    m = (x >= 1)   & (x <= 9);  s[m] = -x[m] + 10.0
    m = (x > 9)   & (x <= 10);  s[m] = -x[m]
    return s


def s_antimonotone(x):
    """Anti-monotone rearrangement: s(x) = -x."""
    return -np.asarray(x, dtype=float)


def s_monotone(x):
    """Monotone rearrangement: s(x) = x + 5."""
    return np.asarray(x, dtype=float) + 5.0


def main():
    print("=" * 70)
    print("  Computing TRUE optimal transport via Kantorovich LP")
    print("  c(x,y) = sqrt(2|x-y|)")
    print("=" * 70)

    # Solve LP at increasing resolutions
#    for n in [100, 200, 400]:
    for n in [100]:
        print(f"\n--- n = {n} ---")
        result = solve_kantorovich(n_disc=n)
        if result is None:
            continue

        x = result['x']
        f_x = rho_plus(x)

        # Compare candidate maps
        maps = {
            'LP (barycentric)': result['s_bary'],
            'LP (deterministic)': result['s_det'],
            '4-piece (kinks -9,1)': s_analytical_4piece(x),
            'Anti-monotone (-x)': s_antimonotone(x),
            'Monotone (x+5)': s_monotone(x),
        }

        print(f"\n  {'Map':>25} {'J':>10}")
        print(f"  {'-'*37}")
        for name, s in maps.items():
            valid = ~np.isnan(s)
            J = compute_map_cost(x[valid], s[valid], f_x[valid])
            print(f"  {name:>25} {J:10.6f}")
        print(f"  {'LP optimal':>25} {result['J_LP']:10.6f}")

    # Final detailed analysis with n=400
    print("\n\n--- Detailed analysis (n=400) ---")
    result = solve_kantorovich(n_disc=400)
    if result is None:
        return

    x = result['x']
    pi_mat = result['pi']

    # Check if the LP solution is a MAP (deterministic) or a genuine coupling
    n_src = len(x)
    n_active_per_row = np.zeros(n_src)
    for i in range(n_src):
        n_active_per_row[i] = np.sum(pi_mat[i, :] > 1e-10)

    print(f"\n  LP transport plan structure:")
    print(f"    Max active targets per source point: {n_active_per_row.max():.0f}")
    print(f"    Mean active targets per source: {n_active_per_row.mean():.1f}")
    print(f"    Fraction with >1 target (mass splitting): "
          f"{np.mean(n_active_per_row > 1.5):.1%}")

    # Plot results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Transport map comparison
    ax = axes[0]
    s_4p = s_analytical_4piece(x)
    valid = ~np.isnan(s_4p)

    ax.scatter(x, result['s_bary'], s=8, alpha=0.6, color='blue', label='LP barycentric', zorder=5)
    ax.scatter(x, result['s_det'], s=4, alpha=0.3, color='green', label='LP deterministic', zorder=4)
    ax.plot(x[valid], s_4p[valid], 'r--', lw=2, label='4-piece (kinks -9,1)', zorder=10)
    ax.plot(x, -x, ':', color='purple', lw=1.5, alpha=0.5, label='Anti-monotone (-x)')
    ax.plot(x, x + 5, ':', color='orange', lw=1.5, alpha=0.5, label='Monotone (x+5)')
    ax.set_xlabel('x'); ax.set_ylabel('s(x)')
    ax.set_title('Optimal Transport Maps')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 10.5)

    # Plot 2: LP transport plan (coupling structure)
    ax = axes[1]
    y = result['y']
    # Show coupling as scatter with size proportional to mass
    for i in range(0, n_src, 2):
        for j in range(len(y)):
            mass = pi_mat[i, j]
            if mass > 1e-10:
                ax.scatter(x[i], y[j], s=mass*50000, alpha=0.3, c='blue', edgecolors='none')
    ax.plot([-10, 10], [-10, 10], 'k:', alpha=0.2)
    ax.set_xlabel('x (source)'); ax.set_ylabel('y (target)')
    ax.set_title('LP Coupling Structure')
    ax.grid(True, alpha=0.3)

    # Plot 3: Number of active targets per source point
    ax = axes[2]
    ax.plot(x, n_active_per_row, '.', ms=3, alpha=0.5)
    ax.set_xlabel('x (source)'); ax.set_ylabel('# active targets')
    ax.set_title('Mass Splitting in LP Solution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'kantorovich_lp_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {save_path}")


if __name__ == '__main__':
    main()
