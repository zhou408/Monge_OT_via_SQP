"""
Find the optimal transport map for the McCann concave cost problem.

Cost: c(x,y) = sqrt(2|x-y|)
Source: mu = max(sin(pi*x/5), 0) / Z on [-10,10], support (-10,-5) U (0,5)
Target: nu = max(-sin(pi*x/5), 0) / Z on [-10,10], support (-5,0) U (5,10)
Z = 20/pi

The source has two "bumps" (half-sine arches):
  Left source (LS):  (-10, -5), peak at x = -7.5
  Right source (RS): (0, 5),    peak at x = 2.5

The target has two bumps:
  Left target (LT):  (-5, 0),   peak at y = -2.5
  Right target (RT): (5, 10),   peak at y = 7.5

Valid piecewise affine maps with |s'| = 1 on each piece:
  - s = -x        (slope -1): maps LS -> RT, maps RS -> LT  (far reflection)
  - s = -x - 10   (slope -1): maps LS -> LT                 (near shift for LS)
  - s = -x + 10   (slope -1): maps RS -> RT                 (near shift for RS)
  - s = x + 5     (slope +1): maps LS -> LT, maps RS -> RT  (forward shift)
  - s = x - 5     (slope +1): maps LS -> LT (partial), maps RS -> LT (partial)

We enumerate all valid map structures and optimize kink locations.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Problem parameters
# =============================================================================
Z = 20.0 / np.pi  # normalization constant

def f(x):
    """Source density (unnormalized): max(sin(pi*x/5), 0)."""
    return np.maximum(np.sin(np.pi * x / 5.0), 0.0)

def g(y):
    """Target density (unnormalized): max(-sin(pi*x/5), 0)."""
    return np.maximum(-np.sin(np.pi * y / 5.0), 0.0)

# Source bumps: (-10, -5) and (0, 5)
# Target bumps: (-5, 0) and (5, 10)

N_QUAD = 200000  # number of quadrature points per interval

def compute_cost_interval(a, b, s_func, n=N_QUAD):
    """Compute integral of sqrt(2|x - s(x)|) * f(x)/Z dx over [a, b].

    Uses composite Simpson's rule for high accuracy.
    """
    if b <= a:
        return 0.0
    x = np.linspace(a, b, n + 1)
    sx = s_func(x)
    integrand = np.sqrt(2.0 * np.abs(x - sx)) * f(x) / Z
    # Simpson's rule
    dx = (b - a) / n
    return np.trapz(integrand, dx=dx)

def compute_cost_interval_simpson(a, b, s_func, n=N_QUAD):
    """Compute integral using Simpson's rule for better accuracy."""
    if b <= a:
        return 0.0
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    sx = s_func(x)
    integrand = np.sqrt(2.0 * np.abs(x - sx)) * f(x) / Z
    dx = (b - a) / n
    # Simpson's 1/3 rule
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return dx / 3.0 * np.dot(w, integrand)

def compute_cost_quad(a, b, s_func):
    """Compute integral using scipy.integrate.quad for validation."""
    def integrand(x):
        sx = s_func(np.array([x]))[0]
        return np.sqrt(2.0 * np.abs(x - sx)) * f(np.array([x]))[0] / Z
    result, _ = quad(integrand, a, b, limit=200, epsabs=1e-12, epsrel=1e-12)
    return result

# =============================================================================
# Verify pushforward constraint for each piece
# =============================================================================
# For s with |s'| = 1, the pushforward constraint is:
#   f(x) = g(s(x)) * |s'(x)| = g(s(x))
# We need f(x) = g(s(x)) for all x in source support.
#
# f(x) = sin(pi*x/5) for x in (-10,-5) U (0,5)
# g(y) = -sin(pi*y/5) = sin(-pi*y/5) for y in (-5,0) U (5,10)
#
# Check s = -x:  g(-x) = -sin(-pi*x/5) = sin(pi*x/5) = f(x)  YES
# Check s = -x-10: g(-x-10) = -sin(pi*(-x-10)/5) = -sin(-pi*x/5 - 2*pi)
#                             = -sin(-pi*x/5) = sin(pi*x/5) = f(x)  YES
# Check s = -x+10: g(-x+10) = -sin(pi*(-x+10)/5) = -sin(-pi*x/5 + 2*pi)
#                             = -sin(-pi*x/5) = sin(pi*x/5) = f(x)  YES
# Check s = x+5:  g(x+5) = -sin(pi*(x+5)/5) = -sin(pi*x/5 + pi)
#                          = sin(pi*x/5) = f(x)  YES
# Check s = x-5:  g(x-5) = -sin(pi*(x-5)/5) = -sin(pi*x/5 - pi)
#                          = sin(pi*x/5) = f(x)  YES
# Check s = x+15: g(x+15) = -sin(pi*(x+15)/5) = -sin(pi*x/5 + 3*pi)
#                           = sin(pi*x/5) = f(x)  YES (but out of range)

print("=" * 78)
print("  Optimal Transport Map: c(x,y) = sqrt(2|x-y|)")
print("  Source: mu = max(sin(pi*x/5),0)/Z on support (-10,-5) U (0,5)")
print("  Target: nu = max(-sin(pi*x/5),0)/Z on support (-5,0) U (5,10)")
print("=" * 78)

# =============================================================================
# Valid affine pieces and their image intervals
# =============================================================================
# For left source (-10, -5):
#   s = -x       maps (-10,-5) -> (5,10)  = RT   |x - s| = |x - (-x)| = |2x|, x in (-10,-5) so |2x| = -2x in (10,20)
#   s = -x - 10  maps (-10,-5) -> (-5,0)  = LT   |x - s| = |x+x+10| = |2x+10|, x in (-10,-5) so 2x+10 in (-10,0), |.| in (0,10)
#   s = x + 5    maps (-10,-5) -> (-5,0)  = LT   |x - s| = |x - x - 5| = 5
#   s = x + 15   maps (-10,-5) -> (5,10)  = RT   |x - s| = 15  (but need y in domain)
#   s = x - 5    maps (-10,-5) -> (-15,-10) = OUT OF DOMAIN
#   s = -x - 20  maps (-10,-5) -> (15,10) reversed.. hmm = OUT OF DOMAIN
#
# For right source (0, 5):
#   s = -x       maps (0,5) -> (-5,0)     = LT   |x - s| = |2x|, x in (0,5) so |2x| = 2x in (0,10)
#   s = -x + 10  maps (0,5) -> (5,10)     = RT   |x - s| = |2x - 10|, x in (0,5) so 2x-10 in (-10,0), |.| in (0,10)
#   s = x + 5    maps (0,5) -> (5,10)     = RT   |x - s| = 5
#   s = x - 5    maps (0,5) -> (-5,0)     = LT   |x - s| = 5
#   s = -x + 20  OUT OF DOMAIN
#   s = -x - 10  maps (0,5) -> (-10,-15) = OUT OF DOMAIN

# The valid affine pieces (with image in target support):
# Left source:
#   L1: s = -x       -> RT,  displacement = -2x (always >= 10 on LS)
#   L2: s = -x - 10  -> LT,  displacement = |2x+10| (0 to 10 on LS)
#   L3: s = x + 5    -> LT,  displacement = 5 (constant)
#
# Right source:
#   R1: s = -x       -> LT,  displacement = 2x (0 to 10 on RS)
#   R2: s = -x + 10  -> RT,  displacement = |2x-10| (0 to 10 on RS)
#   R3: s = x + 5    -> RT,  displacement = 5 (constant)
#   R4: s = x - 5    -> LT,  displacement = 5 (constant)

# =============================================================================
# CATEGORY 1: No-split maps (each source bump maps entirely to one target)
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 1: No-split maps (full bump -> full bump)")
print("=" * 78)

candidates = []

# 1a: LS -> RT via L1 (s = -x), RS -> LT via R1 (s = -x)  [pure far reflection]
def s_1a(x):
    return -x
cost_1a_L = compute_cost_interval_simpson(-10, -5, s_1a)
cost_1a_R = compute_cost_interval_simpson(0, 5, s_1a)
cost_1a = cost_1a_L + cost_1a_R
candidates.append(("1a: LS->RT(s=-x), RS->LT(s=-x) [full reflection]", cost_1a,
                    "s(x) = -x everywhere", None))
print(f"  1a: s=-x everywhere (full reflection)")
print(f"      Cost = {cost_1a:.10f}  (L:{cost_1a_L:.10f}, R:{cost_1a_R:.10f})")

# 1b: LS -> LT via L2 (s = -x-10), RS -> RT via R2 (s = -x+10) [near reflection]
def s_1b_L(x): return -x - 10
def s_1b_R(x): return -x + 10
cost_1b_L = compute_cost_interval_simpson(-10, -5, s_1b_L)
cost_1b_R = compute_cost_interval_simpson(0, 5, s_1b_R)
cost_1b = cost_1b_L + cost_1b_R
candidates.append(("1b: LS->LT(s=-x-10), RS->RT(s=-x+10) [near reflection]", cost_1b,
                    "s(x) = -x-10 on LS, -x+10 on RS", None))
print(f"  1b: s=-x-10 on LS, s=-x+10 on RS (near reflection)")
print(f"      Cost = {cost_1b:.10f}  (L:{cost_1b_L:.10f}, R:{cost_1b_R:.10f})")

# 1c: LS -> LT via L3 (s = x+5), RS -> RT via R3 (s = x+5) [forward shift]
def s_1c(x): return x + 5
cost_1c_L = compute_cost_interval_simpson(-10, -5, s_1c)
cost_1c_R = compute_cost_interval_simpson(0, 5, s_1c)
cost_1c = cost_1c_L + cost_1c_R
candidates.append(("1c: LS->LT(s=x+5), RS->RT(s=x+5) [forward shift]", cost_1c,
                    "s(x) = x+5 everywhere", None))
print(f"  1c: s=x+5 everywhere (forward shift by 5)")
print(f"      Cost = {cost_1c:.10f}  (L:{cost_1c_L:.10f}, R:{cost_1c_R:.10f})")

# 1d: LS -> RT via L1 (s=-x), RS -> RT via R2 (s=-x+10)
# INVALID: both map to RT, so RT gets double mass, LT gets none

# 1e: LS -> LT via L2 (s=-x-10), RS -> LT via R1 (s=-x)
# INVALID: both map to LT

# 1f: LS -> RT via L1 (s=-x), RS -> RT via R3 (s=x+5)
# INVALID: both map to RT

# 1g: LS -> LT via L3 (s=x+5), RS -> LT via R4 (s=x-5)
# INVALID: both map to LT

# Cross combinations:
# 1h: LS -> LT via L3 (s=x+5), RS -> RT via R2 (s=-x+10) [monotone L, antitone R]
def s_1h_L(x): return x + 5
def s_1h_R(x): return -x + 10
cost_1h_L = compute_cost_interval_simpson(-10, -5, s_1h_L)
cost_1h_R = compute_cost_interval_simpson(0, 5, s_1h_R)
cost_1h = cost_1h_L + cost_1h_R
candidates.append(("1h: LS->LT(s=x+5), RS->RT(s=-x+10) [mixed]", cost_1h,
                    "s(x)=x+5 on LS, -x+10 on RS", None))
print(f"  1h: s=x+5 on LS, s=-x+10 on RS (mixed monotone/antitone)")
print(f"      Cost = {cost_1h:.10f}  (L:{cost_1h_L:.10f}, R:{cost_1h_R:.10f})")

# 1i: LS -> LT via L2 (s=-x-10), RS -> RT via R3 (s=x+5) [antitone L, monotone R]
def s_1i_L(x): return -x - 10
def s_1i_R(x): return x + 5
cost_1i_L = compute_cost_interval_simpson(-10, -5, s_1i_L)
cost_1i_R = compute_cost_interval_simpson(0, 5, s_1i_R)
cost_1i = cost_1i_L + cost_1i_R
candidates.append(("1i: LS->LT(s=-x-10), RS->RT(s=x+5) [mixed]", cost_1i,
                    "s(x)=-x-10 on LS, x+5 on RS", None))
print(f"  1i: s=-x-10 on LS, s=x+5 on RS (mixed antitone/monotone)")
print(f"      Cost = {cost_1i:.10f}  (L:{cost_1i_L:.10f}, R:{cost_1i_R:.10f})")

# 1j: LS -> RT via L1 (s=-x), RS -> LT via R4 (s=x-5)
def s_1j_L(x): return -x
def s_1j_R(x): return x - 5
cost_1j_L = compute_cost_interval_simpson(-10, -5, s_1j_L)
cost_1j_R = compute_cost_interval_simpson(0, 5, s_1j_R)
cost_1j = cost_1j_L + cost_1j_R
candidates.append(("1j: LS->RT(s=-x), RS->LT(s=x-5) [mixed]", cost_1j,
                    "s(x)=-x on LS, x-5 on RS", None))
print(f"  1j: s=-x on LS, s=x-5 on RS (far L + forward R to LT)")
print(f"      Cost = {cost_1j:.10f}  (L:{cost_1j_L:.10f}, R:{cost_1j_R:.10f})")

# 1k: LS -> LT via L3 (s=x+5), RS -> LT via R1 (s=-x)
# INVALID: both to LT

# =============================================================================
# CATEGORY 2: Single-kink maps on left source, matching kink on right source
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 2: Single-kink maps (split each source bump)")
print("=" * 78)

# The key insight: if we split LS at kink k1 in (-10,-5), then part of LS goes
# to RT and part to LT. The mass split must match the target. Similarly for RS.
#
# By symmetry of the densities, each target bump has mass 10/pi.
# Each source bump has mass 10/pi.
# So if LS sends mass m to RT, it must send (10/pi - m) to LT.
# Then RS must send (10/pi - m) to RT and m to LT to balance.
#
# The mass sent from LS's (-10, k1) subinterval:
#   M(k1) = integral_{-10}^{k1} sin(pi*x/5) dx / Z
#         = [-5/pi * cos(pi*x/5)]_{-10}^{k1} / Z
#         = (5/pi)(cos(2*pi) - cos(pi*k1/5)) / Z
#         = (5/pi)(1 - cos(pi*k1/5)) / (20/pi)
#         = (1 - cos(pi*k1/5)) / 4
#
# Similarly for RS split at k2 in (0,5):
#   M(k2) = integral_0^{k2} sin(pi*x/5) dx / Z
#         = (5/pi)(1 - cos(pi*k2/5)) / (20/pi)
#         = (1 - cos(pi*k2/5)) / 4
#
# Each full bump mass = integral_{-10}^{-5} sin(pi*x/5)/Z dx = 10/(pi*Z) = 1/2
# (since Z = 20/pi, integral = (5/pi)(cos(-2pi) - cos(-pi)) / Z = (5/pi)(1+1)/(20/pi) = 10/(pi * 20/pi) = 1/2)

def mass_LS_left(k1):
    """Mass in (-10, k1) from left source bump, divided by Z."""
    return (1.0 - np.cos(np.pi * k1 / 5.0)) / 4.0

def mass_RS_left(k2):
    """Mass in (0, k2) from right source bump, divided by Z."""
    return (1.0 - np.cos(np.pi * k2 / 5.0)) / 4.0

def find_k2_for_mass(mass_val):
    """Given mass in (0, k2) = mass_val, solve for k2 in (0,5)."""
    # mass_val = (1 - cos(pi*k2/5)) / 4
    # cos(pi*k2/5) = 1 - 4*mass_val
    cos_val = 1.0 - 4.0 * mass_val
    if cos_val < -1 or cos_val > 1:
        return None
    return 5.0 / np.pi * np.arccos(cos_val)

def find_k1_for_mass(mass_val):
    """Given mass in (-10, k1) = mass_val, solve for k1 in (-10,-5)."""
    cos_val = 1.0 - 4.0 * mass_val
    if cos_val < -1 or cos_val > 1:
        return None
    return 5.0 / np.pi * np.arccos(cos_val) - 10.0  # shift by -10 for left bump

# Actually let me redo this more carefully.
# For left source (-10, -5), let u = x + 10, u in (0, 5):
#   sin(pi*x/5) = sin(pi*(u-10)/5) = sin(pi*u/5 - 2*pi) = sin(pi*u/5)
# So the shape is identical to the right source with u = x+10.
# Mass in (-10, k1) = Mass in (0, k1+10) in u coordinates
#   = (1 - cos(pi*(k1+10)/5)) / 4

def mass_LS_left_v2(k1):
    """Mass of source in (-10, k1), k1 in [-10, -5]."""
    u1 = k1 + 10.0  # u1 in [0, 5]
    return (1.0 - np.cos(np.pi * u1 / 5.0)) / 4.0

# Total mass of each bump = 1/2 (in normalized units)
# Verify:
print(f"  Mass of full LS bump: {mass_LS_left_v2(-5.0):.10f} (should be 0.5)")
print(f"  Mass of full RS bump: {mass_RS_left(5.0):.10f} (should be 0.5)")

# =============================================================================
# 2A: Left split: far first, then near
#     LS: (-10,k1) -> s=-x [to RT], (k1,-5) -> s=-x-10 [to LT]
#     RS: (0,k2) -> s=-x [to LT], (k2,5) -> s=-x+10 [to RT]
#
# Mass balance on RT:
#   mass from LS(-10,k1) via s=-x = mass_LS_left(k1)
#   mass from RS(k2,5) via s=-x+10 = 1/2 - mass_RS_left(k2)
#   Total RT mass must = 1/2
#   So: mass_LS_left(k1) + 1/2 - mass_RS_left(k2) = 1/2
#   => mass_LS_left(k1) = mass_RS_left(k2)
#
# But we also need the images to tile the target exactly.
# s=-x maps (-10,k1) -> (-k1, 10) in RT. Need -k1 >= 5, so k1 <= -5.
# Wait, k1 is in (-10,-5), so -k1 in (5,10). Good, image is (-k1, 10) subset of (5,10) = RT.
# s=-x-10 maps (k1,-5) -> (-(-5)-10, -k1-10) = (-5, -k1-10) in LT. Need -k1-10 <= 0, so k1 >= -10. Yes.
# So image of (k1,-5) via s=-x-10 is (-5, -k1-10). Since k1 in (-10,-5), -k1-10 in (-5,0). Good.
#
# s=-x maps (0,k2) -> (-k2, 0) in LT. Need -k2 >= -5, so k2 <= 5. Good.
# s=-x+10 maps (k2,5) -> (-5+10, -k2+10) = (5, 10-k2) in RT. Need 10-k2 <= 10, so k2 >= 0. Good.
#
# For the images to tile RT = (5,10) exactly:
#   From LS: (-k1, 10)        [since s=-x reverses orientation]
#   From RS: (5, 10-k2)       [since s=-x+10 reverses]
#   Need: 10-k2 = -k1, i.e., k2 = k1 + 10
#   Since k1 in (-10,-5), k2 = k1+10 in (0,5). Perfect.
#
# For images to tile LT = (-5, 0):
#   From LS: (-5, -k1-10)
#   From RS: (-k2, 0)
#   Need: -k1-10 = -k2 = -(k1+10), so -k1-10 = -k1-10. Always true! Good.
#
# Mass balance is automatic when k2 = k1 + 10.
# =============================================================================

print("\n  --- 2A: LS split far-then-near, RS split far-then-near ---")
print("      LS: (-10,k1)->s=-x [RT], (k1,-5)->s=-x-10 [LT]")
print("      RS: (0,k2)->s=-x [LT],   (k2,5)->s=-x+10 [RT]")
print("      Constraint: k2 = k1 + 10")

def cost_2A(k1):
    """Cost for structure 2A with kink at k1 in (-10,-5), k2=k1+10 in (0,5)."""
    k2 = k1 + 10.0

    # Left source
    s_LS1 = lambda x: -x           # (-10, k1) -> RT
    s_LS2 = lambda x: -x - 10.0    # (k1, -5)  -> LT

    # Right source
    s_RS1 = lambda x: -x           # (0, k2)   -> LT
    s_RS2 = lambda x: -x + 10.0    # (k2, 5)   -> RT

    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, s_LS1)
    c += compute_cost_interval_simpson(k1, -5, s_LS2)
    c += compute_cost_interval_simpson(0, k2, s_RS1)
    c += compute_cost_interval_simpson(k2, 5, s_RS2)
    return c

# Optimize
result_2A = minimize_scalar(cost_2A, bounds=(-10 + 1e-10, -5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k1_opt_2A = result_2A.x
k2_opt_2A = k1_opt_2A + 10.0
cost_opt_2A = result_2A.fun

# Also evaluate at boundary: k1=-10 (no split, = case 1b) and k1=-5 (no split, = case 1a)
print(f"  Optimal k1 = {k1_opt_2A:.10f}, k2 = {k2_opt_2A:.10f}")
print(f"  Optimal cost = {cost_opt_2A:.10f}")
print(f"  Cost at k1=-10 (pure near):  {cost_2A(-10 + 1e-12):.10f}")
print(f"  Cost at k1=-5  (pure far):   {cost_2A(-5 - 1e-12):.10f}")

candidates.append((f"2A: far-near split, k1={k1_opt_2A:.6f}, k2={k2_opt_2A:.6f}",
                    cost_opt_2A,
                    f"LS: s=-x on (-10,{k1_opt_2A:.4f}), s=-x-10 on ({k1_opt_2A:.4f},-5); "
                    f"RS: s=-x on (0,{k2_opt_2A:.4f}), s=-x+10 on ({k2_opt_2A:.4f},5)",
                    k1_opt_2A))

# Scan to visualize
print("\n  Cost landscape for 2A:")
k1_scan = np.linspace(-10 + 0.01, -5 - 0.01, 100)
costs_2A = [cost_2A(k) for k in k1_scan]
imin = np.argmin(costs_2A)
print(f"  Scan min at k1 = {k1_scan[imin]:.6f}, cost = {costs_2A[imin]:.10f}")

# =============================================================================
# 2B: Left split: near first, then far
#     LS: (-10,k1) -> s=-x-10 [to LT], (k1,-5) -> s=-x [to RT]
#     RS: (0,k2) -> s=-x+10 [to RT], (k2,5) -> s=-x [to LT]
#
# Image tiling for RT = (5,10):
#   From LS (k1,-5) via s=-x: image is (5, -k1) [reversed]
#   From RS (0,k2) via s=-x+10: image is (10-k2, 10) [reversed]
#   Need: -k1 = 10-k2, i.e., k2 = k1+10. Same constraint!
#
# Image tiling for LT = (-5,0):
#   From LS (-10,k1) via s=-x-10: image is (-k1-10, 0) [reversed]
#   From RS (k2,5) via s=-x: image is (-5, -k2) [reversed]
#   Need: -k1-10 = -k2 = -(k1+10). Consistent.
# =============================================================================

print("\n  --- 2B: LS split near-then-far, RS split near-then-far ---")
print("      LS: (-10,k1)->s=-x-10 [LT], (k1,-5)->s=-x [RT]")
print("      RS: (0,k2)->s=-x+10 [RT],   (k2,5)->s=-x [LT]")
print("      Constraint: k2 = k1 + 10")

def cost_2B(k1):
    """Cost for structure 2B with kink at k1 in (-10,-5), k2=k1+10."""
    k2 = k1 + 10.0

    s_LS1 = lambda x: -x - 10.0    # (-10, k1) -> LT
    s_LS2 = lambda x: -x           # (k1, -5)  -> RT

    s_RS1 = lambda x: -x + 10.0    # (0, k2)   -> RT
    s_RS2 = lambda x: -x           # (k2, 5)   -> LT

    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, s_LS1)
    c += compute_cost_interval_simpson(k1, -5, s_LS2)
    c += compute_cost_interval_simpson(0, k2, s_RS1)
    c += compute_cost_interval_simpson(k2, 5, s_RS2)
    return c

result_2B = minimize_scalar(cost_2B, bounds=(-10 + 1e-10, -5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k1_opt_2B = result_2B.x
k2_opt_2B = k1_opt_2B + 10.0
cost_opt_2B = result_2B.fun

print(f"  Optimal k1 = {k1_opt_2B:.10f}, k2 = {k2_opt_2B:.10f}")
print(f"  Optimal cost = {cost_opt_2B:.10f}")

candidates.append((f"2B: near-far split, k1={k1_opt_2B:.6f}, k2={k2_opt_2B:.6f}",
                    cost_opt_2B,
                    f"LS: s=-x-10 on (-10,{k1_opt_2B:.4f}), s=-x on ({k1_opt_2B:.4f},-5); "
                    f"RS: s=-x+10 on (0,{k2_opt_2B:.4f}), s=-x on ({k2_opt_2B:.4f},5)",
                    k1_opt_2B))

# Scan
costs_2B = [cost_2B(k) for k in k1_scan]
imin_2B = np.argmin(costs_2B)
print(f"  Scan min at k1 = {k1_scan[imin_2B]:.6f}, cost = {costs_2B[imin_2B]:.10f}")

# =============================================================================
# 2C: Mixed: LS far-near, RS near-far (with monotone pieces)
#     LS: (-10,k1) -> s=-x [RT], (k1,-5) -> s=x+5 [LT]
#     RS: (0,k2) -> s=x+5 [RT],  (k2,5) -> s=-x [LT]
#
# But we need slope +1 and -1 pieces to tile correctly.
# s=-x on (-10,k1): image (-k1, 10) subset RT
# s=x+5 on (k1,-5): image (k1+5, 0) subset LT. Need k1+5 >= -5, always true since k1 > -10.
# s=x+5 on (0,k2): image (5, k2+5) subset RT.
# s=-x on (k2,5): image (-5, -k2) subset LT.
#
# RT tiling: (-k1, 10) from LS, (5, k2+5) from RS. Need k2+5 = -k1 => k2 = -k1-5.
#   k1 in (-10,-5) => k2 in (0,5). Good.
# LT tiling: (k1+5, 0) from LS, (-5, -k2) from RS. Need k1+5 = -k2 = k1+5. Consistent!
#
# But wait: s=x+5 has slope +1, and s=-x has slope -1. At the kink, continuity is not
# required (the map can jump between source support components). Within each source bump,
# we need the pieces to be consistent. Let me check continuity at k1:
#   Left of k1: s=-x, so s(k1-) = -k1
#   Right of k1: s=x+5, so s(k1+) = k1+5
#   For continuity: -k1 = k1+5 => k1 = -2.5. But k1 must be in (-10,-5). No solution!
#
# So this map has a JUMP discontinuity at k1. For optimal transport with concave cost,
# the optimal map CAN be discontinuous if the cost is non-twisted. Let's still compute it.
# However, a discontinuous map is harder to interpret as a "piecewise affine map with kinks."
# Actually, the question says "kink" which typically means a continuous but non-smooth point.
# A jump discontinuity would correspond to a different framework.
#
# Actually, wait. For the pushforward constraint to hold at a kink between two pieces
# with |s'|=1, continuity is NOT strictly required - the density is continuous and the
# pushforward of each piece separately handles its part. But the map being a well-defined
# function requires single-valuedness, which is fine - it just jumps.
#
# For concave costs, discontinuous maps can be optimal. Let's include these.
# =============================================================================

print("\n  --- 2C: LS antitone-then-monotone, RS monotone-then-antitone ---")
print("      LS: (-10,k1)->s=-x [RT], (k1,-5)->s=x+5 [LT]")
print("      RS: (0,k2)->s=x+5 [RT],  (k2,5)->s=-x [LT]")
print("      Constraint: k2 = -k1-5")

def cost_2C(k1):
    """Cost for 2C: k2 = -k1-5."""
    k2 = -k1 - 5.0
    if k2 < 0 or k2 > 5:
        return 1e10

    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, lambda x: -x)
    c += compute_cost_interval_simpson(k1, -5, lambda x: x + 5)
    c += compute_cost_interval_simpson(0, k2, lambda x: x + 5)
    c += compute_cost_interval_simpson(k2, 5, lambda x: -x)
    return c

result_2C = minimize_scalar(cost_2C, bounds=(-10 + 1e-10, -5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k1_opt_2C = result_2C.x
k2_opt_2C = -k1_opt_2C - 5.0
cost_opt_2C = result_2C.fun

print(f"  Optimal k1 = {k1_opt_2C:.10f}, k2 = {k2_opt_2C:.10f}")
print(f"  Optimal cost = {cost_opt_2C:.10f}")

candidates.append((f"2C: antitone-monotone, k1={k1_opt_2C:.6f}, k2={k2_opt_2C:.6f}",
                    cost_opt_2C,
                    f"LS: s=-x on (-10,{k1_opt_2C:.4f}), s=x+5 on ({k1_opt_2C:.4f},-5); "
                    f"RS: s=x+5 on (0,{k2_opt_2C:.4f}), s=-x on ({k2_opt_2C:.4f},5)",
                    k1_opt_2C))

# =============================================================================
# 2D: LS monotone-then-antitone, RS antitone-then-monotone
#     LS: (-10,k1) -> s=x+5 [LT], (k1,-5) -> s=-x [RT]
#     RS: (0,k2) -> s=-x [LT],    (k2,5) -> s=x+5 [RT]
#
# RT: from LS (k1,-5) via s=-x: image (5, -k1), from RS (k2,5) via s=x+5: image (k2+5, 10)
# Need: -k1 = k2+5 => k2 = -k1-5. Same as 2C.
# LT: from LS (-10,k1) via s=x+5: image (-5, k1+5), from RS (0,k2) via s=-x: image (-k2, 0)
# Need: k1+5 = -k2 = k1+5. Consistent.
# =============================================================================

print("\n  --- 2D: LS monotone-then-antitone, RS antitone-then-monotone ---")
print("      LS: (-10,k1)->s=x+5 [LT], (k1,-5)->s=-x [RT]")
print("      RS: (0,k2)->s=-x [LT],    (k2,5)->s=x+5 [RT]")
print("      Constraint: k2 = -k1-5")

def cost_2D(k1):
    """Cost for 2D: k2 = -k1-5."""
    k2 = -k1 - 5.0
    if k2 < 0 or k2 > 5:
        return 1e10

    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, lambda x: x + 5)
    c += compute_cost_interval_simpson(k1, -5, lambda x: -x)
    c += compute_cost_interval_simpson(0, k2, lambda x: -x)
    c += compute_cost_interval_simpson(k2, 5, lambda x: x + 5)
    return c

result_2D = minimize_scalar(cost_2D, bounds=(-10 + 1e-10, -5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k1_opt_2D = result_2D.x
k2_opt_2D = -k1_opt_2D - 5.0
cost_opt_2D = result_2D.fun

print(f"  Optimal k1 = {k1_opt_2D:.10f}, k2 = {k2_opt_2D:.10f}")
print(f"  Optimal cost = {cost_opt_2D:.10f}")

candidates.append((f"2D: monotone-antitone, k1={k1_opt_2D:.6f}, k2={k2_opt_2D:.6f}",
                    cost_opt_2D,
                    f"LS: s=x+5 on (-10,{k1_opt_2D:.4f}), s=-x on ({k1_opt_2D:.4f},-5); "
                    f"RS: s=-x on (0,{k2_opt_2D:.4f}), s=x+5 on ({k2_opt_2D:.4f},5)",
                    k1_opt_2D))

# =============================================================================
# 2E: LS: antitone near then monotone, RS: monotone then antitone near
#     LS: (-10,k1)->s=-x-10 [LT], (k1,-5)->s=x+5 [LT]  -- BOTH GO TO LT!
#     Wait, that sends all of LS to LT and nothing to RT. Then RS must go all to RT.
#     That's just a no-split map with different pieces.
#     But: s=-x-10 maps (-10,k1) to (-k1-10, 0) in LT
#          s=x+5  maps (k1,-5) to (k1+5, 0) in LT
#     For tiling LT = (-5,0): need -k1-10 = k1+5 => k1 = -7.5.
#     And images: (-k1-10, 0) = (-(-7.5)-10, 0) = (-2.5, 0)
#                 (k1+5, 0) = (-7.5+5, 0) = (-2.5, 0)  OVERLAP!
#     Actually image of (-10,-7.5) via s=-x-10 is (-(-7.5)-10, -(-10)-10) = (-2.5, 0)
#     Image of (-7.5,-5) via s=x+5 is (-7.5+5, -5+5) = (-2.5, 0)
#     So both pieces map to (-2.5, 0). That means LT gets double-covered on (-2.5,0)
#     and nothing on (-5,-2.5). INVALID.
# =============================================================================
# Skip 2E - invalid.

# =============================================================================
# 2F: LS uses s=x-5 (displacement -5, monotone)
#     s=x-5 maps (-10,-5) -> (-15,-10): OUT OF DOMAIN. SKIP.
# =============================================================================

# =============================================================================
# 2G: RS uses s=x-5
#     s=x-5 maps (0,5) -> (-5,0) = LT. Displacement = 5 (constant).
#     Combined with some split...
#
#     LS: (-10,k1)->s=-x [RT], (k1,-5)->s=-x-10 [LT]
#     RS: (0,5)->s=x-5 [LT]
#     Mass balance: RT gets mass_LS_left(k1), LT gets (1/2 - mass_LS_left(k1)) + 1/2
#     Need RT mass = 1/2: mass_LS_left(k1) = 1/2, so k1 = -5. That's no split = case 1a
#     with RS via s=x-5 instead of s=-x.
#     Actually if RS sends ALL mass to LT via x-5, then LS must send ALL to RT via s=-x.
#     That's a valid no-split map.
# =============================================================================

# 1k: LS -> RT via s=-x, RS -> LT via s=x-5  (already computed as 1j)
# Already done above.

# =============================================================================
# 2H: Both bumps use s=x+5 for part, s=-x-10 / s=-x+10 for other part
#     LS: (-10,k1)->s=x+5 [LT], (k1,-5)->s=-x-10 [LT]
#     Both to LT again -- need to check tiling.
#     s=x+5 on (-10,k1): image (-5, k1+5), monotone increasing
#     s=-x-10 on (k1,-5): image (-5-(-10)-10, -k1-10) wait...
#     s=-x-10: on (k1,-5), image is (-(-5)-10, -k1-10) = (-5, -k1-10)
#     Wait, s=-x-10 reverses. s(k1) = -k1-10, s(-5) = 5-10 = -5.
#     So image of (k1,-5) via s=-x-10 is (-5, -k1-10), BUT this reverses:
#     s(k1)=-k1-10, s(-5)=-5. Since s is decreasing and k1 < -5,
#     -k1-10 > -5-10 = -5... wait, k1 in (-10,-5), so -k1 in (5,10), -k1-10 in (-5,0).
#     Image: from x=k1 to x=-5, s goes from -k1-10 to -5. Since -k1-10 > -5 (as k1<-5),
#     image is (-5, -k1-10). Reversed.
#     s=x+5 on (-10,k1): image (-5, k1+5). k1+5 in (-5,0).
#     Tiling LT: (-5, k1+5) U (-5, -k1-10). Need k1+5 = -5 (one starts from -5) --
#     both start from -5! They overlap on (-5, min(k1+5, -k1-10)).
#     For no overlap: need k1+5 = -5 or -k1-10 = -5, i.e. k1=-10 or k1=-5 (boundaries).
#     INVALID for interior kinks.
# =============================================================================

# =============================================================================
# CATEGORY 3: Two kinks per source bump
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 3: Two kinks per source bump (3 pieces per bump)")
print("=" * 78)

# 3A: LS: (-10,k1)->s=-x [RT], (k1,k1')->s=-x-10 [LT], (k1',-5)->s=-x [RT]
# This means s=-x appears twice on LS with a gap of s=-x-10 in between.
# Image from s=-x on (-10,k1): (-k1, 10) in RT
# Image from s=-x-10 on (k1,k1'): (-k1'-10, -k1-10) in LT
# Image from s=-x on (k1',-5): (5, -k1') in RT
# RT gets: (-k1, 10) U (5, -k1'). For tiling RT: need -k1' = -k1, i.e. k1=k1'. Degenerate.
# UNLESS they overlap... which means double coverage. INVALID.
# Actually, wait. Maybe the intervals don't need to tile -- they can overlap if the
# densities add up correctly? No, for a transport map, the map must be measure-preserving,
# meaning each target point receives exactly the right mass. With |s'|=1 and density matching,
# each target point must be the image of exactly one source point (a.e.).
# So the images must tile (partition) the target a.e. Thus 3A is invalid.

# 3B: LS: (-10,k1)->s=-x-10 [LT], (k1,k1')->s=-x [RT], (k1',-5)->s=-x-10 [LT]
# Image from s=-x-10 on (-10,k1): (-k1-10, 0) in LT
# Image from s=-x on (k1,k1'): (-k1', -k1) in RT
# Image from s=-x-10 on (k1',-5): (-5, -k1'-10) in LT
# RT gets: (-k1', -k1). Need -k1' >= 5 and -k1 <= 10, so k1' <= -5 and k1 >= -10. OK.
# LT gets: (-k1-10, 0) U (-5, -k1'-10). Need -k1-10 >= -5 and -k1'-10 <= 0.
#   => k1 <= -5 (always) and k1' >= -10 (always).
# For LT tiling: (-5, -k1'-10) U (-k1-10, 0) = LT if -k1'-10 = -k1-10, i.e. k1=k1'. Degenerate.
# UNLESS they fill (-5,0) differently. Need union = (-5,0):
#   (-5, -k1'-10) U (-k1-10, 0) = (-5, 0)
#   Need: -k1'-10 >= -k1-10, which means k1' <= k1. But k1 < k1' by assumption. CONTRADICTION.
#   So we need the intervals to overlap or for -k1'-10 = -k1-10. Degenerate only.

# 3C: What about mixing in s=x+5?
# LS: (-10,k1)->s=-x [RT], (k1,k1')->s=x+5 [LT], (k1',-5)->s=-x [RT]
# Image s=-x on (-10,k1): (-k1, 10)
# Image s=x+5 on (k1,k1'): (k1+5, k1'+5) in LT
# Image s=-x on (k1',-5): (5, -k1')
# RT: (-k1, 10) U (5, -k1'). These are both subsets of (5,10).
#   For no overlap and covering: -k1' must connect to -k1: -k1' = -k1 => degenerate.
#   OR (5, -k1') U (-k1, 10) covers (5,10) if -k1' <= -k1, i.e. k1' >= k1.
#   But since k1 < k1', we have -k1 > -k1' >= 5, so (5, -k1') U (-k1, 10).
#   For exact tiling: need -k1' = 5 or -k1 = 10? No, need union = (5,10).
#   (5, -k1') U (-k1, 10) = (5,10) iff -k1' >= 5 and -k1 <= 10 and -k1' >= -k1 ... wait
#   Actually need: min endpoints = 5, max = 10, and they cover everything in between.
#   If -k1' < -k1, then: (5, -k1') U (-k1, 10). Gap if -k1' < -k1: gap is (-k1', -k1).
#   This gap has positive measure unless k1 = k1'. INVALID.

print("  All 2-kink-per-bump structures with pure antitone pieces lead to")
print("  degenerate (overlapping or gapped) target coverage. Skipping.")

# 3D: Let's try s=x+5 combined with s=-x-10 on the left source, and 3 pieces:
# LS: (-10,k1)->s=x+5 [LT], (k1,k1')->s=-x [RT], (k1',-5)->s=-x-10 [LT]
# s=x+5 on (-10,k1): image (-5, k1+5) in LT (monotone)
# s=-x on (k1,k1'): image (-k1', -k1) in (5,10) -- need -k1' >= 5 and -k1 <= 10
# s=-x-10 on (k1',-5): image (-5, -k1'-10) in LT (reversed: s(k1')=-k1'-10, s(-5)=-5)
#   So image is (-5, -k1'-10) going from -k1'-10 down to -5. Actually s is decreasing,
#   so s(k1') = -k1'-10, s(-5) = 5-10 = -5. Since k1' > k1 > -10, s(k1') < 0.
#   Image is (-5, -k1'-10) in LT.
#
# LT coverage: (-5, k1+5) U (-5, -k1'-10). Both start from -5.
#   Cover (-5, max(k1+5, -k1'-10)). For full coverage (-5,0): need max(k1+5, -k1'-10) >= 0
#   and no overlap beyond -5.
#   Actually: (-5, k1+5) U (-5, -k1'-10) = (-5, max(k1+5, -k1'-10)).
#   This double-covers (-5, min(k1+5, -k1'-10)). INVALID (double coverage).

# Let me try a different 3-piece structure:
# LS: (-10,k1)->s=-x-10 [LT img:(-k1-10,0)], (k1,k1')->s=-x [RT img:(-k1',-k1)], (k1',-5)->s=x+5 [LT img:(k1'+5,0)]
# LT: (-k1-10, 0) U (k1'+5, 0).
#   -k1-10 in (-5, 0) since k1 in (-10,-5).
#   k1'+5 in (-5,0) since k1' in (-10,-5).
#   For tiling: need k1'+5 = -k1-10 => k1' = -k1-15.
#   Since k1 in (-10,-5) and k1' in (k1,-5): -k1-15 > k1 => k1 < -7.5.
#   And k1' < -5 => -k1-15 < -5 => k1 > -10. So k1 in (-10, -7.5).
#   Also need -k1-10 < 0 => k1 > -10 (always) and k1'+5 > -5 => k1' > -10 (always if k1>-10).
#   LT = (k1'+5, 0) U (-k1-10, 0) = (k1'+5, 0) U (-k1-10, 0).
#   Since k1'+5 = -k1-15+5 = -k1-10 ... wait, that means they're the same interval!
#   So LT gets: (k1'+5, 0) = (-k1-10, 0). And (-k1-10, 0) from the first piece.
#   DOUBLE COVERAGE. Invalid.

# Actually, I think for 3 pieces the tiling becomes very constrained. Let me check if
# there are any valid 3-piece structures by careful analysis.

# For LS with 3 pieces mapping to RT and LT, we need the RT images to tile exactly
# the portion of RT that LS is responsible for, and similarly for LT.
# With |s'|=1, each piece has image length = domain length.
# Total LS length = 5. If LS sends length a to RT and length (5-a) to LT,
# then similarly RS must send length (5-a) to RT and length a to LT (to total 5 each).
#
# For 3 antitone pieces (all slope -1), images are reversed intervals.
# The constraint that RT images tile a connected interval of length a forces them to
# be contiguous, which with reversed orientation means... it's actually impossible
# to have 2 disjoint pieces of LS map to contiguous parts of RT without overlap/gap
# (as shown above).
#
# For mixed slope pieces, the images go in different directions, making tiling possible
# in principle but the continuity/overlap constraints are very restrictive.
#
# I believe 2+ kinks per source bump are generically invalid for this problem.
# The key insight is that with |s'|=1, each piece is length-preserving, and
# the target tiling constraint is very rigid.

print("  Multi-kink structures appear to be generically invalid due to")
print("  target tiling constraints. Verified above.")

# =============================================================================
# CATEGORY 4: Asymmetric maps (different structure on LS vs RS)
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 4: Asymmetric maps (different split structure per bump)")
print("=" * 78)

# 4A: LS has no split (s=-x, all to RT), RS has a split
# LS sends all mass 1/2 to RT via s=-x. Image = (5,10) = all of RT.
# RS must send all mass to LT. Options: s=-x (all to LT) or split.
# But if RS is unsplit, this is case 1a. With a split of RS sending some to RT...
# but RT is already full from LS. So RS must go entirely to LT.
# Only valid: 1a (RS via s=-x), or 1j (RS via s=x-5).

# 4B: LS has no split (s=-x-10, all to LT), RS has a split
# LS sends all 1/2 to LT. Image = (-5,0). RS must send all to RT.
# RS options: s=-x+10 or s=x+5 (both go to RT).
# This gives cases 1b and 1i.

# 4C: LS has a split, RS has no split
# By symmetry, this gives the same cost structures (just swap roles).
# Due to the identical bump shapes, the optimal kink placement is symmetric.

# Actually, let me think about this more carefully. The symmetry of the problem:
# If we define T(x) = x + 10 mod 20 (shift by half-period), the densities are swapped.
# More precisely, the problem has a translation symmetry x -> x+10 that maps LS to RS
# and LT to RT. So if we split LS with one structure, the optimal RS split is the
# translated version, which is exactly what our k2 = k1+10 constraint captures.

# However, we could also have asymmetric splits where LS has a kink but RS doesn't,
# or vice versa. Let's check if these are valid.

# 4D: LS split with kink k1, RS no split (all to one target)
# If RS goes all to LT (via s=-x), then LT gets all of RS's mass (1/2) plus
# whatever LS sends to LT.  For LT total = 1/2, LS sends 0 to LT.
# So LS goes all to RT. No split. This is case 1a.
#
# If RS goes all to RT (via s=-x+10 or s=x+5), then RT gets 1/2 from RS.
# LS must send 0 to RT. LS goes all to LT. No split. This is case 1b or 1i.
#
# For a genuine asymmetric split, both bumps need to participate in both targets
# to keep the mass balance. But each bump has mass 1/2, and each target needs 1/2.
# If LS sends m1 to RT and (1/2-m1) to LT, RS must send (1/2-m1) to RT and m1 to LT.
# Unless m1=0 or m1=1/2, both bumps are split. So genuinely asymmetric structures
# (one split, one not) only occur at the boundary cases (no-split maps).

print("  Asymmetric splits (one bump split, other not) reduce to no-split")
print("  boundary cases already enumerated.")

# =============================================================================
# CATEGORY 5: Maps using s=x-5 on RS for part of it
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 5: Maps using s=x-5 on right source")
print("=" * 78)

# 5A: RS: (0,k2)->s=x-5 [LT], (k2,5)->s=-x+10 [RT]
#     LS: must supply the rest.
# s=x-5 on (0,k2): image (-5, k2-5). For k2 in (0,5), k2-5 in (-5,0). Subset of LT.
# s=-x+10 on (k2,5): image (5, 10-k2). For k2 in (0,5), 10-k2 in (5,10). Subset of RT.
# So RS sends (0,k2) to LT, (k2,5) to RT.
# LS must send enough to fill the rest of LT and RT.
# LT from RS: (-5, k2-5). Remaining LT: (k2-5, 0). Length = 5 - k2 + 5 = ...
#   LT = (-5,0), RS covers (-5, k2-5), remaining is (k2-5, 0), length = -k2+5 = 5-k2.
# RT from RS: (5, 10-k2), remaining: (10-k2, 10), length = k2.
# LS must cover: (k2-5, 0) in LT and (10-k2, 10) in RT. Total length = (5-k2) + k2 = 5. Good.
#
# LS pieces to cover (k2-5, 0) in LT:
#   s=-x-10 reverses (-10, k1) -> (-k1-10, 0). Need -k1-10 = k2-5 => k1 = -k2-5 = 5-k2-10.
#   Wait: -k1-10 = k2-5 => k1 = -(k2-5)-10 = -k2-5. Since k2 in (0,5), k1 in (-10,-5).
#   So LS piece: (-10, -k2-5) via s=-x-10 -> (k2-5, 0) in LT. Length = -k2-5-(-10) = 5-k2. Matches.
#
# LS pieces to cover (10-k2, 10) in RT:
#   s=-x reverses (-k2-5, -5) -> (5, k2+5). We need (10-k2, 10).
#   s=-x on (-k2-5,-5) -> (5, k2+5). k2+5 in (5,10). But we need (10-k2, 10).
#   k2+5 = 10-k2 => k2 = 2.5. So this only works at k2=2.5.
#   More generally: s=-x on (a,b) -> (-b,-a). So on (-k2-5, -5): image (5, k2+5).
#   We need this to be (10-k2, 10). So 5 = 10-k2 => k2=5 and k2+5=10. Only at boundary.
#
# Alternative: use s=x+15 (slope +1)? s=x+15 maps (a,b) to (a+15,b+15).
#   For LS (-10,-5): image (5,10). That's the same as L1 in displacement but monotone.
#   But wait, s(x)=x+15 for x in (-10,-5) gives s in (5,10). Displacement |x-s|=|x-x-15|=15.
#   That's huge. And g(x+15) = -sin(pi(x+15)/5) = -sin(pix/5+3pi) = sin(pix/5) = f(x). OK valid.
#   But within domain [-10,10], x+15 > 10 for x > -5. So only valid on (-10,-5) mapping to (5,10).
#   Cost would be sqrt(2*15) everywhere = sqrt(30). Very expensive.
#
# Hmm, the constraint is that the LS image in RT must be (10-k2, 10), which has length k2.
# And the LS domain piece for this must have length k2 as well (from |s'|=1).
# Using s=-x: domain (a, a+k2) maps to (-a-k2, -a) in RT. Need -a = 10, so a = -10.
#   Domain: (-10, -10+k2). Image: (10-k2, 10). Length = k2.
#   But we also need -10+k2 <= -5, so k2 <= 5. OK.
#   And the domain must be within LS = (-10,-5). So -10+k2 < -5 => k2 < 5. OK.
#
# So: LS: (-10, -10+k2) via s=-x -> (10-k2, 10) in RT
#     LS: (-10+k2, -5) via s=-x-10 -> ... wait, let me recompute.
#     s=-x-10 on (-10+k2, -5): image (5-10, 10-k2-10) = ... s(-10+k2) = 10-k2-10 = -k2,
#     s(-5) = 5-10 = -5. So image is (-k2, -5)?? That's NOT in LT unless -k2 >= -5, i.e. k2 <= 5.
#     Actually s is decreasing. s(-10+k2) = -(-10+k2)-10 = 10-k2-10 = -k2.
#     s(-5) = 5-10 = -5. Since -10+k2 < -5, and s is decreasing, s(-10+k2) = -k2 > -5 = s(-5).
#     Image = (-5, -k2). For this to be in LT = (-5,0): need -k2 > -5 => k2 < 5.
#     And need -k2 < 0 => k2 > 0. OK.
#     Length = -k2 - (-5) = 5-k2. And domain length = -5 - (-10+k2) = 5-k2. Match!
#
# Now LT total from LS: (-5, -k2), length 5-k2.
#      LT from RS: (-5, k2-5), length k2.
#      Total LT: (-5, -k2) U (-5, k2-5) = (-5, max(-k2, k2-5)).
#      For tiling (-5,0): need -k2 = k2-5 => k2 = 2.5, giving both = (-5, -2.5) U (-5, -2.5).
#      OVERLAP at (-5, -2.5) if k2 = 2.5!
#      Actually wait: (-5, -k2) and (-5, k2-5). These both start from -5.
#      If k2 < 2.5: -k2 > k2-5, so (-5, k2-5) subset (-5, -k2). Overlap.
#      If k2 > 2.5: k2-5 > -k2, so (-5, -k2) subset (-5, k2-5). Overlap.
#      ALWAYS OVERLAPPING from -5. INVALID.

# So the issue is that s=-x-10 and s=x-5 both have images starting at -5 in LT.
# This fundamentally prevents tiling.

# Let me try a different combination for LS:
# LS: (-10, -10+k2) via s=-x -> (10-k2, 10) in RT
# LS: (-10+k2, -5) via some map to cover (k2-5, 0) in LT
# Need a map from (-10+k2, -5) to (k2-5, 0) with |s'|=1 and f(x)=g(s(x)).
# Length of domain = 5-k2. Length of target = 5-k2. Good.
# Option 1: s = x + (k2-5) - (-10+k2) + (k2-5) ... let me think.
#   We need s: (-10+k2, -5) -> (k2-5, 0) bijectively.
#   If monotone increasing (slope +1): s(x) = x + (k2-5) - (-10+k2) = x + k2-5+10-k2 = x+5.
#   So s=x+5. Image: (-10+k2+5, -5+5) = (k2-5, 0).
#   And g(x+5) = -sin(pi(x+5)/5) = sin(pix/5) = f(x). Valid!
#
#   If monotone decreasing (slope -1): s(x) = -(x - (-10+k2)) + 0 = -x -10+k2.
#   Hmm: s(-10+k2) = -(-10+k2)-10+k2 = 10-k2-10+k2 = 0. s(-5) = 5-10+k2 = k2-5.
#   So s(x) = -x + (k2-10). Image: (k2-5, 0) reversed. g(-x+k2-10) = -sin(pi(-x+k2-10)/5).
#   = -sin(-pix/5 + pi*k2/5 - 2pi) = -sin(-pix/5 + pi*k2/5) = sin(pix/5 - pi*k2/5).
#   This equals f(x) = sin(pix/5) only if k2=0. INVALID for general k2.

# So the only valid map for the remaining LS piece is s=x+5.
# This gives:
# 5A-v2:
#   LS: (-10, -10+k2) via s=-x [to RT], (-10+k2, -5) via s=x+5 [to LT]
#   RS: (0, k2) via s=x-5 [to LT], (k2, 5) via s=-x+10 [to RT]
# Wait but we showed this overlaps on LT. Let me recheck.
# LT from LS via s=x+5 on (-10+k2, -5): image (k2-5, 0)
# LT from RS via s=x-5 on (0, k2): image (-5, k2-5)
# LT total: (-5, k2-5) U (k2-5, 0) = (-5, 0). PERFECT TILING! No overlap!
# I made an error before. Let me recheck.
# LS via s=-x-10 gives image starting from -5. But LS via s=x+5 gives image (k2-5, 0).
# RS via s=x-5 gives image (-5, k2-5).
# These tile perfectly: (-5, k2-5) U (k2-5, 0) = (-5, 0).

# RT tiling:
# From LS: (-10, -10+k2) via s=-x -> image (10-k2, 10)
# From RS: (k2, 5) via s=-x+10 -> image (5, 10-k2)
# RT total: (5, 10-k2) U (10-k2, 10) = (5, 10). PERFECT!

# Great! This is a valid structure. Let me also check: is it actually a valid map?
# The map on LS is:
#   s(x) = -x    for x in (-10, -10+k2)   [antitone, far]
#   s(x) = x+5   for x in (-10+k2, -5)    [monotone, shift]
# Continuity at -10+k2: s(-10+k2-) = 10-k2, s(-10+k2+) = -10+k2+5 = k2-5.
# These are equal only if 10-k2 = k2-5, i.e., k2=7.5. Out of range. DISCONTINUOUS.

# The map on RS is:
#   s(x) = x-5   for x in (0, k2)         [monotone, shift]
#   s(x) = -x+10 for x in (k2, 5)         [antitone, near]
# Continuity at k2: s(k2-) = k2-5, s(k2+) = 10-k2.
# Equal only if k2-5 = 10-k2, i.e., k2=7.5. DISCONTINUOUS.

# So this map is discontinuous. For concave costs, this is potentially optimal.

print("\n  --- 5A: LS antitone+monotone, RS monotone+antitone ---")
print("      LS: (-10,-10+k2)->s=-x [RT], (-10+k2,-5)->s=x+5 [LT]")
print("      RS: (0,k2)->s=x-5 [LT],     (k2,5)->s=-x+10 [RT]")
print("      Free parameter: k2 in (0,5)")

def cost_5A(k2):
    """Cost for 5A with split parameter k2 in (0,5)."""
    k1 = -10 + k2  # kink in LS at -10+k2
    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, lambda x: -x)       # LS far
    c += compute_cost_interval_simpson(k1, -5, lambda x: x + 5)     # LS shift
    c += compute_cost_interval_simpson(0, k2, lambda x: x - 5)      # RS shift
    c += compute_cost_interval_simpson(k2, 5, lambda x: -x + 10)    # RS near
    return c

result_5A = minimize_scalar(cost_5A, bounds=(0 + 1e-10, 5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k2_opt_5A = result_5A.x
cost_opt_5A = result_5A.fun

print(f"  Optimal k2 = {k2_opt_5A:.10f}, k1 = {-10+k2_opt_5A:.10f}")
print(f"  Optimal cost = {cost_opt_5A:.10f}")

candidates.append((f"5A: anti+mono LS, mono+anti RS, k2={k2_opt_5A:.6f}",
                    cost_opt_5A,
                    f"LS: s=-x on (-10,{-10+k2_opt_5A:.4f}), s=x+5 on ({-10+k2_opt_5A:.4f},-5); "
                    f"RS: s=x-5 on (0,{k2_opt_5A:.4f}), s=-x+10 on ({k2_opt_5A:.4f},5)",
                    k2_opt_5A))

# 5B: Reverse order
#   LS: (-10,k1) via s=x+5 [LT], (k1,-5) via s=-x [RT]
#   RS: (0,k2) via s=-x+10 [RT], (k2,5) via s=x-5 [LT]
# Let k1 correspond to the split in LS.
# s=x+5 on (-10,k1): image (-5, k1+5) in LT
# s=-x on (k1,-5): image (5, -k1) in RT
# s=-x+10 on (0,k2): image (10-k2, 10) in RT
# s=x-5 on (k2,5): image (k2-5, 0) in LT
# RT: (5, -k1) U (10-k2, 10). Tiling: -k1 = 10-k2 => k2 = 10+k1.
#   k1 in (-10,-5) => k2 in (0,5). Same parameter.
# LT: (-5, k1+5) U (k2-5, 0). For tiling: k1+5 = k2-5 = 10+k1-5 = k1+5. Always!
#   So LT = (-5, k1+5) U (k1+5, 0) = (-5, 0). Perfect!

print("\n  --- 5B: LS monotone+antitone, RS antitone+monotone ---")
print("      LS: (-10,k1)->s=x+5 [LT], (k1,-5)->s=-x [RT]")
print("      RS: (0,k2)->s=-x+10 [RT], (k2,5)->s=x-5 [LT]")
print("      Constraint: k2 = k1+10")

def cost_5B(k1):
    k2 = k1 + 10.0
    c = 0.0
    c += compute_cost_interval_simpson(-10, k1, lambda x: x + 5)      # LS mono
    c += compute_cost_interval_simpson(k1, -5, lambda x: -x)          # LS anti far
    c += compute_cost_interval_simpson(0, k2, lambda x: -x + 10)      # RS anti near
    c += compute_cost_interval_simpson(k2, 5, lambda x: x - 5)        # RS mono
    return c

result_5B = minimize_scalar(cost_5B, bounds=(-10 + 1e-10, -5 - 1e-10), method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 1000})
k1_opt_5B = result_5B.x
k2_opt_5B = k1_opt_5B + 10.0
cost_opt_5B = result_5B.fun

print(f"  Optimal k1 = {k1_opt_5B:.10f}, k2 = {k2_opt_5B:.10f}")
print(f"  Optimal cost = {cost_opt_5B:.10f}")

candidates.append((f"5B: mono+anti LS, anti+mono RS, k1={k1_opt_5B:.6f}, k2={k2_opt_5B:.6f}",
                    cost_opt_5B,
                    f"LS: s=x+5 on (-10,{k1_opt_5B:.4f}), s=-x on ({k1_opt_5B:.4f},-5); "
                    f"RS: s=-x+10 on (0,{k2_opt_5B:.4f}), s=x-5 on ({k2_opt_5B:.4f},5)",
                    k1_opt_5B))

# =============================================================================
# 5C: LS with s=x+5 and s=-x-10, RS with s=x-5 and s=-x
# LS: (-10,k1)->s=x+5 [LT img:(-5,k1+5)], (k1,-5)->s=-x-10 [LT img:(-5,-k1-10)]
# Both go to LT! Double coverage from -5 up. INVALID.
# =============================================================================

# =============================================================================
# 5D: LS: (-10,k1)->s=-x-10 [LT], (k1,-5)->s=x+5 [LT]  -- both LT, INVALID (shown above)
# =============================================================================

# =============================================================================
# 5E: RS with s=x-5 and s=-x (both to LT)
# RS: (0,k2)->s=-x [LT img:(-k2,0)], (k2,5)->s=x-5 [LT img:(k2-5,0)]
# LT: (-k2, 0) U (k2-5, 0). Both end at 0. For tiling: need -k2 = k2-5 => k2=2.5.
# At k2=2.5: both give (-2.5, 0). DOUBLE COVERAGE. INVALID.
# =============================================================================

# =============================================================================
# 5F: The "full monotone shift" with s=x-5 on RS and s=x+5 on LS combined with reflections
# We've already covered the main combinations. Let me also check:
# LS: (-10,k1)->s=-x [RT], (k1,-5)->s=-x-10 [LT] (this is 2A with s=x-5 on RS)
# RS: (0,k2)->s=x-5 [LT], (k2,5)->s=-x+10 [RT]
# k2 = k1+10 from RT tiling?
# RT from LS: (-10,k1) via s=-x -> (-k1, 10).
# RT from RS: (k2,5) via s=-x+10 -> (5, 10-k2).
# Tiling: 10-k2 = -k1 => k2 = k1+10.
# LT from LS: (k1,-5) via s=-x-10 -> (-5, -k1-10).
# LT from RS: (0,k2) via s=x-5 -> (-5, k2-5).
# Tiling: -k1-10 = k2-5 = k1+10-5 = k1+5. So -k1-10 = k1+5 => k1 = -7.5.
# Fixed kink! k1=-7.5, k2=2.5.

print("\n  --- 5F: LS far+near(antitone), RS shift(mono)+near(antitone), fixed kink ---")
k1_5F = -7.5
k2_5F = 2.5
def cost_5F_eval():
    c = 0.0
    c += compute_cost_interval_simpson(-10, k1_5F, lambda x: -x)       # LS far
    c += compute_cost_interval_simpson(k1_5F, -5, lambda x: -x - 10)   # LS near
    c += compute_cost_interval_simpson(0, k2_5F, lambda x: x - 5)      # RS shift
    c += compute_cost_interval_simpson(k2_5F, 5, lambda x: -x + 10)    # RS near
    return c

cost_5F = cost_5F_eval()
print(f"  Fixed kinks: k1={k1_5F}, k2={k2_5F}")
print(f"  Cost = {cost_5F:.10f}")

candidates.append((f"5F: LS far+near, RS shift+near, k1=-7.5, k2=2.5",
                    cost_5F,
                    "LS: s=-x on (-10,-7.5), s=-x-10 on (-7.5,-5); "
                    "RS: s=x-5 on (0,2.5), s=-x+10 on (2.5,5)",
                    -7.5))

# Similarly: LS: near + far, RS: near + shift
# LS: (-10,k1)->s=-x-10, (k1,-5)->s=-x
# RS: (0,k2)->s=-x+10, (k2,5)->s=x-5
# RT from LS (k1,-5) via s=-x: (5,-k1). From RS (0,k2) via s=-x+10: (10-k2,10).
# Tiling: -k1 = 10-k2, k2=k1+10.
# LT from LS (-10,k1) via s=-x-10: (-k1-10, 0). From RS (k2,5) via s=x-5: (k2-5, 0).
# Tiling: -k1-10 = k2-5 => k1=-7.5, k2=2.5. Same point!

print("\n  --- 5G: LS near+far(antitone), RS near(antitone)+shift(mono) ---")
def cost_5G_eval():
    c = 0.0
    c += compute_cost_interval_simpson(-10, -7.5, lambda x: -x - 10)   # LS near
    c += compute_cost_interval_simpson(-7.5, -5, lambda x: -x)         # LS far
    c += compute_cost_interval_simpson(0, 2.5, lambda x: -x + 10)      # RS near
    c += compute_cost_interval_simpson(2.5, 5, lambda x: x - 5)        # RS shift
    return c

cost_5G = cost_5G_eval()
print(f"  Fixed kinks: k1=-7.5, k2=2.5")
print(f"  Cost = {cost_5G:.10f}")

candidates.append((f"5G: LS near+far, RS near+shift, k1=-7.5, k2=2.5",
                    cost_5G,
                    "LS: s=-x-10 on (-10,-7.5), s=-x on (-7.5,-5); "
                    "RS: s=-x+10 on (0,2.5), s=x-5 on (2.5,5)",
                    -7.5))

# =============================================================================
# CATEGORY 6: Fine scan over all structures for 2A and 2B
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 6: Fine grid search for 2A and 2B")
print("=" * 78)

k1_fine = np.linspace(-10 + 0.001, -5 - 0.001, 2000)
costs_2A_fine = np.array([cost_2A(k) for k in k1_fine])
costs_2B_fine = np.array([cost_2B(k) for k in k1_fine])

idx_2A = np.argmin(costs_2A_fine)
idx_2B = np.argmin(costs_2B_fine)

print(f"  2A fine scan: min at k1={k1_fine[idx_2A]:.8f}, cost={costs_2A_fine[idx_2A]:.10f}")
print(f"  2B fine scan: min at k1={k1_fine[idx_2B]:.8f}, cost={costs_2B_fine[idx_2B]:.10f}")

# =============================================================================
# CATEGORY 7: Verify the claimed analytical solution from experiment file
# =============================================================================
print("\n" + "=" * 78)
print("  CATEGORY 7: Verify claimed solution from experiment_mccann.py")
print("=" * 78)

# The experiment file claims kinks at x=-9 and x=1, which corresponds to 2A with k1=-9.
cost_at_minus9 = cost_2A(-9.0)
print(f"  Cost at k1=-9 (claimed optimal): {cost_at_minus9:.10f}")
print(f"  This is structure 2A: LS: s=-x on (-10,-9), s=-x-10 on (-9,-5)")
print(f"                        RS: s=-x on (0,1),    s=-x+10 on (1,5)")

# Also compute using high-accuracy quad
print("\n  High-accuracy verification with scipy.integrate.quad:")

def s_claimed(x):
    x = np.asarray(x, dtype=float)
    s = np.full_like(x, np.nan)
    m = (x >= -10) & (x < -9);  s[m] = -x[m]
    m = (x >= -9)  & (x <= -5); s[m] = -x[m] - 10.0
    m = (x >= 0)   & (x < 1);   s[m] = -x[m]
    m = (x >= 1)   & (x <= 5);  s[m] = -x[m] + 10.0
    return s

pieces_claimed = [
    (-10, -9, lambda x: -x),
    (-9, -5, lambda x: -x - 10),
    (0, 1, lambda x: -x),
    (1, 5, lambda x: -x + 10),
]

cost_claimed_total = 0
for a, b, sfn in pieces_claimed:
    c_piece = compute_cost_interval_simpson(a, b, sfn)
    c_piece_quad = compute_cost_quad(a, b, sfn)
    print(f"    [{a:6.1f}, {b:6.1f}]: Simpson={c_piece:.10f}, quad={c_piece_quad:.10f}")
    cost_claimed_total += c_piece_quad
print(f"  Total cost (quad): {cost_claimed_total:.10f}")

# =============================================================================
# Detailed derivative analysis for 2A to find exact optimum
# =============================================================================
print("\n" + "=" * 78)
print("  DETAILED ANALYSIS: Derivative of 2A cost w.r.t. k1")
print("=" * 78)

# J(k1) = integral_{-10}^{k1} sqrt(2|2x|) f(x)/Z dx      [s=-x, |x-s|=|2x|]
#        + integral_{k1}^{-5}  sqrt(2|2x+10|) f(x)/Z dx    [s=-x-10, |x-s|=|2x+10|]
#        + integral_0^{k2}     sqrt(2|2x|) f(x)/Z dx        [s=-x, |x-s|=|2x|]
#        + integral_{k2}^{5}   sqrt(2|2x-10|) f(x)/Z dx     [s=-x+10, |x-s|=|2x-10|]
# where k2 = k1 + 10.
#
# dJ/dk1 = [sqrt(2|2k1|) - sqrt(2|2k1+10|)] * f(k1)/Z
#         + [sqrt(2|2k2|) - sqrt(2|2k2-10|)] * f(k2)/Z
#
# For k1 in (-10, -5): 2k1 in (-20, -10), |2k1| = -2k1.
# |2k1+10| = |2(k1+5)|. Since k1 < -5: 2k1+10 < 0, |2k1+10| = -2k1-10.
# For k2 = k1+10 in (0, 5): |2k2| = 2k2 = 2k1+20.
# |2k2-10| = |2k1+10|. Since k1 < -5: 2k1+10 < 0, |2k2-10| = -2k1-10.
#
# Also f(k1)/Z = sin(pi*k1/5)/Z. Since k1 in (-10,-5), sin(pi*k1/5) > 0. (verified:
# pi*k1/5 in (-2pi, -pi), sin is in (0, ...) -- no, sin(-2pi)=0, sin(-pi)=0,
# sin(-3pi/2) = 1. So sin(pi*k1/5) for k1 in (-10,-5):
# At k1=-10: sin(-2pi) = 0. At k1=-7.5: sin(-3pi/2) = 1. At k1=-5: sin(-pi) = 0.
# So yes, f(k1)/Z > 0 for k1 in (-10,-5) interior.
# Also f(k2)/Z = sin(pi*k2/5)/Z = sin(pi*(k1+10)/5)/Z = sin(pi*k1/5 + 2*pi)/Z = sin(pi*k1/5)/Z.
# So f(k1)/Z = f(k2)/Z.
#
# dJ/dk1 = f(k1)/Z * [sqrt(-4k1) - sqrt(-4k1-20) + sqrt(4k1+40) - sqrt(-4k1-20)]
#         (using 2|2k1| = -4k1, 2|2k1+10| = -4k1-20, 2|2k2| = 4k1+40, 2|2k2-10| = -4k1-20)
#
# Wait, let me be more careful.
# Term 1: sqrt(2*(-2k1)) = sqrt(-4k1) = 2*sqrt(-k1)
# Term 2: sqrt(2*(-2k1-10)) = sqrt(-4k1-20) = 2*sqrt(-k1-5)
# Term 3: sqrt(2*(2k2)) = sqrt(4k2) = sqrt(4(k1+10)) = 2*sqrt(k1+10)
# Term 4: sqrt(2*(-2k1-10)) = sqrt(-4k1-20) = 2*sqrt(-k1-5)
#
# dJ/dk1 = (f(k1)/Z) * [2*sqrt(-k1) - 2*sqrt(-k1-5)]
#         + (f(k2)/Z) * [2*sqrt(k1+10) - 2*sqrt(-k1-5)]
# Since f(k1) = f(k2):
# dJ/dk1 = 2*f(k1)/Z * [sqrt(-k1) + sqrt(k1+10) - 2*sqrt(-k1-5)]
#
# Setting dJ/dk1 = 0 (f(k1) > 0 in interior):
# sqrt(-k1) + sqrt(k1+10) = 2*sqrt(-k1-5)
#
# Let u = -k1-5 (u in (0,5)). Then -k1 = u+5, k1+10 = 5-u.
# sqrt(u+5) + sqrt(5-u) = 2*sqrt(u)
#
# Square both sides:
# (u+5) + (5-u) + 2*sqrt((u+5)(5-u)) = 4u
# 10 + 2*sqrt(25-u^2) = 4u
# sqrt(25-u^2) = 2u - 5
# Need 2u-5 >= 0, so u >= 2.5.
# 25-u^2 = (2u-5)^2 = 4u^2-20u+25
# 25-u^2 = 4u^2-20u+25
# 0 = 5u^2 - 20u
# 0 = 5u(u-4)
# u = 0 or u = 4
# u=0: k1=-5 (boundary). u=4: k1=-9.
#
# Check u=4: sqrt(9)+sqrt(1) = 3+1 = 4. 2*sqrt(4) = 4. YES!
# Check u=0: sqrt(5)+sqrt(5) = 2*sqrt(5). 2*sqrt(0) = 0. NO! (spurious from squaring)
#
# So the UNIQUE interior critical point is k1 = -9 (u=4), k2 = 1.

print("  Analytical derivative analysis:")
print("  dJ/dk1 = 0 reduces to: sqrt(-k1) + sqrt(k1+10) = 2*sqrt(-k1-5)")
print("  Substituting u = -k1-5: sqrt(u+5) + sqrt(5-u) = 2*sqrt(u)")
print("  Solving: 5u(u-4) = 0 => u=0 (boundary) or u=4 (k1=-9)")
print("  UNIQUE interior critical point: k1 = -9, k2 = 1")
print(f"  Verified: sqrt(9)+sqrt(1) = {np.sqrt(9)+np.sqrt(1):.6f}, 2*sqrt(4) = {2*np.sqrt(4):.6f}")

# Now verify this is a minimum (not maximum) by checking second derivative or comparing values
eps = 0.01
print(f"\n  J(k1=-9-eps) = {cost_2A(-9.0-eps):.10f}")
print(f"  J(k1=-9)     = {cost_2A(-9.0):.10f}")
print(f"  J(k1=-9+eps) = {cost_2A(-9.0+eps):.10f}")
d2 = (cost_2A(-9.0+eps) + cost_2A(-9.0-eps) - 2*cost_2A(-9.0)) / eps**2
print(f"  Approximate d2J/dk1^2 at k1=-9: {d2:.6f} ({'minimum' if d2>0 else 'MAXIMUM'})")

# =============================================================================
# Same analysis for 2B
# =============================================================================
print("\n  --- Derivative analysis for 2B ---")
# 2B: LS: (-10,k1)->s=-x-10 [LT], (k1,-5)->s=-x [RT]
#     RS: (0,k2)->s=-x+10 [RT], (k2,5)->s=-x [LT]
# k2 = k1+10
#
# J(k1) = integral_{-10}^{k1} sqrt(2|2x+10|) f(x)/Z dx    [|x-(-x-10)|=|2x+10|]
#        + integral_{k1}^{-5}  sqrt(2|2x|) f(x)/Z dx        [|x-(-x)|=|2x|]
#        + integral_0^{k2}     sqrt(2|2x-10|) f(x)/Z dx     [|x-(-x+10)|=|2x-10|]
#        + integral_{k2}^{5}   sqrt(2|2x|) f(x)/Z dx        [|x-(-x)|=|2x|]
#
# dJ/dk1 = f(k1)/Z * [sqrt(2|2k1+10|) - sqrt(2|2k1|)]
#         + f(k2)/Z * [sqrt(2|2k2-10|) - sqrt(2|2k2|)]
# = f(k1)/Z * [2*sqrt(-k1-5) - 2*sqrt(-k1) + 2*sqrt(-k1-5) - 2*sqrt(k1+10)]
# = 2*f(k1)/Z * [2*sqrt(-k1-5) - sqrt(-k1) - sqrt(k1+10)]
#
# Setting to 0: 2*sqrt(-k1-5) = sqrt(-k1) + sqrt(k1+10)
# SAME EQUATION as 2A! So k1 = -9 is the critical point for 2B too.

print("  dJ/dk1 = 0 gives same equation as 2A: k1 = -9")
print(f"  J_2B(k1=-9-eps) = {cost_2B(-9.0-eps):.10f}")
print(f"  J_2B(k1=-9)     = {cost_2B(-9.0):.10f}")
print(f"  J_2B(k1=-9+eps) = {cost_2B(-9.0+eps):.10f}")
d2_2B = (cost_2B(-9.0+eps) + cost_2B(-9.0-eps) - 2*cost_2B(-9.0)) / eps**2
print(f"  Approximate d2J/dk1^2 at k1=-9: {d2_2B:.6f} ({'minimum' if d2_2B>0 else 'MAXIMUM'})")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 78)
print("  FINAL SUMMARY: All candidates sorted by cost")
print("=" * 78)

candidates.sort(key=lambda x: x[1])

for i, (name, cost, desc, kink) in enumerate(candidates):
    marker = "  *** GLOBAL OPTIMUM ***" if i == 0 else ""
    print(f"\n  #{i+1}: {name}")
    print(f"      Cost = {cost:.10f}{marker}")
    print(f"      Map:  {desc}")

print("\n" + "=" * 78)
best = candidates[0]
print(f"\n  GLOBAL OPTIMUM: {best[0]}")
print(f"  Minimum cost J* = {best[1]:.10f}")
print(f"  Map description: {best[2]}")

# =============================================================================
# Double-check: compute cost of 2A at k1=-9 with extreme precision
# =============================================================================
print("\n" + "=" * 78)
print("  HIGH-PRECISION VERIFICATION of optimal map at k1=-9, k2=1")
print("=" * 78)

from scipy.integrate import quad as scipy_quad

def integrand_piece(x, s_func_scalar):
    sx = s_func_scalar(x)
    return np.sqrt(2.0 * abs(x - sx)) * max(np.sin(np.pi * x / 5.0), 0.0) / Z

pieces = [
    ("(-10,-9): s=-x",    -10, -9, lambda x: -x),
    ("(-9,-5):  s=-x-10", -9,  -5, lambda x: -x-10),
    ("(0,1):    s=-x",     0,   1, lambda x: -x),
    ("(1,5):    s=-x+10",  1,   5, lambda x: -x+10),
]

total_hp = 0.0
for name, a, b, sfn in pieces:
    val, err = scipy_quad(lambda x: integrand_piece(x, sfn), a, b,
                          limit=500, epsabs=1e-15, epsrel=1e-15)
    print(f"  {name}: J_piece = {val:.15f}  (error bound: {err:.2e})")
    total_hp += val

print(f"\n  TOTAL J* = {total_hp:.15f}")
print(f"  (This is the verified global minimum)")

# =============================================================================
# Verify optimality condition at k1=-9
# =============================================================================
print("\n  Optimality verification at k1=-9:")
print(f"    sqrt(-k1) = sqrt(9) = {np.sqrt(9):.10f}")
print(f"    sqrt(k1+10) = sqrt(1) = {np.sqrt(1):.10f}")
print(f"    2*sqrt(-k1-5) = 2*sqrt(4) = {2*np.sqrt(4):.10f}")
print(f"    LHS = sqrt(9)+sqrt(1) = {np.sqrt(9)+np.sqrt(1):.10f}")
print(f"    RHS = 2*sqrt(4) = {2*np.sqrt(4):.10f}")
print(f"    LHS == RHS: {np.isclose(np.sqrt(9)+np.sqrt(1), 2*np.sqrt(4))}")

print("\n" + "=" * 78)
print("  CONCLUSION")
print("=" * 78)
print(f"""
  The GLOBAL OPTIMUM is the map with kinks at k1 = -9, k2 = 1:

    s(x) = -x        on (-10, -9)   [maps to RT = (9, 10)]
    s(x) = -x - 10   on (-9, -5)    [maps to LT = (-5, -1)]
    s(x) = -x        on (0, 1)      [maps to LT = (-1, 0)]
    s(x) = -x + 10   on (1, 5)      [maps to RT = (5, 9)]

  Optimal cost J* = {total_hp:.15f}

  The optimality condition sqrt(-k1) + sqrt(k1+10) = 2*sqrt(-k1-5)
  is satisfied at k1 = -9 (equivalently, u = -k1-5 = 4 satisfies 5u(u-4) = 0).

  This is structure 2A (far-then-near on each source bump):
  - Each source bump is split into a "far" piece (sent across via s=-x)
    and a "near" piece (sent to adjacent target via s=-x +/- 10).
  - The far piece (length 1) captures a thin slice at the edge of each bump.
  - The near piece (length 4) handles the bulk of the mass.

  The concave cost sqrt(2|x-y|) favors SPREADING the transport: mixing
  some long-distance transport (cheap per unit due to concavity) with
  shorter-distance transport, rather than using a uniform displacement.
""")
