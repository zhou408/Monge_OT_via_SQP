# Monge OT via SQP (1D Experiments)

This repository explores **1D Monge optimal transport** by parameterizing transport maps and solving the resulting constrained problem with **Sequential Quadratic Programming (SQP)**.

## Objective

Given a source density `f` and target density `g`, learn a monotone map `s` that:

1. pushes `f` to `g` (density matching), and
2. keeps transport cost small.

The code compares analytic maps (when available) against SQP-learned maps across several source/target pairs.

## Mathematical Formulation

At the continuous level, this starts from a Monge-style constrained optimization over maps `s`:

```text
min over maps s:  J[s] = integral c(x, s(x)) f(x) dx
subject to:       P_f(s)(y) = g(y)
```

Pushforward density (change of variables):

```text
P_f(s)(y) = sum over x such that s(x)=y of f(x) / |s'(x)|
```

In most experiments, `c(x,s(x)) = 0.5 * (s(x)-x)^2`, so:

```text
J[s] = 0.5 * E_{X~f}[(s(X)-X)^2]
```

McCann-style sections also include a non-quadratic transport cost (`sqrt(2|x-s(x)|)`), smoothed in code for numerical stability.

### What SQP Is (Briefly)

SQP (Sequential Quadratic Programming) solves a nonlinear constrained problem by repeating:

1. linearize constraints near current parameters,
2. locally quadratic-approximate the objective/Lagrangian,
3. solve the resulting KKT linear system for a step,
4. update primal/dual variables (optionally with trust-clip or line search).

### How The Original Problem Is Approximated Here

The original problem is infinite-dimensional (optimize over functions `s`). In these scripts it is approximated as finite-dimensional optimization:

1. **Map parameterization**
   - affine/quadratic maps, or
   - monotone piecewise-linear maps with slopes `a_i = exp(gamma_i) > 0`.
   This gives a finite parameter vector `theta`.

2. **Objective approximation**
   - `J(theta)` is evaluated with analytic moments when available (some affine/quadratic cases), or
   - numerical quadrature / quantile-grid averaging for piecewise-linear maps.
   - solvers typically use Gauss-Newton style Hessian approximations for stability.

3. **Constraint discretization**
   - instead of enforcing `P_f(s)=g` for all `y`, enforce residuals on a finite grid:
   - `zeta_j(theta) = P_f(s_theta)(y_j) - g(y_j) = 0`.
   - Jacobians are analytic in several scripts and finite-difference in others.

With these approximations, each SQP step solves a regularized KKT system:

```text
[ H    A^T ] [Delta_theta] = -[ grad_theta L ]
[ A   -rhoI] [    w      ]   [    zeta      ]
```

with optional trust-clipping / fixed-step / line-search updates.

## Experiments Included

- `experiments/gaussian_linear_experiments.py`
  - Gaussian -> Gaussian with affine map.
- `experiments/gaussian_quadratic_experiments.py`
  - Gaussian -> Gaussian with quadratic map parameterization.
- `experiments/exp_to_exp_experiments.py`
  - Exponential -> Exponential.
- `experiments/gaussian_to_exponential_experiments.py`
  - Gaussian -> Exponential using monotone piecewise-linear maps.
- `experiments/exponential_to_gaussian_experiments.py`
  - Exponential -> Gaussian using monotone piecewise-linear maps.
- `experiments/uniform_to_beta_experiments.py`
  - Uniform -> Beta.
- `experiments/mccann_concave_experiments.py`
  - McCann-style concave / nonconvex examples (piecewise and polynomial variants).

## How to Run

Each script is notebook-style and exits quickly by default. Run full demo blocks with `--demo full`:

```powershell
python experiments/gaussian_linear_experiments.py --demo full
python experiments/gaussian_quadratic_experiments.py --demo full
python experiments/exp_to_exp_experiments.py --demo full
python experiments/gaussian_to_exponential_experiments.py --demo full
python experiments/exponential_to_gaussian_experiments.py --demo full
python experiments/uniform_to_beta_experiments.py --demo full
python experiments/mccann_concave_experiments.py --demo full
```

## Notes

- These files are direct notebook-derived experiment scripts, so some files contain repeated exploratory blocks.
- `experiments/plotting_helpers.py` provides optional shared plotting utilities.
- Parameter defaults in scripts are intentionally preserved from notebook exports.
