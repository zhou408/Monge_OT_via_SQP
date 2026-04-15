# BEYOND BRENIER: SEQUENTIAL QUADRATIC PROGRAMMING FOR GENERAL-COST MONGE OPTIMAL TRANSPORT

This repository contains experiment code for the paper "BEYOND BRENIER: SEQUENTIAL QUADRATIC PROGRAMMING FOR GENERAL-COST MONGE OPTIMAL TRANSPORT" by Zihe Zhou, Harsha Honnappa, and Raghu Pasupathy.

It also includes 1D Monge OT experiment scripts that parameterize transport maps and solve the resulting constrained problem with Sequential Quadratic Programming (SQP).

All experiments in this repository are from the paper, and the code explores computational examples for Monge optimal transport beyond the classical Brenier regime, including general transport costs, nonconvex formulations, and SQP-based numerical methods.

## Repository structure

```text
project_guide/
  README.md
  .gitignore
  docs/
    experiment_map.md
    folder_index.md
    python_file_index.md
  experiments/
    labor_market/
    lagrangian/
    lens/
    mccann/
    nonmonotone/
    plotting_helpers.py
  experiments2/
    labor_market/
    lagrangian/
    lens/
    mccann/
    nonmonotone/
```

## 1D Monge SQP experiments

This repository explores 1D Monge optimal transport by parameterizing transport maps and solving the resulting constrained problem with SQP.

### Objective

Given a source density `f` and target density `g`, learn a monotone map `s` that:

1. pushes `f` to `g` (density matching), and
2. keeps transport cost small.

The code compares analytic maps (when available) against SQP-learned maps across several source/target pairs.

### Mathematical formulation

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

### What SQP is

SQP (Sequential Quadratic Programming) solves a nonlinear constrained problem by repeating:

1. linearize constraints near current parameters,
2. locally quadratic-approximate the objective/Lagrangian,
3. solve the resulting KKT linear system for a step,
4. update primal/dual variables (optionally with trust-clip or line search).

### How the original problem is approximated with SQP

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

## Experiments included

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
  - Implementation based on Robert J. McCann, *Exact solutions to the transportation problem on the line*:
    https://royalsocietypublishing.org/rspa/article-abstract/455/1984/1341/80376/Exact-solutions-to-the-transportation-problem-on

## Experiments

### `experiments/lens`

- `experiment_lens_sqp_gpu.py`
- `freeform_lens_experiment.py`

Experiments for lens and reflector transport, including the `S^1` reflector setting.

### `experiments/lagrangian`

- `experiment_lagrangian_sqp.py`

Experiments for Lagrangian obstacle-cost optimal transport, where the transport cost is defined through an inner path optimization problem.

### `experiments/nonmonotone`

- `nonmonotone_transport_sqp.py`

Experiments for displacement-maximizing and non-monotone transport under concave costs.

### `experiments/mccann`

- `experiment_mccann.py`
- `compute_true_optimal.py`
- `find_optimal_kink.py`

Experiments and supporting analysis for the McCann concave-cost example.

### `experiments/labor_market`

- `labor_market_experiment.py`

Labor-market matching experiments formulated through the same transport framework.

### `experiments2`

This repository also contains a second set of experiment scripts in `experiments2/`,

- `experiments2/lens/experiment_lens_sqp_gpu.py`
  - Main SQP- and GPU-oriented lens and reflector experiment. This manifests the optical path-cost setting from `SQP.tex`, where the cost is given by the freeform illumination / reflector travel time and the Monge map assigns rays deterministically.
- `experiments2/lens/freeform_lens_experiment.py`
  - Freeform lens and reflector design variant exploring the same optical transport model with additional geometry or design flexibility.
- `experiments2/lagrangian/experiment_lagrangian_sqp.py`
  - Lagrangian obstacle-cost experiment that uses an inner path optimization to compute transport costs through a potential landscape, matching the Hamilton-Jacobi / action-cost setting described in `SQP.tex`.
- `experiments2/nonmonotone/nonmonotone_transport_sqp.py`
  - Non-monotone transport experiment for displacement-maximizing concave costs, following the Beta-to-Beta nonconvex example family where the optimal map may violate simple monotonicity structure.
- `experiments2/mccann/experiment_mccann.py`
  - McCann-style concave-cost experiment studying deterministic transport under concave interaction costs and piecewise-structured solutions.
- `experiments2/mccann/compute_true_optimal.py`
  - Auxiliary validation script that computes a reference Kantorovich/LP optimal solution against which the Monge map can be compared.
- `experiments2/mccann/find_optimal_kink.py`
  - Structural analysis helper for identifying piecewise-affine kinks in the McCann model.
- `experiments2/labor_market/labor_market_experiment.py`
  - Labor-market matching experiment that applies the Monge OT framework to deterministic worker–job assignment, reflecting the matching motivation in `SQP.tex`.

## How to run

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

## Documentation

- [docs/experiment_map.md](./docs/experiment_map.md): maps scripts to the paper experiments.
- [docs/python_file_index.md](./docs/python_file_index.md): summarizes every Python file included here.
- [docs/folder_index.md](./docs/folder_index.md): describes the directory layout.

## Notes

- These files are direct notebook-derived experiment scripts, so some files contain repeated exploratory blocks.
- `experiments/plotting_helpers.py` provides optional shared plotting utilities.
- Parameter defaults in scripts are intentionally preserved from notebook exports.
- `experiments2/` mirrors the main experiment families with alternate or newer implementations.
