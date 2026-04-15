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
    coulumb/
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

### `experiments/`

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
  - Implementation of the example from Robert J. McCann, *Exact solutions to the transportation problem on the line*:
    https://royalsocietypublishing.org/rspa/article-abstract/455/1984/1341/80376/Exact-solutions-to-the-transportation-problem-on
  - Reference:
    McCann, R. J. (1999). Exact solutions to the transportation problem on the line. *Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences*, 455(1984), 1341–1380. doi:10.1098/rspa.1999.0364.

### `experiments2/`

- `experiments2/lens/experiment_lens_sqp_gpu.py`
  - SQP- and GPU-oriented lens / reflector experiment, matching the optical transport setting in `SQP.tex`.
- `experiments2/lens/freeform_lens_experiment.py`
  - Freeform lens / reflector design variant exploring the same physical optical cost model.
  - Reference for reflector antenna: Wang, X.-J. (2004). On the design of a reflector antenna II. *Calculus of Variations and Partial Differential Equations*, 20(3), 329–341. doi:10.1007/s00526-003-0239-4.
- `experiments2/lagrangian/experiment_lagrangian_sqp.py`
  - Lagrangian obstacle-cost experiment using inner path optimization costs.
  - Reference: Pooladian, A.-A., Finlay, C., & Oberman, A. (2024). Neural Optimal Transport with Lagrangian Costs. *arXiv preprint arXiv:2406.00288*.
- `experiments2/nonmonotone/nonmonotone_transport_sqp.py`
  - Non-monotone transport experiment for displacement-maximizing concave costs.
- `experiments2/mccann/experiment_mccann.py`
  - Main McCann concave-cost experiment for piecewise-structured deterministic transport.
- `experiments2/mccann/compute_true_optimal.py`
  - Validation helper computing reference optimal solutions.
- `experiments2/mccann/find_optimal_kink.py`
  - Structural analysis helper for McCann kink locations.
- `experiments2/labor_market/labor_market_experiment.py`
  - Labor-market matching experiment applying the Monge OT framework.
- `experiments2/coulumb/experiment_coulomb_sce.py`
  - Coulomb repulsive-cost experiment for strongly correlated electrons (SCE), implementing the self-transport with 1/|x-y| cost from `SQP.tex`.
  - Reference: Colombo, M., De Pascale, L., & Di Marino, S. (2015). Multimarginal optimal transport maps for one-dimensional repulsive costs. *Canadian Journal of Mathematics*, 67(2), 350–368. doi:10.4153/CJM-2014-011-x.



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
