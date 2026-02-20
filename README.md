# Monge OT via SQP (1D Experiments)

This repository explores **1D Monge optimal transport** by parameterizing transport maps and solving the resulting constrained problem with **Sequential Quadratic Programming (SQP)**.

## Objective

Given a source density `f` and target density `g`, learn a monotone map `s` that:

1. pushes `f` to `g` (density matching), and
2. keeps transport cost small.

The code compares analytic maps (when available) against SQP-learned maps across several source/target pairs.

## Mathematical Formulation

The main constrained problem used across scripts is:

\[
\min_{\theta}\; J(\theta)=\frac12\,\mathbb E_{X\sim f}\big[(s_\theta(X)-X)^2\big]
\quad\text{s.t.}\quad
\zeta(y;\theta)=P_f(s_\theta)(y)-g(y)=0
\]

where `P_f(s)` is the pushforward density from change-of-variables:

\[
P_f(s)(y)=\sum_{x:\,s(x)=y}\frac{f(x)}{|s'(x)|}.
\]

On a grid, constraints are enforced pointwise and SQP solves regularized KKT systems of the form:

\[
\begin{bmatrix}
H & A^\top \\
A & -\rho I
\end{bmatrix}
\begin{bmatrix}
\Delta\theta \\
w
\end{bmatrix}
=
-\begin{bmatrix}
\nabla_\theta \mathcal L \\
\zeta
\end{bmatrix}
\]

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
