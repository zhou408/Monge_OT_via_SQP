# Python File Index

This index summarizes every Python file included in this repository snapshot and indicates how each one fits into the paper’s experimental landscape.

## Lens / reflector experiments

### `experiments/lens/experiment_lens_sqp_gpu.py`
### `experiments2/lens/experiment_lens_sqp_gpu.py`

Role:

- Main SQP- and GPU-oriented lens / reflector experiment driver.

Paper relation:

- Part of the lens / reflector experiment family, including the `S^1` reflector setting.

### `experiments/lens/freeform_lens_experiment.py`
### `experiments2/lens/freeform_lens_experiment.py`

Role:

- Freeform lens and reflector experiment script from the same family.

Paper relation:

- Belongs to the lens / reflector experiments and provides a broader exploratory variant of that setup.

## Lagrangian obstacle-cost experiments

### `experiments/lagrangian/experiment_lagrangian_sqp.py`
### `experiments2/lagrangian/experiment_lagrangian_sqp.py`

Role:

- Main Lagrangian obstacle-cost experiment.

Paper relation:

- Directly associated with `SQP.tex`, Experiment `exp:Bilevel`.

## Non-monotone transport experiments

### `experiments/nonmonotone/nonmonotone_transport_sqp.py`
### `experiments2/nonmonotone/nonmonotone_transport_sqp.py`

Role:

- Main non-monotone displacement-maximizing transport experiment.

Paper relation:

- Associated with `exp5_concave_updated.tex`, Experiment `exp:nonmono`.

## McCann concave-cost experiments

### `experiments/mccann/experiment_mccann.py`
### `experiments2/mccann/experiment_mccann.py`

Role:

- Main McCann concave-cost experiment script.

Paper relation:

- Associated with `mccann_experiment.tex`.

### `experiments/mccann/compute_true_optimal.py`
### `experiments2/mccann/compute_true_optimal.py`

Role:

- LP-based validation helper for the McCann example.

Paper relation:

- Supports the Kantorovich / deterministic-structure discussion for the McCann experiment.

### `experiments/mccann/find_optimal_kink.py`
### `experiments2/mccann/find_optimal_kink.py`

Role:

- Structural and kink-analysis helper for the McCann example.

Paper relation:

- Supports the analytical description of the piecewise-affine solution.

## Labor-market experiments

### `experiments/labor_market/labor_market_experiment.py`
### `experiments2/labor_market/labor_market_experiment.py`

Role:

- Labor-market matching experiment formulated through the transport framework.

Paper relation:

- Part of the repository’s paper experiments and application-oriented examples.

## Helper modules

### `experiments/plotting_helpers.py`

Role:

- Plotting and visualization helper used by the original experiment suite.

Paper relation:

- Supports plotting for experiments across the repository.
