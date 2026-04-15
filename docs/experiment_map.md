# Experiment Map

This document maps the code in this repository to the experiments in "BEYOND BRENIER: SEQUENTIAL QUADRATIC PROGRAMMING FOR GENERAL-COST MONGE OPTIMAL TRANSPORT" by Zihe Zhou, Harsha Honnappa, and Raghu Pasupathy.

It covers experiment scripts in both `experiments/` and `experiments2/`. The `experiments2/` folder contains a second set of the main experiment families, often representing newer or alternate implementations of the same research experiments.

## Lens and reflector experiments

Files:

- `../experiments/lens/experiment_lens_sqp_gpu.py`
- `../experiments/lens/freeform_lens_experiment.py`
- `../experiments2/lens/experiment_lens_sqp_gpu.py`
- `../experiments2/lens/freeform_lens_experiment.py`

Paper context:

- Lens / reflector experiments in the paper
- Reflector-on-`S^1` discussion in `SQP.tex`

Notes:

- `experiment_lens_sqp_gpu.py` is the SQP- and GPU-oriented implementation.
- `freeform_lens_experiment.py` is the exploratory variant in the same lens family.
- Both `experiments/` and `experiments2/` include these scripts.

## Lagrangian obstacle-cost experiment

Files:

- `../experiments/lagrangian/experiment_lagrangian_sqp.py`
- `../experiments2/lagrangian/experiment_lagrangian_sqp.py`

Paper context:

- `SQP.tex`, Experiment `exp:Bilevel`
- Section `sec:exp-lagrangian`

Notes:

- This script implements the Lagrangian obstacle-cost experiment.
- Both folders contain the same core experiment driver.

## Non-monotone displacement-maximizing experiment

Files:

- `../experiments/nonmonotone/nonmonotone_transport_sqp.py`
- `../experiments2/nonmonotone/nonmonotone_transport_sqp.py`

Paper context:

- `exp5_concave_updated.tex`
- Experiment `exp:nonmono`

Notes:

- This script implements the `Beta(2,5) -> Beta(5,2)` displacement-maximizing experiment with costs of the form `-|x-y|^p`.
- Both folders contain this script.

## McCann concave-cost experiment

Files:

- `../experiments/mccann/experiment_mccann.py`
- `../experiments/mccann/compute_true_optimal.py`
- `../experiments/mccann/find_optimal_kink.py`
- `../experiments2/mccann/experiment_mccann.py`
- `../experiments2/mccann/compute_true_optimal.py`
- `../experiments2/mccann/find_optimal_kink.py`

Paper context:

- `mccann_experiment.tex`
- Experiment `exp:mccann`

Notes:

- `experiment_mccann.py` is the main experiment script.
- `compute_true_optimal.py` provides LP-based validation.
- `find_optimal_kink.py` supports structural analysis of the piecewise-affine solution.
- Both folders contain this McCann experiment family.

## Labor-market experiment

Files:

- `../experiments/labor_market/labor_market_experiment.py`
- `../experiments2/labor_market/labor_market_experiment.py`

Paper context:

- Labor-market experiments in the paper’s broader experimental program.

Notes:

- This script presents the transport framework in an economic matching setting.
- Both folders contain the same core labor-market experiment.

## Coulomb repulsive-cost experiment

File:

- `../experiments2/coulumb/experiment_coulomb_sce.py`

Paper context:

- Coulomb repulsive costs in `SQP.tex`, physics-inspired transport costs section.

Notes:

- This script implements the strongly correlated electrons (SCE) self-transport problem with 1/|x-y| cost, as described in the DFT setting.

## Helper files

File:

- `../experiments/plotting_helpers.py`

Notes:

- This helper module is used by the original experiment suite for plotting and visualization.
- There is not currently a copy of this helper in `experiments2/`.
