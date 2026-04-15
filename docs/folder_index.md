# Folder Index

This document describes the directory layout of the repository and the purpose of each top-level section.

## `docs`

Purpose:

- Repository documentation for readers who want to connect the code to the paper experiments.

Contents:

- `experiment_map.md`
- `python_file_index.md`
- `folder_index.md`

## `experiments`

Purpose:

- Original experiment scripts from the paper.

Subfolders:

- `lens`
- `lagrangian`
- `nonmonotone`
- `mccann`
- `labor_market`

Other files:

- `plotting_helpers.py`

## `experiments2`

Purpose:

- Second set of experiment scripts, often representing newer or alternate implementations of the same experiment families.

Subfolders:

- `lens`
- `lagrangian`
- `nonmonotone`
- `mccann`
- `labor_market`

## Naming conventions in this repository

- Filenames containing spaces in the original working tree were normalized in this snapshot:
  - `experiment_mccann Code.py` -> `experiment_mccann.py`
  - `compute_true_optimal Code.py` -> `compute_true_optimal.py`
- Original nested file-shaped directories were flattened into standard file paths under the appropriate experiment folders.

## Scope

- This repository focuses on experiment code and lightweight documentation for the paper.
- The structure is designed to make the experimental organization legible to new readers and collaborators.
