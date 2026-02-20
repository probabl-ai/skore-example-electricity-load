# Electricity load forecasting

This example is adapted from https://github.com/probabl-ai/forecasting.

The goal is to have a dataset and prediction pipeline that can be used to
experiment with features of skore and other packages.

The main branch contains a bare-bones version of the project with scripts for
defining and cross-validating a pipeline and running hyperparameter search.
Other branches can be created to explore how this minimal project could be
improved by introducing more tools, good practices, etc.

## Contents

For simplicity, historical data is stored in the repo in `datasets`.
Outputs can be stored in `results` (ignored by git). `utils.py` contains
functions for loading data, defining the pipeline, cross-validation splits etc.
. `eda.py`, `cross_validate.py` and `search.py` are scripts that perform basic
exploratory data analysis, cross-validating the default pipeline, running
hyperparameter search & scoring the best model on a held-out set.

## Challenges

Some things for which current or future versions of skore could help:

- Storing and tracking results
- Comparing results across runs, checking how scores evolve after improving the pipeline
- Summarizing and navigating past experiments
- Ensuring reproducibility of results
- Facilitating the reuse of fitted pipelines
- Enabling collaboration when several people are making different improvements to the pipeline at the same time
- Tracking the Python environment
- Caching computations for faster iterations
- Performing more checks, diagnoses, plots etc to detect errors and possible improvements
- ...
