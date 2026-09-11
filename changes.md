# Changes

## Unreleased

## Changed

* Lab 4: Optional exercise 4 now implements a pre-calculation of acceptance probabilities.
* Lab 4: STUDENT: added missing exercise 3.

## Fixed

* Lab 4: SOLUTIONS fix of code to select correct `lambda_t` in thinning example.

## V4.1.0

### Added

* Lab1: Added `urgent_care_sim.py` a full version of the model built in lab1 for students to browse and use.
* Lab1: Added `urgent_care_sim.ipynb` a notebook to demo use of `urgent_care_sim.py`

### Changed
 
* Lab1: Added call centre image to exercise 9.
* Lab 2: markdown tables for parameters combined into one
* Lab 2: `Scenario` now accepts parameters set to default global constants.  A better design.
* Lab 2: `get_scenario` modified to use new `Scenario` parameter passing format.

### Fixed

* Lab 1: fix of minor typos in markdown for solutions and students copy
* Lab 1: fix of `Patient` and `UrgentCareCallCentre` classes to correctly reference member attribute `self.env`
* Lab 2: parameters in problem description fixed to match the values used by the simulation model

## V4.0.0

## Added 

* Repository essentials: change log + citation file.
* Added new replications algorithm code (implementation of Hoad et al (2010) to Lab 6.

## Changed
* Libraries updated August 2024
* sim-tools and jupyter-lab updated Sept 2025


## Fixed
* Patched lab 1: `Dataframe.append()` is deprecated. Replaced with recommended `pd.concat()`
* Updated np.Inf -> np.inf

## [v3.0.0](https://github.com/health-data-science-OR/stochastic_systems/releases/tag/v3.0.0)

## Changed
* Libraries updated Dec 2022

## Fixed
* Minor update to lab 1.  pd.precision no longer worked.  dataframe.round() used instead.
