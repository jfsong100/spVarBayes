Reproducing Results for Paper: Fast Variational Bayes for Large Spatial
Data
================
Jiafang Song, Abhirup Datta

# Overview

This document describes how to reproduce all simulation, real-world, and
supplementary results in the paper **Fast Variational Bayes for Large
Spatial Data**.

## Get only the `paper/` folder

If only need the code in `paper/` to reproduce the results and do not
want to check out the full repository, use Git sparse checkout.

``` bash
git clone --no-checkout https://github.com/jfsong100/spVarBayes.git
cd spVarBayes
git sparse-checkout init --no-cone
git sparse-checkout set "paper/"
git checkout main
```

The workflow has the following components:

1.  **Environment setup**
2.  **Simulation studies**  
    (Figures 1–3)
3.  **Real-world data analysis**  
    (Figures 4–7, Table 2)
4.  **Supplementary experiments**  
    (Figures G.1–G.2, H.1–H.6, J.1–J.2, K.1–K.2, Table 6)

# Notes on Memory and Runtime

- Large-scale simulations (`n_index = 4, 5`):
  - Require large memory for **DKLGP**, **VNNGP**, and **spVB-MFA-LR**
    (e.g. 100–150 GB for `n_index = 4`, 300 GB for `n_index = 5`).
- The numerical results of this paper is running on a cluster:
  - Request sufficient RAM in the job script (e.g. 150–300 GB, depending
    on `n_index` and method).
  - Running different methods and `n_index` combinations in parallel
    jobs can speed up the complete reproduction.

# Environment Setup

## R / spVarBayes

Install the `spVarBayes` package from GitHub in R:

``` r
# Install devtools if needed
install.packages("devtools")

# Install spVarBayes
devtools::install_github("jfsong100/spVarBayes")

# Install spNNGP
install.packages("spNNGP")

# Install necessary packages for supporting analysis
pkgs = c(
  "BRISC","MASS","fields","Matrix","hdf5r","scoringutils","dplyr","parallel",
  "ggplot2","viridis","patchwork","cowplot","tidyverse","ggrastr","tidyr","scales",
  "gridExtra"
)
missing = pkgs[!vapply(pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0) {
  install.packages(missing, dependencies = TRUE)
}
```

This package provides the following methods:

- `spVB-MFA`
- `spVB-MFA-LR`
- `spVB-NNGP`
- `spVB-NNGP-joint`

Make sure the R version and system compilers are configured so that the
package (and its C++/Rcpp components) install successfully.

## Python / VNNGP Environment

We recommend a dedicated conda environment for **VNNGP**.

``` bash
# Create and activate environment
conda create -n env_vnngp python=3.10
conda activate env_vnngp

# Core dependencies
conda install pytorch==1.13.1
conda install gpytorch==1.10
conda install conda-forge::nb_conda
conda install notebook==6.5.6
conda install jupyter
conda install h5py
conda install pandas
conda install matplotlib
```

Use this environment when running `VNNGP.py` (both simulations and
real-data analysis).

## Python / DKLGP Environment

Create a separate conda environment for **DKLGP**:

``` bash
# Create and activate environment
conda create -n env_dklgp python=3.10
conda activate env_dklgp
```

Clone and install the DKLGP package:

``` bash
# In a folder where you want the DKL-GP source
git clone https://github.com/katzfuss-group/DKL-GP.git

# Direct to that folder (change this)
cd ~/DKL-GP

# Install DKLGP and its Python dependencies
source INSTALL
```

Install Jupyter-related dependencies:

``` bash
conda install conda-forge::nb_conda
conda install notebook==6.5.6
conda install jupyter
conda install h5py
conda install matplotlib
```

Use this environment when running `DKLGP.py` and `DKLGP_default.py`.

# Simulation Studies

The simulation studies in the paper consider sample sizes
$`n = 1000,5000,10000,50000,100000`$, with the corresponding n_index
values $`1,2,3,4,5`$. For a quick code check, we recommend running
`n_index=3`, which reproduces the results for $`n=10000`$. To fully
reproduce the paper’s simulation results, run all n_index values and set
the seed from 1 to 100.

``` bash
# Inside spVarBayes folder, change to the simulations directory
cd paper/simulations
```

## Data Generation

**Script:** `data_generation.R`

- **Inputs:**
  - `n_index` (controls sample size)
  - `t` (seed)
- **Usage (example from shell):**

``` bash
## Example: generate data for n_index = 3 t = 1
Rscript data_generation.R n_index=3 t=1
```

``` bash
## Example: generate data for n_index = 1, 2, 3 and t = 1, 2
for n_index in 1 2 3; do
  for t in 1 2; do
    Rscript data_generation.R n_index=${n_index} t=${t}
  done
done
```

- **Output:**
  - Creates a folder: `data_sim/`
  - For each combination of `n_index` and `t`, saves simulated data.
  - For `n_index = 1, 2, 3`:
    - Also saves the empirical covariance matrix (large files).
  - For `n_index = 4, 5`:
    - Empirical covariance is **not** saved due to memory limits.
    - Only outcomes, covariates, and coordinates are saved.

Details for each scenario are described in Section 4 of the paper.

## R-based Methods: spVarBayes + spNNGP

**Script:** `spVB_spNNGP.R`

- **Inputs:**
  - `n_index`
  - `t` (seed)
- **Usage (example):**

``` bash
## Example: run analysis for n_index = 3 t = 1
Rscript spVB_spNNGP.R n_index=3 t=1
```

``` bash
# e.g. for multiple runs
for n_index in 1 2 3; do
  for t in 1 2; do
    Rscript spVB_spNNGP.R n_index=${n_index} t=${t}
  done
done
```

- **What it does:**
  - Reads data from `data_sim/`.
  - Runs:
    - `spVB-MFA`
    - `spVB-MFA-LR`
    - `spVB-NNGP`
    - `spVB-NNGP-joint`
    - `spNNGP`
  - Creates a folder: `R_results/`.
- **Saved outputs in `R_results/` include:**
  - **Fixed effects:**
    - `output_beta_vector`: point estimates of $`\beta`$
    - `CI_beta`: confidence intervals for $`\beta`$
  - **Hyperparameters:**
    - `output_theta_vector`: point estimates for
      $`\theta=\{\sigma^2,\tau^2,\phi\}`$
  - **Random effects summaries:**
    - `output_list`:
      - Empirical mean and variance
      - Variational approximated mean and variance
      - spNNGP approximated mean and variance
  - **Coverage and interval metrics for random effects:**
    - `CI_w`: confidence intervals, coverage, CRPS, interval scores
  - **KL and timing (per method):**
    - `output_vector`: summarizes KL divergence and running time

## DKLGP Simulations

**Script:** `DKLGP.py` `DKLGP_default.py`

- **Note:** the only difference between `DKLGP.py` and
  `DKLGP_default.py` is convergence options inside the file
  - **DKLGP-default:** set `converged = FALSE`
  - **DKLGP:** set `converged = TRUE`
- **Environment:**

``` bash
conda activate env_dklgp
```

- **Inputs:**
  - `n_index`
  - `seed`
  - A `setups.yaml` file in the **same folder** as `DKLGP.py` and
    `DKLGP_default.py`, containing default settings from the DKLGP
    paper.
- **Usage (example):**

``` bash
python3 DKLGP.py <seed> <n_index>
# e.g.
python3 DKLGP.py 1 3
```

``` bash
# e.g. for multiple runs
for n_index in 1 2 3; do
  for seed in 1 2; do
    python3 DKLGP.py ${seed} ${n_index}
    python3 DKLGP_default.py ${seed} ${n_index}
  done
done
```

``` bash
conda deactivate
```

- **Output:**
  - Creates `DKLGP_results/` with:
    - $`\theta`$ point estimates
    - CRPS, interval scores, coverage for $`w`$
    - Running time (e.g., `KL_vec`)
    - `output_data` (DKLGP approximated mean/variance)

> **Memory note:** For `n_index = 4, 5`, DKLGP requires large memory.

## VNNGP Simulations

**Script:** `VNNGP.py`

- **Environment:**

``` bash
conda activate env_vnngp
```

- **Inputs:**
  - `n_index`
  - `seed`
- **Usage (example):**

``` bash
python3 VNNGP.py <seed> <n_index>
# e.g.
python3 VNNGP.py 1 3
```

``` bash
# e.g. for multiple runs
for n_index in 1 2 3; do
  for seed in 1 2; do
    python3 VNNGP.py ${seed} ${n_index}
  done
done
```

``` bash
conda deactivate
```

- **What it does:**
  - Reads data from `data_sim/`.
  - Runs VNNGP.
- **Output:**
  - Creates `VNNGP_results/` with:
    - θ point estimates
    - CRPS, interval scores, coverage for w
    - Running time (`KL_vec` or similar)
    - `output_data` (VNNGP approximated mean/variance)

> **Memory note:** `n_index = 4, 5` also require large memory for VNNGP.

## Simulation Summary and Figures

Once **all** `n_index` and seed combinations have been run for all
methods (at least run n_index = 3):

1.  **Combined simulation summary:**

    **Script:** `summary.R` This script uses `R_results/`,
    `DKLGP_results/`, and `VNNGP_results/` to create:

    ``` bash
    Rscript summary.R
    ```

2.  **This script creates:**

    - Figure 1, 2, 3
    - Figure H.1, H.2, H.3, H.4, H.5, H.6
    - Figure J.1, J.2
    - Table 6

Make sure the paths expected in `summary.R` match your folder structure.

# Real-World Data Analysis

``` bash
# Inside spVarBayes folder, change to the Real-World Data directory
cd paper/real_world
```

## Data Processing

**Script:** `process_data.R`

- **Usage:**

Note: for a quick test for the code, set `test_code = TRUE` in
process_data.R file.

``` bash
Rscript process_data.R
```

- **Output:**
  - Creates a `data/` folder with:
    - Training data and Test data

## R-based Methods on Real Data

Run the following R scripts (each reads from `data/` and saves results
accordingly):

> **Note:** **spVB-MFA-LR** requires substantial memory and may need to
> be run on a computing cluster. All other scripts could run on a local
> laptop.

``` bash
Rscript spVB_MFA.R
Rscript spVB_MFA_LR.R
Rscript spVB_NNGP_ind.R
Rscript spVB_NNGP_joint.R
Rscript spNNGP.R
```

These fit:

- `spVB-MFA`
- `spVB-MFA-LR`
- `spVB-NNGP` (independent)
- `spVB-NNGP` (joint)
- `spNNGP`

> **Memory note:** `spVB-MFA-LR` require large memory for storing the
> whole covariance matrix

## DKLGP and VNNGP on Real Data

> **Note:** **DKLGP (default)** and **VNNGP** requires substantial
> memory and may need to be run on a computing cluster.

Run DKLGP (default) in the DKLGP environment:

``` bash
conda activate env_dklgp
python3 DKLGP_default.py
conda deactivate
```

Run VNNGP in the VNNGP environment:

``` bash
conda activate env_vnngp
python3 VNNGP.py
conda deactivate
```

Both scripts read from the processed `data/` folder and produce outputs
compatible with the R-based methods.

## Real-Data Summary and Figures

1.  **Combine DKLGP and VNNGP outputs:**

    **Script:** `summary_DKLGP_VNNGP.R`

    ``` bash
    Rscript summary_DKLGP_VNNGP.R
    ```

    This stores combined results in an `output/` folder.

2.  **Create real-data figures and Table 4:**

    **Script:** `summary.R` (real-data version)

    ``` bash
    Rscript summary.R
    ```

    - Produces:
      - Figure 4,5,6,7  
      - Table 2  
    - Output is saved in the `fig/` folder.

# Supplementary Experiments

``` bash
# Inside spVarBayes folder, change to the correct directory in the supplementary folder
cd paper/supplement/G_choices_nn
```

## Choice of Number of Nearest Neighbors (Figures G.1–G.2)

Folder: `G_choices_nn/`  
Script: `n_neighbor.R`

- **Usage:**

``` bash
Rscript n_neighbor.R
```

- **Output:**
  - Figure G.1, G.2

These illustrate different choices for the number of nearest neighbors
for the NNGP variational family.

## Prediction Study (Figures K.1–K.2)

``` bash
# Inside spVarBayes folder, change to the correct directory in the supplementary folder
cd paper/supplement/K_prediction
```

Folder: `K_prediction/`

1.  **Data generation:**

    Script: `data_generation_prediction.R` (or the exact name in your
    folder)

    ``` bash
    for n_index in 1 2; do
       for t in 1 2; do
         Rscript data_generation_prediction.R n_index=${n_index} t=${t}
       done
    done
    ```

    Saves prediction-study data in `data_sim/` under `K_predictions/`.

2.  **R-based methods:**

    Script: `spVB_spNNGP_pred.R`

    ``` bash
    for n_index in 1 2; do
       for t in 1 2; do
         Rscript spVB_spNNGP_pred.R n_index=${n_index} t=${t}
       done
    done
    ```

    Runs all R-based methods for the prediction study.

3.  **VNNGP and DKLGP:**

    ``` bash
    # DKLGP (default version and converged version)
    conda activate env_dklgp
    for n_index in 1 2; do
       for seed in 1 2; do
         python3 DKLGP.py ${seed} ${n_index}
         python3 DKLGP_default.py ${seed} ${n_index}
       done
    done
    conda deactivate
    ```

    ``` bash
    # VNNGP
    conda activate env_vnngp
    for n_index in 1 2; do
       for seed in 1 2; do
         python3 VNNGP.py ${seed} ${n_index}
       done
    done
    conda deactivate
    ```

4.  **Summary and prediction figures:**

    Script: `summary.R` (prediction version)

    ``` bash
    Rscript summary.R
    ```

    - Produces:
      - Figure K.1, K.2
