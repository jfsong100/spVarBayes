# spVarBayes

This R package is based on C/C++ and implements fast variational inference methods for spatial data using Nearest Neighbor Gaussian Processes (NNGP).

## Installation

To install this package in R:

```r
devtools::install_github("jfsong100/spVarBayes")
```

It involves the following functions.

## `spVB_NNGP()`

Fits a structured variational approximation using an NNGP variational distribution for the spatial random effects. This method captures dependencies in the spatial random effects by modeling it as

\[
w \sim \mathcal{N}(\mu, (I - A)^{-1} D (I - A)^{-T})
\]  

where $A$ and $D$ define the NNGP structure. It supports closed-form gradient updates and provides improved uncertainty quantification over mean field. Regression coefficients are modeled independently from the spatial effects in the variational family.

### `spVB_NNGP(joint = TRUE)`

Extends the NNGP variational approach to model **regression coefficients and spatial effects jointly**, using a **joint NNGP variational distribution**. This structure captures posterior correlation between fixed effects and spatial random effects, providing more accurate inference for regression coefficients. It remains computationally scalable using the sparse NNGP framework.



## `spVB_MFA()`

This function performs Mean Field Approximation for spatial data using a Nearest Neighbor Gaussian Process (NNGP) prior on the spatial random effects. The variational distribution assumes independence among parameters, which allows for fast inference but tends to underestimate posterior variance. The method outputs variational means and variances for spatial effects and parameters for the variational distribution for spatial parameters including spatial variance $\sigma^2$, random error $\tau^2$ and spatial decay parameter $\phi$.


## `spVB_MFA(LR=TRUE)`

An enhanced version of spVB_MFA() that applies Linear Response (LR) to improve posterior covariance estimates. This correction compensates for the variance underestimation in standard mean field approximations. The procedure starts with MFA, then uses BRISC estimates for spatial parameters and applies LR to update the covariance structure. It returns an improved covariance matrix for both spatial effects and regression coefficients. However, LR can be unstable if points are too close, and may produce non-positive diagonals in extreme cases.


### `spVB_get_Vw()`

Reconstructs the covariance matrix of the spatial random effects (`w`) based on the variational parameters. For `spVB_MFA`, this returns a diagonal matrix (unless LR = TRUE, in which case covariance is already available as `updated_mat`). For `spVB_NNGP`, it reconstructs the full NNGP covariance matrix using `A` and `D`. For the joint NNGP model, it returns the **joint covariance matrix** for regression coefficients and spatial effects.


### `spVB_beta_sampling()`

Generates **posterior samples of the regression coefficients** from the variational distribution. Available for `spVB_MFA` and `spVB_NNGP` models. For joint NNGP or MFA-LR, use the corresponding joint sampling functions instead.

### `spVB_w_sampling()`

Draws **posterior samples of the spatial random effects (`w`)** using the fitted variational approximation. Applicable to `spVB_MFA` and `spVB_NNGP`.

### `spVB_theta_sampling()`

Samples the spatial covariance parameters — spatial variance $\sigma^2$, random error $\tau^2$ and spatial decay parameter $\phi$ — from their fitted variational distributions. Can be used with any of the core methods.

### `spVB_LR_sampling()`

Provides posterior samples using the **linear response corrected covariance structure** from the `spVB_MFA(LR = TRUE)` model. Returns either:
- Samples of spatial random effects only (if no covariates), or  
- Joint samples of regression coefficients and spatial effects (if covariates are included).


### `spVB_joint_sampling()`

Draws posterior samples from the **joint variational distribution** of regression coefficients and spatial effects fitted by `spVB_NNGP(joint = TRUE)`. 


## Tutorial

You can view the full tutorial here:

[View HTML Tutorial](https://jfsong100.github.io/spVarBayes/spVarBayes-tutorial.html)
