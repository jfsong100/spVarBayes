# ===== Parameters =====
import sys
arguments = sys.argv[1:]
print("arguments",arguments)
print("check point 1")
seed = int(arguments[0])
print("seed",seed)
print("check point 2")
n_index = int(arguments[1])
print("n_index",n_index)

# ===== Import the library after installing the package =====
import torch
import gpytorch
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import yaml
import copy
import sys
import h5py
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from torch.distributions import Normal, Independent
from gps.svigp import get_SVI
from gps.vnn import get_VNN
from gps.exact_gp import get_exact_gp
sys.path.append('../')
from viva import VIVACpp as VIVA, FICCpp as FIC, DiagCpp as Diag, \
    my_train_cpp as my_train
from viva.utils import LogitLikelihood
from viva.utils import KL_GP

# ===== Set scenario here =====
# Options: 'noX', 'withX', 'real'
scenario = 'withX'  # <-- Change this for different runs
global_path = os.getcwd()
save_files = True  # <-- Change this for saving output files
converged = False

# ===== Read data =====
n_train = [1000,5000,10000]
n_test  = [100,500,1000]
n_vec = np.array(n_test) + np.array(n_train)
n = n_vec[n_index - 1]
h5_file_path = global_path + f"/data_sim/n_{n}_seed_{seed}_data.h5"

with h5py.File(h5_file_path, "r") as f:
    y_train      = f["y_train"][:]
    x_train      = f["X_train"][:].T
    w_train      = f["f_train"][:]
    S_train      = f["S_train"][:].T
    y_test       = f["y_test"][:]
    x_test       = f["X_test"][:].T
    w_test       = f["f_test"][:]
    S_test       = f["S_test"][:].T

# ===== Preprocessing =====
def maybe_center_and_regress(y, X, centered=True):
    """
    If `centered` is True, regress y on X and return residuals and beta coefficients.
    Otherwise, return original y and None.

    Parameters:
    - y: (n,) array-like response
    - X: (n, p) array-like covariates
    - centered: bool, whether to regress out X

    Returns:
    - y_new: residuals if centered, else y
    - beta: fitted coefficients if centered, else None
    """
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)

    if centered:
        model = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = model.predict(X)
        y_resid = y - y_pred
        return y_resid.ravel(), model.coef_.ravel()
    else:
        return y.ravel(), None

y_new, beta_hat = maybe_center_and_regress(y_train, x_train, centered=True)


# ===== Preprocessing =====
X_train = torch.from_numpy(S_train).type(torch.float)
y_train = torch.from_numpy(y_new).type(torch.float).squeeze()
X_test = torch.from_numpy(S_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float).squeeze()

X = torch.cat([X_train, X_test], dim=0)
y = torch.cat([y_train, y_test])

f_train = torch.from_numpy(w_train).type(torch.float).squeeze()
f_test = torch.from_numpy(w_test).type(torch.float).squeeze()
f = torch.cat([f_train, f_test])

n_train = y_train.shape[0]
n_test = y_test.shape[0]
d = X_train.shape[1]


# ===== Read parameters =====
with open(f"setups.yaml", 'r') as config_file:
    tuning_parms = yaml.safe_load(config_file)
rho = 1.5
lk_name = "normal"
kernel_name = "MaternKernel"
method = "VIVA"
data_name = f"{kernel_name}_{lk_name}_d{d}"

if converged:
    # Set n_Epoch to [500] for all cases (or select specific ones)
    tuning_parms['opt_VIVA'].pop('n_Epoch', None)
    if n_index == 1:
        tuning_parms['opt_VIVA']['n_Epoch'] = [750]
    elif n_index == 2:
        tuning_parms['opt_VIVA']['n_Epoch'] = [500]
    elif n_index == 3:
        tuning_parms['opt_VIVA']['n_Epoch'] = [500]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ===== Set learning rate based on sample size =====
if n_index == 1:
    tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.1}
elif n_index == 2:
    tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.05}
elif n_index == 3:
    tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.025}
else:
    raise ValueError(f"Unknown scenario: {scenario}")



# ===== Define DKLGP function =====

def work(X, f, y, n_test, rho, lk_name, method, **kwargs):
    n = len(y)
    d = X.size(1)
    n_train = n - n_test
    kwargs_cp = copy.deepcopy(kwargs)
    scenario = kwargs_cp.pop("scenario")
    scale_f_init = kwargs_cp[scenario].pop(
        'scale_f_init', kwargs_cp.pop('scale_f_init', None))
    kernel_name = kwargs_cp['kernel_name']
    kernel_parms = kwargs_cp['kernel_parms']
    kernel_vars_init = kwargs_cp[scenario]['kernel_vars_init']
    kernel_parms.update({'ard_num_dims': d})
    plot_mu_post = kwargs_cp.pop('plot_mu_post', False)
    seed_torch = kwargs_cp.pop('seed', 0)
    torch.manual_seed(seed_torch)
    K = gpytorch.kernels.ScaleKernel(
        getattr(gpytorch.kernels, kernel_name)(**kernel_parms))

    for var_name in kernel_vars_init.keys():
        setattr(K.base_kernel, var_name, kernel_vars_init[var_name])
    K.outputscale = scale_f_init

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = kwargs_cp[scenario].pop(
        'noise_y_init', kwargs_cp.pop('noise_y_init', None))
    classify = False

    with torch.no_grad():
        model = globals()[method](X, y, K, likelihood, rho,
                                  n_test=n_test,
                                  classify=classify, use_ic0=True)

    loss_NLL_func = torch.nn.GaussianNLLLoss()
    mu_post = torch.zeros(y[:n_train].shape)
    mu_post[model.order[:model.n_train]] = model.mu_post
    V_dense = model.V.to_dense()
    covM_maxmin_order = torch.linalg.inv(V_dense @ V_dense.t())
    var_post = torch.zeros(y[:n_train].shape)
    var_post[model.order[:model.n_train]] = covM_maxmin_order.diag()

    epoch_nums = kwargs_cp[scenario].pop(
        'n_Epoch', kwargs_cp.pop('n_Epoch', None))
    for i in range(len(epoch_nums)):
        with torch.no_grad():
            model = globals()[method](X, y, K, likelihood, rho,
                                      n_test=n_test,
                                      classify=classify, use_ic0=True)
            m = int(model.prior_sparsity_train.size(1) / n_train)
            print(f"n = {n}, d = {d}, rho = {rho}: m = {m}", flush=True)
        ELBOLst, K_parmLst = my_train(model, n_Epoch=epoch_nums[i], **kwargs_cp)
        print(ELBOLst)
        model.ELBOLst = ELBOLst
        return model

# ===== Train DKLGP model =====
torch.manual_seed(0)

time0 = time.perf_counter()
model = work(
    X, f, y, n_test, rho, lk_name, method,
    scenario=scenario, data_name=data_name,
    plot_mu_post=False,
    **(tuning_parms['opt_VIVA']))
time1 = time.perf_counter()

sigmasq = model.K.outputscale.item()
tausq = model.likelihood.noise.item()

print(f"{method} used {time1 - time0} "
      f"seconds at rho = {rho}", flush=True)

# ===== Posterior computation =====
mu_post = torch.zeros(y[:n_train].shape)
mu_post[model.order[:model.n_train]] = model.mu_post
V_dense = model.V.to_dense()
covM_maxmin_order = torch.linalg.inv(V_dense @ V_dense.t())
var_post = torch.zeros(y[:n_train].shape)
var_post[model.order[:model.n_train]] = covM_maxmin_order.diag()


# ===== Get Number of Epochs from Config =====
n_epochs = tuning_parms['opt_VIVA'][scenario].get('n_Epoch', tuning_parms['opt_VIVA'].get('n_Epoch', [None]))[0]


# ===== Report Model Runtime and Hyperparameters =====
execute_time = time1 - time0
print("Execution Time:", execute_time)

sigmasq = model.K.outputscale.item()
tausq = model.likelihood.noise.item()
lengthscale = model.K.base_kernel.lengthscale.detach().numpy()

print("Estimated Signal Variance (sigmasq):", sigmasq)
print("Estimated Noise Variance (tausq):", tausq)
print("Estimated lengthscale (1/phi):", lengthscale)


# ===== Prediction =====
w_mean_pred, w_var_pred = model.predict()
y_mean_pred = w_mean_pred.detach()+ x_test @ beta_hat
y_var_pred = (w_var_pred + model.likelihood.noise).detach()


# ===== Plot Predicted Latent Function (w) vs True w =====
plt.figure()
plt.scatter(w_test, w_mean_pred.detach().numpy())
plt.xlabel('True w')
plt.ylabel('Predicted w (w_mean_pred)')
plt.title('Scatter Plot of True vs Predicted w (DKLGP)')

# Add a 45-degree reference line
min_val = min(w_mean_pred.detach().numpy().min(), w_test.min())
max_val = max(w_mean_pred.detach().numpy().max(), w_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')


if save_files:

    if converged:
        method_label = "VIVA"
    else:
        method_label = "VIVA_default"

    save_dir = os.path.join(global_path, "DKLGP_results")
    os.makedirs(save_dir, exist_ok=True)

    excute_time = time1 - time0
    print("time",excute_time)

    output_data = {'t': np.repeat(seed, n_train),
                     'n': np.repeat(n_vec[n_index-1], n_train),
                       'index': range(1,(n_train+1)),
                       'mu_post':mu_post.detach().numpy(),
                       'var_post':var_post.detach().numpy()}

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(global_path + "/DKLGP_results/" + f"output_data_{method_label}_n{n_vec[n_index-1]}_d{d}_seed{seed}.csv",
                  index=False,
                  index_label=False)

    KL_vec = [seed, n_vec[n_index-1], excute_time,sigmasq,tausq]
    KL_vec_df = pd.DataFrame([KL_vec], columns=['seed', 'n','time','sigmasq','tausq'])
    KL_vec_df.to_csv(global_path + "/DKLGP_results/" + f"KL_vec_{method_label}_n{n_vec[n_index-1]}_d{d}_seed{seed}.csv",
                    index=False,
                    index_label=False)

    # Ensure predictions are flattened
    w_pred = w_mean_pred.detach().numpy().squeeze()
    w_var = w_var_pred.detach().numpy().squeeze()
    y_pred = y_mean_pred.detach().numpy().squeeze()
    y_var = y_var_pred.detach().numpy().squeeze()

    # Build dictionary
    pred_data = {
        't': np.repeat(seed, n_test),
        'n': np.repeat(n, n_test),
        'index': list(range(1, n_test + 1)),
        'w_pred': w_pred,
        'w_var': w_var,
        'y_pred': y_pred,
        'y_var': y_var
    }


    pred_data_df = pd.DataFrame(pred_data)
    pred_data_df.to_csv(global_path + "/DKLGP_results/" + f"output_pred_{method_label}_n{n_train}_d{d}_seed{seed}.csv",
                  index=False,
                  index_label=False)


    # Example: Generate w_samples and w from your torch model
    torch.manual_seed(seed)

    # Sample w_test
    mvn_test_w = Independent(Normal(loc=w_mean_pred, scale=torch.sqrt(w_var_pred)), 1)
    w_test_samples = mvn_test_w.rsample((5000,)).T.detach().numpy()  # shape: [n_test, 5000]

    # Sample y_test
    mvn_test_y = Independent(Normal(loc=y_mean_pred, scale=torch.sqrt(y_var_pred)), 1)
    y_test_samples = mvn_test_y.rsample((5000,)).T.detach().numpy()  # shape: [n_test, 5000]

    def eval_posterior_samples(w_samples, w_true, alpha=0.05):
        """
        Evaluate posterior samples against true values using:
        - 95% coverage
        - Interval score (scoringutils::interval_score style, weigh = TRUE)
        - CRPS (sample-based formula)

        Parameters
        ----------
        w_samples : array-like, shape (n, m)
            Posterior samples for n observations, m samples each.
        w_true : array-like, shape (n,) or (n, 1)
            True values for the n observations.
        alpha : float, optional
            Miscoverage level for the central interval (default 0.05 → 95% interval).

        Returns
        -------
        result : dict
            {
            "coverage": float,
            "IS_mean": float,
            "CRPS_mean": float
            }
        """
        w_samples = np.asarray(w_samples)
        w_true = np.asarray(w_true).reshape(-1)  # ensure shape (n,)

        n, m = w_samples.shape

        # 1. Quantiles per observation
        lower = np.quantile(w_samples, alpha / 2.0, axis=1)   # shape (n,)
        upper = np.quantile(w_samples, 1.0 - alpha / 2.0, axis=1)

        # 2. Coverage
        inside = (w_true >= lower) & (w_true <= upper)
        coverage = inside.mean()

        # 3. Interval score (unweighted)
        diff_lower = (w_true < lower)
        diff_upper = (w_true > upper)

        IS_unweighted = (upper - lower) \
            + (2.0 / alpha) * (lower - w_true) * diff_lower \
            + (2.0 / alpha) * (w_true - upper) * diff_upper

        # Weighting as in scoringutils::interval_score(weigh = TRUE)
        IS_weighted = IS_unweighted * (alpha / 2.0)
        IS_score_mean = IS_weighted.mean()

        # 4. CRPS (sample-based)
        samples_sorted = np.sort(w_samples, axis=1)  # shape (n, m)

        # term1: mean |x_k - y|
        term1 = np.mean(np.abs(samples_sorted - w_true[:, None]), axis=1)  # shape (n,)

        # term2: double-sum part
        idx = np.arange(m)  # 0..m-1
        coef = (2 * idx - m + 1) / (m * m)  # shape (m,)
        term2 = np.sum(samples_sorted * coef[None, :], axis=1)  # shape (n,)

        crps = term1 - term2
        crps_score_mean = crps.mean()

        return {
            "coverage": float(coverage),
            "IS_mean": float(IS_score_mean),
            "CRPS_mean": float(crps_score_mean)
        }

    w_pred_results = eval_posterior_samples(w_test_samples, w_test, alpha=0.05)
    y_pred_results = eval_posterior_samples(y_test_samples, y_test, alpha=0.05)
    coverage_w = w_pred_results["coverage"]
    IS_score_w = w_pred_results["IS_mean"]
    crps_score_w = w_pred_results["CRPS_mean"]

    coverage_y = y_pred_results["coverage"]
    IS_score_y = y_pred_results["IS_mean"]
    crps_score_y = y_pred_results["CRPS_mean"]

    print("Test results:")
    print(f"w_test - Coverage: {coverage_w:.3f}, IS: {IS_score_w:.3f}, CRPS: {crps_score_w:.3f}")
    print(f"y_test - Coverage: {coverage_y:.3f}, IS: {IS_score_y:.3f}, CRPS: {crps_score_y:.3f}")


    # Build vector of metrics
    pred_output_vec = [
        seed,
        n_test,
        coverage_w,
        IS_score_w,
        crps_score_w,
        coverage_y,
        IS_score_y,
        crps_score_y
    ]

    pred_output_vec_df = pd.DataFrame([pred_output_vec], columns=["seed", "n",
            "w_test_coverage", "w_test_is_mean", "w_test_crps_mean",
            "y_test_coverage", "y_test_is_mean", "y_test_crps_mean"])

    pred_output_vec_df.to_csv(global_path + "/DKLGP_results/" + f"pred_{method_label}_n{n_train}_d{d}_seed{seed}.csv",
                    index=False,
                    index_label=False)
