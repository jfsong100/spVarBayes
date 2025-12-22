# Parameters
import sys
arguments = sys.argv[1:]
print("arguments",arguments)
print("check point 1")
seed = int(arguments[0])
print("seed",seed)
print("check point 2")
n_index = int(arguments[1])
print("n_index",n_index)

# Import the library after installing the package
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
from sklearn.linear_model import LinearRegression
import scipy.io as sio
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
converged = True   # <-- Change this for DKLGP default or converged

# ===== Read data =====
n_vec = [1000,5000,10000,50000,100000]
n = n_vec[n_index - 1]
h5_file_path = global_path + f"/data_sim/n_{n}_seed_{seed}_data.h5"

with h5py.File(h5_file_path, "r") as f:
    y_gen        = f["y_gen"][:]              # numpy array
    x            = f["X"][:].T                # transpose if you want
    w            = f["f"][:]
    S_ordered    = f["S_ordered"][:].T
    if n <= 10000:
        empirical_mu = f["empirical_mu"][:]
        empirical_var= f["empirical_var"][:]
        empirical_V  = f["empirical_V"][:]


# ===== Preprocessing =====
def maybe_center_and_regress(y, X, centered=True):
    """
    If `centered` is True, regress y on X and return residuals.
    Otherwise, return original y.

    Parameters:
    - y: (n,) array-like response
    - X: (n, p) array-like covariates
    - centered: bool, whether to regress out X

    Returns:
    - y_new: residuals if centered, else y
    """
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)

    if centered:
        model = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = model.predict(X)
        y_resid = y - y_pred
        return y_resid.ravel()
    else:
        return y.ravel()

y_new = maybe_center_and_regress(y_gen, x, centered=True)  # returns residuals

# ===== Preprocessing =====
X_train = torch.from_numpy(S_ordered).type(torch.float)
y_train = torch.from_numpy(y_new).type(torch.float).squeeze()
f_train = torch.from_numpy(w).type(torch.float).squeeze()

X = X_train
y = y_train
f = f_train

#X_test = torch.from_numpy(S_test).type(torch.float)
#y_test = torch.from_numpy(y_test).type(torch.float).squeeze()

#X = torch.cat([X_train, X_test], dim=0)
#y = torch.cat([y_train, y_test])

n_train = y_train.shape[0]
n_test = 0
d = X.shape[1]


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
    elif n_index == 4:
        tuning_parms['opt_VIVA']['n_Epoch'] = [500]
    elif n_index == 5:
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
elif n_index == 4:
    tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.01}
elif n_index == 5:
    tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.01}
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
    # NLL_ic0 = loss_NLL_func(mu_post, f[:n_train], var_post).detach()

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
torch.set_default_dtype(torch.float64)
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


# ===== Plot ELBO over Epochs =====
plt.figure(figsize=(10, 6))
plt.plot(model.ELBOLst, marker='o', linestyle='-', color='b', label='ELBO')
plt.title('ELBO over Epochs')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.legend()
plt.grid(True)


def check_elbo_convergence(elbo_list, window_size=10, tol=1, min_iter=35):
    """
    Check if the average ELBO in a moving window has stabilized.

    Parameters:
    - elbo_list: list of ELBO values
    - window_size: number of recent values to average
    - tol: threshold for the difference in window averages
    - min_iter: only start checking after this many values

    Returns:
    - converged: bool
    """
    if len(elbo_list) < min_iter + 2 * window_size:
        return False

    recent_avg = sum(elbo_list[-window_size:]) / window_size
    previous_avg = sum(elbo_list[-2*window_size:-window_size]) / window_size
    diff = abs(recent_avg - previous_avg)

    return diff < tol
check_elbo_convergence(model.ELBOLst)


# ===== Posterior computation =====
mu_post = torch.zeros(y[:n_train].shape)
mu_post[model.order[:model.n_train]] = model.mu_post
V_dense = model.V.to_dense()
covM_maxmin_order = torch.linalg.inv(V_dense @ V_dense.t())
var_post = torch.zeros(y[:n_train].shape)
var_post[model.order[:model.n_train]] = covM_maxmin_order.diag()

# ===== Get Number of Epochs from Config =====
n_epochs = tuning_parms['opt_VIVA'][scenario].get('n_Epoch', tuning_parms['opt_VIVA'].get('n_Epoch', [None]))[0]

if n <= 10000:
    # ===== Plot Posterior Mean vs Empirical Mean =====
    plt.figure()
    plt.scatter(empirical_mu, mu_post.detach().numpy())
    plt.xlabel('Empirical Mean')
    plt.ylabel('Posterior Mean (mu_post)')
    plt.title(f'Scatter Plot of Empirical vs Posterior Mean (DKLGP, {n_epochs} Epochs)')

    # Add a 45-degree reference line
    min_val = min(mu_post.detach().numpy().min(), empirical_mu.min())
    max_val = max(mu_post.detach().numpy().max(), empirical_mu.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    # ===== Plot Posterior Variance vs Empirical Variance =====
    plt.figure()
    plt.scatter(empirical_var, var_post.detach().numpy())
    plt.xlabel('Empirical Variance')
    plt.ylabel('Posterior Variance (var_post)')
    plt.title(f'Scatter Plot of Empirical vs Posterior Variance (DKLGP, {n_epochs} Epochs)')

    # Add a 45-degree reference line
    min_val = min(var_post.detach().numpy().min(), empirical_var.min())
    max_val = max(var_post.detach().numpy().max(), empirical_var.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# ===== Report Model Runtime and Hyperparameters =====
execute_time = time1 - time0
print("Execution Time:", execute_time)

sigmasq = model.K.outputscale.item()
tausq = model.likelihood.noise.item()
lengthscale = model.K.base_kernel.lengthscale.detach().numpy()

print("Estimated Signal Variance (sigmasq):", sigmasq)
print("Estimated Noise Variance (tausq):", tausq)
print("Estimated lengthscale (1/phi):", lengthscale)

if save_files:

    if converged:
        method_label = "VIVA"
    else:
        method_label = "VIVA_default"

    save_dir = os.path.join(global_path, "DKLGP_results")
    os.makedirs(save_dir, exist_ok=True)

    if n <= 10000:
        KL2 = KL_GP(torch.from_numpy(empirical_mu[model.order].flatten()). \
            type(torch.float64),
                model.mu_post,
                torch.from_numpy(empirical_V[model.order][:, model.order]). \
            type(torch.float64),
                covM_maxmin_order)

        print("KL2",KL2)
    else:
        KL2 = None
        empirical_mu  = [None] * n
        empirical_var = [None] * n

    excute_time = time1 - time0
    print("time",excute_time)

    output_data = {'t': np.repeat(seed, n_train),
                       'n': np.repeat(n_vec[n_index-1], n_train),
                       'index': range(1,(n_train+1)),
                       'empirical_mu': empirical_mu,
                       'empirical_var': empirical_var,
                       'mu_post':mu_post.detach().numpy(),
                       'var_post':var_post.detach().numpy()}

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(global_path + "/DKLGP_results/" + f"output_data_{method_label}_n{n_vec[n_index-1]}_d{d}_seed{seed}.csv",
                  index=False,
                  index_label=False)
    torch.manual_seed(seed)
    # Create the Multivariate Normal distribution
    mvn = torch.distributions.MultivariateNormal(model.mu_post, covM_maxmin_order)

    # Generate a sample from the distribution
    single_sample = mvn.sample((5000,))
    print("Shape of single sample:", single_sample.shape)
    output_sample = torch.zeros(single_sample.shape)
    output_sample[:,model.order] = single_sample

    w_samples = output_sample.T.detach().numpy()

    w = w.flatten()  # if not already 1D

    n, m = w_samples.shape
    alpha = 0.05  # 95% interval

    # 1. 2.5% and 97.5% quantiles per observation (row)
    lower = np.quantile(w_samples, 0.025, axis=1)  # shape (n,)
    upper = np.quantile(w_samples, 0.975, axis=1)  # shape (n,)

    # 2. Coverage: fraction of w[i] inside [lower[i], upper[i]]
    inside = (w >= lower) & (w <= upper)
    coverage = inside.mean()

    # 3. Interval score (same formula as scoringutils::interval_score for one interval)
    #    IS = (u - l) + (2/alpha)*(l - y)*1(y < l) + (2/alpha)*(y - u)*1(y > u)
    diff_lower = (w < lower)
    diff_upper = (w > upper)

    IS_unweighted = (upper - lower) \
        + (2.0 / alpha) * (lower - w) * diff_lower \
        + (2.0 / alpha) * (w - upper) * diff_upper

    # 4. Apply same weighting as scoringutils::interval_score (weigh = TRUE)
    IS_weighted = IS_unweighted * (alpha / 2.0)

    IS_score_mean = IS_weighted.mean()

    # 4. CRPS using sample formula (vectorized over n)
    #    CRPS = 1/m Σ|x_k - y| - 1/(m^2) Σ (2k - m - 1) x_(k), with x_(k) sorted
    samples_sorted = np.sort(w_samples, axis=1)  # shape [n, m]

    # term1: mean absolute error between samples and truth
    term1 = np.mean(np.abs(samples_sorted - w[:, None]), axis=1)  # shape (n,)

    # term2: double-sum part simplified using sorted samples
    idx = np.arange(m)  # 0..m-1
    coef = (2 * idx - m + 1) / (m * m)  # shape (m,)
    term2 = np.sum(samples_sorted * coef[None, :], axis=1)        # shape (n,)

    crps = term1 - term2
    crps_score_mean = crps.mean()

    print("coverage:", coverage)
    print("IS_score_mean:", IS_score_mean)
    print("crps_score_mean:", crps_score_mean)


    KL_vec = [seed, n_vec[n_index-1], KL2, excute_time,sigmasq,tausq,coverage, IS_score_mean, crps_score_mean]
    KL_vec_df = pd.DataFrame([KL_vec], columns=['seed', 'n', 'KL2','time','sigmasq','tausq',"coverage", "is_mean", "crps_mean"])
    KL_vec_df.to_csv(global_path + "/DKLGP_results/" + f"KL_vec_{method_label}_n{n_vec[n_index-1]}_d{d}_seed{seed}.csv",
                    index=False,
                    index_label=False)

else:
    print("Saving skipped — `save_files` is set to False.")
