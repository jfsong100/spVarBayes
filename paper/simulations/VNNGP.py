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

import torch
import gpytorch
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import time
import h5py
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===== Set scenario here =====
# Options: 'noX', 'withX', 'real'
scenario = 'withX'  # <-- Change this for different runs
global_path = os.getcwd()
save_files = True  # <-- Change this for saving output files


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
n_train = y_new.shape[0]
n_test = 0

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

x_train = X[:n_train]
y_train = y[:n_train]
init_post_mean = y_train


# ===== VNNGP Setup =====
m = 15
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 0.01
kernel_name = 'MaternKernel'
kernel_parms = {'nu': 0.5, 'ard_num_dims': 2}

# ===== Set learning rate based on scenario =====
if n_index == 1:
    lr = 0.1
elif n_index == 2:
    lr = 0.01
elif n_index == 3:
    lr = 0.001
elif n_index == 4:
    lr = 0.0005
elif n_index == 5:
    lr = 0.0005
else:
    raise ValueError(f"Unknown scenario: {scenario}")

print(f"Learning rate set to {lr} for scenario: {scenario}")


torch.manual_seed(0)

# ===== VNNGP =====

sys.path.append('..')

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import \
    NNVariationalStrategy
import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal as MVN


class GPModel(ApproximateGP):
    def __init__(self, initial_inducing_response, inducing_points, likelihood, k=256, training_batch_size=256):
        m, d = inducing_points.shape
        self.m = m
        self.k = k
        print("m", m)
        print("k", k)

        #variational_distribution = MeanFieldVariationalDistribution(m)

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)
        
        start_dist = torch.distributions.MultivariateNormal(
            initial_inducing_response,
            torch.diag_embed(torch.ones_like(
                initial_inducing_response
            ) * .5))
        
        variational_distribution.initialize_variational_distribution(start_dist)

        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=training_batch_size
        )

        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            getattr(gpytorch.kernels, kernel_name)(**kernel_parms)
        )
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None and x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=prior, **kwargs)

    


time1 = time.perf_counter()

model = GPModel(
    initial_inducing_response=init_post_mean,
    inducing_points=x_train,
    likelihood=likelihood,
    k=m,
    training_batch_size=256)

loss_MSE_func = torch.nn.MSELoss()
loss_NLL_func = torch.nn.GaussianNLLLoss()
num_batches = model.variational_strategy._total_training_batches


n_Epoch = 500
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(n_Epoch*0.75), int(n_Epoch*0.9)], gamma=0.1)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

epochs_iter = range(n_Epoch)
for epoch in epochs_iter:
    minibatch_iter = range(num_batches)
    
    for minibatch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x=None)
        # Obtain the indices for mini-batch data
        current_training_indices = model.variational_strategy.current_training_indices
        # Obtain the y_batch using indices. It is important to keep the same order of train_x and train_y
        y_batch = y_train[...,current_training_indices]
        loss = -mll(output, y_batch)
            # minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
time2 = time.perf_counter()


# ===== Posterior computation =====
dist_post_f = model(X[:n_train])            
mu_post = dist_post_f.mean
var_post = dist_post_f.variance


if n <= 10000:
    # ===== Plot Posterior Mean vs Empirical Mean =====
    plt.figure()
    plt.scatter(empirical_mu, mu_post.detach().numpy())
    plt.xlabel('Empirical Mean')
    plt.ylabel('Posterior Mean (mu_post)')
    plt.title('Scatter Plot of Empirical vs Posterior Mean (VNNGP)')

    # Add a 45-degree reference line
    min_val = min(mu_post.detach().numpy().min(), empirical_mu.min())
    max_val = max(mu_post.detach().numpy().max(), empirical_mu.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    # ===== Plot Posterior Variance vs Empirical Variance =====
    plt.figure()
    plt.scatter(empirical_var, var_post.detach().numpy())
    plt.xlabel('Empirical Variance')
    plt.ylabel('Posterior Variance (var_post)')
    plt.title('Scatter Plot of Empirical vs Posterior Variance (VNNGP)')

    # Add a 45-degree reference line
    min_val = min(var_post.detach().numpy().min(), empirical_var.min())
    max_val = max(var_post.detach().numpy().max(), empirical_var.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# ===== Report Model Runtime and Hyperparameters =====
execute_time = time2 - time1
print("Execution Time:", execute_time)

sigmasq = model.covar_module.outputscale.item()
tausq = model.likelihood.noise.item()
lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()

print("Estimated Signal Variance (sigmasq):", sigmasq)
print("Estimated Noise Variance (tausq):", tausq)
print("Estimated lengthscale (1/phi):", lengthscale)


if save_files:
    
    save_dir = os.path.join(global_path, "VNNGP_results")
    os.makedirs(save_dir, exist_ok=True)
    mu_w_update = mu_post.detach().numpy()
    sigmasq_w_update = var_post.detach().numpy()

    if n <= 10000:
        import numpy as np
        n = mu_w_update.shape[0]
        ord_idx = np.arange(n)

        diagV = np.diag(empirical_V)
        term1 = np.sum(diagV[ord_idx] / sigmasq_w_update)
        diff  = mu_w_update - empirical_mu[ord_idx]
        term2 = np.sum((diff**2) / sigmasq_w_update)

        term3 = np.sum(np.log(sigmasq_w_update))
        V_sub = empirical_V[np.ix_(ord_idx, ord_idx)]
        sign, logdet = np.linalg.slogdet(V_sub)  # log |det(V_sub)|

        KL_result = (term1 - n + term2 + term3 - logdet) / 2.0

        print("KL_result:", KL_result)
    else:
        KL_result = None
        empirical_mu  = [None] * n
        empirical_var = [None] * n
    
    output_data = {
    't': np.repeat(seed, n),
    'n': np.repeat(n_vec[n_index-1], n),
    'index': list(range(1, n + 1)),
    'empirical_mu': empirical_mu,
    'empirical_var': empirical_var,
    'mu_post': mu_post.detach().numpy(),
    'var_post': var_post.detach().numpy()
    }

    # Optional: check consistency
    lengths = [len(col) for col in output_data.values()]

    # Build header and rows
    header = list(output_data.keys())
    rows = zip(*output_data.values())

    # Save to TXT
    output_path = f"{global_path}/VNNGP_results/output_data_VNNGP_n{n_vec[n_index-1]}_d2_seed{seed}.txt"

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")  # tab-separated header
        for row in rows:
            row_str = "\t".join(map(str, row))
            f.write(row_str + "\n")
    

    # Example: Generate w_samples and w from your torch model
    torch.manual_seed(seed)
    mvn = torch.distributions.MultivariateNormal(mu_post, dist_post_f.covariance_matrix)
    single_sample = mvn.sample((5000,)).T  # shape: [n, 5000]
    print("Shape of single sample:", single_sample.shape)
    w_samples = single_sample.detach().numpy()

    # Ensure that w (true values) is a 1D numpy array
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

    
    sigmasq = model.covar_module.outputscale.item()
    tausq = model.likelihood.noise.item()

    output_vec = [seed, n_vec[n_index-1], KL_result, execute_time, sigmasq, tausq, coverage, IS_score_mean, crps_score_mean]

    # Flatten numpy scalars or arrays
    def flatten(x):
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            else:
                return ",".join(map(str, x.flatten()))
        return x

    cleaned = [flatten(x) for x in output_vec]

    output_vec_path = f"{global_path}/VNNGP_results/KL_vec_VNNGP_n{n_vec[n_index-1]}_d2_seed{seed}.txt"

    with open(output_vec_path, "w") as f:
        header = ["seed", "n", "KL", "time", "sigmasq", "tausq", "coverage", "is_mean", "crps_mean"]
        f.write("\t".join(header) + "\n")
        f.write("\t".join(map(str, cleaned)) + "\n")
    

else:
    print("Saving skipped — `save_files` is set to False.")

