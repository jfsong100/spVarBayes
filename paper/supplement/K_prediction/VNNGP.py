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
import scipy.io as sio
import os
import sys
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


# ===== Report Model Runtime and Hyperparameters =====
execute_time = time2 - time1
print("Execution Time:", execute_time)

sigmasq = model.covar_module.outputscale.item()
tausq = model.likelihood.noise.item()
lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()

print("Estimated Signal Variance (sigmasq):", sigmasq)
print("Estimated Noise Variance (tausq):", tausq)
print("Estimated lengthscale (1/phi):", lengthscale)



# ===== Prediction =====
from torch.utils.data import TensorDataset, DataLoader

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)

model.eval()
likelihood.eval()
means = torch.tensor([0.])
test_mse = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
        diff = torch.pow(preds.mean - y_batch, 2)
        diff = diff.sum(dim=-1) / X_test.size(0) # sum over bsz and scaling
        diff = diff.mean() # average over likelihood_nsamples
        test_mse += diff
means = means[1:]
test_rmse = test_mse.sqrt().item()

w_mean_pred = preds.mean
w_var_pred = preds.variance
y_mean_pred = w_mean_pred.detach() + x_test @ beta_hat
y_var_pred = (w_var_pred + likelihood.noise).detach()


# ===== Plot Predicted Latent Function (w) vs True w =====
plt.figure()
plt.scatter(w_test, w_mean_pred.detach().numpy())
plt.xlabel('True w')
plt.ylabel('Predicted w (w_mean_pred)')
plt.title('Scatter Plot of True vs Predicted w (VNNGP)')

# Add a 45-degree reference line
min_val = min(w_mean_pred.detach().numpy().min(), w_test.min())
max_val = max(w_mean_pred.detach().numpy().max(), w_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')


# ===== Plot Predicted Latent Function (y) vs True y =====
plt.figure()
plt.scatter(y_test, y_mean_pred.detach().numpy())
plt.xlabel('True y')
plt.ylabel('Predicted y (y_mean_pred)')
plt.title('Scatter Plot of True vs Predicted y (VNNGP)')

# Add a 45-degree reference line
min_val = min(y_mean_pred.detach().numpy().min(), y_test.min())
max_val = max(y_mean_pred.detach().numpy().max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')



if save_files:

    save_dir = os.path.join(global_path, "VNNGP_results")
    os.makedirs(save_dir, exist_ok=True)

    mu_w_update = mu_post.detach().numpy()
    sigmasq_w_update = var_post.detach().numpy()
    
    output_data = {
    't': np.repeat(seed, n),
    'n': np.repeat(n_vec[n_index-1], n),
    'index': list(range(1, n + 1)),
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
    
    
    sigmasq = model.covar_module.outputscale.item()
    tausq = model.likelihood.noise.item()

    output_vec = [seed, n_vec[n_index-1], execute_time, sigmasq, tausq]

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
        header = ["seed", "n", "time", "sigmasq", "tausq"]
        f.write("\t".join(header) + "\n")
        f.write("\t".join(map(str, cleaned)) + "\n")
    
    # Ensure predictions are flattened
    w_pred = w_mean_pred.detach().numpy().squeeze()
    w_var = w_var_pred.detach().numpy().squeeze()
    y_pred = y_mean_pred.detach().numpy().squeeze()
    y_var = y_var_pred.detach().numpy().squeeze()

    # Build dictionary
    output_data = {
        't': np.repeat(seed, n_test),
        'n': np.repeat(n, n_test),
        'index': list(range(1, n_test + 1)),
        'w_pred': w_pred,
        'w_var': w_var,
        'y_pred': y_pred,
        'y_var': y_var
    }

    # Check lengths (optional)
    lengths = [len(col) for col in output_data.values()]
    assert all(l == n_test for l in lengths), f"Inconsistent column lengths: {lengths}"

    # Write to TXT file manually
    header = list(output_data.keys())
    rows = zip(*output_data.values())

    output_path = f"{global_path}/VNNGP_results/output_pred_VNNGP_n{n_train}_d{d}_seed{seed}.txt"

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")  # tab-separated
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

        
    # Example: Generate w_samples and w from your torch model
    np.random.seed(seed) 

    # Sample w_test
    w_test_samples = np.random.normal(loc=w_pred, scale=np.sqrt(w_var), size=(5000, w_pred.shape[0])).T

    # Sample y_test
    y_test_samples = np.random.normal(loc=y_pred, scale=np.sqrt(y_var), size=(5000, y_pred.shape[0])).T
    
    
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

    # Flatten numpy scalars if needed
    def flatten(x):
        if isinstance(x, np.ndarray):
            return x.item() if x.size == 1 else ",".join(map(str, x.flatten()))
        return x

    cleaned_pred = [flatten(x) for x in pred_output_vec]

    # Save to TXT
    pred_output_path = f"{global_path}/VNNGP_results/pred_VNNGP_n{n_train}_d{d}_seed{seed}.txt"

    with open(pred_output_path, "w") as f:
        header = [
            "seed", "n", 
            "w_test_coverage", "w_test_is_mean", "w_test_crps_mean",
            "y_test_coverage", "y_test_is_mean", "y_test_crps_mean"
        ]
        f.write("\t".join(header) + "\n")
        f.write("\t".join(map(str, cleaned_pred)) + "\n")




