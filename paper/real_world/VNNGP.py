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


# ===== Set scenario here =====
# Options: 'noX', 'withX', 'real'
scenario = 'real'  # <-- Change this for different runs
global_path = os.getcwd()
save_files = True  # <-- Change this for saving output files

h5_file_path = global_path + "/data/BCEF_data.h5"

with h5py.File(h5_file_path, "r") as f:
    y_train      = f["y"][:]              
    x_reg_train  = f["X"][:].T  
    S_train      = f["S_ordered_scaled"][:].T
    y_test       = f["y_test"][:]              
    x_reg_test   = f["X_test"][:].T  
    S_test       = f["s_test_scaled"][:].T


# ===== Preprocessing =====
#X_train = torch.from_numpy(S_train).type(torch.float)
#y_train = torch.from_numpy(y_new).type(torch.float).squeeze()

y_train = torch.from_numpy(y_train).type(torch.float).squeeze()
X_train = torch.from_numpy(np.hstack((S_train, x_reg_train))).type(torch.float)

#X_test = torch.from_numpy(S_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float).squeeze()
X_test = torch.from_numpy(np.hstack((S_test, x_reg_test))).type(torch.float)

X = torch.cat([X_train, X_test], dim=0)
y = torch.cat([y_train, y_test])
    
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
kernel_parms = {'nu': 0.5, 'ard_num_dims': d}

# ===== Set learning rate based on scenario =====
lr = 0.01


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
        print("sub point 1")
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)
        print("sub point 2")
        start_dist = torch.distributions.MultivariateNormal(
            initial_inducing_response,
            torch.diag_embed(torch.ones_like(
                initial_inducing_response
            ) * .5))
        print("sub point 3")
        variational_distribution.initialize_variational_distribution(start_dist)
        print("sub point 4")
        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=training_batch_size
        )
        print("sub point 5")
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

    
print("Check point 1")

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
print("Check point 2")

n_Epoch = 500
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(n_Epoch*0.75), int(n_Epoch*0.9)], gamma=0.1)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

epochs_iter = range(n_Epoch)
for epoch in epochs_iter:
    print(f"Epoch {epoch + 1}/{n_Epoch}")  # Print current epoch number
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
y_mean_pred = w_mean_pred.detach() 
y_var_pred = (w_var_pred + likelihood.noise).detach()


if save_files: 
    save_dir = os.path.join(global_path, "output")
    os.makedirs(save_dir, exist_ok=True)

    n = y_train.shape[0]
    output_data = {
    'n': np.repeat(n, n),
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
    output_path = f"{global_path}/output/output_data_VNNGP_n{n}_d{d}.txt"

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")  # tab-separated header
        for row in rows:
            row_str = "\t".join(map(str, row))
            f.write(row_str + "\n")
    
     
    sigmasq = model.covar_module.outputscale.item()
    tausq = model.likelihood.noise.item()

    output_vec = [n, execute_time, sigmasq, tausq, lengthscale[0][0],lengthscale[0][1],lengthscale[0][2]]

    # Flatten numpy scalars or arrays
    def flatten(x):
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            else:
                return ",".join(map(str, x.flatten()))
        return x

    cleaned = [flatten(x) for x in output_vec]

    output_vec_path = f"{global_path}/output/KL_vec_VNNGP_n{n}_d{d}.txt"

    with open(output_vec_path, "w") as f:
        header = ["n", "time", "sigmasq", "tausq", "lengthscale1", "lengthscale2", "lengthscale3"]
        f.write("\t".join(header) + "\n")
        f.write("\t".join(map(str, cleaned)) + "\n")
    
    n_test = y_test.shape[0]
    # Ensure predictions are flattened
    w_pred = w_mean_pred.detach().numpy().squeeze()
    w_var = w_var_pred.detach().numpy().squeeze()
    y_pred = y_mean_pred.detach().numpy().squeeze()
    y_var = y_var_pred.detach().numpy().squeeze()

    # Build dictionary
    output_data = {
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

    output_path = f"{global_path}/output/output_pred_VNNGP_n{n}_d{d}.txt"

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")  # tab-separated
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

