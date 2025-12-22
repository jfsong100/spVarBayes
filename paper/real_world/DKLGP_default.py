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
import h5py
import sys
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

print("Begin")

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
#X_test = torch.from_numpy(S_test).type(torch.float)
#y_test = torch.from_numpy(y_test).type(torch.float).squeeze()

y_train = torch.from_numpy(y_train).type(torch.float).squeeze()
X_train = torch.from_numpy(np.hstack((S_train, x_reg_train))).type(torch.float)

#X_test = torch.from_numpy(S_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float).squeeze()
X_test = torch.from_numpy(np.hstack((S_test, x_reg_test))).type(torch.float)

X = torch.cat([X_train, X_test], dim=0)
y = torch.cat([y_train, y_test])

f = None 

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
    
tuning_parms['opt_VIVA']['optimizer_args'] = {'lr': 0.01}
tuning_parms['opt_VIVA'][scenario]['df_student_init'] = d
tuning_parms['opt_VIVA'][scenario]['kernel_vars_init']['lengthscale'] = [0.25] * d
tuning_parms['opt_VIVA'][scenario]


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


print("Start Training")

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


# ===== Plot ELBO over Epochs =====
plt.figure(figsize=(10, 6))
plt.plot(model.ELBOLst, marker='o', linestyle='-', color='b', label='ELBO')
plt.title('ELBO over Epochs')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.legend()
plt.grid(True)
plt.show()

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
y_mean_pred = w_mean_pred.detach()
y_var_pred = (w_var_pred + model.likelihood.noise).detach()


if save_files:

    save_dir = os.path.join(global_path, "output")
    os.makedirs(save_dir, exist_ok=True)

    excute_time = time1 - time0
    print("time",excute_time)

    output_data = {'n': np.repeat(n_train, n_train),
                   'index': range(1,(n_train+1)),
                       'mu_post':mu_post.detach().numpy(),
                       'var_post':var_post.detach().numpy()}

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(global_path + "/output/" + f"output_data_{method}_default_n{n_train}_d{d}.csv",
                  index=False,
                  index_label=False)
    
    KL_vec = [n_train, excute_time,sigmasq,tausq,lengthscale[0][0],lengthscale[0][1],lengthscale[0][2]]
    KL_vec_df = pd.DataFrame([KL_vec], columns=['n','time','sigmasq','tausq',"lengthscale1","lengthscale2","lengthscale3"])
    KL_vec_df.to_csv(global_path + "/output/" + f"KL_vec_{method}_default_n{n_train}_d{d}.csv",
                    index=False,
                    index_label=False)
    
    w_pred = w_mean_pred.detach().numpy().squeeze()
    w_var = w_var_pred.detach().numpy().squeeze()
    y_pred = y_mean_pred.detach().numpy().squeeze()
    y_var = y_var_pred.detach().numpy().squeeze()

    # Build dictionary
    pred_data = {
        'n': np.repeat(n_test, n_test),
        'index': list(range(1, n_test + 1)),
        'w_pred': w_pred,
        'w_var': w_var,
        'y_pred': y_pred,
        'y_var': y_var
    }


    pred_data_df = pd.DataFrame(pred_data)
    pred_data_df.to_csv(global_path + "/output/" + f"output_pred_{method}_default_n{n_train}_d{d}.csv",
                  index=False,
                  index_label=False)
    

else:
    print("Saving skipped — `save_files` is set to False.")

