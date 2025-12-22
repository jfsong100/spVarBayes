########################################
############## load package ############
########################################
library(readr)
library(scoringutils)

########################################
############## Read data ###############
########################################
base_dir  = file.path(getwd())
data.path = file.path(base_dir, "data")
output.path = file.path(base_dir, "output")
load(file.path(data.path,"BCEF_data.RData"))

if(!dir.exists(output.path)){
  dir.create(output.path)
}

########################################
############# Read VNNGP ###############
########################################

output_data_VNNGP = read.delim(file.path(base_dir, "output", paste0("output_data_VNNGP_n",n,"_d3.txt")))
output_pred_VNNGP = read.delim(file.path(base_dir, "output", paste0("output_pred_VNNGP_n",n,"_d3.txt")))
KL_vec_VNNGP      = read.delim(file.path(base_dir, "output", paste0("KL_vec_VNNGP_n",n,"_d3.txt")), header = TRUE)

VNNGP_summary = list(
  w_mean       = output_data_VNNGP$mu_post,
  w_var        = output_data_VNNGP$var_post,  
  theta_mean   = c(KL_vec_VNNGP[3:7]),
  run_time     = KL_vec_VNNGP[2]
)

save(VNNGP_summary,
     file = file.path(output.path,"VNNGP_fit.RData"))

w_mean_pred_VNNGP = output_pred_VNNGP$w_pred
w_var_pred_VNNGP = output_pred_VNNGP$w_var
y_mean_pred_VNNGP = output_pred_VNNGP$y_pred
y_var_pred_VNNGP = output_pred_VNNGP$y_var

n.samples = 5000
set.seed(123)  
samples_matrix = matrix(rnorm(n_test * n.samples,
                              mean = rep(y_mean_pred_VNNGP, times = n.samples),
                              sd = rep(sqrt(y_var_pred_VNNGP), times = n.samples)),
                        nrow = n_test, ncol = n.samples)

y_lb_pred_VNNGP = sapply(1:n_test,
                         FUN = function(i){
                           quantile(samples_matrix[i,],c(0.025))
                         })
y_ub_pred_VNNGP = sapply(1:n_test,
                         FUN = function(i){
                           quantile(samples_matrix[i,],c(0.975))
                         })

VNNGP_crps = mean(crps_sample(y_test, samples_matrix))

VNNGP_is = mean(scoringutils:::interval_score(y_test,
                                              lower = y_lb_pred_VNNGP,
                                              upper = y_ub_pred_VNNGP,
                                              rep(95,length(y_test))))

VNNGP_mse = mean(scoringutils:::se_mean_sample(y_test, as.matrix(y_mean_pred_VNNGP)))

VNNGP_coverage =mean((y_test >= y_lb_pred_VNNGP) * (y_test <= y_ub_pred_VNNGP))

save(w_mean_pred_VNNGP, w_var_pred_VNNGP, 
     y_mean_pred_VNNGP, y_var_pred_VNNGP, 
     y_lb_pred_VNNGP, y_ub_pred_VNNGP, VNNGP_crps, VNNGP_is, VNNGP_mse, VNNGP_coverage,
     file = file.path(output.path,"VNNGP_predict_w_y.RData"))

########################################
############# Read DKLGP ###############
########################################

output_data_DKLGP = read_csv(file.path(base_dir, "output", paste0("output_data_VIVA_default_n",n,"_d3.csv")))
output_pred_DKLGP = read_csv(file.path(base_dir, "output", paste0("output_pred_VIVA_default_n",n,"_d3.csv")))
KL_vec_DKLGP = read_csv(file.path(base_dir, "output", paste0("KL_vec_VIVA_default_n",n,"_d3.csv")))

DKLGP_summary = list(
  w_mean       = output_data_DKLGP$mu_post,
  w_var        = output_data_DKLGP$var_post,  
  theta_mean   = c(KL_vec_DKLGP[3:7]),
  run_time     = KL_vec_DKLGP[2]
)

save(DKLGP_summary,
     file = file.path(output.path,"DKLGP_fit.RData"))

w_mean_pred_DKLGP = output_pred_DKLGP$w_pred
w_var_pred_DKLGP = output_pred_DKLGP$w_var
y_mean_pred_DKLGP = output_pred_DKLGP$y_pred
y_var_pred_DKLGP = output_pred_DKLGP$y_var
n.samples = 5000
set.seed(123)  
samples_matrix = matrix(NA_real_, nrow = n, ncol = n.samples)
valid_idx = which(!is.na(y_mean_pred_DKLGP) & !is.na(y_var_pred_DKLGP))
set.seed(123)
for (i in valid_idx) {
  mu  <- y_mean_pred_DKLGP[i]
  sig <- sqrt(y_var_pred_DKLGP[i])
  samples_matrix[i, ] <- rnorm(n.samples, mean = mu, sd = sig)
}


y_lb_pred_DKLGP = apply(
  samples_matrix,
  MARGIN = 1,
  FUN = function(x) quantile(x, probs = 0.025, na.rm = TRUE)
)
y_ub_pred_DKLGP = apply(
  samples_matrix,
  MARGIN = 1,
  FUN = function(x) quantile(x, probs = 0.975, na.rm = TRUE)
)


DKLGP_crps = mean(crps_sample(y_test[valid_idx], samples_matrix[valid_idx,]))
DKLGP_is = mean(scoringutils:::interval_score(y_test[valid_idx],
                                              lower = y_lb_pred_DKLGP[valid_idx],
                                              upper = y_ub_pred_DKLGP[valid_idx],
                                              rep(95,length(y_test[valid_idx]))))

DKLGP_mse = mean(scoringutils:::se_mean_sample(y_test[valid_idx], as.matrix(y_mean_pred_DKLGP[valid_idx])))

DKLGP_coverage = mean((y_test[valid_idx] >= y_lb_pred_DKLGP[valid_idx]) * (y_test[valid_idx] <= y_ub_pred_DKLGP[valid_idx]))

save(w_mean_pred_DKLGP, w_var_pred_DKLGP, 
     y_mean_pred_DKLGP, y_var_pred_DKLGP, 
     y_lb_pred_DKLGP, y_ub_pred_DKLGP,DKLGP_crps, DKLGP_is, DKLGP_mse, DKLGP_coverage,
     file = file.path(output.path,"DKLGP_predict_w_y.RData"))

