library(spVarBayes)
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
############## Fit model ###############
########################################
BRISC_input = BRISC_estimation(coords = S_ordered, y = y, x = X, sigma.sq = 1,
                               tau.sq = 0.1, phi = 1,
                               nu = 0.5, n.neighbors = 15,
                               n_omp = 1,
                               cov.model = "exponential",
                               search.type = "tree",
                               stabilization = NULL,
                               pred.stabilization = 1e-5,
                               verbose = TRUE, eps = 2e-05,
                               nugget_status = 1,
                               neighbor = NULL, tol = 12)$Theta
BRISC_phi_input = BRISC_input[3]
BRISC_var_input = 1/(1/BRISC_input[2]+1/BRISC_input[1])
print(c("BRISC_estimation for phi is",BRISC_phi_input))
print(c("BRISC_estimation for var is",BRISC_var_input))
BRISC_input

############## MFA-LR ###############
MFA_pre_fit = spVB_MFA(y,X = X,coords=S_ordered, n.neighbors = 15, rho = 0.85,
                       max_iter = 4500, Trace_N = 10,
                       verbose = FALSE, covariates = TRUE, 
                       phi_max_iter = 10,
                       var_input = BRISC_var_input,
                       phi = BRISC_input[3],
                       phi.range = c(3 / 10, 3 / 0.1),
                       ord_type = "Sum_coords", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = TRUE, warm_up = TRUE, lr_adj = 0.1)

###### summarize fitted results
MFA_full_LR = spVB_LR(MFA_pre_fit, get_mat = TRUE, get_para = TRUE, n_omp = 9)

MFA_LR_theta_samples = spVB_theta_sampling(MFA_full_LR, n.samples = 5000)$p.theta.samples
MFA_LR_theta_quantile = sapply(1:2, FUN = function(i){
  quantile(MFA_LR_theta_samples[i,],c(0.025,0.975))
})

MFA_LR_samples = spVB_LR_sampling(MFA_full_LR, n.samples = 5000)
MFA_LR_beta_samples = MFA_LR_samples$p.beta.samples
MFA_LR_beta_quantile = quantile(MFA_LR_beta_samples,c(0.025,0.975))
MFA_LR_w_quantile = t(apply(MFA_LR_samples$p.w.samples[1:n, , drop = FALSE], 1, quantile,probs = c(0.025, 0.975)))

MFA_LR_summary = list(
  w_mean       = MFA_full_LR$w_mu[order(MFA_full_LR$ord)],
  w_var        = diag(MFA_full_LR$updated_mat)[-(1:p)][order(MFA_full_LR$ord)],
  w_quantile   = MFA_LR_w_quantile[order(MFA_full_LR$ord),],      
  theta_mean   = MFA_full_LR$theta,
  theta_quantile = MFA_LR_theta_quantile, 
  beta_mean    = MFA_full_LR$beta,
  beta_var     = diag(MFA_full_LR$updated_mat)[1:p],
  beta_quantile = MFA_LR_beta_quantile,
  run_time     = MFA_full_LR$time + MFA_full_LR$LR_time
)

save(MFA_LR_summary,
     file = file.path(output.path,"MFA_LR_fit.RData"))

###### prediction
MFA_full_LR_predict = predict(MFA_full_LR,coords.0 = s_test,
                              X.0 = X_test,
                              covariates = TRUE,
                              n.samples = 5000,
                              n.report = 5000)

w_mean_pred_MFA_LR = apply(MFA_full_LR_predict$p.w.0, 1, mean)
w_var_pred_MFA_LR = apply(MFA_full_LR_predict$p.w.0, 1, var)

y_mean_pred_MFA_LR = apply(MFA_full_LR_predict$p.y.0, 1, mean)
y_var_pred_MFA_LR = apply(MFA_full_LR_predict$p.y.0, 1, var)

y_lb_pred_MFA_LR = sapply(1:nrow(MFA_full_LR_predict$p.y.0),
                          FUN = function(i){
                            quantile(MFA_full_LR_predict$p.y.0[i,],c(0.025))
                          })
y_ub_pred_MFA_LR = sapply(1:nrow(MFA_full_LR_predict$p.y.0),
                          FUN = function(i){
                            quantile(MFA_full_LR_predict$p.y.0[i,],c(0.975))
                          })

MFA_LR_crps = mean(crps_sample(y_test, MFA_full_LR_predict$p.y.0))
MFA_LR_is = mean(scoringutils:::interval_score(y_test,
                                   lower = y_lb_pred_MFA_LR,
                                   upper = y_ub_pred_MFA_LR,
                                   rep(95,length(y_test))))

MFA_LR_mse = mean(scoringutils:::se_mean_sample(y_test, as.matrix(y_mean_pred_MFA_LR)))
MFA_LR_coverage = mean((y_test >= y_lb_pred_MFA_LR) * (y_test <= y_ub_pred_MFA_LR))

save(w_mean_pred_MFA_LR, w_var_pred_MFA_LR, 
     y_mean_pred_MFA_LR, y_var_pred_MFA_LR, 
     y_lb_pred_MFA_LR, y_ub_pred_MFA_LR, MFA_LR_crps, MFA_LR_is, MFA_LR_mse, MFA_LR_coverage,
     file = file.path(output.path,"MFA_LR_predict_w_y.RData"))
