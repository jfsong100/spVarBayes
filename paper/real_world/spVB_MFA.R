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

############## MFA ###############
MFA_full = spVB_MFA(y,X = X,coords=S_ordered, n.neighbors = 15, rho = 0.85,
                    max_iter = 4000, Trace_N = 30,
                    verbose = FALSE, covariates = TRUE, 
                    phi_max_iter = 10,
                    var_input = BRISC_var_input,
                    phi = BRISC_input[3],
                    phi.range = c(3 / 10, 3 / 0.1),
                    ord_type = "Sum_coords", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = FALSE)

###### summarize fitted results
MFA_full_samples = spVB_w_sampling(MFA_full, n.samples = 5000)
MFA_full_varhat = apply(MFA_full_samples$p.w.samples, 1, var)
MFA_full_w_quantile = t(apply(MFA_full_samples$p.w.samples[1:n, , drop = FALSE], 1, quantile,probs = c(0.025, 0.975)))

MFA_full_beta = spVB_beta_sampling(MFA_full, n.samples = 5000)
MFA_full_beta_varhat = apply(MFA_full_beta$p.beta.samples, 1, var)
MFA_full_beta_quantile = quantile(MFA_full_beta$p.beta.samples,c(0.025,0.975))

MFA_full_theta_samples = spVB_theta_sampling(MFA_full, n.samples = 5000)
MFA_full_theta_quantile = sapply(1:2, FUN = function(i){
  quantile(MFA_full_theta_samples$p.theta.samples[i,],c(0.025,0.975))
})

MFA_summary = list(
  w_mean       = MFA_full$w_mu[order(MFA_full$ord)],
  w_var        = MFA_full_varhat[order(MFA_full$ord)],
  w_quantile   = MFA_full_w_quantile[order(MFA_full$ord),],      
  theta_mean   = MFA_full$theta,
  theta_quantile = MFA_full_theta_quantile, 
  beta_mean    = MFA_full$beta,
  beta_var     = MFA_full_beta_varhat,
  beta_quantile = MFA_full_beta_quantile,
  run_time     = MFA_full$time
)

save(MFA_summary,
     file = file.path(output.path,"MFA_fit.RData"))

###### prediction
MFA_full_predict = predict(MFA_full, coords.0 = s_test, X.0 = X_test, covariates = TRUE,
                           n.samples = 5000, n.report = 5000)

w_mean_pred_MFA = apply(MFA_full_predict$p.w.0, 1, mean)
w_var_pred_MFA = apply(MFA_full_predict$p.w.0, 1, var)
y_mean_pred_MFA = apply(MFA_full_predict$p.y.0, 1, mean)
y_var_pred_MFA = apply(MFA_full_predict$p.y.0, 1, var)

y_lb_pred_MFA = sapply(1:nrow(MFA_full_predict$p.y.0),
                       FUN = function(i){
                         quantile(MFA_full_predict$p.y.0[i,],c(0.025))
                       })
y_ub_pred_MFA = sapply(1:nrow(MFA_full_predict$p.y.0),
                       FUN = function(i){
                         quantile(MFA_full_predict$p.y.0[i,],c(0.975))
                       })

MFA_crps = mean(crps_sample(y_test, MFA_full_predict$p.y.0))
MFA_is = mean(scoringutils:::interval_score(y_test,
                                   lower = y_lb_pred_MFA,
                                   upper = y_ub_pred_MFA,
                                   rep(95,length(y_test))))
MFA_mse = mean(scoringutils:::se_mean_sample(y_test, as.matrix(y_mean_pred_MFA)))
MFA_coverage = mean((y_test >= y_lb_pred_MFA) * (y_test <= y_ub_pred_MFA))

save(w_mean_pred_MFA, w_var_pred_MFA, 
     y_mean_pred_MFA, y_var_pred_MFA, 
     y_lb_pred_MFA, y_ub_pred_MFA, MFA_crps, MFA_is, MFA_mse, MFA_coverage,
     file = file.path(output.path,"MFA_predict_w_y.RData"))

