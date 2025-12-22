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

############## NNGP independent ###############
NNGP_full_m3 = spVB_NNGP(y,X = X,coords=S_ordered, n.neighbors = 15, 
                         n.neighbors.vi = 3,
                         rho = 0.85, max_iter = 3500, Trace_N = 10, 
                         verbose = FALSE, covariates = TRUE, 
                         phi_max_iter = 10,
                         var_input = BRISC_var_input,
                         phi = BRISC_phi_input,
                         phi.range = c(3 / 10, 3 / 0.1),
                         ord_type = "Sum_coords")

###### summarize fitted results
NNGP_ind_samples = spVB_w_sampling(NNGP_full_m3, n.samples = 5000)
NNGP_ind_varhat = apply(NNGP_ind_samples$p.w.samples, 1, var)
NNGP_ind_w_quantile = t(apply(NNGP_ind_samples$p.w.samples[1:n, , drop = FALSE], 1, quantile,probs = c(0.025, 0.975)))

NNGP_ind_beta = spVB_beta_sampling(NNGP_full_m3, n.samples = 5000)
NNGP_ind_beta_varhat = apply(NNGP_ind_beta$p.beta.samples, 1, var)
NNGP_ind_beta_quantile = quantile(NNGP_ind_beta$p.beta.samples,c(0.025,0.975))

NNGP_ind_theta_samples = spVB_theta_sampling(NNGP_full_m3, n.samples = 5000)
NNGP_ind_theta_quantile = apply(NNGP_ind_theta_samples$p.theta.samples[1:2, ,drop = FALSE],1,quantile,probs = c(0.025, 0.975))

NNGP_ind_summary = list(
  w_mean       = NNGP_full_m3$w_mu[order(NNGP_full_m3$ord)],
  w_var        = NNGP_ind_varhat[order(NNGP_full_m3$ord)],
  w_quantile   = NNGP_ind_w_quantile[order(NNGP_full_m3$ord),],      
  theta_mean   = NNGP_full_m3$theta,
  theta_quantile = NNGP_ind_theta_quantile, 
  beta_mean    = NNGP_full_m3$beta,
  beta_var     = NNGP_ind_beta_varhat,
  beta_quantile = NNGP_ind_beta_quantile,
  run_time     = NNGP_full_m3$time
)

save(NNGP_ind_summary,
     file = file.path(output.path,"NNGP_ind_fit.RData"))

###### prediction
NNGP_full_beta_predict = predict(NNGP_full_m3, coords.0 = s_test, X.0 = X_test, covariates = TRUE,
                                 n.samples = 5000,n.report = 5000)

w_mean_pred_NNGP = apply(NNGP_full_beta_predict$p.w.0, 1, mean)
w_var_pred_NNGP = apply(NNGP_full_beta_predict$p.w.0, 1, var)
y_mean_pred_NNGP = apply(NNGP_full_beta_predict$p.y.0, 1, mean)
y_var_pred_NNGP = apply(NNGP_full_beta_predict$p.y.0, 1, var)

y_lb_pred_NNGP = sapply(1:nrow(NNGP_full_beta_predict$p.y.0),
                        FUN = function(i){
                          quantile(NNGP_full_beta_predict$p.y.0[i,],c(0.025))
                        })
y_ub_pred_NNGP = sapply(1:nrow(NNGP_full_beta_predict$p.y.0),
                        FUN = function(i){
                          quantile(NNGP_full_beta_predict$p.y.0[i,],c(0.975))
                        })

NNGP_ind_crps = mean(crps_sample(y_test, NNGP_full_beta_predict$p.y.0))
NNGP_ind_is = mean(scoringutils:::interval_score(y_test,
                                                 lower = y_lb_pred_NNGP,
                                                 upper = y_ub_pred_NNGP,
                                                 rep(95,length(y_test))))
NNGP_ind_mse = mean(scoringutils:::se_mean_sample(y_test, as.matrix(y_mean_pred_NNGP)))
NNGP_ind_coverage = mean((y_test >= y_lb_pred_NNGP) * (y_test <= y_ub_pred_NNGP))

save(w_mean_pred_NNGP, w_var_pred_NNGP, 
     y_mean_pred_NNGP, y_var_pred_NNGP, 
     y_lb_pred_NNGP, y_ub_pred_NNGP, NNGP_ind_crps, NNGP_ind_is, NNGP_ind_mse, NNGP_ind_coverage,
     file = file.path(output.path,"NNGP_ind_predict_w_y.RData"))
