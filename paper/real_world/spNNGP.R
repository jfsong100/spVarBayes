library(spNNGP)
library(BRISC)
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
############## spNNGP ###############
n.samples = 15000

starting = list("beta" = lm(y~X-1)$coefficients, "phi"= BRISC_phi_input, "sigma.sq" = BRISC_input[1], "tau.sq" = BRISC_input[2])
priors = list("phi.Unif" = c(3 / 10, 3 / 0.1), "sigma.sq.IG" = c(2, 40), "tau.sq.IG" = c(2, 10))
cov.model = "exponential"
tuning = list("phi"=0.15, "sigma.sq"=1.5, "tau.sq"=1.15)
cov.model = "exponential"

burnin = 10000
n.samples = 15000
m.s = spNNGP(y~X-1, coords=S_ordered, starting=starting, method="latent",
             n.neighbors=15, priors=priors, tuning=tuning, cov.model=cov.model,
             n.samples=n.samples, n.omp.threads=1, n.report=500, covariates = 1,
             sub.sample = list(start = burnin + 1, thin = 1))

###### summarize fitted results
wvarhat = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, var)
what = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, mean)
w_quantile = apply(m.s$p.w.samples[1:n, (burnin + 1):n.samples, drop = FALSE], 1, quantile,probs = c(0.025, 0.975))

theta_mean = apply(m.s$p.theta.samples[((burnin+1):n.samples),], 2, mean)
theta_quantile = apply(m.s$p.theta.samples[(burnin + 1):n.samples, 1:3, drop = FALSE],2,quantile,probs = c(0.025, 0.975))

beta_mean = mean(m.s$p.beta.samples[((burnin+1):n.samples)])
beta_var = var(m.s$p.beta.samples[((burnin+1):n.samples)])
beta_quantile = quantile(m.s$p.beta.samples[((burnin+1):n.samples),1], c(0.025,0.975))

spNNGP_summary = list(
  w_mean       = what,
  w_var        = wvarhat,
  w_quantile   = w_quantile,      
  theta_mean   = theta_mean,
  theta_quantile = theta_quantile, 
  beta_mean    = beta_mean,
  beta_var     = beta_var,
  beta_quantile = beta_quantile,
  run_time     = m.s$run.time
)

save(spNNGP_summary,
     file = file.path(output.path,"spNNGP_fit.RData"))

###### prediction
p.s = predict(m.s, X.0 = X_test, coords.0 = s_test, n.omp.threads=1, sub.sample = list(start = 10001, thin = 1), n.report = 5000)

###### summarize predicted results
w_mean_pred_spNNGP = apply(p.s$p.w.0, 1, mean)
w_var_pred_spNNGP = apply(p.s$p.w.0, 1, var)
y_mean_pred_spNNGP = apply(p.s$p.y.0, 1, mean)
y_var_pred_spNNGP = apply(p.s$p.y.0, 1, var)
y_lb_pred_spNNGP = apply(p.s$p.y.0[1:n_test, , drop = FALSE], 1, quantile,probs = c(0.025))
y_ub_pred_spNNGP = apply(p.s$p.y.0[1:n_test, , drop = FALSE], 1, quantile,probs = c(0.975))

spNNGP_crps = mean(crps_sample(y_test, p.s$p.y.0))
spNNGP_is = mean(scoringutils:::interval_score(y_test,
                                               lower = y_lb_pred_spNNGP,
                                               upper = y_ub_pred_spNNGP,
                                               rep(95,length(y_test))))
spNNGP_mse = mean(scoringutils:::se_mean_sample(y_test, as.matrix(y_mean_pred_spNNGP)))
spNNGP_coverage = mean((y_test >= y_lb_pred_spNNGP) * (y_test <= y_ub_pred_spNNGP))

save(w_mean_pred_spNNGP, w_var_pred_spNNGP, 
     y_mean_pred_spNNGP, y_var_pred_spNNGP, 
     y_lb_pred_spNNGP, y_ub_pred_spNNGP, spNNGP_crps, spNNGP_is, spNNGP_mse, spNNGP_coverage,
     file = file.path(output.path,"spNNGP_predict_w_y.RData"))
