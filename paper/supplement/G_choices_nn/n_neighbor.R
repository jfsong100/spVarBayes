######################################
# Load packages
######################################
library(BRISC)
library(MASS)
library(fields)
library(RColorBrewer)
library(classInt)
library(psych)
library(rstan)
library(Matrix)
library(magrittr)
library(reticulate)
library(hdf5r)
library(scoringutils)
library(spVarBayes)
library(spNNGP)

######################################
# Input scenario
######################################
t = 12
n_index = 2

base_dir  = normalizePath(file.path(getwd(), "..", ".."), mustWork = TRUE)
data.path = file.path(base_dir, "simulations")
output.path = file.path(base_dir, "supplement","G_choices_nn")
source(file.path(data.path,"data_generation.R"))


######################################
# Helper function
######################################
DL_NNGP<-function(results_NNGP){
  m = results_NNGP$n.neighbors.vi
  if(m==1){
    rowind_vi = c(1:n,2:n)
  }else{
    mi=c(1:(m-1),rep(m,n-m))
    rowind_vi = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
  }
  
  colind_vi = c(1:n,results_NNGP$nnIndx_vi+1)
  
  V_approx = sparseMatrix(i = rowind_vi,
                          j = colind_vi,
                          x = c(rep(1,n),-results_NNGP$A_vi),dims=c(n,n))
  
  D_approx = sparseMatrix(i = seq(1,n,1),
                          j = seq(1,n,1),
                          x = results_NNGP$D_vi,dims=c(n,n))
  mu_w_update = results_NNGP$w_mu
  n = length(mu_w_update)
  ord = results_NNGP$ord
  DL=(sum(diag(t(V_approx) %*% solve(D_approx) %*% V_approx %*% empirical_V[ord,ord])) - n + 
        t(mu_w_update - empirical_mu[ord]) %*% t(V_approx) %*% solve(D_approx) %*% V_approx %*% (mu_w_update - empirical_mu[ord]) +
        sum(log(diag(D_approx)))-determinant(empirical_V[ord,ord],logarithm = T)$modulus )/2
  return(as.numeric(DL))
}

CI_w = function(object, w_true){
  w = w_true[object$ord]
  
  w_samples_result = spVB_w_sampling(object, n.samples = 5000)
  w_samples = w_samples_result[["p.w.samples"]]
  
  quantile = sapply(1:n, function(i) quantile(w_samples[i,],probs = c(0.025,0.975)))
  coverage = sum(sapply(1:n, function(i) w[i] >= quantile[1,i] && w[i] <= quantile[2,i] ))/n
  
  interval_range = rep(95,n)
  IS_score = scoringutils:::interval_score(w,
                                           lower = quantile[1,],
                                           upper = quantile[2,],
                                           interval_range)
  IS_score_mean = mean(IS_score)
  crps_score_mean = mean(crps_sample(w, w_samples))
  
  result_list = list(mean_coverage = coverage,
                     mean_is = IS_score_mean,
                     mean_crps = crps_score_mean)
  return(result_list)
}


run_NNGP = function(n.neighbors.vi){
  
  NNGP_full = spVB_NNGP(y,X = X,coords=S_ordered, n.neighbors = 15, 
                           n.neighbors.vi = n.neighbors.vi,
                           rho = 0.85, max_iter = 1500, Trace_N = 30, 
                           verbose = FALSE, covariates = TRUE, 
                           phi_max_iter = 10,
                           var_input = BRISC_var_input,
                           phi = BRISC_phi_input,
                           mini_batch = FALSE,
                           phi.range = c(phi_min,phi_max),
                           ord_type = "AMMD")
  
  output_vector = rep(0,7)

  CI_NNGP = CI_w(NNGP_full,w_true = w)
  
  output_vector[1] = n.neighbors.vi
  output_vector[2] = n_vec[n_index]
  output_vector[3] = DL_NNGP(NNGP_full)
  output_vector[4] = NNGP_full$time[3]
  output_vector[5] = CI_NNGP$mean_coverage
  output_vector[6] = CI_NNGP$mean_is
  output_vector[7] = CI_NNGP$mean_crps

  w_cov = as.matrix(spVB_get_Vw(NNGP_full))
  w_var = diag(w_cov)
  
  return(list(output_vector = output_vector,
              w_var = w_var))
}


######################################
# Initial values
######################################
BRISC_input = BRISC_estimation(coords = S_ordered, y = y, x = X, sigma.sq = 1,
                               tau.sq = 0.1, phi = 1,
                               nu = 0.5, n.neighbors = 15,
                               n_omp = 1, 
                               cov.model = "exponential",
                               search.type = "tree",
                               stabilization = NULL,
                               pred.stabilization = 1e-5,
                               verbose = TRUE, eps = 2e-05,
                               nugget_status = 1, order = "AMMD",
                               neighbor = NULL, tol = 12)$Theta
BRISC_phi_input = BRISC_input[3]
BRISC_var_input = 1/(1/BRISC_input[2]+1/BRISC_input[1])
print(c("BRISC_estimation for phi is",BRISC_phi_input))
print(c("BRISC_estimation for var is",BRISC_var_input))
BRISC_input
if(n>10000){
  d_max = dist(S_ordered[c(1,n),])
}else{
  d_max = max(as.matrix(dist(S_ordered)))
}

phi_min = min(3/d_max,max(BRISC_phi_input-0.5,0.01))
phi_max = max(30/d_max,BRISC_phi_input+0.1)

######################################
# Run experiments
######################################
set.seed(1)
ord_maxmin = BRISC_order(S_ordered, order = "AMMD")

test_n.neighbors.vi = 20

results_NNGP = vector("list", test_n.neighbors.vi)

output_matrix_NNGP = matrix(NA, nrow = test_n.neighbors.vi, ncol = 7)

w_var_list_NNGP = vector("list", test_n.neighbors.vi)

for (i in 1:test_n.neighbors.vi) {
  res_NNGP = run_NNGP(i)
  
  output_matrix_NNGP[i, ] = res_NNGP$output_vector
  
  w_var_list_NNGP[[i]] = res_NNGP$w_var
}

output_df_NNGP = as.data.frame(output_matrix_NNGP)
colnames(output_df_NNGP) = c("m", "n", "KL", "time", "coverage", "interval_score", "crps")

######################################
# Run spNNGP for reference
######################################
n.samples.vec = c(5000,7500,10000)
n.samples = n.samples.vec[n_index]
starting = list("beta" = lm(y~X-1)$coefficients, "phi"= BRISC_phi_input, "sigma.sq" = BRISC_input[1], "tau.sq" = BRISC_input[2])
tuning = list("phi"=0.15, "sigma.sq"=1.5, "tau.sq"=1.15)
priors = list("phi.Unif"=c(phi_min,phi_max), "sigma.sq.IG"=c(0.1, 1), "tau.sq.IG"=c(1,1))
cov.model = "exponential"
burnin = 2000

intercept = rep(1,n)
m.s = spNNGP(y~X-1, coords=S_ordered, starting=starting, method="latent",
              n.neighbors=15, priors=priors, tuning=tuning, cov.model=cov.model,
              n.samples=n.samples, n.omp.threads=1, n.report=500, covariates = 1)
summary(m.s)
what = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, mean)
m.s.var = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, var)

######################################
# Summarize results
######################################
library(tidyr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggrastr)

long_df = output_df_NNGP %>%
  select(-n) %>%
  pivot_longer(cols = -m, names_to = "metric", values_to = "value")

# ggplot(long_df, aes(x = m, y = value)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~metric, scales = "free_y", ncol = 2) +
#   labs(x = "Number of Nearest Neighbors for the Variational Family", y = "Value") +
#   theme_minimal()

long_df$metric = recode(long_df$metric,
                         "coverage" = "95% Coverage",
                         "crps" = "CRPS",
                         "interval_score" = "Interval Score",
                         "KL" = "KL Divergence",
                         "time" = "Running Time"
)

long_df$metric = factor(long_df$metric, levels = c(
  "95% Coverage", "CRPS", "Interval Score", "KL Divergence", "Running Time"
))

p1 = ggplot(long_df, aes(x = m, y = value)) +
  geom_line(color = "black") +
  geom_point(color = "black") +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  labs(
    x = "Number of Nearest Neighbors for the Variational Family",
    y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.x = element_text(margin = margin(t = 10)),
    panel.grid.minor = element_blank()
  )

p1$layers[[1]] = rasterise(p1$layers[[1]], dpi = 300)

ggsave(file.path(output.path,"NNGP_ind_metric.pdf"), plot = p1,
       width = 12.2, height = 8.64, units = "in",
       dpi = 300, device = cairo_pdf)

w_var_df = map2_dfr(w_var_list_NNGP, 1:test_n.neighbors.vi, ~{
  tibble(
    idx = 1:length(.x),
    var_est = .x,
    neighbors = .y
  )
})

spNNGP_df = tibble(
  idx = 1:n,
  spNNGP_var = m.s.var[ord_maxmin]
)

plot_df = left_join(w_var_df, spNNGP_df, by = "idx")

plot_df$neighbors = factor(
  paste0("mq = ", plot_df$neighbors),
  levels = paste0("mq = ", 1:test_n.neighbors.vi)
)

p2 = ggplot(plot_df, aes(x = spNNGP_var, y = var_est)) +
  geom_point(alpha = 0.3, size = 0.5) +
  facet_wrap(~ neighbors, ncol = 5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    x = "MCMC approximated variance",
    y = "VI approximated variance",
  ) +
  theme_minimal()

p2$layers[[1]] = rasterise(p2$layers[[1]], dpi = 300)

ggsave(file.path(output.path,"NNGP_ind_variance.pdf"), plot = p2,
       width = 12.2, height = 8.64, units = "in",
       dpi = 300, device = cairo_pdf)

