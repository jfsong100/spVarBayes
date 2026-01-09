######################################
# Input scenario
######################################
args = commandArgs(trailingOnly = TRUE)
for (a in args) eval(parse(text = a))
t       = as.integer(t)
n_index = as.integer(n_index)

t
n_index

######################################
# load packages
######################################
library(BRISC)
library(MASS)
library(fields)
library(Matrix)
library(hdf5r)
library(spNNGP)
library(spVarBayes)
library(scoringutils)
library(dplyr)
library(parallel)
ncore = detectCores()

######################################
# Simulation settings
######################################
n_vec       = c(1000, 5000, 10000, 50000, 100000) %>% as.integer()
beta_true   = c(2, 5)    # coefficients
tau2_true   = 0.5        # nugget
phi_true    = 1          # decay parameter
sigma2_true = 10         # spatial variance

######################################
# load data
######################################
base_dir = file.path(getwd())
data.path = file.path(base_dir, "data_sim")
scenario_path = paste0("n_", format(n_vec[n_index], scientific = FALSE), "_seed_", t)
h5_file_path = file.path(data.path, paste0(scenario_path, "_data.h5"))

n = n_vec[n_index]
h5f = hdf5r::H5File$new(h5_file_path, mode = "r")
y = as.vector(h5f[["y_gen"]][] )
X = h5f[["X"]]$read() 
w = as.vector(h5f[["f"]][] )
S_ordered = h5f[["S_ordered"]]$read()
if (n <= 10000) {
  empirical_mu  = as.vector(h5f[["empirical_mu"]][] )
  empirical_var = as.vector(h5f[["empirical_var"]][] )
  empirical_V   = h5f[["empirical_V"]]$read() 
}
h5f$close_all()

p = ncol(X)

######################################
# Initial values using BRISC estimator
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
if(n > 10000){
  d_max = dist(S_ordered[c(1,n),])
}else{
  d_max = max(as.matrix(dist(S_ordered)))
}
phi_min = min(3/d_max,max(BRISC_phi_input*0.5,0.01))
phi_max = max(30/d_max,BRISC_phi_input*1.5)

######################################
# Fit Models
######################################
m_prior = 15 
Trace_MC = 30
max_iter_LR_vec = c(3000, 3000, 4000, 4000, 4000)
max_iter_LR = max_iter_LR_vec[n_index]
# MFA + LR
MFA_pre_fit = spVB_MFA(y = y, X = X, coords=S_ordered, n.neighbors = m_prior, rho = 0.85,
                       max_iter = max_iter_LR, 
                       verbose = FALSE, covariates = TRUE, 
                       phi_max_iter = 10,
                       var_input = BRISC_var_input,
                       phi = BRISC_input[3],
                       phi.range = c(phi_min,phi_max),
                       ord_type = "AMMD", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = TRUE, warm_up = TRUE, lr_adj = 0.1)

MFA_full_LR = spVB_LR(MFA_pre_fit, get_mat = TRUE, get_para = TRUE, n_omp = ncore - 1)

# MFA
MFA_full = spVB_MFA(y,X = X,coords=S_ordered, n.neighbors = m_prior, rho = 0.85,
                    max_iter = 1000, 
                    verbose = FALSE, covariates = TRUE, 
                    phi_max_iter = 10,
                    var_input = BRISC_var_input,
                    phi = BRISC_input[3],
                    phi.range = c(phi_min,phi_max),
                    ord_type = "AMMD", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = FALSE)

# NNGP joint
NNGP_full_joint = spVB_NNGP(y,X = X,coords=S_ordered, n.neighbors = m_prior, 
                            n.neighbors.vi = 3,
                            rho = 0.85, max_iter = 2000, Trace_N = Trace_MC, 
                            verbose = FALSE, covariates = TRUE, 
                            phi_max_iter = 10,
                            var_input = BRISC_var_input,
                            phi = BRISC_phi_input,
                            mini_batch = F,
                            phi.range = c(phi_min,phi_max),
                            ord_type = "AMMD", joint = TRUE)

# NNGP independent
NNGP_full_m3 = spVB_NNGP(y,X = X,coords=S_ordered, n.neighbors = m_prior, 
                         n.neighbors.vi = 3,
                         rho = 0.85, max_iter = 1500, Trace_N = 30, 
                         verbose = FALSE, covariates = TRUE, 
                         phi_max_iter = 10,
                         var_input = BRISC_var_input,
                         phi = BRISC_phi_input,
                         mini_batch = F,
                         phi.range = c(phi_min,phi_max),
                         ord_type = "AMMD")


# spNNGP
n.samples.vec = c(5000,7500,10000,10000,10000)
n.samples = n.samples.vec[n_index]
starting = list("beta" = lm(y~X-1)$coefficients, "phi"= BRISC_phi_input, "sigma.sq" = BRISC_input[1], "tau.sq" = BRISC_input[2])
tuning = list("phi"=0.15, "sigma.sq"=1.5, "tau.sq"=1.15)
priors = list("phi.Unif"=c(phi_min,phi_max), "sigma.sq.IG"=c(0.1, 1), "tau.sq.IG"=c(1,1))
cov.model = "exponential"
burnin.vec = c(2000,2000,2000,5000,5000)
burnin = burnin.vec[n_index]
  
intercept = rep(1,n)
m.s = spNNGP(y~X-1, coords=S_ordered, starting=starting, method="latent",
              n.neighbors=m_prior, priors=priors, tuning=tuning, cov.model=cov.model,
              n.samples=n.samples, n.omp.threads=1, n.report=500, covariates = 1)
summary(m.s)
what = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, mean)
m.s.var = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, var)
m.s.beta.var = c(var(m.s$p.beta.samples[((burnin+1):n.samples),1]),
                 var(m.s$p.beta.samples[((burnin+1):n.samples),2]))

# helper function for calculating KL Divergence
myknn = function(i,s,m){
  if(m>=(i-1)) im<-1:(i-1)
  else 	
  {
    dist=rdist(s[c(1,i),],s[c(1,1:(i-1)),])[-1,-1]
    im<-sort(order(dist)[1:m])
  }
  return(im)
}

DL_MFA_LR<-function(results_MFA_LR,Sigma_star){
  n = results_MFA_LR$n
  imvec = sapply(2:n,myknn,results_MFA_LR$coords,m_prior)
  BF_list = lapply(2:(n), function(i){
    B_list = solve(Sigma_star[imvec[[i - 1]], imvec[[i - 1]]], Sigma_star[i, imvec[[i - 1]]])
    F_list = Sigma_star[i,i] - sum(B_list * Sigma_star[i, imvec[[i - 1]]])
    list(B_list = B_list, F_list = F_list)
  })
  
  colind = c(1:n,unlist(imvec))
  mi=c(1:(m_prior-1),rep(m_prior,n-m_prior))
  
  rowind = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
  
  B_lists = lapply(BF_list, function(x) x$B_list)
  F_lists = lapply(BF_list, function(x) x$F_list)
  
  V=sparseMatrix(i=rowind,j=colind,x=c(rep(1,n),-unlist(B_lists)),dims=c(n,n))
  F=sparseMatrix(i=1:n,j=1:n,x=c(1/Sigma_star[1,1],1/unlist(F_lists)),dims=c(n,n))
  
  mu_w_update = results_MFA_LR$w_mu
  n = length(mu_w_update)
  ord = results_MFA_LR$ord
  
  DL=(sum(diag(t(V) %*% F %*% V %*% empirical_V[ord,ord])) - n + 
        t(mu_w_update - empirical_mu[ord]) %*% t(V) %*% F %*% V %*% (mu_w_update - empirical_mu[ord]) +
        sum(log(1/diag(F)))-determinant(empirical_V[ord,ord],logarithm = T)$modulus )/2
  return(as.numeric(DL))
}

DL_NNGP_joint<-function(results_NNGP_joint,w_beta_cov){
  
  Sigma_star = w_beta_cov[(p+1):(p+n),(p+1):(p+n)]
  imvec = sapply(2:n,myknn,results_NNGP_joint$coords,m_prior)
  BF_list = lapply(2:(n), function(i){
    B_list = solve(Sigma_star[imvec[[i - 1]], imvec[[i - 1]]], Sigma_star[i, imvec[[i - 1]]])
    F_list = Sigma_star[i,i] - sum(B_list * Sigma_star[i, imvec[[i - 1]]])
    list(B_list = B_list, F_list = F_list)
  })
  
  colind = c(1:n,unlist(imvec))
  mi=c(1:(m_prior-1),rep(m_prior,n-m_prior))
  
  rowind = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
  
  B_lists = lapply(BF_list, function(x) x$B_list)
  F_lists = lapply(BF_list, function(x) x$F_list)
  
  V=sparseMatrix(i=rowind,j=colind,x=c(rep(1,n),-unlist(B_lists)),dims=c(n,n))
  F=sparseMatrix(i=1:n,j=1:n,x=c(1/Sigma_star[1,1],1/unlist(F_lists)),dims=c(n,n))
  
  mu_w_update = results_NNGP_joint$w_mu
  n = length(mu_w_update)
  ord = results_NNGP_joint$ord
  
  DL=(sum(diag(t(V) %*% F %*% V %*% empirical_V[ord,ord])) - n + 
        t(mu_w_update - empirical_mu[ord]) %*% t(V) %*% F %*% V %*% (mu_w_update - empirical_mu[ord]) +
        sum(log(1/diag(F)))-determinant(empirical_V[ord,ord],logarithm = T)$modulus )/2
  return(as.numeric(DL))
}

DL_MFA<-function(results_MFA){
  sigmasq_w_update = results_MFA$w_sigma_sq
  mu_w_update = results_MFA$w_mu
  n = length(mu_w_update)
  ord = results_MFA$ord
  DL=(sum(diag(empirical_V)[ord]/sigmasq_w_update) - n + 
        t(mu_w_update - empirical_mu[ord]) %*% diag(1/sigmasq_w_update) %*% (mu_w_update - empirical_mu[ord]) +
        sum(log(sigmasq_w_update))-determinant(empirical_V[ord,ord],logarithm = T)$modulus )/2
  return(as.numeric(DL))
}

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

######################################
# Summarize
######################################
output.path  = file.path(base_dir, "R_results")
if(!dir.exists(output.path)){
  dir.create(output.path)
}

if(n <= 10000){
  w_beta_cov = as.matrix(spVB_get_Vw(NNGP_full_joint))
  w_beta_var = diag(w_beta_cov)
  w_joint_var = w_beta_var[-(1:p)]
  beta_joint_var = w_beta_var[1:p]
  w_cov = as.matrix(spVB_get_Vw(NNGP_full_m3))
  w_var = diag(w_cov)
}else{
  empirical_mu = rep(NA,n)
  empirical_var = rep(NA,n)
  NNGP_full_samples = spVB_w_sampling(NNGP_full_m3, n.samples = 5000)$p.w.samples
  w_var = apply(NNGP_full_samples, 1, var)
  
  NNGP_full_joint_samples = spVB_joint_sampling(NNGP_full_joint, n.samples = 5000)
  NNGP_full_joint_w_samples = NNGP_full_joint_samples$p.w.samples
  NNGP_full_joint_beta_samples = NNGP_full_joint_samples$p.beta.samples
  w_joint_var = apply(NNGP_full_joint_w_samples, 1, var)
  beta_joint_var = apply(NNGP_full_joint_beta_samples, 1, var)
}

output_vector = rep(0,13)
output_list = matrix()

names(output_vector) = c("t","n","Trace_MC","phi_input",
                         "MFA","MFA_LR",
                         "NNGP","NNGP_joint",
                         "MFA_used_time","MFA_LR_used_time",
                         "NNGP_used_time","NNGP_joint_mb_used_time",
                         "spNNGP_time")

output_vector[1] = t
output_vector[2] = n_vec[n_index]
output_vector[3] = Trace_MC
output_vector[4] = BRISC_phi_input
output_vector[5] = ifelse(n<=10000,DL_MFA(MFA_full),NA)
output_vector[6] = ifelse(n<=10000,DL_MFA_LR(MFA_full_LR,MFA_full_LR$updated_mat[-(1:p),-(1:p)]),NA)
output_vector[7] = ifelse(n<=10000,DL_NNGP(NNGP_full_m3),NA)
output_vector[8] = ifelse(n<=10000,DL_NNGP_joint(NNGP_full_joint,w_beta_cov),NA)
output_vector[9] = MFA_full$time[3]
output_vector[10] = MFA_full_LR$time[3] + MFA_full_LR$LR_time[3]
output_vector[11] = NNGP_full_m3$time[3]
output_vector[12] = NNGP_full_joint$time[3]
output_vector[13] = m.s$run.time[3]

print(output_vector)

output_list= cbind(rep(t,n),
                   rep(n_vec[n_index],n),
                   rep(Trace_MC,n),
                   rep(BRISC_phi_input,n),
                   empirical_mu,
                   what,
                   NNGP_full_m3$w_mu,
                   NNGP_full_joint$w_mu,
                   MFA_full$w_mu,
                   MFA_full_LR$w_mu[order(MFA_full_LR$ord)],
                   empirical_var,
                   w_var,
                   w_joint_var,
                   MFA_full$w_sigma_sq,
                   diag(MFA_full_LR$updated_mat[-(1:p),-(1:p)])[order(MFA_full_LR$ord)],
                   m.s.var,
                   NNGP_full_m3$ord,
                   seq(1,n,1))

colnames(output_list) = c("t","n","Trace_MC","phi_input","empirical_mu","spNNGP_what",
                          "NNGP_m3_mu","NNGP_m3_joint_mu",
                          "MFA_mu","MFA_LR_mu",
                          "empirical_var",
                          "NNGP_m3_var","NNGP_m3_joint_var",
                          "MFA_var","MFA_LR_var","spNNGP_var",
                          "order",
                          "index")

output_beta = cbind(rep(t,p),
      rep(n_vec[n_index],p),
      t(cbind(rbind(diag(NNGP_full_m3$beta_cov),
      beta_joint_var,
      diag(MFA_full$beta_cov),
      diag(MFA_full_LR$updated_mat[1:p,1:p]),
      m.s.beta.var))))

colnames(output_beta) = c("t","n",
                       "NNGP","NNGP_joint",
                         "MFA","MFA_LR","spNNGP")
output_beta

output_theta = cbind(rep(t,p),
                    rep(n_vec[n_index],p),
                    t(cbind(rbind(NNGP_full_m3$theta[1:2],
                                  NNGP_full_joint$theta[1:2],
                                  MFA_full$theta[1:2],
                                  MFA_full_LR$theta[1:2],
                                  c(mean(m.s$p.theta.samples[(burnin+1):n.samples,1]),
                                    mean(m.s$p.theta.samples[(burnin+1):n.samples,2]))))))

colnames(output_theta) = c("t","n",
                          "NNGP","NNGP_joint",
                          "MFA","MFA_LR","spNNGP")

output_theta
file_name_output_vector = paste0(output.path,"/VI_NNGP_output_vector","_t",t,"_n",n_vec[n_index],".csv")
file_name_output_beta_vector = paste0(output.path,"/VI_NNGP_output_beta_vector","_t",t,"_n",n_vec[n_index],".csv")
file_name_output_list   = paste0(output.path,"/VI_NNGP_output_list","_t",t,"_n",n_vec[n_index],".csv")
file_name_output_theta_vector = paste0(output.path,"/VI_NNGP_output_theta_vector","_t",t,"_n",n_vec[n_index],".csv")

write.csv(output_vector,file_name_output_vector,row.names = FALSE)
write.csv(output_beta,file_name_output_beta_vector,row.names = FALSE)
write.csv(output_theta,file_name_output_theta_vector,row.names = FALSE)
write.csv(output_list,file_name_output_list,row.names = FALSE)

######################################
# Calculate metrics
######################################

CI_beta = function(object){
  beta_samples_result = spVB_beta_sampling(object, n.samples = 5000)
  beta_samples = beta_samples_result[["p.beta.samples"]]
  
  quantile_beta = sapply(1:p, function(i) quantile(beta_samples[i,],probs = c(0.025,0.975)))
  coverage_beta = sapply(1:p, function(i) beta_true[i] >= quantile_beta[1,i] && beta_true[i] <= quantile_beta[2,i] )
  
  IS_score_beta = scoringutils:::interval_score(beta_true,
                                                lower = quantile_beta[1,],
                                                upper = quantile_beta[2,],
                                                rep(95,p))
  
  crps_score_beta = crps_sample(beta_true, beta_samples)
  
  result_list = list(beta_result = cbind(coverage_beta, IS_score_beta, crps_score_beta))
  
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

CI_beta_w = function(object,w_true){
  w_samples_result = spVB_joint_sampling(object, n.samples = 5000)
  w_samples = w_samples_result[["p.w.samples"]]
  w = w_true[object$ord]
  quantile = sapply(1:n, function(i) quantile(w_samples[i,],probs = c(0.025,0.975)))
  coverage = sum(sapply(1:n, function(i) w[i] >= quantile[1,i] && w[i] <= quantile[2,i] ))/n
  
  interval_range = rep(95,n)
  IS_score = scoringutils:::interval_score(w,
                                           lower = quantile[1,],
                                           upper = quantile[2,],
                                           interval_range)
  IS_score_mean = mean(IS_score)
  crps_score_mean = mean(crps_sample(w, w_samples))
  
  beta_samples = w_samples_result[["p.beta.samples"]]
  
  quantile_beta = sapply(1:p, function(i) quantile(beta_samples[i,],probs = c(0.025,0.975)))
  coverage_beta = sapply(1:p, function(i) beta_true[i] >= quantile_beta[1,i] && beta_true[i] <= quantile_beta[2,i] )
  
  IS_score_beta = scoringutils:::interval_score(beta_true,
                                           lower = quantile_beta[1,],
                                           upper = quantile_beta[2,],
                                           rep(95,p))
  
  crps_score_beta = crps_sample(beta_true, beta_samples)
  
  result_list = list(w_result = c(coverage, IS_score_mean, crps_score_mean),
                     beta_result = cbind(coverage_beta, IS_score_beta, crps_score_beta))
  return(result_list)
}

CI_beta_w_LR = function(object,w_true){
  n = object$n
  w_samples_result = spVB_LR_sampling(object, n.samples = 5000)
  w_samples = w_samples_result[["p.w.samples"]]
  w = w_true[object$ord]
  quantile = sapply(1:n, function(i) quantile(w_samples[i,],probs = c(0.025,0.975)))
  coverage = sum(sapply(1:n, function(i) w[i] >= quantile[1,i] && w[i] <= quantile[2,i] ))/n
  
  interval_range = rep(95,n)
  IS_score = scoringutils:::interval_score(w,
                                           lower = quantile[1,],
                                           upper = quantile[2,],
                                           interval_range)
  IS_score_mean = mean(IS_score)
  crps_score_mean = mean(crps_sample(w, w_samples))
  
  beta_samples = w_samples_result[["p.beta.samples"]]
  
  quantile_beta = sapply(1:p, function(i) quantile(beta_samples[i,],probs = c(0.025,0.975)))
  coverage_beta = sapply(1:p, function(i) beta_true[i] >= quantile_beta[1,i] && beta_true[i] <= quantile_beta[2,i] )
  
  IS_score_beta = scoringutils:::interval_score(beta_true,
                                                lower = quantile_beta[1,],
                                                upper = quantile_beta[2,],
                                                rep(95,p))
  
  crps_score_beta = crps_sample(beta_true, beta_samples)
  
  result_list = list(w_result = c(coverage, IS_score_mean, crps_score_mean),
                     beta_result = cbind(coverage_beta, IS_score_beta, crps_score_beta))
  return(result_list)
}

CI_beta_w_spNNGP = function(m.s, w_true){
  w_samples = m.s$p.w.samples[,(burnin+1):n.samples]
  
  quantile = sapply(1:n, function(i) quantile(w_samples[i,],probs = c(0.025,0.975)))
  coverage = sum(sapply(1:n, function(i) w_true[i] >= quantile[1,i] && w_true[i] <= quantile[2,i] ))/n
  
  interval_range = rep(95,n)
  IS_score = scoringutils:::interval_score(w_true,
                                           lower = quantile[1,],
                                           upper = quantile[2,],
                                           interval_range)
  IS_score_mean = mean(IS_score)
  crps_score_mean = mean(crps_sample(w, w_samples))
  
  beta_samples = t(m.s$p.beta.samples[(burnin+1):n.samples,])
  
  quantile_beta = sapply(1:p, function(i) quantile(beta_samples[i,],probs = c(0.025,0.975)))
  coverage_beta = sapply(1:p, function(i) beta_true[i] >= quantile_beta[1,i] && beta_true[i] <= quantile_beta[2,i] )
  
  IS_score_beta = scoringutils:::interval_score(beta_true,
                                                lower = quantile_beta[1,],
                                                upper = quantile_beta[2,],
                                                rep(95,p))
  
  crps_score_beta = crps_sample(beta_true, beta_samples)
  
  return(list(mean_coverage = coverage,
              mean_is = IS_score_mean,
              mean_crps = crps_score_mean,
              beta_result = cbind(coverage_beta, IS_score_beta, crps_score_beta)))
}

CI_NNGP = CI_w(NNGP_full_m3,w_true = w)
CI_NNGP_joint = CI_beta_w(NNGP_full_joint,w_true = w)
CI_MFA = CI_w(MFA_full,w_true = w)
CI_MFA_LR = CI_beta_w_LR(MFA_full_LR,w_true = w)
CI_spNNGP = CI_beta_w_spNNGP(m.s,w_true = w)

output_vector_CI_w = matrix(c(CI_NNGP$mean_coverage, CI_NNGP_joint$w_result[1], CI_spNNGP$mean_coverage, CI_MFA$mean_coverage, CI_MFA_LR$w_result[1],
                              CI_NNGP$mean_is, CI_NNGP_joint$w_result[2], CI_spNNGP$mean_is, CI_MFA$mean_is, CI_MFA_LR$w_result[2],
                              CI_NNGP$mean_crps, CI_NNGP_joint$w_result[3], CI_spNNGP$mean_crps, CI_MFA$mean_crps, CI_MFA_LR$w_result[3]),nrow = 5)
output_vector_CI_w

CI_beta_NNGP = CI_beta(NNGP_full_m3)
CI_beta_MFA = CI_beta(MFA_full)

first_beta = matrix(c( CI_beta_NNGP$beta_result[1, ],
                CI_NNGP_joint$beta_result[1, ],
                CI_spNNGP$beta_result[1, ],
                CI_beta_MFA$beta_result[1, ],
                CI_MFA_LR$beta_result[1, ] ),nrow = 3)

second_beta = matrix(c( CI_beta_NNGP$beta_result[2, ],
                 CI_NNGP_joint$beta_result[2, ],
                 CI_spNNGP$beta_result[2, ],
                 CI_beta_MFA$beta_result[2, ],
                 CI_MFA_LR$beta_result[2, ] ),nrow = 3)

output_vector_CI_beta = rbind(first_beta, second_beta)
row.names(output_vector_CI_beta) = c(paste0(rep("beta1_",3),c("coverage","is","crps")),
                                     paste0(rep("beta2_",3),c("coverage","is","crps")))
colnames(output_vector_CI_beta) = c("NNGP","NNGP_joint","spNNGP","MFA","MFA_LR")
output_vector_CI_beta

file_name_output_w_CI = paste0(output.path,"/VI_NNGP_output_vector_CI_w","_t",t,"_n",n_vec[n_index],".csv")
write.csv(output_vector_CI_w,file_name_output_w_CI,row.names = FALSE)
file_name_output_beta_CI = paste0(output.path,"/VI_NNGP_output_vector_CI_beta","_t",t,"_n",n_vec[n_index],".csv")
write.csv(output_vector_CI_beta,file_name_output_beta_CI,row.names = FALSE)
