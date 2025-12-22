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
library(rhdf5)
library(spNNGP)
library(spVarBayes)
library(scoringutils)
library(dplyr)
library(parallel)
ncore = detectCores()
######################################
# Simulation settings
######################################
tau2_true = 0.5
beta_true = c(2,5)
phi_true = 1
sigma2_true = 10

n_train = c(1000,5000,10000)
n_test  = c(100 ,500 ,1000)
n_vec = n_train + n_test
n=n_vec[n_index]

######################################
# load data
######################################
base_dir  = file.path(getwd())
data.path = file.path(base_dir,"data_sim")
scenario_path = paste0("n_",format(n_vec[n_index], scientific = FALSE),"_seed_",t)
h5_file_path = paste0(data.path, "/",scenario_path, "_data.h5")

n          = n_vec[n_index]
y_train    = as.vector(h5read(h5_file_path, "y_train"))
X_train    = h5read(h5_file_path, "X_train")
w_train    = as.vector(h5read(h5_file_path, "f_train"))
S_train    = h5read(h5_file_path, "S_train")
y_test     = as.vector(h5read(h5_file_path, "y_test"))
X_test     = h5read(h5_file_path, "X_test")
w_test     = as.vector(h5read(h5_file_path, "f_test"))
S_test     = h5read(h5_file_path, "S_test")
p = ncol(X_train)

######################################
# Initial values using BRISC estimator
######################################
BRISC_input = BRISC_estimation(coords = S_train, y = y_train, x = X_train, sigma.sq = 1,
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

if(n_train[n_index]>10000){
  d_max = dist(S_train[c(1,length(y_train)),])
}else{
  d_max = max(as.matrix(dist(S_train)))
}

phi_min = min(3/d_max,max(BRISC_phi_input*0.5,0.01))
phi_max = max(30/d_max,BRISC_phi_input*1.5)

######################################
# Fit Models
######################################
m_prior = 15 
Trace_MC = 30
# MFA + LR
max_iter_LR_vec = c(3000, 3000, 4000)
max_iter_LR = max_iter_LR_vec[n_index]
MFA_pre_fit = spVB_MFA(y = y_train,X = X_train,coords=S_train, n.neighbors = m_prior, rho = 0.85,
                       max_iter = max_iter_LR, 
                       verbose = FALSE, covariates = TRUE, 
                       phi_max_iter = 10,
                       var_input = BRISC_var_input,
                       phi = BRISC_input[3],
                       phi.range = c(phi_min,phi_max),
                       ord_type = "AMMD", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = TRUE, warm_up = TRUE, lr_adj = 0.1)

MFA_full_LR = spVB_LR(MFA_pre_fit, get_mat = TRUE, get_para = TRUE, n_omp = (ncore-1))

# MFA
MFA_full = spVB_MFA(y = y_train,X = X_train,coords=S_train, n.neighbors = m_prior, rho = 0.85,
                    max_iter = 1000, 
                    verbose = FALSE, covariates = TRUE, 
                    phi_max_iter = 10,
                    var_input = BRISC_var_input,
                    phi = BRISC_input[3],
                    phi.range = c(phi_min,phi_max),
                    ord_type = "AMMD", tau_sq_input = BRISC_input[2], sigma_sq_input = BRISC_input[1], LR = FALSE)

# NNGP joint
NNGP_full_joint = spVB_NNGP(y = y_train,X = X_train,coords=S_train,n.neighbors = m_prior, 
                            n.neighbors.vi = 3,
                            rho = 0.85, max_iter = 2000, Trace_N = Trace_MC, 
                            verbose = FALSE, covariates = TRUE, 
                            phi_max_iter = 10,
                            var_input = BRISC_var_input,
                            phi = BRISC_phi_input,
                            mini_batch = FALSE,
                            phi.range = c(phi_min,phi_max),
                            ord_type = "AMMD", joint = TRUE)

w_beta_cov = spVB_get_Vw(NNGP_full_joint)
w_beta_var = diag(w_beta_cov)

# NNGP independent
NNGP_full_m3 = spVB_NNGP(y = y_train,X = X_train,coords=S_train, n.neighbors = m_prior, 
                         n.neighbors.vi = 3,
                         rho = 0.85, max_iter = 1500, Trace_N = Trace_MC, 
                         verbose = FALSE, covariates = TRUE, 
                         phi_max_iter = 10,
                         var_input = BRISC_var_input,
                         phi = BRISC_phi_input,
                         mini_batch = FALSE,
                         phi.range = c(phi_min,phi_max),
                         ord_type = "AMMD")

# spNNGP
n.samples.vec <- c(5000,7500,10000)
n.samples = n.samples.vec[n_index]
starting <- list("beta" = lm(y_train~X_train-1)$coefficients, "phi"= BRISC_phi_input, "sigma.sq"=BRISC_input[1], "tau.sq"=BRISC_input[2])
tuning <- list("phi"=0.15, "sigma.sq"=1.5, "tau.sq"=1.15)
priors <- list("phi.Unif"=c(phi_min,phi_max), "sigma.sq.IG"=c(0.1, 1), "tau.sq.IG"=c(1,1))
cov.model <- "exponential"

intercept = rep(1,length(y_train))
m.s <- spNNGP(y_train~X_train-1, coords=S_train, starting=starting, method="latent",
              n.neighbors=m_prior, priors=priors, tuning=tuning, cov.model=cov.model,
              n.samples=n.samples, n.omp.threads=1, n.report=500, covariates = 1)

burnin = 2000

summary(m.s)
what = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, mean)
m.s.var = apply(m.s$p.w.samples[,((burnin+1):n.samples)], 1, var)
summary(m.s)

######################################
# Summarize
######################################
output.path = file.path(base_dir,"R_results")
if(!dir.exists(output.path)){
  dir.create(output.path)
}
output_vector = rep(0,19)
output_list = matrix()

names(output_vector) = c("t","n","Trace_MC","phi_input",
                         "NNGP_used_time","NNGP_joint_mb_used_time",
                         "MFA_used_time","MFA_LR_used_time",
                         "spNNGP_time",
                         "NNGP_sigmasq","NNGP_joint_sigmasq","MFA_sigmasq","MFA_LR_sigmasq","spNNGP_sigmasq",
                         "NNGP_tausq","NNGP_joint_tausq","MFA_tausq","MFA_LR_tausq","spNNGP_tausq")

output_vector[1] = t
output_vector[2] = n_train[n_index]
output_vector[3] = Trace_MC
output_vector[4] = BRISC_phi_input
output_vector[5] = NNGP_full_m3$time[3]
output_vector[6] = NNGP_full_joint$time[3]
output_vector[7] = MFA_full$time[3]
output_vector[8] = MFA_full_LR$time[3] + MFA_full_LR$LR_time[3]
output_vector[9] = m.s$run.time[3]

output_vector[10] = NNGP_full_m3$theta[1]
output_vector[11] = NNGP_full_joint$theta[1]
output_vector[12] = MFA_full$theta[1]
output_vector[13] = MFA_full_LR$theta[1]
output_vector[14] = mean(m.s$p.theta.samples[((burnin+1):n.samples),1])

output_vector[15] = NNGP_full_m3$theta[2]
output_vector[16] = NNGP_full_joint$theta[2]
output_vector[17] = MFA_full$theta[2]
output_vector[18] = MFA_full_LR$theta[2]
output_vector[19] = mean(m.s$p.theta.samples[((burnin+1):n.samples),2])

print(output_vector)

output_list= cbind(rep(t,n_train[n_index]),
                   rep(n_train[n_index],n_train[n_index]),
                   rep(Trace_MC,n_train[n_index]),
                   rep(BRISC_phi_input,n_train[n_index]),
                   what,
                   NNGP_full_m3$w_mu,
                   NNGP_full_joint$w_mu,
                   MFA_full$w_mu,
                   MFA_full_LR$w_mu,
                   diag(spVB_get_Vw(NNGP_full_m3)),
                   w_beta_var[(p+1):(p+length(y_train))],
                   MFA_full$w_sigma_sq,
                   diag(MFA_full_LR$updated_mat)[(p+1):(p+length(y_train))],
                   m.s.var,
                   NNGP_full_m3$ord,
                   seq(1,n_train[n_index],1))

colnames(output_list) = c("t","n","Trace_MC","phi_input","spNNGP_what",
                          "NNGP_m3_mu","NNGP_m3_joint_mu",
                          "MFA_mu","MFA_LR_mu",
                          "NNGP_m3_var","NNGP_m3_joint_var",
                          "MFA_var","MFA_LR_var","spNNGP_var",
                          "order",
                          "index")
 
file_name_output_vector = paste0(output.path,"/VI_NNGP_output_vector","_t",t,"_n",n_train[n_index],".csv")
file_name_output_list   = paste0(output.path,"/VI_NNGP_output_list","_t",t,"_n",n_train[n_index],".csv")

write.csv(output_vector,file_name_output_vector,row.names = FALSE)
write.csv(output_list,file_name_output_list,row.names = FALSE)


######################################
# Making predictions on test data
######################################
NNGP_full_m3_predict = predict(NNGP_full_m3,
                               coords.0 = S_test,
                               X.0 = X_test,
                               covariates = TRUE,
                               n.samples = 5000)

NNGP_joint_m3_predict = predict(NNGP_full_joint,
                               coords.0 = S_test,
                               X.0 = X_test,
                               covariates = TRUE,
                               n.samples = 5000)

MFA_full_predict = predict(MFA_full,
                           coords.0 = S_test,
                           X.0 = X_test,
                           covariates = TRUE,
                           n.samples = 5000)

MFA_LR_predict = predict(MFA_full_LR,
                         coords.0 = S_test,
                         X.0 = X_test,
                         covariates = TRUE,
                             n.samples = 5000)

p.s = predict(m.s, X.0 = X_test, coords.0 = S_test,
               sub.sample = list(start = (burnin+1), thin = 1),
               n.omp.threads=1)

######################################
# Metrics 
######################################
calculate_IS = function(w_test,p.w.0){
  library(scoringutils)
  interval_range = rep(95,length(w_test))
  quantile = sapply(1:length(y_test), function(i) quantile(p.w.0[i,],probs = c(0.025,0.975)))
  IS_score = scoringutils:::interval_score(w_test,
                            lower = quantile[1,],
                            upper = quantile[2,],
                            interval_range)
  return(IS_score)
}

calculate_coverage= function(w_test,p.w.0){
  quantile = sapply(1:length(y_test), function(i) quantile(p.w.0[i,],probs = c(0.025,0.975)))
  coverage = sum(sapply(1:length(y_test), function(i) w_test[i] >= quantile[1,i] && w_test[i] <= quantile[2,i] ))/length(y_test)
  return(coverage)
}

#### w ####
interval_score_data = data.frame(cbind(calculate_IS(w_test,MFA_full_predict$p.w.0),
                                       calculate_IS(w_test,MFA_LR_predict$p.w.0),
                                       calculate_IS(w_test,NNGP_full_m3_predict$p.w.0),
                                       calculate_IS(w_test,NNGP_joint_m3_predict$p.w.0),
                                       calculate_IS(w_test,p.s$p.w.0)
                                       ))
names(interval_score_data) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                               "spNNGP")


w_IS_mean = colMeans(interval_score_data)
w_IS_mean
w_MSE = c(mean(scoringutils::se_mean_sample(w_test, MFA_full_predict$p.w.0)),
          mean(scoringutils::se_mean_sample(w_test, MFA_LR_predict$p.w.0)),
          mean(scoringutils::se_mean_sample(w_test, NNGP_full_m3_predict$p.w.0)),
          mean(scoringutils::se_mean_sample(w_test, NNGP_joint_m3_predict$p.w.0)),
          mean(scoringutils::se_mean_sample(w_test, p.s$p.w.0)))

names(w_MSE) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                 "spNNGP")
w_MSE
w_coverage = c(calculate_coverage(w_test,MFA_full_predict$p.w.0),
               calculate_coverage(w_test,MFA_LR_predict$p.w.0),
               calculate_coverage(w_test,NNGP_full_m3_predict$p.w.0),
               calculate_coverage(w_test,NNGP_joint_m3_predict$p.w.0),
               calculate_coverage(w_test,p.s$p.w.0))
w_coverage
names(w_coverage) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                      "spNNGP")

w_crps = c(mean(crps_sample(w_test, MFA_full_predict$p.w.0)),
           mean(crps_sample(w_test, MFA_LR_predict$p.w.0)),
           mean(crps_sample(w_test, NNGP_full_m3_predict$p.w.0)),
           mean(crps_sample(w_test, NNGP_joint_m3_predict$p.w.0)),
           mean(crps_sample(w_test, p.s$p.w.0)))

names(w_crps) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                  "spNNGP")
w_crps

#### y ####
interval_score_data_y=data.frame(cbind(calculate_IS(y_test,MFA_full_predict$p.y.0),
                                       calculate_IS(y_test,MFA_LR_predict$p.y.0),
                                       calculate_IS(y_test,NNGP_full_m3_predict$p.y.0),
                                       calculate_IS(y_test,NNGP_joint_m3_predict$p.y.0),
                                       calculate_IS(y_test,p.s$p.y.0)
))
names(interval_score_data_y) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                                 "spNNGP")


y_IS_mean = colMeans(interval_score_data_y)
y_MSE = c(mean(se_mean_sample(y_test, MFA_full_predict$p.y.0)),
          mean(se_mean_sample(y_test, MFA_LR_predict$p.y.0)),
          mean(se_mean_sample(y_test, NNGP_full_m3_predict$p.y.0)),
          mean(se_mean_sample(y_test, NNGP_joint_m3_predict$p.y.0)),
          mean(se_mean_sample(y_test, p.s$p.y.0)))

names(y_MSE) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                "spNNGP")

y_coverage = c(calculate_coverage(y_test,MFA_full_predict$p.y.0),
               calculate_coverage(y_test,MFA_LR_predict$p.y.0),
               calculate_coverage(y_test,NNGP_full_m3_predict$p.y.0),
               calculate_coverage(y_test,NNGP_joint_m3_predict$p.y.0),
               calculate_coverage(y_test,p.s$p.y.0))

names(y_coverage) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                      "spNNGP")

y_crps = c(mean(crps_sample(y_test, MFA_full_predict$p.y.0)),
           mean(crps_sample(y_test, MFA_LR_predict$p.y.0)),
           mean(crps_sample(y_test, NNGP_full_m3_predict$p.y.0)),
           mean(crps_sample(y_test, NNGP_joint_m3_predict$p.y.0)),
           mean(crps_sample(y_test, p.s$p.y.0)))

names(y_crps) = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                  "spNNGP")


third_elements <- c("interval_score", "crps","MSE", "coverage")
second_elements <- c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP joint",
                     "spNNGP")

combination_w <- apply(expand.grid("w", second_elements, third_elements), 1, function(x) paste(x, collapse = "_"))
combination_y <- apply(expand.grid("y", second_elements, third_elements), 1, function(x) paste(x, collapse = "_"))

output_vector = c(t,n_train[n_index],
                  w_IS_mean,w_crps,w_MSE,w_coverage,
                  y_IS_mean,y_crps,y_MSE,y_coverage)

names(output_vector) = c("t","n",
                         combination_w,combination_y)

file_name_output_vector = paste0(output.path, "/VI_NNGP_pred_vector","_t",t,"_n",n_train[n_index],".csv")

write.csv(output_vector,file_name_output_vector,row.names = FALSE)



output_pred_list= cbind(rep(t,length(y_test)),
                   rep(n_test[n_index],length(y_test)),
                   y_test,
                   rowMeans(NNGP_full_m3_predict$p.y.0),
                   rowMeans(NNGP_joint_m3_predict$p.y.0),
                   rowMeans(MFA_full_predict$p.y.0),
                   rowMeans(MFA_LR_predict$p.y.0),
                   rowMeans(p.s$p.y.0),
                   w_test,
                   rowMeans(NNGP_full_m3_predict$p.w.0),
                   rowMeans(NNGP_joint_m3_predict$p.w.0),
                   rowMeans(MFA_full_predict$p.w.0),
                   rowMeans(MFA_LR_predict$p.w.0),
                   rowMeans(p.s$p.w.0),
                   seq(1,length(y_test),1))

colnames(output_pred_list) = c("t","n",
                               "y_test",
                          "NNGP_m3_y_test","NNGP_joint_m3_y_test",
                          "MFA_y_test", "MFA_LR_y_test",
                          "spNNGP_y_test",
                          "w_test",
                          "NNGP_m3_w_test","NNGP_joint_m3_w_test",
                          "MFA_w_test", "MFA_LR_w_test",
                          "spNNGP_w_test",
                          "index")
file_name_output_pred_list   = paste0(output.path,"/VI_NNGP_pred_list","_t",t,"_n",n_train[n_index],".csv")
write.csv(output_pred_list,file_name_output_pred_list,row.names = FALSE)


