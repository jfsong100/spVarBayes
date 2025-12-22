library(tidyr)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)
######################################
# Output path
######################################
base_dir  = file.path(getwd())
output.path  = file.path(base_dir,"figures")
if(!dir.exists(output.path)){
  dir.create(output.path)
}

##################################################
# Input spVB and spNNGP results from R_results
##################################################
file_list= list.files(file.path(base_dir,  "R_results"))
VI_output_list = file_list[grepl("VI_NNGP_output_list_t", file_list, fixed = TRUE)]
VI_output_list_beta = file_list[grepl("VI_NNGP_output_beta_vector", file_list, fixed = TRUE)]
VI_output_vector_list = file_list[grepl("VI_NNGP_output_vector_t", file_list, fixed = TRUE)]
VI_output_list_theta = file_list[grepl("VI_NNGP_output_theta_vector", file_list, fixed = TRUE)]
output_vector_CI_w = file_list[grepl("VI_NNGP_output_vector_CI_w", file_list, fixed = TRUE)]
output_vector_CI_beta = file_list[grepl("VI_NNGP_output_vector_CI_beta", file_list, fixed = TRUE)]

output_vector_CI_w_coverage_list = matrix(NA,ncol=7)
output_vector_CI_w_is_list = matrix(NA,ncol=7)
output_vector_CI_w_crps_list = matrix(NA,ncol=7)

output_vector_CI_beta1_coverage_list = matrix(NA,ncol=7)
output_vector_CI_beta1_is_list = matrix(NA,ncol=7)
output_vector_CI_beta1_crps_list = matrix(NA,ncol=7)

output_vector_CI_beta2_coverage_list = matrix(NA,ncol=7)
output_vector_CI_beta2_is_list = matrix(NA,ncol=7)
output_vector_CI_beta2_crps_list = matrix(NA,ncol=7)

output_list_t = matrix(NA,ncol=13)
output_list_total = matrix(NA,ncol=18)
output_list_total_beta1 = matrix(NA,ncol=7)
output_list_total_beta2 = matrix(NA,ncol=7)

output_list_total_sigmasq = matrix(NA,ncol=7)
output_list_total_tausq = matrix(NA,ncol=7)

colnames(output_list_total_beta1) = 
  colnames(output_list_total_beta2) = 
  colnames(output_list_total_sigmasq) = 
  colnames(output_list_total_tausq) = 
  c("t","n","NNGP","NNGP_joint",
    "MFA","MFA_LR","spNNGP")


colnames(output_list_total) = c("t","n","Trace_MC","phi_input","empirical_mu","spNNGP_what",
                                "NNGP_m3_mu","NNGP_m3_joint_mu",
                                "MFA_mu","MFA_LR_mu",
                                "empirical_var",
                                "NNGP_m3_var","NNGP_m3_joint_var",
                                "MFA_var","MFA_LR_var","spNNGP_var",
                                "order",
                                "index")

NNGPVI_path = file.path(base_dir,  "R_results")
setwd(NNGPVI_path)
for (i in 1:length(VI_output_vector_list)) {
  filename = VI_output_vector_list[i]
  n = as.integer(as.numeric(str_match(filename, "_n([^.]*)")[, 2]))
  output_list_t_sub = unlist(read.csv(VI_output_vector_list[i]))
  output_list_t_sub[2] = n
  
  output_list_t = rbind(output_list_t,t(output_list_t_sub))
  sub_data_NNGP = read.csv(VI_output_list[i])
  sub_data_NNGP$n = n
  sub_data_NNGP$empirical_mu = sub_data_NNGP$empirical_mu[sub_data_NNGP$order]
  sub_data_NNGP$empirical_var = sub_data_NNGP$empirical_var[sub_data_NNGP$order]
  sub_data_NNGP$spNNGP_what = sub_data_NNGP$spNNGP_what[sub_data_NNGP$order]
  sub_data_NNGP$spNNGP_var = sub_data_NNGP$spNNGP_var[sub_data_NNGP$order]
  sub_data_NNGP$MFA_LR_mu = sub_data_NNGP$MFA_LR_mu[sub_data_NNGP$order]
  sub_data_NNGP$MFA_LR_var = sub_data_NNGP$MFA_LR_var[sub_data_NNGP$order]
  
  output_list_total = rbind(output_list_total,sub_data_NNGP)
  
  output_list_total_beta1 = rbind(output_list_total_beta1,read.csv(VI_output_list_beta[i])[1,])
  output_list_total_beta2 = rbind(output_list_total_beta2,read.csv(VI_output_list_beta[i])[2,])
  
  output_list_total_sigmasq = rbind(output_list_total_sigmasq,read.csv(VI_output_list_theta[i])[1,])
  output_list_total_tausq = rbind(output_list_total_tausq,read.csv(VI_output_list_theta[i])[2,])
  
  filename = output_vector_CI_w[i]
  t = str_match(filename, "_t(\\d+)")[,2] %>% as.integer()
  n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
  ci_w_sub = unlist(read.csv(output_vector_CI_w[i]))
  output_vector_CI_w_coverage_list = rbind(output_vector_CI_w_coverage_list,c(t,n,read.csv(output_vector_CI_w[i])[,1]))
  output_vector_CI_w_is_list = rbind(output_vector_CI_w_is_list,c(t,n,read.csv(output_vector_CI_w[i])[,2]))
  output_vector_CI_w_crps_list = rbind(output_vector_CI_w_crps_list,c(t,n,read.csv(output_vector_CI_w[i])[,3]))
  
  filename = output_vector_CI_beta[i]
  t = str_match(filename, "_t(\\d+)")[,2] %>% as.integer()
  n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
  output_vector_CI_beta1_coverage_list = rbind(output_vector_CI_beta1_coverage_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[1,])))
  output_vector_CI_beta1_is_list = rbind(output_vector_CI_beta1_is_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[2,])))
  output_vector_CI_beta1_crps_list = rbind(output_vector_CI_beta1_crps_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[3,])))
  
  output_vector_CI_beta2_coverage_list = rbind(output_vector_CI_beta2_coverage_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[4,])))
  output_vector_CI_beta2_is_list = rbind(output_vector_CI_beta2_is_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[5,])))
  output_vector_CI_beta2_crps_list = rbind(output_vector_CI_beta2_crps_list,c(t,n,as.numeric(read.csv(output_vector_CI_beta[i])[6,])))
  

}

##################################################
# Input DKLGP_default and DKLGP results from DKLGP_results
##################################################
file_list_DKL= list.files(file.path(base_dir,  "DKLGP_results"))
KL_list_DKL = c(file_list_DKL[grepl("KL_vec_VIVA_n", file_list_DKL, fixed = TRUE)])
VI_output_list_DKL = c(file_list_DKL[grepl("output_data_VIVA_n", file_list_DKL, fixed = TRUE)])

file_list_DKL_default= list.files(file.path(base_dir,  "DKLGP_results"))
KL_list_DKL_default = file_list_DKL_default[grepl("KL_vec_VIVA_default", file_list_DKL_default, fixed = TRUE)]
VI_output_list_DKL_default = file_list_DKL_default[grepl("output_data_VIVA_default", file_list_DKL_default, fixed = TRUE)]

output_list_t_DKL = matrix(NA,ncol=9)
output_list_total_DKL = matrix(NA,ncol=7)

colnames(output_list_total_DKL) = c("t","n","index","empirical_mu","empirical_var",
                                 "mu_post","var_post")

setwd(file.path(base_dir,  "DKLGP_results"))
for (i in 1:length(KL_list_DKL)) {

  if(file.exists(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",read.csv(KL_list_DKL[i])[1],"_n",read.csv(KL_list_DKL[i])[2],".csv"))){
    
    output_list_t_DKL = rbind(output_list_t_DKL,c((read.csv(KL_list_DKL[i])[1:2]%>% as.numeric()),
                                                  read.csv(KL_list_DKL[i])[3] %>% 
                                                    str_extract("[0-9]+\\.?[0-9]+") %>% 
                                                    as.numeric(),
                                                  read.csv(KL_list_DKL[i])[4:9]%>% as.numeric()))
    
    order = read.csv(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",read.csv(KL_list_DKL[i])[1],"_n",read.csv(KL_list_DKL[i])[2],".csv"))$order
    index = read.csv(VI_output_list_DKL[i])$index
    sub_data = read.csv(VI_output_list_DKL[i])[order,]
    sub_data$index = index
    output_list_total_DKL = rbind(output_list_total_DKL,sub_data)
  }

}

output_list_t_DKL_default = matrix(NA,ncol=9)
output_list_total_DKL_default = matrix(NA,ncol=7)

colnames(output_list_total_DKL_default) = c("t","n","index","empirical_mu","empirical_var",
                                            "mu_post","var_post")
options(warn = 1)
for (i in 1:length(KL_list_DKL_default)) {
  if(file.exists(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",read.csv(KL_list_DKL_default[i])[1],"_n",read.csv(KL_list_DKL_default[i])[2],".csv"))){
    output_list_t_DKL_default = rbind(output_list_t_DKL_default,
                                      c(read.csv(KL_list_DKL_default[i])[1:2]%>% as.numeric(),
                                        read.csv(KL_list_DKL_default[i])[3] %>% 
                                          str_extract("[0-9]+\\.?[0-9]+") %>% 
                                          as.numeric(),
                                        read.csv(KL_list_DKL_default[i])[4:9]%>% as.numeric()))
    
    
    order = read.csv(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",read.csv(KL_list_DKL_default[i])[1],"_n",read.csv(KL_list_DKL_default[i])[2],".csv"))$order
    index = read.csv(paste0("output_data_VIVA_default","_n",read.csv(KL_list_DKL_default[i])[2],"_d2_seed",read.csv(KL_list_DKL_default[i])[1],".csv"))$index
    sub_data = read.csv(paste0("output_data_VIVA_default","_n",read.csv(KL_list_DKL_default[i])[2],"_d2_seed",read.csv(KL_list_DKL_default[i])[1],".csv"))[order,]
    sub_data$index = index
    output_list_total_DKL_default = rbind(output_list_total_DKL_default,sub_data)
    
  }
  
}

##################################################
# Input VNNGP results from VNNGP_results
##################################################
file_list_VNNGP= list.files(file.path(base_dir,  "VNNGP_results"))
KL_list_VNNGP = file_list_VNNGP[grepl("KL_vec_VNNGP", file_list_VNNGP, fixed = TRUE)]
VI_output_list_VNNGP = file_list_VNNGP[grepl("output_data_VNNGP", file_list_VNNGP, fixed = TRUE)]

output_list_t_VNNGP = matrix(NA,ncol=9)
output_list_total_VNNGP = matrix(NA,ncol=7)

colnames(output_list_total_VNNGP) = c("t","n","index","empirical_mu","empirical_var",
                                      "mu_post","var_post")
options(warn = 1)
setwd(file.path(base_dir,  "VNNGP_results"))

for (i in 1:length(KL_list_VNNGP)) {
  VNNGP_data = read_delim(KL_list_VNNGP[i])
  filename = KL_list_VNNGP[i]
  t = str_match(filename, "_seed(\\d+)")[,2] %>% as.integer()
  n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
  
  find_files = file.exists(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",t,"_n",n,".csv"))
  print(paste0("find is ",find_files))
  if(file.exists(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",t,"_n",n,".csv"))){
    
    output_list_t_VNNGP = rbind(output_list_t_VNNGP,
                                c(VNNGP_data[1]%>% as.numeric(),n,
                                  VNNGP_data[3] %>% 
                                    str_extract("[0-9]+\\.?[0-9]+") %>% 
                                    as.numeric(),
                                  VNNGP_data[4:9]%>% as.numeric()))
    
    order = read.csv(paste0(NNGPVI_path,"/","VI_NNGP_output_list","_t",VNNGP_data[1],"_n",n,".csv"))$order
    index = read_delim(paste0("output_data_VNNGP","_n",n,"_d2_seed",VNNGP_data[1],".txt"))$index
    sub_data = read_delim(paste0("output_data_VNNGP","_n",n,"_d2_seed",VNNGP_data[1],".txt"))[order,]
    
    sub_data$index = index
    output_list_total_VNNGP = rbind(output_list_total_VNNGP,sub_data)
  }
}

##################################################
# Combine all methods results
##################################################
#### approximated mean and variance for spatial random effects #### 
output_list_total =  output_list_total[!is.na(output_list_total[,1]),]
output_list_total_DKL = output_list_total_DKL[!is.na(output_list_total_DKL[,1]),]
output_list_total_DKL_default = output_list_total_DKL_default[!is.na(output_list_total_DKL_default[,1]),]
output_list_total_VNNGP = output_list_total_VNNGP[!is.na(output_list_total_VNNGP[,1]),]

colnames(output_list_total_DKL) = c("t","n","index","empirical_mu","empirical_var","DKL_mu","DKL_var")
colnames(output_list_total_DKL_default) = c("t","n","index","empirical_mu","empirical_var","DKL_default_mu","DKL_default_var")
colnames(output_list_total_VNNGP) = c("t","n","index","empirical_mu","empirical_var","VNNGP_mu","VNNGP_var")

df1 = output_list_total
df2 = output_list_total_DKL[,c("t", "n", "index", "DKL_mu", "DKL_var")]
df3 = output_list_total_DKL_default[,c("t", "n", "index", "DKL_default_mu", "DKL_default_var")]
df4 = output_list_total_VNNGP[,c("t", "n", "index", "VNNGP_mu", "VNNGP_var")]

dfs_to_merge = list(df1, df2, df3, df4)

output_list_total_sub = Reduce(function(x, y) merge(x, y, by = c("t", "n", "index"),all = TRUE), dfs_to_merge)

#### metrics for spatial random effects #### 
colnames(output_list_t) = c("t","n","Trace_MC","phi_input",
                            "MFA","MFA_LR",
                            "NNGP","NNGP_joint",
                            "MFA_used_time","MFA_LR_used_time",
                            "NNGP_used_time","NNGP_joint_mb_used_time",
                            "spNNGP_time")

colnames(output_list_t_DKL) = c("t","n","DKL","DKL_time","DKL_sigmasq","DKL_tausq","DKL_coverage","DKL_is","DKL_crps")
colnames(output_list_t_DKL_default) = c("t","n","DKL_default","DKL_default_time","DKL_default_sigmasq","DKL_default_tausq","DKL_default_coverage","DKL_default_is","DKL_default_crps")
colnames(output_list_t_VNNGP) = c("t","n","VNNGP","VNNGP_time","VNNGP_sigmasq","VNNGP_tausq","VNNGP_coverage","VNNGP_is","VNNGP_crps")

output_list_t_combine = list(output_list_t, output_list_t_DKL, 
                              output_list_t_DKL_default,output_list_t_VNNGP)

list_of_dfs = list(unique(output_list_t_DKL[, c("t", "n")]), 
                    unique(output_list_t_DKL_default[, c("t", "n")]), 
                    unique(output_list_t[, c("t", "n")]),
                    unique(output_list_t_VNNGP[, c("t", "n")]))

intersect_rows = Reduce(function(x, y) merge(x, y, by = c("t", "n")), list_of_dfs)

merged_data_frames = lapply(output_list_t_combine, function(df) {
  merge(df, intersect_rows, by = c("t", "n"))
})

output_list_all = Reduce(function(x, y) merge(x, y, by = c("t", "n"),all = TRUE), output_list_t_combine)

output_list_intersect = Reduce(function(x, y) merge(x, y, by = c("t", "n")), output_list_t_combine)

output_list_intersect = output_list_intersect[is.na(output_list_intersect$n)==F,]

df_kl_base = as.data.frame(output_list_t) %>% filter(!is.na(n)) %>% select(t, n, NNGP,NNGP_joint,MFA,MFA_LR)
df_kl_dkl = as.data.frame(output_list_t_DKL) %>% filter(!is.na(n))%>% select(t, n, DKL)
df_kl_dkl_default = as.data.frame(output_list_t_DKL_default) %>% filter(!is.na(n))%>% select(t, n, DKL_default)
df_kl_vnngp = as.data.frame(output_list_t_VNNGP) %>% filter(!is.na(n))%>% select(t, n, VNNGP)

df_kl_all = df_kl_base %>%
  full_join(df_kl_dkl, by = c("t", "n")) %>%
  full_join(df_kl_dkl_default, by = c("t", "n")) %>%
  full_join(df_kl_vnngp, by = c("t", "n"))

colnames(df_kl_all) = c("t", "n", "spVB-NNGP", "spVB-NNGP-joint", "spVB-MFA", "spVB-MFA-LR",
                         "DKLGP", "DKLGP-default", "VNNGP")

kl_long = df_kl_all %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "value") %>%
  mutate(metric = "KL Divergence")

colnames(output_vector_CI_w_coverage_list) = 
  colnames(output_vector_CI_w_is_list) = 
  colnames(output_vector_CI_w_crps_list) = c("t","n","NNGP","NNGP joint","spNNGP","MFA","MFA LR")

df_cov_base = as.data.frame(output_vector_CI_w_coverage_list)
df_cov_dkl = as.data.frame(output_list_t_DKL) %>% select(t, n, DKL_coverage)
df_cov_dkl_default = as.data.frame(output_list_t_DKL_default) %>% select(t, n, DKL_default_coverage)
df_cov_vnngp = as.data.frame(output_list_t_VNNGP) %>% select(t, n, VNNGP_coverage)

df_cov_all = df_cov_base %>%
  full_join(df_cov_dkl, by = c("t", "n")) %>%
  full_join(df_cov_dkl_default, by = c("t", "n")) %>%
  full_join(df_cov_vnngp, by = c("t", "n"))

colnames(df_cov_all) = c("t", "n", "spVB-NNGP", "spVB-NNGP-joint", "spNNGP", "spVB-MFA", "spVB-MFA-LR",
                          "DKLGP", "DKLGP-default", "VNNGP")

coverage_long = df_cov_all %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "value") %>%
  mutate(metric = "Coverage")


df_is_base = as.data.frame(output_vector_CI_w_is_list)
df_is_dkl = as.data.frame(output_list_t_DKL) %>% select(t, n, DKL_is)
df_is_dkl_default = as.data.frame(output_list_t_DKL_default) %>% select(t, n, DKL_default_is)
df_is_vnngp = as.data.frame(output_list_t_VNNGP) %>% select(t, n, VNNGP_is)

df_is_all = df_is_base %>%
  full_join(df_is_dkl, by = c("t", "n")) %>%
  full_join(df_is_dkl_default, by = c("t", "n")) %>%
  full_join(df_is_vnngp, by = c("t", "n"))

colnames(df_is_all) = c("t", "n", "spVB-NNGP", "spVB-NNGP-joint", "spNNGP", "spVB-MFA", "spVB-MFA-LR",
                         "DKLGP", "DKLGP-default", "VNNGP")

is_long = df_is_all %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "value") %>%
  mutate(metric = "Interval Score")

df_crps_base = as.data.frame(output_vector_CI_w_crps_list)
df_crps_dkl = as.data.frame(output_list_t_DKL) %>% select(t, n, DKL_crps)
df_crps_dkl_default = as.data.frame(output_list_t_DKL_default) %>% select(t, n, DKL_default_crps)
df_crps_vnngp = as.data.frame(output_list_t_VNNGP) %>% select(t, n, VNNGP_crps)

df_crps_all = df_crps_base %>%
  full_join(df_crps_dkl, by = c("t", "n")) %>%
  full_join(df_crps_dkl_default, by = c("t", "n")) %>%
  full_join(df_crps_vnngp, by = c("t", "n"))

colnames(df_crps_all) = c("t", "n", "spVB-NNGP", "spVB-NNGP-joint", "spNNGP", "spVB-MFA", "spVB-MFA-LR",
                           "DKLGP", "DKLGP-default", "VNNGP")

crps_long = df_crps_all %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "value") %>%
  mutate(metric = "CRPS")

#### metrics for fixed effects #### 
colnames(output_vector_CI_beta1_coverage_list) = 
  colnames(output_vector_CI_beta1_is_list) = 
  colnames(output_vector_CI_beta1_crps_list) = c("t","n","NNGP","NNGP joint","spNNGP","MFA","MFA LR")

colnames(output_vector_CI_beta2_coverage_list) = 
  colnames(output_vector_CI_beta2_is_list) = 
  colnames(output_vector_CI_beta2_crps_list) = c("t","n","NNGP","NNGP joint","spNNGP","MFA","MFA LR")



##################################################
# Save results if needed
##################################################
# save.image(file.path(base_dir,  "summary_data.RData"))


##################################################
# Create plots
##################################################
library(tidyr)
library(ggrastr)
library(ggplot2)

custom_labeller_2 = function(variables) {
  
  if(any(grepl("-NNGP", variables))) {
    label1 = lapply(variables, function(x) {
      sapply(x, function(value) {
        parts = strsplit(value, "_")[[1]]
        if (length(parts) == 2) {
          paste(parts[1], "\n", parts[2])  # Using newline character
        } else {
          value  # Return the original value if it does not contain "_"
        }
      })
    })
  } else {
    label1 = lapply(variables, function(x) {
      sapply(x, function(value) {
        paste("n =", value)  # Add "n = " in front of each number
      })
    })
  }
  
  return(label1)
}
output_list_total_wide = output_list_total_sub[is.na(output_list_total_sub$n)==F, c("n","spNNGP_what",
                                                   "NNGP_m3_mu","NNGP_m3_joint_mu",
                                                   "MFA_mu","MFA_LR_mu","DKL_mu","DKL_default_mu","VNNGP_mu")]
colnames(output_list_total_wide) =c("n","estimated.w.spNNGP",
                                    "estimated.w.NNGP.m3","estimated.w.NNGP.joint.m3",
                                    "estimated.w.MFA","estimated.w.MFA.LR","estimated.w.DKL","estimated.w.DKL.default","estimated.w.VNNGP")

output_list_total_long = output_list_total_wide %>% pivot_longer(cols=c("estimated.w.NNGP.m3","estimated.w.NNGP.joint.m3",
                                                                                  "estimated.w.MFA","estimated.w.MFA.LR",
                                                                                  "estimated.w.DKL","estimated.w.DKL.default",
                                                                                  "estimated.w.VNNGP"),
                                                                           names_to='model',
                                                                           values_to='estimated w')

output_list_total_long$model = factor(output_list_total_long$model,
                                      levels = c("estimated.w.MFA","estimated.w.MFA.LR","estimated.w.VNNGP",
                                                 "estimated.w.NNGP.m3","estimated.w.NNGP.joint.m3",
                                                 "estimated.w.DKL.default",'estimated.w.DKL',"estimated.w.spNNGP"),
                                      labels = c("spVB-MFA","spVB-MFA-LR","VNNGP",
                                                 "spVB-NNGP","spVB-NNGP-joint",
                                                 "DKLGP-default","DKLGP","spNNGP"))


output_list_totalV = output_list_total_sub[, c("t","n",
                                               "NNGP_m3_var","NNGP_m3_joint_var",
                                               "MFA_var","MFA_LR_var","spNNGP_var","DKL_var","DKL_default_var","VNNGP_var")]
colnames(output_list_totalV) = c("t","n",
                                 "estimated.var.w.NNGP.m3","estimated.var.w.NNGP.joint.m3",
                                 "estimated.var.w.MFA" ,
                                 "estimated.var.w.MFA.LR" ,
                                 "estimated.var.w.spNNGP",
                                 "estimated.var.w.DKL" ,
                                 "estimated.var.w.DKL.default",
                                 "estimated.var.w.VNNGP")

output_list_totalV_long = output_list_totalV %>% pivot_longer(cols=c("estimated.var.w.NNGP.m3","estimated.var.w.NNGP.joint.m3",
                                                                      "estimated.var.w.MFA" ,
                                                                      "estimated.var.w.MFA.LR" ,
                                                                      "estimated.var.w.DKL" ,
                                                                      "estimated.var.w.DKL.default",
                                                                      "estimated.var.w.VNNGP"),
                                                               names_to='model',
                                                               values_to='estimated var w')

output_list_totalV_long$model = factor(output_list_totalV_long$model,
                                       levels = c("estimated.var.w.MFA","estimated.var.w.MFA.LR",
                                                  "estimated.var.w.VNNGP",
                                                  "estimated.var.w.NNGP.m3","estimated.var.w.NNGP.joint.m3",
                                                  'estimated.var.w.DKL.default','estimated.var.w.DKL',
                                                  "estimated.var.w.spNNGP"),
                                       labels = c("spVB-MFA","spVB-MFA-LR","VNNGP",
                                                  "spVB-NNGP","spVB-NNGP-joint",
                                                  "DKLGP-default","DKLGP","spNNGP"))
##################################################
# Create Figure 1 (a)
##################################################
p1a = ggplot(output_list_total_long %>% filter(n==10000)) + 
  geom_point(aes(`estimated.w.spNNGP`,`estimated w`), alpha = 0.3)+
  facet_grid(~model,labeller = as_labeller(custom_labeller_2)) + 
  geom_abline(intercept = 0, color = "red", size = 0.3)+ 
  theme(legend.position = "none")+
  labs(x = "MCMC Estimated Posterior Mean for w (spNNGP)", y = "Approximated Posterior Mean for w") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(aspect.ratio = 1)+
  theme(strip.text = element_text(size = 15))+
  theme(plot.title = element_text(size = 15))+
  theme(
    axis.title.x = element_text(size = 15),  
    axis.text.x  = element_text(size = 15),
    axis.title.y  = element_text(size = 15),
    axis.text.y  = element_text(size = 15) 
  )

p1a$layers[[1]] = rasterise(p1a$layers[[1]], dpi = 300, scale = 1)

ggsave(file.path(output.path,"Figure_1a_w_mean_sub.pdf"), plot = p1a,
       width = 18, height = 4.78, units = "in", 
       device = cairo_pdf)

##################################################
# Create Figure 1 (b)
##################################################
p1b = ggplot(output_list_totalV_long %>% filter(n==10000)) + 
  geom_point(aes(`estimated.var.w.spNNGP`,`estimated var w`), alpha = 0.3)+
  facet_grid(~model,labeller = as_labeller(custom_labeller_2), scales = "free_y") + 
  geom_abline(intercept = 0, color = "red", size = 0.3)+ 
  theme(legend.position = "none")+
  labs(x = "MCMC Estimated Posterior Variance for w (spNNGP)", y = "Approximated Posterior Variance for w") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(aspect.ratio = 1)+
  theme(strip.text = element_text(size = 15))+
  theme(plot.title = element_text(size = 15))+
  theme(
    axis.title.x = element_text(size = 15),  
    axis.text.x  = element_text(size = 15),
    axis.title.y  = element_text(size = 15),
    axis.text.y  = element_text(size = 15) 
  )

p1b$layers[[1]] = rasterise(p1b$layers[[1]], dpi = 300, scale = 1)

ggsave(file.path(output.path,"Figure_1b_w_var_sub.pdf"), plot = p1b,
       width = 18, height = 4.78, units = "in", 
       device = cairo_pdf)

##################################################
# Create Figure H1
##################################################
p1 = ggplot(output_list_total_long) + 
  geom_point(aes(`estimated.w.spNNGP`,`estimated w`), alpha = 0.3)+
  facet_grid(n~model,labeller = as_labeller(custom_labeller_2)) + 
  geom_abline(intercept = 0, color = "red", size = 0.3)+ 
  theme(legend.position = "none")+
  labs(x = "MCMC Estimated Posterior Mean for w (spNNGP)", y = "Approximated Posterior Mean for w") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(aspect.ratio = 1)+
  theme(strip.text = element_text(size = 10))+
  theme(plot.title = element_text(size = 11))

p1$layers[[1]] = rasterise(p1$layers[[1]], dpi = 300)

ggsave(file.path(output.path,"Figure_H1_w_mean_all.pdf"), plot = p1,
       width = 14, height = 8.64, units = "in",
       dpi = 300, device = cairo_pdf)

##################################################
# Create Figure H2
##################################################
p2 = ggplot(output_list_totalV_long) + 
  geom_point(aes(`estimated.var.w.spNNGP`,`estimated var w`), alpha = 0.3)+
  facet_grid(n~model,labeller = as_labeller(custom_labeller_2), scales = "free_y") + 
  geom_abline(intercept = 0, color = "red", size = 0.3)+ 
  theme(legend.position = "none")+
  labs(x = "MCMC Estimated Posterior Variance for w (spNNGP)", y = "Approximated Posterior Variance for w") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(aspect.ratio = 1)+
  theme(strip.text = element_text(size = 10))+
  theme(plot.title = element_text(size = 11))

p2$layers[[1]] = rasterise(p2$layers[[1]], dpi = 300)

ggsave(file.path(output.path,"Figure_H1_w_var_all.pdf"), plot = p2,
       width = 14, height = 8.64, units = "in",
       dpi = 300, device = cairo_pdf)


##################################################
# Create Figure 2, Figure H3,H4,H5,H6
##################################################
library(ggplot2)
library(cowplot) 
library(patchwork)

method_colors = c(
  "spVB-MFA" = "#b4d9b8", 
  "spVB-MFA.mb" = "#66c2a5",
  "spVB-MFA-LR" = "#00bfc4",
  "VNNGP" = "#01665e",
  "spVB-NNGP m=1" = "#5e3a8c",  
  "spVB-NNGP.mb m=1" = "#b39ddb", 
  "spVB-NNGP" = "#2171b5",
  "spVB-NNGP-joint" = "#264778",
  "spVB-NNGP.mb m=3" = "#6baed6",
  "spVB-NNGP m=5" = "#3b4994",
  "spVB-NNGP.mb m=5" = "#7986cb",
  "spNNGP" = "#d7301f",
  "DKLGP" = "#e6ab02",
  "DKLGP-default" = "#a6761d",
  "INFVB" = "darksalmon"
)

# Define method mapping
method_labels = c(
  "MFA" = "spVB-MFA",
  "MFA LR" = "spVB-MFA-LR",
  "NNGP" = "spVB-NNGP",
  "NNGP joint" = "spVB-NNGP-joint",
  "spNNGP" = "spNNGP",
  "DKL" = "DKLGP",
  "DKL_default" = "DKLGP-default",
  "VNNGP" = "VNNGP"
)

all_metrics_long = bind_rows(coverage_long, is_long, crps_long, kl_long)
all_metrics_long = all_metrics_long %>% filter(!is.na(n) & n <= 10000)
all_metrics_long$method = factor(all_metrics_long$method, 
                                 levels = c("spVB-MFA","spVB-MFA-LR","VNNGP",
                                            "spVB-NNGP", "spVB-NNGP-joint",
                                            "DKLGP-default","DKLGP","spNNGP"))

metrics_main = all_metrics_long %>%
  filter(metric %in% c("KL Divergence", "CRPS")) %>%
  filter(n == 10000)

kl_plot = metrics_main %>%
  filter(metric == "KL Divergence", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  coord_cartesian(ylim = c(0, 6000)) +
  labs(x = "Method", y = "KL Divergence") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
    strip.text = element_text(size = 12),
    axis.title.x  = element_text(size = 14),
    axis.title.y  = element_text(size = 14),
    strip.text.x  = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)


crps_plot = metrics_main %>%
  filter(metric == "CRPS", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  labs(x = "Method", y = "CRPS") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
    strip.text = element_text(size = 12),
    axis.title.x  = element_text(size = 14),
    axis.title.y  = element_text(size = 14),
    strip.text.x  = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)

combined_plot = kl_plot + crps_plot +
  plot_layout(ncol = 2, guides = "collect") 

# combined_plot

ggsave(file.path(output.path,"Figure_2_training_metics_sub.pdf"), plot = combined_plot,
       width = 9.85, height = 5.7, units = "in", 
       device = cairo_pdf)

metrics_diagnostic = all_metrics_long %>%
  filter(metric %in% c("Coverage", "Interval Score"))

kl_all = all_metrics_long %>%
  filter(metric == "KL Divergence", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  coord_cartesian(ylim = c(0, 6000)) +
  labs(x = "Method", y = "KL Divergence") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    strip.text = element_text(size = 10),
    axis.title.x  = element_text(size = 12),
    axis.title.y  = element_text(size = 12),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)

ggsave(file.path(output.path,"Figure_H3_kl_all.pdf"), plot = kl_all,
       width = 9.75, height = 4.5, units = "in", 
       device = cairo_pdf)

crps_all = all_metrics_long %>%
  filter(metric == "CRPS", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  labs(x = "Method", y = "CRPS") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    strip.text = element_text(size = 10),
    axis.title.x  = element_text(size = 12),
    axis.title.y  = element_text(size = 12),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)

ggsave(file.path(output.path,"Figure_H4_crps_all.pdf"), plot = crps_all,
       width = 9.75, height = 4.5, units = "in", 
       device = cairo_pdf)

is_all = all_metrics_long %>%
  filter(metric == "Interval Score", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  labs(x = "Method", y = "Interval Score") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    strip.text = element_text(size = 10),
    axis.title.x  = element_text(size = 12),
    axis.title.y  = element_text(size = 12),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)

ggsave(file.path(output.path,"Figure_H5_is_all.pdf"), plot = is_all,
       width = 9.75, height = 4.5, units = "in", 
       device = cairo_pdf)


coverage_all = all_metrics_long %>%
  filter(metric == "Coverage", is.finite(value)) %>%
  ggplot(aes(x = method, y = value, color = method)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~ n, scales = "free_y", labeller = label_both) +
  labs(x = "Method", y = "95% Coverage") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    strip.text = element_text(size = 10),
    axis.title.x  = element_text(size = 12),
    axis.title.y  = element_text(size = 12),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_color_manual(values = method_colors)+
  geom_hline(data = data.frame(metric = "Coverage", yint = 0.95),
             aes(yintercept = yint), color = "red", linetype = "dashed", inherit.aes = FALSE)

ggsave(file.path(output.path,"Figure_H6_coverage_all.pdf"), plot = coverage_all,
       width = 9.75, height = 4.5, units = "in", 
       device = cairo_pdf)


##################################################
# Create Table 3
##################################################
method_recode <- c(
  MFA        = "spVB-MFA",
  MFA.LR     = "spVB-MFA-LR",
  NNGP       = "spVB-NNGP",
  NNGP.joint = "spVB-NNGP-joint",
  spNNGP     = "spNNGP"
)

output_vector_CI_beta1_coverage_list = data.frame(output_vector_CI_beta1_coverage_list)
output_vector_CI_beta1_coverage_list = output_vector_CI_beta1_coverage_list[!is.na(output_vector_CI_beta1_coverage_list$n),]

beta1_tab = output_vector_CI_beta1_coverage_list %>%
  filter(n <= 10000) %>%
  group_by(n) %>%
  summarise(across(c(NNGP, NNGP.joint, spNNGP, MFA, MFA.LR),
                   ~ mean(.x, na.rm = TRUE))) %>%
  pivot_longer(-n, names_to = "method_raw", values_to = "beta1")

output_vector_CI_beta2_coverage_list = data.frame(output_vector_CI_beta2_coverage_list)
output_vector_CI_beta2_coverage_list = output_vector_CI_beta2_coverage_list[!is.na(output_vector_CI_beta2_coverage_list$n),]

beta2_tab = output_vector_CI_beta2_coverage_list %>%
  filter(n <= 10000) %>%
  group_by(n) %>%
  summarise(across(c(NNGP, NNGP.joint, spNNGP, MFA, MFA.LR),
                   ~ mean(.x, na.rm = TRUE))) %>%
  pivot_longer(-n, names_to = "method_raw", values_to = "beta2")

res_tab = left_join(beta1_tab, beta2_tab,
                     by = c("n", "method_raw")) %>%
  mutate(Method = recode(method_raw, !!!method_recode),
         Method = factor(Method, levels = method_recode[c(
           "MFA", "MFA.LR", "NNGP", "NNGP.joint", "spNNGP"
         )])) %>%
  arrange(n, Method) %>%
  select(n, Method, beta1, beta2)

write.csv(res_tab, file.path(output.path,"beta_coverage.csv"), row.names = FALSE)

##################################################
# Create Figure 3
##################################################
mean_time = output_list_all[,c("t","n","NNGP_used_time","NNGP_joint_mb_used_time",
                                  "MFA_used_time","MFA_LR_used_time",
                                  "spNNGP_time",
                                  "DKL_time","DKL_default_time","VNNGP_time"
)] %>% filter(!is.na(n)) %>% group_by(n) %>% summarise(
  `spVB-NNGP` = mean(NNGP_used_time),
  `spVB-NNGP-joint` = mean(NNGP_joint_mb_used_time),
  `spVB-MFA` = mean(MFA_used_time),
  `spVB-MFA-LR` = mean(MFA_LR_used_time),
  spNNGP = mean(spNNGP_time),
  DKLGP = mean(DKL_time),
  `DKLGP-default` = mean(DKL_default_time),
  VNNGP = mean(VNNGP_time)
)

mean_time_long = mean_time %>%
  pivot_longer(
    cols      = -n,              # everything except n
    names_to  = "methods",       # column for method names
    values_to = "time"           # column for time values
  )
mean_time_long$n = factor(mean_time_long$n,levels = c(1000,5000,10000,50000,100000))
mean_time_long$methods = factor(mean_time_long$methods,levels = c("spVB-MFA","spVB-MFA-LR","VNNGP",
                                                                  "spVB-NNGP", "spVB-NNGP-joint",
                                                                  "DKLGP-default","DKLGP","spNNGP"))
time_sub = ggplot(mean_time_long %>% filter(n != "1000"), aes(x = n, y = time, fill = methods, group = methods)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = method_colors) +
  scale_color_manual(values = method_colors) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))+
  labs(x = "Sample Size", y = "Running Time (seconds)", fill = "Methods", color = "Methods") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_line(color = "gray85"),
    panel.grid.minor.x = element_blank(),
    axis.ticks.x = element_line(color = "black"),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9)
  )
# time_sub
ggsave(file.path(output.path,"Figure_3_time_sub.pdf"),  plot = time_sub,
       width = 9.75, height = 4.25, units = "in", 
       device = cairo_pdf)


##################################################
# Create Figure I1 I2
##################################################
methods = c("NNGP", "NNGP_joint", "MFA", "MFA_LR", "spNNGP", "DKL", "DKL_default", "VNNGP")
methods_sub = c("NNGP", "NNGP_joint", "MFA", "MFA_LR", "spNNGP")

tausq_total = output_list_total_tausq %>%
  select(t, n, all_of(methods_sub)) %>%
  pivot_longer(cols = all_of(methods_sub), names_to = "method", values_to = "tausq")

tausq_mean = output_list_intersect %>%
  select(t, n,
         DKL = DKL_tausq,
         DKL_default = DKL_default_tausq,
         VNNGP = VNNGP_tausq) %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "tausq")

tausq_df = bind_rows(tausq_total, tausq_mean)

tausq_df$method = recode(tausq_df$method,
                            "MFA" = "spVB-MFA",
                            "MFA_LR" = "spVB-MFA-LR",
                            "VNNGP" = "VNNGP",
                            "NNGP" = "spVB-NNGP",
                            "NNGP_joint" = "spVB-NNGP-joint",
                            "spNNGP" = "spNNGP",
                            "DKL" = "DKLGP",
                            "DKL_default" = "DKLGP-default"
)

# Factor order for plot
tausq_df$method = factor(tausq_df$method, levels = c(
  "spVB-MFA",
  "spVB-MFA-LR",
  "VNNGP",
  "spVB-NNGP",
  "spVB-NNGP-joint",
  "DKLGP-default",
  "DKLGP",
  "spNNGP"
))

tausq_df = tausq_df %>% filter(!is.na(tausq), n<= 10000)

tau_fig = ggplot(tausq_df, aes(x = method, y = tausq, fill = method)) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ n, scales = "free_y", labeller = labeller(n = function(n) paste0("n = ", n))) +
  geom_hline(yintercept = 0.5, color = "red", linetype = "dashed") +
  scale_fill_manual(values = method_colors) +
  labs(
    x = "Method",
    y = "Point estimation of tausq" ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 11),
    legend.position = "none"
  )
# tau_fig

ggsave(file.path(output.path,"Figure_I2_tausq_point.pdf"), plot = tau_fig,
       width = 9.82, height = 5.05, units = "in", 
       device = cairo_pdf)

# Extract relevant columns from output_list_total_sigmasq
sigmasq_total = output_list_total_sigmasq %>%
  select(t, n, all_of(methods_sub)) %>%
  pivot_longer(cols = all_of(methods_sub), names_to = "method", values_to = "sigmasq")

sigmasq_mean = output_list_intersect %>%
  select(t, n,
         DKL = DKL_sigmasq,
         DKL_default = DKL_default_sigmasq,
         VNNGP = VNNGP_sigmasq) %>%
  pivot_longer(cols = -c(t, n), names_to = "method", values_to = "sigmasq")

sigmasq_df = bind_rows(sigmasq_total, sigmasq_mean)

sigmasq_df$method = recode(sigmasq_df$method,
                            "MFA" = "spVB-MFA",
                            "MFA_LR" = "spVB-MFA-LR",
                            "VNNGP" = "VNNGP",
                            "NNGP" = "spVB-NNGP",
                            "NNGP_joint" = "spVB-NNGP-joint",
                            "spNNGP" = "spNNGP",
                            "DKL" = "DKLGP",
                            "DKL_default" = "DKLGP-default"
)

# Factor order for plot
sigmasq_df$method = factor(sigmasq_df$method, levels = c(
  "spVB-MFA",
  "spVB-MFA-LR",
  "VNNGP",
  "spVB-NNGP",
  "spVB-NNGP-joint",
  "DKLGP-default",
  "DKLGP",
  "spNNGP"
))

sigmasq_df = sigmasq_df %>% filter(!is.na(sigmasq), n<= 10000)

sigma_fig = ggplot(sigmasq_df, aes(x = method, y = sigmasq, fill = method)) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ n, scales = "free_y", labeller = labeller(n = function(n) paste0("n = ", n))) +
  geom_hline(yintercept = 10, color = "red", linetype = "dashed") +
  scale_fill_manual(values = method_colors) +
  labs(
    x = "Method",
    y = "Point estimation of sigmasq" ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 11),
    legend.position = "none"
  )

# sigma_fig

ggsave(file.path(output.path,"Figure_I1_sigmasq_point.pdf"), plot = sigma_fig,
       width = 9.82, height = 5.05, units = "in", 
       device = cairo_pdf)

