######################################
# Input scenario (comment out the following for cluster)
######################################
args = commandArgs(trailingOnly = TRUE)
for (a in args) eval(parse(text = a))
t       = as.integer(t)
n_index = as.integer(n_index)

t
n_index

######################################
# Load packages
######################################
# load packages
library(BRISC)
library(MASS)
library(fields)
library(Matrix)
library(rhdf5)

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

maxmin_order = T
save_files = T

seed = as.integer(t+n*1000) 
set.seed(seed)
x_axis = runif(n,0,10)
y_axis = runif(n,0,10)
s = cbind(x_axis,y_axis)

if(maxmin_order){
  ord_maxmin = BRISC_order(s, order = "AMMD")
  S_ordered <- s[ord_maxmin,]
  ord = 1:n
}else{
  ord_sum <- order(s[,1] + s[,2])
  S_ordered <- s[ord_sum,]
  ord = 1:n
}

wgen=BRISC_simulation(S_ordered, sim_number = 1,
                      seeds =  seed, sigma.sq = sigma2_true,
                      tau.sq = 0, phi = phi_true, n.neighbors = 100)
w=as.vector(wgen$output.data)
X = cbind(rnorm(n,0,1),rnorm(n,0,1))
y=as.vector(as.vector(X %*% beta_true) + w + rnorm(n,mean=0,sd=sqrt(tau2_true)))

######################################
# Split data into training and testing
######################################
set.seed(seed)
train_index = sort(sample(1:n,n_train[n_index]))
y_train = y[train_index]
y_test = y[-train_index]

w_train = w[train_index]
w_test = w[-train_index]

S_train = S_ordered[train_index,]
S_test = S_ordered[-train_index,]

X_train = X[train_index,]
X_test = X[-train_index,]

######################################
# Save files
######################################
base_dir  = file.path(getwd())
print(base_dir)
data.path = file.path(base_dir,"data_sim")
if(!dir.exists(data.path)){
  dir.create(data.path)
}

if(save_files){
  scenario_path = paste0("n_",format(n_vec[n_index], scientific = FALSE),"_seed_",t)
  h5_file_path = paste0(data.path, "/",scenario_path, "_data.h5")
  
  # Create the HDF5 file, overwriting it if it already exists
  if (file.exists(h5_file_path)) {
    file.remove(h5_file_path)
  }
  h5createFile(h5_file_path)
  
  # Write each dataset to the HDF5 file
  h5write(y_train, h5_file_path, "y_train")
  h5write(X_train, h5_file_path, "X_train")
  h5write(w_train, h5_file_path, "f_train")
  h5write(S_train, h5_file_path, "S_train")
  
  h5write(y_test, h5_file_path, "y_test")
  h5write(X_test, h5_file_path, "X_test")
  h5write(w_test, h5_file_path, "f_test")
  h5write(S_test, h5_file_path, "S_test")
}
