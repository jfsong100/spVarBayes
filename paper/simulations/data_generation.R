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
library(BRISC)
library(fields)
library(Matrix)
library(rhdf5)

######################################
# Simulation settings
######################################
n_vec       = c(1000, 5000, 10000, 50000, 100000)
beta_true   = c(2, 5)    # coefficients
tau2_true   = 0.5        # nugget
phi_true    = 1          # decay parameter
sigma2_true = 10         # spatial variance

maxmin_order = T
save_files = T
n=n_vec[n_index]

m_prior = 15              ## number of nearest neighbor for prior

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

####################################################################
# Calculate reference posterior mean and variance for small n<=10000
####################################################################
if(n <= 10000){
  #### calculates nngp precision matrix (without storing the full distance matrix)
  #### s=matrix of locations (each row is a location)
  #### m=neighbor size, phi=exponential decay
  myknn <- function(i,s,m){
    if(m>=(i-1)) im<-1:(i-1)
    else 	
    {
      dist=rdist(s[c(1,i),],s[c(1,1:(i-1)),])[-1,-1]
      im<-sort(order(dist)[1:m])
    }
    return(im)
  }
  #### distance matrix for location i and its neighbors ####
  Dimgen = function(i,imvec,s)	dist(s[c(i,imvec[[i-1]]),])
  nngp_precision=function(s,m,phi){
    
    n=nrow(s)
    m=min(m,n-1)
    imvec <- sapply(2:n,myknn,s,m)
    Dimvec <- sapply(2:n,Dimgen,imvec,s)
    colind = c(1:n,unlist(imvec))
    mi=c(1:(m-1),rep(m,n-m))
    rowind = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
    
    wimvec=sapply(2:n, function(i,Dimvec,par) {D=exp(-par*as.matrix(Dimvec[[i-1]]));as.vector(solve(D[-1,-1])%*%D[1,-1])},Dimvec,phi)
    fimvec=sapply(2:n, function(i,Dimvec,wimvec,par) {D=as.matrix(Dimvec[[i-1]]);1-exp(-par*D[1,-1])%*%wimvec[[i-1]]}, Dimvec,wimvec,phi)
    
    V=sparseMatrix(i=rowind,j=colind,x=c(rep(1,n),-unlist(wimvec)),dims=c(n,n))
    F=sparseMatrix(i=1:n,j=1:n,x=c(1,1/fimvec),dims=c(n,n))
    nngpprec=t(V)%*%F%*%V
  }
  
  ## Calculate reference posterior mean and variance
  M_NNGP=nngp_precision(s=S_ordered,m=m_prior,phi=phi_true)
  empirical_V = as.matrix(solve(M_NNGP/sigma2_true+1/tau2_true*diag(n)))
  empirical_mu = as.vector(solve((M_NNGP/sigma2_true+1/tau2_true*diag(n)),(y-as.vector(X%*%beta_true))/tau2_true))
}

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
  h5write(y, h5_file_path, "y_gen")
  h5write(X, h5_file_path, "X")
  h5write(w, h5_file_path, "f")
  h5write(S_ordered, h5_file_path, "S_ordered")
  if(n<=10000){
    h5write(empirical_mu, h5_file_path, "empirical_mu")
    h5write(diag(empirical_V), h5_file_path, "empirical_var")
    h5write(empirical_V, h5_file_path, "empirical_V")
  }
}

