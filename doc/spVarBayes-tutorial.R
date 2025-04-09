## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
library(spVarBayes)
library(RANN)

## -----------------------------------------------------------------------------
library(spVarBayes)

## -----------------------------------------------------------------------------
rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

set.seed(12)
n <- 1500

coords <- cbind(runif(n,0,5), runif(n,0,5))

# Remove close points
remove_close_points <- function(x, y, threshold) {

  points <- cbind(x, y)
  
  repeat {
    neighbors <- nn2(data = points, k = 2, searchtype = "radius", radius = threshold)
    distances <- neighbors$nn.dists[, 2]
    if (all(is.na(distances) | distances >= threshold)) {
      break 
    }
    closest_index <- which.min(distances)
    points <- points[-closest_index, ]
  }
  
  list(x = points[, 1], y = points[, 2])
}

remove_coords <- remove_close_points(coords[,1], coords[,2], 0.015)
cleaned_x <- remove_coords$x
cleaned_y <- remove_coords$y

coords <- cbind(cleaned_x,cleaned_y)

n = nrow(coords)

x <- cbind(rnorm(n), rnorm(n))
B <- as.matrix(c(1,5))

sigma2_true <- 5
tau2_true <- 1
phi_true <- 6

D <- as.matrix(dist(coords))
R <- exp(-phi_true*D)
w <- rmvn(1, rep(0,n), sigma2_true*R)
y <- rnorm(n, x%*%B + w, sqrt(tau2_true))

# Split into training set and test set
n_train <- 1000
train_index = sample(1:n, n_train)
y_train = y[train_index]
x_train = x[train_index,]
w_train = w[train_index,]
coords_train = coords[train_index,]

y_test = y[-train_index]
x_test = x[-train_index,]
w_test = w[-train_index,]
coords_test = coords[-train_index,]


## -----------------------------------------------------------------------------
NNGP <- spVB_NNGP(y = y_train,X = x_train,coords=coords_train, n.neighbors = 15, 
                       n.neighbors.vi = 3,
                       rho = 0.85, max_iter = 1500, covariates = TRUE)
plot(w_train,NNGP$w_mu[order(NNGP$ord)])
abline(0,1,col="red")
w_var_NNGP <- spVB_get_Vw(NNGP)

## -----------------------------------------------------------------------------
NNGP_w_samples <- spVB_w_sampling(NNGP, n.samples = 5000)$p.w.samples
NNGP_predict <- predict(NNGP, coords.0 = coords_test, X.0 = x_test, covariates = TRUE, n.samples = 5000)
plot(w_test, apply(NNGP_predict$p.w.0, 1, mean))
abline(0,1,col="red")


## -----------------------------------------------------------------------------
NNGP_joint <- spVB_NNGP(y = y_train,X = x_train,coords=coords_train, n.neighbors = 15, 
                        n.neighbors.vi = 3,
                        rho = 0.85, max_iter = 1500, covariates = TRUE, joint = TRUE)

plot(w_train,NNGP_joint$w_mu[order(NNGP_joint$ord)])
abline(0,1,col="red")
w_var_NNGP_joint <- spVB_get_Vw(NNGP_joint)

## -----------------------------------------------------------------------------
NNGP_joint_w_samples <- spVB_joint_sampling(NNGP_joint, n.samples = 5000)$p.w.samples
NNGP_joint_predict <- predict(NNGP_joint, coords.0 = coords_test, X.0 = x_test, covariates = TRUE, n.samples = 5000)
plot(w_test, apply(NNGP_joint_predict$p.w.0, 1, mean))
abline(0,1,col="red")


## -----------------------------------------------------------------------------
MFA <- spVB_MFA(y = y_train,X = x_train,coords=coords_train, covariates = TRUE, 
                n.neighbors = 15, rho = 0.85, max_iter = 1000, LR = FALSE)
plot(w_train,MFA$w_mu[order(MFA$ord)])
abline(0,1,col="red")


## -----------------------------------------------------------------------------
MFA_w_samples <- spVB_w_sampling(MFA, n.samples = 5000)$p.w.samples
MFA_predict <- predict(MFA, coords.0 = coords_test, X.0 = x_test, covariates = TRUE, n.samples = 5000)
plot(w_test, apply(MFA_predict$p.w.0, 1, mean))
abline(0,1,col="red")


## -----------------------------------------------------------------------------
MFA_LR <- spVB_MFA(y = y_train,X = x_train,coords=coords_train, covariates = TRUE, 
                   n.neighbors = 15, rho = 0.85, max_iter = 1000, LR = TRUE)
plot(w_train,MFA_LR$w_mu[order(MFA_LR$ord)])
abline(0,1,col="red")


## -----------------------------------------------------------------------------
MFA_LR_w_samples <- spVB_LR_sampling(MFA_LR, n.samples = 5000)$p.w.samples
MFA_LR_predict <- predict(MFA_LR, coords.0 = coords_test, X.0 = x_test, covariates = TRUE, n.samples = 5000)
plot(w_test, apply(MFA_LR_predict$p.w.0, 1, mean))
abline(0,1,col="red")


