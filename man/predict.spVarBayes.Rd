\name{predict.spVarBayes}

\alias{predict.spVarBayes}

\title{Function for prediction at new locations using \code{spVarBayes} models.}

\description{
  The function \code{predict} collects posterior predictive samples
  for a set of new locations given an object of
  class \code{spVarBayes}. Some code blocks are borrowed from the R packages: spNNGP: Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes https://CRAN.R-project.org/package=spNNGP and BRISC: Fast Inference for Large Spatial Datasets using BRISC https://CRAN.R-project.org/package=BRISC.
}

\usage{
  \method{predict}{spVarBayes}(object, X.0, coords.0,
                               n.samples, n.omp.threads = 1, seed = 1,
                               verbose = TRUE, covariates = TRUE, phi.fix = FALSE, n.report=100, ...)
}

\arguments{
  \item{object}{an object of class \code{spVarBayes}. }
  \item{X.0}{the design matrix for prediction locations. An
    intercept should be provided in the first column if one is specified
    in \code{spVarBayes} model.}
  
  \item{coords.0}{the spatial coordinates corresponding to
    \code{X.0}. }
  
  \item{covariates}{if \code{TRUE}, matrix of the covariates is needed as the input.}
  
  \item{n.samples}{the number of posterior samples to collect.}
  
  \item{n.omp.threads}{a positive integer indicating
    the number of threads to use for SMP parallel processing. The package must
    be compiled for OpenMP support. For most Intel-based machines, we recommend setting
    \code{n.omp.threads} up to the number of hyperthreaded cores. Note, \code{n.omp.threads} > 1 might not
    work on some systems.}
  
  \item{seed}{seed which is used in generation of the posterior samples. Default value is 1.}
  
  \item{verbose}{if \code{TRUE}, model specification and progress of the
    sampler is printed to the screen. Otherwise, nothing is printed to
    the screen.  }
  
  \item{n.report}{the interval to report sampling progress. }
  \item{...}{currently no additional arguments.  }
}

\value{
  An object of class \code{predict.spVarBayes} which is a list comprising:
  \describe{
    \item{p.y.0}{a matrix that holds the response variable posterior predictive samples where rows correspond to \code{coords.0} and columns are samples.}
    \item{p.w.0}{a matrix that holds the random effect posterior predictive samples where rows correspond to \code{coords.0} and columns are samples.}
    \item{p.w.samples}{posterior samples for the spatial random effects for the fitted data; rows correspond to locations and columns to samples.}
    \item{p.theta.samples}{posterior samples for the spatial covariance parameters (rows: sigma.sq, tau.sq, phi; columns: samples).}
    \item{p.beta.samples}{posterior samples for the regression coefficients (rows: number of covariates \code{p}; columns: samples).}
    \item{run.time}{execution time as reported by \code{proc.time()}.}
  }
}


\references{
  Datta, A., S. Banerjee, A.O. Finley, and A.E. Gelfand. (2016)
  Hierarchical Nearest-Neighbor Gaussian process models for large
  geostatistical datasets. \emph{Journal of the American Statistical
    Association}, \doi{10.1080/01621459.2015.1044091}.
  
}

\author{
  Jiafang Song \email{jsong100@jhu.edu}, \cr
  Abhirup Datta \email{abhidatta@jhu.edu}
}



\examples{
  
  rmvn <- function(n, mu=0, V = matrix(1)){
    p <- length(mu)
    if(any(is.na(match(dim(V),p))))
      stop("Dimension problem!")
    D <- chol(V)
    t(matrix(rnorm(n*p), ncol=p) \%*\% D + rep(mu,rep(n,p)))
  }
  
  ##Make some data
  set.seed(12)
  n <- 2000
  
  coords <- cbind(runif(n,0,5), runif(n,0,5))
  
  x <- cbind(1, rnorm(n))
  B <- as.matrix(c(1,5))
  
  sigma2_true <- 5
  tau2_true <- 1
  phi_true <- 6
  
  D <- as.matrix(dist(coords))
  R <- exp(-phi_true*D)
  
  w <- rmvn(1, rep(0,n), sigma2_true*R)
  y <- rnorm(n, x \%*\% B + w, sqrt(tau2_true))
  
  ##Split into training data and test data
  set.seed(12)
  train_index <- sample(n,1500)
  coords_train <- coords[train_index,]
  y_train <- y[train_index]
  w_train <- w[train_index]
  x_train <- x[train_index,]
  
  coords_test <- coords[-train_index,]
  y_test <- y[-train_index]
  w_test <- w[-train_index]
  x_test <- x[-train_index,]
  
  # load packages
  library(spVarBayes)
  
  # Perform MFA
  MFA <- spVB_MFA(y = y_train,X = x_train,coords=coords_train, covariates = TRUE, 
                  n.neighbors = 15, rho = 0.85, max_iter = 1000, LR = FALSE)
  MFA_predict <- predict(MFA, coords.0 = coords_test, 
                         X.0 = x_test, covariates = TRUE, n.samples = 5000)
  plot(w_test, apply(MFA_predict$p.w.0, 1, mean))
  abline(0,1,col="red")
  
  # Perform MFA with linear response
  MFA_LR <- spVB_MFA(y = y_train,X = x_train,coords=coords_train, covariates = TRUE, 
                     n.neighbors = 15, rho = 0.85, max_iter = 1000, LR = TRUE)
  MFA_LR_predict <- predict(MFA_LR, coords.0 = coords_test, 
                            X.0 = x_test, covariates = TRUE, n.samples = 5000)
  plot(w_test, apply(MFA_LR_predict$p.w.0, 1, mean))
  abline(0,1,col="red")
  
  # Perform NNGP
  NNGP <- spVB_NNGP(y = y_train,X = x_train,coords=coords_train, n.neighbors = 15, 
                    n.neighbors.vi = 3,
                    rho = 0.85, max_iter = 1500, covariates = TRUE)
  NNGP_predict <- predict(NNGP, coords.0 = coords_test, 
                          X.0 = x_test, covariates = TRUE, n.samples = 5000)
  plot(w_test, apply(NNGP_predict$p.w.0, 1, mean))
  abline(0,1,col="red")
  
  # Perform NNGP joint model
  NNGP_joint <- spVB_NNGP(y = y_train,X = x_train,coords=coords_train, n.neighbors = 15, 
                          n.neighbors.vi = 3,
                          rho = 0.85, max_iter = 1500, covariates = TRUE, joint = TRUE)
  
  NNGP_joint_predict <- predict(NNGP_joint, coords.0 = coords_test, 
                                X.0 = x_test, covariates = TRUE, n.samples = 5000)
  plot(w_test, apply(NNGP_joint_predict$p.w.0, 1, mean))
  abline(0,1,col="red")
  
}

\keyword{prediction}
