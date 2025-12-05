\name{spVB_LR}
\alias{spVB_LR}
\title{Function for computing Linear Reponse after fitting mean-field approximation.}

\description{
  The function \code{spVB_LR} computes Linear Reponse correction for Mean Field Approximation Variational Inference.
}

\usage{
spVB_LR <- function(object, get_mat = TRUE, get_para = TRUE, n_omp = 1, 
                        sigma.sq.IG = c(0.1,1),
                        tau.sq.IG = c(0.1,0.1),...)
}

\arguments{

  \item{object}{an object of class \code{spVarBayes}, obtained as an output of \code{spVB_MFA}. }

  \item{get_mat}{if \code{TRUE}, the linear response corrected variance covariance matrix for regression coefficients and spatial random effects is returned. Otherwise, only variance vector is obtained. }

  \item{get_para}{if \code{TRUE}, the spatial parameters will be updated using the linear response corrected variance covariance matrix. Otherwise, spatial parameters remain unchanged. }

  \item{sigma.sq.IG}{Inverse-Gamma prior parameters for sigma.sq. The
    hyperparameters of the inverse-Gamma are
    passed as a vector of length two, with the first and second elements corresponding
    to the \emph{shape} and \emph{scale}, respectively. Default value is c(0.1,1)}

  \item{tau.sq.IG}{Inverse-Gamma prior parameters for tau.sq. The
    hyperparameters of the inverse-Gamma are
    passed as a vector of length two, with the first and second elements corresponding
    to the \emph{shape} and \emph{scale}, respectively. Default value is c(0.1,0.1)}
    
  \item{...}{currently no additional arguments.  }

}

\value{
  A list comprising of the following:

  \item{updated_mat}{if LR is \code{TRUE}, the linear response corrected variance covariance matrix for regression coefficients and spatial random effects. The dimension will be \code{(p+n) * (p+n)} }
    
  \item{LR_time}{if LR is \code{TRUE}, execution time for computing linear response reported using
    \code{proc.time()}. }

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
n <- 1000


coords <- cbind(runif(n,0,1), runif(n,0,1))

x <- cbind(1, rnorm(n))
B <- as.matrix(c(1,5))

sigma2_true <- 5
tau2_true <- 1
phi_true <- 6

D <- as.matrix(dist(coords))
R <- exp(-phi_true*D)
w <- rmvn(1, rep(0,n), sigma2_true*R)
y <- rnorm(n, x \%*\% B + w, sqrt(tau2_true))

##Fit a MFA VB model
MFA_pre_fit <- spVB_MFA(y = y,X = x, coords = coords, covariates = TRUE, 
                n.neighbors = 15, rho = 0.85, max_iter = 3000, LR = TRUE)

MFA_LR = spVB_LR(MFA_pre_fit, get_mat = TRUE, get_para = TRUE, n_omp = 5)

}

\keyword{model}
