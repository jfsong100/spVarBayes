\name{spVB_get_Vw}
\alias{spVB_get_Vw}
\title{Function for computing covariance matrix of the variational distribution for spatial random effects.}

\description{
  The function \code{spVB_get_Vw} computes the covariance matrix of the variational distribution for spatial random effects given an object of class \code{spVarBayes}.
}

\usage{
spVB_get_Vw <- function(object,...)
}

\arguments{

  \item{object}{an object of class \code{spVarBayes}, obtained as an output of \cr \code{spVB_NNGP} or \code{spVB_MFA}. }

}

\value{
  A list comprising of the following:

  \item{V_w}{a \code{sparseMatrix} object.}

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
  t(matrix(rnorm(n*p), ncol=p)\%*\%D + rep(mu,rep(n,p)))
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
y <- rnorm(n, x\%*\%B + w, sqrt(tau2_true))



##Fit a NNGP VB model
NNGP_full <- spVB_NNGP(y,X = x,coords=coords,n.neighbors = 15,
                      n.neighbors.vi = 3,
                      max_iter = 1500, 
                      covariates = TRUE)

## Get cov matrix for NNGP_full model
w_var_NNGP_full <- spVB_get_Vw(NNGP_full)

}

\keyword{model}
