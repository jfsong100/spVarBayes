\name{spVB_beta_sampling}
\alias{spVB_beta_sampling}
\title{Function for computing posterior samples for regression coefficients.}

\description{
  The function \code{spVB_beta_sampling} computes posterior samples for regression coefficients given an object of class \code{spVarBayes}.
}

\usage{
spVB_beta_sampling <- function(object,
                              n.samples,
                              seed = 1,
                              verbose = TRUE,
                              n.report=100, ...)
}

\arguments{

  \item{object}{an object of class \code{spVarBayes}, obtained as an output of \cr \code{spVB_NNGP} or \code{spVB_MFA}. }

  \item{n.samples}{the number of posterior samples to collect.}

  \item{seed}{seed which is used in generation of the posterior samples. Default value is 1.}

  \item{verbose}{if \code{TRUE}, model specification and progress of the
    sampler is printed to the screen. Otherwise, nothing is printed to
    the screen.  }

  \item{n.report}{the interval to report sampling progress. }
  \item{...}{currently no additional arguments.  }

}

\value{
  A list comprising of the following:

   \item{p.beta.samples}{posterior samples for the regression coefficients where rows are the number of covariates \code{p} columns are samples.}

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



##Fit a NNGP VB model
NNGP_full <- spVB_NNGP(y,X = x,coords=coords,n.neighbors = 15,
                      n.neighbors.vi = 3,
                      max_iter = 1500, Trace_N = 30,
                      covariates = TRUE)

## Get posterior samples for regression coefficients
p.beta.samples <- spVB_beta_sampling(NNGP_full,n.samples = 5000)$p.beta.samples
summary(p.beta.samples[1,])

}

\keyword{model}
