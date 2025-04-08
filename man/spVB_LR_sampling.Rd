\name{spVB_LR_sampling}
\alias{spVB_LR_sampling}
\title{Function for computing posterior samples for mean field approximation with linear response.}

\description{
  The function \code{spVB_LR_sampling} computes posterior samples for mean field approximation with linear response given an object of class \code{spVarBayes}. It will return samples for spatial random effects and regression coefficients.
}

\usage{
spVB_LR_sampling <- function(object,
                              n.samples,
                              seed = 1,
                              verbose = TRUE,
                              n.report=100, ...)
}

\arguments{

  \item{object}{an object of class \code{spVarBayes}, obtained as an output of \cr \code{MFA-LR}. }

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

  \item{p.beta.samples}{posterior samples for the regression coefficients where rows are locations corresponding to fitted data and columns are samples.}
  
  \item{p.w.samples}{posterior samples for the spatial random effects where rows are locations corresponding to fitted data and columns are samples.}

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



##Fit a MFA VB model with linear response
MFA_LR <- spVB_MFA(y,X = x,coords=coords, covariates = TRUE, 
                n.neighbors = 15, rho = 0.85, max_iter = 1000, LR = TRUE)

## Get posterior samples for spatial random effects
p.w.samples <- spVB_LR_sampling(MFA_LR,n.samples = 5000)$p.w.samples


}

\keyword{model}
