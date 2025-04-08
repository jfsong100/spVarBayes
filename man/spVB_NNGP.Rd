\name{spVB_NNGP}
\alias{spVB_NNGP}
\title{Function for Approximating Posterior Distribution using Nearest Neighbor Gaussian Process Variational Inference}

\description{
  The function \code{spVB_NNGP} fits Gaussian Bayesian spatial regression models using Nearest Neighbor Gaussian Processes (NNGP).
  Some code blocks are borrowed from the R packages: spNNGP: Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes https://CRAN.R-project.org/package=spNNGP and BRISC: Fast Inference for Large Spatial Datasets using BRISC https://CRAN.R-project.org/package=BRISC.
}

\usage{
spVB_NNGP <- function(y, X, coords, covariates = TRUE, n.neighbors = 15, n.neighbors.vi = 3,
                      sigma.sq = 1, tau.sq = 0.5, phi = NULL, sigma.sq.IG = c(0.1,1),
                      tau.sq.IG = c(0.1,0.1), phi.range = NULL, var_input = NULL,
                      n_omp = 1, cov.model = "exponential", nu = 1.5, search.type = "tree",tol = 12,
                      verbose = FALSE, max_iter = 2000, min_iter = 1000, stop_K = FALSE, K = 20,
                      N_phi = 5, Trace_N = 50, phi_max_iter = 0, rho = 0.85,
                      mini_batch = FALSE, mini_batch_size = 128, ord_type = "Sum_coords", joint = FALSE)
}

\arguments{
   \item{y}{an \eqn{n}{n} length vector of response at the observed coordinates. }

   \item{X}{an \eqn{n \times p}{n x p} matrix of the covariates in the observation coordinates. {X = NULL} if no covariates as the input.}

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2}{R^2} (e.g., easting and northing). }

  \item{covariates}{if \code{TRUE}, matrix of the covariates is needed as the input.}

  \item{n.neighbors}{number of neighbors used in the NNGP prior. This must be an integer.}

  \item{n.neighbors.vi}{number of neighbors used in the NNGP variational distribution. This must be an integer.}

  \item{sigma.sq}{starting value of sigma square. Default value is 1. }

  \item{tau.sq}{starting value of tau square. Default value is 0.5. }

  \item{phi}{starting value of phi. If there is no input, BRSIC estimation will be used. }

  \item{sigma.sq.IG}{Inverse-Gamma prior parameters for sigma.sq. The
    hyperparameters of the inverse-Gamma are
    passed as a vector of length two, with the first and second elements corresponding
    to the \emph{shape} and \emph{scale}, respectively. Default value is c(0.1,1)}

  \item{tau.sq.IG}{Inverse-Gamma prior parameters for tau.sq. The
    hyperparameters of the inverse-Gamma are
    passed as a vector of length two, with the first and second elements corresponding
    to the \emph{shape} and \emph{scale}, respectively. Default value is c(0.1,0.1)}

  \item{phi.range}{the range of phi can take is passed as a vector of length two with the first
    and second elements corresponding to the lower and upper support,
    respectively. Default value is \code{NULL}}

  \item{var_input}{the diagonal values in the covariance matrix of the variational distribution. The input can either be a scalar or a vector of length n. If there is no input, BRISC estimation will be used.}

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support. Default is 1. }

  \item{cov.model}{a quoted keyword that specifies the covariance
    function used to model the spatial dependence structure among the
    observations. Currently only \code{"exponential"} is supported.
    Future supported covariance model key words are:
    \code{"exponential"}, \code{"matern"}, \code{"spherical"}, and
    \code{"gaussian"}.  See below for details.}

  \item{nu}{starting value of nu, only required for matern covariance model. Default value is 1.5. }

  \item{search.type}{a quoted keyword that specifies type of nearest
    neighbor search algorithm. Supported method key words are: \code{"cb"} and
    \code{"brute"}. The \code{"cb"} should generally be much
    faster. If locations do not have identical coordinate values on
    the axis used for the nearest neighbor ordering (see
    \code{ord} argument) then \code{"cb"} and \code{"brute"} should
    produce identical neighbor sets. However, if there are identical
    coordinate values on the axis used for nearest neighbor ordering,
    then \code{"cb"} and \code{"brute"} might produce different, but
    equally valid, neighbor sets, e.g., if data are on a grid. }

  \item{tol}{the input observation coordinates, response and the covariates are rounded to this many places after the decimal. The default value is 12. }


  \item{verbose}{if \code{TRUE}, model specification and progress of the
    sampler is printed to the screen. Otherwise, nothing is printed to
    the screen.}

  \item{max_iter}{the maximum number of iterations to run the model, Default value is 2000. }

  \item{min_iter}{the minimum number of iterations to run the model when using change in the average of lower bound as the stopping rule, Default value is 1000. }

  \item{stop_K}{if \code{TRUE}, model considers using change in the average of lower bound as the stopping rule. }

    \item{K}{the number of consecutive times when using change in the average of lower bound as the stopping rule. The algorithm terminates when the estimated evidence lower bound falls below maximum value of estimated evidence lower bound for more than \code{K} consecutive times. Default is 20.}

    \item{N_phi}{the number of candidates for alpha_phi and beta_phi in the shifted beta distribution. Default value is 5. }

    \item{Trace_N}{the number of Monte Carlo samples. Default value is 50. We recommend a larger value for a small sigma.sq. }

    \item{phi_max_iter}{the number of maximum iterations for updating parameter phi. Default value is 10. }

    \item{rho}{the input rate for AdaDelta optimizer. Default value is 0.85. }

    \item{mini_batch}{if \code{TRUE}, the model will update parameters using mini batch. }

    \item{mini_batch_size}{the number of batch size when using mini batch. Default is 256.}

    \item{ord_type}{keyword that specifies the ordering scheme to be used in ordering the observations.
    Supported keywords are: "AMMD" and "Sum_coords" for approximate Maximum Minimum Distance and sum of
    coordinate based ordering, respectively. "AMMD" ordering is implemented using BRISC pacakge. Default value is "Sum_coords". \code{n>65} is required for "AMMD".}

    \item{joint}{if \code{TRUE}, a joint model for regression coefficients and spatial random effects will be applied, otherwise, the model will assume an independent block.}


  \item{...}{currently no additional arguments.  }
}

\value{
  An object of class \code{spVarBayes} with additional class designations for
  \code{method} and \code{family}. The return object is a list comprising:

  \item{beta}{a vector of estimated posterior mean for the regression coefficients.}

  \item{beta_cov}{a vector of the elements of the estimated posterior covariance matrix for the regression coefficients.}

  \item{theta}{a vector of estimated posterior mean for the spatial covariance parameters with each tag corresponding to a parameter name. }

  \item{theta_para}{a vector with each tag corresponding to a parameter name. Variance parameters, simga.sq and tau.sq, are assumed to follow an inverse-Gamma distribution, whereas the spatial decay phi is assumed to follow Shifted Beta distributions. The hyperparameters of the inverse-Gamma are returned as a vector of length two, with the first and second elements corresponding to the shape and scale, respectively. The hyperparameters of the Shifted Beta are also passed as a vector of length two with the first and second elements corresponding to the alpha and beta shape parameters, respectively.}

  \item{w_mu}{a vector of estimated posterior mean for the spatial random effects.}

  \item{A_vi}{a vector of the non-zero elements of A in covariance matrix of the variational distribution for the spatial random effects.}

  \item{D_vi}{a vector of the diagonal elements of D in covariance matrix of the variational distribution for the spatial random effects.}

  \item{ord}{an index vector of length \code{n} used for the nearest neighbor search. This can be used for reordering the output to the original input order. }

  \item{time}{execution time for parameter estimation reported using
    \code{proc.time()}. }

  The return object will include additional objects used for subsequent
  prediction and/or input for other functions.
}

\details{
Model parameters can be fixed at their \code{starting} values by setting their
\code{tuning} values to zero.

The \emph{no nugget} model is specified by setting \code{tau.sq} to zero
in the \code{starting} and \code{tuning} lists.
}

\references{

Datta, A., S. Banerjee, A.O. Finley, and A.E. Gelfand. (2016) Hierarchical Nearest-Neighbor Gaussian process models for large geostatistical datasets. Journal of the American Statistical Association, 111:800-812.

Saha, A., & Datta, A. (2018). BRISC: bootstrap for rapid inference on spatial covariances. Stat, e184, DOI: 10.1002/sta4.184.

Guinness, J. (2018) Permutation and Grouping Methods for Sharpening Gaussian Process Approximations, Technometrics, DOI: 10.1080/00401706.2018.1437476,
https://github.com/joeguinness/gp_reorder/tree/master/R .

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
NNGP <- spVB_NNGP(y,X = x,coords=coords, n.neighbors = 15, 
                            n.neighbors.vi = 3,
                            rho = 0.85, max_iter = 1500, covariates = TRUE)

plot(w,NNGP[order(NNGP$ord)])
abline(0,1,col="red")

##Fit a NNGP VB joint model 
NNGP_joint <- spVB_NNGP(y,X = x,coords=coords, n.neighbors = 15, 
                            n.neighbors.vi = 3,
                            rho = 0.85, max_iter = 1500, covariates = TRUE, joint = TRUE)

plot(w,NNGP_joint$w_mu[order(NNGP_joint$ord)])
abline(0,1,col="red")



}

\keyword{model}
