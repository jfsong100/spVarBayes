predict.spVarBayes <- function(object, X.0, coords.0,covariates = TRUE,
                           n.samples,
                           n.omp.threads = 1,
                           seed = 1,
                           verbose = TRUE,
                           phi.fix = FALSE,
                           n.report=100, ...){

  ####################################################
  ##Check for unused args
  ####################################################
  formal.args <- names(formals(sys.function(sys.parent())))

  elip.args <- list(...)
  for(i in names(elip.args)){
    if(! i %in% formal.args)
      warning("'",i, "' is not an argument")
  }

  if(missing(object)){stop("error: predict expects object\n")}
  if(!class(object)[1] == "spVarBayes"){
    stop("error: requires an output object of class spVarBayes\n")
  }

  p <- 0

  if(covariates){
    if(is.null(X.0)){stop("error: X must be input if covariates is TRUE")}
    if(!is.matrix(X.0)){stop("error: X must be a matrix")}
    p <- ncol(X.0)
    X <- object$X
  }

  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)

  y <- object$y
  coords <- object$coords
  #family <- object$family ##family code gaussian=1, binomial=2, ...

  family.indx <- 1
  ##if(class(object)[3] == "binomial"){
  # if(out$type[2] == "binomial"){
  #   family.indx <- 2
  # }

  n <- length(y)

  theta.para <- object$theta_para
  #zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;


  zetasq.alpha <- theta.para[1] # IG distribution
  zetasq.beta <- theta.para[2]

  tausq.alpha <- theta.para[3] # IG distribution
  tausq.beta <- theta.para[4]

  phi.alpha <- theta.para[5] # beta distribution
  phi.beta <- theta.para[6]

  phimin <- object$phi.range[1]
  phimax <- object$phi.range[2]

  set.seed(seed)

  if(phi.fix){
    p.phi.samples <- rep(object$theta[3],n.samples)
  }else{
    p.phi.samples <- rbeta(n.samples,phi.alpha,phi.beta)*(phimax - phimin) + phimin
  }

  p.sigmasq.samples <- rigamma(n.samples,zetasq.alpha,zetasq.beta)
  p.tausq.samples <- rigamma(n.samples,tausq.alpha,tausq.beta)

  p.theta.samples <- t(cbind(p.sigmasq.samples,
                           p.tausq.samples,
                           p.phi.samples))

  ##if(class(object)[2] == "latent"){
  if(object$VI_family == "NNGP"){
    #cat(c("Sampling from NNGP variational distribution"), "\n")
    # p.w.samples <- t(as.matrix(rmvnorm(n.samples,object$w_mu,get_Vw(n, object$n.neighbors.vi,
    #                                                               object$nnIndx_vi,
    #                                                               object$A_vi, object$S_vi))))
    if(covariates){
      if(object$joint){
        p.samples <- spVB_joint_sampling(object,n.samples = n.samples)
        p.beta.samples <- p.samples$p.beta.samples
        p.w.samples <- p.samples$p.w.samples
      }else{
        p.beta.samples <- t(as.matrix(rmvnorm(n.samples,object$beta,object$beta_cov)))
        p.w.samples <- spVB_w_sampling(object,n.samples = n.samples)$p.w.samples
      }
    }else{
      p.w.samples <- spVB_w_sampling(object,n.samples = n.samples)$p.w.samples
    }
    
  }else if(object$VI_family == "MFA-LR"){
    p.samples <- spVB_LR_sampling(object,n.samples = n.samples)
    p.w.samples <- p.samples$p.w.samples
    if(covariates){
      p.beta.samples <- p.samples$p.beta.samples
    }
  }else{
    #cat(c("Sampling from mean-field variational distribution"), "\n")
    #p.w.samples = t(as.matrix(rmvnorm(n.samples,object$w_mu,diag(object$w_sigma_sq))))
    p.w.samples <- spVB_w_sampling(object,n.samples = n.samples)$p.w.samples
    if(covariates){
      p.beta.samples <- t(as.matrix(rmvnorm(n.samples,object$beta,object$beta_cov)))
    }
  }

  n.neighbors <- object$n.neighbors
  cov.model <- object$cov.model
  ##Covariance model
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1


  ##if(class(object)[2] == "latent"){


  ##check X.0 and coords.0
  #if(missing(X.0)){stop("error: X.0 must be specified\n")}
  #if(!any(is.data.frame(X.0), is.matrix(X.0))){stop("error: X.0 must be a data.frame or matrix\n")}
  #if(ncol(X.0) != ncol(X)){ stop(paste("error: X.0 must have ",p," columns\n"))}

  if(missing(coords.0)){stop("error: coords.0 must be specified\n")}
  if(!any(is.data.frame(coords.0), is.matrix(coords.0))){stop("error: coords.0 must be a data.frame or matrix\n")}
  if(!ncol(coords.0) == 2){stop("error: coords.0 must have two columns\n")}

  q <- nrow(coords.0)

  ##get nn indx
  nn.indx.0 <- nn2(coords, coords.0, k=n.neighbors)$nn.idx-1 ##obo for cNNGP.cpp indexing

  if(covariates){
    storage.mode(X) <- "double"
    storage.mode(X.0) <- "double"
    storage.mode(p.beta.samples) <- "double"
  }

  storage.mode(y) <- "double"
  storage.mode(coords) <- "double"
  storage.mode(n) <- "integer"
  #storage.mode(p) <- "integer"
  storage.mode(n.neighbors) <- "integer"

  storage.mode(coords.0) <- "double"
  storage.mode(q) <- "integer"
  storage.mode(p) <- "integer"

  storage.mode(p.theta.samples) <- "double"
  ##if(class(object)[2] == "latent"){
  storage.mode(p.w.samples) <- "double"

  storage.mode(n.samples) <- "integer"
  storage.mode(cov.model.indx) <- "integer"
  storage.mode(nn.indx.0) <- "integer"
  storage.mode(n.omp.threads) <- "integer"
  storage.mode(verbose) <- "integer"
  storage.mode(n.report) <- "integer"
  storage.mode(family.indx) <- "integer"

  ptm <- proc.time()

  if(!covariates){
    # print(c("dim theta",dim(p.theta.samples)))
    # print(c("dim w",dim(p.w.samples)))
    out <- c(out, .Call("NobetaPredict", y, coords, n, n.neighbors, coords.0, q, nn.indx.0,
                        p.theta.samples,
                        p.w.samples, n.samples, family.indx, cov.model.indx, n.omp.threads, verbose, n.report))

  }else{
    # print(c("dim beta",dim(p.beta.samples)))
    # print(c("dim theta",dim(p.theta.samples)))
    # print(c("dim w",dim(p.w.samples)))
    out <- c(out, .Call("WithbetaPredict", y, X, coords, n, p, n.neighbors, coords.0, X.0, q, nn.indx.0,
                        p.theta.samples, p.beta.samples,
                        p.w.samples, n.samples, family.indx, cov.model.indx, n.omp.threads, verbose, n.report))
    out$p.beta.samples <- p.beta.samples
  }
  out$p.theta.samples <- p.theta.samples
  out$p.w.samples <- p.w.samples
  out$nn.indx.0 <- nn.indx.0
  out$run.time <- proc.time() - ptm


  class(out) <- "predict.spVarBayes"
  out
}


