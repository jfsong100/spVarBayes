spVB_beta_sampling <- function(object,
                              n.samples,
                              seed = 1,
                              verbose = TRUE,
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

  if(!object$covariates){
    stop("error: no covariates are included in the model\n")
  }

  if(object$joint){
    warning('We recommend using spVB_joint_sampling for NNGP joint model.')
  }
  
  p <- ncol(object$X)


  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)


  n <- object$n

  set.seed(seed)

  p.beta.samples <- t(as.matrix(rmvnorm(n.samples,object$beta,object$beta_cov)))
  out$p.beta.samples <- p.beta.samples

  out
}
