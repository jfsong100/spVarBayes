spVB_theta_sampling <- function(object,
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

  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)


  n <- object$n

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

  p.phi.samples <- rbeta(n.samples,phi.alpha,phi.beta)*(phimax - phimin) + phimin
  p.sigmasq.samples <- rigamma(n.samples,zetasq.alpha,zetasq.beta)
  p.tausq.samples <- rigamma(n.samples,tausq.alpha,tausq.beta)

  p.theta.samples <- t(cbind(p.sigmasq.samples,
                             p.tausq.samples,
                             p.phi.samples))

  out$p.theta.samples <- p.theta.samples

  out
}
