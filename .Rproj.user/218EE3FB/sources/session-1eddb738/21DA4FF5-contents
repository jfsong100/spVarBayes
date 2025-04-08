spVB_joint_sampling <- function(object,
                           n.samples,
                           seed = 1,
                           verbose = TRUE,
                           n.report=100, ...){

  # formal.args <- names(formals(sys.function(sys.parent())))
  #
  # elip.args <- list(...)
  # for(i in names(elip.args)){
  #   if(! i %in% formal.args)
  #     warning("'",i, "' is not an argument")
  # }
  #
  # if(missing(object)){stop("error: predict expects object\n")}
  # if(!class(object)[1] == "NNGPVI"){
  #   stop("error: requires an output object of class NNGPVI\n")
  # }

  if(!class(object)[1] == "spVarBayes"){
    stop("error: requires an output object of class spVarBayes\n")
  }
  
  if(!object$VI_family == "NNGP"){
    stop(
      paste0(
        "error: this function is only for NNGP joint model.\n",
        "For mean field approximation, please use spVB_w_sampling.\n"
      )
    )
  }
  
  if(!object$joint){
    stop("error: for independent NNGP, please use spVB_w_sampling\n")
  }
  
  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)

  family.indx <- 1

  n <- object$n

  set.seed(seed)

  sim <- matrix(rnorm(n.samples*n))

  nnIndxLU_vi <- object$nnIndxLU_vi
  nnIndx_vi <- object$nnIndx_vi
  A_vi <- object$A_vi
  S_vi <- object$D_vi
  w_mu <- object$w_mu
  
  storage.mode(nnIndxLU_vi) <- "integer"
  storage.mode(nnIndx_vi) <- "integer"
  storage.mode(A_vi) <- "double"
  storage.mode(S_vi) <- "double"
  storage.mode(w_mu) <- "double"
  
  storage.mode(n) <- "integer"
  storage.mode(n.samples) <- "integer"
  storage.mode(sim) <- "double"
  ptm <- proc.time()

  cat(c("Joint Sampling from NNGP variational distribution for beta and w."), "\n")
  
  
  set.seed(seed+1)
  
  p <- length(object$beta)
  sim_beta <- matrix(rnorm(n.samples*p))
  
  E_vi <- object$E_vi
  L_beta <- object$L_beta
  A_beta <- object$A_beta
  mu_beta <- object$beta
  IndxLU_beta <- object$IndxLU_beta
  
  storage.mode(E_vi) <- "double"
  storage.mode(L_beta) <- "double"
  storage.mode(A_beta) <- "double"
  
  storage.mode(mu_beta) <- "double"
  storage.mode(sim_beta) <- "double"
  storage.mode(IndxLU_beta) <- "integer"
  storage.mode(p) <- "integer"
  
  result <- .Call("NNGP_joint_samplingcpp",
                  n, nnIndxLU_vi, nnIndx_vi, w_mu, A_vi, S_vi, sim, n.samples, p, sim_beta, E_vi, A_beta, L_beta, mu_beta, IndxLU_beta, PACKAGE = "spVarBayes")
  
  

  out$run.time <- proc.time() - ptm
  out$p.w.samples <- matrix(result$w_sample, ncol = n.samples)
  out$p.beta.samples = matrix(result$beta_sample, ncol = n.samples)

  out
}
