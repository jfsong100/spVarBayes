spVB_w_sampling <- function(object,
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

  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)

  family.indx <- 1

  n <- object$n

  set.seed(seed)

  sim <- matrix(rnorm(n.samples*n))

  if(object$VI_family == "NNGP"){
    nnIndxLU_vi = object$nnIndxLU_vi
    nnIndx_vi = object$nnIndx_vi
    A_vi = object$A_vi
    S_vi = object$D_vi
    w_mu = object$w_mu

    storage.mode(nnIndxLU_vi) <- "integer"
    storage.mode(nnIndx_vi) <- "integer"
    storage.mode(A_vi) <- "double"
    storage.mode(S_vi) <- "double"
    storage.mode(w_mu) <- "double"
  }else{
    w_mu = object$w_mu
    w_sigma_sq = object$w_sigma_sq
    storage.mode(w_mu) <- "double"
    storage.mode(w_sigma_sq) <- "double"
  }

  storage.mode(n) <- "integer"
  storage.mode(n.samples) <- "integer"
  storage.mode(sim) <- "double"
  ptm <- proc.time()

  if(object$VI_family == "NNGP"){
    cat(c("Sampling from NNGP variational distribution"), "\n")
    result <- .Call("NNGP_samplingcpp",
                    n, nnIndxLU_vi, nnIndx_vi, w_mu, A_vi, S_vi, sim, n.samples, PACKAGE = "spVarBayes")

    #out <- c(out, .Call("NNGP_samplingcpp", n, nnIndxLU_vi, nnIndx_vi, w_mu, A_vi, S_vi, sim, n.samples))

  }else if(object$VI_family == "MFA"){
    cat(c("Sampling from mean-field variational distribution"), "\n")
    result <- .Call("MFA_samplingcpp",
                    n, w_mu, w_sigma_sq, sim, n.samples, PACKAGE = "spVarBayes")

    #out <- c(out, .Call("MFA_samplingcpp", n, w_mu, w_sigma_sq, sim, n.samples))
  }


  out$run.time <- proc.time() - ptm
  out$p.w.samples <- matrix(result$w_sample, ncol = n.samples)
  out
}
