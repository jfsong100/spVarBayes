spVB_LR_sampling <- function(object,
                           n.samples,
                           seed = 1,
                           verbose = TRUE,
                           n.report=100, ...){

  ##call
  out <- list()
  out$call <- match.call()
  out$object.class <- class(object)

  family.indx <- 1

  if(!class(object)[1] == "spVarBayes"){
    stop("error: requires an output object of class spVarBayes\n")
  }
  
  if(!object$VI_family == "MFA-LR"){
    stop(
      paste0(
        "error: this function is only for mean field approximation with linear response\n",
        "For mean field approximation or NNGP, please use spVB_w_sampling\n",
        "NNGP joint model, please use spVB_joint_sampling\n"
      )
    )
  }
  
  n <- object$n

  set.seed(seed)

  ptm <- proc.time()

  B_q <- object$B_q
  F_q <- object$F_q
  
  if(object$covariates){
    
    cat(c("Joint Sampling from linear response corrected mean-field variational distribution for beta and w."), "\n")
    
    p <- length(object$beta)
    sim <- matrix(rnorm(n.samples*(n+p)))
    u <- solve(B_q,matrix(sim, ncol = n.samples)*sqrt(F_q))
    u <- as.matrix(u)
    out$sim <- matrix(sim, ncol = n.samples)
    out$p.beta.samples <- u[1:p,] + object$beta
    out$p.w.samples <- u[-(1:p),] + object$w_mu

  }else{
    cat(c("Sampling from mean-field variational distribution with linear reponse."), "\n")
    
    sim <- matrix(rnorm(n.samples*n))
    u <- solve(B_q,matrix(sim, ncol = n.samples)*sqrt(F_q))
    u <- as.matrix(u)
    out$sim <- matrix(sim, ncol = n.samples)
    out$p.w.samples <- u + object$w_mu
  }



  out$run.time <- proc.time() - ptm

  out
}
