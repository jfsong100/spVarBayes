spVB_LR <- function(object, get_mat = TRUE, n_omp = 1,
                        sigma.sq.IG = c(0.1,1),
                        tau.sq.IG = c(0.1,0.1)) {

  Trace_N <- object$Trace_N
  p <- 0
  n <- object$n
  
  if(object$covariates){
    beta_mu <- object$beta
    beta_sigmasq <- diag(object$beta_cov)
    p <- length(beta_mu)

    Inter_mat <- .Call("construct_I_VH", object$n, object$X, object$theta[2], object$nnIndxLU, object$nnIndx,
                       object$numIndxCol, object$nnIndxnnCol, object$cumnumIndxCol,
                       object$B, object$F, c(beta_sigmasq, object$w_sigma_sq))

    if(p==1){
      if(!get_mat){
        num <- n_omp
        cat("-------------------------------------------------------", "\n")
        cat(c("Default Number of Threads is", unname(num)), "\n")
        cat(c("Compute the Linear response for variance", unname(num)), "\n")
        
        RcppParallel::setThreadOptions(numThreads = num)
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_diagonal_parallel", Inter_mat, c(beta_sigmasq, object$w_sigma_sq), 1000, 1e-6)
        time2 <- proc.time()

      }else{
        result_list <- list()
        num <- n_omp
        cat("-------------------------------------------------------", "\n")
        cat(c("   Default Number of Threads is", unname(num)), "\n")
        cat(c("   Compute the Linear response for covariance matrix", "\n"))
        RcppParallel::setThreadOptions(numThreads = num)
        # cat(c("Compute the nearest positive definite for covariance matrix", "\n"))
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_matrix_parallel", Inter_mat, c(beta_sigmasq, object$w_sigma_sq))
        # results_pd <- .Call("nearest_positive_definite", mat_results[-(1:p),-(1:p)], 1e-6, 100)
        object$updated_mat = updated_mat
        cat("-------------------------------------------------------", "\n")
        cat(c("   Update spatial parameters \n"))
        cat("-------------------------------------------------------", "\n")
        ### Update tausq ###
        b_tau_update = tau.sq.IG[2] + (sum(qr.resid(qr(object$X), object$y - object$w_mu)^2) + p*object$theta[2] + sum(diag(updated_mat)[-(1:p)]))/2
        
        ### Update sigmasq ###
        LR_mat_decompose <- spVB_LR_chol(object)
        prior_mat <- spVB_prior(object)
        
        B_q <- LR_mat_decompose$V
        F_q <- LR_mat_decompose$F
        B_mat <- prior_mat$B_mat
        F_mat <- prior_mat$F_mat
        
        set.seed(1)
        sim <- matrix(rnorm(Trace_N*(n+p)))
        u <- solve(B_q,matrix(sim, ncol = Trace_N)*sqrt(F_q))
        
        MNNGP <- t(B_mat) %*% solve(F_mat) %*% B_mat
        U <- u[-(1:p), 1:Trace_N, drop = FALSE]
        
        b_sigma_update <- sigma.sq.IG[2] + (sum(colSums((MNNGP %*% U) * U))/Trace_N + sum((B_mat %*% object$w_mu)^2 / diag(F_mat)))*object$theta[1]*0.5
        time2 <- proc.time()
        
        
      }
    }else{
      X = object$X
      mat1 = t(X) %*% X
      diag(mat1) = rep(0,p)
      beta_premat_pp = -(object$beta_cov %*% mat1)/object$theta[2]
      beta_premat_pn = object$beta_cov %*% t(X)

      Inter_mat <- .Call("construct_I_VH_p", object$n, p, object$X, object$theta[2], object$nnIndxLU, object$nnIndx,
                         object$numIndxCol, object$nnIndxnnCol, object$cumnumIndxCol,
                         object$B, object$F, c(beta_sigmasq, object$w_sigma_sq), beta_premat_pp, beta_premat_pn)

      if(get_mat){
        num <- n_omp
        cat("-------------------------------------------------------", "\n")
        cat(c("   Default Number of Threads is", unname(num)), "\n")
        cat(c("   Compute the Linear response for variance \n"))
        RcppParallel::setThreadOptions(numThreads = num)
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_full_p_parallel", Inter_mat, object$beta_cov ,object$w_sigma_sq, p, 1000, 1e-6)
        object$updated_mat = updated_mat
        
        cat("-------------------------------------------------------", "\n")
        cat(c("   Update spatial parameters \n"))
        cat("-------------------------------------------------------", "\n")
        ### Update tausq ###
        # b_tau_update = tau.sq.IG[2] + (sum(qr.resid(qr(object$X), object$y - object$w_mu)^2) + p*object$theta[2] + sum(diag(updated_mat)[-(1:p)]))/2
        # 
        # ### Update sigmasq ###
        # LR_mat_decompose = spVB_LR_chol(object)
        # prior_mat = spVB_prior(object)
        # 
        # B_q = LR_mat_decompose$V
        # F_q = LR_mat_decompose$F
        # B_mat = prior_mat$B_mat
        # F_mat = prior_mat$F_mat
        # 
        # set.seed(1)
        # sim <- matrix(rnorm(Trace_N*(n+p)))
        # u <- solve(B_q,matrix(sim, ncol = Trace_N)*sqrt(F_q))
        # 
        # MNNGP <- t(B_mat) %*% solve(F_mat) %*% B_mat
        # U <- u[-(1:p), 1:Trace_N, drop = FALSE]
        # 
        # b_sigma_update = sigma.sq.IG[2] + (sum(colSums((MNNGP %*% U) * U))/Trace_N + sum((B_mat %*% object$w_mu)^2 / diag(F_mat)))*object$theta[1]*0.5
        time2 <- proc.time()
      }else{
        num <- n_omp
        cat("-------------------------------------------------------", "\n")
        cat(c("   Default Number of Threads is", unname(num)), "\n")
        cat(c("   Compute the Linear response for variance \n"))
        RcppParallel::setThreadOptions(numThreads = num)
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_diagonal_parallel_p", Inter_mat, object$beta_cov ,object$w_sigma_sq, p, 1000, 1e-6)
        time2 <- proc.time()
      }

    }

  }else{
    num <- n_omp
    cat("-------------------------------------------------------", "\n")
    cat(c("   Default Number of Threads is", unname(num)), "\n")
    cat(c("   Compute the Linear response for variance \n"))
    Inter_mat <- .Call("construct_I_VH_nop", object$n, object$theta[2], object$nnIndxLU, object$nnIndx,
                       object$numIndxCol, object$nnIndxnnCol, object$cumnumIndxCol,
                       object$B, object$F, object$w_sigma_sq)
    RcppParallel::setThreadOptions(numThreads = num)
    time1 <- proc.time()
    updated_mat <- .Call("compute_Hinv_V_full_nop_parallel", Inter_mat, object$w_sigma_sq)
    object$updated_mat <- updated_mat
    cat("-------------------------------------------------------", "\n")
    cat(c("   Update spatial parameters \n"))
    cat("-------------------------------------------------------", "\n")
    b_tau_update <- tau.sq.IG[1] + (sum((object$y-object$w_mu)^2) + sum(diag(updated_mat)))/2
    
    ### Update sigmasq ###
    LR_mat_decompose <- spVB_LR_chol(object)
    prior_mat <- spVB_prior(object)
    
    B_q <- LR_mat_decompose$V
    F_q <- LR_mat_decompose$F
    B_mat <- prior_mat$B_mat
    F_mat <- prior_mat$F_mat
    
    set.seed(1)
    sim <- matrix(rnorm(Trace_N*(n)))
    u <- solve(B_q,matrix(sim, ncol = Trace_N)*sqrt(F_q))
    
    MNNGP <- t(B_mat) %*% solve(F_mat) %*% B_mat
    U <- u
    
    b_sigma_update <- sigma.sq.IG[2] + (sum(colSums((MNNGP %*% U) * U))/Trace_N + sum((B_mat %*% object$w_mu)^2 / diag(F_mat)))*object$theta[1]*0.5
    time2 <- proc.time()
  }

  Theta <- object$theta
  if(object$cov.model!="matern"){
    names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi")
  }else{names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi", "nu")}
  Theta_para <- object$theta_para
  if(object$cov.model!="matern"){
    names(Theta_para) <- c("sigma.sq.alpha", "sigma.sq.beta",
                           "tau.sq.alpha", "tau.sq.beta",
                           "phi.alpha","phi.beta")
  }

  # Theta[2] <- b_tau_update/Theta_para[3]
  # Theta_para[4] <- b_tau_update
  # 
  # Theta[1] <- b_sigma_update/Theta_para[1]
  # Theta_para[2] <- b_sigma_update
  result_list = object
  result_list$VI_family <-  "MFA-LR"
  result_list$LR_time <- time2 - time1
  # result_list$theta <- Theta
  # result_list$theta_para <- Theta_para
  # result_list$updated_mat <- updated_mat
  # result_list$B_q <- B_q
  # result_list$F_q <- F_q
  # result_list$sim <- sim
  # result_list$B_mat <- B_mat
  # result_list$F_mat <- F_mat
  
  return(result_list)
}
