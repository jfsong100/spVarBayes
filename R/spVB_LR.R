spVB_LR <- function(object, get_mat = TRUE, get_para = TRUE, n_omp = 1, n_large = FALSE,
                        sigma.sq.IG = c(0.1,1),
                        tau.sq.IG = c(0.1,0.1)) {

  Trace_N <- object$Trace_N
  p <- 0
  n <- object$n
  
  if(!get_mat){
    get_para = FALSE
    warning('Only compute the diagonals, spatial parameters will not be updated.')
  }
  
  if(n_large){
    warning('For large sample size, We recommend using decomposed matrix.')
    y = object$y
    coords = object$coords
    n.neighbors = object$n.neighbors
    search.type = "tree"
    
    cov.model = object$cov.model
    
    ##Covariance model
    cov.model.names <- c("exponential","spherical","matern","gaussian")
    cov.model.indx <- which(cov.model == cov.model.names) - 1
    storage.mode(cov.model.indx) <- "integer"
    
    nu.Unif = 0
    nu = 1.5
    
    storage.mode(sigma.sq.IG) <- "double"
    storage.mode(tau.sq.IG) <- "double"
    storage.mode(nu.Unif) <- "double"
    
    nu.starting = nu
    storage.mode(nu.starting) <- "double"
    
    
    ##Search type
    search.type.names <- c("brute", "tree", "cb")
    if(!search.type %in% search.type.names){
      stop("error: specified search.type '",search.type,"' is not a valid option; choose from ", paste(search.type.names, collapse=", ", sep="") ,".")
    }
    search.type.indx <- which(search.type == search.type.names)-1
    storage.mode(search.type.indx) <- "integer"
    
    n.omp.threads <- as.integer(1)
    storage.mode(n.omp.threads) <- "integer"
    fix_nugget <- 1
    w_mu = object$w_mu
    theta_input = object$theta
    
    storage.mode(n) <- "integer"
    storage.mode(y) <- "double"
    storage.mode(coords) <- "double"
    storage.mode(w_mu) <- "double"
    storage.mode(theta_input) <- "double"
    storage.mode(n.neighbors) <- "integer"
    storage.mode(fix_nugget) <- "double"
    B_q <- numeric(length(object$nnIndx))  
    F_q <- numeric(n)    
    result_list = object
    result_list$VI_family <-  "MFA-LR"
    
    num <- n_omp
    RcppParallel::setThreadOptions(numThreads = num)
    time1 <- proc.time()
    result_big <- .Call("spVarBayes_MFA_LR_update_nocovariates_bigcpp",
                        y, n, n.neighbors, coords, cov.model.indx,
                        sigma.sq.IG, tau.sq.IG,
                        search.type.indx, n.omp.threads, fix_nugget, nu.Unif, nu.starting,
                        w_mu, theta_input, Inter_mat, object$w_sigma_sq, PACKAGE = "spVarBayes")
    time2 <- proc.time()
    result_list$B_q = result_big$B_q
    result_list$F_q = result_big$F_q
    result_list$LR_time <- time2 - time1
    Theta = result_big$theta
    Theta_para = result_big$theta_para
    
    Theta[3] = object$theta[3]
    Theta_para[5:6] = object$theta_para[5:6]
    if(object$cov.model!="matern"){
      names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi")
    }else{names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi", "nu")}
    
    if(object$cov.model!="matern"){
      names(Theta_para) <- c("sigma.sq.alpha", "sigma.sq.beta",
                             "tau.sq.alpha", "tau.sq.beta",
                             "phi.alpha","phi.beta")
    }
    
    result_list$theta <- Theta
    result_list$theta_para <- Theta_para
    
  }else{
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
          cat("-------------------------------------------------------", "\n")
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
          cat("-------------------------------------------------------", "\n")
          RcppParallel::setThreadOptions(numThreads = num)
          # cat(c("Compute the nearest positive definite for covariance matrix", "\n"))
          time1 <- proc.time()
          updated_mat <- .Call("compute_Hinv_V_matrix_parallel", Inter_mat, c(beta_sigmasq, object$w_sigma_sq))
          # results_pd <- .Call("nearest_positive_definite", mat_results[-(1:p),-(1:p)], 1e-6, 100)
          object$updated_mat = updated_mat
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
          cat("-------------------------------------------------------", "\n")
          RcppParallel::setThreadOptions(numThreads = num)
          time1 <- proc.time()
          updated_mat <- .Call("compute_Hinv_V_full_p_parallel", Inter_mat, object$beta_cov ,object$w_sigma_sq, p)
          
          object$updated_mat = updated_mat
          
          if(get_para){
            cat(c("   Update spatial parameters \n"))
            cat("-------------------------------------------------------", "\n")
            ## Update tausq ###
            b_tau_update = tau.sq.IG[2] + (sum(qr.resid(qr(object$X), object$y - object$w_mu)^2) + p*object$theta[2] + sum(diag(updated_mat)[-(1:p)]))/2
            
            ### Update sigmasq ###
            LR_mat_decompose = spVB_LR_chol(object)
            prior_mat = spVB_prior(object)
            
            B_q = LR_mat_decompose$V
            F_q = LR_mat_decompose$F
            B_mat = prior_mat$B_mat
            F_mat = prior_mat$F_mat
            
            set.seed(1)
            sim <- matrix(rnorm(Trace_N*(n+p)))
            u <- solve(B_q,matrix(sim, ncol = Trace_N)*sqrt(F_q))
            
            MNNGP <- t(B_mat) %*% solve(F_mat) %*% B_mat
            U <- u[-(1:p), 1:Trace_N, drop = FALSE]
            
            b_sigma_update = sigma.sq.IG[2] + (sum(colSums((MNNGP %*% U) * U))/Trace_N + sum((B_mat %*% object$w_mu)^2 / diag(F_mat)))*object$theta[1]*0.5
            
          }
          time2 <- proc.time()
        }else{
          num <- n_omp
          cat("-------------------------------------------------------", "\n")
          cat(c("   Default Number of Threads is", unname(num)), "\n")
          cat(c("   Compute the Linear response for variance \n"))
          cat("-------------------------------------------------------", "\n")
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
      cat("-------------------------------------------------------", "\n")
      Inter_mat <- .Call("construct_I_VH_nop", object$n, object$theta[2], object$nnIndxLU, object$nnIndx,
                         object$numIndxCol, object$nnIndxnnCol, object$cumnumIndxCol,
                         object$B, object$F, object$w_sigma_sq)
      RcppParallel::setThreadOptions(numThreads = num)
      
      if(get_mat){
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_full_nop_parallel", Inter_mat, object$w_sigma_sq)
        object$updated_mat <- updated_mat
        
        if(get_para){
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
          
        }
        time2 <- proc.time()
        
      }else{
        time1 <- proc.time()
        updated_mat <- .Call("compute_Hinv_V_diag_nop_parallel", Inter_mat, object$w_sigma_sq)
        time2 <- proc.time()
      }
      
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
    
    result_list = object
    result_list$VI_family <-  "MFA-LR"
    result_list$updated_mat <- updated_mat
    result_list$LR_time <- time2 - time1
    
    if(get_para){
      Theta[2] <- b_tau_update/Theta_para[3]
      Theta_para[4] <- b_tau_update
      
      Theta[1] <- b_sigma_update/Theta_para[1]
      Theta_para[2] <- b_sigma_update
      
      result_list$theta <- Theta
      result_list$theta_para <- Theta_para
      result_list$B_q <- B_q
      result_list$F_q <- F_q
      result_list$sim <- sim
      result_list$B_mat <- B_mat
      result_list$F_mat <- F_mat
    }
  }
  
  
  

  return(result_list)
}
