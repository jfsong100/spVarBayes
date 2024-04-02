spVarBayes_NNGP <- function(y, X, coords, zeta.sq = 1, 
                            tau.sq = 1, phi = 1, nu = 1.5, n.neighbors = 15, n.neighbors.vi = 1, 
                            n_omp = 1,converge_percent = 0.5,
                              cov.model = "exponential", search.type = "tree", 
                              zeta.sq.IG = c(0.1,1), 
                              tau.sq.IG = c(0.1,0.1), 
                              phi.range = c(1.5,3), 
                              var_input = NULL,
                              phi_input = NULL,
                              verbose = TRUE, 
                              tol = 12, 
                              vi_threshold = 0.01, 
                              rho = 0.85, 
                              max_iter = 500, 
                              N_phi = 5, Trace_N = 100, 
                              covariates = TRUE, initial_mu = 1, phi_max_iter = 25, 
                              rho_phi = 0.5, mini_batch_size = 128, mini_batch = F,
                              phi_methods = "sgd", reorder = F, shuffle = F
){
  
  n <- nrow(coords)
  p <- 0
  if(covariates){
    p <- ncol(X)
  }
  
  ##Coords
  if(!is.matrix(coords)){stop("error: coords must n-by-2 matrix of xy-coordinate locations")}
  if(ncol(coords) != 2 || nrow(coords) != n){
    stop("error: either the coords have more than two columns or then number of rows is different than
         data used in the model formula")
  }
  coords <- round(coords, tol)

  if(nu < 0 ){stop("error: nu must be non-negative")}

  n.neighbors.opt <- min(100, n-1)

  if(!is.null(n.neighbors)){
    if(n.neighbors < n.neighbors.opt) warning('We recommend using higher n.neighbors especially for small Phi')
  }

  if(is.null(n.neighbors)){
    n.neighbors <- n.neighbors.opt
  }

  
  ##Covariance model
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1
  storage.mode(cov.model.indx) <- "integer"

  ####################################################
  ##Priors
  ####################################################
  if(is.null(var_input)|is.null(phi_input)){
    BRISC_input = BRISC_estimation(coords = coords, y = y, x = X, sigma.sq = 1,
                                   tau.sq = 0.1, phi = 1,
                                   nu = 0.5, n.neighbors = 15,
                                   n_omp = 1, 
                                   cov.model = "exponential",
                                   search.type = "tree",
                                   stabilization = NULL,
                                   pred.stabilization = 1e-5,
                                   verbose = TRUE, eps = 2e-05,
                                   nugget_status = 1, 
                                   neighbor = NULL, tol = 12)$Theta
    BRISC_phi_input = BRISC_input[3]
    BRISC_var_input = 1/(1/BRISC_input[2]+1/BRISC_input[1])
    
    if(is.null(var_input)){
      print(c("Using BRISC estimation for variance of w"))
      print(c("BRISC_estimation for var is",BRISC_var_input))
      var_input = BRISC_var_input
    }
    if(is.null(phi_input)){
      print(c("Using BRISC estimation for phi"))
      print(c("BRISC_estimation for phi is",BRISC_phi_input))
      phi_input = BRISC_phi_input
    }
    
    
  }
  
  
  #zeta.sq.IG <- 0
  #tau.sq.IG <- 0
  nu.Unif <- 0
  #phi.range <- 0

  
  storage.mode(zeta.sq.IG) <- "double"
  storage.mode(tau.sq.IG) <- "double"
  storage.mode(phi.range) <- "double"
  storage.mode(nu.Unif) <- "double"
  
  ##Parameter values
  if(cov.model!="matern"){
    initiate <- c(zeta.sq, tau.sq, phi)
    names(initiate) <- c("zeta.sq", "tau.sq", "phi")
  }
  else{
    initiate <- c(zeta.sq, tau.sq, phi, nu)
    names(initiate) <- c("zeta.sq", "tau.sq", "phi", "nu")}

  zeta.sq.starting <- zeta.sq.IG
  tau.sq.starting <- tau.sq.IG
  phi.starting <- phi.range
  nu.starting <- nu

  storage.mode(zeta.sq.starting) <- "double"
  storage.mode(tau.sq.starting) <- "double"
  storage.mode(phi.starting) <- "double"
  storage.mode(nu.starting) <- "double"


  ##Search type
  search.type.names <- c("brute", "tree", "cb")
  if(!search.type %in% search.type.names){
    stop("error: specified search.type '",search.type,"' is not a valid option; choose from ", paste(search.type.names, collapse=", ", sep="") ,".")
  }
  search.type.indx <- which(search.type == search.type.names)-1
  storage.mode(search.type.indx) <- "integer"

  ##Option for Multithreading if compiled with OpenMp support
  n.omp.threads <- as.integer(n_omp)
  storage.mode(n.omp.threads) <- "integer"
  
  if(length(var_input)==1){
    var_input = rep(var_input,n)
  }

  fix_nugget <- 1
  ##type conversion
  storage.mode(n) <- "integer"
  storage.mode(p) <- "integer"
  storage.mode(coords) <- "double"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(n.neighbors.vi) <- "integer"
  storage.mode(verbose) <- "integer"
  storage.mode(fix_nugget) <- "double"
  storage.mode(max_iter) <- "integer"
  storage.mode(rho) <- "double"
  storage.mode(vi_threshold) <- "double"
  storage.mode(N_phi) <- "integer"
  storage.mode(Trace_N) <- "integer"
  storage.mode(converge_percent) <- "double"
  storage.mode(var_input) <- "double"
  storage.mode(phi_input) <- "double"
  storage.mode(initial_mu) <- "integer"
  storage.mode(phi_max_iter) <- "integer"
  storage.mode(rho_phi) <- "double"
  storage.mode(mini_batch_size) <- "integer"
  
  p1<- proc.time()
  ord = 1:n
  if(reorder){
    ord <- order(coords[,1] + coords[,2])
    coords <- coords[ord,]
  }
 
  if(p>0){X <- X[ord,,drop=FALSE]}
  y <- y[ord]
  
  print(c("Using NNGP Gaussian family for Variational Approximation"))
  print(paste("Using",n.neighbors,"nearest neighbors for the prior"))
  print(paste("Using",n.neighbors.vi,"nearest neighbors for the variational family"))
  
  if(covariates){
    if(mini_batch){
      print("Include Covariates X and using Mini Batch")
      result <- .Call("spVarBayes_NNGP_mb_betacpp", y, X, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, zeta.sq.IG, tau.sq.IG, phi.range, nu.Unif, zeta.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, vi_threshold,converge_percent,var_input,phi_input,phi_max_iter, rho_phi, initial_mu,mini_batch_size, PACKAGE = "spVarBayes")
    }else{
      print("Include Covariates X and using Full Batch")
      result <- .Call("spVarBayes_NNGP_betacpp", y, X, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, zeta.sq.IG, tau.sq.IG, phi.range, nu.Unif, zeta.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, vi_threshold,converge_percent,var_input,phi_input,phi_max_iter, rho_phi, initial_mu, PACKAGE = "spVarBayes")
    }
  }else{
    if(mini_batch){
      print("No Covariates X and using Mini Batch with Scaled Beta for Phi")
      result <- .Call("spVarBayes_NNGP_nocovariates_mb_betacpp", y, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, zeta.sq.IG, tau.sq.IG, phi.range, nu.Unif, zeta.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, vi_threshold,converge_percent,var_input,phi_input,phi_max_iter, rho_phi, initial_mu,mini_batch_size, PACKAGE = "spVarBayes")
    }else{
      print("No Covariates X and using Full Batch with Scaled Beta for Phi")
      result <- .Call("spVarBayes_NNGP_nocovariates_betacpp",  y, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, zeta.sq.IG, tau.sq.IG, phi.range, nu.Unif, zeta.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, vi_threshold,converge_percent,var_input,phi_input,phi_max_iter, rho_phi, initial_mu, PACKAGE = "spVarBayes")
    }
  }
  

  p2 <- proc.time()

  Theta <- result$theta
  if(cov.model!="matern"){
    names(Theta) <- c("zeta.sq", "tau.sq", "phi")
  }else{names(Theta) <- c("zeta.sq", "tau.sq", "phi", "nu")}
  Theta_para <- result$theta_para
  if(cov.model!="matern"){
    names(Theta_para) <- c("zeta.sq.alpha", "zeta.sq.beta",
                           "tau.sq.alpha", "tau.sq.beta",
                           "phi.alpha","phi.beta")
  }
  
  result_list <- list ()
  result_list$n <- n
  result_list$y <- y
  result_list$X <- X
  result_list$coords <- coords
  result_list$n.neighbors <- n.neighbors
  result_list$n.neighbors.vi <- n.neighbors.vi
  result_list$cov.model <- cov.model
  result_list$time <-  p2 - p1
  result_list$nnIndxLU <- result$nnIndxLU
  result_list$CIndx <- result$CIndx
  result_list$nnIndx <- result$nnIndx
  result_list$numIndxCol <- result$numIndxCol
  result_list$cumnumIndxCol <- result$cumnumIndxCol
  result_list$nnIndxCol <- result$nnIndxCol
  result_list$nnIndxnnCol <- result$nnIndxnnCol
  result_list$nnIndxLU_vi <- result$nnIndxLU_vi
  result_list$nnIndx_vi <- result$nnIndx_vi
  result_list$numIndxCol_vi <- result$numIndxCol_vi
  result_list$cumnumIndxCol_vi <- result$cumnumIndxCol_vi
  result_list$nnIndxCol_vi <- result$nnIndxCol_vi
  result_list$nnIndxnnCol_vi <- result$nnIndxnnCol_vi
  result_list$B <- result$B
  result_list$F <- result$F
  if(covariates){
    result_list$beta <- result$beta
    result_list$beta_cov <- result$beta_cov
  }
  result_list$theta <- Theta
  result_list$phi.range <- phi.range
  result_list$theta_para <-  Theta_para
  result_list$w_mu <- result$w_mu
  result_list$A_vi <-  result$A_vi
  result_list$S_vi <-  result$S_vi
  result_list$uiIndx <-  result$uiIndx
  result_list$uIndx <-  result$uIndx
  result_list$uIndxLU <-  result$uIndxLU
  result_list$iter <-  result$iter
  result_list$ELBO_vec <-  result$ELBO_vec
  result_list$ord <-  ord
  
  class(result_list) <- "spVarBayes"
  
  result_list
}
