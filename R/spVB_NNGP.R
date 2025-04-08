spVB_NNGP <- function(y, X, coords, covariates = TRUE, n.neighbors = 15, n.neighbors.vi = 3,
                      sigma.sq = 1, tau.sq = 0.5, phi = NULL, sigma.sq.IG = c(0.1,1),
                      tau.sq.IG = c(0.1,0.1), phi.range = NULL, var_input = NULL,
                      n_omp = 1, cov.model = "exponential", nu = 1.5, search.type = "tree",tol = 12,
                      verbose = FALSE, max_iter = 2000, min_iter = 1000, stop_K = FALSE, K = 20,
                      N_phi = 5, Trace_N = 30, phi_max_iter = 0, rho = 0.85,
                      mini_batch = FALSE, mini_batch_size = 128, ord_type = "Sum_coords", joint = FALSE){

  n <- nrow(coords)
  p <- 0
  if(covariates){
    if(is.null(X)){stop("error: X must be input if covariates is TRUE")}
    if(!is.matrix(X)){stop("error: X must be a matrix")}
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
    if(n.neighbors < n.neighbors.opt) warning('We recommend using higher n.neighbors especially for small Phi.')
    if(n.neighbors < 10) warning('We recommend using higher n.neighbors at least 10.')
  }

  if(!is.null(n.neighbors.vi)){
    if(n.neighbors.vi == 0) stop("error: With n.neighbors.vi = 0, the model is reduced to a mean field approximation. Try to use spVB-MFA function")
  }

  if(is.null(n.neighbors)){
    n.neighbors <- n.neighbors.opt
  }

  if(!mini_batch){
    mini_batch_size <- n
  }

  if(mini_batch & mini_batch_size > n){
    stop("error: mini batch size must be smaller or equal to n")
  }

  if(rho <=0 | rho>=1){
    stop("error: rho should be a value between 0 and 1")
  }

  ##Covariance model
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1
  storage.mode(cov.model.indx) <- "integer"

  initial_mu <- 1
  cat("----------------------------------------", "\n")
  cat("          Model description", "\n")
  cat("----------------------------------------", "\n")
  cat(c("Using NNGP Gaussian family for Variational Approximation."), "\n")
  cat("\n")
  cat(paste("Using",n.neighbors,"nearest neighbors for the prior."), "\n")
  cat("\n")
  cat(paste("Using",n.neighbors.vi,"nearest neighbors for the variational family."), "\n")
  cat("\n")

  if(ord_type == "Sum_coords"){
    cat("Data is ordered using Sum_coords ordering.", "\n")
    cat("\n")
    ord <- order(coords[,1] + coords[,2])
    coords <- coords[ord,]
  }else if(ord_type == "AMMD"){
    cat("Data is ordered using Max-Min ordering.", "\n")
    cat("\n")
    set.seed(1)
    ord <- BRISC_order(coords, order = "AMMD")
    coords <- coords[ord,]
  }else{
    stop("error: Please insert a valid ordering scheme choice given by Sum_coords or AMMD.")
  }

  if(p>0){X <- X[ord,,drop=FALSE]}
  y <- y[ord]




  ####################################################
  ##Priors
  ####################################################
  suppressWarnings({
    sink(tempfile())
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
                                   neighbor = NULL, tol = 12, order = ord_type)$Theta
    BRISC_phi_input = BRISC_input[3]
    BRISC_var_input = 1/(1/BRISC_input[2]+1/BRISC_input[1])
    sink()
  })


  if(BRISC_input[1]<1){
    warning('We recommend using larger number of Monte Carlo samples especially for small sigma.sq.')
  }

  if(is.null(var_input)|is.null(phi)){
    if(is.null(var_input)){
      cat(c("Using BRISC estimation for diagonals elements in the variational distribuiton for spatial random effects."), "\n")
      cat("\n")
      cat(c("BRISC estimation for var is",unname(round(BRISC_var_input,6))),".", "\n")
      cat("\n")
      var_input = BRISC_var_input
    }
    if(is.null(phi)){
      cat(c("Using BRISC estimation for phi."), "\n")
      cat("\n")
      cat(c("BRISC estimation for phi is",unname(round(BRISC_phi_input,6))), ".","\n")
      cat("\n")
      phi = BRISC_phi_input
    }
  }


  if(is.null(phi.range)){
    if(n>10000){
      len <- max(coords[,1]) - min(coords[,1])
      wid <- max(coords[,2]) - min(coords[,2])
      d_max <- sqrt(len^2 + wid^2)
      warning('We recommend a phi range input especially for large sample size.')
    }else{
      d_max <- max(as.matrix(dist(coords)))
    }
    #d_max <- 10*sqrt(2)
    phi_min <- min(3/d_max,max(BRISC_phi_input-0.5,0.01))
    phi_max <- max(30/d_max,BRISC_phi_input+1)
    phi.range <- c(phi_min,phi_max)
  }


  if(phi < phi.range[1] | phi > phi.range[2]){
    stop("error: phi input must be in the phi range.")
  }


  #sigma.sq.IG <- 0
  #tau.sq.IG <- 0
  nu.Unif <- 0
  #phi.range <- 0


  storage.mode(sigma.sq.IG) <- "double"
  storage.mode(tau.sq.IG) <- "double"
  storage.mode(phi.range) <- "double"
  storage.mode(nu.Unif) <- "double"

  ##Parameter values
  # if(cov.model!="matern"){
  #   initiate <- c(sigma.sq, tau.sq, phi)
  #   names(initiate) <- c("sigma.sq", "tau.sq", "phi")
  # }
  # else{
  #   initiate <- c(sigma.sq, tau.sq, phi, nu)
  #   names(initiate) <- c("sigma.sq", "tau.sq", "phi", "nu")}

  sigma.sq.starting <- sigma.sq
  tau.sq.starting <- tau.sq
  phi.starting <- phi
  nu.starting <- nu

  storage.mode(sigma.sq.starting) <- "double"
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
    var_input <- rep(var_input,n)
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
  storage.mode(min_iter) <- "integer"
  storage.mode(K) <- "integer"
  storage.mode(stop_K) <- "integer"
  storage.mode(rho) <- "double"
  #storage.mode(vi_threshold) <- "double"
  storage.mode(N_phi) <- "integer"
  storage.mode(Trace_N) <- "integer"

  storage.mode(var_input) <- "double"
  storage.mode(phi) <- "double"
  storage.mode(initial_mu) <- "integer"
  storage.mode(phi_max_iter) <- "integer"
  # storage.mode(rho_phi) <- "double"
  storage.mode(mini_batch_size) <- "integer"

  p1<- proc.time()

  cat("----------------------------------------", "\n")
  cat("          Model fitting", "\n")
  cat("----------------------------------------", "\n")

  if(covariates){
    if(mini_batch){
      cat("spVB-NNGP model fit with covariates X and using mini batch", "\n")
      cat("\n")
      result <- .Call("spVarBayes_NNGP_mb_betacpp",
                      y, X, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter,var_input,phi,phi_max_iter, initial_mu, mini_batch_size, min_iter, K, stop_K ,PACKAGE = "spVarBayes")
    }else{
      #cat("Include Covariates X and using Full Batch")
      cat("spVB-NNGP model fit with covariates X and using full batch", "\n")
      cat("\n")
      if(joint){
        cat("Joint modeling for coefficients beta and w", "\n")
        result <- .Call("spVarBayes_NNGP_beta_w_jointcpp",
                        y, X, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter,var_input,phi,phi_max_iter, initial_mu, min_iter, K, stop_K, PACKAGE = "spVarBayes")
      }else{
        cat("Assume block independence for coefficients beta and w", "\n")
        result <- .Call("spVarBayes_NNGP_betacpp",
                        y, X, n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter,var_input,phi,phi_max_iter, initial_mu, min_iter, K, stop_K, PACKAGE = "spVarBayes")
      }
    }
  }else{
    if(mini_batch){
      #cat("No Covariates X and using Mini Batch with Scaled Beta for Phi")
      cat("spVB-NNGP model fit without covariates X and using mini batch", "\n")
      cat("\n")
      result <- .Call("spVarBayes_NNGP_nocovariates_mb_betacpp",
                      y,    n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter,var_input,phi,phi_max_iter, initial_mu,mini_batch_size, min_iter, K, stop_K, PACKAGE = "spVarBayes")
    }else{
      #cat("No Covariates X and using Full Batch with Scaled Beta for Phi")
      cat("spVB-NNGP model fit without covariates X and using full batch", "\n")
      cat("\n")
      result <- .Call("spVarBayes_NNGP_nocovariates_betacpp",
                      y,    n, p, n.neighbors, n.neighbors.vi, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, var_input,phi,phi_max_iter, initial_mu, min_iter, K, stop_K, PACKAGE = "spVarBayes")
    }
  }

  p2 <- proc.time()

  Theta <- result$theta
  if(cov.model!="matern"){
    names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi")
  }else{names(Theta) <- c("1/E[1/sigma.sq]", "1/E[1/tau.sq]", "phi", "nu")}
  Theta_para <- result$theta_para

  if(cov.model!="matern"){
    names(Theta_para) <- c("sigma.sq.alpha", "sigma.sq.beta",
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
  # result_list$CIndx <- result$CIndx
  result_list$nnIndx <- result$nnIndx
  result_list$numIndxCol <- result$numIndxCol
  result_list$cumnumIndxCol <- result$cumnumIndxCol
  result_list$nnIndxCol <- result$nnIndxCol
  result_list$nnIndxnnCol <- result$nnIndxnnCol
  result_list$nnIndxLU_vi <- result$nnIndxLU_vi
  result_list$nnIndx_vi <- result$nnIndx_vi
  # result_list$numIndxCol_vi <- result$numIndxCol_vi
  # result_list$cumnumIndxCol_vi <- result$cumnumIndxCol_vi
  # result_list$nnIndxCol_vi <- result$nnIndxCol_vi
  # result_list$nnIndxnnCol_vi <- result$nnIndxnnCol_vi
  result_list$B <- result$B
  result_list$F <- result$F
  if(covariates){
    result_list$beta <- result$beta
    beta_cov <- matrix(result$beta_cov,length(result$beta))
    if(length(result$beta) > 1){
      beta_cov <- beta_cov + t(beta_cov) - diag(diag(beta_cov))
    }
    result_list$beta_cov <- beta_cov
  }
  result_list$joint <- joint
  if(joint){
    result_list$L_beta <- result$L_beta
    result_list$A_beta <- result$A_beta
    result_list$E_vi <- result$E_vi
    # result_list$L_beta_track <- result$L_beta_track
    # result_list$A_beta_track <- result$A_beta_track
    # result_list$E_vi_track <- result$E_vi_track
    result_list$IndxLU_beta <- result$IndxLU_beta
  }
  result_list$theta <- Theta
  result_list$phi.range <- phi.range
  result_list$theta_para <-  Theta_para
  result_list$w_mu <- result$w_mu
  result_list$A_vi <-  result$A_vi
  result_list$D_vi <-  result$S_vi
  # result_list$uiIndx <-  result$uiIndx
  # result_list$uIndx <-  result$uIndx
  # result_list$uIndxLU <-  result$uIndxLU
  result_list$iter <-  result$iter - 1
  result_list$ELBO_vec <-  result$ELBO_vec
  result_list$ord <-  ord
  result_list$VI_family <-  "NNGP"
  result_list$covariates <- covariates
  # result_list$a_track <- result$a_track
  # result_list$gamma_track <- result$gamma_track

  class(result_list) <- "spVarBayes"

  result_list
}
