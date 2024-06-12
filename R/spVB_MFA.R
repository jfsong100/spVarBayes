spVB_MFA <- function(y, X, coords, covariates = TRUE, n.neighbors = 15,
                     sigma.sq = 1, tau.sq = 0.5, phi = NULL, sigma.sq.IG = c(0.1,1),
                     tau.sq.IG = c(0.1,0.1), phi.range = NULL, var_input = NULL,
                     n_omp = 1, cov.model = "exponential", nu = 1.5, search.type = "tree",tol = 12,
                     verbose = FALSE, max_iter = 2000, min_iter = 750, stop_K = FALSE, K = 20,
                     N_phi = 5, Trace_N = 50, phi_max_iter = 50, rho = 0.85,
                     mini_batch = T, mini_batch_size = 256, reorder = "Sum_coords"){



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

  if(tau.sq < 0 ){stop("error: tau.sq must be non-negative")}
  if(sigma.sq < 0 ){stop("error: sigma.sq must be non-negative")}
  #if(phi < 0 ){stop("error: phi must be non-negative")}
  if(nu < 0 ){stop("error: nu must be non-negative")}

  n.neighbors.opt <- min(100, n-1)

  if(!is.null(n.neighbors)){
    if(n.neighbors < n.neighbors.opt) warning('We recommend using higher n.neighbors especially for small Phi')
    if(n.neighbors < 10) warning('We recommend using higher n.neighbors at least 10')
  }

  if(is.null(n.neighbors)){
    n.neighbors <- n.neighbors.opt
  }

  if(!mini_batch){
    mini_batch_size = n
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

  ##Stabilization
  # if(is.null(stabilization)){
  #   if(cov.model == "exponential"){
  #     stabilization = FALSE
  #   }
  #   if(cov.model != "exponential"){
  #     stabilization = TRUE
  #   }
  # }
  #
  # if(!isTRUE(stabilization)){
  #   if(cov.model != "exponential"){
  #     warning('We recommend using stabilization for spherical, Matern and Gaussian model')
  #   }
  # }
  initial_mu = 1

  if(reorder == "Sum_coords"){
    print("Using Sum_coords ordering")
    ord <- order(coords[,1] + coords[,2])
    coords <- coords[ord,]
  }else if(reorder == "AMMD"){
    print("Using Maxmin ordering")
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

  if(BRISC_input[1]<1){
    warning('We recommend using larger number of Monte Carlo samples especially for small sigma.sq')
  }

  if(is.null(var_input)|is.null(phi)){
    if(is.null(var_input)){
      print(c("Using BRISC estimation for variance of w"))
      print(c("BRISC_estimation for var is",BRISC_var_input))
      var_input = BRISC_var_input
    }
    if(is.null(phi)){
      print(c("Using BRISC estimation for phi"))
      print(c("BRISC_estimation for phi is",BRISC_phi_input))
      phi = BRISC_phi_input
    }
  }

  if(is.null(phi.range)){
    if(n>10000){
      len = max(coords[,1]) - min(coords[,1])
      wid = max(coords[,2]) - min(coords[,2])
      d_max = sqrt(len^2 + wid^2)
      warning('We recommend a phi range input especially for large sample size')
    }else{
      d_max = max(as.matrix(dist(coords)))
    }
    #d_max = 10*sqrt(2)
    phi_min = min(3/d_max,max(BRISC_phi_input-0.5,0.01))
    phi_max = max(30/d_max,BRISC_phi_input+1)
    phi.range = c(phi_min,phi_max)
  }


  if(phi < phi.range[1] | phi > phi.range[2]){
    stop("error: phi input must be in the phi range")
  }


  #sigma.sq.IG <- 0
  #tau.sq.IG <- 0
  nu.Unif <- 0
  #phi.range <- 0
  if(length(var_input)==1){
    var_input = rep(var_input,n)
  }

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


  if(!mini_batch){
    mini_batch_size = n
  }
  ##Option for Multithreading if compiled with OpenMp support
  n.omp.threads <- as.integer(n_omp)
  storage.mode(n.omp.threads) <- "integer"

  fix_nugget <- 1
  ##type conversion
  storage.mode(n) <- "integer"
  storage.mode(p) <- "integer"
  storage.mode(coords) <- "double"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(verbose) <- "integer"
  storage.mode(fix_nugget) <- "double"
  storage.mode(max_iter) <- "integer"
  storage.mode(rho) <- "double"
  storage.mode(min_iter) <- "integer"
  storage.mode(K) <- "integer"

  #storage.mode(vi_threshold) <- "double"
  storage.mode(N_phi) <- "integer"
  storage.mode(Trace_N) <- "integer"
  #storage.mode(converge_percent) <- "double"
  storage.mode(phi) <- "double"
  storage.mode(var_input) <- "double"
  storage.mode(initial_mu) <- "integer"
  storage.mode(phi_max_iter) <- "integer"
  storage.mode(mini_batch_size) <- "integer"
  #storage.mode(rho_phi) <- "double"

  p1<- proc.time()


  print(c("Using Mean-field Approximation family for Variational Approximation"))

  if(covariates){
    print(c("With covariates"))
    result <- .Call("spVarBayes_MFA_mb_betacpp",
                    y, X, n, p, n.neighbors, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, var_input,phi, phi_max_iter,initial_mu,mini_batch_size, min_iter, K, stop_K,PACKAGE = "spVarBayes")
  }else{
    print(c("No covariates"))
    result <- .Call("spVarBayes_MFA_nocovariates_mb_betacpp",
                    y,    n, p, n.neighbors, coords, cov.model.indx, rho, sigma.sq.IG, tau.sq.IG, phi.range, nu.Unif, sigma.sq.starting, tau.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, fix_nugget, N_phi, Trace_N, max_iter, var_input,phi, phi_max_iter,initial_mu,mini_batch_size, min_iter, K, stop_K,PACKAGE = "spVarBayes")
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
  result_list$cov.model <- cov.model
  result_list$time <-  p2 - p1
  result_list$nnIndxLU <- result$nnIndxLU
  # result_list$CIndx <- result$CIndx
  result_list$nnIndx <- result$nnIndx
  result_list$B <- result$B
  result_list$F <- result$F
  if(covariates){
    result_list$beta <- result$beta
    beta_cov = matrix(result$beta_cov,length(result$beta))
    beta_cov = beta_cov + t(beta_cov) - diag(diag(beta_cov))
    result_list$beta_cov <- beta_cov
  }
  result_list$theta <- Theta
  result_list$phi.range <- phi.range
  result_list$theta_para <-  Theta_para
  result_list$w_mu <- result$w_mu
  result_list$w_sigma_sq <-  result$w_sigma_sq
  result_list$iter <-  result$iter - 1
  result_list$ELBO_vec <-  result$ELBO_vec
  result_list$ord <-  ord
  result_list$VI_family <-  "MFA"
  result_list$covariates <- covariates
  # result_list$gradient_mu_vec <-  result$gradient_mu_vec
  # result_list$MFA_sigmasq_grad_vec <-  result$MFA_sigmasq_grad_vec

  class(result_list) <- "spVarBayes"

  result_list
}
