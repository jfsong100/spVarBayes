spVB_quadratic <- function(object, w, sigmasq_input, phi_input, n_omp = 1, cov.model = "exponential", nu = 1.5, search.type = "tree", ... ){

  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1
  storage.mode(cov.model.indx) <- "integer"

  ##Search type
  search.type.names <- c("brute", "tree", "cb")
  if(!search.type %in% search.type.names){
    stop("error: specified search.type '",search.type,"' is not a valid option; choose from ", paste(search.type.names, collapse=", ", sep="") ,".")
  }
  search.type.indx <- which(search.type == search.type.names)-1
  storage.mode(search.type.indx) <- "integer"

  n.omp.threads <- as.integer(n_omp)
  storage.mode(n.omp.threads) <- "integer"

  fix_nugget <- 1
  ##type conversion
  n = object$n
  m = object$m
  coords = object$coords
  storage.mode(n) <- "integer"
  storage.mode(coords) <- "double"
  storage.mode(m) <- "integer"
  storage.mode(fix_nugget) <- "double"

  storage.mode(w) <- "double"
  storage.mode(phi_input) <- "double"
  storage.mode(sigmasq_input) <- "double"


  results <- .Call("prior_densitycpp", w, sigmasq_input, phi_input,
                   n, m, coords, cov.model.indx,
                   search.type.indx, n.omp.threads, fix_nugget)

  results

}
