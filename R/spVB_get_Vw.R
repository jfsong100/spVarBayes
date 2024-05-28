spVB_get_Vw <- function(object){

  if(!class(object)[1] == "spVarBayes"){
    stop("error: requires an output object of class spVarBayes\n")
  }

  if(object$VI_family == "NNGP"){

    n = object$n

    if(n > 10000){
      warning("For a large n, we recommend using a sampling function instead of directly obtaining the covariance matrix.")
    }

    m = object$n.neighbors.vi
    nnIndx_vi = object$nnIndx_vi
    A_vi = object$A_vi
    S_vi = object$D_vi

    if(m==1){
      rowind_vi = c(1:n,2:n)
    }else{
      mi=c(1:(m-1),rep(m,n-m))
      rowind_vi = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
    }

    colind_vi = c(1:n,nnIndx_vi+1)

    V_approx = sparseMatrix(i = rowind_vi,
                            j = colind_vi,
                            x = c(rep(1,n),-A_vi),dims=c(n,n))

    D_approx = sparseMatrix(i = seq(1,n,1),
                            j = seq(1,n,1),
                            x = S_vi,dims=c(n,n))

    V_w = solve(V_approx) %*% D_approx %*% t(solve(V_approx))

  }else if(object$VI_family == "MFA"){
    V_w = diag(object$w_sigma_sq)
  }else{
    stop("error: The input variational family is not supported\n")
  }

  return(V_w)
}
