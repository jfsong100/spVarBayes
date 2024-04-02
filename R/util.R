rigamma <- function(n, a, b){
    1/rgamma(n = n, shape = a, rate = b)
}

get_Vw = function(n, m, nnIndx_vi, A_vi, S_vi){
  
  #n = object$n
  #m = object$n.neighbors.vi
  
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

  return(V_w)
}