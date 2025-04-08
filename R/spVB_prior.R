spVB_prior = function(object){
  
  out <- list()
  out$call <- match.call()
  
  nnIndx = object$nnIndx
  B = object$B
  F = object$F
  n = object$n
  m_dat = object$n.neighbors
    
  if(m_dat==1){
    rowind = c(1:n,2:n)
  }else{
    mi=c(1:(m_dat-1),rep(m_dat,n-m_dat))
    rowind = c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
  }
  
  colind = c(1:n,nnIndx+1)
  
  B_mat = sparseMatrix(i = rowind,
                          j = colind,
                          x = c(rep(1,n),-B),dims=c(n,n))
  
  F_mat = sparseMatrix(i = seq(1,n,1),
                          j = seq(1,n,1),
                          x = F,dims=c(n,n))

  out$B_mat = B_mat
  out$F_mat = F_mat
  out
}