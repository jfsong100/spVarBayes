spVB_LR_chol = function(object){
  
  out <- list()
  out$call <- match.call()

  family.indx <- 1
  
  n <- object$n
  
  myknn <- function(i,s,m){
    if(m>=(i-1)) im<-1:(i-1)
    else
    {
      dist=fields::rdist(s[c(1,i),],s[c(1,1:(i-1)),])[-1,-1]
      im<-(order(dist)[1:m])
    }
    return(im)
  }
  
  coords <- object$coords
  m_dat <- object$n.neighbors
  Sigma_star <- object$updated_mat
  
  if(object$covariates){
    p <- length(object$beta)
    imvec_w <- sapply(2:n,myknn,coords,m_dat)
    imvec_beta <- sapply(2:p,function(i){1:(i-1)})
    
    imvec <- list()
    for(i in 2:p){
      imvec[[i-1]] <- 1:(i-1)
    }
    
    imvec[[p+1-1]] <- 1:p
    for(i in 2:n){
      imvec[[p+i-1]] <- c(1:p,imvec_w[[i-1]]+p)
    }
    
    BF_list <- lapply(2:(n+p), function(i){
      B_list <- solve(Sigma_star[imvec[[i - 1]], imvec[[i - 1]]], Sigma_star[i, imvec[[i - 1]]])
      F_list <- Sigma_star[i,i] - sum(B_list * Sigma_star[i, imvec[[i - 1]]])
      list(B_list = B_list, F_list = F_list)
    })
    
    colind <- c(1:(n+p),unlist(imvec))
    mi=c(1:(p-1),
         p,
         p+(1:(m_dat-1)),
         p+(rep(m_dat,n-m_dat)))
    rowind <- c(1:(n+p),unlist(sapply(2:(n+p), function(i,mi) rep(i,mi[i-1]), mi)))
    
    B_lists <- lapply(BF_list, function(x) x$B_list)
    F_lists <- lapply(BF_list, function(x) x$F_list)
    
    V=sparseMatrix(i=rowind,j=colind,x=c(rep(1,(n+p)),-unlist(B_lists)),dims=c(n+p,n+p))
    F <- c(Sigma_star[1,1],unlist(F_lists))

    out$V <- V
    out$F <- F

  }else{
    imvec <- sapply(2:n,myknn,coords,m_dat)
    BF_list = lapply(2:(n), function(i){
      B_list = solve(Sigma_star[imvec[[i - 1]], imvec[[i - 1]]], Sigma_star[i, imvec[[i - 1]]])
      F_list = Sigma_star[i,i] - sum(B_list * Sigma_star[i, imvec[[i - 1]]])
      list(B_list = B_list, F_list = F_list)
    })
    
    colind <- c(1:n,unlist(imvec))
    mi <- c(1:(m_dat-1),rep(m_dat,n-m_dat))
    
    rowind <- c(1:n,unlist(sapply(2:n, function(i,mi) rep(i,mi[i-1]), mi)))
    
    B_lists <- lapply(BF_list, function(x) x$B_list)
    F_lists <- lapply(BF_list, function(x) x$F_list)
    
    V=sparseMatrix(i=rowind,j=colind,x=c(rep(1,n),-unlist(B_lists)),dims=c(n,n))
    F <- c(Sigma_star[1,1],unlist(F_lists))

    out$V <- V
    out$F <- F
  }
  out
}