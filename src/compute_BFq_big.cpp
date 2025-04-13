#include <RcppEigen.h>
#include <Rcpp.h>
#include <RcppParallel.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>

#include <vector>

using namespace Rcpp;
using namespace Eigen;
using namespace RcppParallel;

struct UpdateWorker : public Worker {
 
  const SparseLU<SparseMatrix<double> > *solver_ptr;
  const NumericVector V_diag;     
  const IntegerVector nnIndx;       
  const IntegerVector nnIndxLU;      
  const int n;                    
  NumericVector B_q;
  NumericVector F_q;
  
  UpdateWorker(const SparseLU<SparseMatrix<double> >* solver_ptr_,
               const NumericVector &V_diag_,
               const IntegerVector &nnIndx_,
               const IntegerVector &nnIndxLU_,
               int n_,
               NumericVector &B_q_,
               NumericVector &F_q_)
    : solver_ptr(solver_ptr_), V_diag(V_diag_), nnIndx(nnIndx_),
      nnIndxLU(nnIndxLU_), n(n_), B_q(B_q_), F_q(F_q_) { }
  
  void operator()(std::size_t begin, std::size_t end) {
    int inc = 1;
    double one = 1.0, zero = 0.0;
    char lower = 'L';
    int info = 0;
    
    for (std::size_t i = begin; i < end; i++) {
      if (i == 0) {
      
        VectorXd v = VectorXd::Zero(n);
        v(0) = V_diag[0];
        VectorXd solVec = solver_ptr->solve(v);
        B_q[i] = 0;
        F_q[i] = solVec(0);
      } else {

        int r = nnIndxLU[n + i];
        int startIdx = nnIndxLU[i];
        
        std::vector<double> subVec(r, 0.0);
        std::vector<double> subMat(r * r, 0.0);
        
        for (int k = 0; k < r; k++) {
          int global_col = nnIndx[startIdx + k];
          VectorXd v = VectorXd::Zero(n);
          v(global_col) = V_diag[global_col];
          VectorXd solVec = solver_ptr->solve(v);
          subVec[k] = solVec(i);
          // For each local index l, store the corresponding entry from solVec into the submatrix.
          for (int l = 0; l < r; l++) {
            int global_col_l = nnIndx[startIdx + l];
            subMat[k * r + l] = solVec(global_col_l);
          }
        }
        
        // Invert the submatrix using LAPACK's dpotrf and dpotri.
        F77_NAME(dpotrf)(&lower, &r, subMat.data(), &r, &info FCONE);
        if (info != 0) {
          ::Rf_error("c++ error: dpotrf failed\n");
        }
        F77_NAME(dpotri)(&lower, &r, subMat.data(), &r, &info FCONE);
        if (info != 0) {
          ::Rf_error("c++ error: dpotri failed\n");
        }
        
        F77_NAME(dsymv)(&lower, &r, &one, subMat.data(), &r,
                 subVec.data(), &inc, &zero, &B_q[startIdx], &inc FCONE);
        
        VectorXd v = VectorXd::Zero(n);
        v(i) = V_diag[i];
        VectorXd solVec_diag = solver_ptr->solve(v);
        double HinvVii = solVec_diag(i);
        
        // Compute the dot product of the computed B_q segment and subVec.
        double dot_val = F77_NAME(ddot)(&r, &B_q[startIdx], &inc, subVec.data(), &inc);
        F_q[i] = HinvVii - dot_val;
      } 
    } 
  } 
};

 extern "C" SEXP updateBFq_cpp(SEXP H_, 
                              SEXP V_diag_, 
                              SEXP nnIndx_, 
                              SEXP nnIndxLU_, 
                              SEXP B_q_, 
                              SEXP F_q_) { 


     Map<SparseMatrix<double> > H(as<Map<SparseMatrix<double> > >(H_));
     int n = H.rows();
     
     NumericVector V_diag(V_diag_);
     IntegerVector nnIndx(nnIndx_);
     IntegerVector nnIndxLU(nnIndxLU_);
     NumericVector B_q(B_q_);
     NumericVector F_q(F_q_);
     
     // Compute the sparse LU factorization of H.
     SparseLU<SparseMatrix<double> > solver;
     solver.compute(H);
     if (solver.info() != Success) {
       stop("Sparse LU decomposition failed!");
     }
     
     // Create the parallel worker with the solver pointer and input/output vectors.
     UpdateWorker worker(&solver, V_diag, nnIndx, nnIndxLU, n, B_q, F_q);
     
     // Run the parallelFor loop over rows [0, n).
     parallelFor(0, n, worker);
     
     // Return the updated vectors in a list.
     return List::create(Named("B_q") = B_q,
                         Named("F_q") = F_q);
   
   
 }
 