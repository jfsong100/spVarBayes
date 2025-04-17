#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

extern "C" SEXP compute_Hinv_V_topblock(SEXP H_, SEXP V_top_, SEXP p_) {

  const SparseMatrix<double>& H     = as<Map<SparseMatrix<double>>>(H_);
  const MatrixXd&        V_top     = as<Map<MatrixXd>>(V_top_);
  int                    p         = as<int>(p_);
  int                    n_plus_p = H.rows();

  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success)
    stop("Sparse LU decomposition of H failed");
  
  // 1) Build the p x p block Z_top = (H^{-1} * V_top)
  MatrixXd Z_top(p, p);
  for(int i = 0; i < p; ++i) {
    VectorXd v = VectorXd::Zero(n_plus_p);
    for(int j = 0; j < p; ++j) {
      v[j] = V_top(j, i);
    }
    VectorXd z = solver.solve(v);
    if (solver.info() != Success)
      stop("Solve for top block column " + std::to_string(i) + " failed");
    
    // extract only the first p entries
    for(int j = 0; j < p; ++j) {
      Z_top(j, i) = z[j];
    }
  }
  
  // 2) Symmetrize
  Z_top = 0.5 * (Z_top + Z_top.transpose());
  
  // 3) Eigen-correct to enforce PSD
  SelfAdjointEigenSolver<MatrixXd> es(Z_top);
  VectorXd eigenvals = es.eigenvalues();
  MatrixXd eigenvecs = es.eigenvectors();
  
  for(int i = 0; i < p; ++i) {
    if (eigenvals[i] < 0) {
      eigenvals[i] = 1e-8;
    }
  }
  
  MatrixXd Z_top_psd = eigenvecs * eigenvals.asDiagonal() * eigenvecs.transpose();
  
  return wrap(Z_top_psd);
}

