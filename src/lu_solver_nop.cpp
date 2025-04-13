#include <RcppEigen.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace Eigen;
using namespace RcppParallel;

extern "C" SEXP compute_Hinv_V_diag_nop(SEXP H_, SEXP V_diag_) {
  
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);
  int n = H.rows();
  
  NumericVector diag_res(n);
  
  SparseLU<SparseMatrix<double>> solver_diag;
  solver_diag.compute(H);
  if (solver_diag.info() != Success) {
    stop("Sparse LU decomposition failed in solver_diag!");
  }
  
  for (int j = 0; j < n; ++j) {
    Eigen::VectorXd v_col = Eigen::VectorXd::Zero(n);
    v_col[j] = V_diag[j];
    Eigen::VectorXd z_col = solver_diag.solve(v_col);
    if (solver_diag.info() != Success) {
      stop("Solving column failed in solver_diag!");
    }
    diag_res[j] = z_col[j];
  }
  
  return wrap(diag_res);
}

struct ParallelMatDiagSolver : public Worker {
  const SparseMatrix<double>& H;
  const VectorXd& V_diag;
  NumericVector &results; 
  SparseLU<SparseMatrix<double>>& solver_diag;
  
  // Constructor to initialize the worker
  ParallelMatDiagSolver(const SparseMatrix<double> &H_,
                        const VectorXd &V_diag_,
                        NumericVector &results_,
                        SparseLU<SparseMatrix<double>> &solver_diag_)
    : H(H_), V_diag(V_diag_), results(results_), solver_diag(solver_diag_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int n = H.rows();
    for (std::size_t j = begin; j < end; ++j) {
      Eigen::VectorXd v_col = Eigen::VectorXd::Zero(n);
      v_col[j] = V_diag[j];
      Eigen::VectorXd z_col = solver_diag.solve(v_col);
      if (solver_diag.info() == Success) {
        results[j] = z_col[j];
      } else {
        results[j] = NA_REAL;
      }
    }
  }
};


extern "C" SEXP compute_Hinv_V_diag_nop_parallel(SEXP H_, SEXP V_diag_) {
  const SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int n = H.rows();
  
  NumericVector results(H.rows());
  
  SparseLU<SparseMatrix<double>> solver_diag;
  solver_diag.compute(H);
  if (solver_diag.info() != Success) {
    stop("Sparse LU decomposition failed in solver_diag!");
  }
  
  
  ParallelMatDiagSolver worker(H, V_diag, results, solver_diag);
  parallelFor(0, n, worker);
  
  return wrap(results);
}