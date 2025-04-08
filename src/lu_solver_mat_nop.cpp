#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace Eigen;

extern "C" SEXP compute_Hinv_V_full_nop(SEXP H_, SEXP V_diag_, SEXP max_iter_, SEXP tol_) {
  const SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int max_iter = as<int>(max_iter_);
  double tol = as<double>(tol_);
  int n = H.rows();
  MatrixXd Z(n, n);
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    stop("Sparse LU decomposition failed!");
  }
  for (int j = 0; j < n; ++j) {
    VectorXd v_col = VectorXd::Zero(n);
    v_col[j] = V_diag[j];
    VectorXd z_col = solver.solve(v_col);
    if (solver.info() != Success) {
      stop("Solving column failed!");
    }
    Z.col(j) = z_col;
  }
  return wrap(Z);
}

struct ParallelMatnopSolver : public Worker {
  const Eigen::SparseMatrix<double> &H;
  const Eigen::VectorXd &V_diag;
  RcppParallel::RMatrix<double> results;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver;

  ParallelMatnopSolver(const Eigen::SparseMatrix<double> &H_,
                       const Eigen::VectorXd &V_diag_,
                       Rcpp::NumericMatrix &results_,
                       Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver_)
    : H(H_), V_diag(V_diag_), results(results_), solver(solver_) {}

  void operator()(std::size_t begin, std::size_t end) {
    int n = H.rows();
    for (std::size_t col = begin; col < end; ++col) {
      Eigen::VectorXd v_col = Eigen::VectorXd::Zero(n);
      v_col[col] = V_diag[col];
      Eigen::VectorXd z_col = solver.solve(v_col);
      if (solver.info() == Eigen::Success) {
        for (int row = 0; row < n; ++row) {
          results(row, col) = z_col(row);
        }
      } else {
        for (int row = 0; row < n; ++row) {
          results(row, col) = NA_REAL;
        }
      }
    }
  }
};

extern "C" SEXP compute_Hinv_V_full_nop_parallel(SEXP H_, SEXP V_diag_) {
  const Eigen::SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int n = H.rows();
  Rcpp::NumericMatrix results(n, n);
  Eigen::SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    stop("Sparse LU decomposition failed!");
  }
  ParallelMatnopSolver worker(H, V_diag, results, solver);
  parallelFor(0, n, worker);
  return wrap(results);
}
