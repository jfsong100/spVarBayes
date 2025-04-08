#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace Eigen;


extern "C" SEXP compute_Hinv_V_full(SEXP H_, SEXP V_top_, SEXP V_diag_, SEXP p_, SEXP max_iter_, SEXP tol_) {

  // Map the input matrices
  const SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const MatrixXd& V_top = as<Map<MatrixXd>>(V_top_);
  const VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int p = as<int>(p_);
  int max_iter = as<int>(max_iter_);
  double tol = as<double>(tol_);

  int n_plus_p = H.rows();
  int n = n_plus_p - p;

  MatrixXd Z(n_plus_p, n_plus_p);  // Full H^{-1}V

  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    stop("Sparse LU decomposition failed!");
  }

  // Top-left V block
  for (int i = 0; i < p; ++i) {
    VectorXd V_col = VectorXd::Zero(n_plus_p);
    for (int j = 0; j < p; ++j) {
      V_col[j] = V_top(j, i);
    }
    VectorXd z_i = solver.solve(V_col);
    if (solver.info() != Success) {
      stop("Solving z_i failed!");
    }
    Z.col(i) = z_i;
  }

  // Bottom-right V diagonal block
  for (int j = 0; j < n; ++j) {
    int idx = p + j;
    VectorXd V_col = VectorXd::Zero(n_plus_p);
    V_col[idx] = V_diag[j];

    VectorXd z_j = solver.solve(V_col);
    if (solver.info() != Success) {
      stop("Solving z_j failed!");
    }
    Z.col(idx) = z_j;
  }

  return wrap(Z);  // Return full H^{-1}V
}



struct ParallelMatpSolver : public Worker {
  const Eigen::SparseMatrix<double> &H;
  const Eigen::MatrixXd &V_top;
  const Eigen::VectorXd &V_diag;
  RcppParallel::RMatrix<double> results;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver;
  int p;

  ParallelMatpSolver(const Eigen::SparseMatrix<double> &H_,
                     const Eigen::MatrixXd &V_top_,
                     const Eigen::VectorXd &V_diag_,
                     Rcpp::NumericMatrix &results_,
                     Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver_,
                     int p_)
    : H(H_), V_top(V_top_), V_diag(V_diag_), results(results_), solver(solver_), p(p_) {}

  void operator()(std::size_t begin, std::size_t end) {
    int n_plus_p = H.rows();
    for (std::size_t col = begin; col < end; ++col) {
      Eigen::VectorXd v_col = Eigen::VectorXd::Zero(n_plus_p);

      if (col < p) {
        for (int row = 0; row < p; ++row) {
          v_col[row] = V_top(row, col);
        }
      } else {
        v_col[col] = V_diag[col - p];
      }

      Eigen::VectorXd z_col = solver.solve(v_col);
      if (solver.info() == Eigen::Success) {
        for (int row = 0; row < n_plus_p; ++row) {
          results(row, col) = z_col(row);
        }
      } else {
        for (int row = 0; row < n_plus_p; ++row) {
          results(row, col) = NA_REAL;
        }
      }
    }
  }
};


extern "C" SEXP compute_Hinv_V_full_p_parallel(SEXP H_, SEXP V_top_, SEXP V_diag_, SEXP p_) {
  const Eigen::SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const Eigen::MatrixXd& V_top = as<Map<MatrixXd>>(V_top_);
  const Eigen::VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int p = as<int>(p_);
  int n_plus_p = H.rows();

  Rcpp::NumericMatrix results(n_plus_p, n_plus_p);

  Eigen::SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    stop("Sparse LU decomposition failed!");
  }

  ParallelMatpSolver worker(H, V_top, V_diag, results, solver, p);
  parallelFor(0, n_plus_p, worker);

  return wrap(results);
}
