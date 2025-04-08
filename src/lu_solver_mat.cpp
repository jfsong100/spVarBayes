#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace Eigen;


extern "C" SEXP compute_Hinv_V(SEXP H_, SEXP V_diag_, SEXP max_iter_, SEXP tol_) {
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);
  int max_iter = as<int>(max_iter_);
  double tol = as<double>(tol_);

  int n = H.rows(); // Number of rows in H
  Eigen::MatrixXd results(n, n); 

  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }

  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
    e_i(i) = 1.0;

    Eigen::VectorXd z_i = solver.solve(e_i);
    if (solver.info() != Success) {
      Rcpp::stop("Solving for z_i failed!");
    }
    results.col(i) = z_i * V_diag(i);
    //results.row(i) = (z_i.array() * V_diag.array()).matrix().transpose();
  }

  return wrap(results);
}


struct ParallelMatSolver : public Worker {
  const Eigen::SparseMatrix<double> &H;  // Sparse matrix
  const Eigen::VectorXd &V_diag;
  Rcpp::NumericMatrix &results;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver;

  // Constructor to initialize the worker
  ParallelMatSolver(const Eigen::SparseMatrix<double> &H_,
                 const Eigen::VectorXd &V_diag_,
                 Rcpp::NumericMatrix &results_,
                 Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver_)
    : H(H_), V_diag(V_diag_), results(results_), solver(solver_) {}

  // Parallel worker function
  void operator()(std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      // standard basis vector e_i
      Eigen::VectorXd e_i = Eigen::VectorXd::Zero(H.rows());
      e_i(i) = 1.0;
      // Solve H z_i = e_i
      Eigen::VectorXd z_i = solver.solve(e_i);

      if (solver.info() == Eigen::Success) {
        // Compute e_i^T H^{-1} V e_i
        for (std::size_t j = 0; j < H.rows(); ++j) {
          results(j, i) = z_i(j) * V_diag(i);
        }
      } else {
        for (std::size_t j = 0; j < H.rows(); ++j) {
          results(j, i) = NA_REAL;
        }
      }
    }
  }
};


extern "C" SEXP compute_Hinv_V_matrix_parallel(SEXP H_, SEXP V_diag_) {
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);

  int n = H.rows();
  Rcpp::NumericMatrix results(n,n);

  // Perform Sparse LU decomposition of H
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Eigen::Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }else {
    Rcpp::Rcout << "SparseLU decomposition succeeded!" << std::endl;
  }

  // Create the parallel solver worker
  ParallelMatSolver solver_worker(H, V_diag, results, solver);

  parallelFor(0, n, solver_worker);

  return wrap(results);
}


