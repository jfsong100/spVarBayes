#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace Eigen;


extern "C" SEXP compute_Hinv_V_diagonal(SEXP H_, SEXP V_diag_, SEXP max_iter_, SEXP tol_) {
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);
  int max_iter = as<int>(max_iter_);
  double tol = as<double>(tol_);

  int n = H.rows(); 
  NumericVector results(n);

  // Perform Sparse LU decomposition of H
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }

  // Loop over all i = 1, ..., n and solve H z_i = e_i
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
    e_i(i) = 1.0;

    Eigen::VectorXd z_i = solver.solve(e_i);
    if (solver.info() != Success) {
      Rcpp::stop("Solving for z_i failed!");
    }

    results[i] = z_i[i] * V_diag[i];
  }

  return wrap(results);
}

// Parallel worker to solve each system H z_i = e_i
struct ParallelSolver : public Worker {
  const Eigen::SparseMatrix<double> &H;  
  const Eigen::VectorXd &V_diag;
  Rcpp::NumericVector &results;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver;

  // Constructor to initialize the worker
  ParallelSolver(const Eigen::SparseMatrix<double> &H_,
                 const Eigen::VectorXd &V_diag_,
                 Rcpp::NumericVector &results_,
                 Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver_)
    : H(H_), V_diag(V_diag_), results(results_), solver(solver_) {}

  // Parallel worker function
  void operator()(std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      // standard basis vector e_i
      Eigen::VectorXd e_i = Eigen::VectorXd::Zero(H.rows());
      e_i(i) = 1.0;

      Eigen::VectorXd z_i = solver.solve(e_i);

      if (solver.info() == Eigen::Success) {
       
        results[i] = z_i[i] * V_diag[i];
      } else {
        results[i] = NA_REAL;
      }
    }
  }
};


extern "C" SEXP compute_Hinv_V_diagonal_parallel(SEXP H_, SEXP V_diag_) {
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);

  int n = H.rows();
  Rcpp::NumericVector results(n);

  // Perform Sparse LU decomposition of H
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Eigen::Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }

  // Create the parallel solver worker
  ParallelSolver solver_worker(H, V_diag, results, solver);

  parallelFor(0, n, solver_worker);

  return wrap(results);
}


