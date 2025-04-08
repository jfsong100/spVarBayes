#include <RcppEigen.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace Eigen;
using namespace RcppParallel;


extern "C" SEXP compute_Hinv_V_diagonal_p(SEXP H_, SEXP V_top_, SEXP V_diag_, SEXP p_, SEXP max_iter_, SEXP tol_) {
  
  const Eigen::SparseMatrix<double>& H = as<Eigen::Map<Eigen::SparseMatrix<double>>>(H_);
  const Eigen::MatrixXd& V_top = as<Eigen::Map<Eigen::MatrixXd>>(V_top_);
  const Eigen::VectorXd& V_diag = as<Eigen::Map<Eigen::VectorXd>>(V_diag_);
  int p = as<int>(p_); // Dimension of the top-left p x p submatrix
  int max_iter = as<int>(max_iter_);
  double tol = as<double>(tol_);

  int n_plus_p = H.rows();  
  int n = n_plus_p - p;    
  NumericVector results(n_plus_p);

  // Perform Sparse LU decomposition of H
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }

  // Compute the diagonal for the top-left p x p block of V
  for (int i = 0; i < p; ++i) {
    Eigen::VectorXd V_column = Eigen::VectorXd::Zero(n_plus_p);
    for (int j = 0; j < p; ++j) {
      V_column[j] = V_top(j, i); 
    }

    Eigen::VectorXd z_i = solver.solve(V_column);
    if (solver.info() != Success) {
      Rcpp::stop("Solving for z_i failed!");
    }

    results[i] = z_i[i]; 
  }


  for (int j = 0; j < n; ++j) {
    int idx = p + j; 
    Eigen::VectorXd e_j = Eigen::VectorXd::Zero(n_plus_p);
    e_j[idx] = V_diag[j]; 

    Eigen::VectorXd z_j = solver.solve(e_j);
    if (solver.info() != Success) {
      Rcpp::stop("Solving for z_j failed!");
    }

    results[idx] = z_j[idx]; 
  }

  return wrap(results);
}



// Parallel worker to solve for diagonal elements of H^{-1} V
struct ParallelSolverp : public Worker {
  const SparseMatrix<double> &H;       
  const MatrixXd &V_top;             
  const VectorXd &V_diag;           
  NumericVector &results;            
  SparseLU<SparseMatrix<double>> &solver;
  int p;                              
  int n;                              

  // Constructor to initialize the worker
  ParallelSolverp(const SparseMatrix<double> &H_,
                 const MatrixXd &V_top_,
                 const VectorXd &V_diag_,
                 NumericVector &results_,
                 SparseLU<SparseMatrix<double>> &solver_,
                 int p_, int n_)
    : H(H_), V_top(V_top_), V_diag(V_diag_), results(results_), solver(solver_), p(p_), n(n_) {}

  // Parallel worker function
  void operator()(std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      Eigen::VectorXd V_column = Eigen::VectorXd::Zero(H.rows());

      if (i < p) {
       
        for (int j = 0; j < p; ++j) {
          V_column[j] = V_top(j, i);
        }
      } else {
        
        V_column[i] = V_diag[i - p];
      }

     
      Eigen::VectorXd z_i = solver.solve(V_column);

      if (solver.info() == Eigen::Success) {
        
        results[i] = z_i[i];
      } else {
        results[i] = NA_REAL; 
      }
    }
  }
};


extern "C" SEXP compute_Hinv_V_diagonal_parallel_p(SEXP H_, SEXP V_top_, SEXP V_diag_, SEXP p_) {
  const SparseMatrix<double>& H = as<Map<SparseMatrix<double>>>(H_);
  const MatrixXd& V_top = as<Map<MatrixXd>>(V_top_);
  const VectorXd& V_diag = as<Map<VectorXd>>(V_diag_);
  int p = as<int>(p_);
  int n = H.rows() - p;

  NumericVector results(H.rows());

  // Perform Sparse LU decomposition of H
  SparseLU<SparseMatrix<double>> solver;
  solver.compute(H);
  if (solver.info() != Success) {
    Rcpp::stop("Sparse LU decomposition failed!");
  }

 
  ParallelSolverp solver_worker(H, V_top, V_diag, results, solver, p, n);

  parallelFor(0, H.rows(), solver_worker);

  return wrap(results);
}
