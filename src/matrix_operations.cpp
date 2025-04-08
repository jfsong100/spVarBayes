#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

// Function to construct the H matrix
extern "C" SEXP construct_H(SEXP n_, SEXP X_, SEXP tau2_true_, SEXP nnIndxLU_, SEXP nnIndx_,
                           SEXP numIndxCol_, SEXP nnIndxnnCol_, SEXP cumnumIndxCol_, SEXP B_, SEXP F_) {

  int n = as<int>(n_);
  NumericVector X(X_);
  double tau2_true = as<double>(tau2_true_);
  IntegerVector nnIndxLU(nnIndxLU_);
  IntegerVector nnIndx(nnIndx_);
  IntegerVector numIndxCol(numIndxCol_);
  IntegerVector nnIndxnnCol(nnIndxnnCol_);
  IntegerVector cumnumIndxCol(cumnumIndxCol_);
  NumericVector B(B_);
  NumericVector F(F_);

  Eigen::SparseMatrix<double> H(n + 1, n + 1);
  std::vector<Triplet<double>> tripletList;

  // Fill the H matrix
  for (int i = 0; i < n; ++i) {
    int index = i+1;
    tripletList.push_back(Triplet<double>(index, 0, -1.0 / tau2_true * X[i]));
    tripletList.push_back(Triplet<double>(0, index, -1.0 / tau2_true * X[i]));

    if (nnIndxLU[n + i] > 0) {
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        int row = i + 1;
        int col = nnIndx[nnIndxLU[i] + l] + 1;
        double value = -B[nnIndxLU[i] + l] / F[i];
        tripletList.push_back(Triplet<double>(row, col, value));
        tripletList.push_back(Triplet<double>(col, row, value));
      }
    }

    if(numIndxCol[i] > 0){
      for (int j = 0; j < numIndxCol[i]; j++) {
        int k = nnIndxnnCol[cumnumIndxCol[i] - i + j];
        int find_index;
        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if (neighbor_index == i) {
            find_index = c;
          }
        }

        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if(neighbor_index != i){
            int row = i + 1;
            int col = nnIndx[nnIndxLU[k] + c] + 1;
            double value = B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2;

            tripletList.push_back(Triplet<double>(row, col, value));
            tripletList.push_back(Triplet<double>(col, row, value));
          }

        }

      }
    }
  }

  H.setFromTriplets(tripletList.begin(), tripletList.end());
  return wrap(H);
}

// Function to construct I_VH matrix
extern "C" SEXP construct_I_VH(SEXP n_, SEXP X_, SEXP tau2_true_, SEXP nnIndxLU_, SEXP nnIndx_,
                              SEXP numIndxCol_, SEXP nnIndxnnCol_, SEXP cumnumIndxCol_,
                              SEXP B_, SEXP F_, SEXP V_diag_) {

  int n = as<int>(n_);
  NumericVector X(X_);
  double tau2_true = as<double>(tau2_true_);
  IntegerVector nnIndxLU(nnIndxLU_);
  IntegerVector nnIndx(nnIndx_);
  IntegerVector numIndxCol(numIndxCol_);
  IntegerVector nnIndxnnCol(nnIndxnnCol_);
  IntegerVector cumnumIndxCol(cumnumIndxCol_);
  NumericVector B(B_);
  NumericVector F(F_);
  NumericVector V_diag(V_diag_);

  Eigen::SparseMatrix<double> H(n + 1, n + 1);
  std::vector<Triplet<double>> tripletList;

  for (int i = 0; i < n + 1; ++i) {
    tripletList.push_back(Triplet<double>(i, i, 1.0));  // Identity diagonal entries
  }

  for (int i = 0; i < n; ++i) {
    int index = i + 1;
    tripletList.push_back(Triplet<double>(index, 0, 1.0 / tau2_true * X[i] * V_diag[index]));
    tripletList.push_back(Triplet<double>(0, index, 1.0 / tau2_true * X[i] * V_diag[0]));

    if (nnIndxLU[n + i] > 0) {
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        int row = i + 1;
        int col = nnIndx[nnIndxLU[i] + l] + 1;
        double value1 = B[nnIndxLU[i] + l] / F[i] * V_diag[row];
        double value2 = B[nnIndxLU[i] + l] / F[i] * V_diag[col];
        tripletList.push_back(Triplet<double>(row, col, value1));
        tripletList.push_back(Triplet<double>(col, row, value2));
      }
    }

    if(numIndxCol[i] > 0){
      for (int j = 0; j < numIndxCol[i]; j++) {
        int k = nnIndxnnCol[cumnumIndxCol[i] - i + j];
        int find_index;
        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if (neighbor_index == i) {
            find_index = c;
          }
        }

        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if(neighbor_index != i){
            int row = i + 1;
            int col = nnIndx[nnIndxLU[k] + c] + 1;
            double value1 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[row];
            double value2 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[col];

            tripletList.push_back(Triplet<double>(row, col, value1));
            tripletList.push_back(Triplet<double>(col, row, value2));
          }
        }
      }
    }
  }

  H.setFromTriplets(tripletList.begin(), tripletList.end());
  return wrap(H);
}

// Function to construct V_VHV matrix
extern "C" SEXP construct_V_VHV(SEXP n_, SEXP X_, SEXP tau2_true_, SEXP nnIndxLU_, SEXP nnIndx_,
                               SEXP numIndxCol_, SEXP nnIndxnnCol_, SEXP cumnumIndxCol_,
                               SEXP B_, SEXP F_, SEXP V_diag_) {

  int n = as<int>(n_);
  NumericVector X(X_);
  double tau2_true = as<double>(tau2_true_);
  IntegerVector nnIndxLU(nnIndxLU_);
  IntegerVector nnIndx(nnIndx_);
  IntegerVector numIndxCol(numIndxCol_);
  IntegerVector nnIndxnnCol(nnIndxnnCol_);
  IntegerVector cumnumIndxCol(cumnumIndxCol_);
  NumericVector B(B_);
  NumericVector F(F_);
  NumericVector V_diag(V_diag_);

  Eigen::SparseMatrix<double> H(n + 1, n + 1);
  std::vector<Triplet<double>> tripletList;

  for (int i = 0; i < n + 1; ++i) {
    tripletList.push_back(Triplet<double>(i, i, V_diag[i])); 
  }

  for (int i = 0; i < n; ++i) {
    int index = i + 1;
    tripletList.push_back(Triplet<double>(index, 0, 1.0 / tau2_true * X[i] * V_diag[index]* V_diag[0]));
    tripletList.push_back(Triplet<double>(0, index, 1.0 / tau2_true * X[i] * V_diag[0]* V_diag[index]));

    if (nnIndxLU[n + i] > 0) {
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        int row = i + 1;
        int col = nnIndx[nnIndxLU[i] + l] + 1;
        double value1 = B[nnIndxLU[i] + l] / F[i] * V_diag[row]* V_diag[col];
        tripletList.push_back(Triplet<double>(row, col, value1));
        tripletList.push_back(Triplet<double>(col, row, value1));
      }
    }

    if(numIndxCol[i] > 0){
      for (int j = 0; j < numIndxCol[i]; j++) {
        int k = nnIndxnnCol[cumnumIndxCol[i] - i + j];
        int find_index;
        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if (neighbor_index == i) {
            find_index = c;
          }
        }

        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if(neighbor_index != i){
            int row = i + 1;
            int col = nnIndx[nnIndxLU[k] + c] + 1;
            double value1 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[row]* V_diag[col];
            tripletList.push_back(Triplet<double>(row, col, value1));
            tripletList.push_back(Triplet<double>(col, row, value1));
          }
        }
      }
    }
  }

  H.setFromTriplets(tripletList.begin(), tripletList.end());
  return wrap(H);
}


extern "C" SEXP construct_I_VH_p(SEXP n_, SEXP p_, SEXP X_, SEXP tau2_true_, SEXP nnIndxLU_, SEXP nnIndx_,
                              SEXP numIndxCol_, SEXP nnIndxnnCol_, SEXP cumnumIndxCol_,
                              SEXP B_, SEXP F_, SEXP V_diag_, SEXP beta_premat_pp_, SEXP beta_premat_pn_,
                              SEXP beta_premat_np_) {

  int n = as<int>(n_);
  int p = as<int>(p_);
  NumericMatrix X(X_);
  NumericMatrix beta_premat_pp(beta_premat_pp_);
  NumericMatrix beta_premat_pn(beta_premat_pn_);
  double tau2_true = as<double>(tau2_true_);
  IntegerVector nnIndxLU(nnIndxLU_);
  IntegerVector nnIndx(nnIndx_);
  IntegerVector numIndxCol(numIndxCol_);
  IntegerVector nnIndxnnCol(nnIndxnnCol_);
  IntegerVector cumnumIndxCol(cumnumIndxCol_);
  NumericVector B(B_);
  NumericVector F(F_);
  NumericVector V_diag(V_diag_);

  Eigen::SparseMatrix<double> H(n + p, n + p);
  std::vector<Triplet<double>> tripletList;

  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < p; ++j) {
      if (k == j) {
       
        tripletList.push_back(Triplet<double>(k, j, 1.0 - beta_premat_pp(k, j)));
      } else {
        
        tripletList.push_back(Triplet<double>(k, j, -beta_premat_pp(k, j)));
      }
    }
  }

  
  for (int i = p; i < n + p; ++i) {
    tripletList.push_back(Triplet<double>(i, i, 1.0));
  }

  for (int i = 0; i < n; ++i) {
    int index = i + p;
    
    for(int j = 0; j < p; ++j){
      tripletList.push_back(Triplet<double>(index, j, 1.0 / tau2_true * (X(i,j) * V_diag[index]) ));
      tripletList.push_back(Triplet<double>(j, index, 1.0 / tau2_true * beta_premat_pn(j,i)));
    }


    if (nnIndxLU[n + i] > 0) {
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        int row = i + p;
        int col = nnIndx[nnIndxLU[i] + l] + p;
        double value1 = B[nnIndxLU[i] + l] / F[i] * V_diag[row];
        double value2 = B[nnIndxLU[i] + l] / F[i] * V_diag[col];
        tripletList.push_back(Triplet<double>(row, col, value1));
        tripletList.push_back(Triplet<double>(col, row, value2));
      }
    }

    if(numIndxCol[i] > 0){
      for (int j = 0; j < numIndxCol[i]; j++) {
        int k = nnIndxnnCol[cumnumIndxCol[i] - i + j];
        int find_index;
        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if (neighbor_index == i) {
            find_index = c;
          }
        }

        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if(neighbor_index != i){
            int row = i + p;
            int col = nnIndx[nnIndxLU[k] + c] + p;
            double value1 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[row];
            double value2 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[col];

            tripletList.push_back(Triplet<double>(row, col, value1));
            tripletList.push_back(Triplet<double>(col, row, value2));
          }
        }
      }
    }
  }

  H.setFromTriplets(tripletList.begin(), tripletList.end());
  return wrap(H);
}


extern "C" SEXP construct_I_VH_nop(SEXP n_, SEXP tau2_true_, SEXP nnIndxLU_, SEXP nnIndx_,
                                SEXP numIndxCol_, SEXP nnIndxnnCol_, SEXP cumnumIndxCol_,
                                SEXP B_, SEXP F_, SEXP V_diag_) {

  int n = as<int>(n_);
  double tau2_true = as<double>(tau2_true_);
  IntegerVector nnIndxLU(nnIndxLU_);
  IntegerVector nnIndx(nnIndx_);
  IntegerVector numIndxCol(numIndxCol_);
  IntegerVector nnIndxnnCol(nnIndxnnCol_);
  IntegerVector cumnumIndxCol(cumnumIndxCol_);
  NumericVector B(B_);
  NumericVector F(F_);
  NumericVector V_diag(V_diag_);

  Eigen::SparseMatrix<double> H(n, n);
  std::vector<Triplet<double>> tripletList;

  
  for (int i = 0; i < n; ++i) {
    tripletList.push_back(Triplet<double>(i, i, 1.0));
  }

  for (int i = 0; i < n; ++i) {
    int index = i;

    if (nnIndxLU[n + i] > 0) {
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        int row = i;
        int col = nnIndx[nnIndxLU[i] + l];
        double value1 = B[nnIndxLU[i] + l] / F[i] * V_diag[row];
        double value2 = B[nnIndxLU[i] + l] / F[i] * V_diag[col];
        tripletList.push_back(Triplet<double>(row, col, value1));
        tripletList.push_back(Triplet<double>(col, row, value2));
      }
    }

    if(numIndxCol[i] > 0){
      for (int j = 0; j < numIndxCol[i]; j++) {
        int k = nnIndxnnCol[cumnumIndxCol[i] - i + j];
        int find_index;
        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if (neighbor_index == i) {
            find_index = c;
          }
        }

        for (int c = 0; c < nnIndxLU[n + k]; c++) {
          int neighbor_index = nnIndx[nnIndxLU[k] + c];
          if(neighbor_index != i){
            int row = i;
            int col = nnIndx[nnIndxLU[k] + c];
            double value1 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[row];
            double value2 = -B[nnIndxLU[k] + c]*B[nnIndxLU[k] + find_index]/F[k]/2 * V_diag[col];

            tripletList.push_back(Triplet<double>(row, col, value1));
            tripletList.push_back(Triplet<double>(col, row, value2));
          }
        }
      }
    }
  }

  H.setFromTriplets(tripletList.begin(), tripletList.end());
  return wrap(H);
}
