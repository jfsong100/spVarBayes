#include <string>
#include <limits>
#include "util.h"
#include "nngp_fun.h"

#include <vector>
#include <algorithm>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>

using namespace std;

void zeros_minibatch(double *a, int n, int BatchSize, int *nBatchLU, int batch_index){
  int i_mb;
  for(i_mb = 0; i_mb < BatchSize; i_mb++)
    a[nBatchLU[batch_index] + i_mb] = 0.0;
}

void vecsum_minibatch(double *cum_vec, double *input_vec, int scale,int n, int BatchSize, int *nBatchLU, int batch_index){
  int i_mb;
  for(i_mb = 0; i_mb < BatchSize; i_mb++)
    cum_vec[nBatchLU[batch_index] + i_mb] += input_vec[nBatchLU[batch_index] + i_mb]/scale;

}

void get_num_nIndx_col(int *nnIndx, int nIndx, int *numIndxCol){
  for(int i = 0; i < nIndx; i++){
    numIndxCol[nnIndx[i]] += 1;
  }
}

void get_cumnum_nIndx_col(int *numIndxCol, int n, int *cumnumIndxCol){
  int q = 0;
  for(int i = 0; i < (n-1); i++){
    q += numIndxCol[i] + 1;
    cumnumIndxCol[i+1] = q;
  }
}

void get_sum_nnIndx(int *sumnnIndx, int n, int m){
  int q = 0;
  for(int i = 0; i < (n-1); i++){
    if(i < m){
      q += i;
    }
    sumnnIndx[i] = q;
  }
}

void findPositions(int *positions, int *arr, int nIndx, int x) {
  int numPositions = 1;
  positions[0] = x;
  for (int j = 0; j < nIndx; j++) {
    if (arr[j] == x) {
      positions[numPositions] = j;
      numPositions++;
    }
  }
}

void get_nnIndx_col(int *nnIndx, int n, int nIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol){
  int numPositions;
  for(int i = 0; i < n ; i++){
    numPositions = numIndxCol[i];
    int *positions = (int *) R_alloc((numPositions+1), sizeof(int)); zeros_int(positions, (numPositions+1));
    findPositions(positions, nnIndx, nIndx, i);
    for (int l = 0; l < (numPositions + 1); l++) {
      nnIndxCol[cumnumIndxCol[i]+l] = positions[l];
    }
  }
}

void get_nnIndx_nn_col(int *nnIndx, int n, int m, int nIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol, int *sumnnIndx){
  for (int i = 0; i < n; i++) {
    if(numIndxCol[i] > 0){
      for (int l = 0 ; l < numIndxCol[i]; l++) {
        //int ind = floor(nnIndxCol[cumnumIndxCol[i] + l + 1] / m);
        int ind = (floor((nnIndxCol[1 + cumnumIndxCol[i] + l] - sumnnIndx[i+l] )/m));
        nnIndxnnCol[cumnumIndxCol[i] - (i - 1) + l - 1] = ind + min(i+l+1,m);
      }
    }
  }
}

//Update B and F:

double updateBF_logdet(double *B, double *F, double *c, double *C, double *D, double *d, int *nnIndxLU, int *CIndx, int n, double *theta, int covModel, int nThreads, double fix_nugget){
  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';
  double logDet = 0;
  double nu = 0;
  //check if the model is 'matern'
  if (covModel == 2) {
    nu = theta[2];
  }

  double *bk = (double *) R_alloc(nThreads*(static_cast<int>(1.0+5.0)), sizeof(double));


  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(5.0));
  int threadID = 0;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID)
#endif
  for(i = 0; i < n; i++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    //theta[0] = alphasquareIndex, theta[1] = phiIndex, theta[2] = nuIndex (in case of 'matern')
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        c[nnIndxLU[i]+k] = spCor(d[nnIndxLU[i]+k], theta[1], nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          C[CIndx[i]+l*nnIndxLU[n+i]+k] = spCor(D[CIndx[i]+l*nnIndxLU[n+i]+k], theta[1], nu, covModel, &bk[threadID*nb]);
          if(l == k){
            C[CIndx[i]+l*nnIndxLU[n+i]+k] += theta[0]*fix_nugget;
          }
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[CIndx[i]], &nnIndxLU[n+i], &c[nnIndxLU[i]], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = 1 - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[nnIndxLU[i]], &inc) + theta[0]*fix_nugget;
    }else{
      B[i] = 0;
      F[i] = 1+ theta[0]*fix_nugget;
    }
  }
  for(i = 0; i < n; i++){
    logDet += log(F[i]);
  }

  return(logDet);
}

void solve_B_F(double *B, double *F, double *norm_residual_boot, int n, int *nnIndxLU, int *nnIndx, double *residual_boot){

  residual_boot[0] = norm_residual_boot[0] * sqrt(F[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = norm_residual_boot[i];
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      sum = sum + B[nnIndxLU[i] + l] * residual_boot[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
    }
    residual_boot[i] = sum * sqrt(F[i]);
  }
}

void product_B_F(double *B, double *F, double *residual_nngp, int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp){
  norm_residual_nngp[0] = residual_nngp[0]/sqrt(F[0]);

  double sum;
  for (int i = 1; i < n; i++) {
    sum = 0.0;
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
    }
    norm_residual_nngp[i] = sum + residual_nngp[i] / sqrt(F[i]);
  }
}

void product_B_F_vec(double *B, double *F, double *input_vec, int n, int *nnIndxLU, int *nnIndx, double *output_vec, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol){

  for(int i = 0; i < n; i++){
    input_vec[i] = input_vec[i]/sqrt(F[i]);
  }
  output_vec[n-1] = input_vec[n-1];
  double sum;
  for (int i = 0; i < (n - 1); i++) {

    sum = 0.0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        sum = sum - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]];
      }
    }
    output_vec[i] = sum + input_vec[i];
  }
}

void diagonal(double *B, double *F, double *diag_output, int n, int *nnIndx, int *nnIndxLU){
  zeros(diag_output,n);
  double a = 0;
  int i, j, k;

  double *u =(double *) R_alloc(n, sizeof(double));
  //double *v =(double *) R_alloc(n, sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for private(a, j)
#endif
  for(k = 0; k < n; k++){
    zeros(u,n);
    u[k] = 1.0;

    for(i = 0; i < n; i++){
      a = 0;
      for(j = 0; j < nnIndxLU[n+i]; j++){
        a += B[nnIndxLU[i]+j]*u[nnIndx[nnIndxLU[i]+j]];
      }
      diag_output[k] += (u[i] - a)*(u[i] - a)/F[i];
    }

  }


}

void update_uvec(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi){
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }
}

void a_gradient_fun(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                    double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    double *u_vec_temp, double *u_vec_temp2) {
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //gradient_output[nnlist_a_position[[i]]] = u[nnlist_ordered[[i]]]*gradient[i]

  for(int i = 1; i < n; i++){
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      //a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]);
      a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);

      //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
    }
  }

}

void a_gradient_fun2(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                    int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    double *u_vec_temp, double *u_vec_temp2, int zetaSqIndx,
                    double *F_inv, double *B_over_F, double *Bmat_over_F,
                    int *nnIndxLUSq, int *nnIndxwhich) {
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }
  int inc = 1;
  double one_over_zeta_sq = 1.0/theta[zetaSqIndx];

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  //product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  //product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  product_B_F_combine(u_vec, u_vec_temp, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
                      n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, nnIndxwhich);

  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);
  //gradient_output[nnlist_a_position[[i]]] = u[nnlist_ordered[[i]]]*gradient[i]

  for(int i = 1; i < n; i++){
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      //a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]);
      a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);

      //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
    }
  }

}

void a_gradient_fun_all(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int m_vi, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        double *u_vec_temp, double *u_vec_temp2,
                        double *derivative_neighbour, double *derivative_neighbour_a, double *derivative_store) {
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //gradient_output[nnlist_a_position[[i]]] = u[nnlist_ordered[[i]]]*gradient[i]


  for(int k = 0; k < n-2; k++){
    // number of neighbours for the kth column
    int num_m_vi_k = nnIndxLU_vi[n + k + 1];

    // begin from k+1 th row
    // for k+1 th row it equals to the vector of u
    // which is u_vec[nnIndx_vi[nnIndxLU_vi[k] + l]], l = 0,1,..,num_m_vi_k
    // store as k+1th row
    // derivative_store[(k+1)*num_m_vi_k+l] = u_vec[nnIndx_vi[nnIndxLU_vi[k] + l]], l = 0,1,..,num_m_vi_k

    for(int l = 0; l < nnIndxLU_vi[n + k + 1]; l++){
      derivative_store[(k+1)*num_m_vi_k+l] = u_vec[nnIndx_vi[nnIndxLU_vi[k + 1] + l]];
      a_gradient[nnIndxLU_vi[k+1] + l] += derivative_store[(k+1)*num_m_vi_k+l] * (-u_vec[k+1]/theta[tauSqIndx] - u_vec_temp2[k+1] + gradient_const[k+1]);

    }


    for(int j = k+2; j < n; j++){
      if(derivative_neighbour[j]>0){
        for(int l = 0; l < nnIndxLU_vi[n + k + 1]; l++){
          derivative_store[(j)*num_m_vi_k+l] = derivative_store[(j-1)*num_m_vi_k+l] * derivative_neighbour_a[j];
          a_gradient[nnIndxLU_vi[k+1] + l] += derivative_store[(j)*num_m_vi_k+l] * (-u_vec[j]/theta[tauSqIndx] - u_vec_temp2[j] + gradient_const[j]);
        }
      }
    }
    // and for the k+2 th row,
    // needs to create a vector du_i+1/du_i
  }

  int num_m_vi_k = nnIndxLU_vi[n + n - 1];
  for(int l = 0; l < nnIndxLU_vi[n + n - 1]; l++){
    derivative_store[(n-1)*num_m_vi_k+l] = u_vec[nnIndx_vi[nnIndxLU_vi[n-1] + l]];
    a_gradient[nnIndxLU_vi[n-1] + l] += derivative_store[(n-1)*num_m_vi_k+l] * (-u_vec[n-1]/theta[tauSqIndx] - u_vec_temp2[n-1] + gradient_const[n-1]);
  }

}


void gamma_gradient_fun(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                        double *u_vec_temp, double *u_vec_temp2, double *gradient){
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);
  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //double *gradient = (double *) R_alloc(n, sizeof(double));

  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i];
  }

  gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;

  for (int i = 0; i < (n - 1); i++) {
    sum = gradient[i];
    if(numIndxCol_vi[i] > 0){
      for (int l = 0; l < numIndxCol_vi[i]; l++) {
        sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
      }
    }
    gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
  }

}

void gamma_gradient_fun2(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                        int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                        double *u_vec_temp, double *u_vec_temp2, double *gradient, int zetaSqIndx,
                        double *F_inv, double *B_over_F, double *Bmat_over_F,
                        int *nnIndxLUSq, int *nnIndxwhich){
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);
  int inc = 1;
  double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
  //product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  //product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //double *gradient = (double *) R_alloc(n, sizeof(double));
  product_B_F_combine(u_vec, u_vec_temp, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
                      n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, nnIndxwhich);

  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);

  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i];
  }

  gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;

  for (int i = 0; i < (n - 1); i++) {
    sum = gradient[i];
    if(numIndxCol_vi[i] > 0){
      for (int l = 0; l < numIndxCol_vi[i]; l++) {
        sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
      }
    }
    gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
  }

}


void gamma_gradient_fun_all(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                            double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                            int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                            double *u_vec_temp, double *u_vec_temp2, double *gradient,
                            double *derivative_neighbour, double *derivative_neighbour_a,
                            double *derivative_store_gamma){
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++){
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
  }

  gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;

  for(int k = 0; k < n - 1; k++){
    derivative_store_gamma[k] = epsilon_vec[k] * sqrt(S_vi[k]);
    gamma_gradient[k] += gradient[k] * derivative_store_gamma[k] + 1;

    for(int j = k + 1; j < n; j++){
      if(derivative_neighbour[j]>0){
        derivative_store_gamma[j] = derivative_store_gamma[j-1] * derivative_neighbour_a[j];
        gamma_gradient[k] += gradient[j] * derivative_store_gamma[j];
      }

    }

  }

}

void ELBO_u_vec(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                double *u_vec_mean, int Trace_MC, double ELBO_MC){
  ELBO_MC = 0.0;
  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
  ELBO_MC += pow(u_vec[0],2)/theta[tauSqIndx];
  u_vec_mean[0] += u_vec[0]/Trace_MC;
  double sum;

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
    u_vec_mean[i] += u_vec[i]/Trace_MC;
    ELBO_MC += pow(u_vec[i],2)/theta[tauSqIndx];
  }

  ELBO_MC += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);

}


void product_B_F_minibatch(double *B, double *F, double *residual_nngp,
                           int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                           int BatchSize, int *nBatchLU, int batch_index){
  int i;
  double sum;
  int i_mb;


  for (i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){norm_residual_nngp[0] = residual_nngp[0]/sqrt(F[0]);}else{
      sum = 0.0;
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
      }
      norm_residual_nngp[i] = sum + residual_nngp[i] / sqrt(F[i]);
    }
  }


}

void product_B_F_vec_minibatch(double *B, double *F, double *input_vec,
                               int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                               int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                               int BatchSize, int *nBatchLU, int batch_index, int nBatch){
  int i;
  int i_mb;
  double sum;

  for (i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == (n-1)){output_vec[n-1] = input_vec[n-1]/sqrt(F[i]);}else{
      sum = 0.0;
      if(numIndxCol[i] > 0){
        for (int l = 0; l < numIndxCol[i]; l++) {
          sum = sum - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]] / sqrt(F[nnIndxnnCol[cumnumIndxCol[i] - i + l]]);
        }
      }
      output_vec[i] = sum + input_vec[i]/sqrt(F[i]);
    }
  }

}

void updateBF(double *B, double *F, double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU, int n, int m, double zetaSq, double phi, double nu, int covModel, double *bk, double nuUnifb){

  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(nuUnifb));
  int threadID = 0;
  double e;
  int mm = m*m;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID, e)
#endif
  for(i = 0; i < n; i++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c[m*threadID+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C[mm*threadID+l*nnIndxLU[n+i]+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = zetaSq - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
    }else{
      B[i] = 0;
      F[i] = zetaSq;
    }
  }

}

void updateBF2(double *B, double *F, double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU, int n, int m, double phi, double nu, int covModel, double *bk, double nuUnifb){

  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(nuUnifb));
  int threadID = 0;
  double e;
  int mm = m*m;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID, e)
#endif
  for(i = 0; i < n; i++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c[m*threadID+k] = spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C[mm*threadID+l*nnIndxLU[n+i]+k] = spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = 1 - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
    }else{
      B[i] = 0;
      F[i] = 1;
    }
  }

}

void updateBF_minibatch(double *B, double *F, double *c, double *C,
                        double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                        double zetaSq, double phi, double nu, int covModel, double *bk, double nuUnifb,
                        int BatchSize, int *nBatchLU, int batch_index){

  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(nuUnifb));
  int threadID = 0;
  double e;
  int mm = m*m;
  int i_mb;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID, e)
#endif
  for(i_mb = 0; i_mb < BatchSize; i_mb++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    i = nBatchLU[batch_index] + i_mb;
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c[m*threadID+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C[mm*threadID+l*nnIndxLU[n+i]+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = zetaSq - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
    }else{
      B[i] = 0;
      F[i] = zetaSq;
    }
  }

}


double Q_mini_batch(double *B, double *F, double *u_mb, double *v_mb,
                    int BatchSize, int *nBatchLU, int batch_index, int n,
                    int *nnIndx, int *nnIndxLU){

  double a, b, q = 0;
  int i, j;

#ifdef _OPENMP
#pragma omp parallel for private(a, b, j) reduction(+:q)
#endif

  for(int i_mb = 0; i_mb < BatchSize; i_mb++){
    i = nBatchLU[batch_index] + i_mb;
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*u_mb[nnIndx[nnIndxLU[i]+j]];
      b += B[nnIndxLU[i]+j]*v_mb[nnIndx[nnIndxLU[i]+j]];
    }
    q += (u_mb[i] - a)*(v_mb[i] - b)/F[i];
  }

  return(q);
}


void update_uvec_minibatch(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi,
                 int n, int *nnIndxLU_vi, int *nnIndx_vi,
                 int BatchSize, int *nBatchLU, int batch_index){
  // similar code as solve_B_F in BRISC package
  int i;
  double sum;

  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){
      u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
    }else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }
  }

}


void gamma_gradient_fun_minibatch(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                        double *u_vec_temp, double *u_vec_temp2, double *gradient,
                        int BatchSize, int *nBatchLU, int batch_index, int nBatch){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;


  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){ u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);}else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, BatchSize, nBatchLU, batch_index);
  product_B_F_vec_minibatch(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, nBatch);
  //double *gradient = (double *) R_alloc(n, sizeof(double));

  for(int i_mb = 0; i_mb < BatchSize; i_mb++){
    i = nBatchLU[batch_index] + i_mb;
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
  }



  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
    }

  }


}


void a_gradient_fun_minibatch(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                    double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    double *u_vec_temp, double *u_vec_temp2,
                    int BatchSize, int *nBatchLU, int batch_index, int nBatch) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i;


  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){ u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);}else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }

  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, BatchSize, nBatchLU, batch_index);
  product_B_F_vec_minibatch(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, nBatch);

  for(int i_mb = 0; i_mb < BatchSize; i_mb++){
    i = nBatchLU[batch_index] + i_mb;
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]);
      }
    }
  }


}


double var_est(double *u_vec, double *epsilon_vec, double *B, double *F, int *nnIndx, int *nnIndxLU, int n,
               double *w_mu, int BatchSize, int *nBatchLU, int batch_index, int nBatch){
  int i_mb,i,j;
  double var_estimation = 0.0;
  double a,b;
  for(i_mb = 0; i_mb < BatchSize; i_mb++){
    i = nBatchLU[batch_index] + i_mb;
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*(u_vec[nnIndx[nnIndxLU[i]+j]]+w_mu[nnIndx[nnIndxLU[i]+j]]);
    }
    var_estimation += (u_vec[i] + w_mu[i] - a)*(u_vec[i] + w_mu[i] - a)/F[i];
  }
  return var_estimation;
}


void mu_grad(double *w_mu, double *B, double *F, int n,
             int *nnIndx, int *nnIndxLU, int *nnIndxCol,
             int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
             int BatchSize, int *nBatchLU, int batch_index, int nBatch,
             double *w_mu_temp){
  int i, i_mb;
  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    sum1 = w_mu[i];
    if(i > 0){
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum1 = sum1 - B[nnIndxLU[i] + l] * w_mu[nnIndx[nnIndxLU[i] + l]];
      }
    }
    sum1 = sum1/F[i];
    Rprintf("the value of sum at %i : %f \n",i, sum1);

    sum3 = 0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        Rprintf("the value of i_l: %i \n",i_l);
        sum2 = w_mu[i_l];
        for (int k = 0; k < nnIndxLU[n + i_l]; k++) {
         sum2 = sum2 - B[nnIndxLU[i_l] + k] * w_mu[nnIndx[nnIndxLU[i_l] + k]];
        }
        Rprintf("the value of B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] : %f \n",B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] );
        Rprintf("the value of sum2: %f \n",sum2/F[i_l]);
        sum2 = sum2*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
        sum3 += sum2;
      }
    }
    Rprintf("the value of sum3 at %i : %f \n",i, sum3);
    w_mu_temp[i] = (sum1 - sum3);
  }

}

void find_set(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
              int BatchSize, int *nBatchLU, int batch_index,
              int *result_arr, int &result_index, int *temp_arr,
              int &temp_index, int *tempsize_vec, int *seen_values) {

  // Use a binary flag array for quick lookup


  temp_index = 0; // Clear the temporary array
  //Rprintf("val is ");
  // Populate temp_arr and seen_values for the current batch_index
  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int val = nBatchLU[batch_index] + i_mb;
    temp_arr[temp_index++] = val;
    seen_values[val] = 1;
    //Rprintf("%i ", val);
  }
  //Rprintf("\n");

  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int i = nBatchLU[batch_index] + i_mb;
    // Rprintf("nnIndxLU is ");
    // for (int l = 0; l < nnIndxLU[n + i]; l++) {
    //   int new_index = nnIndx[nnIndxLU[i] + l];
    //   if (!seen_values[new_index]) {
    //     temp_arr[temp_index++] = new_index;
    //     seen_values[new_index] = 1;
    //     Rprintf("%i ",new_index);
    //   }
    // }
    // Rprintf("\n");
    //Rprintf("numIndxCol is ");
    for (int k = 0; k < numIndxCol[i]; k++) {
      int new_index = nnIndxnnCol[cumnumIndxCol[i] - i + k];
      if (!seen_values[new_index]) {
        temp_arr[temp_index++] = new_index;
        seen_values[new_index] = 1;
        //Rprintf("%i ",new_index);
      }
    }
    //Rprintf("\n");
  }

  tempsize_vec[batch_index] = temp_index;
  qsort(temp_arr, temp_index, sizeof(int), [](const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
  });

  // Append temp_arr to result_arr
  for (int i = 0; i < temp_index; i++) {
    result_arr[result_index++] = temp_arr[i];
  }


}



void update_uvec_minibatch_plus(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi,
                                int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                int batch_index,
                                int *final_result_vec, int *nBatchLU_temp, int tempsize) {
  // similar code as solve_B_F in BRISC package
  int i;
  double sum;

  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){
      u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);
    }else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }
  }

}

double Q_mini_batch_plus(double *B, double *F, double *u_mb, double *v_mb,
                    int batch_index, int n,
                    int *nnIndx, int *nnIndxLU,
                    int *final_result_vec, int *nBatchLU_temp, int tempsize){

  double a, b, q = 0;
  int i, j;

#ifdef _OPENMP
#pragma omp parallel for private(a, b, j) reduction(+:q)
#endif

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*u_mb[nnIndx[nnIndxLU[i]+j]];
      b += B[nnIndxLU[i]+j]*v_mb[nnIndx[nnIndxLU[i]+j]];
    }
    q += (u_mb[i] - a)*(v_mb[i] - b)/F[i];
  }

  return(q);
}

void updateBF_minibatch_plus(double *B, double *F, double *c, double *C,
                        double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                        double zetaSq, double phi, double nu, int covModel, double *bk, double nuUnifb,
                        int batch_index,
                        int* final_result_vec, int *nBatchLU_temp, int tempsize){

  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(nuUnifb));
  int threadID = 0;
  double e;
  int mm = m*m;
  int i_mb;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID, e)
#endif
  for(i_mb = 0; i_mb < tempsize; i_mb++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c[m*threadID+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C[mm*threadID+l*nnIndxLU[n+i]+k] = zetaSq*spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = zetaSq - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
    }else{
      B[i] = 0;
      F[i] = zetaSq;
    }
  }

}


void product_B_F_minibatch_plus(double *B, double *F, double *residual_nngp,
                           int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                           int batch_index,
                           int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i;
  double sum;
  int i_mb;


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){norm_residual_nngp[0] = residual_nngp[0]/sqrt(F[0]);}else{
      sum = 0.0;
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
      }
      norm_residual_nngp[i] = sum + residual_nngp[i] / sqrt(F[i]);
    }
  }


}

void product_B_F_vec_minibatch_plus(double *B, double *F, double *input_vec,
                               int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                               int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                               int batch_index,
                               int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i;
  int i_mb;
  double sum;

  // for(i_mb = 0; i_mb < BatchSize; i_mb++){
  //   i = nBatchLU[batch_index] + i_mb;
  //   input_vec[i] = input_vec[i]/sqrt(F[i]);
  // }


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){output_vec[n-1] = input_vec[n-1]/sqrt(F[n-1]);}else{
      sum = 0.0;
      if(numIndxCol[i] > 0){
        for (int l = 0; l < numIndxCol[i]; l++) {
          sum = sum - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]] / sqrt(F[nnIndxnnCol[cumnumIndxCol[i] - i + l]]);
        }
      }
      output_vec[i] = sum + input_vec[i]/sqrt(F[i]);
    }
  }

}

void zeros_minibatch_plus(double *a, int n, int batch_index,
                          int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i_mb;
  int i;
  for(i_mb = 0; i_mb < tempsize; i_mb++)
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    a[i] = 0.0;
}

void vecsum_minibatch_plus(double *cum_vec, double *input_vec, int scale,int n,
                           int batch_index, int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i_mb;
  int i;
  for(i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    cum_vec[i] += input_vec[i]/scale;
  }


}



void gamma_gradient_fun_minibatch_plus(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                  double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                  int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                  double *u_vec_temp, double *u_vec_temp2, double *gradient,
                                  int batch_index,
                                  int *final_result_vec, int *nBatchLU_temp, int tempsize){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){ u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);}else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  //double *gradient = (double *) R_alloc(n, sizeof(double));

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
  }



  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
    }

  }


}


void a_gradient_fun_minibatch_plus(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                              double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                              double *u_vec_temp, double *u_vec_temp2,
                              int batch_index,
                              int *final_result_vec, int *nBatchLU_temp, int tempsize) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i;


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){ u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);}else{
      sum = epsilon_vec[i] * sqrt(S_vi[i]);
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
      }
      u_vec[i] = sum;
    }

  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  //product_B_F_minibatch(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, BatchSize, nBatchLU, batch_index);
  //product_B_F_vec_minibatch(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, nBatch);
  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, batch_index, final_result_vec, nBatchLU_temp, tempsize);


  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]);
      }
    }
  }


}



void find_set_mb(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
              int BatchSize, int *nBatchLU, int batch_index,
              int *result_arr, int &result_index, int *temp_arr,
              int &temp_index, int *tempsize_vec, int *seen_values) {

  // Use a binary flag array for quick lookup


  temp_index = 0; // Clear the temporary array
  //Rprintf("val is ");
  // Populate temp_arr and seen_values for the current batch_index
  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int val = nBatchLU[batch_index] + i_mb;
    temp_arr[temp_index++] = val;
    seen_values[val] = 1;
    //Rprintf("%i ", val);
  }
  //Rprintf("\n");

  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int i = nBatchLU[batch_index] + i_mb;
    //Rprintf("nnIndxLU is ");
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      int new_index = nnIndx[nnIndxLU[i] + l];
      if (!seen_values[new_index]) {
        temp_arr[temp_index++] = new_index;
        seen_values[new_index] = 1;
        //Rprintf("%i ",new_index);
      }
    }
    //Rprintf("\n");
    //Rprintf("numIndxCol is ");
    // for (int k = 0; k < numIndxCol[i]; k++) {
    //   int new_index = nnIndxnnCol[cumnumIndxCol[i] - i + k];
    //   if (!seen_values[new_index]) {
    //     temp_arr[temp_index++] = new_index;
    //     seen_values[new_index] = 1;
    //     //Rprintf("%i ",new_index);
    //   }
    // }
    //Rprintf("\n");
  }

  tempsize_vec[batch_index] = temp_index;
  qsort(temp_arr, temp_index, sizeof(int), [](const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
  });

  // Append temp_arr to result_arr
  for (int i = 0; i < temp_index; i++) {
    result_arr[result_index++] = temp_arr[i];
  }


}

double Expectation_B_F(double *B, double *F, double *w_mu, double *u_vec,
                     int n, int *nnIndxLU, int *nnIndx,
                     int BatchSize, int *nBatchLU, int batch_index){
  int i;
  double sum;
  int i_mb;
  double var;
  var = 0;
  for (i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){var += pow((w_mu[0]+u_vec[0]),2)/F[0];}else{
      sum = w_mu[i] + u_vec[i];
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum = sum - B[nnIndxLU[i] + l] * (u_vec[nnIndx[nnIndxLU[i] + l]] +  w_mu[nnIndx[nnIndxLU[i] + l]]);
      }
      var += pow(sum,2) / F[i];
    }
  }

  return(var);

}


// Comparator function for qsort
int compare_ints(const void* a, const void* b) {
  int arg1 = *(const int*)a;
  int arg2 = *(const int*)b;

  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}


void find_set_nngp(int n, int *nnIndx, int *nnIndxLU, int BatchSize, int *nBatchLU, int batch_index,
                   int *seen_values,
                   int *intersect_result, int *intersect_sizes, int *intersect_start_indices,
                   int *complement_first_result, int *complement_first_sizes, int *complement_first_start_indices,
                   int *complement_second_result, int *complement_second_sizes, int *complement_second_start_indices,
                   int &intersect_result_index, int &complement_first_result_index, int &complement_second_result_index) {

  int current_intersect_size = 0;
  int current_complement_first_size = 0;
  int current_complement_second_size = 0;
  zeros_int(seen_values, n);

  // First, mark the elements of the first set
  for (int i = 0; i < BatchSize; ++i) {
    int val = nBatchLU[batch_index] + i;
    // Do not add to intersect_result here, wait until confirmation that it's an intersection
    seen_values[val] = 1; // Mark as seen in the first set
  }

  // Second, go through the elements of the second set and determine intersection and complements
  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int i = nBatchLU[batch_index] + i_mb;
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      int new_index = nnIndx[nnIndxLU[i] + l];

      if (seen_values[new_index] == 0) { // If not seen in the first set
        // Add to complement of second set if it's not already added
        if (seen_values[new_index] != 2) {
          complement_second_result[complement_second_result_index + current_complement_second_size++] = new_index;
          seen_values[new_index] = 2; // Mark as seen only in the second set
        }
      } else if (seen_values[new_index] == 1) { // If seen in the first set
        // Add to intersection only once
        if (seen_values[new_index] != 3) {
          intersect_result[intersect_result_index + current_intersect_size++] = new_index;
          seen_values[new_index] = 3; // Mark as seen in both
        }
      }
    }
  }

  // Now determine the complement of the first set
  for (int i = 0; i < BatchSize; ++i) {
    int val = nBatchLU[batch_index] + i;

    // If only in the first set, it's part of the complement of the first set
    if (seen_values[val] == 1) {
      complement_first_result[complement_first_result_index + current_complement_first_size++] = val;
    }
  }

  // Sort each set
  qsort(intersect_result + intersect_result_index, current_intersect_size, sizeof(int), compare_ints);
  qsort(complement_first_result + complement_first_result_index, current_complement_first_size, sizeof(int), compare_ints);
  qsort(complement_second_result + complement_second_result_index, current_complement_second_size, sizeof(int), compare_ints);

  // Update indices and sizes
  intersect_start_indices[batch_index] = intersect_result_index;
  intersect_sizes[batch_index] = current_intersect_size;
  intersect_result_index += current_intersect_size;

  complement_first_start_indices[batch_index] = complement_first_result_index;
  complement_first_sizes[batch_index] = current_complement_first_size;
  complement_first_result_index += current_complement_first_size;

  complement_second_start_indices[batch_index] = complement_second_result_index; // Should point to the start of this batch's segment
  complement_second_sizes[batch_index] = current_complement_second_size;
  complement_second_result_index += current_complement_second_size; // Should accumulate the size
}


void mu_grad_intersect(double *y, double *w_mu,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                         int BatchSize, int *nBatchLU, int batch_index,
                         int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                         double *theta, int tauSqIndx,
                         double *B, double *F,
                         int *intersect_start_indices, int *intersect_sizes,
                         int* final_intersect_vec,
                         double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;

  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
   i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
   sum1 = w_mu[i];
   if(i > 0){
     for (int l = 0; l < nnIndxLU[n + i]; l++) {
       sum1 = sum1 - B[nnIndxLU[i] + l] * w_mu[nnIndx[nnIndxLU[i] + l]];
     }
   }
   sum1 = sum1/F[i];


   sum3 = 0;
   if(numIndxCol[i] > 0){
     for (int l = 0; l < numIndxCol[i]; l++) {
       // l is the lth that the i is whose neighbor
       // transfer to i_l
       i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

       sum2 = w_mu[i_l];
       for (int k = 0; k < nnIndxLU[n + i_l]; k++) {
         sum2 = sum2 - B[nnIndxLU[i_l] + k] * w_mu[nnIndx[nnIndxLU[i_l] + k]];
       }

       sum2 = sum2*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
       sum3 += sum2;
     }
   }

   gradient_mu_temp[i] = (y[i] - w_mu[i])/theta[tauSqIndx] - sum1 + sum3;
  }

}

void mu_grad_complement_1(double *y, double *w_mu,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                         int BatchSize, int *nBatchLU, int batch_index,
                         int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                         double *theta, int tauSqIndx,
                         double *B, double *F,
                         int *complement_first_start_indices, int *complement_first_sizes,
                         int* final_complement_1_vec,
                         double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;

  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    sum1 = w_mu[i];
    if(i > 0){
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum1 = sum1 - B[nnIndxLU[i] + l] * w_mu[nnIndx[nnIndxLU[i] + l]];
      }
    }
    sum1 = sum1/F[i];


    gradient_mu_temp[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - sum1;
  }

}

void mu_grad_complement_2(double *w_mu,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                         int BatchSize, int *nBatchLU, int batch_index,
                         int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                         double *theta, int tauSqIndx,
                         double *B, double *F,
                         int *complement_second_start_indices, int *complement_second_sizes,
                         int* final_complement_2_vec,
                         double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;

  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];

    sum3 = 0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

        sum2 = w_mu[i_l];
        for (int k = 0; k < nnIndxLU[n + i_l]; k++) {
          sum2 = sum2 - B[nnIndxLU[i_l] + k] * w_mu[nnIndx[nnIndxLU[i_l] + k]];
        }

        sum2 = sum2*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
        sum3 += sum2;
      }
    }

    gradient_mu_temp[i] =  sum3;
  }

}


void gamma_gradient_fun_minibatch_nngp(double *y, double *w_mu, double *u_vec,
                                       double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *gradient, double *w_mu_temp,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;

  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    w_mu_temp[i] = w_mu[i] + u_vec[i];
  }

  mu_grad_intersect(y, w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                    BatchSize, nBatchLU, batch_index,
                    numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                    B, F, intersect_start_indices, intersect_sizes,
                    final_intersect_vec,
                    gradient);

  mu_grad_complement_1(y, w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                    BatchSize, nBatchLU, batch_index,
                    numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                    B, F, complement_first_start_indices, complement_first_sizes,
                    final_complement_1_vec,
                    gradient);

  mu_grad_complement_2(w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                    BatchSize, nBatchLU, batch_index,
                    numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                    B, F, complement_second_start_indices, complement_second_sizes,
                    final_complement_2_vec,
                    gradient);

  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    //Rprintf("gamma i is : %i",i);
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          //Rprintf("index : %i \n",nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]);
          //Rprintf("number : %f \n",gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]]);
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]);
    }

  }

  for(i = 0; i < BatchSize; i++){
    gamma_gradient[nBatchLU[batch_index] + i] += 1;
  }


}

void a_gradient_fun_minibatch_nngp(double *y, double *w_mu, double *u_vec,
                                   double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *gradient, double *w_mu_temp,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    w_mu_temp[i] = w_mu[i] + u_vec[i];
  }

  mu_grad_intersect(y, w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                    BatchSize, nBatchLU, batch_index,
                    numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                    B, F, intersect_start_indices, intersect_sizes,
                    final_intersect_vec,
                    gradient);

  mu_grad_complement_1(y, w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                       BatchSize, nBatchLU, batch_index,
                       numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                       B, F, complement_first_start_indices, complement_first_sizes,
                       final_complement_1_vec,
                       gradient);

  mu_grad_complement_2(w_mu_temp, n, nnIndx, nnIndxLU, nnIndxCol,
                       BatchSize, nBatchLU, batch_index,
                       numIndxCol, nnIndxnnCol, cumnumIndxCol, theta, tauSqIndx,
                       B, F, complement_second_start_indices, complement_second_sizes,
                       final_complement_2_vec,
                       gradient);


  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * gradient[i];
        //Rprintf("gradient[i]: %f \n",gradient[i]);
      }
    }
  }


}


void product_B_F_minibatch_term1(double *B, double *F, double *residual_nngp,
                                int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                                int batch_index,
                                int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i;
  double sum;
  int i_mb;


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){norm_residual_nngp[0] = residual_nngp[0]/(F[0]);}else{
      sum = 0.0;
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]]/ (F[i]);
      }
      norm_residual_nngp[i] = sum + residual_nngp[i] / (F[i]);
    }
  }


}


void gamma_gradient_fun_minibatch_test(double *y, double *w_mu_update,
                                       double *w_vec_temp_dF, double *w_vec_temp2,
                                       double *u_vec, double *epsilon_vec, double *gamma_gradient,
                                       double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                       int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    //    gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];

    gradient[i] =  - u_vec_temp2[i] + ( - u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (- u_vec[i])/theta[tauSqIndx];
    //    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];

    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];

    //gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) ;
    }

  }

  // for (int i_mb = 0; i_mb < tempsize; i_mb++) {
  //   i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
  //   gamma_gradient[i] += 1;
  // }
  for(i = 0; i < BatchSize; i++){
    gamma_gradient[nBatchLU[batch_index] + i] += 1;
  }


}

void a_gradient_fun_minibatch_test(double *y, double *w_mu_update,
                                   double *w_vec_temp_dF, double *w_vec_temp2,
                                   double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const,
                                   double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                   int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i, i_mb;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + (- u_vec[i])/theta[tauSqIndx];

    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (- u_vec[i])/theta[tauSqIndx];
    //gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];
    //gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];

    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i]);
      }
    }
  }


}


void gamma_gradient_fun_minibatch_beta(double *y, double *X, double *beta, int p, double *w_mu_update,
                                       double *w_vec_temp_dF, double *w_vec_temp2,
                                       double *u_vec, double *epsilon_vec, double *gamma_gradient,
                                       double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                       int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;
  const int inc = 1;

  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    //    gradient[i] =  - u_vec_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
    gradient[i] =  - u_vec_temp2[i] + (- u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    //gradient[i] =  - u_vec_temp_dF[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];

    gradient[i] =  - u_vec_temp_dF[i] + (- u_vec[i])/theta[tauSqIndx];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    //    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];

    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) ;
    }

  }

  // for (int i_mb = 0; i_mb < tempsize; i_mb++) {
  //   i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
  //   gamma_gradient[i] += 1;
  // }
  for(i = 0; i < BatchSize; i++){
    gamma_gradient[nBatchLU[batch_index] + i] += 1;
  }


}

void a_gradient_fun_minibatch_beta(double *y, double *X, double *beta, int p, double *w_mu_update,
                                   double *w_vec_temp_dF, double *w_vec_temp2,
                                   double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const,
                                   double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                   int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i, i_mb;
  const int inc = 1;

  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
    gradient[i] =  - u_vec_temp2[i] + ( - u_vec[i])/theta[tauSqIndx];

  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + ( - u_vec[i])/theta[tauSqIndx];
   //    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];

    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    //gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i]);
      }
    }
  }


}



void gamma_gradient_fun_minibatch_all(double *y, double *w_mu_update,
                                       double *w_vec_temp_dF, double *w_vec_temp2,
                                       double *u_vec, double *epsilon_vec, double *gamma_gradient,
                                       double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                       int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec,
                                       double *derivative_neighbour, double *derivative_neighbour_a,
                                       double *derivative_store_gamma){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i, end_int;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    end_int = final_result_vec[nBatchLU_temp[batch_index] + tempsize - 1] + 1;
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      // sum = gradient[i];
      // if(numIndxCol_vi[i] > 0){
      //   for (int l = 0; l < numIndxCol_vi[i]; l++) {
      //     sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
      //   }
      // }
      // gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) ;
      derivative_store_gamma[i] = epsilon_vec[i] * sqrt(S_vi[i]);
      gamma_gradient[i] += gradient[i] * derivative_store_gamma[i];

      for(int j = i + 1; j < end_int; j++){
        if(derivative_neighbour[j]>0){
          derivative_store_gamma[j] = derivative_store_gamma[j-1] * derivative_neighbour_a[j];
          gamma_gradient[i] += gradient[j] * derivative_store_gamma[j];
        }
      }
    }

  }

  // for (int i_mb = 0; i_mb < tempsize; i_mb++) {
  //   i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
  //   gamma_gradient[i] += 1;
  // }
  for(i = 0; i < BatchSize; i++){
    gamma_gradient[nBatchLU[batch_index] + i] += 1;
  }


}

void a_gradient_fun_minibatch_all(double *y, double *w_mu_update,
                                   double *w_vec_temp_dF, double *w_vec_temp2,
                                   double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const,
                                   double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                   int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec,
                                   double *derivative_neighbour, double *derivative_neighbour_a, double *derivative_store) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i, i_mb, end_int;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    end_int = final_result_vec[nBatchLU_temp[batch_index] + tempsize - 1] + 1;
    if(i>0 & i < n-1){
      // for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      //   a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i]);
      // }

      int num_m_vi_k = nnIndxLU_vi[n + i];
      for(int l = 0; l < nnIndxLU_vi[n + i]; l++){
        derivative_store[(i)*num_m_vi_k+l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
        a_gradient[nnIndxLU_vi[i] + l] += derivative_store[(i)*num_m_vi_k+l] * (gradient[i]);
      }


      for(int j = i+1; j < end_int; j++){
        if(derivative_neighbour[j]>0){
          for(int l = 0; l < nnIndxLU_vi[n + i]; l++){
            derivative_store[(j)*num_m_vi_k+l] = derivative_store[(j-1)*num_m_vi_k+l] * derivative_neighbour_a[j];
            a_gradient[nnIndxLU_vi[i] + l] += derivative_store[(j)*num_m_vi_k+l] * (gradient[j]);
          }
        }
      }

    }if(i == n-1){
      int num_m_vi_k = nnIndxLU_vi[n + n - 1];
      for(int l = 0; l < nnIndxLU_vi[n + n - 1]; l++){
        derivative_store[(n-1)*num_m_vi_k+l] = u_vec[nnIndx_vi[nnIndxLU_vi[n-1] + l]];
        a_gradient[nnIndxLU_vi[n-1] + l] += derivative_store[(n-1)*num_m_vi_k+l] * (gradient[n-1]);
      }
    }
  }


}


void MFA_sigmasq_grad_term1(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx,
                            double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            double *gradient_sigmasq_temp) {

  int i;
  double sum;
  int i_mb;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;
  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    // Rprintf("i is : %i \n", i);
    sum3 = 0;
    // Rprintf("i's neighbor: \n ");
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        if(i_l < end_point & i_l > ini_point){
          // Rprintf("%i \n", i_l);
          sum3 += B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
        }
      }
    }
    // Rprintf("\n");
    gradient_sigmasq_temp[i] =  sum3;
  }

}


void MFA_sigmasq_grad(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,
                      int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                      int BatchSize, int *nBatchLU, int batch_index,
                      int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                      double *theta, int tauSqIndx,
                      double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                      int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                      int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                      int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){

  int i_mb, i;
  // Rprintf("intersect: ");
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    // Rprintf("%i ",i);
    MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - 1/F[i] - gradient_sigmasq_temp[i]) * 0.5 * sigma_sq[i];
  }
  // Rprintf("\n");
  // Rprintf("complement_first: ");
  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    // Rprintf("%i ",i);
    MFA_sigmasq_grad_vec[i] =  (1/sigma_sq[i] - 1/theta[tauSqIndx] - 1/F[i]) * 0.5 * sigma_sq[i];
    //MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - 1/F[i] - gradient_sigmasq_temp[i]) * 0.5;

  }
  // Rprintf("\n");
  // Rprintf("complement_second: ");
  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    // Rprintf("%i ",i);
    MFA_sigmasq_grad_vec[i] = - gradient_sigmasq_temp[i] * 0.5 * sigma_sq[i];
    //MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - 1/F[i] - gradient_sigmasq_temp[i]) * 0.5;
  }
  // Rprintf("\n");

}

void product_B_F_minibatch_dF(double *B, double *F, double *residual_nngp,
                              int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                              int BatchSize, int *nBatchLU, int batch_index){
  int i;
  double sum;
  int i_mb;


  for (i_mb = 0; i_mb < BatchSize; i_mb++) {
    i = nBatchLU[batch_index] + i_mb;
    if(i == 0){norm_residual_nngp[0] = residual_nngp[0]/(F[0]);}else{
      sum = 0.0;
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]]/ (F[i]);
      }
      norm_residual_nngp[i] = sum + residual_nngp[i] / (F[i]);
    }
  }


}


void mu_grad_intersect_fix(double *y, double *w_mu,
                       int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                       int BatchSize, int *nBatchLU, int batch_index,
                       int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                       double *theta, int tauSqIndx,
                       double *B, double *F,
                       int *intersect_start_indices, int *intersect_sizes,
                       int* final_intersect_vec,
                       double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;

  int i_l;
  double sum1, sum2, sum3;
  // Rprintf("w mu intersect: \n");
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    // Rprintf("i is:  %i \n", i);
    sum1 = w_mu[i];
    // Rprintf("i's sons \n");
    if(i > 0){
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        sum1 = sum1 - B[nnIndxLU[i] + l] * w_mu[nnIndx[nnIndxLU[i] + l]];
        // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
      }
    }
    sum1 = sum1/F[i];
    // Rprintf("\n");
    // Rprintf("i's neighbors \n");
    sum3 = 0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

        if(i_l < end_point & i_l > ini_point){
          // Rprintf("%i ",i_l);
          sum2 = w_mu[i_l];
          for (int k = 0; k < nnIndxLU[n + i_l]; k++) {
            sum2 = sum2 - B[nnIndxLU[i_l] + k] * w_mu[nnIndx[nnIndxLU[i_l] + k]];
            // Rprintf("B at %i * w at %i \n",nnIndxLU[i_l] + k,nnIndx[nnIndxLU[i_l] + k]);
          }

          sum2 = sum2*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
          // Rprintf("times B at %i \n", nnIndxCol[ 1 + cumnumIndxCol[i] + l] );
          sum3 += sum2;
        }

      }
    }
    // Rprintf("\n");
    gradient_mu_temp[i] = (y[i] - w_mu[i])/theta[tauSqIndx] - sum1 + sum3;
  }

}

void mu_grad_complement_1_fix(double *y, double *w_mu,
                          int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                          int BatchSize, int *nBatchLU, int batch_index,
                          int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                          double *theta, int tauSqIndx,
                          double *B, double *F,
                          int *complement_first_start_indices, int *complement_first_sizes,
                          int* final_complement_1_vec,
                          double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;
  // Rprintf("w mu complement_1: \n");
  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    // Rprintf("i is:  %i \n", i);
    sum1 = w_mu[i];
    // Rprintf("i's sons \n");
    if(i > 0){
      for (int l = 0; l < nnIndxLU[n + i]; l++) {
        // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
        sum1 = sum1 - B[nnIndxLU[i] + l] * w_mu[nnIndx[nnIndxLU[i] + l]];
      }
    }
    sum1 = sum1/F[i];


    gradient_mu_temp[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - sum1;
  }
  // Rprintf("\n");
}

void mu_grad_complement_2_fix(double *w_mu,
                          int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                          int BatchSize, int *nBatchLU, int batch_index,
                          int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                          double *theta, int tauSqIndx,
                          double *B, double *F,
                          int *complement_second_start_indices, int *complement_second_sizes,
                          int* final_complement_2_vec,
                          double *gradient_mu_temp) {

  int i;
  double sum;
  int i_mb;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;
  int i_l;
  double sum1, sum2, sum3;
  // Rprintf("w mu complement_2: \n");
  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    // Rprintf("i is:  %i \n", i);
    sum3 = 0;
    // Rprintf("i's neighbors \n");
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        if(i_l < end_point & i_l > ini_point){
          sum2 = w_mu[i_l];
          // Rprintf("%i ",i_l);
          for (int k = 0; k < nnIndxLU[n + i_l]; k++) {
            sum2 = sum2 - B[nnIndxLU[i_l] + k] * w_mu[nnIndx[nnIndxLU[i_l] + k]];
          }

          sum2 = sum2*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
          sum3 += sum2;
        }

      }
    }

    gradient_mu_temp[i] =  sum3;
  }
  // Rprintf("\n");
}


void product_B_F_vec_minibatch_plus_fix(double *B, double *F, double *input_vec,
                                    int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                                    int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                    int BatchSize, int *nBatchLU, int batch_index,
                                    int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i;
  int i_mb;
  double sum;
  int i_l;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;

  // for(i_mb = 0; i_mb < BatchSize; i_mb++){
  //   i = nBatchLU[batch_index] + i_mb;
  //   input_vec[i] = input_vec[i]/sqrt(F[i]);
  // }


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){output_vec[n-1] = input_vec[n-1]/sqrt(F[n-1]);}else{
      sum = 0.0;
      if(numIndxCol[i] > 0){
        for (int l = 0; l < numIndxCol[i]; l++) {
          i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
          if(i_l < end_point & i_l > ini_point){
            sum = sum - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]] / sqrt(F[nnIndxnnCol[cumnumIndxCol[i] - i + l]]);

          }
        }
      }
      output_vec[i] = sum + input_vec[i]/sqrt(F[i]);
    }
  }

}


void shuffleArray(int *array, int n) {
  // Shuffling the array
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }

}


void find_set_nngp_shuffle(int *shuffle_array, int n, int *nnIndx, int *nnIndxLU,
                           int BatchSize, int *nBatchLU, int batch_index,
                           int *seen_values,
                           int *intersect_result, int *intersect_sizes, int *intersect_start_indices,
                           int *complement_first_result, int *complement_first_sizes, int *complement_first_start_indices,
                           int *complement_second_result, int *complement_second_sizes, int *complement_second_start_indices,
                           int &intersect_result_index, int &complement_first_result_index, int &complement_second_result_index) {

  int current_intersect_size = 0;
  int current_complement_first_size = 0;
  int current_complement_second_size = 0;
  zeros_int(seen_values, n);

  // First, mark the elements of the first set
  for (int i = 0; i < BatchSize; ++i) {
    int val = shuffle_array[nBatchLU[batch_index] + i]; // Using shuffled value directly
    seen_values[val] = 1; // Mark as seen in the first set
  }

  // Second, go through the elements of the second set and determine intersection and complements
  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int i = shuffle_array[nBatchLU[batch_index] + i_mb];
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      int new_index = nnIndx[nnIndxLU[i] + l];

      if (seen_values[new_index] == 0) { // If not seen in the first set
        // Add to complement of second set if it's not already added
        if (seen_values[new_index] != 2) {
          complement_second_result[complement_second_result_index + current_complement_second_size++] = new_index;
          seen_values[new_index] = 2; // Mark as seen only in the second set
        }
      } else if (seen_values[new_index] == 1) { // If seen in the first set
        // Add to intersection only once
        if (seen_values[new_index] != 3) {
          intersect_result[intersect_result_index + current_intersect_size++] = new_index;
          seen_values[new_index] = 3; // Mark as seen in both
        }
      }
    }
  }

  // Now determine the complement of the first set
  for (int i = 0; i < BatchSize; ++i) {
    int val = shuffle_array[nBatchLU[batch_index] + i];

    // If only in the first set, it's part of the complement of the first set
    if (seen_values[val] == 1) {
      complement_first_result[complement_first_result_index + current_complement_first_size++] = val;
    }
  }

  // Sort each set
  qsort(intersect_result + intersect_result_index, current_intersect_size, sizeof(int), compare_ints);
  qsort(complement_first_result + complement_first_result_index, current_complement_first_size, sizeof(int), compare_ints);
  qsort(complement_second_result + complement_second_result_index, current_complement_second_size, sizeof(int), compare_ints);

  // Update indices and sizes
  intersect_start_indices[batch_index] = intersect_result_index;
  intersect_sizes[batch_index] = current_intersect_size;
  intersect_result_index += current_intersect_size;

  complement_first_start_indices[batch_index] = complement_first_result_index;
  complement_first_sizes[batch_index] = current_complement_first_size;
  complement_first_result_index += current_complement_first_size;

  complement_second_start_indices[batch_index] = complement_second_result_index; // Should point to the start of this batch's segment
  complement_second_sizes[batch_index] = current_complement_second_size;
  complement_second_result_index += current_complement_second_size; // Should accumulate the size
}


void find_set_mb_shuffle(int *shuffle_array, // Added shuffled_array parameter
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                 int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                 int BatchSize, int *nBatchLU, int batch_index,
                 int *result_arr, int &result_index, int *temp_arr,
                 int &temp_index, int *tempsize_vec, int *seen_values) {

  temp_index = 0; // Clear the temporary array

  // Populate temp_arr and seen_values for the current batch_index using shuffle_array
  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int val = shuffle_array[nBatchLU[batch_index] + i_mb]; // Use shuffled index
    temp_arr[temp_index++] = val;
    seen_values[val] = 1;
  }

  for (int i_mb = 0; i_mb < BatchSize; i_mb++) {
    int i = shuffle_array[nBatchLU[batch_index] + i_mb]; // Use shuffled index
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      int new_index = nnIndx[nnIndxLU[i] + l];
      if (!seen_values[new_index]) {
        temp_arr[temp_index++] = new_index;
        seen_values[new_index] = 1;
      }
    }
  }

  tempsize_vec[batch_index] = temp_index;
  qsort(temp_arr, temp_index, sizeof(int), [](const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
  });

  // Append temp_arr to result_arr
  for (int i = 0; i < temp_index; i++) {
    result_arr[result_index++] = temp_arr[i];
  }
}

bool isValueInArraySubset(int value, int *array, int startIndex, int endIndex) {
  for (int i = startIndex; i < endIndex; i++) {
    if (array[i] == value) {
      return true; // Value found
    }
  }
  return false; // Value not found
}


void update_inFlags(int *shuffle_array, int *inFlags, int nm,
                    int n, int *nnIndxLU, int *nnIndx,
                    int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    int BatchSize, int *nBatchLU, int batch_index,
                    int *final_result_vec, int *nBatchLU_temp, int tempsize) {
  int i, i_mb, i_l;
  int ini_point = nBatchLU[batch_index];
  int end_point = ini_point + BatchSize;

  // Iterate over each element in final_result_vec
  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if (i < n) {
      if (numIndxCol[i] > 0) {
        for (int l = 0; l < numIndxCol[i]; l++) {
          i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
          // Check if i_l is within the subset of shuffle_array
          inFlags[nm*batch_index + nnIndxCol[ 1 + cumnumIndxCol[i] + l]] = isValueInArraySubset(i_l, shuffle_array, ini_point, end_point);
        }
      }
    }
  }
}

void product_B_F_vec_minibatch_plus_shuffle(int *shuffle_array, int *inFlags, int nm,
                                            double *B, double *F, double *input_vec,
                                        int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                                        int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                        int BatchSize, int *nBatchLU, int batch_index,
                                        int *final_result_vec, int *nBatchLU_temp, int tempsize){
  int i;
  int i_mb;
  double sum;
  int i_l;
  int ini_point = nBatchLU[batch_index] ;
  int end_point = nBatchLU[batch_index] + BatchSize;

  // for(i_mb = 0; i_mb < BatchSize; i_mb++){
  //   i = nBatchLU[batch_index] + i_mb;
  //   input_vec[i] = input_vec[i]/sqrt(F[i]);
  // }


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){output_vec[n-1] = input_vec[n-1]/sqrt(F[n-1]);}else{
      sum = 0.0;
      if(numIndxCol[i] > 0){
        for (int l = 0; l < numIndxCol[i]; l++) {
          i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
          //if(isValueInArraySubset(i_l, shuffle_array, ini_point, end_point))
          //Rprintf("i_l %i at bool is: %i \n",i_l, inFlags[nm*batch_index + nnIndxCol[ 1 + cumnumIndxCol[i] + l]]);
          if(inFlags[nm*batch_index + nnIndxCol[ 1 + cumnumIndxCol[i] + l]] ){
            sum = sum - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]] / sqrt(F[nnIndxnnCol[cumnumIndxCol[i] - i + l]]);

          }
        }
      }
      output_vec[i] = sum + input_vec[i]/sqrt(F[i]);
    }
  }

}


void MFA_sigmasq_grad_term1_shuffle(int *shuffle_array, int *inFlags, int nm,
                                    int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx,
                            double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            double *gradient_sigmasq_temp) {

  int i;
  double sum;
  int i_mb;
  int ini_point = nBatchLU[batch_index] ;
  int end_point = nBatchLU[batch_index] + BatchSize;
  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    // Rprintf("i is : %i \n", i);
    sum3 = 0;
    // Rprintf("i's neighbor: \n ");
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        if(inFlags[nm*batch_index + nnIndxCol[ 1 + cumnumIndxCol[i] + l]] ){
          // Rprintf("%i \n", i_l);
          sum3 += B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
        }
      }
    }
    // Rprintf("\n");
    gradient_sigmasq_temp[i] =  sum3;
  }

}


void gamma_gradient_fun_minibatch_shuffle(int *shuffle_array, int *inFlags, int nm,
                                          double *y, double *w_mu_update,
                                       double *w_vec_temp_dF, double *w_vec_temp2,
                                       double *u_vec, double *epsilon_vec, double *gamma_gradient,
                                       double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                       int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){
  // similar code as solve_B_F in BRISC package
  double sum;
  int i_mb;
  int i;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  //product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_shuffle(shuffle_array , inFlags, nm, B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }


  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) ;
    }

  }

  // for (int i_mb = 0; i_mb < tempsize; i_mb++) {
  //   i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
  //   gamma_gradient[i] += 1;
  // }
  // for(i = 0; i < BatchSize; i++){
  //   gamma_gradient[nBatchLU[batch_index] + i] += 1;
  // }
  for(i = 0; i < BatchSize; i++){
    gamma_gradient[shuffle_array[nBatchLU[batch_index] + i]] += 1;
  }

}

void a_gradient_fun_minibatch_shuffle(int *shuffle_array, int *inFlags, int nm,
                                      double *y, double *w_mu_update,
                                   double *w_vec_temp_dF, double *w_vec_temp2,
                                   double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const,
                                   double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                                   int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i, i_mb;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  product_B_F_minibatch_plus(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  //product_B_F_vec_minibatch_plus_fix(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
  product_B_F_vec_minibatch_plus_shuffle(shuffle_array ,inFlags, nm, B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  product_B_F_minibatch_term1(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);

  //double *gradient = (double *) R_alloc(n, sizeof(double));
  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp_dF[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i] - w_vec_temp2[i] + w_vec_temp_dF[i];
    //gradient[i] =  - u_vec_temp2[i] + (y[i] - w_mu_update[i] - u_vec[i])/theta[tauSqIndx] - w_vec_temp2[i];
  }

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i]);
      }
    }
  }


}





double Q_mini_batch_shuffle(int *shuffle_array, double *B, double *F, double *u_mb, double *v_mb,
                    int BatchSize, int *nBatchLU, int batch_index, int n,
                    int *nnIndx, int *nnIndxLU){

  double a, b, q = 0;
  int i, j;

#ifdef _OPENMP
#pragma omp parallel for private(a, b, j) reduction(+:q)
#endif

  for(int i_mb = 0; i_mb < BatchSize; i_mb++){
    i = shuffle_array[nBatchLU[batch_index] + i_mb];
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*u_mb[nnIndx[nnIndxLU[i]+j]];
      b += B[nnIndxLU[i]+j]*v_mb[nnIndx[nnIndxLU[i]+j]];
    }
    q += (u_mb[i] - a)*(v_mb[i] - b)/F[i];
  }

  return(q);
}


void updateBF_quadratic(double *B_temp, double *F_temp, double *Bmat_over_F_temp,
                          double *F_inv, double *B_over_F, double *Bmat_over_F,
                          int nIndx, int nIndSqx,
                          int *nnIndxLUSq,
                          int Trace_N,
                          double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU,
                          int n, int m,
                          double nu, int covModel, double *bk, double nuUnifb,
                          double a_phi, double b_phi,
                          double phimax, double phimin) {

  int i, j, k;

  int inc = 1;
  double one = 1.0;
  double F_inv_temp;
  double phi_can;
  zeros(F_inv,n);
  zeros(B_over_F,nIndx);
  zeros(Bmat_over_F,nIndSqx);

  for(k = 0; k < Trace_N; k++){
    phi_can = rbeta(a_phi, b_phi)*(phimax - phimin) + phimin;
    //phi_can = 1.175;
    updateBF2(B_temp, F_temp, c, C, coords, nnIndx, nnIndxLU, n, m, phi_can, nu, covModel, bk, nuUnifb);

    zeros(Bmat_over_F_temp,nIndSqx);

    for(i = 0; i < n; i++){
      F_inv[i] += 1/F_temp[i];

      for(j = 0; j < nnIndxLU[n+i]; j++){
        B_over_F[nnIndxLU[i]+j] += B_temp[nnIndxLU[i]+j]/F_temp[i];
      }

      if(i > 0){
        F_inv_temp = 1/F_temp[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B_temp[nnIndxLU[i]], &inc, &B_temp[nnIndxLU[i]], &inc, &Bmat_over_F_temp[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F_temp[nnIndxLUSq[i]], &inc);
      }

    }
    add_to_vec(Bmat_over_F_temp, Bmat_over_F, nIndSqx);

  }

  double Trace_N_inv = 1.0/Trace_N;
  //Rprintf("Trace_N_inv is %f ", Trace_N_inv);
  F77_NAME(dscal)(&n, &Trace_N_inv, F_inv, &inc);
  F77_NAME(dscal)(&nIndx, &Trace_N_inv, B_over_F, &inc);
  F77_NAME(dscal)(&nIndSqx, &Trace_N_inv, Bmat_over_F, &inc);

}


double E_quadratic(double *eta_vec,
                   double *F_inv, double *B_over_F, double *Bmat_over_F,
                   int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq) {
  double q,a,b,c;
  int i, j, k;
  int inc = 1;
  double one = 1.0;
  int num_m;
  q = 0;
  for(i = 0; i < n; i++){
    a = 0;
    b = 0;
    c = 0;
    num_m = nnIndxLU[n+i];
    for(j = 0; j < num_m; j++){
      a += B_over_F[nnIndxLU[i]+j]*eta_vec[nnIndx[nnIndxLU[i]+j]];

      c += Bmat_over_F[nnIndxLUSq[i]+j*num_m+j]*pow(eta_vec[nnIndx[nnIndxLU[i]+j]],2);
      for(k = (j+1); k < num_m; k++){
        b += Bmat_over_F[nnIndxLUSq[i]+j*num_m+k]*eta_vec[nnIndx[nnIndxLU[i]+j]]*eta_vec[nnIndx[nnIndxLU[i]+k]];
      }
    }
    //Rprintf("b at i %i is %f ",i,b);
    q += pow(eta_vec[i],2)*F_inv[i] - 2*a*eta_vec[i] + 2*b + c;
  }

  return(q);
}


void product_B_F_combine(double *eta_vec, double *mid_vec, double *output_vec,
                         double *F_inv, double *B_over_F, double *Bmat_over_F,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                         int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                         int *nnIndxwhich){

  mid_vec[0] = eta_vec[0] * F_inv[0];

  double sum;
  for (int i = 1; i < n; i++) {
    sum = 0.0;
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      sum = sum - B_over_F[nnIndxLU[i] + l] * eta_vec[nnIndx[nnIndxLU[i] + l]];
    }
    mid_vec[i] = sum + eta_vec[i] * F_inv[i];
  }

  double sum21;
  double sum22;
  int l_ind;
  int num_m;
  int j_ind;
  for (int i = 0; i < (n - 1); i++) {

    sum21 = 0.0;
    sum22 = 0.0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        //sum2 = sum2 - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]];
        sum21 += B_over_F[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * eta_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]];
        l_ind = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        num_m = nnIndxLU[n+l_ind];
        j_ind = nnIndxwhich[cumnumIndxCol[i] - i + l];

        for(int k = 0; k < num_m; k++){
          sum22 += Bmat_over_F[nnIndxLUSq[l_ind]+j_ind*num_m+k]*eta_vec[nnIndx[nnIndxLU[l_ind]+k]];
        }

      }
    }
    output_vec[i] = mid_vec[i] - sum21 + sum22;
  }

  output_vec[n-1] = mid_vec[n-1];
}

void get_nnIndxwhich(int *nnIndxwhich, int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                     int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol){
  int l_ind;
  int num_m;
  int j_ind;
  for (int i = 0; i < n; i++) {
    if(numIndxCol[i] > 0){
      for (int l = 0 ; l < numIndxCol[i]; l++) {
        l_ind = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        num_m = nnIndxLU[n+l_ind];
        j_ind = 0;

        for (int s = 0; s < num_m; s++) {
          if (nnIndx[nnIndxLU[l_ind] + s] == i) {
            break;
          } else {
            j_ind++;
          }
        }

        nnIndxwhich[cumnumIndxCol[i] - (i - 1) + l - 1] = j_ind;
      }
    }
  }
}

double E_quadratic_mb(double *eta_vec,
                   double *F_inv, double *B_over_F, double *Bmat_over_F,
                   int BatchSize, int *nBatchLU, int batch_index,
                   int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq) {
  double q,a,b,c;
  int i, j, k;
  int inc = 1;
  double one = 1.0;
  int num_m;
  q = 0;
  for(int i_mb = 0; i_mb < BatchSize; i_mb++){
    i = nBatchLU[batch_index] + i_mb;
    a = 0;
    b = 0;
    c = 0;
    num_m = nnIndxLU[n+i];
    for(j = 0; j < num_m; j++){
      a += B_over_F[nnIndxLU[i]+j]*eta_vec[nnIndx[nnIndxLU[i]+j]];

      c += Bmat_over_F[nnIndxLUSq[i]+j*num_m+j]*pow(eta_vec[nnIndx[nnIndxLU[i]+j]],2);
      for(k = (j+1); k < num_m; k++){
        b += Bmat_over_F[nnIndxLUSq[i]+j*num_m+k]*eta_vec[nnIndx[nnIndxLU[i]+j]]*eta_vec[nnIndx[nnIndxLU[i]+k]];
      }
    }
    //Rprintf("b at i %i is %f ",i,b);
    q += pow(eta_vec[i],2)*F_inv[i] - 2*a*eta_vec[i] + 2*b + c;
  }

  return(q);
}


void updateBF_quadratic_mb(double *B_temp, double *F_temp, double *Bmat_over_F_temp,
                        double *F_inv, double *B_over_F, double *Bmat_over_F,
                        int nIndx, int nIndSqx,
                        int *nnIndxLUSq,
                        int Trace_N,
                        double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU,
                        int BatchSize, int *nBatchLU, int batch_index, int* final_result_vec, int *nBatchLU_temp, int tempsize,
                        int n, int m,
                        double nu, int covModel, double *bk, double nuUnifb,
                        double a_phi, double b_phi,
                        double phimax, double phimin) {

  int i, j, k, i_mb;

  int inc = 1;
  double one = 1.0;
  double F_inv_temp;
  double phi_can;
  zeros(F_inv,n);
  zeros(B_over_F,nIndx);
  zeros(Bmat_over_F,nIndSqx);

  for(k = 0; k < Trace_N; k++){
    phi_can = rbeta(a_phi, b_phi)*(phimax - phimin) + phimin;
    // phi_can = 1.175;
    //updateBF2(B_temp, F_temp, c, C, coords, nnIndx, nnIndxLU, n, m, phi_can, nu, covModel, bk, nuUnifb);
    updateBF_minibatch_plus2(B_temp, F_temp, c, C, coords,  nnIndx,  nnIndxLU,  n,  m,
                             phi_can, nu, covModel, bk, nuUnifb, batch_index, final_result_vec,  nBatchLU_temp, tempsize);

    zeros(Bmat_over_F_temp,nIndSqx);

    // for(i_mb = 0; i_mb < BatchSize; i_mb++){
    //   i = nBatchLU[batch_index] + i_mb;
    //   F_inv[i] += 1/F_temp[i];
    //
    //   for(j = 0; j < nnIndxLU[n+i]; j++){
    //     B_over_F[nnIndxLU[i]+j] += B_temp[nnIndxLU[i]+j]/F_temp[i];
    //   }
    //
    //   if(i > 0){
    //     F_inv_temp = 1/F_temp[i];
    //     F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B_temp[nnIndxLU[i]], &inc, &B_temp[nnIndxLU[i]], &inc, &Bmat_over_F_temp[nnIndxLUSq[i]], &nnIndxLU[n+i]);
    //     F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F_temp[nnIndxLUSq[i]], &inc);
    //   }
    //
    // }
#ifdef _OPENMP
#pragma omp parallel for private(i, F_inv_temp) // Parallelize the outer loop
#endif
    for (i_mb = 0; i_mb < BatchSize; i_mb++) {
      i = nBatchLU[batch_index] + i_mb;
      double F_inv_val = 1.0 / F_temp[i];
      F_inv[i] += F_inv_val;

      int nnIndxLU_i = nnIndxLU[i];
      int nnIndxLU_ni = nnIndxLU[n + i];

      for (j = 0; j < nnIndxLU_ni; j++) {
        B_over_F[nnIndxLU_i + j] += B_temp[nnIndxLU_i + j] * F_inv_val;
      }

      if (i > 0) {
        F_inv_temp = F_inv_val;
        F77_NAME(dger)(&nnIndxLU_ni, &nnIndxLU_ni, &one, &B_temp[nnIndxLU_i], &inc, &B_temp[nnIndxLU_i], &inc, &Bmat_over_F_temp[nnIndxLUSq[i]], &nnIndxLU_ni);
        F77_NAME(dscal)(&nnIndxLUSq[n + i], &F_inv_temp, &Bmat_over_F_temp[nnIndxLUSq[i]], &inc);
      }
    }

    add_to_vec(Bmat_over_F_temp, Bmat_over_F, nIndSqx);

  }

  double Trace_N_inv = 1.0/Trace_N;
  //Rprintf("Trace_N_inv is %f ", Trace_N_inv);
  F77_NAME(dscal)(&n, &Trace_N_inv, F_inv, &inc);
  F77_NAME(dscal)(&nIndx, &Trace_N_inv, B_over_F, &inc);
  F77_NAME(dscal)(&nIndSqx, &Trace_N_inv, Bmat_over_F, &inc);

}


void updateBF_minibatch_plus2(double *B, double *F, double *c, double *C,
                             double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                             double phi, double nu, int covModel, double *bk, double nuUnifb,
                             int batch_index,
                             int* final_result_vec, int *nBatchLU_temp, int tempsize){

  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  //bk must be 1+(int)floor(alpha) * nthread
  int nb = 1+static_cast<int>(floor(nuUnifb));
  int threadID = 0;
  double e;
  int mm = m*m;
  int i_mb;

#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID, e)
#endif
  for(i_mb = 0; i_mb < tempsize; i_mb++){
#ifdef _OPENMP
    threadID = omp_get_thread_num();
#endif
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c[m*threadID+k] = spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        for(l = 0; l <= k; l++){
          e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C[mm*threadID+l*nnIndxLU[n+i]+k] = spCor(e, phi, nu, covModel, &bk[threadID*nb]);
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc FCONE);
      F[i] = 1 - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
    }else{
      B[i] = 0;
      F[i] = 1;
    }
  }

}


void product_B_F_combine_mb(double *eta_vec, double *mid_vec, double *mid2_vec, double *output_vec,
                         double *F_inv, double *B_over_F, double *Bmat_over_F,
                         int BatchSize, int *nBatchLU, int batch_index,
                         int *final_result_vec, int *nBatchLU_temp, int tempsize,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                         int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                         int *nnIndxwhich){


  int i_mb, i;
  double sum;
  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == 0){mid_vec[0] = eta_vec[0] * F_inv[0];}
    sum = 0.0;
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      sum = sum - B_over_F[nnIndxLU[i] + l] * eta_vec[nnIndx[nnIndxLU[i] + l]];
    }
    mid_vec[i] = sum + eta_vec[i] * F_inv[i];
  }

  double sum21;
  double sum22;
  int l_ind;
  int num_m;
  int j_ind;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;


  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    sum21 = 0.0;
    sum22 = 0.0;
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        //sum2 = sum2 - B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * input_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]];
        l_ind = nnIndxnnCol[cumnumIndxCol[i] - i + l];
        if(l_ind < end_point & l_ind > ini_point){
          num_m = nnIndxLU[n+l_ind];
          j_ind = nnIndxwhich[cumnumIndxCol[i] - i + l];
          sum21 += B_over_F[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ] * eta_vec[nnIndxnnCol[cumnumIndxCol[i] - i + l]];
          for(int k = 0; k < num_m; k++){
            sum22 += Bmat_over_F[nnIndxLUSq[l_ind]+j_ind*num_m+k]*eta_vec[nnIndx[nnIndxLU[l_ind]+k]];
          }
        }
      }
    }
    mid2_vec[i] = sum21 - sum22;
    if(i ==(n-1)){output_vec[n-1] = mid_vec[n-1];}
    output_vec[i] = mid_vec[i] - sum21 + sum22;
  }


}


void gamma_gradient_mb_fun2(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                         int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                         int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                         double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient, int zetaSqIndx,
                         double *F_inv, double *B_over_F, double *Bmat_over_F,
                         int *nnIndxLUSq, int *nnIndxwhich,
                         int batch_index, int BatchSize, int *nBatchLU,
                         int *final_result_vec, int *nBatchLU_temp, int tempsize,
                         int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                         int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                         int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){
  // similar code as solve_B_F in BRISC package
  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  int inc = 1;
  double one_over_zeta_sq = 1.0/theta[zetaSqIndx];

  // product_B_F_combine(u_vec, u_vec_temp, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
  //                     n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, nnIndxwhich);
  //
  // F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);
  product_B_F_combine_mb(u_vec, u_vec_temp, u_vec_temp_dF, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
                         BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                         n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                         nnIndxwhich);

  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp, &inc);
  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp_dF, &inc);
  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);
  int i, i_mb;

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + ( - u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    //gradient[i] =  - u_vec_temp_dF[i] + (- u_vec[i])/theta[tauSqIndx];
    gradient[i] =  - u_vec_temp[i] + (- u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    //gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];
    gradient[i] = u_vec_temp_dF[i];
  }

  double sum = 0.0;
  for (int i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i == (n-1)){
      gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]);
    }else{
      sum = gradient[i];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
      gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) ;
    }

  }

  // for (int i_mb = 0; i_mb < tempsize; i_mb++) {
  //   i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
  //   gamma_gradient[i] += 1;
  // }
  for(i = 0; i < BatchSize; i++){
    gamma_gradient[nBatchLU[batch_index] + i] += 1;
  }


}

void a_gradient_mb_fun2(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient, int zetaSqIndx,
                                double *F_inv, double *B_over_F, double *Bmat_over_F,
                                int *nnIndxLUSq, int *nnIndxwhich,
                                int batch_index, int BatchSize, int *nBatchLU,
                                int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) {
  // similar code as solve_B_F in BRISC package
  double sum;
  int i, i_mb;


  update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             batch_index, final_result_vec, nBatchLU_temp, tempsize);

  zeros(u_vec_temp, n);
  zeros(u_vec_temp_dF, n);
  zeros(u_vec_temp2, n);

  int inc = 1;
  double one_over_zeta_sq = 1.0/theta[zetaSqIndx];

  // product_B_F_combine(u_vec, u_vec_temp, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
  //                     n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, nnIndxwhich);
  //
  // F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);
  product_B_F_combine_mb(u_vec, u_vec_temp, u_vec_temp_dF, u_vec_temp2, F_inv,  B_over_F, Bmat_over_F,
                         BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                         n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                         nnIndxwhich);

  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp, &inc);
  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp_dF, &inc);
  F77_NAME(dscal)(&n, &one_over_zeta_sq, u_vec_temp2, &inc);

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient[i] =  - u_vec_temp2[i] + ( - u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    //gradient[i] =  - u_vec_temp_dF[i] + (- u_vec[i])/theta[tauSqIndx];
    gradient[i] =  - u_vec_temp[i] + (- u_vec[i])/theta[tauSqIndx];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    //gradient[i] = - u_vec_temp2[i] + u_vec_temp_dF[i];
    gradient[i] = u_vec_temp_dF[i];
  }

  for(int i_mb = 0; i_mb < tempsize; i_mb++){
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    if(i>0){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i]);
      }
    }
  }


}


void MFA_updateBF_quadratic(double *B_temp, double *F_temp, double *Bmat_over_F_temp,
                           double *F_inv, double *B_over_F, double *Bmat_over_F, double *Bsq_over_F,
                           int nIndx, int nIndSqx,
                           int *nnIndxLUSq,
                           int Trace_N,
                           double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU,
                           int n, int m,
                           double nu, int covModel, double *bk, double nuUnifb,
                           double a_phi, double b_phi,
                           double phimax, double phimin) {

  int i, j, k, i_mb;

  int inc = 1;
  double one = 1.0;
  double F_inv_temp;
  double phi_can;
  zeros(F_inv,n);
  zeros(B_over_F,nIndx);
  zeros(Bsq_over_F,nIndx);
  zeros(Bmat_over_F,nIndSqx);
  for(k = 0; k < Trace_N; k++){
    phi_can = rbeta(a_phi, b_phi)*(phimax - phimin) + phimin;
    //phi_can = 1.175;
    updateBF2(B_temp, F_temp, c, C, coords, nnIndx, nnIndxLU, n, m, phi_can, nu, covModel, bk, nuUnifb);
    // updateBF_minibatch_plus2(B_temp, F_temp, c, C, coords,  nnIndx,  nnIndxLU,  n,  m,
    //                          phi_can, nu, covModel, bk, nuUnifb, batch_index, final_result_vec,  nBatchLU_temp, tempsize);

    zeros(Bmat_over_F_temp,nIndSqx);

    for(int i = 0; i < n; i++){
      F_inv[i] += 1/F_temp[i];

      for(j = 0; j < nnIndxLU[n+i]; j++){
        B_over_F[nnIndxLU[i]+j] += B_temp[nnIndxLU[i]+j]/F_temp[i];
        Bsq_over_F[nnIndxLU[i]+j] += B_temp[nnIndxLU[i]+j]*B_temp[nnIndxLU[i]+j]/F_temp[i];
      }

      if(i > 0){
        F_inv_temp = 1/F_temp[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B_temp[nnIndxLU[i]], &inc, &B_temp[nnIndxLU[i]], &inc, &Bmat_over_F_temp[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F_temp[nnIndxLUSq[i]], &inc);
      }

    }
    add_to_vec(Bmat_over_F_temp, Bmat_over_F, nIndSqx);

  }


  double Trace_N_inv = 1.0/Trace_N;
  //Rprintf("Trace_N_inv is %f ", Trace_N_inv);
  F77_NAME(dscal)(&n, &Trace_N_inv, F_inv, &inc);
  F77_NAME(dscal)(&nIndx, &Trace_N_inv, B_over_F, &inc);
  F77_NAME(dscal)(&nIndSqx, &Trace_N_inv, Bmat_over_F, &inc);
  F77_NAME(dscal)(&nIndx, &Trace_N_inv, Bsq_over_F, &inc);

}

void MFA_sigmasq_grad_term1_rephi(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx,
                            double *Bsq_over_F, int *nnIndxLUSq, int *nnIndxwhich,
                            int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            double *gradient_sigmasq_temp) {

  int i;
  double sum;
  int i_mb, num_m, j_ind;
  int ini_point = nBatchLU[batch_index] - 1;
  int end_point = nBatchLU[batch_index] + BatchSize;
  int i_l;
  double sum1, sum2, sum3;
  for (i_mb = 0; i_mb < tempsize; i_mb++) {
    i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
    // Rprintf("i is : %i \n", i);
    sum3 = 0;
    // Rprintf("i's neighbor: \n ");
    if(numIndxCol[i] > 0){
      for (int l = 0; l < numIndxCol[i]; l++) {
        // l is the lth that the i is whose neighbor
        // transfer to i_l
        i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

        if(i_l < end_point & i_l > ini_point){
          // Rprintf("%i \n", i_l);
          //sum3 += B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
          num_m = nnIndxLU[n+i_l];
          j_ind = nnIndxwhich[cumnumIndxCol[i] - i + l];
          sum3 += Bsq_over_F[nnIndxCol[ 1 + cumnumIndxCol[i] + l] ];
        }
      }
    }
    // Rprintf("\n");
    gradient_sigmasq_temp[i] =  sum3;
  }

}


void MFA_sigmasq_grad_rephi(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,
                      int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                      int BatchSize, int *nBatchLU, int batch_index,
                      int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                      double *theta, int tauSqIndx, int zetaSqIndx,
                      double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                      int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                      int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                      int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){

  int i_mb, i;

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i]) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] =  (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx]) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = - gradient_sigmasq_temp[i] * 0.5 * sigma_sq[i];
  }


}



void MFA_sigmasq_grad_revise1(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,double *w_a,
                            int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx, int zetaSqIndx,
                            double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                            int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                            int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){

  int i_mb, i;
  double sum_asq_over_expJ = asq_over_expJ(sigma_sq, w_a, n);

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i] - pow(w_a[i]/sigma_sq[i],2)/(1+sum_asq_over_expJ)) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] =  (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - pow(w_a[i]/sigma_sq[i],2)/(1+sum_asq_over_expJ)) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = (- gradient_sigmasq_temp[i] - pow(w_a[i]/sigma_sq[i],2)/(1+sum_asq_over_expJ)) * 0.5 * sigma_sq[i];
  }


}

double asq_over_expJ(double *sigma_sq, double *w_a, int n){
  double sum = 0.0;

  for(int i = 0; i < n; i++){
    sum += w_a[i] * w_a[i] /sigma_sq[i];
  }
  return(sum);

}

double ad_over_expJ(double *sigma_sq, double *w_a, double *w_d, int n){
  double sum = 0.0;

  for(int i = 0; i < n; i++){
    sum += w_a[i] * w_d[i] /sigma_sq[i];
  }
  return(sum);

}


void MFA_sigmasq_grad_revise2(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq, double *w_a, double *w_d,
                             int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                             int BatchSize, int *nBatchLU, int batch_index,
                             int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                             double *theta, int tauSqIndx, int zetaSqIndx,
                             double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                             int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                             int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                             int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){

  int i_mb, i;
  double sum_asq_over_expJ = asq_over_expJ(sigma_sq, w_a, n);
  double sum_dsq_over_expJ = asq_over_expJ(sigma_sq, w_d, n);
  double sum_ad_over_expJ  = ad_over_expJ(sigma_sq, w_a, w_d, n);
  double gradient_sub = 0.0;
  double deno = 1 +  sum_asq_over_expJ + sum_dsq_over_expJ + sum_asq_over_expJ*sum_dsq_over_expJ - sum_ad_over_expJ * sum_ad_over_expJ;

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    gradient_sub = - pow(w_a[i]/sigma_sq[i],2)*(1+sum_dsq_over_expJ) - pow(w_d[i]/sigma_sq[i],2)*(1+sum_asq_over_expJ) + 2*sum_ad_over_expJ*w_a[i]*w_d[i]/sigma_sq[i]/sigma_sq[i];

    MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i] + gradient_sub/deno ) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    gradient_sub = - pow(w_a[i]/sigma_sq[i],2)*(1+sum_dsq_over_expJ) - pow(w_d[i]/sigma_sq[i],2)*(1+sum_asq_over_expJ) + 2*sum_ad_over_expJ*w_a[i]*w_d[i]/sigma_sq[i]/sigma_sq[i];

    MFA_sigmasq_grad_vec[i] =  (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] + gradient_sub/deno) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    gradient_sub = - pow(w_a[i]/sigma_sq[i],2)*(1+sum_dsq_over_expJ) - pow(w_d[i]/sigma_sq[i],2)*(1+sum_asq_over_expJ) + 2*sum_ad_over_expJ*w_a[i]*w_d[i]/sigma_sq[i]/sigma_sq[i];

    MFA_sigmasq_grad_vec[i] = (- gradient_sigmasq_temp[i] + gradient_sub/deno) * 0.5 * sigma_sq[i];
  }


}


void D_vec_gradient(int n, int c, double *D_vec, double *G, double *result, double *work, double *output) {

  // Step 1: Compute D^T G^-1 D (result is a c x c matrix)
  for (int s = 0; s < c; ++s) {
    for (int t = 0; t <= s; ++t) { // Only compute lower triangle
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        double Ds = D_vec[i + s * n]; // Element D(i, s)
        double Dt = D_vec[i + t * n]; // Element D(i, t)
        sum += Ds * Dt / G[i];        // Weighted sum
      }
      result[s + t * c] = sum; // Store in column-major order
    }
  }


  for (int i = 0; i < c; ++i) {
    result[i + i * c] += 1.0; // Add 1 to the diagonal of M
  }


  char const *lower = "L";
  int info;
  F77_NAME(dpotrf)(lower, &c, result, &c, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
  F77_NAME(dpotri)(lower, &c, result, &c, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

  for (int s = 0; s < c; ++s) {
    for (int t = s + 1; t < c; ++t) { // Fill the upper triangle
      result[s + t * c] = result[t + s * c];
    }
  }

  // Step 4: Compute G^-1 D and store in work
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      work[i + j * n] = D_vec[i + j * n] / G[i]; // G^-1 D
    }
  }

  // Step 5: Compute (D^T G^-1 D)^{-1} * G^-1 D using dsymv
  const double one = 1.0;
  const double zero = 0.0;
  const int inc = 1;
  F77_NAME(dgemm)("N", "N", &n, &c, &c, &one, work, &n, result, &c, &zero, output, &n FCONE FCONE);

  // Final gradient result is stored in `work`
}


void G_vec_gradient(int n, int c, double *D_vec, double *G, double *result, double *output) {


  // Step 0: Compute D^T G^-1 D (result is a c x c matrix)
  for (int s = 0; s < c; ++s) {
    for (int t = 0; t <= s; ++t) { // Only compute lower triangle
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        double Ds = D_vec[i + s * n]; // Element D(i, s)
        double Dt = D_vec[i + t * n]; // Element D(i, t)
        sum += Ds * Dt / G[i];        // Weighted sum
      }
      result[s + t * c] = sum; // Store in column-major order
    }
  }


  for (int i = 0; i < c; ++i) {
    result[i + i * c] += 1.0; // Add 1 to the diagonal of M
  }


  char const *lower = "L";
  int info;
  F77_NAME(dpotrf)(lower, &c, result, &c, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
  F77_NAME(dpotri)(lower, &c, result, &c, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

  for (int s = 0; s < c; ++s) {
    for (int t = s + 1; t < c; ++t) { // Fill the upper triangle
      result[s + t * c] = result[t + s * c];
    }
  }

  // Step 2: Compute B = M^-1 D^T G^-1 using dgemm
  //double *B = malloc(c * n * sizeof(double)); // Temporary storage for B
  double *work = (double *) R_alloc(c, sizeof(double)); zeros(work, c);



  for (int i = 0; i < n; ++i) {
    // Step 1: Compute G^-1 D for the i-th row
    for (int j = 0; j < c; ++j) {
      work[j] = D_vec[i + j * n] / G[i]; // G^-1 D(i, j)
    }

    // Step 2: Compute (G^-1 D)_i * M^-1 using dgemv
    // double *row_result = malloc(c * sizeof(double)); // Temporary row vector
    double *row_result = (double *) R_alloc(c, sizeof(double)); zeros(row_result, c);
    const double alpha = 1.0, beta = 0.0;
    const int inc = 1;
    F77_NAME(dgemv)("N", &c, &c, &alpha, result, &c, work, &inc, &beta, row_result, &inc FCONE);

    // Step 3: Accumulate the sum of squares of the resulting row
    for (int j = 0; j < c; ++j) {
      output[i] += work[j] * row_result[j];
    }

  }


}


void MFA_sigmasq_grad_reviseq(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,double *G_vec_second,
                              int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                              int BatchSize, int *nBatchLU, int batch_index,
                              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                              double *theta, int tauSqIndx, int zetaSqIndx,
                              double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                              int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                              int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                              int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec){

  int i_mb, i;

  for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
    i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i] - G_vec_second[i]) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
    i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] =  (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - G_vec_second[i]) * 0.5 * sigma_sq[i];
  }

  for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
    i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
    MFA_sigmasq_grad_vec[i] = (- gradient_sigmasq_temp[i] - G_vec_second[i]) * 0.5 * sigma_sq[i];
  }


}


void update_vvec(double *v_vec, double *epsilon_vec, double *A_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi){
  // similar code as solve_B_F in BRISC package
  v_vec[0] = epsilon_vec[0];
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i];
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * v_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    v_vec[i] = sum;
  }
}

void sum_vec(double *A, double *X, int np, double *A_p_X){
  for (int i = 0; i < np; i++){
    A_p_X[i] = A[i] + X[i];
  }
}


void update_uvec_ubvec(double *u_vec, double *ub_vec,
                       double *epsilon_vec, double *z_vec,
                       double *A_vi, double *A_beta, double *L_beta,
                       double *S_vi, double *E_vi,
                       int n, int p,
                       int *nnIndxLU_vi, int *nnIndx_vi,
                       int *IndxLU_beta){
  const int inc = 1;
  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]);
  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      ub_vec[j] =  sum;
    }
  }



  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum + F77_NAME(ddot)(&p, &A_beta[i], &n, ub_vec, &inc);
  }


}


void gamma_l_gradient_fun(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                        double *u_vec_temp, double *u_vec_temp2, double *gradient,
                        double *ub_vec, double *z_vec, int p, double *L_beta, double *A_beta,
                        double *X, double *gradient_beta, double *XtX, double *tmp_Xtu, double *l_gradient, double *E_vi,
                        int *numIndxCol_beta, int *cumnumIndxCol_beta, int *IndxCol_beta, int *IndxLU_beta){

  const int inc = 1;
  const double one = 1.0;
  const double zero = 0.0;
  char const *ytran = "T";

  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]);
  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      ub_vec[j] =  sum;
    }
  }

  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum + F77_NAME(ddot)(&p, &A_beta[i], &n, ub_vec, &inc);
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);
  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //double *gradient = (double *) R_alloc(n, sizeof(double));

  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] - F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc)/theta[tauSqIndx];
  }

  gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;

  for (int i = 0; i < (n - 1); i++) {
    sum = gradient[i];
    if(numIndxCol_vi[i] > 0){
      for (int l = 0; l < numIndxCol_vi[i]; l++) {
        sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
      }
    }
    gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
  }

  //////////////////////////////
  F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, u_vec, &inc, &zero, tmp_Xtu, &inc FCONE);

  for(int j = 0; j < p; j++){
    gradient_beta[j] = -(tmp_Xtu[j] + F77_NAME(ddot)(&p, &XtX[j], &p, ub_vec, &inc))/theta[tauSqIndx];
  }

  if(p == 1){
    sum = gradient_beta[0];

    for(int i = 0; i < n; i++){
      sum = sum + gradient[i] * A_beta[i];

      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]] * A_beta[i];
        }
      }

    }

    l_gradient[0] = sum * z_vec[0] * sqrt(E_vi[0]) + 1;

  }else{
    // l1
    for(int j = 0; j < (p-1); j++){
      sum = gradient_beta[j];
      if(numIndxCol_beta[j] > 0){
        for(int k = 0; k < numIndxCol_beta[j]; k++){
          sum = sum + gradient_beta[IndxCol_beta[j]+k] * L_beta[numIndxCol_beta[p+cumnumIndxCol_beta[j] + k]];
        }
      }

      for(int i = 0; i < n; i++){
        sum = sum + gradient[i] * A_beta[i+j*n];

        if(numIndxCol_vi[i] > 0){
          for (int l = 0; l < numIndxCol_vi[i]; l++) {
            sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]] * A_beta[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]+j*n];
          }
        }

      }

      l_gradient[j] = sum * z_vec[j] * sqrt(E_vi[j]) + 1;
    }

    sum = gradient_beta[p-1];
    for(int i = 0; i < n; i++){
      sum = sum + gradient[i] * A_beta[i+(p-1)*n];
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]] * A_beta[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]+(p-1)*n];
        }
      }
    }



    l_gradient[p-1] = sum * z_vec[p-1] * sqrt(E_vi[p-1]) + 1;
  }

}


void a_Abeta_Lbeta_gradient_fun(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                    double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    double *u_vec_temp, double *u_vec_temp2,
                    double *ub_vec, double *z_vec, int p, double *L_beta, double *A_beta, double *X, double *E_vi, int *IndxLU_beta,
                    double *A_beta_gradient, double *L_beta_gradient, double *gradient, double *gradient_beta, double *tmp_Xtu, double *XtX,
                    int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi) {
  // similar code as solve_B_F in BRISC package
  const int inc = 1;
  const double one = 1.0;
  const double zero = 0.0;
  char const *ytran = "T";
  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]);
  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      ub_vec[j] =  sum;
    }
  }

  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum + F77_NAME(ddot)(&p, &A_beta[i], &n, ub_vec, &inc);
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //gradient_output[nnlist_a_position[[i]]] = u[nnlist_ordered[[i]]]*gradient[i]
  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] - F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc)/theta[tauSqIndx];
  }

  for(int i = 1; i < n; i++){
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      // a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);
      a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * gradient[i];

      //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
    }
  }



  for(int j = 0; j < p; j++){
    for(int i = 0; i < n; i++){
      sum = gradient[i];

      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }

      A_beta_gradient[i+j*n] = sum * ub_vec[j];
    }
  }

  F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, u_vec, &inc, &zero, tmp_Xtu, &inc FCONE);

  for(int j = 0; j < p; j++){
    gradient_beta[j] = -(tmp_Xtu[j] + F77_NAME(ddot)(&p, &XtX[j], &p, ub_vec, &inc))/theta[tauSqIndx];
  }

  if(p>1){
    int L_beta_ind = p * (p-1)/2;
    // for(int j = 1; j < p; j++){
    //   for(int k = 0; k < IndxLU_beta[p+j]; k ++){
    //     L_beta_gradient[IndxLU_beta[j] + k] = gradient_beta[j] * ub_vec[k];
    //   }
    // }
    sum = gradient_beta[1];
    for(int i = 0; i < n; i++){
      sum = sum + gradient[i] * A_beta[i+n];

      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]]* A_beta[i+n];
        }
      }

    }
    L_beta_gradient[0] = sum * ub_vec[0];

  }

}


void update_ubvec(double *ub_vec, double *u_vec, double *z_vec,
                       double *A_w, double *L_beta,
                       double *E_vi,
                       int n, int p, int num_aw,
                       int *nnIndxLU_vi, int *nnIndx_vi,
                       int *IndxLU_beta){

  const int inc = 1;
  double sum1 = 0.0;
  for(int i = 0; i < num_aw; i++){
    sum1 = sum1 + A_w[i] * u_vec[i];
  }

  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]) + sum1;



  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      sum1 = 0.0;
      for(int i = 0; i < num_aw; i++){
        sum1 = sum1 + A_w[i+j*num_aw] * u_vec[i];
      }

      ub_vec[j] =  sum + sum1;
    }
  }



}



void gamma_l_gradient_fun2(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                          double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                          int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                          double *u_vec_temp, double *u_vec_temp2, double *gradient,
                          double *ub_vec, double *z_vec, int p, double *L_beta, double *A_w,
                          double *X, double *gradient_beta, double *XtX, double *tmp_Xtu, double *l_gradient, double *E_vi,
                          int *numIndxCol_beta, int *cumnumIndxCol_beta, int *IndxCol_beta, int *IndxLU_beta, int num_aw){

  const int inc = 1;
  const double one = 1.0;
  const double zero = 0.0;
  char const *ytran = "T";

  double sum1 = 0.0;
  for(int i = 0; i < num_aw; i++){
    sum1 = sum1 + A_w[i] * u_vec[i];
  }

  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]) + sum1;



  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      sum1 = 0.0;
      for(int i = 0; i < num_aw; i++){
        sum1 = sum1 + A_w[i+j*num_aw] * u_vec[i];
      }

      ub_vec[j] =  sum + sum1;
    }
  }

  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);
  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //double *gradient = (double *) R_alloc(n, sizeof(double));

  //////////////////////////////
  F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, u_vec, &inc, &zero, tmp_Xtu, &inc FCONE);

  for(int j = 0; j < p; j++){
    gradient_beta[j] = -(tmp_Xtu[j] + F77_NAME(ddot)(&p, &XtX[j], &p, ub_vec, &inc))/theta[tauSqIndx];
  }

  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] - F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc)/theta[tauSqIndx];
  }

  gamma_gradient[n-1] = gradient[n-1] * epsilon_vec[n-1] * sqrt(S_vi[n-1]) + 1;

  for (int i = 0; i < (n - 1); i++) {

    sum = gradient[i];
    if(i < num_aw){

      for(int j = 0; j < p; j++){
        sum = sum + gradient_beta[j] * A_w[i+j*num_aw];
      }

      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }

    }else{
      if(numIndxCol_vi[i] > 0){
        for (int l = 0; l < numIndxCol_vi[i]; l++) {
          sum = sum + A_vi[ nnIndxCol_vi[ 1 + cumnumIndxCol_vi[i] + l] ] * gradient[nnIndxnnCol_vi[cumnumIndxCol_vi[i] - i + l]];
        }
      }
    }

    gamma_gradient[i] = sum * epsilon_vec[i] * sqrt(S_vi[i]) + 1;
  }



  if(p == 1){
    sum = gradient_beta[0];

    l_gradient[0] = sum * z_vec[0] * sqrt(E_vi[0]) + 1;

  }else{
    // l1
    for(int j = 0; j < (p-1); j++){
      sum = gradient_beta[j];
      if(numIndxCol_beta[j] > 0){
        for(int k = 0; k < numIndxCol_beta[j]; k++){
          sum = sum + gradient_beta[IndxCol_beta[j]+k] * L_beta[numIndxCol_beta[p+cumnumIndxCol_beta[j] + k]];
        }
      }
      l_gradient[j] = sum * z_vec[j] * sqrt(E_vi[j]) + 1;
    }

    sum = gradient_beta[p-1];
    l_gradient[p-1] = sum * z_vec[p-1] * sqrt(E_vi[p-1]) + 1;
  }

}


void a_Abeta_Lbeta_gradient_fun2(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                double *u_vec_temp, double *u_vec_temp2,
                                double *ub_vec, double *z_vec, int p, double *L_beta, double *A_w, double *X, double *E_vi, int *IndxLU_beta,
                                double *A_w_gradient, double *L_beta_gradient, double *gradient, double *gradient_beta, double *tmp_Xtu, double *XtX,
                                int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi, int num_aw) {
  // similar code as solve_B_F in BRISC package
  const int inc = 1;
  const double one = 1.0;
  const double zero = 0.0;
  char const *ytran = "T";

  double sum1 = 0.0;
  for(int i = 0; i < num_aw; i++){
    sum1 = sum1 + A_w[i] * u_vec[i];
  }

  ub_vec[0] = z_vec[0] * sqrt(E_vi[0]) + sum1;



  double sum;
  if(p > 1){
    for(int j = 1; j < p; j++){
      sum = z_vec[j] * sqrt(E_vi[j]);
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        sum = sum + L_beta[IndxLU_beta[j] + k] * ub_vec[k];
      }
      sum1 = 0.0;
      for(int i = 0; i < num_aw; i++){
        sum1 = sum1 + A_w[i+j*num_aw] * u_vec[i];
      }

      ub_vec[j] =  sum + sum1;
    }
  }

  u_vec[0] = epsilon_vec[0] * sqrt(S_vi[0]);

  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(S_vi[i]);
    for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
      sum = sum + A_vi[nnIndxLU_vi[i] + l] * u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]];
    }
    u_vec[i] = sum;
  }

  zeros(u_vec_temp, n);
  zeros(u_vec_temp2, n);

  product_B_F(B, F, u_vec, n, nnIndxLU, nnIndx, u_vec_temp);
  product_B_F_vec(B, F, u_vec_temp, n, nnIndxLU, nnIndx, u_vec_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
  //gradient_output[nnlist_a_position[[i]]] = u[nnlist_ordered[[i]]]*gradient[i]

  F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, u_vec, &inc, &zero, tmp_Xtu, &inc FCONE);

  for(int j = 0; j < p; j++){
    gradient_beta[j] = -(tmp_Xtu[j] + F77_NAME(ddot)(&p, &XtX[j], &p, ub_vec, &inc))/theta[tauSqIndx];
  }

  for(int i = 0; i < n; i++){
    //gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i];
    gradient[i] = -u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] - F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc)/theta[tauSqIndx];
  }

  if(num_aw == 1){
    for(int i = 1; i < n; i++){
      for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
        // a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);
        a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * gradient[i];

        //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
      }
    }
  }else{
    for(int i = 1; i < n; i++){

      if(i < num_aw){
        sum1 = 0.0;
        for(int j = 0; j < p; j++){
          sum1 = sum1 + gradient_beta[j] * A_w[i + j*num_aw];
        }


        for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
          // a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);
          a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (gradient[i] + sum1);

          //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
        }
      }else{
        for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
          // a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * (-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i]);
          a_gradient[nnIndxLU_vi[i] + l] = u_vec[nnIndx_vi[nnIndxLU_vi[i] + l]] * gradient[i];

          //Rprintf("gradient[i]: %f \n",(-u_vec[i]/theta[tauSqIndx] - u_vec_temp2[i] + gradient_const[i]));
        }
      }


    }
  }




  for(int j = 0; j < p; j++){
    for(int k = 0; k < num_aw; k++){
      A_w_gradient[k + j*num_aw] = gradient_beta[j] * u_vec[k];
    }

  }


  if(p>1){
    int L_beta_ind = p * (p-1)/2;
    for(int j = 1; j < p; j++){
      for(int k = 0; k < IndxLU_beta[p+j]; k ++){
        L_beta_gradient[IndxLU_beta[j] + k] = gradient_beta[j] * ub_vec[k];
      }
    }

  }


}



void updateBFq(double *B_q, double *F_q, double *c_q, double *C_q, double *HinvV_full, int *nnIndx, int *nnIndxLU, int n, int m, int p){
  
  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';
  
  int threadID = 0;
  int mm = m*m;
  

  for(i = 0; i < n; i++){

    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        // e = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
        c_q[m*threadID+k] = HinvV_full[(nnIndx[nnIndxLU[i] + k] + p)*(n+p) + i + p];
        for(l = 0; l <= k; l++){
          // e = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
          C_q[mm*threadID+l*nnIndxLU[n+i]+k] = HinvV_full[(nnIndx[nnIndxLU[i] + k] + p)*(n+p) + nnIndx[nnIndxLU[i]+l] + p];
        }
      }
      F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C_q[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotrf failed\n");}
      F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C_q[mm*threadID], &nnIndxLU[n+i], &info FCONE); if(info != 0){error("c++ error: 1 dpotri failed\n");}
      F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C_q[mm*threadID], &nnIndxLU[n+i], &c_q[m*threadID], &inc, &zero, &B_q[nnIndxLU[i]], &inc FCONE);
      F_q[i] = HinvV_full[(i + p) * (n + p) + (i + p)] - F77_NAME(ddot)(&nnIndxLU[n+i], &B_q[nnIndxLU[i]], &inc, &c_q[m*threadID], &inc);
    }else{
      B_q[i] = 0;
      F_q[i] = HinvV_full[(i + p) * (n + p) + (i + p)];
    }
  }
  
}

void update_uvec_lr(double *u_vec, double *epsilon_vec, double *B_q, double *F_q, int n, int *nnIndxLU, int *nnIndx){
  // similar code as solve_B_F in BRISC package
  u_vec[0] = epsilon_vec[0] * sqrt(F_q[0]);
  double sum;
  for (int i = 1; i < n; i++) {
    sum = epsilon_vec[i] * sqrt(F_q[i]);
    for (int l = 0; l < nnIndxLU[n + i]; l++) {
      sum = sum + B_q[nnIndxLU[i] + l] * u_vec[nnIndx[nnIndxLU[i] + l]];
    }
    u_vec[i] = sum;
  }
}
