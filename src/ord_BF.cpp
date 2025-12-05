#define USE_FC_LEN_T
#include <string>
#include <stdio.h>
#include <stdlib.h>
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
#include <Rconfig.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>


using namespace std;


#ifndef FCONE
#define FCONE
#endif






extern "C" {


  SEXP spVarBayes_build_ord_BFcpp(SEXP n_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP sType_r, 
                               SEXP nuUnif_r, SEXP sigmasq_input_r, SEXP phi_input_r,SEXP tau_input_r, SEXP nuStarting_r){


    int h, i, j, k, l, s, info, nProtect=0;
    int nThreads = 1;
    const int inc = 1;
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int m_vi = INTEGER(m_vi_r)[0];
    double *coords = REAL(coords_r);
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    

    double nuUnifa = 0, nuUnifb = 0;
    if(corName == "matern"){
      nuUnifa = REAL(nuUnif_r)[0]; nuUnifb = REAL(nuUnif_r)[1];
    }

    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(corName != "matern"){
      nTheta = 3;//zeta^2, tau^2, phi
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//zeta^2, tau^2, phi, nu
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
    }

    double *theta = (double *) R_alloc(nTheta, sizeof(double));
    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

    SEXP nnIndx_r; PROTECT(nnIndx_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndx = INTEGER(nnIndx_r);

    //int *nnIndx = (int *) R_alloc(nIndx, sizeof(int));

    double *d = (double *) R_alloc(nIndx, sizeof(double));

    SEXP nnIndxLU_r; PROTECT(nnIndxLU_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU = INTEGER(nnIndxLU_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m, coords, nnIndx, d, nnIndxLU);
    }else{
      mkNNIndxCB(n, m, coords, nnIndx, d, nnIndxLU);
    }

    int *CIndx = (int *) R_alloc(2*n, sizeof(int));
    
    // SEXP CIndx_r; PROTECT(CIndx_r = allocVector(INTSXP, 2*n)); nProtect++; int *CIndx = INTEGER(CIndx_r); //index for D and C.

    //int *CIndx = (int *) R_alloc(2*n, sizeof(int));
    for(i = 0, j = 0; i < n; i++){//zero should never be accessed
      j += nnIndxLU[n+i]*nnIndxLU[n+i];
      if(i == 0){
        CIndx[n+i] = 0;
        CIndx[i] = 0;
      }else{
        CIndx[n+i] = nnIndxLU[n+i]*nnIndxLU[n+i];
        CIndx[i] = CIndx[n+i-1] + CIndx[i-1];
      }
    }

    SEXP numIndxCol_r; PROTECT(numIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol = INTEGER(numIndxCol_r); zeros_int(numIndxCol, n);
    get_num_nIndx_col(nnIndx, nIndx, numIndxCol);

    SEXP cumnumIndxCol_r; PROTECT(cumnumIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol = INTEGER(cumnumIndxCol_r); zeros_int(cumnumIndxCol,n);
    get_cumnum_nIndx_col(numIndxCol, n, cumnumIndxCol);

    int *nnIndxCol = (int *) R_alloc(nIndx+n, sizeof(int));
    
    // SEXP nnIndxCol_r; PROTECT(nnIndxCol_r = allocVector(INTSXP, nIndx+n)); nProtect++; int *nnIndxCol = INTEGER(nnIndxCol_r); 
    zeros_int(nnIndxCol, nIndx+n);
    get_nnIndx_col(nnIndx, n, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol);

    int *sumnnIndx = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx, n-1);
    get_sum_nnIndx(sumnnIndx, n, m);

    SEXP nnIndxnnCol_r; PROTECT(nnIndxnnCol_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndxnnCol = INTEGER(nnIndxnnCol_r); zeros_int(nnIndxnnCol, n);
    get_nnIndx_nn_col(nnIndx, n, m, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, sumnnIndx);


    double *D = (double *) R_alloc(j, sizeof(double));

    for(i = 0; i < n; i++){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        for(l = 0; l <= k; l++){
          D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
        }
      }
    }
    int mm = m*m;
    SEXP B_r; PROTECT(B_r = allocVector(REALSXP, nIndx)); nProtect++; double *B = REAL(B_r);
    SEXP F_r; PROTECT(F_r = allocVector(REALSXP, n)); nProtect++; double *F = REAL(F_r);

    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));

    double *c =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C = (double *) R_alloc(mm*nThreads, sizeof(double));

    theta[zetaSqIndx] = REAL(sigmasq_input_r)[0];
    theta[tauSqIndx] = REAL(tau_input_r)[0];
    //theta[phiIndx] = REAL(phiStarting_r)[0];
    theta[phiIndx] = REAL(phi_input_r)[0];
    if(corName == "matern"){
      theta[nuIndx] = REAL(nuStarting_r)[0];
    }
    ////////////
    double a, v, b, e, mu, var, aij, phiCand, nuCand = 0, nu = 0;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

    
    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx_vi = static_cast<int>(static_cast<double>(1+m_vi)/2*m_vi+(n-m_vi-1)*m_vi);
    
    SEXP nnIndx_vi_r; PROTECT(nnIndx_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndx_vi = INTEGER(nnIndx_vi_r);
    
    double *d_vi = (double *) R_alloc(nIndx_vi, sizeof(double));
    
    SEXP nnIndxLU_vi_r; PROTECT(nnIndxLU_vi_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).
   
    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }else{
      mkNNIndxCB(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }
    
    
    int mm_vi = m_vi*m_vi;

    SEXP numIndxCol_vi_r; PROTECT(numIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol_vi = INTEGER(numIndxCol_vi_r); zeros_int(numIndxCol_vi, n);
    get_num_nIndx_col(nnIndx_vi, nIndx_vi, numIndxCol_vi);
    
    SEXP cumnumIndxCol_vi_r; PROTECT(cumnumIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol_vi = INTEGER(cumnumIndxCol_vi_r); zeros_int(cumnumIndxCol_vi,n);
    get_cumnum_nIndx_col(numIndxCol_vi, n, cumnumIndxCol_vi);
    
    SEXP nnIndxCol_vi_r; PROTECT(nnIndxCol_vi_r = allocVector(INTSXP, nIndx_vi+n)); nProtect++; int *nnIndxCol_vi = INTEGER(nnIndxCol_vi_r); zeros_int(nnIndxCol_vi, n);
    get_nnIndx_col(nnIndx_vi, n, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi);
    
    int *sumnnIndx_vi = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx_vi, n-1);
    get_sum_nnIndx(sumnnIndx_vi, n, m_vi);
    
    SEXP nnIndxnnCol_vi_r; PROTECT(nnIndxnnCol_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndxnnCol_vi = INTEGER(nnIndxnnCol_vi_r); zeros_int(nnIndxnnCol_vi, n);
    get_nnIndx_nn_col(nnIndx_vi, n, m_vi, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi, sumnnIndx_vi);
    
    SEXP result_r, resultName_r;
    int nResultListObjs = 12;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, nnIndxLU_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("nnIndxLU"));

    SET_VECTOR_ELT(result_r, 1, nnIndx_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("nnIndx"));

    SET_VECTOR_ELT(result_r, 2, numIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("numIndxCol"));

    SET_VECTOR_ELT(result_r, 3, cumnumIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("cumnumIndxCol"));

    SET_VECTOR_ELT(result_r, 4, nnIndxnnCol_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("nnIndxnnCol"));

    SET_VECTOR_ELT(result_r, 5, B_r);
    SET_VECTOR_ELT(resultName_r, 5, mkChar("B"));

    SET_VECTOR_ELT(result_r, 6, F_r);
    SET_VECTOR_ELT(resultName_r, 6, mkChar("F"));
    
    SET_VECTOR_ELT(result_r, 7, nnIndxLU_vi_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("nnIndxLU_vi"));
    
    SET_VECTOR_ELT(result_r, 8, nnIndx_vi_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("nnIndx_vi"));
    
    SET_VECTOR_ELT(result_r, 9, numIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("numIndxCol_vi"));
    
    SET_VECTOR_ELT(result_r, 10, cumnumIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("cumnumIndxCol_vi"));
    
    SET_VECTOR_ELT(result_r, 11, nnIndxnnCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("nnIndxnnCol_vi"));
    

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }



}
