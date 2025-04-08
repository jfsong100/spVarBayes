#define USE_FC_LEN_T
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "util.h"
#include "nngp_fun.h"
#include <vector>


#ifdef _OPENMP
#include <omp.h>
#endif

// #include <cppad/cppad.hpp>
// #include <cppad/example/cppad_eigen.hpp>
// #include <cassert>
// #include <Eigen/Dense>
// #include <Eigen/Core>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rconfig.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>


using namespace std;

//
// using CppAD::AD;
// using CppAD::ADFun;
// using Eigen::Matrix;
// using Eigen::Dynamic;
//
// typedef Matrix< AD<double> , Dynamic, Dynamic > a_matrix;
// typedef Matrix< AD<double> , Dynamic , 1>       a_vector;
// typedef Matrix< double     , Dynamic, Dynamic > matrix;
// typedef Matrix< double ,     Dynamic , 1>       vector;
// typedef CppAD::AD<double> ADdouble;


#ifndef FCONE
#define FCONE
#endif


extern "C" {

  SEXP NobetaPredict(SEXP y_r, SEXP coords_r, SEXP n_r, SEXP m_r,
                    SEXP coords0_r,
                    SEXP q_r, SEXP nnIndx0_r,
                    SEXP thetaSamples_r,
                    SEXP wSamples_r,
                    SEXP nSamples_r,
                    SEXP family_r,
                    SEXP covModel_r,
                    SEXP nThreads_r, SEXP verbose_r,
                    SEXP nReport_r){

    int h, i, j, k, l, s, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    char const *lower = "L";

    //get args
    double *y = REAL(y_r);
    double *coords = REAL(coords_r);
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int mm = m*m;

    double *coords0 = REAL(coords0_r);
    int q = INTEGER(q_r)[0];

    int *nnIndx0 = INTEGER(nnIndx0_r);
    double *theta = REAL(thetaSamples_r);
    double *w = REAL(wSamples_r);

    int nSamples = INTEGER(nSamples_r)[0];
    int family = INTEGER(family_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    int nReport = INTEGER(nReport_r)[0];

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\t     Prediction description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("Model fit with %i observations.\n\n", n);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Predicting at %i locations.\n\n", q);
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i threads.\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(family == 1){
      if(corName != "matern"){
        nTheta = 3;//zeta^2, tau^2, phi
        zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
      }else{
        nTheta = 4;//zeta^2, tau^2, phi, nu
        zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
      }
    }else{//family is binomial
      if(corName != "matern"){
        nTheta = 2;//zeta^2, phi
        zetaSqIndx = 0; phiIndx = 1;
      }else{
        nTheta = 3;//zeta^2, phi, nu
        zetaSqIndx = 0; phiIndx = 1; nuIndx = 2;
      }
    }

    //get max nu
    double nuMax = 0;
    int nb = 0;

    if(corName == "matern"){
      for(i = 0; i < nSamples; i++){
        if(theta[i*nTheta+nuIndx] > nuMax){
          nuMax = theta[i*nTheta+nuIndx];
        }
      }

      nb = 1+static_cast<int>(floor(nuMax));
    }

    double *bk = (double *) R_alloc(nThreads*nb, sizeof(double));

    double *C = (double *) R_alloc(nThreads*mm, sizeof(double)); zeros(C, nThreads*mm);
    double *c = (double *) R_alloc(nThreads*m, sizeof(double)); zeros(c, nThreads*m);
    double *tmp_m  = (double *) R_alloc(nThreads*m, sizeof(double));
    double phi = 0, nu = 0, zetaSq = 0, tauSq = 0, d;
    int threadID = 0, status = 0;

    SEXP y0_r, w0_r;
    PROTECT(y0_r = allocMatrix(REALSXP, q, nSamples)); nProtect++;
    PROTECT(w0_r = allocMatrix(REALSXP, q, nSamples)); nProtect++;
    double *y0 = REAL(y0_r);
    double *w0 = REAL(w0_r);

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\t\t         Predicting\n");
      Rprintf("----------------------------------------\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int zIndx = -1;
    double *wZ = (double *) R_alloc(q*nSamples, sizeof(double));

    double *yZ = NULL;
    if(family == 1){
      yZ = (double *) R_alloc(q*nSamples, sizeof(double));
    }

    GetRNGstate();

    for(i = 0; i < q*nSamples; i++){
      wZ[i] = rnorm(0.0,1.0);
    }

    if(family == 1){
      for(i = 0; i < q*nSamples; i++){
        yZ[i] = rnorm(0.0,1.0);
      }
    }

    PutRNGstate();

    for(i = 0; i < q; i++){
#ifdef _OPENMP
#pragma omp parallel for private(threadID, phi, nu, zetaSq, tauSq, k, l, d, info)
#endif
      for(s = 0; s < nSamples; s++){
#ifdef _OPENMP
        threadID = omp_get_thread_num();
#endif
        phi = theta[s*nTheta+phiIndx];
        if(corName == "matern"){
          nu = theta[s*nTheta+nuIndx];
        }
        zetaSq = theta[s*nTheta+zetaSqIndx];

        if(family == 1){
          tauSq = theta[s*nTheta+tauSqIndx];
        }

        for(k = 0; k < m; k++){
          d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords0[i], coords0[q+i]);
          c[threadID*m+k] = zetaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
          for(l = 0; l < m; l++){
            d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords[nnIndx0[i+q*l]], coords[n+nnIndx0[i+q*l]]);
            C[threadID*mm+l*m+k] = zetaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
          }
        }

        F77_NAME(dpotrf)(lower, &m, &C[threadID*mm], &m, &info FCONE); if(info != 0){error("c++ error: dpotrf failed\n");}
        F77_NAME(dpotri)(lower, &m, &C[threadID*mm], &m, &info FCONE); if(info != 0){error("c++ error: dpotri failed\n");}

        F77_NAME(dsymv)(lower, &m, &one, &C[threadID*mm], &m, &c[threadID*m], &inc, &zero, &tmp_m[threadID*m], &inc FCONE);

        d = 0;
        for(k = 0; k < m; k++){
          d += tmp_m[threadID*m+k]*w[s*n+nnIndx0[i+q*k]];
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        zIndx++;

        w0[s*q+i] = sqrt(zetaSq - F77_NAME(ddot)(&m, &tmp_m[threadID*m], &inc, &c[threadID*m], &inc))*wZ[zIndx] + d;

        if(family == 1){
          y0[s*q+i] = sqrt(tauSq)*yZ[zIndx] +  w0[s*q+i]; //F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) +
        }else{//binomial
          y0[s*q+i] = w0[s*q+i]; //F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) +
        }

      }

      if(verbose){
        if(status == nReport){
          Rprintf("Location: %i of %i, %3.2f%%\n", i, q, 100.0*i/q);
#ifdef Win32
          R_FlushConsole();
#endif
          status = 0;
        }
      }
      status++;
      R_CheckUserInterrupt();
    }

    if(verbose){
      Rprintf("Location: %i of %i, %3.2f%%\n", i, q, 100.0*i/q);
#ifdef Win32
      R_FlushConsole();
#endif
    }

    //make return object
    SEXP result_r, resultName_r;
    int nResultListObjs = 2;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, y0_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("p.y.0"));

    SET_VECTOR_ELT(result_r, 1, w0_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("p.w.0"));

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);

    return(result_r);

  }

  SEXP WithbetaPredict(SEXP y_r, SEXP X_r, SEXP coords_r, SEXP n_r, SEXP p_r, SEXP m_r,
                     SEXP coords0_r, SEXP X0_r,
                     SEXP q_r, SEXP nnIndx0_r,
                     SEXP thetaSamples_r,
                     SEXP betaSamples_r,
                     SEXP wSamples_r,
                     SEXP nSamples_r,
                     SEXP family_r,
                     SEXP covModel_r,
                     SEXP nThreads_r, SEXP verbose_r,
                     SEXP nReport_r){

    int h, i, j, k, l, s, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    char const *lower = "L";

    //get args
    double *y = REAL(y_r);
    double *X = REAL(X_r);
    double *coords = REAL(coords_r);
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int mm = m*m;

    double *X0 = REAL(X0_r);
    double *coords0 = REAL(coords0_r);
    int q = INTEGER(q_r)[0];
    int p = INTEGER(p_r)[0];

    int *nnIndx0 = INTEGER(nnIndx0_r);
    double *beta = REAL(betaSamples_r);
    double *theta = REAL(thetaSamples_r);
    double *w = REAL(wSamples_r);

    int nSamples = INTEGER(nSamples_r)[0];
    int family = INTEGER(family_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    int nReport = INTEGER(nReport_r)[0];

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\t     Prediction description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("Model fit with %i observations.\n\n", n);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Predicting at %i locations.\n\n", q);
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i threads.\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(family == 1){
      if(corName != "matern"){
        nTheta = 3;//zeta^2, tau^2, phi
        zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
      }else{
        nTheta = 4;//zeta^2, tau^2, phi, nu
        zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
      }
    }else{//family is binomial
      if(corName != "matern"){
        nTheta = 2;//zeta^2, phi
        zetaSqIndx = 0; phiIndx = 1;
      }else{
        nTheta = 3;//zeta^2, phi, nu
        zetaSqIndx = 0; phiIndx = 1; nuIndx = 2;
      }
    }

    //get max nu
    double nuMax = 0;
    int nb = 0;

    if(corName == "matern"){
      for(i = 0; i < nSamples; i++){
        if(theta[i*nTheta+nuIndx] > nuMax){
          nuMax = theta[i*nTheta+nuIndx];
        }
      }

      nb = 1+static_cast<int>(floor(nuMax));
    }

    double *bk = (double *) R_alloc(nThreads*nb, sizeof(double));

    double *C = (double *) R_alloc(nThreads*mm, sizeof(double)); zeros(C, nThreads*mm);
    double *c = (double *) R_alloc(nThreads*m, sizeof(double)); zeros(c, nThreads*m);
    double *tmp_m  = (double *) R_alloc(nThreads*m, sizeof(double));
    double phi = 0, nu = 0, zetaSq = 0, tauSq = 0, d;
    int threadID = 0, status = 0;

    SEXP y0_r, w0_r;
    PROTECT(y0_r = allocMatrix(REALSXP, q, nSamples)); nProtect++;
    PROTECT(w0_r = allocMatrix(REALSXP, q, nSamples)); nProtect++;
    double *y0 = REAL(y0_r);
    double *w0 = REAL(w0_r);

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\t\t         Predicting\n");
      Rprintf("----------------------------------------\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int zIndx = -1;
    double *wZ = (double *) R_alloc(q*nSamples, sizeof(double));

    double *yZ = NULL;
    if(family == 1){
      yZ = (double *) R_alloc(q*nSamples, sizeof(double));
    }

    GetRNGstate();

    for(i = 0; i < q*nSamples; i++){
      wZ[i] = rnorm(0.0,1.0);
    }

    if(family == 1){
      for(i = 0; i < q*nSamples; i++){
        yZ[i] = rnorm(0.0,1.0);
      }
    }

    PutRNGstate();

    for(i = 0; i < q; i++){
#ifdef _OPENMP
#pragma omp parallel for private(threadID, phi, nu, zetaSq, tauSq, k, l, d, info)
#endif
      for(s = 0; s < nSamples; s++){
#ifdef _OPENMP
        threadID = omp_get_thread_num();
#endif
        phi = theta[s*nTheta+phiIndx];
        //Rprintf("phi %f",phi);
        if(corName == "matern"){
          nu = theta[s*nTheta+nuIndx];
        }
        zetaSq = theta[s*nTheta+zetaSqIndx];
        //Rprintf("zetaSq %f",zetaSq);
        if(family == 1){
          tauSq = theta[s*nTheta+tauSqIndx];
        }
        //Rprintf("tauSq %f",tauSq);
        for(k = 0; k < m; k++){
          d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords0[i], coords0[q+i]);
          c[threadID*m+k] = zetaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
          for(l = 0; l < m; l++){
            d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords[nnIndx0[i+q*l]], coords[n+nnIndx0[i+q*l]]);
            C[threadID*mm+l*m+k] = zetaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
          }
        }

        F77_NAME(dpotrf)(lower, &m, &C[threadID*mm], &m, &info FCONE); if(info != 0){error("c++ error: dpotrf failed\n");}
        F77_NAME(dpotri)(lower, &m, &C[threadID*mm], &m, &info FCONE); if(info != 0){error("c++ error: dpotri failed\n");}

        F77_NAME(dsymv)(lower, &m, &one, &C[threadID*mm], &m, &c[threadID*m], &inc, &zero, &tmp_m[threadID*m], &inc FCONE);

        d = 0;
        for(k = 0; k < m; k++){
          d += tmp_m[threadID*m+k]*w[s*n+nnIndx0[i+q*k]];
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        zIndx++;

        w0[s*q+i] = sqrt(zetaSq - F77_NAME(ddot)(&m, &tmp_m[threadID*m], &inc, &c[threadID*m], &inc))*wZ[zIndx] + d;

        if(family == 1){
          y0[s*q+i] = F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) + sqrt(tauSq)*yZ[zIndx] +  w0[s*q+i]; //F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) +
        }else{//binomial
          y0[s*q+i] = F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) + w0[s*q+i]; //F77_NAME(ddot)(&p, &X0[i], &q, &beta[s*p], &inc) +
        }

      }

      if(verbose){
        if(status == nReport){
          Rprintf("Location: %i of %i, %3.2f%%\n", i, q, 100.0*i/q);
#ifdef Win32
          R_FlushConsole();
#endif
          status = 0;
        }
      }
      status++;
      R_CheckUserInterrupt();
    }

    if(verbose){
      Rprintf("Location: %i of %i, %3.2f%%\n", i, q, 100.0*i/q);
#ifdef Win32
      R_FlushConsole();
#endif
    }

    //make return object
    SEXP result_r, resultName_r;
    int nResultListObjs = 2;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, y0_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("p.y.0"));

    SET_VECTOR_ELT(result_r, 1, w0_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("p.w.0"));

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);

    return(result_r);

  }

  SEXP NNGP_joint_samplingcpp(SEXP n_r, SEXP nnIndxLU_vi_r, SEXP nnIndx_vi_r,
                          SEXP w_mu_r,
                          SEXP A_vi_r, SEXP S_vi_r,
                          SEXP sim_r, SEXP sim_number_r,
                          SEXP p_r, SEXP sim_beta_r,
                          SEXP E_vi_r, SEXP A_beta_r, SEXP L_beta_r, SEXP mu_beta_r,
                          SEXP IndxLU_beta_r){

    int i, j, k, l, index, nProtect=0;
    int n = INTEGER(n_r)[0];
    double *sim = REAL(sim_r);
    double *A_vi = REAL(A_vi_r);
    double *S_vi = REAL(S_vi_r);
    double *w_mu = REAL(w_mu_r);
    int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r);
    int *nnIndx_vi = INTEGER(nnIndx_vi_r);
    int sim_number = INTEGER(sim_number_r)[0];
    int tot_length = n*sim_number;

    int p = INTEGER(p_r)[0];
    double *sim_beta = REAL(sim_beta_r);
    double *E_vi = REAL(E_vi_r);
    double *A_beta = REAL(A_beta_r);
    double *L_beta = REAL(L_beta_r);
    double *mu_beta = REAL(mu_beta_r);
    int *IndxLU_beta = INTEGER(IndxLU_beta_r);

    SEXP sim_cor_r; PROTECT(sim_cor_r = allocVector(REALSXP, tot_length)); nProtect++; double *sim_cor = REAL(sim_cor_r);
    SEXP w_sample_r; PROTECT(w_sample_r = allocVector(REALSXP, tot_length)); nProtect++; double *w_sample = REAL(w_sample_r);

    SEXP sim_cor_beta_r; PROTECT(sim_cor_beta_r = allocVector(REALSXP, p*sim_number)); nProtect++; double *sim_cor_beta = REAL(sim_cor_beta_r);
    SEXP beta_sample_r; PROTECT(beta_sample_r = allocVector(REALSXP, p*sim_number)); nProtect++; double *beta_sample = REAL(beta_sample_r);

    for(int im = 0; im < sim_number; im++){
      // update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
      // update_uvec_ubvec(u_vec, ub_vec, epsilon_vec,  z_vec, A_vi, A_beta, L_beta,
      //                   S_vi, E_vi, n, p, nnIndxLU_vi, nnIndx_vi, IndxLU_beta);
      update_uvec_ubvec(&sim_cor[n*im], &sim_cor_beta[p*im], &sim[n*im],  &sim_beta[p*im], A_vi, A_beta, L_beta,
                        S_vi, E_vi, n, p, nnIndxLU_vi, nnIndx_vi, IndxLU_beta);

      // update_uvec(&sim_cor[n*im], &sim[n*im],  A_vi,  S_vi, n, nnIndxLU_vi, nnIndx_vi);
    }

    for(int im = 0; im < sim_number; im++){
      for(int i = 0; i < n; i++){
        index = im * n + i;
        w_sample[index] = sim_cor[index] + w_mu[i];
      }

      for(int j = 0; j < p; j++){
        index = im * p + j;
        beta_sample[index] = sim_cor_beta[index] + mu_beta[j];
      }

    }

    //return stuff
    SEXP result_r, resultName_r;
    int nResultListObjs = 6;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, sim_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("norm_sim"));

    SET_VECTOR_ELT(result_r, 1, sim_cor_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("sim"));

    SET_VECTOR_ELT(result_r, 2, w_sample_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("w_sample"));

    SET_VECTOR_ELT(result_r, 3, sim_beta_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("norm_sim_beta"));

    SET_VECTOR_ELT(result_r, 4, sim_cor_beta_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("sim_beta"));

    SET_VECTOR_ELT(result_r, 5, beta_sample_r);
    SET_VECTOR_ELT(resultName_r, 5, mkChar("beta_sample"));
    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);


    return(result_r);
  }

  SEXP NNGP_samplingcpp(SEXP n_r, SEXP nnIndxLU_vi_r, SEXP nnIndx_vi_r,
                        SEXP w_mu_r,
                        SEXP A_vi_r, SEXP S_vi_r,
                        SEXP sim_r, SEXP sim_number_r){

    int i, j, k, l, index, nProtect=0;
    int n = INTEGER(n_r)[0];
    double *sim = REAL(sim_r);
    double *A_vi = REAL(A_vi_r);
    double *S_vi = REAL(S_vi_r);
    double *w_mu = REAL(w_mu_r);
    int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r);
    int *nnIndx_vi = INTEGER(nnIndx_vi_r);
    int sim_number = INTEGER(sim_number_r)[0];
    int tot_length = n*sim_number;

    SEXP sim_cor_r; PROTECT(sim_cor_r = allocVector(REALSXP, tot_length)); nProtect++; double *sim_cor = REAL(sim_cor_r);
    SEXP w_sample_r; PROTECT(w_sample_r = allocVector(REALSXP, tot_length)); nProtect++; double *w_sample = REAL(w_sample_r);

    for(int im = 0; im < sim_number; im++){
      update_uvec(&sim_cor[n*im], &sim[n*im],  A_vi,  S_vi, n, nnIndxLU_vi, nnIndx_vi);
    }

    for(int im = 0; im < sim_number; im++){
      for(int i = 0; i < n; i++){
        index = im * n + i;
        w_sample[index] = sim_cor[index] + w_mu[i];
      }
    }

    //return stuff
    SEXP result_r, resultName_r;
    int nResultListObjs = 3;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, sim_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("norm_sim"));

    SET_VECTOR_ELT(result_r, 1, sim_cor_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("sim"));

    SET_VECTOR_ELT(result_r, 2, w_sample_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("w_sample"));

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);


    return(result_r);
  }

  SEXP MFA_samplingcpp(SEXP n_r,
                          SEXP w_mu_r,
                          SEXP w_sigma_sq_r,
                          SEXP sim_r, SEXP sim_number_r){
    int i, j, k, l, index, nProtect=0;
    int n = INTEGER(n_r)[0];
    double *sim = REAL(sim_r);
    double *w_mu = REAL(w_mu_r);
    double *w_sigma_sq = REAL(w_sigma_sq_r);

    int sim_number = INTEGER(sim_number_r)[0];
    int tot_length = n*sim_number;

    SEXP sim_cor_r; PROTECT(sim_cor_r = allocVector(REALSXP, tot_length)); nProtect++; double *sim_cor = REAL(sim_cor_r);
    SEXP w_sample_r; PROTECT(w_sample_r = allocVector(REALSXP, tot_length)); nProtect++; double *w_sample = REAL(w_sample_r);

    for(int im = 0; im < sim_number; im++){
      for(int i = 0; i < n; i++){
        index = im * n + i;
        sim_cor[index] = sim[index] * sqrt(w_sigma_sq[i]);
        // Rprintf("sim_cor[%i]: %f \n", index, sim_cor[index]);
        // Rprintf("sim[%i]: %f \n", index, sim[index]);
        // Rprintf("w_sigma_sq[%i]: %f \n", i, sqrt(w_sigma_sq[i]));
        w_sample[index] = sim_cor[index] + w_mu[i];
      }
    }

    //return stuff
    SEXP result_r, resultName_r;
    int nResultListObjs = 3;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, sim_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("norm_sim"));

    SET_VECTOR_ELT(result_r, 1, sim_cor_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("sim"));

    SET_VECTOR_ELT(result_r, 2, w_sample_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("w_sample"));

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);


    return(result_r);
  }

}

