#include <string>
#include <limits>
#include "util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>

double max_ind(double *a, int n){
  int max_ind = 0;
  double max;
  for (int i = 0; i < n; i++) {
    if (a[i] > max) {
      max = a[i]; // Update max if a greater element is found
      max_ind = i;
    }
  }
  return max_ind;
}

void zeros(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0.0;
}

void zeros_int(int *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0;
}

void ones(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 1.0;
}

void ones_int(int *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 1;
}
double sumsq(double *u_vec, int n){
  double sum = 0.0;
  for(int i = 0; i < n; i++){
    sum += pow(u_vec[i],2);
  }
  return(sum);
}


double var_est(double *u_vec, int n){
  double sum = 0.0;
  double u_mean = 0.0;
  for(int i = 0; i < n; i++){
    u_mean += u_vec[i];
  }
  u_mean = u_mean/n;
  for(int i = 0; i < n; i++){
    sum += pow((u_vec[i]-u_mean),2);
  }
  return(sum);
}

void sum_two_vec(double *vec1, double *vec2, double *output_vec, int n){
  for(int i = 0; i < n; i++){
    output_vec[i] = vec1[i] + vec2[i];
  }
}

void add_to_vec(double *vec1, double *vec2, int n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    vec2[i] += vec1[i];
  }
}

void vecsum(double *cum_vec, double *input_vec, int scale,int n){
  for(int i = 0; i < n; i++){
    cum_vec[i] += input_vec[i]/scale;
  }
}

void vecsumsq(double *cum_vec, double *input_vec, int scale,int n){
  for(int i = 0; i < n; i++){
    cum_vec[i] += pow(input_vec[i],2)/scale;
  }
}

void get_nBatchIndx(int n, int nBatch, int n_mb, int *nBatchIndx, int *nBatchLU){
  if(nBatch>1){
    for(int i = 0; i < (nBatch - 1); i++){
      nBatchLU[i] = i*n_mb;
      nBatchIndx[i] = n_mb;
    }
    nBatchLU[nBatch - 1] = (nBatch - 1)*n_mb;
    nBatchIndx[nBatch - 1] = n - (nBatch - 1)*n_mb;
  }else{
    nBatchLU[0] = 0;
    nBatchIndx[0] = n_mb;
  }
}

void create_sign(double *vec, int *sign_vec,int n_per){
  for(int i = 0; i < n_per; i++){
    if(vec[i]>0){sign_vec[i] = 1;}else{sign_vec[i] = 0;}
  }
}

void checksign(int *sign_vec1, int *sign_vec2, int *check_vec, int n_per){
  for(int i = 0; i < n_per; i++){
    if(sign_vec1[i]!=sign_vec2[i]){check_vec[i] = 1;}else{check_vec[i] = 0;}
  }
}

double prodsign(int *check_vec, int n_per){
  double prod = 1;
  for(int i = 0; i < n_per; i++){
    prod = prod * check_vec[i];
  }
  return prod;
}
double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}
double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}
void getNNIndx(int i, int m, int &iNNIndx, int &iNN){

  if(i == 0){
    iNNIndx = 0;//this should never be accessed
    iNN = 0;
    return;
  }else if(i < m){
    iNNIndx = static_cast<int>(static_cast<double>(i)/2*(i-1));
    iNN = i;
    return;
  }else{
    iNNIndx = static_cast<int>(static_cast<double>(m)/2*(m-1)+(i-m)*m);
    iNN = m;
    return;
  }
}

void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){

  int i, j, iNNIndx, iNN;
  double d;

  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

  for(i = 0; i < nIndx; i++){
    nnDist[i] = std::numeric_limits<double>::infinity();
  }

#ifdef _OPENMP
#pragma omp parallel for private(j, iNNIndx, iNN, d)
#endif
  for(i = 0; i < n; i++){
    getNNIndx(i, m, iNNIndx, iNN);
    nnIndxLU[i] = iNNIndx;
    nnIndxLU[n+i] = iNN;
    if(i != 0){
      for(j = 0; j < i; j++){
        d = dist2(coords[i], coords[n+i], coords[j], coords[n+j]);
        if(d < nnDist[iNNIndx+iNN-1]){
          nnDist[iNNIndx+iNN-1] = d;
          nnIndx[iNNIndx+iNN-1] = j;
          rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
        }
      }
    }
  }
}

std::string getCorName(int i){

  if(i == 0){
    return "exponential";
  }else if(i == 1){
    return "spherical";
  }else if(i == 2){
    return "matern";
  }else if(i == 3){
    return "gaussian";
  }else{
    error("c++ error: cov.model is not correctly specified");
  }

}

double spCor(double &D, double &phi, double &nu, int &covModel, double *bk){

  //0 exponential
  //1 spherical
  //2 matern
  //3 gaussian

  if(covModel == 0){//exponential

    return exp(-phi*D);

  }else if(covModel == 1){//spherical

    if(D > 0 && D <= 1.0/phi){
      return 1.0 - 1.5*phi*D + 0.5*pow(phi*D,3);
    }else if(D >= 1.0/phi){
      return 0.0;
    }else{
      return 1.0;
    }
  }else if(covModel == 2){//matern

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    if(D*phi > 0.0){
      return pow(D*phi, nu)/(pow(2, nu-1)*gammafn(nu))*bessel_k_ex(D*phi, nu, 1.0, bk);//thread safe bessel
    }else{
      return 1.0;
    }
  }else if(covModel == 3){//gaussian

    return exp(-1.0*(pow(phi*D,2)));

  }else{
    error("c++ error: cov.model is not correctly specified");
  }
}

//Description: computes the quadratic term.
double Q(double *B, double *F, double *u, double *v, int n, int *nnIndx, int *nnIndxLU){

  double a, b, q = 0;
  int i, j;

#ifdef _OPENMP
#pragma omp parallel for private(a, b, j) reduction(+:q)
#endif
  for(i = 0; i < n; i++){
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*u[nnIndx[nnIndxLU[i]+j]];
      b += B[nnIndxLU[i]+j]*v[nnIndx[nnIndxLU[i]+j]];
    }
    q += (u[i] - a)*(v[i] - b)/F[i];
  }

  return(q);
}


int which(int a, int *b, int n){
  int i;
  for(i = 0; i < n; i++){
    if(a == b[i]){
      return(i);
    }
  }

  error("c++ error: which failed");
  return -9999;
}
///////////////////////////////////////////////////////////////////
//code book
///////////////////////////////////////////////////////////////////

//Description: using the fast mean-distance-ordered nn search by Ra and Kim 1993
//Input:
//ui = is the index for which we need the m nearest neighbors
//m = number of nearest neighbors
//n = number of observations, i.e., length of u
//sIndx = the NNGP ordering index of length n that is pre-sorted by u
//u = x+y vector of coordinates assumed sorted on input
//rSIndx = vector or pointer to a vector to store the resulting nn sIndx (this is at most length m for ui >= m)
//rNNDist = vector or point to a vector to store the resulting nn Euclidean distance (this is at most length m for ui >= m)


double dmi(double *x, double *c, int inc){
  return pow(x[0]+x[inc]-c[0]-c[inc], 2);
}

double dei(double *x, double *c, int inc){
  return pow(x[0]-c[0],2)+pow(x[inc]-c[inc],2);
}

void fastNN(int m, int n, double *coords, int ui, double *u, int *sIndx, int *rSIndx, double *rSNNDist){

  int i,j,k;
  bool up, down;
  double dm, de;

  //rSNNDist will hold de (i.e., squared Euclidean distance) initially.
  for(i = 0; i < m; i++){
    rSNNDist[i] = std::numeric_limits<double>::infinity();
  }

  i = j = ui;

  up = down = true;

  while(up || down){

    if(i == 0){
      down = false;
    }

    if(j == (n-1)){
      up = false;
    }

    if(down){

      i--;

      dm = dmi(&coords[sIndx[ui]], &coords[sIndx[i]], n);

      if(dm > 2*rSNNDist[m-1]){
        down = false;

      }else{
        de = dei(&coords[sIndx[ui]], &coords[sIndx[i]], n);

        if(de < rSNNDist[m-1] && sIndx[i] < sIndx[ui]){
          rSNNDist[m-1] = de;
          rSIndx[m-1] = sIndx[i];
          rsort_with_index(rSNNDist, rSIndx, m);
        }

      }
    }//end down

    if(up){

      j++;

      dm = dmi(&coords[sIndx[ui]], &coords[sIndx[j]], n);

      if(dm > 2*rSNNDist[m-1]){
        up = false;

      }else{
        de = dei(&coords[sIndx[ui]], &coords[sIndx[j]], n);

        if(de < rSNNDist[m-1] && sIndx[j] < sIndx[ui]){
          rSNNDist[m-1] = de;
          rSIndx[m-1] = sIndx[j];
          rsort_with_index(rSNNDist, rSIndx, m);
        }

      }

    }//end up

  }

  for(i = 0; i < m; i++){
    rSNNDist[i] = sqrt(rSNNDist[i]);
  }

  return;
}


void mkNNIndxCB(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){
  int i, iNNIndx, iNN;

  int *sIndx = new int[n];
  double *u = new double[n];

  for(i = 0; i < n; i++){
    sIndx[i] = i;
    u[i] = coords[i]+coords[n+i];
  }

  rsort_with_index(u, sIndx, n);

  //make nnIndxLU and fill nnIndx and d
#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
  for(i = 0; i < n; i++){ //note this i indexes the u vector
    getNNIndx(sIndx[i], m, iNNIndx, iNN);
    nnIndxLU[sIndx[i]] = iNNIndx;
    nnIndxLU[n+sIndx[i]] = iNN;
    fastNN(iNN, n, coords, i, u, sIndx, &nnIndx[iNNIndx], &nnDist[iNNIndx]);
  }
}




//trees
Node *miniInsert(Node *Tree, double *coords, int index, int d,int n){

  int P = 2;

  if(Tree==NULL){
    return new Node(index);
  }

  if(coords[index]<=coords[Tree->index]&&d==0){
    Tree->left=miniInsert(Tree->left,coords,index,(d+1)%P,n);
  }

  if(coords[index]>coords[Tree->index]&&d==0){
    Tree->right=miniInsert(Tree->right,coords,index,(d+1)%P,n);
  }

  if(coords[index+n]<=coords[Tree->index+n]&&d==1){
    Tree->left=miniInsert(Tree->left,coords,index,(d+1)%P,n);
  }

  if(coords[index+n]>coords[Tree->index+n]&&d==1){
    Tree->right=miniInsert(Tree->right,coords,index,(d+1)%P,n);
  }

  return Tree;
}

void get_nn(Node *Tree, int index, int d, double *coords, int n, double *nnDist, int *nnIndx, int iNNIndx, int iNN, int check){

  int P = 2;

  if(Tree==NULL){
    return;
  }

  double disttemp= dist2(coords[index],coords[index+n],coords[Tree->index],coords[Tree->index+n]);

  if(index!=Tree->index && disttemp<nnDist[iNNIndx+iNN-1]){
    nnDist[iNNIndx+iNN-1]=disttemp;
    nnIndx[iNNIndx+iNN-1]=Tree->index;
    //fSort(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
    rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
  }

  Node *temp1=Tree->left;
  Node *temp2=Tree->right;

  if(d==0){

    if(coords[index]>coords[Tree->index]){
      std::swap(temp1,temp2);
    }

    get_nn(temp1,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN, check);

    if(fabs(coords[Tree->index]-coords[index])>nnDist[iNNIndx+iNN-1]){
      return;
    }

    get_nn(temp2,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN, check);
  }

  if(d==1){

    if(coords[index+n]>coords[Tree->index+n]){
      std::swap(temp1,temp2);
    }

    get_nn(temp1,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN,check);

    if(fabs(coords[Tree->index+n]-coords[index+n])>nnDist[iNNIndx+iNN-1]){
      return;
    }

    get_nn(temp2,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN,check);
  }

}

void mkUIndx0(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU){

  int iNNIndx, iNN, i, j, k, l, h;

  for(i = 0, l = 0; i < n; i++){
    uIndxLU[i] = l;
    for(j = 0, h = 0; j < n; j++){
      getNNIndx(j, m, iNNIndx, iNN);
      for(k = 0; k < iNN; k++){
        if(nnIndx[iNNIndx+k] == i){
          uIndx[l+h] = j;
          h++;
        }
      }
    }
    l += h;
    uIndxLU[n+i] = h;
    R_CheckUserInterrupt();
  }
}

void mkUIndx1(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU){

  int iNNIndx, iNN, i, j, k, l, h;

  for(i = 0, l = 0; i < n; i++){
    uIndxLU[i] = l;
    for(j = n-1, h = 0; j > i; j--){
      getNNIndx(j, m, iNNIndx, iNN);
      for(k = 0; k < iNN; k++){
        if(nnIndx[iNNIndx+k] == i){
          uIndx[l+h] = j;
          h++;
        }
      }
    }
    l += h;
    uIndxLU[n+i] = h;
    R_CheckUserInterrupt();
  }
}


void mkUIndx2(int n, int m, int* nnIndx, int *nnIndxLU, int* uIndx, int* uIndxLU){

  int i, j, k;
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

  //int *j_A = new int[nIndx]; is nnIndx
  int *i_nnIndx = new int[n+1];
  //int *j_A_csc = new int[nIndx];//uIndx
  int *i_A_csc = new int[n+1];

  for(i = 0, k = 0; i < n; i++){
    if(nnIndxLU[n+i] == 0){//excludes rows with no elements, i.e., the first row because it is zero by design A[0,0] = 0
      i_nnIndx[0] = 0;
    }else{
      i_nnIndx[k] = i_nnIndx[k-1]+nnIndxLU[n+i-1];
    }
    k++;
  }
  i_nnIndx[n] = i_nnIndx[0]+nIndx;

  crs_csc(n, i_nnIndx, nnIndx, i_A_csc, uIndx);

  for(i = 0; i < n; i++){
    uIndxLU[i] = i_A_csc[i];
    uIndxLU[i+n] = i_A_csc[i+1]-i_A_csc[i];
  }

  delete[] i_nnIndx;
  delete[] i_A_csc;

}

void crs_csc(int n, int *i_A, int *j_A, int *i_B, int *j_B){

  int i, j, col, cumsum, temp, row, dest, last;

  int nnz = i_A[n];

  for(i = 0; i < n; i++){
    i_B[i] = 0;
  }

  for(i = 0; i < nnz; i++){
    i_B[j_A[i]]++;
  }

  //cumsum the nnz per column to get i_B[]
  for(col = 0, cumsum = 0; col < n; col++){
    temp  = i_B[col];
    i_B[col] = cumsum;
    cumsum += temp;
  }
  i_B[n] = nnz;

  for(row = 0; row < n; row++){
    for(j = i_A[row]; j < i_A[row+1]; j++){
      col  = j_A[j];
      dest = i_B[col];

      j_B[dest] = row;
      i_B[col]++;
    }
  }

  for(col = 0, last = 0; col <= n; col++){
    temp  = i_B[col];
    i_B[col] = last;
    last = temp;
  }
}


void mkNNIndxTree0(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){

  int i, iNNIndx, iNN;
  double d;
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
  int BUCKETSIZE = 10;


  for(i = 0; i < nIndx; i++){
    nnDist[i] = std::numeric_limits<double>::infinity();
  }

  Node *Tree=NULL;
  int time_through=-1;

  for(i=0;i<n;i++){
    getNNIndx(i, m, iNNIndx, iNN);
    nnIndxLU[i] = iNNIndx;
    nnIndxLU[n+i] = iNN;
    if(time_through==-1){
      time_through=i;
    }

    if(i!=0){
      for(int j = time_through; j < i; j++){
	getNNIndx(i, m, iNNIndx, iNN);
	d = dist2(coords[i], coords[i+n], coords[j], coords[n+j]);
	if(d < nnDist[iNNIndx+iNN-1]){
	  nnDist[iNNIndx+iNN-1] = d;
	  nnIndx[iNNIndx+iNN-1] = j;

	  //fSort(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
	  rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
	}
      }


      if(i%BUCKETSIZE==0){

#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
	for(int j=time_through;j<time_through+BUCKETSIZE;j++){

	  getNNIndx(j, m, iNNIndx, iNN);
	  get_nn(Tree,j,0, coords,n, nnDist,nnIndx,iNNIndx,iNN,i-BUCKETSIZE);
	}


	for(int j=time_through;j<time_through+BUCKETSIZE;j++){
	  Tree=miniInsert(Tree,coords,j,0, n);
	}

	time_through=-1;
      }
      if(i==n-1){

#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
	for(int j=time_through;j<n;j++){
	  getNNIndx(j, m, iNNIndx, iNN);
	  get_nn(Tree,j,0, coords,n, nnDist,nnIndx,iNNIndx,iNN,i-BUCKETSIZE);
	}

      }
    }
    if(i==0){
      Tree=miniInsert(Tree,coords,i,0,n);
      time_through=-1;
    }
  }

  delete Tree;
}
