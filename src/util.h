#include <string>
double max_ind(double *a, int n);
void zeros(double *a, int n);
void zeros_int(int *a, int n);
void ones(double *a, int n);
void ones_int(int *a, int n);
double sumsq(double *u_vec, int n);
double var_est(double *u_vec, int n);
void sum_two_vec(double *vec1, double *vec2, double *output_vec, int n);
void add_to_vec(double *vec1, double *vec2, int n);
void vecsum(double *cum_vec, double *input_vec, int scale,int n);
void vecsumsq(double *cum_vec, double *input_vec, int scale,int n);

void get_nBatchIndx(int n, int nBatch, int n_mb, int *nBatchIndx, int* nBatchLU);
void create_sign(double *vec, int *sign_vec,int n_per);
void checksign(int *sign_vec1, int *sign_vec2, int *check_vec, int n_per);
double prodsign(int *check_vec, int n_per);
double dist2(double &a1, double &a2, double &b1, double &b2);
double logit(double theta, double a, double b);
double logitInv(double z, double a, double b);
//Description: given a location's index i and number of neighbors m this function provides the index to i and number of neighbors in nnIndx
void getNNIndx(int i, int m, int &iNNIndx, int &iNN);

//Description: creates the nearest neighbor index given pre-ordered location coordinates.
//Input:
//n = number of locations
//m = number of nearest neighbors
//coords = ordered coordinates for the n locations
//Output:
//nnIndx = set of nearest neighbors for all n locations (on return)
//nnDist = euclidean distance corresponding to nnIndx (on return)
//nnIndxLU = nx2 look-up matrix with row values correspond to each location's index in nnIndx and number of neighbors (columns 1 and 2, respectively)
//Note: nnIndx and nnDist must be of length (1+m)/2*m+(n-m-1)*m on input. nnIndxLU must also be allocated on input.
void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);


std::string getCorName(int i);

double spCor(double &D, double &phi, double &nu, int &covModel, double *bk);

double Q(double *B, double *F, double *u, double *v, int n, int *nnIndx, int *nnIndxLU);
int which(int a, int *b, int n);
void mkNNIndxCB(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);
void mkUIndx0(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU);
void mkUIndx1(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU);
void mkUIndx2(int n, int m, int* nnIndx, int *nnIndxLU, int* uIndx, int* uIndxLU);
void crs_csc(int n, int *i_A, int *j_A, int *i_B, int *j_B);
//trees
struct Node{
	int index; // which point I am
	Node *left;
	Node *right;
	Node (int i) { index = i; left = right = NULL; }
};

Node *miniInsert(Node *Tree, double *coords, int index, int d,int n);

void get_nn(Node *Tree, int index, int d, double *coords, int n, double *nnDist, int *nnIndx, int iNNIndx, int iNN, int check);

void mkNNIndxTree0(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);
