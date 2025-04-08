#include <string>

void zeros_minibatch(double *a, int n, int BatchSize, int *nBatchLU, int batch_index);
void vecsum_minibatch(double *cum_vec, double *input_vec, int scale,int n, int BatchSize, int *nBatchLU, int batch_index);
void get_num_nIndx_col(int *nnIndx, int nIndx, int *numIndxCol);
void get_cumnum_nIndx_col(int *numIndxCol, int n, int *cumnumIndxCol);
void get_sum_nnIndx(int *sumnnIndx, int n, int m);
void findPositions(int *positions, int *arr, int nIndx, int x);
void get_nnIndx_col(int *nnIndx, int n, int nIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol);
void get_nnIndx_nn_col(int *nnIndx, int n, int m, int nIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol, int *sumnnIndx);

double updateBF_logdet(double *B, double *F, double *c, double *C, double *D, double *d, int *nnIndxLU, int *CIndx, int n, double *theta, int covModel, int nThreads, double fix_nugget);
void solve_B_F(double *B, double *F, double *norm_residual_boot, int n, int *nnIndxLU, int *nnIndx, double *residual_boot);
void product_B_F(double *B, double *F, double *residual_nngp, int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp);
void product_B_F_vec(double *B, double *F, double *input_vec, int n, int *nnIndxLU, int *nnIndx, double *output_vec, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol);
void diagonal(double *B, double *F, double *diag_output, int n, int *nnIndx, int *nnIndxLU);

void update_uvec(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi);
void a_gradient_fun(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                    double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    double *u_vec_temp, double *u_vec_temp2);
void a_gradient_fun_all(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int m_vi, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        double *u_vec_temp, double *u_vec_temp2,
                        double *derivative_neighbour, double *derivative_neighbour_a, double *derivative_store);

void gamma_gradient_fun(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                        double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                        int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                        double *u_vec_temp, double *u_vec_temp2, double *gradient);

void gamma_gradient_fun_all(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                            double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                            int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                            double *u_vec_temp, double *u_vec_temp2, double *gradient,
                            double *derivative_neighbour, double *derivative_neighbour_a,
                            double *derivative_store_gamma);
void ELBO_u_vec(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx,
                double *u_vec_mean, int Trace_MC, double ELBO_MC);
void product_B_F_minibatch(double *B, double *F, double *residual_nngp,
                           int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                           int BatchSize, int *nBatchLU, int batch_index);
void product_B_F_vec_minibatch(double *B, double *F, double *input_vec,
                               int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                               int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                               int BatchSize, int *nBatchLU, int batch_index, int nBatch);
void updateBF_minibatch(double *B, double *F, double *c, double *C,
                        double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                        double zetaSq, double phi, double nu, int covModel, double *bk, double nuUnifb,
                        int BatchSize, int *nBatchLU, int batch_index);
void updateBF(double *B, double *F, double *c, double *C, double *coords,
              int *nnIndx, int *nnIndxLU, int n, int m, double zetaSq, double phi,
              double nu, int covModel, double *bk, double nuUnifb);
double Q_mini_batch(double *B, double *F, double *u_mb, double *v_mb,
                    int BatchSize, int *nBatchLU, int batch_index, int n,
                    int *nnIndx, int *nnIndxLU);
void update_uvec_minibatch(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi,
                           int n, int *nnIndxLU_vi, int *nnIndx_vi,
                           int BatchSize, int *nBatchLU, int batch_index);
void gamma_gradient_fun_minibatch(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                             double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                             int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                             double *u_vec_temp, double *u_vec_temp2, double *gradient,
                             int BatchSize, int *nBatchLU, int batch_index, int nBatch);
void a_gradient_fun_minibatch(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                              double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                              double *u_vec_temp, double *u_vec_temp2,
                              int BatchSize, int *nBatchLU, int batch_index, int nBatch) ;
double var_est(double *u_vec, double *epsilon_vec, double *B, double *F, int *nnIndx, int *nnIndxLU, int n,
               double *w_mu, int BatchSize, int *nBatchLU, int batch_index, int nBatch);
void mu_grad(double *w_mu, double *B, double *F, int n,
        int *nnIndx, int *nnIndxLU, int *nnIndxCol,
        int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
        int BatchSize, int *nBatchLU, int batch_index, int nBatch,
        double *w_mu_temp, double zetaSq);
void find_set(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
              int BatchSize, int *nBatchLU, int batch_index,
              int *result_arr, int &result_index, int *temp_arr,
              int &temp_index, int *tempsize_vec, int *seen_values);
void update_uvec_minibatch_plus(double *u_vec, double *epsilon_vec, double *A_vi, double *S_vi,
                                int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                int batch_index,
                                int *final_result_vec, int *nBatchLU_temp, int tempsize);
double Q_mini_batch_plus(double *B, double *F, double *u_mb, double *v_mb,
                         int batch_index, int n,
                         int *nnIndx, int *nnIndxLU,
                         int *final_result_vec, int *nBatchLU_temp, int tempsize);
void updateBF_minibatch_plus(double *B, double *F, double *c, double *C,
                             double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                             double zetaSq, double phi, double nu, int covModel, double *bk, double nuUnifb,
                             int batch_index,
                             int *final_result_vec, int *nBatchLU_temp, int tempsize);
void product_B_F_minibatch_plus(double *B, double *F, double *residual_nngp,
                               int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                               int batch_index,
                               int *final_result_vec, int *nBatchLU_temp, int tempsize);
void product_B_F_vec_minibatch_plus(double *B, double *F, double *input_vec,
                                   int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                                   int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   int batch_index,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize);
void zeros_minibatch_plus(double *a, int n, int batch_index,int *final_result_vec, int *nBatchLU_temp, int tempsize);
void vecsum_minibatch_plus(double *cum_vec, double *input_vec, int scale,int n,
                           int batch_index,int *final_result_vec, int *nBatchLU_temp, int tempsize);

void gamma_gradient_fun_minibatch_plus(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *gradient,
                                       int batch_index,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize);

void a_gradient_fun_minibatch_plus(double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                              double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                              double *u_vec_temp, double *u_vec_temp2,
                              int batch_index,
                              int *final_result_vec, int *nBatchLU_temp, int tempsize);

void find_set_mb(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                 int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                 int BatchSize, int *nBatchLU, int batch_index,
                 int *result_arr, int &result_index, int *temp_arr,
                 int &temp_index, int *tempsize_vec, int *seen_values) ;
double Expectation_B_F(double *B, double *F, double *w_mu, double *u_vec,
                       int n, int *nnIndxLU, int *nnIndx,
                       int BatchSize, int *nBatchLU, int batch_index);
// Comparator function for qsort
int compare_ints(const void* a, const void* b);

void find_set_nngp(int n, int *nnIndx, int *nnIndxLU, int BatchSize, int *nBatchLU, int batch_index,
                   int *seen_values,
                   int *intersect_result, int *intersect_sizes, int *intersect_start_indices,
                   int *complement_first_result, int *complement_first_sizes, int *complement_first_start_indices,
                   int *complement_second_result, int *complement_second_sizes, int *complement_second_start_indices,
                   int &intersect_result_index, int &complement_first_result_index, int &complement_second_result_index) ;

void mu_grad_intersect(double *y, double *w_mu,
                       int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                       int BatchSize, int *nBatchLU, int batch_index,
                       int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                       double *theta, int tauSqIndx,
                       double *B, double *F,
                       int *intersect_start_indices, int *intersect_sizes,
                       int* final_intersect_vec,
                       double *gradient_mu_temp) ;

void mu_grad_complement_1(double *y, double *w_mu,
                          int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                          int BatchSize, int *nBatchLU, int batch_index,
                          int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                          double *theta, int tauSqIndx,
                          double *B, double *F,
                          int *complement_first_start_indices, int *complement_first_sizes,
                          int* final_complement_1_vec,
                          double *gradient_mu_temp);
void mu_grad_complement_2(double *w_mu,
                          int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                          int BatchSize, int *nBatchLU, int batch_index,
                          int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                          double *theta, int tauSqIndx,
                          double *B, double *F,
                          int *complement_second_start_indices, int *complement_second_sizes,
                          int* final_complement_2_vec,
                          double *gradient_mu_temp);


void gamma_gradient_fun_minibatch_nngp(double *y, double *w_mu, double *u_vec,
                                       double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *gradient, double *w_mu_temp,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);


void a_gradient_fun_minibatch_nngp(double *y, double *w_mu, double *u_vec,
                                   double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *gradient, double *w_mu_temp,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);
void product_B_F_minibatch_term1(double *B, double *F, double *residual_nngp,
                                 int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                                 int batch_index,
                                 int *final_result_vec, int *nBatchLU_temp, int tempsize);
void gamma_gradient_fun_minibatch_test(double *y, double *w_mu_update,
                                       double *w_vec_temp_dF, double *w_vec_temp2,
                                       double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                       double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                       int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                                       double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                       int batch_index, int BatchSize, int *nBatchLU,
                                       int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                       int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                       int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);


void a_gradient_fun_minibatch_test(double *y, double *w_mu_update,
                                   double *w_vec_temp_dF, double *w_vec_temp2,
                                   double *u_vec, double *epsilon_vec, double *a_gradient, double *gradient_const, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                   double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                   double *u_vec_temp, double *u_vec_temp2, double *u_vec_temp_dF, double *gradient,
                                   int batch_index, int BatchSize, int *nBatchLU,
                                   int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                   int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                                   int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

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
                                       int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

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
                                   int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

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
                                      double *derivative_store_gamma);

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
                                  double *derivative_neighbour, double *derivative_neighbour_a, double *derivative_store);
void MFA_sigmasq_grad_term1(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx,
                            double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            double *gradient_sigmasq_temp);

void MFA_sigmasq_grad(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,
                      int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                      int BatchSize, int *nBatchLU, int batch_index,
                      int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                      double *theta, int tauSqIndx,
                      double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                      int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                      int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                      int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);
void product_B_F_minibatch_dF(double *B, double *F, double *residual_nngp,
                              int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp,
                              int BatchSize, int *nBatchLU, int batch_index);

void mu_grad_intersect_fix(double *y, double *w_mu,
                           int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                           int BatchSize, int *nBatchLU, int batch_index,
                           int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                           double *theta, int tauSqIndx,
                           double *B, double *F,
                           int *intersect_start_indices, int *intersect_sizes,
                           int* final_intersect_vec,
                           double *gradient_mu_temp);
void mu_grad_complement_1_fix(double *y, double *w_mu,
                              int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                              int BatchSize, int *nBatchLU, int batch_index,
                              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                              double *theta, int tauSqIndx,
                              double *B, double *F,
                              int *complement_first_start_indices, int *complement_first_sizes,
                              int* final_complement_1_vec,
                              double *gradient_mu_temp);

void mu_grad_complement_2_fix(double *w_mu,
                              int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                              int BatchSize, int *nBatchLU, int batch_index,
                              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                              double *theta, int tauSqIndx,
                              double *B, double *F,
                              int *complement_second_start_indices, int *complement_second_sizes,
                              int* final_complement_2_vec,
                              double *gradient_mu_temp);

void product_B_F_vec_minibatch_plus_fix(double *B, double *F, double *input_vec,
                                        int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                                        int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                        int BatchSize, int *nBatchLU, int batch_index,
                                        int *final_result_vec, int *nBatchLU_temp, int tempsize);

void shuffleArray(int *array, int n);

void find_set_nngp_shuffle(int *shuffle_array, int n, int *nnIndx, int *nnIndxLU,
                           int BatchSize, int *nBatchLU, int batch_index,
                           int *seen_values,
                           int *intersect_result, int *intersect_sizes, int *intersect_start_indices,
                           int *complement_first_result, int *complement_first_sizes, int *complement_first_start_indices,
                           int *complement_second_result, int *complement_second_sizes, int *complement_second_start_indices,
                           int &intersect_result_index, int &complement_first_result_index, int &complement_second_result_index);

void find_set_mb_shuffle(int *shuffle_array, // Added shuffled_array parameter
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                         int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                         int BatchSize, int *nBatchLU, int batch_index,
                         int *result_arr, int &result_index, int *temp_arr,
                         int &temp_index, int *tempsize_vec, int *seen_values);
bool isValueInArraySubset(int value, int *array, int startIndex, int endIndex) ;
void update_inFlags(int *shuffle_array, int *inFlags, int nm,
                    int n, int *nnIndxLU, int *nnIndx,
                    int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                    int BatchSize, int *nBatchLU, int batch_index,
                    int *final_result_vec, int *nBatchLU_temp, int tempsize);

void product_B_F_vec_minibatch_plus_shuffle(int *shuffle_array, int *inFlags, int nm,
                                            double *B, double *F, double *input_vec,
                                            int n, int *nnIndxLU, int *nnIndx, double *output_vec,
                                            int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                            int BatchSize, int *nBatchLU, int batch_index,
                                            int *final_result_vec, int *nBatchLU_temp, int tempsize);

void MFA_sigmasq_grad_term1_shuffle(int *shuffle_array, int *inFlags, int nm,
                                    int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                                    int BatchSize, int *nBatchLU, int batch_index,
                                    int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                                    double *theta, int tauSqIndx,
                                    double *B, double *F, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                    double *gradient_sigmasq_temp) ;


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
                                          int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

void a_gradient_fun_minibatch_shuffle(int *shuffle_array, int *inFlags,int nm,
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
                                      int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec) ;


double Q_mini_batch_shuffle(int *shuffle_array, double *B, double *F, double *u_mb, double *v_mb,
                            int BatchSize, int *nBatchLU, int batch_index, int n,
                            int *nnIndx, int *nnIndxLU);

double E_quadratic(double *eta_vec,
                   double *F_inv, double *B_over_F, double *Bmat_over_F,
                   int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq);

void updateBF2(double *B, double *F, double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU, int n, int m, double phi, double nu, int covModel, double *bk, double nuUnifb);

void a_gradient_fun2(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                     int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                     double *u_vec_temp, double *u_vec_temp2, int zetaSqIndx,
                     double *F_inv, double *B_over_F, double *Bmat_over_F,
                     int *nnIndxLUSq, int *nnIndxwhich);

void gamma_gradient_fun2(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                         int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                         int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                         double *u_vec_temp, double *u_vec_temp2, double *gradient, int zetaSqIndx,
                         double *F_inv, double *B_over_F, double *Bmat_over_F,
                         int *nnIndxLUSq, int *nnIndxwhich);

void updateBF_quadratic(double *B_temp, double *F_temp, double *Bmat_over_F_temp,
                        double *F_inv, double *B_over_F, double *Bmat_over_F,
                        int nIndx, int nIndSqx,
                        int *nnIndxLUSq,
                        int Trace_N,
                        double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU,
                        int n, int m,
                        double nu, int covModel, double *bk, double nuUnifb,
                        double a_phi, double b_phi,
                        double phimax, double phimin);

void product_B_F_combine(double *eta_vec, double *mid_vec, double *output_vec,
                         double *F_inv, double *B_over_F, double *Bmat_over_F,
                         int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                         int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol, int *nnIndxwhich);

void get_nnIndxwhich(int *nnIndxwhich, int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                     int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol);

double E_quadratic_mb(double *eta_vec,
                      double *F_inv, double *B_over_F, double *Bmat_over_F,
                      int BatchSize, int *nBatchLU, int batch_index,
                      int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq);

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
                           double phimax, double phimin);

void updateBF_minibatch_plus2(double *B, double *F, double *c, double *C,
                              double *coords, int *nnIndx, int *nnIndxLU, int n, int m,
                              double phi, double nu, int covModel, double *bk, double nuUnifb,
                              int batch_index,
                              int* final_result_vec, int *nBatchLU_temp, int tempsize);

void product_B_F_combine_mb(double *eta_vec, double *mid_vec, double *mid2_vec, double *output_vec,
                            double *F_inv, double *B_over_F, double *Bmat_over_F,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            int n, int *nnIndx, int *nnIndxLU, int *nnIndxLUSq,
                            int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                            int *nnIndxwhich);

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
                            int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);


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
                        int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

void MFA_updateBF_quadratic(double *B_temp, double *F_temp, double *Bmat_over_F_temp,
                               double *F_inv, double *B_over_F, double *Bmat_over_F, double *Bsq_over_F,
                               int nIndx, int nIndSqx,
                               int *nnIndxLUSq,
                               int Trace_N,
                               double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU,
                               int n, int m,
                               double nu, int covModel, double *bk, double nuUnifb,
                               double a_phi, double b_phi,
                               double phimax, double phimin);


void MFA_sigmasq_grad_term1_rephi(int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                                  int BatchSize, int *nBatchLU, int batch_index,
                                  int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                                  double *theta, int tauSqIndx,
                                  double *Bsq_over_F, int *nnIndxLUSq, int *nnIndxwhich,
                                  int *final_result_vec, int *nBatchLU_temp, int tempsize,
                                  double *gradient_sigmasq_temp);

void MFA_sigmasq_grad_rephi(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,
                            int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                            int BatchSize, int *nBatchLU, int batch_index,
                            int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                            double *theta, int tauSqIndx, int zetaSqIndx,
                            double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                            int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                            int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                            int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

double asq_over_expJ(double *sigma_sq, double *w_a, int n);


void MFA_sigmasq_grad_revise1(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,double *w_a,
                             int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                             int BatchSize, int *nBatchLU, int batch_index,
                             int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                             double *theta, int tauSqIndx, int zetaSqIndx,
                             double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                             int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                             int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                             int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

double ad_over_expJ(double *sigma_sq, double *w_a, double *w_d, int n);

void MFA_sigmasq_grad_revise2(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq, double *w_a, double *w_d,
                              int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                              int BatchSize, int *nBatchLU, int batch_index,
                              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                              double *theta, int tauSqIndx, int zetaSqIndx,
                              double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                              int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                              int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                              int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);

void D_vec_gradient(int n, int c, double *D_vec, double *G, double *result, double *work, double *output);

void G_vec_gradient(int n, int c, double *D_vec, double *G, double *result, double *output);

void MFA_sigmasq_grad_reviseq(double *MFA_sigmasq_grad_vec, double *gradient_sigmasq_temp, double *sigma_sq,double *G_vec_second,
                              int n, int *nnIndx, int *nnIndxLU, int *nnIndxCol,
                              int BatchSize, int *nBatchLU, int batch_index,
                              int *numIndxCol, int *nnIndxnnCol, int *cumnumIndxCol,
                              double *theta, int tauSqIndx, int zetaSqIndx,
                              double *F_inv, int *final_result_vec, int *nBatchLU_temp, int tempsize,
                              int *intersect_start_indices, int *intersect_sizes, int* final_intersect_vec,
                              int *complement_first_start_indices, int *complement_first_sizes, int* final_complement_1_vec,
                              int *complement_second_start_indices, int *complement_second_sizes, int* final_complement_2_vec);
void update_vvec(double *v_vec, double *epsilon_vec, double *A_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi);

void sum_vec(double *A, double *X, int np, double *A_p_X);

void update_uvec_ubvec(double *u_vec, double *ub_vec,
                       double *epsilon_vec, double *z_vec,
                       double *A_vi, double *A_beta, double *L_beta,
                       double *S_vi, double *E_vi,
                       int n, int p,
                       int *nnIndxLU_vi, int *nnIndx_vi,
                       int *IndxLU_beta);

void gamma_l_gradient_fun(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                          double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                          int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                          double *u_vec_temp, double *u_vec_temp2, double *gradient,
                          double *ub_vec, double *z_vec, int p, double *L_beta, double *A_beta,
                          double *X, double *gradient_beta, double *XtX, double *tmp_Xtu, double *l_gradient, double *E_vi,
                          int *numIndxCol_beta, int *cumnumIndxCol_beta, int *IndxCol_beta, int *IndxLU_beta);

void a_Abeta_Lbeta_gradient_fun(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                double *u_vec_temp, double *u_vec_temp2,
                                double *ub_vec, double *z_vec, int p, double *L_beta, double *A_beta, double *X, double *E_vi, int *IndxLU_beta,
                                double *A_beta_gradient, double *L_beta_gradient, double *gradient, double *gradient_beta, double *tmp_Xtu, double *XtX,
                                int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi);

void update_ubvec(double *ub_vec, double *u_vec, double *z_vec,
                  double *A_w, double *L_beta,
                  double *E_vi,
                  int n, int p, int num_aw,
                  int *nnIndxLU_vi, int *nnIndx_vi,
                  int *IndxLU_beta);

void gamma_l_gradient_fun2(double *u_vec, double *epsilon_vec, double *gamma_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                           double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                           int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi,
                           double *u_vec_temp, double *u_vec_temp2, double *gradient,
                           double *ub_vec, double *z_vec, int p, double *L_beta, double *A_w,
                           double *X, double *gradient_beta, double *XtX, double *tmp_Xtu, double *l_gradient, double *E_vi,
                           int *numIndxCol_beta, int *cumnumIndxCol_beta, int *IndxCol_beta, int *IndxLU_beta, int num_aw);

void a_Abeta_Lbeta_gradient_fun2(double *u_vec, double *epsilon_vec, double *a_gradient, double *A_vi, double *S_vi, int n, int *nnIndxLU_vi, int *nnIndx_vi,
                                 double *B, double *F, int *nnIndx, int *nnIndxLU, double *theta, int tauSqIndx, int *cumnumIndxCol, int *numIndxCol, int *nnIndxCol, int *nnIndxnnCol,
                                 double *u_vec_temp, double *u_vec_temp2,
                                 double *ub_vec, double *z_vec, int p, double *L_beta, double *A_w, double *X, double *E_vi, int *IndxLU_beta,
                                 double *A_w_gradient, double *L_beta_gradient, double *gradient, double *gradient_beta, double *tmp_Xtu, double *XtX,
                                 int *cumnumIndxCol_vi, int *numIndxCol_vi, int *nnIndxCol_vi, int *nnIndxnnCol_vi, int num_aw);

void updateBFq(double *B_q, double *F_q, double *c_q, double *C_q, double *HinvV_full, int *nnIndx, int *nnIndxLU, int n, int m, int p);

void update_uvec_lr(double *u_vec, double *epsilon_vec, double *B_q, double *F_q, int n, int *nnIndxLU, int *nnIndx);
