#ifndef MATRIX_UTIL
#define MATRIX_UTIL

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

void remove_vector_slice(gsl_vector ** pp_vector, int slice_start, int slice_size);
void remove_matrix_slice(gsl_matrix ** pp_matrix, int slice_start, int slice_size);

void realloc_vector(gsl_vector ** pp_vector, int l);
void realloc_matrix(gsl_matrix ** pp_matrix, int rows, int cols);

void resize_vector(gsl_vector ** pp_vector, int l);
void resize_matrix(gsl_matrix ** pp_matrix, int rows, int cols, int fill_identity);

double psdmatrix_inv_logdet(const gsl_matrix * A, gsl_matrix * invA);
double matrix_inv_det(gsl_matrix * A, gsl_matrix * invA);

void weighted_mean(gsl_matrix * p_points, gsl_vector * p_weights, gsl_vector * p_result);
void weighted_covar(gsl_matrix * p_points, gsl_vector * p_mean, gsl_vector * p_weights, gsl_matrix * p_result);
void weighted_cross_covar(gsl_matrix * p_points1, gsl_vector * p_mean1, gsl_matrix * p_points2, gsl_vector * p_mean2, gsl_vector * p_weights, gsl_matrix * p_result);

void matrix_add_to_diagonal(gsl_matrix * m, gsl_vector * v);

#endif 
