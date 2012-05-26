#ifndef MATRIX_UTIL
#define MATRIX_UTIL

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

void remove_vector_slice(gsl_vector ** pp_vector, int slice_start, int slice_size);
void remove_matrix_slice(gsl_matrix ** pp_matrix, int slice_start, int slice_size);
void resize_vector(gsl_vector ** pp_vector, int l);
void resize_matrix(gsl_matrix ** pp_matrix, int rows, int cols, int fill_identity);
double matrix_inv_det(gsl_matrix * A, gsl_matrix * invA);

gsl_vector * weighted_mean(gsl_matrix * p_points, gsl_vector * p_weights);
}
gsl_matrix * weighted_covar(gsl_matrix * p_points, gsl_vector * p_mean, gsl_vector * p_weights);
gsl_matrix * weighted_cross_covar(gsl_matrix * p_points1, gsl_vector * p_mean1, gsl_matrix * p_points2, gsl_vector * p_mean2, gsl_vector * p_weights);

void matrix_add_to_diagonal(gsl_matrix * m, gsl_vector * v);

#endif MATRIX_UTIL
