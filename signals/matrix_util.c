#include "matrix_util.h"

#include "../sigvisa.h"
#include "math.h"

/* Removes a slice from a vector, starting at the given index. */
void remove_vector_slice(gsl_vector ** pp_vector, int slice_start, int slice_size) {

 if (*pp_vector == NULL) {
    return;
  }

  int s = slice_start;
  int l = slice_size;
  int n = (*pp_vector)->size;

  if (l == n) {
    gsl_vector_free(*pp_vector);
    *pp_vector = NULL;
    return;
  }

  gsl_vector * new_vector = gsl_vector_alloc(n - l);
  for(int i=0; i < s; ++i) {
    gsl_vector_set(new_vector, i,  gsl_vector_get(*pp_vector, i));
  }
  for(int i=s+l; i < n; ++i) {
    gsl_vector_set(new_vector, i-l,  gsl_vector_get(*pp_vector, i));
  }
  gsl_vector_free(*pp_vector);
  *pp_vector = new_vector;
}

/* Removes a slice from a square matrix, starting at the given row/column index. */
void remove_matrix_slice(gsl_matrix ** pp_matrix, int slice_start, int slice_size) {

  if (*pp_matrix == NULL) {
    return;
  }

  int s = slice_start;
  int l = slice_size;
  int n = (*pp_matrix)->size1;

  if (l == n) {
    gsl_matrix_free(*pp_matrix);
    *pp_matrix = NULL;
    return;
  }

  gsl_matrix * new_matrix = gsl_matrix_alloc(n-l, n-l);

  gsl_matrix_view topleft_block_old = gsl_matrix_submatrix(*pp_matrix, 0, 0, s, s);
  gsl_matrix_view topleft_block_new = gsl_matrix_submatrix(*pp_matrix, 0, 0, s, s);
  gsl_matrix_memcpy(&topleft_block_new.matrix, &topleft_block_old.matrix);

  gsl_matrix_view topright_block_old = gsl_matrix_submatrix(*pp_matrix, 0, s+l, s, n-s-l);
  gsl_matrix_view topright_block_new = gsl_matrix_submatrix(*pp_matrix, 0, s, s, n-s-l);
  gsl_matrix_memcpy(&topright_block_new.matrix, &topright_block_old.matrix);

  gsl_matrix_view bottomleft_block_old = gsl_matrix_submatrix(*pp_matrix, s+l, 0, n-s-l, s);
  gsl_matrix_view bottomleft_block_new = gsl_matrix_submatrix(*pp_matrix, s, 0, n-s-l, s);
  gsl_matrix_memcpy(&bottomleft_block_new.matrix, &bottomleft_block_old.matrix);

  gsl_matrix_view bottomright_block_old = gsl_matrix_submatrix(*pp_matrix, s+l, s+l, n-s-l, n-s-l);
  gsl_matrix_view bottomright_block_new = gsl_matrix_submatrix(*pp_matrix, s, s, n-s-l, n-s-l);
  gsl_matrix_memcpy(&bottomright_block_new.matrix, &bottomright_block_old.matrix);

  gsl_matrix_free(*pp_matrix);
  *pp_matrix = new_matrix;
}

/* Resizes the given vector to have the given length. Excess entries
   are discarded (if the vector is shrinking) while new entries are
   filled with zeros (if it is growing). */
void resize_vector(gsl_vector ** pp_vector, int l) {
  
  assert(l >= (*pp_vector)->size);

  if (*pp_vector == NULL) {
    *pp_vector = gsl_vector_alloc(l);
    gsl_vector_set_zero(*pp_vector);
  } else {
    gsl_vector * new_vector = gsl_vector_alloc(l);
    gsl_vector_set_zero(new_vector);
    for (int i=0; i < MIN(l, (*pp_vector)->size); ++i) {
	gsl_vector_set(new_vector, i, gsl_vector_get(*pp_vector, i));
    }
    gsl_vector_free(*pp_vector);
    *pp_vector = new_vector;
  }
}

void realloc_vector(gsl_vector ** pp_vector, int l) {
  if (*pp_vector == NULL) {
    *pp_vector = gsl_vector_alloc(l);
  } else {
    gsl_vector_free(*pp_vector);
    *pp_vector = gsl_vector_alloc(l);
  }
}

void realloc_matrix(gsl_matrix ** pp_matrix, int rows, int cols) {
  if (*pp_matrix == NULL) {
    *pp_matrix = gsl_matrix_alloc(rows,cols);
  } else {
    gsl_matrix_free(*pp_matrix);
    *pp_matrix = gsl_matrix_alloc(rows,cols);
  }
}

/* Resize the given matrix to have the given numbers of rows and
   columns. If either number is smaller than current, excess entries
   are dropped; if larger, then new entries are filled in with zeros
   or with the identity matrix, according to whether fill_identity is
   set. */
void resize_matrix(gsl_matrix ** pp_matrix, int rows, int cols, int fill_identity) {
  
  if (*pp_matrix == NULL) {
    *pp_matrix = gsl_matrix_alloc(rows,cols);
    if (fill_identity) {
      gsl_matrix_set_identity(*pp_matrix);
    } else {
      gsl_matrix_set_zero(*pp_matrix);
    }
  } else {

    gsl_matrix * new_matrix = gsl_matrix_alloc(rows, cols);
    if (fill_identity) {
      gsl_matrix_set_identity(new_matrix);
    } else {
      gsl_matrix_set_zero(new_matrix);
    }

    for (int i=0; i < MIN(rows, (*pp_matrix)->size1); ++i) {
      for(int j=0; j < MIN(cols, (*pp_matrix)->size2); ++j) {
	gsl_matrix_set(new_matrix, i, j, gsl_matrix_get(*pp_matrix, i, j));
      }
    }
    gsl_matrix_free(*pp_matrix);
    *pp_matrix = new_matrix;
  }
}


double matrix_inv_det(gsl_matrix * A, gsl_matrix * invA) {
  gsl_matrix * LU = gsl_matrix_alloc(A->size1,A->size2);
  gsl_permutation *p = gsl_permutation_alloc(A->size1);
  int signum;
  gsl_matrix_memcpy(LU, A);
  gsl_linalg_LU_decomp(LU, p, &signum);
  gsl_linalg_LU_invert(LU, p, invA);

  double det = gsl_linalg_LU_det (LU, signum);

  gsl_matrix_free(LU);
  gsl_permutation_free(p);
  return det;
}


double psdmatrix_inv_logdet(const gsl_matrix * A, gsl_matrix * invA) {

  gsl_matrix_memcpy(invA, A);
  gsl_linalg_cholesky_decomp(invA);
  gsl_vector_view diag = gsl_matrix_diagonal(invA);
  
  double det = 0;
  for(int i=0; i < diag.vector.size; ++i) {
    det += 2* log(gsl_vector_get(&diag.vector, i));
  }
  gsl_linalg_cholesky_invert(invA);
  
  return det;
}

void weighted_mean(gsl_matrix * p_points, gsl_vector * p_weights, gsl_vector * p_result) {
  // points are columns of p_points
  int n = p_points->size2;

  gsl_vector_set_zero(p_result);
  for (int i=0; i < n; ++i) {
    gsl_vector_view pt = gsl_matrix_column(p_points, i);
    gsl_blas_daxpy (gsl_vector_get(p_weights, i), &pt.vector, p_result);
  }
  
}

void weighted_covar(gsl_matrix * p_points, gsl_vector * p_mean, gsl_vector * p_weights, gsl_matrix * p_result) {
  // points are columns of p_points
  int d = p_points->size1;
  int n = p_points->size2;

  gsl_matrix_set_zero(p_result);
  gsl_vector * r = gsl_vector_alloc(d);
  for (int i=0; i < n; ++i) {
    gsl_vector_view pt = gsl_matrix_column(p_points, i);
    gsl_vector_memcpy(r, &pt.vector);
    gsl_vector_sub(r, p_mean);
    gsl_blas_dsyr(CblasLower, gsl_vector_get(p_weights, i), r, p_result);
  }
  gsl_vector_free(r);
  
}

void weighted_cross_covar(gsl_matrix * p_points1, gsl_vector * p_mean1, gsl_matrix * p_points2, gsl_vector * p_mean2, gsl_vector * p_weights, gsl_matrix * p_result) {
  // points are columns of p_points
  int d1 = p_points1->size1;
  int d2 = p_points2->size1;
  int n = p_points1->size2;

  gsl_matrix_set_zero(p_result);
  gsl_vector * r1 = gsl_vector_alloc(d1);
  gsl_vector * r2 = gsl_vector_alloc(d2);
  for (int i=0; i < n; ++i) {
    gsl_vector_view pt1 = gsl_matrix_column(p_points1, i);
    gsl_vector_memcpy(r1, &pt1.vector);
    gsl_vector_sub(r1, p_mean1);

    gsl_vector_view pt2 = gsl_matrix_column(p_points2, i);
    gsl_vector_memcpy(r2, &pt2.vector);
    gsl_vector_sub(r2, p_mean2);

    gsl_blas_dger (gsl_vector_get(p_weights, i), r1, r2, p_result);
  }
  gsl_vector_free(r1);
  gsl_vector_free(r2);

}

void matrix_add_to_diagonal(gsl_matrix * m, gsl_vector * v) {
  gsl_vector_view diag = gsl_matrix_diagonal(m);
  gsl_vector_add(&diag.vector, v);
}
