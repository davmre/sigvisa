#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "SignalModelUtil.h"


ArrivalWaveform_t * append_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_active=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->next_active == NULL) {
    // if at the end of the list, add to the end
    p_head->next_active = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_active = append_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * remove_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  if (p_head == NULL) {
    p_new_head = NULL;
  } else if (p_head == p_arr) {
    p_new_head = p_arr->next_active;
  } else if (p_head->next_active == NULL) {
    p_new_head = p_head;
  } else {
    // otherwise, delete recursively
    p_head->next_active = remove_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_st(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_start=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->start_time > p_arr->start_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_start = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_start == NULL) {
    // if at the end of the list, add to the end
    p_head->next_start = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_start = insert_st(p_head->next_start, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_et(ArrivalWaveform_t * p_head, 
				 ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_end=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->end_time > p_arr->end_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_end = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_end == NULL) {
    // if at the end of the list, add to the end
    p_head->next_end = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_end = insert_et(p_head->next_end, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}


int AR_add_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n) {

  int arridx = 0;

  if (*pp_means == NULL || *pp_covars == NULL) {

    *pp_means = gsl_vector_alloc(n);
    gsl_vector_set_zero(*pp_means);

    *pp_covars = gsl_matrix_alloc(n,n);
    gsl_matrix_set_identity(*pp_covars);

  } else {
    
    int l = (*pp_means)->size;
    arridx = l/n;
    gsl_vector * new_means = gsl_vector_alloc(l + n);
    gsl_vector_set_zero(new_means);
    for(int i=0; i < l; ++i) {
      gsl_vector_set(new_means, i,  gsl_vector_get(*pp_means, i));
    }
    gsl_vector_free(*pp_means);
    *pp_means = new_means;

    gsl_matrix * new_covars = gsl_matrix_alloc(l + n, l + n);
    gsl_matrix_set_identity(new_covars);
    for (int i=0; i < l; ++i) {
      for(int j=0; j < l; ++j) {
	gsl_matrix_set(new_covars, i, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    gsl_matrix_free(*pp_covars);
    *pp_covars = new_covars;
  }
  return arridx;
}

void AR_remove_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n, int arridx) {
  if (*pp_means == NULL || *pp_covars == NULL) {
    return;
  } 

  int l = (*pp_means)->size;
  if (l == n) {
    gsl_vector_free(*pp_means);
    gsl_matrix_free(*pp_covars);
    *pp_means = NULL;
    *pp_covars = NULL;
  } else {
    gsl_vector * new_means = gsl_vector_alloc(l - n);
    for(int i=0; i < arridx*n; ++i) {
      gsl_vector_set(new_means, i,  gsl_vector_get(*pp_means, i));
    }
    for(int i=(arridx+1)*n; i < l; ++i) {
      gsl_vector_set(new_means, i-n,  gsl_vector_get(*pp_means, i));
    }
    gsl_vector_free(*pp_means);
    *pp_means = new_means;

    gsl_matrix * new_covars = gsl_matrix_alloc(l - n, l - n);

    for (int i=0; i < arridx*n; ++i) {
      for(int j=0; j < arridx*n; ++j) {
	gsl_matrix_set(new_covars, i, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything up by n rows
    for (int i=(arridx+1)*n; i < l; ++i) {
      for(int j=0; j < (arridx+1)*n; ++j) {
	gsl_matrix_set(new_covars, i-n, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything in by n columns
    for (int i=0; i < (arridx+1)*n; ++i) {
      for(int j=(arridx+1)*n; j < l; ++j) {
	gsl_matrix_set(new_covars, i, j-n, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything in by n columns
    for (int i=i=(arridx+1)*n; i < l; ++i) {
      for(int j=(arridx+1)*n; j < l; ++j) {
	gsl_matrix_set(new_covars, i-n, j-n, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    gsl_matrix_free(*pp_covars);
    *pp_covars = new_covars;

  }
}

void AR_transitions(int n, int k, double * ar_coeffs, 
		    int n_arrs, 
		    gsl_matrix ** pp_transition) {
  if (*pp_transition != NULL) {
    gsl_matrix_free(*pp_transition);
  } 
  

  if (n_arrs == 0) {
    *pp_transition = NULL;
    return;
  } 

  *pp_transition = gsl_matrix_alloc(n*n_arrs, n*n_arrs);

  gsl_matrix_set_zero(*pp_transition);

  for (int arr=0; arr < n_arrs; ++arr) {
    gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n, ar_coeffs[0]);
    for (int i=1; i < n; ++i) {
      gsl_matrix_set(*pp_transition, arr*n+i-1, arr*n+i, 1);
      gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n+i, ar_coeffs[i]);
    }
   
  }

}

void AR_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs) {
  if (*pp_obs != NULL) {
    gsl_matrix_free(*pp_obs);
  }
  
  if (n_arrs == 0) {
    *pp_obs = NULL;
    return;
  } 

  *pp_obs = gsl_matrix_alloc(k, n*n_arrs);
  gsl_matrix_set_zero(*pp_obs);


  for (int arr=0; arr < n_arrs; ++arr) {
    ArrivalWaveform_t * p_a = active_arrivals;
   int curr_chan=0;
    for (int i=0; i < NUM_CHANS; ++i) {
      if (p_segment->p_channels[i] != NULL) {
	switch (i) {
	case CHAN_BHZ:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	case CHAN_BHN:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	case CHAN_BHE:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	}
	curr_chan++;
      }
    }

    active_arrivals=active_arrivals->next_active;
  }


}

void AR_predict(gsl_vector * p_means, gsl_matrix * p_covars, gsl_matrix * p_transition, double noise_sigma2, int n) {
  int num_arrs = p_means->size/n;
  
  gsl_vector * tmp = gsl_vector_alloc(p_means->size);
  gsl_blas_dgemv (CblasNoTrans, 1, p_transition, p_means, 0, tmp);
  gsl_vector_memcpy(p_means, tmp);
  gsl_vector_free(tmp);

  gsl_matrix * mtmp = gsl_matrix_alloc(p_covars->size1, p_covars->size2);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, p_covars, p_transition, 0, mtmp);  
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, p_transition, mtmp, 0, p_covars);
  
  gsl_matrix_free(mtmp);
  for(int i=0; i < num_arrs; ++i) {
    int idx = (i+1)*n-1;
    gsl_matrix_set(p_covars, idx, idx, gsl_matrix_get(p_covars, idx, idx) + noise_sigma2);
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

void AR_update(gsl_vector * p_means, gsl_matrix * p_covars, 
	       gsl_matrix * p_obs, gsl_vector * obs_perturb, 
	       gsl_matrix * obs_covar, int n, int k, 
	       gsl_vector * y, gsl_matrix * S) {

  gsl_blas_dgemv (CblasNoTrans, 1, p_obs, p_means, 0, y);
  //LogTrace("predicted perturb %lf versus obs %lf",  pred_obs_perturb, obs_perturb_mean);
  gsl_vector_sub(y, obs_perturb);
  gsl_vector_scale(y, -1);


  gsl_matrix * Sinv = gsl_matrix_alloc(k,k);
  gsl_matrix * PHt = gsl_matrix_alloc(p_means->size, k);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, p_covars, p_obs, 0, PHt);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, p_obs, PHt, 0, S);
  gsl_matrix_add (S, obs_covar);
  matrix_inv_det(S, Sinv);

  gsl_matrix * K = gsl_matrix_alloc(p_means->size, k);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, PHt, Sinv, 0, K);

  gsl_vector *mean_update = gsl_vector_alloc(p_means->size);
  gsl_blas_dgemv(CblasNoTrans, 1, K, y, 0, mean_update);
  gsl_vector_add(p_means, mean_update);

  gsl_matrix * I = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_matrix_set_identity (I);

  gsl_matrix * new_covar = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, -1, K, p_obs, 1, I);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, I, p_covars, 0, new_covar);
  gsl_matrix_memcpy(p_covars, new_covar);

  gsl_matrix_free(Sinv);
  gsl_matrix_free(PHt);
  gsl_matrix_free(K);
  gsl_vector_free(mean_update);
  gsl_matrix_free(I);
  gsl_matrix_free(new_covar);
}
