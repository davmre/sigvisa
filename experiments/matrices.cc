#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

#include <random>
#include <stdio.h>

void fill_u(matrix<double> & U, vector<double> & d) {
  U.clear();
  d.clear();

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);



  for (unsigned i=0; i < U.size1(); ++i) {
    U(i,i) = 1.0;
    for (unsigned j=i+1; j < U.size2(); ++j) {
      U(i,j) = distribution(generator);
    }
  }

  for (unsigned i=0; i < d.size(); ++i) {
    d(i) = distribution(generator);
  }

}

void udu(matrix<double> &M, matrix<double> &U, vector<double> &d, unsigned int state_size) {
  // this method stolen from pykalman.sqrt.bierman by Daniel Duckworth (BSD license)
  /*Construct the UDU' decomposition of a positive, semidefinite matrix M

    Parameters
    ----------
    M : [n, n] array
        Matrix to factorize

    Returns
    -------
    UDU : UDU_decomposition of size n
        UDU' representation of M
  */

  unsigned int n = state_size;

  // make M upper triangular: not sure if this is necessary?
  for (unsigned i=0; i < n; ++i) {
    for (unsigned j=0; j < i; ++j) {
      M(i,j) = 0;
    }
  }

  d.clear();
  U.clear();
  for (unsigned i=0; i < n; ++i) {
    U(i,i) = 1;
  }

  for (unsigned j=n; j >= 2; --j) {
    d(j - 1) = M(j - 1, j - 1);
    double alpha = 0.0;
    double beta = 0.0;
    if (d(j - 1) > 0) {
      alpha = 1.0 / d(j - 1);
    } else {
      if (fabs(d(j-1) > 1e-5) ) {
	printf("WARNING: nonpositive d[%d] %f in udu decomp\n", j-1, d(j-1));
	exit(-1);
      }
      d(j-1) = 0.0;
      alpha = 0.0;
    }
    for (unsigned k=1; k < j; ++k) {
      beta = M(k - 1, j - 1);
      U(k - 1, j - 1) = alpha * beta;

      // M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
      for (unsigned kk=0; kk < k; ++kk) {
	M(kk, k - 1) = M(kk, k - 1) - beta * U(kk, j - 1);
      }
    }
  }
  d(0) = M(0, 0);

  if (d(0) < 0) {
    printf("ERROR: udu decomposition on non-posdef matrix with M(0,0)=%f\n", M(0,0));
    exit(-1);
  }
}

void udu_ptr(matrix<double> &M_mat, matrix<double> &U_mat, vector<double> &d_vec, unsigned int state_size) {

  unsigned int n = M_mat.size1();
  double *M = &(M_mat(0,0));
  double *U = &(U_mat(0,0));
  double *d = &(d_vec(0));

  // this method stolen from pykalman.sqrt.bierman by Daniel Duckworth (BSD license)
  /*Construct the UDU' decomposition of a positive, semidefinite matrix M

    Parameters
    ----------
    M : [n, n] array
        Matrix to factorize

    Returns
    -------
    UDU : UDU_decomposition of size n
        UDU' representation of M
  */



  // make M upper triangular: not sure if this is necessary?
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned idx=i*n; idx < i*n+i; ++idx) {
      M[idx] = 0;
    }
  }

  // initialize U as the identity
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned idx=i*n; idx < i*n+state_size; ++idx) {
      U[idx] = 0;
    }
  }
  for (unsigned i=0; i < state_size; ++i) {
    U[i*n+i] = 1;
  }

  for (unsigned j=state_size-1; j >= 1; --j) {
    d[j] = M[j*n+j];
    double alpha = 0.0;
    double beta = 0.0;
    if (d[j] > 0) {
      alpha = 1.0 / d[j];
    } else {
      if (fabs(d[j] > 1e-5) ) {
	printf("WARNING: nonpositive d[%d] %f in udu decomp\n", j, d[j]);
	exit(-1);
      }
      d[j] = 0.0;
      alpha = 0.0;
    }
    for (unsigned k=0; k < j; ++k) {
      beta = M[k*n+ j ];
      U[k*n+ j] = alpha * beta;
      for (unsigned kk=0; kk <= k; ++kk) {
	M[kk*n+ k] = M[kk*n+k] - beta * U[kk*n+j];
      }
    }
  }
  d[0] = M[0];

  if (d[0] < 0) {
    printf("ERROR: udu decomposition on non-posdef matrix with M(0,0)=%f\n", M[0]);
    exit(-1);
  }
}


void compute_explicit_cov(matrix<double> &U,
			  vector<double> &d,
			  matrix<double> &mtmp,
			  matrix<double> &P) {

  unsigned int n = d.size();

  // construct the cov matrix
  for (unsigned i=0; i < n; ++i) {
    subrange(mtmp, 0, n, i, i+1) = subrange(U, 0, n, i, i+1);
    subrange(mtmp, 0, n, i, i+1) *= d(i);
  }
  noalias(subrange(P, 0, n, 0, n))		\
    = prod(subrange(mtmp, 0, n, 0, n),
	   trans(subrange(U, 0, n, 0, n)));
}

void cov_by_hand(matrix<double> &U,
		 vector<double> &d,
		 matrix<double> &mtmp,
		 matrix<double> &P) {
  unsigned int n = d.size();

  for (unsigned i=0; i < n; ++i) {
    for (unsigned j=0; j < n; ++j) {
      double accum = 0;
      for (unsigned k=0; k < n; ++k) {
	accum += U(i, k)* U(j,k) * d(k);
      }
      P(i,j) = accum;
    }
  }
}


#include <cblas.h>
void cov_atlas(matrix<double> &U,
	       vector<double> &d,
	       matrix<double> &mtmp,
	       matrix<double> &P,
	       int state_size,
	       int prev_state_size) {
  unsigned int n = d.size();
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned j=0; j < prev_state_size; ++j) {
      mtmp(i,j) = U(i,j) * d(j);
    }
  }

  cblas_dgemm(CblasRowMajor,
	      CblasNoTrans,
	      CblasTrans,
	      state_size,  prev_state_size, state_size, 1.0,
	      &(mtmp(0,0)), mtmp.size2(),
	      &(U(0,0)), U.size2(),
	      0.0, &(P(0,0)), P.size2());


  /*
Problem with the triangular case:
    After transition, U is not triangular in general. In fact it's not square in general...
   */
}

void write_mat_col(const char *fname, const matrix<double> & m) {

  FILE *f = fopen(fname, "w");
  if (f == NULL)
    {
      printf("Error opening file!\n");
      exit(1);
    }

  for(unsigned i=0; i < m.size1(); ++i) {
    for(unsigned j=0; j < m.size2(); ++j) {
      fprintf(f, "%.12f ", m(i,j));
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");

  fclose(f);
}

bool matrix_eq(matrix<double> &m1, matrix<double> &m2, double tol) {
  for(unsigned i=0; i < m1.size1(); ++i) {
    for(unsigned j=0; j < m1.size2(); ++j) {
      if (fabs(m1(i,j) - m2(i,j)) > tol) {
	printf("unequal\n");
	return false;
      }
    }
  }
  printf("equal\n");
  return true;
}

int main() {

  int n = 150;
  int iters = 100;

  matrix<double> U(n, n);
  matrix<double> P(n, n);
  matrix<double> tmp(n, n);
  vector<double> d(n);
  fill_u(U, d);

  printf("filled\n");

  matrix<double> U_out(n, n);
  vector<double> d_out(n);


  matrix<double> P_true(n, n);
  matrix<double> P_hand(n, n);
  matrix<double> P_atlas(n, n);
  //compute_explicit_cov(U, d, tmp, P_true);
  //cov_by_hand(U, d, tmp, P_hand);
  //cov_atlas(U, d, tmp, P_atlas, n, n);
  //matrix_eq(P_true, P_atlas, 1e-8);

  /*
  write_mat_col("P_true.txt", P_true);
  write_mat_col("P_hand.txt", P_hand);
  write_mat_col("P_atlas.txt", P_atlas);
  */
  cov_atlas(U, d, tmp, P, n, n);
  P_true = P;
  for (unsigned i=0; i < iters; ++i) {
    udu_ptr(P, U_out, d_out, n);
    //P = P_true;
    //udu_ptr(&(P(0,0)), &(U_out(0,0)), &(d_out(0)),
    //	    n, n);

    //printf("udu\n");
    P = P_true;
  }

  bool eq = matrix_eq(U_out, U, 1e-8);


  return 0;

}
