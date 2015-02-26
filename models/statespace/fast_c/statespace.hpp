#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric::ublas;

#include <google/dense_hash_map>
#include <boost/functional/hash.hpp>
using google::dense_hash_map;

//typedef matrix_range< matrix<double> > mr;
//typedef vector_range< vector <double> > vr;

#define PI 3.14159265359

//double gdb_get( matrix<double> & m, int i, int j) {
//  return m(i,j);
//}


class StateSpaceModel {

public:
  virtual int apply_transition_matrix(const double * x,
					int k, double * result) = 0;

  virtual void transition_bias(int k, double * result) = 0;
  virtual void transition_noise_diag(int k, double * result) = 0;
  virtual double apply_observation_matrix(const  double * x, int k) = 0;

  virtual void apply_observation_matrix(const matrix<double,column_major> &X, int row_offset, int k, double * result, double *result_tmp, int n) = 0;
  virtual double observation_bias(int k) = 0;
  virtual double observation_noise(int k) = 0;
  virtual int prior_mean(double * result) = 0;
  virtual int prior_vars(double * result) = 0;
  virtual bool stationary(int k) = 0;

  virtual ~StateSpaceModel() { return; };

  int max_dimension;
};


class CompactSupportSSM : public StateSpaceModel {
public:
  int n_basis;
  int n_steps;



  CompactSupportSSM(const vector<int> & start_idxs, const vector<int> & end_idxs,
		    const vector<int> & identities, const matrix<double> & basis_prototypes,
		    const vector<double> & coef_means,  const vector<double> & coef_vars,
		    double obs_noise, double bias);

  ~CompactSupportSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double,column_major> &X, int row_offset, int k, double * result, double *result_tmp, int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);

private:
dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  > active_indices;
  matrix<int> active_basis;

  const vector<int> & start_idxs;
  const vector<int> & end_idxs;
  const vector<int> & identities;
  const matrix <double> & basis_prototypes;

  const vector<double> & coef_means;
  const vector<double> & coef_vars;
  double obs_noise;
  double bias;


};

class ARSSM : public StateSpaceModel {
public:
  ARSSM(const vector<double> & params, double error_var,
	  double obs_noise, double bias);
  ~ARSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double,column_major> &X, int row_offset, int k, double * result, double *result_tmp, int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);

private:
  const vector<double> & params;
  const double error_var;
  const double obs_noise;
  const double bias;
};

class TransientCombinedSSM : public StateSpaceModel {
public:
TransientCombinedSSM(std::vector<StateSpaceModel *> ssms, const vector<int> start_idxs,
			 const vector<int> & end_idxs, const std::vector<vector<double> * > scales,
			 double obs_noise);
~TransientCombinedSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double,column_major> &X, int row_offset, int k, double * result, double *result_tmp, int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);

private:
  std::vector<StateSpaceModel *> ssms;
  const vector<int> start_idxs;
  const vector<int> end_idxs;
  const std::vector<vector<double> * > scales;

  const double obs_noise;
  const int n_ssms;
  int n_steps;

  int active_ssm_cache1_k;
  int active_ssm_cache1_v;
  int active_ssm_cache2_k;
  int active_ssm_cache2_v;

  vector<int> ssms_tmp;
  std::vector<int> changepoints;

  matrix<int> active_sets;

  int active_set_idx(int k);

};

class FilterState {
public:

  int state_size;

  bool wasnan;
  bool at_fixed_point;
  double alpha;
  matrix<double,column_major> obs_U;
  vector<double> obs_d;
  vector<double> gain;
  matrix<double,column_major> pred_U;
  vector<double> pred_d;

  matrix<double,column_major> tmp_U1;
  matrix<double,column_major> tmp_U2;
  matrix<double> P;

  vector<double> f;
  vector<double> v;

  vector<double> xk;

  double eps_stationary;

  FilterState (int max_dimension, double eps_stationary);
  void init_priors(StateSpaceModel &ssm);

};

double kalman_observe_sqrt(StateSpaceModel &ssm, FilterState &cache, int k, double zk);
void kalman_predict_sqrt(StateSpaceModel &ssm, FilterState &cache, int k);
double filter_likelihood(StateSpaceModel &ssm, const vector<double> &z);
