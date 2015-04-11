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

#ifdef DEBUG
#define D(x) x
#else
#define D(x)
#endif

class StateSpaceModel {

public:
  virtual int apply_transition_matrix(const double * x,
					int k, double * result) = 0;
  virtual int apply_transition_matrix( const matrix<double> &X,
					 unsigned int x_row_offset,
					 int k,
					 matrix<double> &result,
					 unsigned int r_row_offset,
					 unsigned int n)  = 0;
  virtual void transition_bias(int k, double * result) = 0;
  virtual void transition_noise_diag(int k, double * result) = 0;
  virtual double apply_observation_matrix(const  double * x, int k) = 0;

  virtual void apply_observation_matrix(const matrix<double> &X, unsigned int row_offset, int k, double * result, double *result_tmp, unsigned int n) = 0;
  virtual double observation_bias(int k) = 0;
  virtual double observation_noise(int k) = 0;
  virtual int prior_mean(double * result) = 0;
  virtual int prior_vars(double * result) = 0;
  virtual bool stationary(int k) = 0;

  virtual ~StateSpaceModel() { return; };

  unsigned int max_dimension;
  bool is_cssm;
};

class FilterState {
public:

  unsigned int state_size;

  bool wasnan;
  bool at_fixed_point;
  double alpha;
  double pred_z;
  matrix<double> obs_U;
  vector<double> obs_d;
  vector<double> gain;
  matrix<double> pred_U;
  vector<double> pred_d;

  matrix<double> tmp_U1;
  matrix<double> tmp_U2;
  matrix<double> P;

  vector<double> f;
  vector<double> v;

  vector<double> xk;

  double eps_stationary;

  FilterState (int max_dimension, double eps_stationary);
  void init_priors(StateSpaceModel &ssm);

};


class CompactSupportSSM : public StateSpaceModel {
public:
  unsigned int n_basis;
  unsigned int n_steps;
  const vector<double> & coef_means;
  const vector<double> & coef_vars;


  CompactSupportSSM(const vector<int> & start_idxs, const vector<int> & end_idxs,
		    const vector<int> & identities, const matrix<double> & basis_prototypes,
		    const vector<double> & coef_means,  const vector<double> & coef_vars,
		    double obs_noise, double bias);

  ~CompactSupportSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  int apply_transition_matrix( const matrix<double> &X,
				 unsigned int x_row_offset,
				 int k,
				 matrix<double> &result,
				 unsigned int r_row_offset,
				 unsigned int n) ;
  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double> &X, unsigned int row_offset, int k, double * result, double *result_tmp, unsigned int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);
  void extract_coefs(const vector<double> &x,
		     const matrix<double> &P,
		     unsigned int state_offset,
		     int k,
		     vector<double> & coef_means,
		     vector<double> & coef_vars);

private:
dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  > active_indices;
  matrix<int> active_basis;

  const vector<int> & start_idxs;
  const vector<int> & end_idxs;
  const vector<int> & identities;
  const matrix <double> & basis_prototypes;

  double obs_noise;
  double bias;


};

class ARSSM : public StateSpaceModel {
public:
  ARSSM(vector<double> & params, double error_var,
	  double obs_noise, double bias);
  ~ARSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  int apply_transition_matrix( const matrix<double> &X,
				 unsigned int x_row_offset,
				 int k,
				 matrix<double> &result,
				 unsigned int r_row_offset,
				 unsigned int n) ;
  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double> &X, unsigned int row_offset, int k, double * result, double *result_tmp, unsigned int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);

  vector<double> & params;
  double error_var;
  double obs_noise;
  double bias;


};

class TransientCombinedSSM : public StateSpaceModel {
public:
TransientCombinedSSM(std::vector<StateSpaceModel *> & ssms, const vector<int> & start_idxs,
			 const vector<int> & end_idxs, const std::vector<const double * > & scales,
			 double obs_noise);
~TransientCombinedSSM();

  int apply_transition_matrix(const double *x, int k, double * result);
  int apply_transition_matrix( const matrix<double> &X,
				 unsigned int x_row_offset,
				 int k,
				 matrix<double> &result,
				 unsigned int r_row_offset,
				 unsigned int n) ;

  void transition_bias (int k, double * result) ;
  void transition_noise_diag(int k, double * result);
  double apply_observation_matrix(const double * x, int k);
  void apply_observation_matrix(const matrix<double> &X, unsigned int row_offset, int k, double * result, double *result_tmp, unsigned int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(double * result);
  int prior_vars(double * result);
  bool stationary(int k);

  void init_coef_priors(std::vector<vector<double> > & cmeans,
			std::vector<vector<double> > & cvars);

  void extract_all_coefs(FilterState &cache, int k,
			 std::vector<vector<double> > & cmeans,
			 std::vector<vector<double> > & cvars);
  void extract_component_means(double *xk, int k,
			       std::vector<vector<double> > & means);
  void extract_component_vars(matrix<double> &P, matrix<double> &P_tmp, int k,
			       std::vector<vector<double> > & vars);

  const unsigned int n_ssms;
private:

  std::vector<StateSpaceModel *> ssms;
  const vector<int> start_idxs;
  const vector<int> end_idxs;
  const std::vector<const double * > scales;

  const double obs_noise;
  unsigned int n_steps;

  int active_ssm_cache1_k;
  int active_ssm_cache1_v;
  int active_ssm_cache2_k;
  int active_ssm_cache2_v;

  vector<unsigned int> ssms_tmp;
  std::vector<int> changepoints;

  matrix<int> active_sets;

  int active_set_idx(int k);

};


double kalman_observe_sqrt(StateSpaceModel &ssm, FilterState &cache, int k, double zk);
void kalman_predict_sqrt(StateSpaceModel &ssm, FilterState &cache, int k);
double filter_likelihood(StateSpaceModel &ssm, const vector<double> &z);
void mean_obs(StateSpaceModel &ssm, vector<double> & result);
void obs_var(StateSpaceModel &ssm, vector<double> & result);
void prior_sample(StateSpaceModel &ssm, vector<double> & result, unsigned long seed);

double tssm_component_means(TransientCombinedSSM &tssm,
			  const vector<double> &z,
			  std::vector<vector<double> > & means);
double tssm_component_vars(TransientCombinedSSM &tssm,
			  const vector<double> &z,
			  std::vector<vector<double> > & vars);
double all_filtered_cssm_coef_marginals(TransientCombinedSSM &ssm,
				      const vector<double> &z,
				      std::vector<vector<double> > & cmeans,
				      std::vector<vector<double> > & cvars);
  void step_obs_likelihoods(StateSpaceModel &ssm, const vector<double> &z,
			    vector<double> & ells,
			    vector<double> & preds,
			    vector<double> & alphas);
