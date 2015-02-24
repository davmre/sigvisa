#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric::ublas;

#include <google/dense_hash_map>
#include <boost/functional/hash.hpp>
using google::dense_hash_map;

//typedef matrix_range< matrix<double> > mr;
//typedef vector_range< vector <double> > vr;

#define PI 3.14159265

class StateSpaceModel {

public:
  virtual int apply_transition_matrix(const vector<double> &x, int k, vector<double> &result) = 0;
  virtual void transition_bias(int k, vector<double> & result) = 0;
  virtual void transition_noise_diag(int k, vector<double> & result) = 0;
  virtual double apply_observation_matrix(const vector<double> &x, int k) = 0;
  virtual void apply_observation_matrix(const matrix<double> &X, int k, vector<double> &result, int n) = 0;
  virtual double observation_bias(int k) = 0;
  virtual double observation_noise(int k) = 0;
  virtual int prior_mean(vector<double> &result) = 0;
  virtual int prior_vars(vector<double> &result) = 0;
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

  int apply_transition_matrix (const vector<double> &x, int k, vector<double> &result);
  void transition_bias (int k, vector<double> & result) ;
  void transition_noise_diag(int k, vector<double> & result);
  double apply_observation_matrix(const vector<double> &x, int k);
  void apply_observation_matrix(const matrix<double> &X, int k, vector<double> &result, int n);
  double observation_bias(int k);
  double observation_noise(int k);
  int prior_mean(vector<double> &result);
  int prior_vars(vector<double> &result);
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


class FilterState {
public:

  int state_size;

  bool wasnan;
  bool at_fixed_point;
  double alpha;
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

double kalman_observe_sqrt(StateSpaceModel &ssm, FilterState &cache, int k, double zk);
void kalman_predict_sqrt(StateSpaceModel &ssm, FilterState &cache, int k);
double filter_likelihood(StateSpaceModel &ssm, const vector<double> &z);
