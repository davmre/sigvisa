#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric::ublas;

#include "statespace.hpp"

class PyCSSSM {
public:

  PyCSSSM(const pyublas::numpy_vector<int> & start_idxs, const pyublas::numpy_vector<int> & end_idxs,
	  const pyublas::numpy_vector<int> & identities, const pyublas::numpy_matrix<double> & basis_prototypes,
	  const pyublas::numpy_vector<double> & coef_means, const pyublas::numpy_vector<double> & coef_vars,
	  const double obs_noise, const double bias) {

    /* Warning: "Indexed access to numpy_vector is much slower than iterator access."
     * For speed purposes, may be worth copying the numpy vectors into "standard" ublas vectors,
     * unless I want to go through and make sure we're using iterators everywhere possible. */

    sidx.resize(start_idxs.size());
    sidx.assign(start_idxs);

    eidx.resize(end_idxs.size());
    eidx.assign(end_idxs);

    ids.resize(identities.size());
    ids.assign(identities);

    bp.resize(basis_prototypes.size1(), basis_prototypes.size2());
    bp.assign(basis_prototypes);

    cm.resize(coef_means.size());
    cm.assign(coef_means);

    cv.resize(coef_vars.size());
    cv.assign(coef_vars);

    this->ssm = new CompactSupportSSM(sidx, eidx, ids, bp, cm, cv, obs_noise, bias);
  };

  ~PyCSSSM() {
    delete this->ssm;
  };

  double run_filter(pyublas::numpy_vector<double> z) {
    return filter_likelihood(*(this->ssm), z);
  };

private:
  CompactSupportSSM * ssm;

  vector<int> sidx;
  vector<int> eidx;
  vector<int> ids;
  matrix<double> bp;
  vector<double> cm;
  vector<double> cv;

};


class PyARSSM {
public:

  PyARSSM(const pyublas::numpy_vector<double> & params, double error_var,
	  const double obs_noise, const double bias) {

    p.resize(params.size());
    p.assign(params);

    this->ssm = new ARSSM(p, error_var, obs_noise, bias);
  };

  ~PyARSSM() {
    delete this->ssm;
  };

  double run_filter(pyublas::numpy_vector<double> z) {
    return filter_likelihood(*(this->ssm), z);
  };

private:
  ARSSM * ssm;

  vector<double> p;

};

class PyTSSM {
public:

  PyTSSM(const pyublas::numpy_vector<double> & params, double error_var,
	 const double obs_noise, const double bias) {

    p.resize(params.size());
    p.assign(params);

    this->ssm = new TransientCombinedSSM(p, error_var, obs_noise, bias);
  };

  ~PyARSSM() {
    delete this->ssm;
  };

  double run_filter(pyublas::numpy_vector<double> z) {
    return filter_likelihood(*(this->ssm), z);
  };

private:
  TransientCombinedSSM * ssm;

  vector<double> p;

  std::vector<StateSpaceModel *> ssms;
  vector<int> start_idxs;
  vector<int> & end_idxs;
  std::vector<vector<double> * > scales;

};



namespace bp = boost::python;
BOOST_PYTHON_MODULE(ssms_c) {
  bp::class_<PyCSSSM>("CompactSupportSSM", bp::init< \
	  pyublas::numpy_vector<int> const &,  pyublas::numpy_vector<int> const &,
	  pyublas::numpy_vector<int> const &,  pyublas::numpy_matrix<double> const &,
	  pyublas::numpy_vector<double> const & ,  pyublas::numpy_vector<double> const &,
	  double const , double const>())
    .def("run_filter", &PyCSSSM::run_filter);

  bp::class_<PyARSSM>("ARSSM", bp::init< \
		      pyublas::numpy_vector<double> const &,  double const,
		      double const , double const>())
    .def("run_filter", &PyARSSM::run_filter);

}
