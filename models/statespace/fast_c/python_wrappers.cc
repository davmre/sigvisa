#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/list.hpp>

#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric::ublas;

#include "statespace.hpp"

class PyCSSSM  {
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

  pyublas::numpy_vector<double> py_mean_obs(int n) {
    vector<double> result(n);
    mean_obs(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_obs_var(int n) {
    vector<double> result(n);
    obs_var(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_prior_sample(int n) {
    vector<double> result(n);
    prior_sample(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  int py_max_dimension() {
    return ssm->max_dimension;
  };


  void set_coef_prior(const pyublas::numpy_vector<double> & cmeans,
		      const pyublas::numpy_vector<double> & cvars) {
    cm.assign(cmeans);
    cv.assign(cvars);
  }

  boost::python::tuple get_coef_prior() {
    pyublas::numpy_vector<double> m(cm.size());
    pyublas::numpy_vector<double> v(cv.size());
    m.assign(cm);
    v.assign(cv);
    return boost::python::make_tuple(m,v);
  }

  StateSpaceModel *ssm;
  vector<double> cm;
  vector<double> cv;

private:

  vector<int> sidx;
  vector<int> eidx;
  vector<int> ids;
  matrix<double> bp;

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

  pyublas::numpy_vector<double> py_mean_obs(int n) {
    vector<double> result(n);
    mean_obs(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_obs_var(int n) {
    vector<double> result(n);
    obs_var(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_prior_sample(int n) {
    vector<double> result(n);
    prior_sample(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };


  StateSpaceModel *ssm;
private:

  vector<double> p;

};

class PyTSSM  {
public:

  PyTSSM(const boost::python::list & components, double obs_noise) {

    /* TODO: figure out how to unpack components */
    /* also remember potential memory manegement issues with passed-in SSMs */

    // this->ssm = new TransientCombinedSSM(obs_noise);

    int n_components = boost::python::len(components);
    this->start_idxs.resize(n_components);
    this->end_idxs.resize(n_components);

    /* loop over the Python list of components, to build up the
     * C++ structures needed to initialize the TSSM */
    for (int i=0; i < n_components; ++i) {
      const boost::python::tuple & t = boost::python::extract<boost::python::tuple>(components[i]);

      /* If SSM is None, push a null pointer onto the list
       * of SSMs, otherwise, push an SSM pointer. */
      if (t[0]==boost::python::api::object()) {
	this->ssms.push_back(NULL);
	this->pyssms.push_back(NULL);
      } else {

	boost::python::extract<PyARSSM> get_ar(t[0]);
	boost::python::extract<PyCSSSM> get_cs(t[0]);

	StateSpaceModel *ssm;
	if (get_ar.check()) {
	  ssm = get_ar().ssm;
	} else if (get_cs.check()) {
	  ssm = get_cs().ssm;
	} else {
	  PyErr_SetString(PyExc_RuntimeError, "scale array is not long enough to cover life of SSM component");
	}

	// increase the Python refcount on the SSM object, so that our
	// pointer to its internal C++ SSM will remain valid as long
	// as this TSSM is active (even if the SSM otherwise goes out
	// of scope in the Python code).
	PyObject * pyssm = boost::python::object(t[0]).ptr();
	Py_INCREF(pyssm);
	this->pyssms.push_back(pyssm);

	this->ssms.push_back(ssm);
      }

      /* the start_idx and npts integer components are easy to handle */
      int start_idx = boost::python::extract<int>(t[1]);
      unsigned int npts = boost::python::extract<int>(t[2]);
      this->start_idxs[i] = start_idx;
      this->end_idxs[i] = start_idx+npts;

      /* I hate C++ types enough that we're just going to represent the scaling vector as a (double *),
       * or NULL if we were apssed a Python None. */
      if (t[3]==boost::python::api::object()) {
	this->scales.push_back(NULL);

      } else {

	const pyublas::numpy_vector<double> & scale_vec = boost::python::extract<pyublas::numpy_vector<double> >(t[3]);

	if (scale_vec.size() < npts) {
	  PyErr_SetString(PyExc_RuntimeError, "scale array is not long enough to cover life of SSM component");
	}

	const double * scale_ptr = &(scale_vec(0));
	this->scales.push_back(scale_ptr);

	// Keep a reference to the Numpy array object to prevent it
	// getting garbage collected.
	PyObject * pyvec = boost::python::object(t[3]).ptr();
	Py_INCREF(pyvec);
	this->vec_refs.push_back(pyvec);
      }
    }

    this->ssm = new TransientCombinedSSM(this->ssms, this->start_idxs, this->end_idxs, this->scales, obs_noise);
  };

  ~PyTSSM() {

    // release all the Python references we've been holding
    // (SSM objects and scale vectors)
    std::vector<PyObject *>::iterator it;
    for (it = pyssms.begin(); it < pyssms.end(); ++it) {
      if (*it != NULL) Py_DECREF(*it);
    }
    for (it = vec_refs.begin(); it < vec_refs.end(); ++it) {
      Py_DECREF(*it);
    }
  };

  double run_filter(pyublas::numpy_vector<double> z) {
    return filter_likelihood(*(this->ssm), z);
  };

  pyublas::numpy_vector<double> py_mean_obs(int n) {
    vector<double> result(n);
    mean_obs(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_prior_sample(int n) {
    vector<double> result(n);
    prior_sample(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };

  pyublas::numpy_vector<double> py_obs_var(int n) {
    vector<double> result(n);
    obs_var(*(this->ssm), result);
    return pyublas::numpy_vector<double>(result);
  };


  boost::python::list component_means(const pyublas::numpy_vector<double> z) {
    std::vector<vector<double> > means(this->ssms.size());
    for (unsigned i=0; i < means.size(); ++i) {
      int st = this->start_idxs[i];
      int et = this->end_idxs[i];
      int npts = et-st;
      means[i].resize(npts);
      means[i].clear();
    }
    tssm_component_means(*(this->ssm), z, means);

    boost::python::list l;
    for(unsigned i=0; i < means.size(); ++i) {
      pyublas::numpy_vector<double> v(means[i]);
      l.append(v);
    }
    return l;
  }

  boost::python::list marginals(const pyublas::numpy_vector<double> z) {
    std::vector<vector<double> > cmeans;
    std::vector<vector<double> > cvars;
    all_filtered_cssm_coef_marginals(*(this->ssm), z, cmeans, cvars);

    boost::python::list l;
    for(unsigned i=0; i < cmeans.size(); ++i) {
      pyublas::numpy_vector<double> m(cmeans[i]);
      pyublas::numpy_vector<double> v(cvars[i]);
      l.append( boost::python::make_tuple(m,v) );
    }
    return l;

  }

  /*

  int get_n_coefs(int i) {
    if (!this->ssms[i]) {
      printf("ERROR: trying to set means for a NULL ssm.\n");
      exit(-1);
    }
    if (!this->ssms[i]->is_cssm) {
      printf("ERROR: trying to set means for a non-CSSM ssm.\n");
      exit(-1);
    }
    CompactSupportSSM *ssm = (CompactSupportSSM *) (this->ssms[i]);
    return ssm->n_basis;
  }
  */

  boost::python::object get_component(int i) {
    if (this->pyssms[i]) {
      boost::python::handle<> h(this->pyssms[i]);
      return boost::python::object(h);
    } else {
      return boost::python::object();
    }
  }


  TransientCombinedSSM *ssm;
private:
  vector<double> p;

  // keep references to the Python objects whose internal states we depend on,
  // to prevent the Python garbage collector from killing them.
  std::vector<PyObject *> pyssms;
  std::vector<PyObject *> vec_refs;

  std::vector<StateSpaceModel *> ssms;
  vector<int> start_idxs;
  vector<int> end_idxs;
  std::vector<const double * > scales;

};

namespace bp = boost::python;
BOOST_PYTHON_MODULE(ssms_c) {


  bp::class_<PyCSSSM>("CompactSupportSSM", bp::init< \
	  pyublas::numpy_vector<int> const &,  pyublas::numpy_vector<int> const &,
	  pyublas::numpy_vector<int> const &,  pyublas::numpy_matrix<double> const &,
	  pyublas::numpy_vector<double> const & ,  pyublas::numpy_vector<double> const &,
	  double const , double const>())
    .def("run_filter", &PyCSSSM::run_filter)
    .def("mean_obs", &PyCSSSM::py_mean_obs)
    .def("prior_sample", &PyCSSSM::py_prior_sample)
    .def("obs_var", &PyCSSSM::py_obs_var)
    .def("set_coef_prior", &PyCSSSM::set_coef_prior)
    .def("get_coef_prior", &PyCSSSM::get_coef_prior)
    .def("max_dimension", &PyCSSSM::py_max_dimension)
     ;


  bp::class_<PyARSSM>("ARSSM", bp::init< \
		      pyublas::numpy_vector<double> const &,  double const,
		      double const , double const>())
    .def("run_filter", &PyARSSM::run_filter)
    .def("mean_obs", &PyARSSM::py_mean_obs)
    .def("prior_sample", &PyARSSM::py_prior_sample)
    .def("obs_var", &PyARSSM::py_obs_var);

  bp::class_<PyTSSM>("TransientCombinedSSM", bp::init<boost::python::list const &,
		     double >())
    .def("run_filter", &PyTSSM::run_filter)
    .def("mean_obs", &PyTSSM::py_mean_obs)
    .def("prior_sample", &PyTSSM::py_prior_sample)
    .def("obs_var", &PyTSSM::py_obs_var)
    .def("component_means", &PyTSSM::component_means)
    .def("all_filtered_cssm_coef_marginals", &PyTSSM::marginals)
    //.def("get_n_coefs", &PyTSSM::get_n_coefs)
    .def("get_component", &PyTSSM::get_component);
}
