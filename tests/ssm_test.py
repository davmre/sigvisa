from sigvisa.ssms_c import CompactSupportSSM
import pyublas
import numpy as np
import time

from sigvisa.models.wiggles.wavelets import construct_implicit_basis_simple, construct_basis_simple, construct_implicit_basis_C

def cssm(N=64, run_test=True):

    #starray, etarray, idarray, m, N = construct_implicit_basis_C(5.0, "db4_2.0_3_30")



    start_times, end_times, identities, prototypes, N = construct_implicit_basis_simple(N, "db4", "zpd")
    starray = np.array(start_times, dtype=np.int32)
    etarray = np.array(end_times, dtype=np.int32)
    idarray = np.array(identities, dtype=np.int32)

    n1 = len(prototypes)
    n2 = np.max([len(l) for l in prototypes])
    print n1, n2

    m = np.matrix(np.ones((n1, n2), dtype=np.float64)*np.nan)
    for i, p in enumerate(prototypes):
        m[i,:len(p)] = p

    n = len(starray)
    cmeans = np.zeros((n,), dtype=np.float64)
    cvars = np.ones((n,), dtype=np.float64)

    np.random.seed(0)
    cvars = np.abs(np.random.randn(n))
    z = np.array(np.random.randn(110))

    z[1:4] = np.nan
    z[100:110] = np.nan


    import sigvisa.models.statespace.compact_support as cs
    t0 = time.time()
    ic = cs.ImplicitCompactSupportSSM(start_times, end_times, identities, prototypes, cmeans, cvars, 0.0, 0.01)
    t1 = time.time()
    if run_test:
        ll2 = ic.run_filter(z)
        t2 = time.time()

    """
        basis = construct_basis_simple(N, "db4", "zpd")
        pyc = cs.CompactSupportSSM(basis, cmeans, cvars, obs_noise=0.01)
        ll3 = pyc.run_filter(z)
        assert( np.abs(ll2-ll3) < 1e-8 )
        """

    pyublas.set_trace(True)


    t3 = time.time()
    c = CompactSupportSSM(starray, etarray,
                          idarray, m, cmeans, cvars, 0.01, 0.0)
    t4 = time.time()
    if run_test:
        ll = c.run_filter(z)
        t5 = time.time()

        print "python", t1-t0, t2-t1, "ll", ll2
        print "C", t4-t3, t5-t4, "ll", ll
        return
        ov_py = ic.obs_var(N)
        ov_c = c.obs_var(N)
        assert( (np.abs(ov_py - ov_c) < 1e-8).all())

        print ov_c

    return ic, c

def arssm(N=1024, run_test=True):

    params = np.array((0.6, 0.2, 0.05))

    np.random.seed(0)
    z = np.random.randn(N)

    from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
    em = ErrorModel(mean=0.0, std=np.sqrt(0.1))
    arm = ARModel(params, em)


    from sigvisa.ssms_c import ARSSM
    ar_c = ARSSM(params, 0.1, 0.0, 0.0)
    t0 = time.time()
    if run_test:
        ll_c = ar_c.run_filter(z)
        t1 = time.time()

    import sigvisa.models.statespace.ar as ar
    ar_py = ar.ARSSM(params, 0.1, mean=0.0)
    t2 = time.time()

    if run_test:
        ll_py = ar_py.run_filter(z)
        t3 = time.time()

        ll_arm = arm.log_p(z)
        t4 = time.time()

        print "python", t3-t2, "ll", ll_py
        print "armodel", t4-t3, "ll", ll_arm
        print "C", t1-t0, "ll", ll_c

        py_var = ar_py.obs_var(20)
        c_var = ar_c.obs_var(20)
        assert( (np.abs(py_var-c_var) < 1e-8).all())

    return ar_py, ar_c

def tssm(run_test=True):

    from sigvisa.ssms_c import TransientCombinedSSM

    import sigvisa.models.statespace.transient as transient
    from sigvisa.models.statespace.dummy import DummySSM

    cs_py, cs_c = cssm(run_test=False)
    ar_py, ar_c = arssm(run_test=False)

    N = 64 #cs_py.n_steps

    scale1 = np.sin(np.linspace(-10, 10, 112 ))
    #scale1 = np.ones((112,))

    components_c = [(ar_c, 0, 256, None ), (cs_c, 2, 112, scale1), (None, 10, 32, scale1)]
    tssm_c = TransientCombinedSSM(components_c, 0.01)

    components_py = [(ar_py, 0, 256, None ), (cs_py, 2, 112, scale1), (DummySSM(bias=1.0), 10, 32, scale1)]
    tssm_py = transient.TransientCombinedSSM(components_py, 0.01)

    if run_test:
        np.random.seed(0)
        z = np.random.randn(256)

        z[0:5] = np.nan
        z[100:130] = np.nan
        z[253] = np.nan

        import gc

        t0 = time.time()
        ll1 = tssm_py.run_filter(z)
        t1 = time.time()
        ll2 = tssm_c.run_filter(z)
        t2 = time.time()



        #csssm = tssm_c.get_component(1)
        #cm, cv = csssm.get_coef_prior()
        #cm2 = np.random.randn(len(cm))
        #csssm.set_coef_prior(cm2, cv)
        #ll3 = tssm_c.run_filter(z)
        ll3 = ll2

        print "python tssm time %f ll %f" % (t1-t0, ll1)
        print "c tssm time %f ll %f %f" % (t2-t1, ll2, ll3)



        t2 = time.time()
        means_c = tssm_c.component_means(z)
        t3 = time.time()
        means_py = tssm_py.component_means(z)
        t4 = time.time()

        gc.collect()

        for mc, mp in zip(means_c, means_py):
            assert( (np.abs(mc-mp) < 1e-8).all() )

        print "python components time %f " % (t4-t3)
        print "c components time %f " % (t3-t2)

        t5 = time.time()
        marginals_c = tssm_c.all_filtered_cssm_coef_marginals(z)
        t6 = time.time()
        marginals_py = tssm_py.all_filtered_cssm_coef_marginals(z)
        t7=  time.time()
        gc.collect()
        for k in marginals_py.keys():
            assert( (np.abs(marginals_py[k][0] - marginals_c[k][0]) < 1e-8).all())
            assert( (np.abs(marginals_py[k][1] - marginals_c[k][1]) < 1e-8).all())

        print "python coef marginals time %f " % (t7-t6)
        print "c coef marginals time %f " % (t6-t5)



        mean_c = tssm_c.mean_obs(len(z))
        mean_py = tssm_py.mean_obs(len(z))
        gc.collect()
        assert( (np.abs(mean_c-mean_py) < 1e-8).all())
        print "prior means match!"

        var_c = tssm_c.obs_var(len(z))
        var_py = tssm_py.obs_var(len(z))
        gc.collect()
        assert( (np.abs(var_c-var_py) < 1e-8).all())
        print "prior vars match!"


        sample_c = tssm_c.prior_sample(len(z))
        sample_py = tssm_py.prior_sample(len(z))
        np.savetxt("sc.txt", sample_c)
        np.savetxt("sp.txt", sample_py)

    return tssm_c

def tssm_memory_test():
    tssm_c = tssm(run_test=False)
    import gc
    gc.collect()
    np.random.seed(0)
    z = np.random.randn(256)
    ll2 = tssm_c.run_filter(z)
    print ll2


def cssm_vs_true_test():
    from sigvisa.models.wiggles.wavelets import construct_wavelet_basis
    import sigvisa.models.statespace.compact_support as cs
    import scipy.stats

    basis = construct_wavelet_basis(2.0, "db2_1.0_99_30", sort=True)

    cssm = cs.CompactSupportSSM(basis, obs_noise=1.0)

    input_N = basis.shape[1]
    tmp_basis = basis[:, :input_N]
    np.random.seed(0)

    z = np.random.randn(input_N)
    ll1 = cssm.run_filter_naive(z)
    ll2 = cssm.run_filter(z)

    m = np.zeros(input_N)
    C = np.dot(tmp_basis.T, tmp_basis) + np.eye(input_N)
    rv = scipy.stats.multivariate_normal(mean=m, cov=C)
    ll3 = rv.logpdf(z)
    print "python sqrt filter", ll2
    print "python naive filter", ll1
    print "full Gaussian", ll3
    assert( np.max(np.abs(ll1-ll2) )< 1e-8 )
    assert( np.max(np.abs(ll1-ll3) )< 1e-8 )


def true_gaussian_filtering(z, m, C):
    n = len(z)
    ells = []
    for i in range(n):
        Kstar = C[i, :i]
        Kinv = np.linalg.inv(C[:i,:i])

        pred_m = np.dot(Kstar, np.dot(Kinv, z[:i]))
        pred_v = C[i,i] - np.dot(Kstar, np.dot(Kinv, Kstar))
        ell = scipy.stats.norm(loc=pred_m, scale=np.sqrt(pred_v)).logpdf(z[i])
        print "step %d pred %.4f alpha %.4f z %.4f y %.4f ell %f" % (i, pred_m, pred_v, z[i], z[i]-pred_m, ell)
        ells.append(ell)
    return np.sum(ells)


if __name__ == "__main__":
    try:
        cssm()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()

        pdb.post_mortem(tb)
