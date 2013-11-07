import numpy as np
from sigvisa.learn.train_param_common import learn_gp, subsample_data

from sigvisa.models.distributions import InvGamma, LogNormal, Gaussian, Uniform
from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa.infer.optimize.optim_utils import construct_optim_params

def sort_events( X, Y):
    n,m = X.shape
    print X.shape
    print Y.shape
    combined = np.hstack([X, Y])
    combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
    X_sorted = np.array(combined_sorted[:, :m], copy=True, dtype=float)
    y_sorted = combined_sorted[:, m:]
    return X_sorted, y_sorted

def preprocess():
    X = np.loadtxt('wiggle_X.txt')
    Y = np.loadtxt('wiggle_Y.txt')

    clean_rows = np.isfinite(Y[:, 0])

    X_clean = X[clean_rows, :]
    Y_clean = Y[clean_rows, :]

    X_sorted, Y_sorted = sort_events(X_clean, Y_clean)

    n, m = Y_sorted.shape

    for i in range(n):
        Y_sorted[i,:] /= np.linalg.norm(Y_sorted[i,:], 2)

    Y_normalized = Y_sorted - np.mean(Y_sorted, axis=0)
    Y_normalized = Y_normalized / np.std(Y_normalized, axis=0)

    np.savetxt('wiggle_X_good.txt', X_sorted)
    np.savetxt('wiggle_Y_good.txt', Y_sorted)



def debug(X, Y, i=26):

    lscale = 10.0
    k = 700

    train_idx = np.arange(0, 2*k, 2)
    test_idx = train_idx + 1

    X_train = np.array(X[train_idx,:], copy=True)
    Y_train = np.array(Y[train_idx,i], copy=True)


    std = np.std(Y_train)
    v = (std**2.0)/2.0
    prior_model = SparseGP(X=np.array(((0.0, 0.0, 0.0),)), y=np.array((0.0,)), sta='ASAR', hyperparams=[v,v,lscale,])
    prior_ll = prior_model.log_p(x=Y[test_idx,i], cond=X[:k,:])
    print "prior ll", prior_ll

    posterior_model = SparseGP(X=X_train, y=Y_train, sta='ASAR', hyperparams=[v,v,lscale,])
    posterior_ll = posterior_model.log_p(x=Y[test_idx,i], cond=X[test_idx,:])

    indep_model = Gaussian(mean=0.0, std=std)
    baseline_lp = np.sum([indep_model.log_p(x=y) for y in Y[test_idx,i]])

    print "lscale %f ll_prior %f ll_posterior %f baseline %f" % (lscale, prior_ll, posterior_ll, baseline_lp)


def learn(X, Y):
    models = []
    #priors = [InvGamma(beta=0.1, alpha=.1), InvGamma(beta=0.1, alpha=.1), LogNormal(mu=2, sigma=3)]
    priors = [LogNormal(mu=2, sigma=3),]
    #optim_params = construct_optim_params("'method': 'bfgscoord', 'normalize': False, 'bfgs_factr': 1e7, 'bfgscoord_iters': 2, 'random_inits': 3")
    optim_params = construct_optim_params("'method': 'bfgscoord', 'normalize': False, 'bfgs_factr': 1e10, 'bfgscoord_iters': 2")

    n,m = Y.shape

    f = open('learned_hparams.txt', 'w')

    for i in range(60):
        """
        for r in (.000001, .001, .01, .1, .5, .9):
            model = SparseGP(X=X, y=Y[:,i], sta='ASAR', hyperparams=[r, 11.85,], compute_ll=True)
            noise_var, signal_var = model.noise_var, model.wfn_params[0]
            model2 = SparseGP(X=np.array(((0.0, 0.0, 0.0),)), y=np.array((0.0,)), sta='ASAR', hyperparams=[noise_var, signal_var, 11.85,])
            model2ll = model2.log_p(x=Y[:,i], cond=X)
            print "r %f ll %f ll2 %f" % (r, model.ll, model2ll)

        ystd = np.std(Y[:,i])
        ymodel=Gaussian(mean=0.0, std=ystd)
        lp = np.sum([ymodel.log_p(x=y) for y in Y[:,i]])
        print "gaussian lp", lp
        continue
        """

        model = learn_gp(X=X, y=Y[:,i], sta='ASAR', kernel_str='lld', params=[.01, .01, 10.0,], priors=[Uniform(0, 10.0), Uniform(0, 10.0), LogNormal(2,2.0),], optim_params = optim_params, k=None, center_mean=True)
        models.append(model)
        model.save_trained_model('param_%d.gp' % i)
        f.write('amp %d params %s\n' % (i, model.hyperparams))
    f.close()

def eval_predict(X, Y, i, train_idx, test_idx):
    print "called at", i
    X_train = np.array(X[train_idx,:], copy=True)
    y_train = np.array(Y[train_idx,i], copy=True)

    base_model = SparseGP(fname='param_%d.gp' % i)

    trained_model = SparseGP(X=X_train, y=y_train, sta='ASAR', hyperparams=base_model.hyperparams, sparse_invert=False, center_mean=True, build_tree=False)
    trained_model.save_trained_model('param_%d_%d.gp' % (i, len(train_idx)))
    print "trained model, evaluating log_p"

    X_test = np.array(X[test_idx, :], copy=True)
    y_test = np.array(Y[test_idx, i], copy=True)

    baseline_mean = np.mean(y_train)
    baseline_std = np.std(y_train)
    baseline_model = Gaussian(mean=baseline_mean, std=baseline_std)

    #trained_model = SparseGP(X=X_train, y=y_train, sta='ASAR', hyperparams=[.001, 10.0])

    y_lp = [trained_model.log_p(x=y, cond=np.reshape(x, (1, -1))) for (x,y) in zip(X_test, y_test)]
    #y_lp = trained_model.log_p(x=y_test, cond=X_test)
    baseline_lp = [baseline_model.log_p(x=y) for y in y_test]

    #print y_pred
    #print y_test
    print "evaluated log_p, predicting..."
    y_pred = [trained_model.predict(cond=np.reshape(x, (1, -1))) for x in X_test]
    pred_mad = np.mean(np.abs(y_test-y_pred))
    baseline_mad = np.mean(np.abs(y_test-baseline_mean))

    result_str =  "param %d params %s baseline_lp %.4f pred_lp %.4f baseline_mad %.4f pred_mad %.4f" % (i, trained_model.hyperparams, np.sum(baseline_lp), np.sum(y_lp), baseline_mad, pred_mad)
    print result_str
    f = open('wiggle_results.txt', 'a')
    f.write(result_str + "\n")
    f.close()

X = np.loadtxt('wiggle_X_good.txt')
Y = np.loadtxt('wiggle_Y_good.txt')

#debug(X, Y)


k = 1200
smallX = np.array(X[:k,:], copy=True)
smallY = np.array(Y[:k,:], copy=True)
learn(smallX,smallY)


#preprocess()


np.random.seed(0)
n_train = 7000
n_test = 2000
p = np.random.permutation(n_train+n_test)
train_idx = p[:n_train]
test_idx = p[n_train:]
for i in range(60):
    eval_predict(X, Y, i, train_idx, test_idx)
