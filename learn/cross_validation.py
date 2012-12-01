import numpy as np

import sys, traceback, pdb
from optparse import OptionParser

from learn.train_coda_models import learn_model, load_model, get_model_fname, get_training_data, analyze_model_fname
from sigvisa import *
from learn.SpatialGP import distfns, SpatialGP, start_params
from database.signal_data import *

class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def cv_generator(n, k=3):
    data = np.random.permutation(n)
    fold_size = n/k
    folds = [data[i*fold_size:(i+1)*fold_size] for i in range(k)]
    folds[k-1] = data[(k-1)*fold_size:]
    for i in range(k):
        train = np.array(())
        for j in range(k):
            if j != i:
                train = np.concatenate([train, folds[j]])
        test = folds[i]
        yield ([int (t) for t in train], [int(t) for t in test])

def save_cv_folds(X, y, evids, cv_dir, folds=3):
    np.savetxt(os.path.join(cv_dir, "X.txt"), X)
    np.savetxt(os.path.join(cv_dir, "y.txt"), y)
    np.savetxt(os.path.join(cv_dir, "evids.txt"), evids)

    for i, (train, test) in enumerate(cv_generator(len(y), k=folds)):
        np.savetxt(os.path.join(cv_dir, "fold_%02d_train.txt" % i), train)
        np.savetxt(os.path.join(cv_dir, "fold_%02d_test.txt" % i), test)


def train_cv_models(cv_dir, model_type):
    X = np.loadtxt(os.path.join(cv_dir, "X.txt"))
    y = np.loadtxt(os.path.join(cv_dir, "y.txt"))
    evids = np.loadtxt(os.path.join(cv_dir, "evids.txt"))

    d = analyze_model_fname(os.path.join(cv_dir, "abcdefg.model"))

    i = 0
    while os.path.exists(os.path.join(cv_dir, "fold_%02d_train.txt" % i)):

        train = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_train.txt" % i))]
        test = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_test.txt" % i))]

        trainX = X[train, :]
        trainy = y[train]
        trainevids = evids[train]

        evidhash = hashlib.sha1(trainevids).hexdigest()[0:8]
        fname = ".".join(["fold_%02d" % i, evidhash, model_type])
        fullpath = os.path.join(cv_dir, fname)
        if os.path.exists(fullpath):
            print "model %s already exists, skipping..." % fullpath
            i += 1
            continue

        logfile_name = os.path.join(cv_dir, "fold_%02d_train.log" % i)
        logfile = open(logfile_name, 'w')
        print "training model, writing log to", logfile_name
        with RedirectStdStreams(stdout=logfile, stderr=logfile):
            model = learn_model(trainX, trainy, model_type, target=d['target'])
        logfile.close()
        model.save_trained_model(fullpath)

        i += 1

def cv_eval_models(cv_dir, model_type):
    X = np.loadtxt(os.path.join(cv_dir, "X.txt"))
    y = np.loadtxt(os.path.join(cv_dir, "y.txt"))
    evids = np.loadtxt(os.path.join(cv_dir, "evids.txt"))

    residuals = []
    residual_evids = []

    i = 0
    while os.path.exists(os.path.join(cv_dir, "fold_%02d_train.txt" % i)):
        train = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_train.txt" % i))]
        test = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_test.txt" % i))]

        testX = X[test, :]
        testy = y[test]
        trainevids = evids[train]
        testevids = evids[test]
        evidhash = hashlib.sha1(trainevids).hexdigest()[0:8]

        fname = ".".join(["fold_%02d" % i, evidhash, model_type])
        model = load_model(os.path.join(cv_dir, fname), model_type)
        residuals += [model.predict(testX[i]) - testy[i] for i in range(len(testevids))]
        residual_evids.extend(testevids)
        i += 1

    mean_abs_error = np.mean(np.abs(residuals))
    median_abs_error = np.median(np.abs(residuals))

    f_results = open(os.path.join(cv_dir, "%s_results.txt" % model_type), 'w')
    f_results.write('mean_abs_error %f\n' % mean_abs_error)
    f_results.write('median_abs_error %f\n' % median_abs_error)
    f_results.close()

    return mean_abs_error, median_abs_error

def main():
    parser = OptionParser()

    s = Sigvisa()

    parser.add_option("-s", "--site", dest="site", default=None, type="str", help="site for which to train models")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("--run_iter", dest="run_iter", default="latest", type="str", help="run iteration (latest)")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phase", dest="phase", default=",".join(s.phases), type="str", help="phase for which to train models)")
    parser.add_option("-t", "--target", dest="target", default="decay", type="str", help="target parameter name (decay)")
    parser.add_option("-m", "--model_types", dest="model_types", default="gp_dad_log,constant,linear_distance", type="str", help="types of model to train (gp_dad_log,constant,linear_distance)")

    (options, args) = parser.parse_args()

    site = options.site
    chan = options.chan
    phase = options.phase
    band = options.band
    target = options.target
    model_types = options.model_types.split(',')

    run_name = options.run_name

    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(s.cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)

    X, y, evids = get_training_data(run_name, run_iter, site, chan, band, [phase,], target)

    model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_types[0], evids, model_name="paired_exp", prefix="eval")

    cv_dir = os.path.dirname(model_fname)

#    print "generating cross-validation folds in dir", cv_dir
#    save_cv_folds(X,y,evids, cv_dir)

    for model_type in model_types:
        print "training cross-validation models for", model_type
        train_cv_models(cv_dir, model_type)

    print "DONE TRAINING, NOW EVALUATING!"
    for model_type in model_types:
        mean_error, median_error = cv_eval_models(cv_dir, model_type)
        print "%s abs error: mean %.4f median %.4f" % (model_type, mean_error, median_error)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
