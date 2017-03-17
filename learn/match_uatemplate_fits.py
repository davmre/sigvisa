import numpy as np
import os
import uuid
import cPickle as pickle
import itertools
from collections import defaultdict

from sigvisa import Sigvisa
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.utils.fileutils import mkdir_p
from sklearn import linear_model
from sigvisa.source.event import get_event

from optparse import OptionParser

from sigvisa.models.logistic_regression import LogisticRegressionModel


def data_for_fitid(fitid, phase_set):

    """
    return an array X of input features, and a dict ys mapping phases
    to boolean values indicating whether they were "detected" by the 
    fit uatemplates (associating in the context of the given phase set).
    """

    def match_phases(phase_atimes, uatemplate_atimes, phase_set):
        phases_to_match = [phase for phase in phase_atimes.keys() if phase in phase_set]

        if len(phases_to_match) == 0:
            return {}

        possible_associations = defaultdict(list)
        for phase in phases_to_match:
            possible_associations[phase].append((None, -25.0))
            atime = phase_atimes[phase]
            for uaid, uatime in uatemplate_atimes.items():
                score = -np.abs(uatime - atime)
            possible_associations[phase].append((uaid, score))
            

        vals = [possible_associations[k] for k in phases_to_match]
        best_score = -np.inf
        best_assoc = None
        for assoc in itertools.product(*vals):
            uaids, scores = zip(*assoc)

            nontrivial_uaids = [t for t in uaids if t is not None]
            if len(set(nontrivial_uaids)) != len(nontrivial_uaids):
                # duplicate, assigned to two phases
                continue

            total_score = np.sum(scores)
            if total_score > best_score:
                best_score = total_score
                best_assoc = uaids

        phases_matched = dict([(phase, best_assoc[i] is not None) for i, phase in enumerate(phases_to_match)])
        return phases_matched

    s = Sigvisa()
    sql_query = "select evid, dist from sigvisa_coda_fit where fitid=%d" % fitid
    evid, dist = s.sql(sql_query)[0]
    ev = get_event(evid)

    sql_query = "select phase, arrival_time from sigvisa_coda_fit_phase where fitid=%d" % fitid
    phases = dict(s.sql(sql_query))

    sql_query = "select uaid, arrival_time from sigvisa_uatemplate_tuning where fitid=%d" % fitid
    uatemplates = dict(s.sql(sql_query))
    
    X = (ev.mb, np.exp(ev.mb), ev.depth, np.sqrt(ev.depth), dist, np.log(dist))

    if len(uatemplates) == 0:
        ys = {}
    elif len(uatemplates) == 1 and uatemplates.values()[0] is None:
        ys = dict([(phase, False) for phase in phase_set])
    else:
        ys = match_phases(phases, uatemplates, phase_set)


    return X, ys

def get_training_data(fitids, sta, phase_set):
    
    Xs = {}
    ys = {}
    for phase in phase_set:
        Xs[phase]= []
        ys[phase] = []
    
    for fitid in fitids:
        X, fit_ys = data_for_fitid(fitid, phase_set)
        for phase in fit_ys.keys():
            Xs[phase].append(X)
            ys[phase].append(fit_ys[phase])

    results = {}
    for phase in phase_set:
        results[phase] = np.asarray(Xs[phase]), np.asarray(ys[phase])

    return results
    

def get_fitids(runid):
    s = Sigvisa()
    sql_query = "select fitid, sta from sigvisa_coda_fit where runid=%d" % (runid)

    fitids = defaultdict(list)
    for fitid, sta in s.sql(sql_query):
        fitids[sta].append(fitid)

    return fitids

def train_model(Xs, ys, sta, phase):

    x_mean = np.mean(Xs, axis=0)
    Xs_centered = Xs -x_mean

    x_scale = np.std(Xs_centered, axis=0)
    Xs_scaled = Xs_centered / (x_scale + 1e-6)

    if np.sum(ys) == 0:
        # if phase is never detected, no way to fit a
        # reasonable regression
        weights = np.zeros((Xs.shape[1],))
        intercept = -10.0
    else:
        logreg = linear_model.LogisticRegression()
        logreg.fit(Xs_scaled, ys)
        weights = logreg.coef_.flatten()
        intercept = float(logreg.intercept_)

    model = LogisticRegressionModel(weights, intercept, 
                                    x_mean=x_mean, x_scale=x_scale,
                                    sta=sta, phase=phase)


    """
    x0 = Xs[0, :]
    p1 = model.predict_prob(x0)
    xs0 = Xs_scaled[0, :]
    p2 = logreg.predict_proba(x0)
    """

    return model

def save_model(sta, phase, runid, model, phase_set):

    s = Sigvisa()
    mkdir_p(os.path.join(s.homedir, "parameters", "detection_models"))
    model_fname =  os.path.join("parameters", "detection_models", "%s_%s_%d_%s.pkl" % (sta, phase, runid, str(uuid.uuid4())[:8]))
    with open(os.path.join(s.homedir, model_fname), "wb") as f:
        pickle.dump(model, f)

    phase_context_str = ','.join(sorted(phase_set))
    sql_query = "insert into sigvisa_hough_detection_model (fitting_runid, sta, phase, phase_context, model_fname) values (%d, '%s', '%s', :phase_set, :model_fname)" % (runid, sta, phase)
    modelid = execute_and_return_id(s.dbconn, sql_query, "modelid", phase_set=phase_context_str, model_fname=model_fname)
    return modelid

def main():

    parser = OptionParser()

    parser.add_option("--runid", dest="runid", default=None, type="int", help="fitid for which to fit uatemplates")
    parser.add_option("--phase_context", dest="phase_context", default=None, type="str", help="")
    parser.add_option("--dummy", dest="dummy", default=False, action="store_true", help="")

    (options, args) = parser.parse_args()

    phase_context = set(options.phase_context.split(","))
    runid = options.runid

    fitids_by_sta = get_fitids(runid)

    for sta, fitids in fitids_by_sta.items():
        training_data_by_phase = get_training_data(fitids, sta, phase_context)
        print "loaded training data at", sta
        for phase in training_data_by_phase.keys():
            Xs, ys = training_data_by_phase[phase]
            if len(ys) <= 5:
                print "not enough fits (%d) to train %s, %s" % (len(ys), sta, phase)
                continue
            print "training model for %s on %d fits" % (phase, len(ys))
            model = train_model(Xs, ys, sta, phase)
            if not options.dummy:
                modelid = save_model(sta, phase, runid, model, phase_context)
                print "saved modelid %d for %s, %s in context %s" % (modelid, sta, phase, phase_context)



if __name__ == "__main__":
    main()
