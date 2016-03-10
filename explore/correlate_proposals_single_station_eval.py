import numpy as np
from sigvisa import Sigvisa
import os

from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event

from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.graph.region import Region
from sigvisa.signals.common import Waveform
from sigvisa.models.distributions import Gaussian,TruncatedGaussian


from sigvisa.models.ttime import tt_predict
from sigvisa.infer.template_xc import fastxc

from sigvisa.utils.geog import dist_km
from sigvisa.infer.template_xc import fastxc

from sigvisa.explore.correlate_validation_events import load_arrivals

from sigvisa.infer.correlations.historical_signal_library import get_historical_signals
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage
from sigvisa.infer.correlations.weighted_event_posterior import build_ttr_model_array, hack_ev_time_posterior_with_weight

import cPickle as pickle



def load_alignments(sta, phase):
    fname = "alignments_%s_%s.pkl" % (sta, phase)
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    return results

def load_proposal_weights(sta, phase):
    fname = "weights_%s_%s.pkl" % (sta, phase)
    try:
        with open(fname, 'rb') as f:
            weights = pickle.load(f)
    except IOError as e:
        raise e

        #print "weights not found for %s %s, loading alignments..." % (sta, phase)
        #alignments = load_alignments(sta, phase)
        #print "computing weights..."
        #weights = compute_proposal_weights(alignments)
        #with open(fname, 'wb') as f:
        #    pickle.dump(weights, f)
        #print "done"
    return weights
        

def compute_proposal_weights(alignments):
    
    evids = [k for k in alignments.keys() if k not in ("xs", "gpsignals")]

    weights = {}

    for evid in evids:
        rdict = alignments[evid]

        training_xcs = []
        training_iid_weights = []
        training_ar_weights = []
        
        for (xcs, iids, ars) in zip(rdict["xcs"], rdict["iid_advantages"], rdict["ar_advantages"]):
            C = np.max(iids)
            posterior = np.exp(iids-C)
            Z = np.sum(posterior)
            posterior /= Z
            logZ = np.log(Z) + C
            training_iid_weights.append(logZ)

            C = np.max(ars)
            posterior = np.exp(ars-C)
            Z = np.sum(posterior)
            posterior /= Z
            logZ = np.log(Z) + C
            training_ar_weights.append(logZ)

            training_xcs.append(np.max(xcs))
           
        weights[evid] = rdict["ev"], (np.array(training_xcs), np.array(training_iid_weights), np.array(training_ar_weights))

    weights["xs"] = alignments["xs"]

    return weights

def proposal_lp(ev, xs, weights, gaussian_width_deg, depth_width, uniform_prob):
    x = [ev.lon, ev.lat, ev.depth]

    londist = Gaussian(ev.lon, gaussian_width_deg)
    latdist = Gaussian(ev.lat, gaussian_width_deg)
    depthdist = TruncatedGaussian(ev.depth, depth_width, a=0)

    lp = -np.inf
    for (xx, weight) in zip(xs, weights):
        lw = np.log(weight)
        lw += londist.log_p(xx[0,0])
        lw += latdist.log_p(xx[0,1])
        lw += depthdist.log_p(xx[0,2])
        lp = np.logaddexp(lp, lw)

    proposal_logprob = np.log((1-uniform_prob)) + lp

    # hardcode the western US region
    uniform_ev_logprob = np.log(uniform_prob * 1.0/(26.0) * 1.0/(16.0) * 1.0/(700))
    
    return np.logaddexp(proposal_logprob, uniform_ev_logprob)

def is_proposable(ev, xs, weights, weight_threshold=0.1, distance_threshold_km=30.0):
    for (xx, weight) in zip(xs, weights):
        if weight > weight_threshold:
            if dist_km((xx[0,0], xx[0,1]), (ev.lon, ev.lat) ) < distance_threshold_km:
                return True
    return False


def reweight_uniform_top(weights, n=20):
    cutoff = sorted(weights)[-n]

    new_weights = np.ones(weights.shape)
    new_weights[weights < cutoff] = 0.0

    assert (np.sum(new_weights) == n)
    new_weights /= n
    return new_weights

def reweight_temper_exp(weights, temper):

    new_weights = weights * temper
    new_weights -= np.max(new_weights)
    new_weights = np.exp(new_weights)
    new_weights /= np.sum(new_weights)
    return new_weights

def reweight_uniform_all(weights, uniform_prob=0.05):
    total_weight = np.sum(weights)
    added_uniform_weight = total_weight / (1./uniform_prob - 1.)

    new_weights = weights + added_uniform_weight / len(weights)
    new_weights /= np.sum(weights)
    return new_weights


def normalize(weights):
    new_weights = new_weights / np.sum(new_weights)
    return new_weights

def evaluate_proposals(weights, weight_extractor, gaussian_width_deg, 
                       depth_width, uniform_prob, 
                       weight_threshold, distance_threshold_km):

    total_lp = 0.0
    good_proposal_count = 0

    training_xs = weights["xs"]

    for evid in weights.keys():
        if evid =="xs":
            continue

        ev, w = weights[evid]
        training_ev_weights = weight_extractor(w)

        lp = proposal_lp(ev, training_xs, training_ev_weights, gaussian_width_deg, depth_width, uniform_prob)
        total_lp += lp

        if is_proposable(ev, training_xs, training_ev_weights, weight_threshold, distance_threshold_km):
            good_proposal_count += 1
    
    return total_lp, good_proposal_count


def loop_proposals(weights):
    results = {}


    """
    results["xc_uniform"] = evaluate_proposals(weights, 
                                               lambda w : reweight_uniform_all(reweight_uniform_top(w[0])),
                                               gaussian_width_deg=0.02, uniform_prob=0.05,
                                               depth_width=10.0, distance_threshold_km=20.0,
                                               weight_threshold=0.04)
    """
    results["iid_uniform"] = evaluate_proposals(weights, 
                                                lambda w : reweight_uniform_all(reweight_uniform_top(w[1], n=9)),
                                                gaussian_width_deg=0.02, uniform_prob=0.05,
                                                depth_width=10.0, distance_threshold_km=20.0,
                                                weight_threshold=0.1)


    results["iid_normalized"] = evaluate_proposals(weights, 
                                                   lambda w : reweight_uniform_all(normalize(w[1])),
                                                   gaussian_width_deg=0.02, uniform_prob=0.05,
                                                   depth_width=10.0, distance_threshold_km=20.0,
                                                   weight_threshold=0.1)


    """
    results["ar_uniform"] = evaluate_proposals(weights, 
                                               lambda w : reweight_uniform_all(reweight_uniform_top(w[2])),
                                               gaussian_width_deg=0.02, uniform_prob=0.05,
                                               depth_width=10.0, distance_threshold_km=20.0,
                                               weight_threshold=0.04)
    """


    for temper in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0):
        k = "tempered_%.3f" % temper

        """

        results["xc_"+k] = evaluate_proposals(weights, 
                                              lambda w : reweight_uniform_all(reweight_temper_exp(w[0], temper=temper)),
                                              gaussian_width_deg=0.02, uniform_prob=0.05,
                                              depth_width=10.0, distance_threshold_km=20.0,
                                              weight_threshold=0.04)
        """

        results["iid_"+k] = evaluate_proposals(weights, 
                                               lambda w : reweight_uniform_all(reweight_temper_exp(w[1], temper=temper)),
                                               gaussian_width_deg=0.02, uniform_prob=0.05,
                                               depth_width=10.0, distance_threshold_km=20.0,
                                               weight_threshold=0.1)

        """
        results["ar_"+k] = evaluate_proposals(weights, 
                                              lambda w : reweight_uniform_all(reweight_temper_exp(w[2], temper=temper)),
                                              gaussian_width_deg=0.02, uniform_prob=0.05,
                                              depth_width=10.0, distance_threshold_km=20.0,
                                              weight_threshold=0.04)
        """

    return results

def format_performance(sta, phase, performance):
    s = ""
    for k, (total_lp, good_proposals) in sorted(performance.items()):
        s += "%s %s %s: lp %.2f, proposals %d\n" % (sta, phase, k, total_lp, good_proposals)
    return s

def tune_likelihood(sta, phase):
    weights = load_proposal_weights(sta, phase)
    results = []
    for gaussian_width_deg in (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        for depth_width in (5.0, ):
            lp, good_proposals = evaluate_proposals(weights, 
                                                    lambda w : reweight_uniform_all(reweight_uniform_top(w[0])),
                                                    gaussian_width_deg=gaussian_width_deg, uniform_prob=0.05,
                                                    depth_width=depth_width, distance_threshold_km=20.0,
                                                    weight_threshold=0.04)
            results.append((gaussian_width_deg, depth_width, lp))
    return results


if __name__ == "__main__":   

    sta="ANMO"
    phase="Pg"


    stas = "ANMO,ELK,IL31,KDAK,NEW,PFO,TX01,ULM,YBH,YKR8".split(",")
    phases = ["P", "S", "Lg", "Pg"]

    
    for sta in stas:
        for phase in phases:
            try:

                #results = tune_likelihood(sta, phase)
                #for (width, depth, lp) in results:
                #    print sta, phase, "width %.2f depth %.1f lp %.1f" % (width, depth, lp)

                weights = load_proposal_weights(sta, phase)
                performance = loop_proposals(weights)

                s = format_performance(sta, phase, performance)
                print s

                #fname = "proposals_iid_%s_%s.txt" % (sta, phase)
                #with open(fname, "w") as f:
                #    f.write(s)
                #    print "wrote", fname
            except Exception as e:
                print sta, phase, e
                continue
