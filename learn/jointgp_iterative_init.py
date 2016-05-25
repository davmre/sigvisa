import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.infer.template_xc import fastxc
from sigvisa.graph.sigvisa_graph import SigvisaGraph, MAX_TRAVEL_TIME
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.graph.region import Region
from sigvisa.source.event import Event
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser

import itertools
from collections import defaultdict
from sigvisa.utils.geog import dist_km
from sigvisa.signals.common import Waveform

from sigvisa.utils.array import index_to_time, time_to_index, time_to_index_offset
import numpy.ma as ma

def score_event_correlations(sg, sta):

    # get the wn and first arriving phase for each event
    eid_atimes = {}
    for wn in sg.station_waves[sta]:
        arrivals = wn.arrivals()
        eids = [eid for (eid, phase) in arrivals if eid > 0]
        for eid in eids:
            sidxs = [(phase, wn.arrival_start_idx(eid, phase)) for (eeid, phase) in arrivals if eeid==eid]
            sorted_phases = sorted(sidxs, key = lambda x : x[1])
            first_arriving_phase = sorted_phases[0][0]
            eid_atimes[eid] = (wn, first_arriving_phase)

    # extract a signal for each event
    eid_signals = {}
    eid_signals_short = {}
    for eid, (wn, phase) in eid_atimes.items():
        eid_signals[eid] = wn.get_arrival_signal(eid, phase, 30.0, pre_s = 5.0)
        eid_signals_short[eid] = wn.get_arrival_signal(eid, phase, 20.0, pre_s = 0.0)
    
    # correlate all pairs of events
    xcpairs = {}
    xcs_by_eid = defaultdict(list)
    xcs_by_eid_full= defaultdict(list)
    for (eid1, eid2) in itertools.combinations(eid_signals.keys(), 2):
        xc1 = np.max(fastxc(eid_signals_short[eid1], eid_signals[eid2]))
        xc2 = np.max(fastxc(eid_signals_short[eid2], eid_signals[eid1]))
        xc = max(xc1, xc2)

        xcpairs[(eid1, eid2)] = xc
        xcs_by_eid[eid1].append(xc)
        xcs_by_eid[eid2].append(xc)

        xcs_by_eid_full[eid1].append((eid2, xc))
        xcs_by_eid_full[eid2].append((eid1, xc))



    # score each event by its best two correlations
    #evscores = dict([(eid, np.max(xcs_by_eid[eid])) for eid in xcs_by_eid.keys()])

    evscores = {}
    for eid, xcs in xcs_by_eid.items():
        best_xcs = sorted(xcs)[-2:]
        evscores[eid] = np.sum(best_xcs)

    return evscores, eid_atimes, xcs_by_eid_full

def select_prototype_events(sg, evscores, N=3, dist_threshold=25.0):

    """
    - add the top N (3-5?) events with good correlation scores
    - then ensure that each event that has >3 neighbors within
          25km, has two prototypes in the set. so I loop over all events
          that have >2 25km neighbors, and if none of those neighbors
          is in the set, I add the highest-correlation one
          (potentially the event itself). repeat until the set is stable.
    """

    evs = dict([(eid, sg.get_event(eid)) for eid in evscores.keys()])
    eids_by_score_sorted = sorted(evscores.items(), key = lambda x : -x[1])

    prototype_eids = set()
    for i in range(N):
        prototype_eids.add(eids_by_score_sorted[i][0])

    neighbors = {}
    for eid, ev in evs.items():
        my_neighbors = []
        for eid2, ev2 in evs.items():
            d = dist_km((ev.lon, ev.lat), (ev2.lon, ev2.lat)) + np.abs(ev.depth - ev2.depth)
            if d < dist_threshold:
                my_neighbors.append(eid2)
        neighbors[eid] = my_neighbors

    substantial_neighbors = dict([(eid, eid_neighbors) for (eid, eid_neighbors) in neighbors.items() if len(eid_neighbors) > 4])

    stable_set = False
    while not stable_set:
        stable_set = True
        for eid, eid_neighbors in substantial_neighbors.items():
            represented = np.sum([neid in prototype_eids for neid in eid_neighbors]) >= 2
            if not represented:
                neighbor_scores = [(eid, evscores[eid]) for eid in eid_neighbors if eid not in prototype_eids]
                best_neighbor, best_score = sorted(neighbor_scores, key = lambda x : -x[1])[0]
                prototype_eids.add(best_neighbor)
                print "eid %d not represented, adding neighbor %d with score %.2f" % (eid, best_neighbor, best_score)
                stable_set = False
                break

    return prototype_eids

def select_prototype_events2(sg, evscores, xcs_by_eid, N=3, dist_threshold=25.0, xc_threshold=0.4):

    """
    make sure every event that has a nearby correlated event, should have one in the proototype set
    """

    evs = dict([(eid, sg.get_event(eid)) for eid in evscores.keys()])
    eids_by_score_sorted = sorted(evscores.items(), key = lambda x : -x[1])

    prototype_eids = set()
    for i in range(N):
        prototype_eids.add(eids_by_score_sorted[i][0])

    
    correlating_eids = {}
    for eid, ev in evs.items():
        sorted_eids = sorted(xcs_by_eid[eid], key = lambda x : -x[1])
        t = [eid2 for (eid2, xc) in sorted_eids if xc > xc_threshold]
        if len(t) > 0:
            correlating_eids[eid] = t


    stable_set = False
    while not stable_set:
        stable_set = True
        for eid, correlating_list in correlating_eids.items():
            represented = np.sum([neid in prototype_eids for neid in correlating_list]) >= 1
            if not represented:
                best_neighbor = [neid for neid in correlating_list if neid not in prototype_eids][0]
                best_score = 0.0
                #neighbor_scores = [(eid, evscores[eid]) for eid in correlating_list if eid not in prototype_eids]
                #best_neighbor, best_score = sorted(neighbor_scores, key = lambda x : -x[1])[0]
                prototype_eids.add(best_neighbor)
                print "eid %d not represented, adding neighbor %d with score %.2f" % (eid, best_neighbor, best_score)

                stable_set = False
                break

    return prototype_eids


def construct_sg_for_eids(sg1, eids, eid_atimes, model_type="gp_joint", model_dict=None, **kwargs):

    new_sg = SigvisaGraph(template_model_type=model_type, 
                          template_shape = sg1.template_shape,
                          wiggle_family=sg1.wiggle_family, 
                          min_mb=sg1.min_mb,
                          wiggle_model_type=model_type,
                          model_dict=model_dict,
                          raw_signals=True, 
                          base_srate=sg1.base_srate,
                          runids=sg1.runids, 
                          hack_param_constraint=True,
                          phases=sg1.phases,
                          jointgp_param_run_init=sg1.jointgp_param_run_init,
                          hack_ttr_max=sg1.hack_ttr_max,
                          uatemplate_rate=sg1.uatemplate_rate,
                          skip_levels=sg1.skip_levels,
                          force_event_wn_matching=True,
                          **kwargs)

    wave_pairs = []
    for eid in eids:
        wn, firstphase = eid_atimes[eid]
        s, stime, etime = wn.get_event_signal(eid, pre_s = 20.0, post_s = 50.0)
        mask = np.isnan(s)
        s_masked = ma.masked_array(s, mask)
        wave = Waveform(data=s_masked, stime=stime, sta=wn.sta, srate=wn.srate, filter_str=wn.filter_str, chan=wn.chan)

        sidx = time_to_index(stime, wn.st, wn.srate)
        eidx = time_to_index(etime, wn.st, wn.srate)

        #try:
        wn2 = new_sg.add_wave(wave, disable_conflict_checking=True)
        #except Exception as e: 
        #    print e
        #    continue

        wave_pairs.append((eid, wn, wn2))

    for (eid, wn1, wn2) in wave_pairs:
        evnodes = new_sg.add_event(sg1.get_event(eid), eid=eid)

        phases = [phase for (eid2, phase) in wn1.arrivals() if eid2==eid]
        for phase in phases:
            tmvals = sg1.get_template_vals(eid, wn1.sta, phase, wn1.band, wn1.chan)
            new_tmnodes = new_sg.get_template_nodes(eid, wn2.sta, phase, wn2.band, wn2.chan)
            for param, val in tmvals.items():
                k,n = new_tmnodes[param]
                n.set_value(val, key=k)

    # copy over uatemplates
    for (eid, wn1, wn2) in wave_pairs:
        for (eid2, phase) in wn1.arrivals():
            if eid2 == eid: continue
            tmvals, _ = wn1.get_template_params_for_arrival(eid2, phase)
            phase_sidx, phase_eidx = wn1.template_idx_window(eid=eid2, phase=phase, 
                                                             pre_arrival_slack_s=0.0,
                                                             post_fade_slack_s=0.0)
            phase_stime = index_to_time(phase_sidx, wn1.st, wn1.srate)
            phase_etime = index_to_time(phase_eidx, wn1.st, wn1.srate)

            if phase_etime > wn2.st and phase_stime < wn2.et:
                new_sg.create_unassociated_template(wn2, tmvals["arrival_time"], initial_vals=tmvals, nosort=True)

    new_sg._topo_sort()
    new_sg.current_log_p()

    try:
        new_sg.seed = sg1.seed
    except:
        new_sg.seed = 0

    return new_sg

def optimize_prototypes(sg1, prototype_eids, eid_atimes, old_run_dir):

    new_sg = construct_sg_for_eids(sg1, prototype_eids, eid_atimes)

    logger = MCMCLogger( write_template_vals=True, dump_interval_s=10.0, print_interval_s=10.0, write_gp_hparams=True, max_dumps=2, run_dir=old_run_dir+".prototypes")
    run_open_world_MH(new_sg, steps=400,
                      enable_event_openworld=False,
                      enable_event_moves=False,
                      enable_phase_openworld=False,
                      enable_template_openworld=False,
                      enable_template_moves=True,
                      enable_hparam_moves=True,
                      special_mb_moves=False,
                      template_move_type="rw",
                      logger=logger)

    return new_sg

def fit_eids_from_prototype_model(sg_full, sg_prototype, sta, eids, eid_atimes, old_run_dir):

    models = {}
    sg_prototype.current_log_p()
    for (param, band, chan, phase), (jgp, hnodes) in sg_prototype._joint_gpmodels[sta].items():
        gp = jgp.train_gp()
        if gp is None:
            continue

        models[(param, band, chan, phase)] = gp


    sg_indep = construct_sg_for_eids(sg_full, eids, eid_atimes, model_dict=models, model_type="dict", dummy_fallback=True)
    for sta, wns in sg_indep.station_waves.items():
        for wn in wns:
            wn.hack_wavelets_as_iid = True

    logger = MCMCLogger( write_template_vals=True, dump_interval_s=10.0, print_interval_s=10.0, write_gp_hparams=True, max_dumps=2, run_dir=old_run_dir+".align_indep")
    run_open_world_MH(sg_indep, steps=200,
                      enable_event_openworld=False,
                      enable_event_moves=False,
                      enable_phase_openworld=False,
                      enable_template_openworld=False,
                      enable_template_moves=True,
                      enable_hparam_moves=True,
                      special_mb_moves=False,
                      template_move_type="rw",
                      disable_moves=["atime_xc"],
                      logger=logger)
    return sg_indep

def repatriate_fits(sg_full, sg_single):

    for sta, wns in sg_single.station_waves.items():
        for wn in wns:
            for (eid, phase) in wn.arrivals():
                if eid < 1: continue

                tmvals = sg_single.get_template_vals(eid, wn.sta, phase, wn.band, wn.chan)
                tmnodes = sg_full.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)
                for param, val in tmvals.items():
                    k,n = tmnodes[param]
                    n.set_value(val, key=k)

                wn2 = list(tmnodes["coda_decay"][1].children)[0]
                wn2.nm_node.set_value(wn.nm)

def jointgp_iterative_align_init(sg, base_run_dir):
    assert(len(sg.station_waves.keys())==1)
    sta = sg.station_waves.keys()[0]

    evscores, eid_atimes, xcs_by_eid = score_event_correlations(sg, sta)
    #prototype_eids = select_prototype_events(sg, evscores)
    prototype_eids = select_prototype_events2(sg, evscores, xcs_by_eid)
    print "selected prototype eids", prototype_eids
    
    new_sg = optimize_prototypes(sg, prototype_eids, eid_atimes, base_run_dir)
    repatriate_fits(sg, new_sg)

    indep_eids = [eid for eid in eid_atimes.keys() if eid not in prototype_eids]
    sg_indep = fit_eids_from_prototype_model(sg, new_sg, sta, indep_eids, eid_atimes, base_run_dir)
    repatriate_fits(sg, sg_indep)

    return sg

def main():

    with open("/home/dmoore/python/sigvisa/logs/mcmc/01405/step_000000/pickle.sg", 'rb') as f:
        sg4 = pickle.load(f)

    jointgp_iterative_align_init(sg4)

    with open("updated_sg4_indep.sg", "wb") as f:
        pickle.dump(sg4, f)

if __name__ == "__main__":
    main()
