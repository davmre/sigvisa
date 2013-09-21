import numpy as np
import numpy.ma as ma
import copy
import sys
import traceback
import pdb
import pickle


from sigvisa import Sigvisa
from sigvisa.graph.array_node import lldlld_X
from sigvisa.graph.sigvisa_graph import get_param_model_id
from sigvisa.infer.propose import generate_hough_array, propose_event_from_hough, event_prob_from_hough
from sigvisa.learn.train_param_common import load_modelid
from sigvisa.models.distributions import Gaussian
from sigvisa.models.ev_prior import event_from_evnodes
from sigvisa.models.ttime import tt_residual
from sigvisa.models.templates.coda_height import amp_transfer
from sigvisa.utils.counter import Counter
from sigvisa.source.event import get_event


def unass_template_logprob(sg, sta, template_dict):
    """

    return the log prob of a set of template parameters, under the
    model of unassociated templates at a station sta.

    """

    # HACK
    assert(len(sg.station_waves[sta]) == 1)
    wn = sg.station_waves[sta][0]

    tg = sg.template_generator(phase="UA")
    wg = sg.wiggle_generator(phase="UA", srate=wn.srate)


    lp = 0.0
    lp += -np.log(float(wn.npts)/wn.srate) # arrival time
    for param in tg.params():
        model = tg.unassociated_model(param, nm=wn.nm)
        lp += model.log_p(template_dict[param])
    for param in wg.params():
        model = wg.unassociated_model(param, nm=wn.nm)
        lp += model.log_p(template_dict[param])
    return lp

def param_logprob(sg, site, sta, ev, phase, chan, band, param, val, basisid=None):

    """

    return the log probability for an individual template parameter,
    as generated by an event phase arrival, WITHOUT interfering with
    the graph.

    """

    model_type = sg._tm_type(param, site=site)

    s = Sigvisa()
    if s.is_array_station(site) and sg.arrays_joint:
        modelid = get_param_model_id(runid=sg.runid, sta=site,
                                     phase=phase, model_type=model_type,
                                     param=param, template_shape=sg.template_shape,
                                     chan=chan, band=band, basisid=basisid)
        cond = lldlld_X(ev, sta)
    else:
        modelid = get_param_model_id(runid=sg.runid, sta=sta,
                                     phase=phase, model_type=model_type,
                                     param=param, template_shape=sg.template_shape,
                                     chan=chan, band=band, basisid=basisid)
        cond = ev

    model = load_modelid(modelid)
    return model.log_p(x = val, cond = cond)

def ev_phase_template_logprob(sg, sta, eid, phase, template_dict):

    """

    return log p(template params in template_dict) under the distribution generated by a phase arrival from event eid at station sta.

    """

    ev = event_from_evnodes(sg.evnodes[eid])

    s = Sigvisa()
    site = s.get_array_site(sta)

    # HACK
    assert(len(sg.station_waves[sta]) == 1)
    wn = sg.station_waves[sta][0]
    wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
    wiggle_params = set(wg.params())

    if 'tt_residual' not in template_dict:
        template_dict['tt_residual'] = tt_residual(ev, sta, template_dict['arrival_time'], phase=phase)

    # TODO: implement multiple bands/chans
    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]

    if 'amp_transfer' not in template_dict and "coda_height" in template_dict:
        template_dict['amp_transfer'] = amp_transfer(ev, band, phase, template_dict['coda_height'])
    # note if coda height is not specified, we'll ignore amp params
    # entirely: this is used by the create-new-template proposer
    # (in phase_template_proposal_logp etc)

    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]

    lp = 0
    for (param, val) in template_dict.items():
        if param in ('arrival_time', 'coda_height'): continue
        if param in wiggle_params:
            basisid = wg.basisid
        else:
            basisid = None
        lp_param = param_logprob(sg, site, sta, ev, phase, chan, band, param, val, basisid=basisid)
        lp += lp_param
    return lp


def template_association_logodds(sg, sta, tmid, eid, phase):

    tmnodes = sg.uatemplates[tmid]
    param_values = dict([(k, n.get_value()) for (k,n) in tmnodes.items()])

    lp_unass = unass_template_logprob(sg, sta, param_values)
    lp_ev = ev_phase_template_logprob(sg, sta, eid, phase, param_values)

    return lp_ev - lp_unass


def template_association_distribution(sg, sta, eid, phase):
    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]

    c = Counter()
    for tmid in sg.uatemplate_ids[(sta,chan,band)]:
        c[tmid] += np.exp(template_association_logodds(sg, sta, tmid, eid, phase))

    # if there are no unassociated templates, there's nothing to sample.
    n_u = len(sg.uatemplate_ids[(sta, chan, band)])
    if n_u == 0:
        c[None] = 1.0
        return c

    c[None] = np.exp(sg.ntemplates_sta_log_p(sta, n=n_u) - sg.ntemplates_sta_log_p(sta, n=n_u-1))
    c.normalize()
    return c

def sample_template_to_associate(sg, sta, eid, phase):
    """
    Propose associating an unassociate template at sta with the
    (eid,phase) arrival, with probability proportional to the odds
    ratio p_{E,P}(T)/p_U(T). Alternately propose creating a new
    template, with probability proportional to p(N_U = n_U)/p(N_U =
    n_U - 1).

    Return:
    tmid: the unassociated template id proposed for
          association. (value of None indicates proposing a creation
          move)
    assoc_logprob: log probability of the proposal

    """

    c = template_association_distribution(sg, sta, eid, phase)
    tmid = c.sample()
    assoc_logprob = np.log(c[tmid])

    return tmid, assoc_logprob

def associate_template(sg, sta, tmid, eid, phase):
    """

    Transform the graph to associate the template tmid with the arrival of eid/phase at sta.

    """

    tmnodes = sg.uatemplates[tmid]

    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    values = dict([(k, n.get_value()) for (k, n) in tmnodes.items()])
    sg.set_template(eid, sta, phase, band, chan, values)
    sg.destroy_unassociated_template(tmnodes, nosort=True)
    return

def unassociate_template(sg, sta, eid, phase, tmid=None):

    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    ev_tmvals = sg.get_template_vals(eid, sta, phase, band, chan)

    wave_node = sg.station_waves[sta][0]
    atime = ev_tmvals['arrival_time']
    tmnodes = sg.create_unassociated_template(wave_node, atime, wiggles=True, nosort=True,
                                           tmid=tmid, initial_vals=ev_tmvals)
    tmid = tmnodes.values()[0].tmid

    return tmid

def deassociation_prob(sg, sta, eid, phase, deletion_prob=False):

    # return prob of deassociating (or of deleting, if deletion_prob=True).

    s = Sigvisa()
    site = s.get_array_site(sta)


    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    ev_tmvals = sg.get_template_vals(eid, sta, phase, band, chan)

    unass_lp = unass_template_logprob(sg, sta, ev_tmvals)

    n_u = len(sg.uatemplate_ids[(sta,chan,band)])
    ntemplates_ratio_log = sg.ntemplates_sta_log_p(sta, n=n_u+1) - sg.ntemplates_sta_log_p(sta, n=n_u)


    deassociation_ratio_log = unass_lp + ntemplates_ratio_log

    wave_node = sg.station_waves[sta][0]
    signal_lp_with_template = wave_node.log_p()
    arrivals = copy.copy(wave_node.arrivals())
    arrivals.remove((eid, phase))
    signal_lp_without_template = wave_node.log_p(arrivals=arrivals)
    deletion_ratio_log = signal_lp_without_template - signal_lp_with_template

    log_normalizer = np.logaddexp(deassociation_ratio_log, deletion_ratio_log)

    if deletion_prob:
        return np.exp(deletion_ratio_log - log_normalizer)
    else:
        return np.exp(deassociation_ratio_log - log_normalizer)

def sample_deassociation_proposal(sg, sta, eid, phase):
    p = deassociation_prob(sg, sta, eid, phase)
    u = np.random.rand()
    deassociate = u < p
    deassociate_lp = np.log(p) if deassociate else np.log(1-p)
    return deassociate, deassociate_lp

def get_signal_based_amplitude_distribution(sg, sta, tmvals, peak_period_s = 1.0):
    wn = sg.station_waves[sta][0]
    peak_time = tmvals['arrival_time'] + tmvals['peak_offset']
    peak_idx = int((peak_time - wn.st) * wn.srate)
    peak_period_samples = int(peak_period_s * wn.srate)
    peak_data=wn.get_value()[peak_idx - peak_period_samples:peak_idx + peak_period_samples]

    # if we land outside of the signal window, or during an unobserved (masked) portion,
    # we'll just sample from the event-conditional prior instead
    if ma.count(peak_data) == 0:
        return None

    peak_height = peak_data.mean()

    env_height = max(peak_height - wn.nm.c, wn.nm.c/100.0)



    return Gaussian(mean=np.log(env_height), std = 1.0)

def propose_phase_template(sg, sta, eid, phase):
    # sample a set of params for a phase template from an appropriate distribution (as described above).
    # return as an array.

    s = Sigvisa()
    site = s.get_array_site(sta)

    # we assume that add_event already sampled all the params parent-conditionally
    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    tmvals = sg.get_template_vals(eid, sta, phase, band, chan)
    if 'amp_transfer' in tmvals:
        del tmvals['amp_transfer']

    amp_dist = get_signal_based_amplitude_distribution(sg, sta, tmvals)
    if amp_dist is not None:

        amplitude = amp_dist.sample()

        del tmvals['coda_height']

        # compute log-prob of non-amplitude parameters
        lp = ev_phase_template_logprob(sg, sta, eid, phase, tmvals)
        tmvals['coda_height'] = amplitude
        lp += amp_dist.log_p(amplitude)

    else:
        lp = ev_phase_template_logprob(sg, sta, eid, phase, tmvals)

    return tmvals, lp

def phase_template_proposal_logp(sg, sta, eid, phase, tmvals):
    # return the logprob of params from the proposal distribution

    tmvals = copy.copy(tmvals)
    if 'amp_transfer' in tmvals:
        del tmvals['amp_transfer']

    amplitude = tmvals['coda_height']
    amp_dist = get_signal_based_amplitude_distribution(sg, sta, tmvals)

    if amp_dist is not None:
        del tmvals['coda_height']
        lp = amp_dist.log_p(amplitude)
    else:
        lp = 0.0
    lp += ev_phase_template_logprob(sg, sta, eid, phase, tmvals)

    return lp

def death_proposal_log_ratio(sg, eid):

    lp_unass = 0
    lp_ev = 0

    evnodes = sg.evnodes[eid]
    ev = event_from_evnodes(evnodes)
    eid = ev.eid

    for (site, elements) in sg.site_elements.items():
        assert (len(list(sg.site_bands[site])) == 1)
        assert (len(list(sg.site_chans[site])) == 1)
        for sta in elements:
            for phase in sg.phases:
                for chan in sg.site_chans[site]:
                    for band in sg.site_bands[site]:
                        tmvals = sg.get_template_vals(eid, sta, phase, band, chan)

                        lp_unass_tmpl = unass_template_logprob(sg, sta, tmvals)
                        lp_ev_tmpl = ev_phase_template_logprob(sg, sta, eid, phase, tmvals)
                        lp_unass += lp_unass_tmpl
                        lp_ev += lp_ev_tmpl

    return lp_unass - lp_ev

def death_proposal_distribution(sg):
    c = Counter()
    for eid in sg.evnodes.keys():
        c[eid] = death_proposal_log_ratio(sg, eid)

    #
    c_log = copy.copy(c)
    if len(c) > 0:

        log_normalizer=np.float('-inf')
        for v in c_log.values():
            log_normalizer = np.logaddexp(v, log_normalizer)
        for k in c_log.keys():
            c_log[k] -= log_normalizer

        v = np.max(c.values())
        for eid in c.keys():
            c[eid] = np.exp(c[eid] - v)
        c.normalize()

    return c, c_log

def sample_death_proposal(sg):
    c, c_log = death_proposal_distribution(sg)
    if len(c) == 0:
        return None, 1.0
    eid = c.sample()
    return eid, c_log[eid]

def death_proposal_logprob(sg, eid):
    c, c_log = death_proposal_distribution(sg)
    if len(c) == 0:
        return 1.0
    return c_log[eid]

def ev_death_move(sg):

    lp_old = sg.current_log_p()

    eid, eid_logprob = sample_death_proposal(sg)
    if eid is None:
        return False
    ev = event_from_evnodes(sg.evnodes[eid])

    move_logprob = eid_logprob
    n_current_events = len(sg.evnodes)
    reverse_logprob = -np.log(n_current_events) # this accounts for the different "positions" we can birth an event into

    forward_fns = []
    inverse_fns = []
    inverse_fns.append(lambda : sg.add_event(ev, eid=eid))

    tmids = []
    tmid_i = 0

    deassociations = []

    # loop over phase arrivals at each station and propose either
    # associating an existing unass. template with the new event, or
    # creating a new template.
    # don't modify the graph, but generate a list of functions
    # to execute the forward and reverse moves
    for elements in sg.site_elements.values():
        for sta in elements:

            s = Sigvisa()
            site = s.get_array_site(sta)
            assert (len(list(sg.site_bands[site])) == 1)
            band = list(sg.site_bands[site])[0]
            assert (len(list(sg.site_chans[site])) == 1)
            chan = list(sg.site_chans[site])[0]

            for phase in sg.phases:
                deassociate, deassociate_logprob = sample_deassociation_proposal(sg, sta, eid, phase)
                deassociations.append((sta, phase, deassociate, tmid_i))
                if deassociate:
                    # deassociation will produce a new uatemplated
                    # with incrementing tmid. We keep track of this
                    # tmid (kind of a hack) to ensure that we
                    # reassociate the same template if the move gets
                    # rejected.
                    forward_fns.append(lambda sta=sta,phase=phase: tmids.append(unassociate_template(sg, sta, eid, phase)))
                    inverse_fns.append(lambda sta=sta,phase=phase,tmid_i=tmid_i: associate_template(sg, sta, tmids[tmid_i], eid, phase))
                    tmid_i += 1
                    #print "proposing to deassociate at %s (lp %.1f)" % (sta, deassociate_logprob)

                else:
                    template_param_array = sg.get_template_vals(eid, sta, phase, band, chan)
                    inverse_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    tmp = phase_template_proposal_logp(sg, sta, eid, phase, template_param_array)
                    reverse_logprob += tmp
                    #print "proposing to delete at %s (lp %f)"% (sta, deassociate_logprob)

                move_logprob += deassociate_logprob

    # order of operations:
    # first, deassociate the templates we need to deassociate
    # second, calculate probabilities of re-associating them to the event (while it's still around)
    # finally, kill the event
    for fn in forward_fns:
        fn()

    for (sta, phase, deassociate, tmid_i) in deassociations:
        c = template_association_distribution(sg, sta, eid, phase)
        if deassociate:
            tmid = tmids[tmid_i]
            tmp = np.log(c[tmid])
            reverse_logprob += tmp
        else:
            tmp = np.log(c[None])
            reverse_logprob += tmp

    sg.remove_event(eid)
    # no need to topo sort since remove_event does it for us

    lp_new = sg.current_log_p()

    hough_array = generate_hough_array(sg, stime=sg.event_start_time, etime=sg.end_time, bin_width_deg=4.0)
    ev_logprob = np.log(event_prob_from_hough(ev, hough_array, sg.event_start_time, sg.end_time))
    reverse_logprob += ev_logprob

    """
    print "move lp", move_logprob
    print "reverse lp", reverse_logprob
    print "new lp", lp_new
    print "old lp", lp_old
    print "MH acceptance ratio", (lp_new + reverse_logprob) - (lp_old + move_logprob)
    """
    assert(np.isfinite((lp_new + reverse_logprob) - (lp_old + move_logprob)))
    u = np.random.rand()
    move_accepted = (lp_new + reverse_logprob) - (lp_old + move_logprob)  > np.log(u)
    if move_accepted:
        print "move accepted"
        return True
    else:
        #print "move rejected"
        for fn in inverse_fns:
            fn()
        sg._topo_sort()
        #print "changes reverted"
        return False


def ev_birth_move(sg):
    lp_old = sg.current_log_p()

    hough_array = generate_hough_array(sg, stime=sg.event_start_time, etime=sg.end_time, bin_width_deg=4.0)
    proposed_ev, ev_prob = propose_event_from_hough(hough_array, sg.event_start_time, sg.end_time)
    print "proposed ev", proposed_ev
    #proposed_ev = get_event(evid=5393637)
    #ev_prob = 1.0

    forward_fns = []
    inverse_fns = []
    associations = []

    n_current_events = len(sg.evnodes)
    move_logprob = -np.log(n_current_events+1) # we imagine there are n+1 "positions" we can birth an event into
    move_logprob += np.log(ev_prob)

    # add an event, WITH all its template nodes initialized to parent-sampled values.
    # we need to replace these values before computing any signal-based probabilities.
    # luckily,
    evnodes = sg.add_event(proposed_ev, sample_templates=True)
    eid = evnodes['mb'].eid
    #print "added proposed event to graph"

    # loop over phase arrivals at each station and propose either
    # associating an existing unass. template with the new event, or
    # creating a new template.
    # don't modify the graph, but generate a list of functions
    # to execute the forward and reverse moves
    for elements in sg.site_elements.values():
        for sta in elements:

            s = Sigvisa()
            site = s.get_array_site(sta)
            assert (len(list(sg.site_bands[site])) == 1)
            band = list(sg.site_bands[site])[0]
            assert (len(list(sg.site_chans[site])) == 1)
            chan = list(sg.site_chans[site])[0]

            for phase in sg.phases:
                tmid, assoc_logprob = sample_template_to_associate(sg, sta, eid, phase)
                if tmid is not None:
                    forward_fns.append(lambda sta=sta,phase=phase,tmid=tmid: associate_template(sg, sta, tmid, eid, phase))
                    inverse_fns.append(lambda sta=sta,phase=phase: unassociate_template(sg, sta, eid, phase))
                    associations.append((sta, phase, True))
                    #print "proposing to associate template %d at %s,%s with assoc lp %.1f" % (tmid, sta, phase, assoc_logprob)
                    tmpl_lp  = 0.0
                else:
                    template_param_array, tmpl_lp = propose_phase_template(sg, sta, eid, phase)
                    forward_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    #inverse_fns.append(lambda : delete_template(sg, sta, eid, phase))
                    associations.append((sta, phase, False))

                sta_phase_logprob = assoc_logprob + tmpl_lp
                move_logprob += sta_phase_logprob


    inverse_fns.append(lambda : sg.remove_event(eid))

    # execute all the forward moves
    for fn in forward_fns:
        fn()
    sg._topo_sort()

    # compute log probability of the reverse move.
    # we have to do this in a separate loop so that
    # we can execute all the forward moves first.
    reverse_logprob = death_proposal_logprob(sg, eid)
    for (sta, phase, associated) in associations:
        reverse_logprob += np.log(deassociation_prob(sg, sta, eid, phase, deletion_prob=not associated))

    lp_new = sg.current_log_p()

    """
    print "move lp", move_logprob
    print "reverse lp", reverse_logprob
    print "new lp", lp_new
    print "old lp", lp_old
    print "MH acceptance ratio", (lp_new + reverse_logprob) - (lp_old + move_logprob)
    """

    assert(np.isfinite((lp_new + reverse_logprob) - (lp_old + move_logprob)))

    u = np.random.rand()
    move_accepted = (lp_new + reverse_logprob) - (lp_old + move_logprob)  > np.log(u)
    if move_accepted:
        print "move accepted"
        assert( evnodes['loc']._mutable[evnodes['loc'].key_prefix + 'depth'])
        return True
    else:
        #print "move rejected"
        for fn in inverse_fns:
            fn()
        # no need to topo sort here since remove_event does it for us

        #print "changes reverted"
        return False

def main():


    f = open('cached_templates2.sg', 'rb')
    sg = pickle.load(f)
    f.close()

    # update the pickled graph to something resembling the current structure
    sg.runid = 17
    from collections import defaultdict
    sg.extended_evnodes = defaultdict(list)
    sg.event_rate = 0.00126599049
    sg.event_start_time = sg.start_time - 2000.0

    np.random.seed(0)
    print ev_birth_move(sg)

    f = open('postbirth.sg', 'wb')
    pickle.dump(sg, f)
    f.close()
    """

    f = open('postbirth.sg', 'rb')
    sg = pickle.load(f)
    f.close()
    """

    print ev_death_move(sg)

    print "final lp", sg.current_log_p()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
