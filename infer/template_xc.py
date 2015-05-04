import numpy as np

from sigvisa.utils.geog import dist_km
from sigvisa.models.ttime import tt_predict
from sigvisa.models.signal_model import ObservedSignalNode
from sigvisa.infer.mcmc_basic import MH_accept
import scipy.weave as weave
from scipy.weave import converters

def fastxc(a, b):
    """
    Inputes: np arrays a,b, where len(a) < len(b).
    Computes the normalized cross-correlation
    |w'a|/(||a||||w||),
    where w is a sliding window over b.
    """

    # assume len(a) < len(b)
    n = len(b) - len(a)+1
    m = len(a)

    # remove nans
    a[np.isnan(a)] = 0.0
    b[np.isnan(b)] = 0.0

    r = np.zeros((n,))
    a_normed = a / np.linalg.norm(a)
    code="""
for(int i=0; i < n; ++i) {
    double b_norm = 0;
    double cc = 0;
    for (int j=0; j < m; ++j) {
        cc += a_normed(j)*b(i+j);
        b_norm += b(i+j)*b(i+j);
    }
    cc /= sqrt(b_norm);
    r(i) = cc;
}
"""
    weave.inline(code,['n', 'm', 'a_normed', 'b', 'r',],type_converters = converters.blitz,verbose=2,compiler='gcc')
    """
    for i in range(n):
        window = b[i:i+len(a)]
        w_normed = window / np.linalg.norm(window)
        r[i] = np.dot(a_normed, w_normed)
    """

    r[np.isnan(r)] = 0.0

    return r


def sample_xc_source_arrival(sg, eid_target, phase, sta, band, chan):
    eids = set(sg.evnodes.keys())
    eids.remove(eid_target)
    eids = list(eids)

    ev_target = sg.get_event(eid_target)
    tries = 10
    while tries > 0:
        eid = np.random.choice(eids)
        ev_src = sg.get_event(eid)
        dist = dist_km((ev_target.lon, ev_target.lat),
                       (ev_src.lon, ev_src.lat))
        if dist < 100: break
        tries -= 1

    tmnodes = sg.get_template_nodes(eid, sta, phase, band, chan)
    wn = [n for n in tmnodes['coda_height'][1].children if isinstance(n, ObservedSignalNode)][0]
    return eid, wn

def get_arrival_signal(sg, eid, phase, wn, pre_s, post_s, pred_atime=False, atime=None):

    if atime is None:
        if pred_atime:
            ev = sg.get_event(eid)
            atime = ev.time + tt_predict(ev, wn.sta, phase=phase)
        else:
            v, tg = wn.get_template_params_for_arrival(eid, phase)
            atime = v['arrival_time']

    st_idx = int((atime-pre_s - wn.st)*wn.srate)
    et_idx = int((atime+post_s - wn.st)*wn.srate)

    st_idx_clipped = max(st_idx, 0)
    et_idx_clipped = min(et_idx, wn.npts)

    if et_idx_clipped <= st_idx_clipped:
        raise Exception("arrival is not supported at this wave node")

    d = wn.get_value()
    return d[st_idx_clipped:et_idx_clipped], atime, (st_idx_clipped-st_idx)

def atime_proposal_distribution_from_xc(sg, eid_src, eid_target, phase,
                                        wn_src, wn_target,
                                        temp=1.0):

    if wn_target.srate != wn_src.srate:
        raise Exception("cross-correlation requires matching srates")
    srate = float(wn_target.srate)

    # load 10s immediately following the *current hypothesized arrival
    # time* for the source event.
    signal_src, atime_src, idx_src = get_arrival_signal(sg, eid_src, phase, wn_src,
                                                        pre_s=0.0, post_s=10.0,
                                                        pred_atime=False)

    # load a 40s time period around the *predicted* arrival time for
    # the target event. (using the current hypothesis would require
    # us to use a different window for the backwards proposal, since
    # the proposal will change the hypothesis, whereas the predicted
    # time is unchanged).
    signal_target, atime_target, idx_target = get_arrival_signal(sg, eid_target, phase, wn_target,
                                                                 pre_s=15.0, post_s=25.0,
                                                                 pred_atime=True)

    # slide the source arrival over the target
    xc = fastxc(signal_src, signal_target)
    xcdist = np.exp(temp*xc)

    # add a traveltime prior
    target_window_soffset = -15.0 + idx_target/srate
    target_window_eoffset = target_window_soffset + len(xc)/srate
    x = np.linspace(target_window_soffset, target_window_eoffset, len(xc))
    prior = np.exp(-np.abs(x/3.0))

    xcdist *= prior
    xcdist /= np.sum(xcdist)

    def idx_to_atime(idx):
        # if src signal was clipped, this will give us the
        # *correct* target index at which the arrival should
        # start
        st_idx = idx-idx_src

        # target_window_offset + st_idx/srate is the adjustment
        # to the *source* atime: e.g., if idx=0 then the source
        # aligns with the *beginning* of the target window.
        # If the window_soffset is -15, this means the source
        # aligns 15s before the predicted target.

        # wait what?

        # the source would have to move 15s earlier to match up.
        # To get new target atime, we *subtract* this adjustment
        # from the predicted target atime.
        atime = atime_target + target_window_soffset + st_idx/srate
        return atime

    def atime_to_idx(atime):
        # st_idx gives the index into the target window for this atime
        st_idx = int(np.round(srate*(atime - atime_target - target_window_soffset)))

        # now we correct to get the index we'd expect the source window
        # to align at, given that it might be clipped
        return st_idx+idx_src

    return xcdist, idx_to_atime, atime_to_idx


def xc_move(sg, wave_node, eid, phase, tmnodes, propose_peak=False, **kwargs):
    wn_target = wave_node
    eid_target = eid

    k_atime, n_atime = tmnodes['arrival_time']
    current_atime = n_atime.get_value(key=k_atime)

    relevant_nodes = [wn_target,]
    relevant_nodes += [n_atime.parents[n_atime.default_parent_key()],] if n_atime.deterministic() else [n_atime,]

    eid_src, wn_src = sample_xc_source_arrival(sg, eid_target, phase, wn_target.sta, wn_target.band, wn_target.chan)
    xcdist, idx_to_atime, atime_to_idx = atime_proposal_distribution_from_xc(sg, eid_src, eid_target, phase, wn_src, wn_target, temp=10.0)

    proposed_idx = np.random.choice(np.arange(len(xcdist)), p=xcdist)

    proposed_atime = idx_to_atime(proposed_idx)

    log_qforward = np.log(xcdist[proposed_idx])
    backwards_idx = atime_to_idx(current_atime)

    log_qbackward = np.log(xcdist[backwards_idx])

    #print "atime_xc %s %d: atime %.1f proposing %.1f qf %.1f qb %.1f" % (wn_target.sta, eid, current_atime, proposed_atime, log_qforward, log_qbackward)

    return current_atime, proposed_atime, log_qforward, log_qbackward, k_atime, n_atime, relevant_nodes

def atime_xc_move(sg, wave_node, eid, phase, tmnodes, **kwargs):
    """
    Propose a new arrival time based on cross-correlation with a random
    source event.
    """

    try:
        current_atime, proposed_atime, log_qforward, log_qbackward, k_atime, n_atime, relevant_nodes = xc_move(sg, wave_node, eid, phase, tmnodes, **kwargs)
    except Exception as e:
        print e
        return False

    return MH_accept(sg, keys=(k_atime,),
                     oldvalues = (current_atime,),
                     newvalues = (proposed_atime,),
                     log_qforward = log_qforward,
                     log_qbackward = log_qbackward,
                     node_list = (n_atime,),
                     relevant_nodes = relevant_nodes,)

def constpeak_atime_xc_move(sg, wave_node, eid, phase, tmnodes, **kwargs):
    """
    Propose a new arrival time from cross-correlation, while also
    adjusting the offset to leave the peak time constant. The hope is
    that this is more likely to be accepted, since changing the peak
    time (as the basic atime move does) is a very disruptive thing.
    """

    try:
        current_atime, proposed_atime, log_qforward, log_qbackward, k_atime, n_atime, relevant_nodes = xc_move(sg, wave_node, eid, phase, tmnodes, **kwargs)
    except Exception as e:
        print e
        return False

    k_offset, n_offset = tmnodes['peak_offset']
    current_offset = n_offset.get_value(key=k_offset)
    peak = current_atime + np.exp(current_offset)

    proposed_offset = np.log(peak-proposed_atime)
    if np.isnan(proposed_offset):
        # if the proposed atime is *after* the current peak,
        # it's not possible to maintain the current peak time.
        return False

    relevant_nodes += [n_offset,]

    return MH_accept(sg, keys=(k_atime,k_offset),
                     oldvalues = (current_atime,current_offset),
                     newvalues = (proposed_atime,proposed_offset),
                     log_qforward = log_qforward,
                     log_qbackward = log_qbackward,
                     node_list = (n_atime,n_offset),
                     relevant_nodes = relevant_nodes,)
