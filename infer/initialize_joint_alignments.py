import numpy as np
from sigvisa import Sigvisa

from sigvisa.utils.geog import dist_km

from sigvisa.models.ttime import tt_predict
from sigvisa.infer.correlations.ar_correlation_model import ar_advantage, iid_advantage
from sigvisa.infer.template_xc import fastxc
from collections import defaultdict
from sigvisa.models.distributions import Laplacian
from sigvisa.utils.array import index_to_time, time_to_index

def align_atimes(sg, sta, phase, 
                 patch_len_s=12.0, 
                 buffer_len_s=8.0, 
                 nearby_ev_threshold_km=15.0,
                 n_random_inits=3,
                 n_coord_ascent_iters=5,
                 plot=False,
                 center_at_current_atime=False,
                 temper_signal=5.0):

    def extract_windows():
        windows = {}
        noise_models = {}
        pred_atimes = {}
        srate = None
        
        # get the set of events with arrivals for this sta/phase
        # for each ev, get a window of signal around its predicted atime
        for eid in sg.evnodes.keys():
            try:
                wn = sg.get_arrival_wn(sta, eid, phase, band=None, chan=None)
            except:
                continue

            if srate is None:
                srate = wn.srate
            else:
                assert(srate == wn.srate)

            ev = sg.get_event(eid)
            if center_at_current_atime:
                pred_atime = wn.get_template_params_for_arrival(eid, phase)[0]["arrival_time"]
            else:
                pred_atime = ev.time + tt_predict(ev, sta, phase)

            pred_atimes[eid] = pred_atime

            sidx = time_to_index(pred_atime - buffer_len_s, wn.st, srate)
            eidx = sidx + int((patch_len_s + 2*buffer_len_s) * srate)
            windows[eid] = wn.get_value()[sidx:eidx].copy()

            for (other_eid, other_phase) in wn.arrivals():
                if other_eid == eid and other_phase==phase: continue
                other_atime = wn.get_template_params_for_arrival(other_eid, other_phase)[0]["arrival_time"]
                #other_ev = sg.get_event(other_eid)
                #other_pred_atime = other_ev.time + tt_predict(other_ev, sta, other_phase)
                other_idx = time_to_index(other_atime, stime=(pred_atime - buffer_len_s), srate=srate)
                if other_idx > buffer_len_s*srate and other_idx < len(windows[eid]):
                    print "zeroing", eid, "starting at", other_idx, "from", other_eid, other_phase
                    windows[eid][other_idx:] = 0.0

            noise_models[eid] = wn.nm
        return windows, noise_models, pred_atimes, srate 

    def get_nearby_eids(sg, windows):
        # then for each ev with signal, get the indices of all other evs within the distance threshold
        nearby_eids = dict()
        for eid in windows.keys():
            nearby_eids[eid] = set()
            ev = sg.get_event(eid)
            for eid2 in windows.keys():
                if eid2 == eid: continue
                ev2 = sg.get_event(eid2)
                if dist_km((ev.lon, ev.lat), (ev2.lon, ev2.lat)) < nearby_ev_threshold_km:
                    nearby_eids[eid].add(eid2)
        return nearby_eids
                

    # we want to maximize, over all events, the log-probability of
    #    - the tt residual model of atimes, plus
    #    - the ar/iid advantage for the average of patches from nearby events
    # with respect to the indices that determine atimes/patches for each event. 
    
    windows, noise_models, pred_atimes, srate  = extract_windows()
    if len(windows) < 2:
        return

    nearby_eids = get_nearby_eids(sg, windows)
    
    max_align = int(buffer_len_s * 2 * srate) 
    patch_len = int(patch_len_s * srate)
    
    ttr_model = Laplacian(0, 3.0)
    ttrs = np.linspace(-buffer_len_s, buffer_len_s, max_align)
    ttr_prior = np.array([ttr_model.log_p(s) for s in ttrs])
    
    def random_init_alignments(seed):
        np.random.seed(seed)        
        alignments = dict()
        for eid in nearby_eids.keys():
            offset_s = np.random.randn() * 2            
            align_idx = (buffer_len_s + offset_s) * srate
            align_idx = max(align_idx, 0)
            align_idx = min(align_idx, max_align)
            alignments[eid] = align_idx
        return alignments

    def extract_patch(window, align_idx):
        patch = window[align_idx:align_idx+patch_len].copy()
        patch -= np.mean(patch)
        patch /= np.linalg.norm(patch)
        return patch
    
    def extract_patches(alignments):
        patches = {}
        for eid, align_idx in alignments.items():            
            patches[eid] = extract_patch(windows[eid], align_idx)
        return patches
    
    def alignment_objective(alignments):
        # alignment objective that we want to maximize, as a function
        # of arrival times (alignments) for each phase.
        # the objective is the sum of traveltime model log-probabilities,
        # with an approximate signal probability for each event computed by 
        #assuming that the surrounding signal window is sampled from an AR 
        # noise process, except for the patch starting at the arrival time
        # which also contains some scaled version of the average of patches 
        # from nearby events (where the exact scaling is unknown and optimized 
        # out analytically inside the ar_advantage function)

        patches = extract_patches(alignments)
        
        score = 0.0
        for eid in nearby_eids.keys():
            window = windows[eid]
            nm = noise_models[eid]
            
            nearby_patches = [patches[nearby_eid] for nearby_eid in nearby_eids[eid]]
            if len(nearby_patches) > 0:                
                mean_neighbor = np.mean(nearby_patches, axis=0)
            else:
                mean_neighbor =  np.zeros((patch_len,))
                
            #signal_lps = ar_advantage(window, mean_neighbor, nm) / temper_signal
            #signal_lps = iid_advantage(window, mean_neighbor) / np.std(window)
            signal_lps = np.exp(fastxc(mean_neighbor, window) * temper_signal)[:len(ttr_prior)]
            likelihood = signal_lps + ttr_prior
            arrival_idx = alignments[eid]
            score += likelihood[arrival_idx]
        return score
    
    def coord_ascent(alignments):
        
        # loop over events in a random order, and for each event,
        # choose the alignment that maximizes the *local* score.
        # note this is not quite the same as the global score,
        # because we don't account for how the alignment of this 
        # event will change the mean patch against which other 
        # events are correlated, so this is not a perfect
        # coord ascent move and is not guaranteed to increase
        # the global objective. 
        patches = extract_patches(alignments)
        random_eids = np.random.permutation(nearby_eids.keys())
        for eid in random_eids:
            window = windows[eid]
            nm = noise_models[eid]
            
            nearby_patches = [patches[nearby_eid] for nearby_eid in nearby_eids[eid]]
            if len(nearby_patches) > 0:                
                mean_neighbor = np.mean(nearby_patches, axis=0)
            else:
                mean_neighbor =  np.zeros((patch_len,))

            #signal_lps = ar_advantage(window, mean_neighbor, nm) / temper_signal
            #signal_lps = iid_advantage(window, mean_neighbor) / np.std(window)
            signal_lps = np.exp(fastxc(mean_neighbor, window) * temper_signal)[:len(ttr_prior)]
            likelihood = signal_lps + ttr_prior
            
            old_arrival_idx = alignments[eid]
            new_arrival_idx = np.argmax(likelihood)

            # TODO should this change be made batchwise or online?
            alignments[eid] = new_arrival_idx
            patches[eid] = extract_patch(windows[eid], new_arrival_idx)
            
        return alignments
    
    best_alignments = None
    best_score = -np.inf
    for random_init in np.arange(n_random_inits):
        alignments = random_init_alignments(random_init)
        old_score = -np.inf
        for i in range(n_coord_ascent_iters):
            new_alignments = coord_ascent(alignments)
            score = alignment_objective(alignments)            
            print "init", random_init, "iter", i, "score", score
            if score == old_score:
                break
            old_score = score
        if score > best_score:
            best_alignments = alignments
            best_score = score
     
    def plot_alignments(alignments):

        import matplotlib.pylab as plt

        patches = extract_patches(alignments)
        for eid in alignments.keys():
            window = windows[eid]
            nm = noise_models[eid]
            
            nearby_patches = [patches[nearby_eid] for nearby_eid in nearby_eids[eid]]
            if len(nearby_patches) > 0:                
                mean_neighbor = np.mean(nearby_patches, axis=0)
            else:
                mean_neighbor =  np.zeros((patch_len,))
            
            plt.figure(figsize=(12, 4))
            plt.plot(window / np.linalg.norm(window))
            arrival_idx = alignments[eid]
            coords = np.arange(arrival_idx, arrival_idx + len(mean_neighbor))
            plt.plot(coords, mean_neighbor)
            ttr = (arrival_idx / srate) - buffer_len_s
            plt.title("eid %d align %d ttr %.2fs" % (eid, arrival_idx, ttr))
            
        return score
    
    def set_alignments(alignments):
        for eid in alignments.keys():
            wn = sg.get_arrival_wn(sta, eid, phase, band=None, chan=None)
            
            pred_atime = pred_atimes[eid]
            align_idx = alignments[eid]

            window_sidx = time_to_index(pred_atime - buffer_len_s, wn.st, wn.srate)
            window_stime = index_to_time(window_sidx, wn.st, wn.srate)

            # don't set atimes to exact idx boundaries
            # because it might trigger floating-point rounding errors in other code.
            jitter = 0.1/srate

            aligned_atime = align_idx / srate + window_stime + jitter

            tmnodes = sg.get_template_nodes(eid, sta, phase, wn.band, wn.chan)
            k_atime, n_atime = tmnodes["arrival_time"]
            n_atime.set_value(aligned_atime, key=k_atime)
    
    print "best score", best_score
    set_alignments(best_alignments)

    if plot:
        plot_alignments(best_alignments)
    
    
    

    
