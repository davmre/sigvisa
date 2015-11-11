import numpy as np
import cPickle as pickle
import sys

from sigvisa.infer.propose_hough import CTFProposer, get_uatemplates,global_hough,normalize_global,visualize_hough_array, score_assoc
from sigvisa.models.ttime import tt_predict
from sigvisa.source.event import Event

with open(sys.argv[1], 'rb') as f:
    sg_inferred = pickle.load(f)

#with open(sys.argv[2], 'rb') as f:
#    sg_leb = pickle.load(f)


def get_hough_bins(ev, hc, uatemplates_by_sta):
    coord = (ev.lon, ev.lat, ev.depth, ev.mb,ev.time)
    v = hc.coords_to_index(coord)

    return bins, evs


def get_hough_bins(sg, hc, uatemplates_by_sta):
    bins = []
    evs = []
    for eid, evnodes in sg.extended_evnodes.items():
        ev = sg.get_event(eid)
        snrs = np.array([np.exp(n.get_value())/list(n.children)[0].nm.c for n in evnodes if "coda_height" in n.label])
        solid_dets = np.sum(snrs > 1)

        matches = 0
        pmatches = 0
        pmatch_stas = []
        for wn, (atimes, amps, tmids) in uatemplates_by_sta.items():
            sta = wn.sta
            evtimes = np.array([n.get_value() for n in evnodes if "arrival_time" in n.label and sta in n.label])
            for evtime in evtimes:
                diffs = [np.abs(at-evtime) for at in atimes]
                #print "sta", sta, "evtime", evtime, "diffs", diffs
                if len(diffs) == 0: continue
                if np.min(diffs) < 20: matches += 1
        
            try:
                pred_p = tt_predict(ev, sta, "P") + ev.time
                pdiffs = [np.abs(at-pred_p) for at in atimes]
                if len(pdiffs) > 0 and np.min(pdiffs) < 20: 
                    pmatches += 1
                    pmatch_stas.append((sta, np.argmin(pdiffs)))
            except ValueError:
                pass

        
        print "eid", eid, "snrs", snrs, "solid", solid_dets, "matched", matches, "pmatched", pmatches
        if pmatches < 3: 
            continue
        else:
            print "accepting event", ev
            print "pmatches at", pmatch_stas

        coord = (ev.lon, ev.lat, ev.depth, ev.mb,ev.time)
        #coord = (ev.lon, ev.lat, ev.depth, ev.time, ev.mb)
        v = hc.coords_to_index(coord)
        bins.append(v)
        evs.append(ev)

    arr = hc.create_array(dtype=np.float32, fill_val=0.0)
    for bin in bins:
        arr[bin] += 1.0
    visualize_hough_array(arr, sg.station_waves.keys(), fname="/home/dmoore/public_html/true_array.png", ax=None, timeslice=None)


    return bins, evs
        

def get_hough_array(sg, hc, uatemplates_by_sta):
    global_array,assocs, nll = global_hough(sg, hc, uatemplates_by_sta, save_debug=False)
    global_dist = normalize_global(global_array.copy(), nll, one_event_semantics=True, hc=hc)

    visualize_hough_array(global_dist, sg.station_waves.keys(), fname="/home/dmoore/public_html/hough_array.png", ax=None, timeslice=None, region=sg.inference_region)
    np.save("hough_array", global_dist)
    print "hough_array.png/npy"
    return global_array, global_dist, assocs, nll


def proposal_likelihood(dist, bins):
    lp = 0.0
    for bin in bins:
        p = dist[bin]
        lp += np.log(p)
        print "lp += %f from ev" % np.log(p)

    ndist = 1.0-dist
    for bin in bins:
        ndist[bin] = 1.0
    nlp = np.sum(np.log(ndist))
    print "lp += %f from nulls" % nlp
    lp += nlp
    return lp

#sg_inferred.uatemplate_rate = 0.001
ctf = CTFProposer(sg_inferred, [1.0,0.2, 0.05], depthbin_bounds=[0,10,50,150,400,700], mbbins=[12,1,1], offset=False, phases=['P', 'S', 'Lg', 'PcP', 'ScP', 'pP', 'Pg']) #["P","Lg", "S", "PcP", "ScP", "pP"])
uatemplates_by_sta_full = get_uatemplates(sg_inferred)

#stas =  ['AKBB', 'MK31', 'KBZ', 'SONA0', 'AAK', 'AKTO', 'ZAA0', 'INK', 'USA0', 'TOA0', 'NB200', "LPAZ"]
#uatemplates_by_sta = dict([(sta, uatemplates_by_sta_full[sta]) for sta in uatemplates_by_sta_full.keys() if sta in stas])
uatemplates_by_sta = uatemplates_by_sta_full

#true_bins, true_evs = get_hough_bins(sg_leb, ctf.global_hc, uatemplates_by_sta)
global_array, global_dist, assocs, nll = get_hough_array(sg_inferred, ctf.global_hc, uatemplates_by_sta)

#lp = proposal_likelihood(global_dist, true_bins)
#print "proposal score", lp



#for bbin in true_bins:
#    print "bin", bbin
#    score_assoc(sg_inferred, ctf.global_hc, uatemplates_by_sta, assocs, bbin)
#    print "bin score", global_array[bbin], global_array[bbin] -nll

true_ev = Event(lon=-105.427, lat=43.731, depth=0.0, time=1239041017.07, mb=4.0)
coord = (true_ev.lon, true_ev.lat, true_ev.depth, true_ev.mb, true_ev.time)
v = ctf.global_hc.coords_to_index(coord)
print "true ev bin", v, true_ev
print "true ev reconstructed ", ctf.global_hc.index_to_coords(v)
print "true ev bin score", global_array[v],  global_array[v]-nll

best_bin = np.unravel_index(np.argmax(global_dist), global_dist.shape)
print "argmax bin", best_bin, 
print "argmax ev", ctf.global_hc.index_to_coords(best_bin)
#score_assoc(sg_inferred, ctf.global_hc, uatemplates_by_sta, assocs, best_bin)
print "bin score", global_array[best_bin],  global_array[best_bin]-nll
    
"""r = []
for i in range(20):
    ev, lp, _ = ctf.propose_event(sg_inferred, fix_result=None, one_event_semantics=True)
    r.append((ev, lp))
for i, (ev, lp) in enumerate(r):
    print i, lp, ev.lon, ev.lat, ev.depth, ev.time, ev.mb
"""
