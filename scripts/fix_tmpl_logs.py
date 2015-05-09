import numpy as np
import os
import cPickle as pickle
from collections import defaultdict
from sigvisa.utils.fileutils import mkdir_p

vals = {'noise_var': 0.01,
        'signal_var': 1.0,
        'horiz_lscale': 300,
        'depth_lscale': 5.0}

tmpls = defaultdict(str)
for fn in sorted(os.listdir('.')):
    if fn.startswith("step"):
        step = int(fn.split("_")[1])
        with open(os.path.join(fn, 'pickle.sg'), 'rb') as f:
            sg = pickle.load(f)

        for sta in sg.station_waves.keys():
            for eid in sg.evnodes.keys():
                for phase in sg.ev_arriving_phases(eid, sta):
                    try:
                        wn = [wn for wn in sg.station_waves[sta] if (eid, phase) in wn.arrivals()][0]
                    except IndexError:
                        continue

                    tmvals = sg.get_template_vals(eid, sta, phase, wn.band, wn.chan)

                    fname = ("ev_%05d" % eid, "tmpl_%d_%s_%s" % (eid, wn.label, phase))
                    tmpls[fname] += ('%06d %f %f %f %f %f\n' % (step,
                                                                tmvals['arrival_time'],
                                                                tmvals['peak_offset'],
                                                                tmvals['coda_height'],
                                                                tmvals['peak_decay'],
                                                                tmvals['coda_decay']))


        print "read step", step

for d, fname in tmpls.keys():
    mkdir_p(d)
    with open(os.path.join(d, fname), 'w') as f:
        f.write(tmpls[(d, fname)])
    print "wrote", fname
