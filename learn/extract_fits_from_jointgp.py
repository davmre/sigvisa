import numpy as np

import time

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser


from sigvisa import Sigvisa
#from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *

from sigvisa.learn.fit_shape_params_mcmc import compute_wavelet_messages, compute_template_messages, save_template_params
from sigvisa.infer.mcmc_logger import MCMCLogger

"""
extract fits from a jointGP
"""




def main(sg_fname, run_name, run_iter=1, burnin=20):
     #with open("/home/dmoore/python/sigvisa/logs/mcmc/01962/step_000019/pickle.sg", 'rb') as f:
     #    sg_joint = pickle.load(f)
     with open(sg_fname, 'rb') as f:
          sg = pickle.load(f)

     # assume the fname is of the form run_dir/step_xxxxxx/pickle.sg
     run_dir = os.path.dirname(os.path.dirname(sg_fname))

     s = Sigvisa()
     cursor = s.dbconn.cursor()
     runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=True)
     print "runid", runid

     logger = MCMCLogger(run_dir=run_dir, write_template_vals=True)

     for eid in sg.evnodes.keys():
          evid = sg.evnodes[eid]["loc"].evid
          assert(evid is not None)

          wns = [wn for wns in sg.station_waves.values() for wn in wns]
          for wn in wns:
               eids = set([eeid for (eeid, phase) in wn.arrivals()])
               if eid not in eids: continue

               

               # compute template posterior, and set the graph state to the best template params
               messages, best_tmvals = compute_template_messages(sg, wn, logger, 
                                                                 burnin=burnin,
                                                                 target_eid=eid)
               for (vals, nodes) in best_tmvals.values():
                    for (v, n) in zip(vals, nodes):
                         n.set_value(v)

               # compute wavelet posterior
               wavelet_messages, wavelet_posteriors = compute_wavelet_messages(sg, wn)

               for k, v in wavelet_messages.items():
                    messages[k][sg.wiggle_family] = v
                    messages[k][sg.wiggle_family + "_posterior"] = wavelet_posteriors[k]

               print "SAVING PARAMS FOR WN", wn
               fitids = save_template_params(sg, eid, evid, "", "", -1, runid, messages)
               if len(fitids) == 0:
                    import pdb; pdb.set_trace()
               print "saved to fitids", fitids
               
     s.dbconn.commit()
     cursor.close()


if __name__=="__main__":

     parser = OptionParser()

     parser.add_option("--sg_fname", dest="sg_fname", default=None, type="str", help="")
     parser.add_option("--burnin", dest="burnin", default=20, type="int", help="")
     parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
     parser.add_option("-i", "--run_iter", dest="run_iter", default=1, type="int", help="run iteration")
     
     (options, args) = parser.parse_args()

     main(sg_fname=options.sg_fname, run_name=options.run_name, 
          run_iter=options.run_iter, burnin=options.burnin)