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

if __name__=="__main__":

     parser = OptionParser()

     parser.add_option("--cluster_fname", dest="cluster_fname", default=None, type="str", help="")
     parser.add_option("--runid", dest="runid", type="int", help="")
     parser.add_option("--sta", dest="sta", type="str", help="")
     
     (options, args) = parser.parse_args()

     s = Sigvisa()
     cursor = s.dbconn.cursor()

     evids = np.loadtxt(options.cluster_fname, dtype=int)

     for evid in evids:
         sql_query = "select fitid from sigvisa_coda_fit where runid=%d and evid=%d and sta='%s' " % (options.runid, evid, options.sta)
         cursor.execute(sql_query)
         r = cursor.fetchall()
         for fitid in np.array(r).flatten():
              sql_query = "delete from sigvisa_coda_fit_phase where fitid=%d" % fitid
              cursor.execute(sql_query)
              sql_query = "delete from sigvisa_coda_fit where fitid=%d" % fitid
              cursor.execute(sql_query)
              print "deleted existing fit", fitid

     cursor.close()
     s.dbconn.commit()
