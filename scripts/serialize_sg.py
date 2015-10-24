import numpy as np

from sigvisa.graph.sigvisa_graph import SigvisaGraph
import cPickle as pickle
import sys

sgfile = sys.argv[1]
outfile = sys.argv[2]

with open(sgfile, 'rb') as f:
    sg = pickle.load(f)

sg.serialize_to_file(outfile)
