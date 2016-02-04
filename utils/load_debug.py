import pdb
import sys
import cPickle as pickle

fname = sys.argv[1]
with open(fname, "rb") as f:
    sg = pickle.load(f)
import pdb; pdb.set_trace()
