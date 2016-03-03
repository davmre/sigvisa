import pdb
import sys
import cPickle as pickle

fname = sys.argv[1]
with open(fname, "rb") as f:
    thing = pickle.load(f)
print thing
