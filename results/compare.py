import numpy as np

from database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_MB_COL,\
     EV_TIME_COL

from utils.geog import dist_deg, dist_km

import mwmatching

DELTA_TIME = 50                         # in seconds
DELTA_DIST = 5                          # in degrees


def find_matching(gold_events, guess_events):
  """
  We want a max cardinality min cost matching.
  Returns a list of pairs of gold,guess indices
  """
  edges = []
  for goldnum, gold in enumerate(gold_events):
    for guessnum, guess in enumerate(guess_events):
      if ((abs(gold[EV_TIME_COL] - guess[EV_TIME_COL]) <= DELTA_TIME)
          and (dist_deg(gold[:2], guess[:2]) <= DELTA_DIST)):
        edges.append((goldnum, len(gold_events)+guessnum,
                      -dist_deg(gold[:2], guess[:2])))
  
  mat = mwmatching.maxWeightMatching(edges, maxcardinality=True)
  indices = []
  for i in range(len(gold_events)):
    if i < len(mat) and mat[i] >= 0:
      assert(mat[i] >= len(gold_events))
      indices.append((i, mat[i] - len(gold_events)))
  
  return indices

def find_true_false_guess(gold_events, guess_events):
  mat = find_matching(gold_events, guess_events)
  true = [j for (i,j) in mat]
  true_set = set(true)
  false = [i for i in range(len(guess_events)) if i not in true_set]

  return true, false, mat

def find_unmatched(gold_events, guess_events):
  mat = find_matching(gold_events, guess_events)
  mat_gold = set()
  for i,j in mat:
    mat_gold.add(i)
  return [x for x in range(len(gold_events)) if x not in mat_gold]

def f1_and_error(gold_events, guess_events):
  indices = find_matching(gold_events, guess_events)
  # compute precision
  if len(guess_events):
    p = 100. * float(len(indices)) / len(guess_events)
  else:
    p = 100.
  # compute recall
  if len(gold_events):
    r = 100. * float(len(indices)) / len(gold_events)
  else:
    r = 100.
  # compute f1
  if p==0 or r==0:
    f = 0.
  else:
    f = 2 * p * r / (p + r)
  # compute avg error
  if len(indices):
    errs = np.array([dist_km((gold_events[i,EV_LON_COL],
                              gold_events[i,EV_LAT_COL]),
                                  (guess_events[j, EV_LON_COL],
                                   guess_events[j, EV_LAT_COL]))
                     for (i,j) in indices])
    err = (np.average(errs), np.std(errs))
  else:
    err = (0.,0.)

  return f, p, r, err
