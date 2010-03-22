import os

from database.dataset import *

import priors.NumEventPrior
import priors.EventLocationPrior

def main(param_dirname):
  start_time, end_time, leb_events = read_events("training")
  
  priors.NumEventPrior.learn(os.path.join(param_dirname, "NumEventPrior.txt"),
                             start_time, end_time, leb_events)
  
  priors.EventLocationPrior.learn(os.path.join(param_dirname,
                                               "EventLocationPrior.txt"),
                                  leb_events)
  


if __name__ == "__main__":
  main("parameters")
  
