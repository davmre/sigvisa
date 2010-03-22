import os

from database.dataset import *

import priors.NumEventPrior
import priors.EventLocationPrior

def main(config_dirname):
  start_time, end_time, leb_events = read_events("training")
  
  priors.NumEventPrior.learn(os.path.join("parameters", "NumEventPrior.txt"),
                             start_time, end_time, leb_events)
  
  priors.EventLocationPrior.learn(os.path.join("parameters",
                                               "EventLocationPrior.txt"),
                                  leb_events)
  


if __name__ == "__main__":
  main("config")
  
