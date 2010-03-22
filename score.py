import os

from database.dataset import *
import netvisa

def main(param_dirname):
  start_time, end_time, leb_events = read_events("validation")

  model = netvisa.NetModel(start_time, end_time,
                           os.path.join(param_dirname, "NumEventPrior.txt"))

  print "Score", netvisa.score_world(model, leb_events)

if __name__ == "__main__":
  main("parameters")
