from database.dataset import *
import priors.NumEventPrior

def main(config_dirname):
  start_time, end_time, leb_events = read_events("training")
  
  priors.NumEventPrior.learn(config_dirname, start_time, end_time, leb_events)
  


if __name__ == "__main__":
  main("config")
  
