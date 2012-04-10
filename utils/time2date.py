# converts a unix timestamp to a data in UTC
import sys, time

def main():
  if len(sys.argv) != 2:
    print "Usage: python -m utils.time2date timestamp"
    sys.exit(1)

  timetuple = time.gmtime(int(sys.argv[1]))
  print time.strftime("%Y-%m-%d %H:%M:%S", timetuple)
  
if __name__ == "__main__":
  main()
