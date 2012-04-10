# convert a UTC date to a UNIX timestamp 
import sys, time, calendar

def main():
  if len(sys.argv) != 3:
    print "Usage: python -m utils.date2time YYYY-MM-DD HH:MI:SS"
    sys.exit(1)

  timetuple = time.strptime(" ".join(sys.argv[1:]), "%Y-%m-%d %H:%M:%S")

  print calendar.timegm(timetuple)
  
if __name__ == "__main__":
  main()
