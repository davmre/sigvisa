# convert the isc bulletin into a csv file
# with the following schema:
#
# create table isc_events (
#  eventid     int not null,
#  location    varchar(100) not null,
#  author      varchar(10) not null,
#  origid      int not null,
#  lon         double not null,
#  lat         double not null,
#  smaj        double not null,
#  smin        double not null,
#  depth       double not null,
#  deptherr    double not null,
#  time        double not null,
#  timeerr     double not null,
#  ndef        int,
#  nsta        int,
#  gap         double,
# 
#  primary key (eventid, author),
#  index (time),
#  index (author, time)
# ) engine = myisam;
# 

import sys, calendar

FEET2KM = 0.0003048

def INT(string):
  string = string.strip()
  if not len(string):
    return -999
  else:
    return int(string)

def main():
  if len(sys.argv) != 2:
    print "Error: Usage python isc2csv.py <ISC-bulletin>"
    sys.exit(1)

  in_fname = sys.argv[1]
  out_fname = in_fname+".csv"

  in_fp = open(in_fname)
  out_fp = open(out_fname, "w")

  print "Converting %s to %s" % (in_fname, out_fname)

  print >> out_fp, "eventid,region,author,lon,lat,depth,time,mb,ndef,nsta,gap"
  UNKNOWN, READ_ORIGINS, READ_MAGS, READ_DETS = range(4)
  
  state = UNKNOWN

  # initially, we have no event-id
  eventid, region, origins = None, None, None

  orig_read_cnt = 0
  orig_write_cnt = 0
  
  for line in in_fp:
    line = line.rstrip()

    # when we come to a new event or at the end of the file
    # flush out any saved origins if their corresponding magnitude is not found
    if (line.startswith("Event") or line.startswith("STOP")) \
           and origins is not None:
      for (eventid, region, author, longitude, latitude, depth, tim, ndef,
           nsta, gap) in origins.itervalues():
        mb = -999.0
        print >> out_fp, '%s,"%s",%s,%f,%f,%f,%f,%f,%d,%d,%d' \
              % (eventid, region, author, longitude, latitude, depth,
                 tim, mb, ndef, nsta, gap)
        orig_write_cnt += 1
      state = READ_DETS
      origins = None
      
      
    # read an Event line
    if line.startswith("Event"):
      if orig_read_cnt != orig_write_cnt:
        import pdb
        pdb.set_trace()
        
      # parse out the event-id...
      eventid_end_idx = line.index(" ", 6)
      eventid = line[6:eventid_end_idx]
      # ... and the region
      region = line[eventid_end_idx+1:]
      # prepare to read origins from authors
      origins = {}
      
      state = READ_ORIGINS
      continue
        

    # read the origin line
    if state == READ_ORIGINS and line[:4].isdigit():
      year = int(line[:4])
      mon = int(line[5:7])
      day = int(line[8:10])
      hr = int(line[11:13])
      mi = int(line[14:16])
      sec = int(line[17:19])
      fracsecstr = "0." + line[20:22].strip()
      fracsec = float(fracsecstr)

      tim = calendar.timegm((year, mon, day, hr, mi, sec)) + fracsec
        
      latitude = float(line[36:44])
      longitude = float(line[45:54])
      
      # read depth
      depthstr = line[71:77].strip()
      # convert depth from feet to km if necessary
      if depthstr.endswith("f"):
        depth = FEET2KM * float(depthstr[:-1])
      elif not len(depthstr):
        depth = -999.0
      else:
        depth = float(depthstr)

      ndef = INT(line[83:87])
      nsta = INT(line[88:92])
      gap = INT(line[93:96])
      
      author = line[118:124].rstrip()
      origid = int(line[125:136])
      
      origins[origid] = (eventid, region, author, longitude, latitude,
                         depth, tim, ndef, nsta, gap)
      orig_read_cnt += 1
      continue
    
    if line.startswith("Magnitude"):
      state = READ_MAGS
      continue

    if state == READ_MAGS and line.startswith("mb "):
      mb = float(line[7:10])
      origid = int(line[27:38])
      if origid in origins:
        (eventid, region, author, longitude, latitude, depth, tim, ndef, nsta,
         gap) = origins[origid]
        del origins[origid]
        
        print >> out_fp, '%s,"%s",%s,%f,%f,%f,%f,%f,%d,%d,%d' \
              % (eventid, region, author, longitude, latitude, depth,
                 tim, mb, ndef, nsta, gap)
        orig_write_cnt += 1
        continue

  in_fp.close()
  out_fp.close()

  print "Read %d origs, Written %d origs" % (orig_read_cnt, orig_write_cnt)
  
if __name__ == "__main__":
  try:
    main()
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
  
