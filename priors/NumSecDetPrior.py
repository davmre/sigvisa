def learn(param_fname, leb_seclist):
  cnt = 0
  tot = 0
  for seclist in leb_seclist:
    for dets in seclist:
      cnt += 1
      tot += len(dets) - 2

  rate = float(tot) / cnt
  print "secondary detection rate is %.3f per primary detection" % rate

  fp = open(param_fname, "w")
  
  print >>fp, rate
  
  fp.close()

