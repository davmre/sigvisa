import numpy as np
import urllib2
import time

#orids = np.loadtxt("filtered_orids")
#orids = np.arange(1, 2069)
orids = [2102, 2156, 2215, 2233, 2243, 2244, 2279]

for orid in orids:
    t0 = time.time()
    try:
        print "orid %d" % orid
        url = "http://kampos.banatao.berkeley.edu:8001/sigvisa/bulletin/%d/signals.html" % orid
        response = urllib2.urlopen(url)
        html = response.read()
        t1 = time.time()
        print "success in %.1fs" % (t1-t0)
    except Exception as e:
        print "failed: %s" % str(e)
    
