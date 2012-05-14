from utils import LogNormal

from database.dataset import *

NUM_PRIOR = 10                          # number of prior points

def learn(param_filename, options, earthmodel, detections, leb_events,
          leb_evlist):
  # separate the true and false detections
  true_dets = set()
  for detlist in leb_evlist:
    for phase, detnum in detlist:
      true_dets.add(detnum)
  
  false_dets = set(i for i in range(len(detections)) if i not in true_dets)

  # learn the overall distribution of SNR and sample some points for
  # an empirical Bayes prior
  true_mean, true_sigma = LogNormal.estimate([detections[detnum, DET_SNR_COL]
                                              for detnum in true_dets])

  false_mean, false_sigma = LogNormal.estimate([detections[detnum, DET_SNR_COL]
                                                for detnum in false_dets])
  

  true_prior = [LogNormal.sample(true_mean, true_sigma)
                for _ in xrange(NUM_PRIOR)]
  false_prior = [LogNormal.sample(false_mean, false_sigma)
                 for _ in xrange(NUM_PRIOR)]

  if options.verbose:
    print "True SNR: Overall mean, sigma:", true_mean, true_sigma
    print "False SNR: Overall mean, sigma:", false_mean, false_sigma
    
  site_true = dict((siteid, []) for siteid in range(earthmodel.NumSites()))
  site_false = dict((siteid, []) for siteid in range(earthmodel.NumSites()))
  
  for detnum in true_dets:
    site_true[int(detections[detnum, DET_SITE_COL])].append(detections[detnum,
                                                                DET_SNR_COL])
  
  for detnum in false_dets:
    site_false[int(detections[detnum, DET_SITE_COL])].append(detections[detnum,
                                                                DET_SNR_COL])

  if options.verbose:
    print "Site specific SNR true mean, sigma, false mean, sigma"

  fp = open(param_filename, "w")

  print >>fp, earthmodel.NumSites()
  
  for siteid in range(earthmodel.NumSites()):
    site_true_mean, site_true_sigma = LogNormal.estimate(site_true[siteid]
                                                         + true_prior)
    
    site_false_mean, site_false_sigma = LogNormal.estimate(site_false[siteid]
                                                           + false_prior)

    if options.verbose:
      print "[%d]: True %.2f +- %.2f  False %.2f +- %.2f" \
            % (siteid, site_true_mean, site_true_sigma, site_false_mean,
               site_false_sigma)
    
    print >>fp, site_true_mean, site_true_sigma, site_false_mean, \
          site_false_sigma
  
  fp.close()
