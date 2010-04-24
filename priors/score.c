#include <math.h>
#include <stdlib.h>

#include "../netvisa.h"

/* returns 0 if the event - phase is impossible to be detected at the
 * site, 1 otherwise 
 * if detected returns log probability of detection given event 
 *                     - log probability noise 
 * else returns log probability of mis-detection given event */
static int score_event_site_phase_int(NetModel_t * p_netmodel,
                                      const Event_t * p_event,
                                      int siteid, int phaseid,
                                      double distance, double pred_az,
                                      int * p_detcnt,
                                      double * p_detsc, 
                                      double * p_dettimesc,
                                      double * p_detslosc,
                                      double * p_detazsc,
                                      double * p_detphasesc,
                                      double * p_snrsc)
{
  EarthModel_t * p_earth;
  double pred_arrtime;
  int detnum;
  int numtimedefphases;

  p_earth = p_netmodel->p_earth;
  
  assert(EarthModel_IsTimeDefPhase(p_earth, phaseid));
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                        p_event->evlat, p_event->evdepth,
                                        p_event->evtime, phaseid,
                                        siteid);

  /* check if the site is in the shadow zone for the event - phase */
  if (pred_arrtime < 0)
    return 0;

  /* check if the site is up */
  if (!NetModel_IsSiteUp(p_netmodel, siteid, pred_arrtime))
    return 0;
  
  detnum = p_event->p_detids[siteid * numtimedefphases + phaseid];
        
  if (detnum != -1)
  {
    double pred_slow;
    Detection_t * det;

    det = p_netmodel->p_detections + detnum;

    *p_detcnt += 1;

          
    *p_dettimesc += ArrivalTimePrior_LogProb(&p_netmodel->arr_time_prior,
                                             det->time_det, pred_arrtime,
                                             det->deltim_det, siteid,
                                             phaseid);

    *p_dettimesc -= NumFalseDet_LogTimeRate(&p_netmodel
                                            ->num_falsedet_prior,
                                            siteid);

    pred_slow = EarthModel_ArrivalSlowness(p_earth, p_event->evlon,
                                           p_event->evlat,
                                           p_event->evdepth, phaseid,
                                           siteid);

    *p_detslosc += ArrivalSlownessPrior_LogProb(&p_netmodel
                                                ->arr_slo_prior,
                                                det->slo_det, 
                                                pred_slow,
                                                det->delslo_det,
                                                siteid, phaseid);
    *p_detslosc -= LOGPROB_UNIFORM_SLOWNESS;

    *p_detazsc += ArrivalAzimuthPrior_LogProb(&p_netmodel->arr_az_prior,
                                              det->azi_det, 
                                              pred_az,
                                              det->delaz_det,
                                              siteid, phaseid);
    *p_detazsc -= LOGPROB_UNIFORM_AZIMUTH;

    *p_detphasesc += ArrivalPhasePrior_LogProb(&p_netmodel
                                               ->arr_phase_prior,
                                               det->phase_det,
                                               phaseid);
          
    *p_detphasesc -= FalseArrivalPhasePrior_LogProb(&p_netmodel
                                                    ->arr_phase_prior,
                                                    det->phase_det);

#ifdef VISA_SCORE_SNR
    *p_snrsc += ArrivalSNRPrior_LogProb(&p_netmodel->arr_snr_prior,
                                        det->site_det, det->snr_det);
    
    *p_snrsc -= FalseArrivalSNRPrior_LogProb(&p_netmodel->arr_snr_prior,
                                             det->site_det, det->snr_det);
#endif

  }

  *p_detsc += EventDetectionPrior_LogProb(&p_netmodel->event_det_prior,
                                          detnum == -1? 0 : 1,
                                          p_event->evdepth,
                                          p_event->evmag,
                                          distance, siteid, phaseid);

  return 1;
}

/* returns 0 if the event can't be detected at the site */
int score_event_site_phase(NetModel_t * p_netmodel,
                           const Event_t * p_event,
                           int siteid, int phaseid,
                           double distance, double pred_az,
                           double * p_score)
{
  int      poss;
  int      detcnt;
  double   detsc;
  double   dettimesc;
  double   detslosc;
  double   detazsc;
  double   detphasesc;
  double   detsnrsc;

  detcnt = 0;
  detsc = dettimesc = detslosc = detazsc = detphasesc = detsnrsc = 0;
  
  poss = score_event_site_phase_int(p_netmodel, p_event, siteid, phaseid,
                                    distance, pred_az, &detcnt, &detsc,
                                    &dettimesc, &detslosc, &detazsc,
                                    &detphasesc, &detsnrsc);

  *p_score = detsc + dettimesc + detslosc + detazsc + detphasesc + detsnrsc;

  return poss;
}


static void score_event_int(NetModel_t * p_netmodel, Event_t * p_event, 
                 double * p_numsc, double * p_locsc, double * p_magsc,
                 double * p_detsc, double * p_dettimesc,
                 double * p_detazsc, double * p_detslosc,
                 double * p_detphasesc, double * p_detsnrsc,
                 int * p_poss_detcnt, int * p_detcnt)
{
  EarthModel_t * p_earth;
  Detection_t * p_detections;
  int numtimedefphases;
  int numsites;
  int siteid;
  int phaseid;

  p_earth = p_netmodel->p_earth;
  p_detections = p_netmodel->p_detections;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  *p_numsc += NumEventPrior_LogTimeRate(&p_netmodel->num_event_prior);
  
  *p_locsc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  *p_magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
                                    p_event->evmag, 0);

  for (siteid=0; siteid<numsites; siteid++)
  {
    double distance, pred_az;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                siteid);
    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat, siteid);
    
    for (phaseid=0; phaseid<numtimedefphases; phaseid++)
    {
      *p_poss_detcnt += score_event_site_phase_int(p_netmodel, p_event,
                                                   siteid, phaseid,
                                                   distance, pred_az,
                                                   p_detcnt,
                                                   p_detsc, p_dettimesc,
                                                   p_detslosc, p_detazsc,
                                                   p_detphasesc,
                                                   p_detsnrsc);   
    }
  }
}

double score_event(NetModel_t * p_netmodel, Event_t * p_event)
{
  double numsc, locsc, magsc, detsc, dettimesc, detazsc, detslosc, detphasesc;
  double detsnrsc;
  double score;
  int possdetcnt, detcnt;

  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = detsnrsc = 0;
  possdetcnt = detcnt = 0;
  
  score_event_int(p_netmodel, p_event, &numsc, &locsc, &magsc, &detsc, 
                  &dettimesc, &detazsc, &detslosc, &detphasesc, &detsnrsc,
                  &possdetcnt, &detcnt);
  
  score = numsc + locsc + magsc + detsc + dettimesc + detazsc + detslosc
    + detphasesc + detsnrsc;

  return score;
}

double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events, int verbose)
{
  double score;
  double numsc;
  double magsc;
  double locsc;
  double detsc;
  double dettimesc;
  double detazsc;
  double detslosc;
  double detphasesc;
  double detsnrsc;
  
  int poss_detcnt;
  int detcnt;
  
  int i;

  if (verbose)
  {
    printf ("%d events:\n", numevents);
  }
  
  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = detsnrsc = 0;
  poss_detcnt = detcnt = 0;
  
  for (i=0; i<numevents; i++)
  {
    score_event_int(p_netmodel, p_events + i, &numsc, &locsc, &magsc, &detsc,
                    &dettimesc, &detazsc, &detslosc, &detphasesc,
                    &detsnrsc,
                    &poss_detcnt, &detcnt);
  }
  
  if (verbose)
  {
    printf("Event #,Time: score %lf, avg %lf\n", numsc, numsc/numevents);
    printf("Event Location: score %lf, avg %lf\n", locsc, locsc/numevents);
    printf("Event Mag: score %lf, avg %lf\n", magsc, magsc/numevents);
    printf("Event Det: score %lf, avg-event %lf avg-det %lf\n", detsc, 
           detsc/numevents, detsc/poss_detcnt);
    printf("Det Time: score %lf, avg-event %lf avg-det %lf\n", dettimesc, 
           dettimesc/numevents, dettimesc/detcnt);
    printf("Det Azimuth: score %lf, avg-event %lf avg-det %lf\n", detazsc, 
           detazsc/numevents, detazsc/detcnt);
    printf("Det Slowness: score %lf, avg-event %lf avg-det %lf\n", detslosc, 
           detslosc/numevents, detslosc/detcnt);
    printf("Det Phase: score %lf, avg-event %lf avg-det %lf\n", detphasesc, 
           detphasesc/numevents, detphasesc/detcnt);
    printf("Det SNR: score %lf, avg-event %lf avg-det %lf\n", detsnrsc, 
           detsnrsc/numevents, detsnrsc/detcnt);
    printf("Avg. # Detections: Possible %lf, Actual %lf\n", 
           ((double) poss_detcnt) / ((double) numevents),
           ((double) detcnt) / ((double) numevents));
  }
  
  score = numsc + locsc + magsc + detsc + dettimesc + detslosc + detazsc +
    detphasesc + detsnrsc;
  
  if (verbose)
    printf("Total: %lf Avg. %lf\n", score, score/numevents);
  
  return score;
}
