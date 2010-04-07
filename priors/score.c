#include <math.h>
#include <stdlib.h>

#include "../netvisa.h"

void score_event(NetModel_t * p_netmodel, Event_t * p_event, 
                 double * p_numsc, double * p_locsc, double * p_magsc,
                 double * p_detsc, double * p_dettimesc,
                 double * p_detazsc, double * p_detslosc,
                 double * p_detphasesc,
                 int * p_poss_detcnt, int * p_detcnt)
{
  EarthModel_t * p_earth;
  Detection_t * p_detections;
  int numphases;
  int numsites;
  int siteid;
  int phaseid;

  p_earth = p_netmodel->p_earth;
  p_detections = p_netmodel->p_detections;
  
  numsites = p_netmodel->event_det_prior.numsites;
  numphases = p_netmodel->event_det_prior.numphases;

  *p_numsc += NumEventPrior_LogTimeRate(&p_netmodel->num_event_prior);
    
  *p_locsc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  *p_magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
                                    p_event->evmag, 0);

  for (siteid=0; siteid<numsites; siteid++)
  {
    double distance;
    
    /* check if the site is up */
    if (!NetModel_IsSiteUp(p_netmodel, siteid, p_event->evtime)
        || !NetModel_IsSiteUp(p_netmodel, siteid,
                              p_event->evtime + UPTIME_QUANT))
      continue;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                siteid);
    
    for (phaseid=0; phaseid<numphases; phaseid++)
    {
      /* check if the site is in the shadow zone for the event */
      if (EarthModel_IsTimeDefPhase(p_earth, phaseid))
      {
        double pred_arrtime, pred_slow, pred_az;
        int detnum;
        
        pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                              p_event->evlat, p_event->evdepth,
                                              p_event->evtime, phaseid,
                                              siteid);

        pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                            p_event->evlat, siteid);

        pred_slow = EarthModel_ArrivalSlowness(p_earth, p_event->evlon,
                                               p_event->evlat,
                                               p_event->evdepth, phaseid,
                                               siteid);

        if (pred_arrtime < 0)
          continue;
        
        detnum = p_event->p_detids[siteid * numphases + phaseid];
        
        *p_poss_detcnt += 1;
        
        *p_detsc += EventDetectionPrior_LogProb(&p_netmodel->event_det_prior,
                                                detnum == -1? 0 : 1,
                                                p_event->evdepth, 
                                                p_event->evmag,
                                                distance, siteid, phaseid);

        if (detnum != -1)
        {
          Detection_t * det;

          *p_detcnt += 1;

          det = p_detections + detnum;
          
          *p_dettimesc += ArrivalTimePrior_LogProb(&p_netmodel->arr_time_prior,
                                                   det->time_det, pred_arrtime,
                                                   det->deltim_det, siteid,
                                                   phaseid);

          *p_dettimesc -= NumFalseDet_LogTimeRate(&p_netmodel
                                                  ->num_falsedet_prior,
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
        }
      }
    }
  }
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
  
  int poss_detcnt;
  int detcnt;
  
  int i;

  if (verbose)
  {
    printf ("%d events:\n", numevents);
  }
  
  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = 0;
  poss_detcnt = detcnt = 0;
  
  for (i=0; i<numevents; i++)
  {
    score_event(p_netmodel, p_events + i, &numsc, &locsc, &magsc, &detsc,
                & dettimesc, &detazsc, &detslosc, &detphasesc,
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
    printf("Avg. # Detections: Possible %lf, Actual %lf\n", 
           ((double) poss_detcnt) / ((double) numevents),
           ((double) detcnt) / ((double) numevents));
  }
  
  score = numsc + locsc + magsc + detsc + dettimesc + detslosc + detazsc +
    detphasesc;
  
  if (verbose)
    printf("Total: %lf Avg. %lf\n", score, score/numevents);
  
  return score;
}
