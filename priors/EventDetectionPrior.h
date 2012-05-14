
#define NUM_EDA_COEFFS 15

#define EDA_COEFF_INTERCEPT 0
#define EDA_COEFF_MAG       1
#define EDA_COEFF_DEPTH     2
#define EDA_COEFF_DIST      3
#define EDA_COEFF_DIST0     4
#define EDA_COEFF_DIST35    5
#define EDA_COEFF_DIST40    6
#define EDA_COEFF_DIST12520 7
#define EDA_COEFF_DIST12540 8
#define EDA_COEFF_DIST145   9
#define EDA_COEFF_DIST170  10
#define EDA_COEFF_DIST175  11
#define EDA_COEFF_MAG6     12
#define EDA_COEFF_MAG68    13
#define EDA_COEFF_MD       14

typedef struct EventDetectionPrior_t
{
  int numsites;
  int numtimedefphases;
  double * p_coeffs; /* numsites x numtimedefphases x NUM_EDA_COEFFS */
} EventDetectionPrior_t;

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * dist,
                                     const char * filename);

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * prior,
                                   int is_detected,
                                   double evdepth, double evmag, double dist,
                                   int siteid, int phaseid);

void EventDetectionPrior_UnInit(EventDetectionPrior_t * dist);
