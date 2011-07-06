
typedef struct PhaseAmp_t
{
  double intercept;
  double mb_coeff;
  double depth_coeff;
  double ttime_coeff;

  double std;                                /* standard deviation */
  
} PhaseAmp_t;

typedef struct ArrivalAmplitudePrior_t
{
  int numsites;
  int numphases;
  double min_logamp;
  double max_logamp;
  double step_logamp;
  int numstep;
  
  double * p_site_false;                     /* numsites x numstep */
  PhaseAmp_t * p_site_phase_amp;             /* numsites x numphases */

} ArrivalAmplitudePrior_t;

void ArrivalAmplitudePrior_Init_Params(ArrivalAmplitudePrior_t * prior,
                                       const char * filename);

double ArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * prior,
                                     double mb, double depth, double ttime,
                                     int siteid, int phaseid, double amp);

double FalseArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * 
                                          prior, int siteid, double amplitude);

double FalseArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * 
                                      prior, int siteid, double amplitude);

void ArrivalAmplitudePrior_UnInit(ArrivalAmplitudePrior_t * prior);
