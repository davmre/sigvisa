typedef struct NumEventPrior_t
{
  double num_event_rate;
  double log_num_event_rate;
} NumEventPrior_t;

#define NumEventPrior_LogTimeRate(prior) ((prior)->log_num_event_rate)

void NumEventPrior_Init_Params(NumEventPrior_t * prior, const char * filename);

double NumEventPrior_LogProb(const NumEventPrior_t * dist, int val,
                             double time_interval);
