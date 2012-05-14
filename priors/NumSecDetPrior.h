typedef struct NumSecDetPrior_t
{
  double num_secdet_rate;
  double log_num_secdet_rate;
} NumSecDetPrior_t;

#define NumSecDetPrior_LogRate(prior) ((prior)->log_num_secdet_rate)

void NumSecDetPrior_Init_Params(NumSecDetPrior_t * prior,
                                const char * filename);

double NumSecDetPrior_LogProb(const NumSecDetPrior_t * dist, int val);
