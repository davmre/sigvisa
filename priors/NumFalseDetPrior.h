typedef struct NumFalseDetPrior_t
{
  int numsites;

  double * falserates;                       /* one per site */
  double * logfalserates;

} NumFalseDetPrior_t;

#define NumFalseDet_TimeRate(prior, siteid) ((prior)->falserates[(siteid)])

#define NumFalseDet_LogTimeRate(prior, siteid) \
((prior)->logfalserates[(siteid)])

void NumFalseDetPrior_Init_Params(NumFalseDetPrior_t * prior,
                                  const char * filename);

double NumFalseDetPrior_LogProb(const NumFalseDetPrior_t * prior,
                                int numfalse, int siteid);


void NumFalseDetPrior_UnInit(NumFalseDetPrior_t * prior);
