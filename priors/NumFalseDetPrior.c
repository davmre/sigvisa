#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"


void NumFalseDetPrior_Init_Params(NumFalseDetPrior_t * prior,
                                  const char * filename)
{
  FILE * fp;
  int siteid;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (1 != fscanf(fp, "%d\n", &prior->numsites))
  {
    fprintf(stderr, "error reading num sites from %s\n", filename);
    exit(1);
  }

  prior->falserates = (double *)calloc(prior->numsites,
                                       sizeof(*prior->falserates));

  prior->logfalserates = (double *)calloc(prior->numsites,
                                          sizeof(*prior->logfalserates));

  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    if (1 != fscanf(fp, "%lg\n", &prior->falserates[siteid])) 
    {
      fprintf(stderr, "NumFalseDetPrior: can't read false rate for siteid %d"
              " in file %s", siteid, filename);
      exit(1);
    }

    prior->logfalserates[siteid] = log(prior->falserates[siteid]);
  }
  
  fclose(fp);
}

double NumFalseDetPrior_LogProb(const NumFalseDetPrior_t * prior,
                                int numfalse, int siteid)
{
  assert(siteid < prior->numsites);
  
  return Poisson_logprob(prior->falserates[siteid], numfalse);
}

void NumFalseDetPrior_UnInit(NumFalseDetPrior_t * prior)
{
  free(prior->falserates);
  free(prior->logfalserates);
}

