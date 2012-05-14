#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../sigvisa.h"

void EventMagPrior_Init_Params(EventMagPrior_t * prior, int nparams,
                               const char * filename)
{
  FILE * fp;
  
  assert(1 == nparams);

  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open %s\n", filename);
    exit(1);
  }
  
  if (2 != fscanf(fp, "%lf %lf\n", &prior->min_mag, &prior->mag_rate))
  {
    fprintf(stderr, "Unable to read min magnitude and magrate in %s\n", 
            filename);
    exit(1);
  }

  fclose(fp);
}


double EventMagPrior_LogProb(const EventMagPrior_t * dist, double val,
                             int nargs)
{
  assert(0 == nargs);

  if (val < dist->min_mag)
    return -HUGE_VAL;
 
  return log(dist->mag_rate) - dist->mag_rate * (val - dist->min_mag);
}

