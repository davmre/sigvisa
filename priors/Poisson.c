#include <math.h>

/* log(n!) for n > 0 */
static double log_fact(int n)
{
  int i;
  double sum;
  
  sum = 0;
  for (i=2; i<=n; i++)
    sum += log((double) i);
  return sum;
}


double Poisson_logprob(double rate, int val)
{
  double ans;
  
  
  if (val < 0)
    return -HUGE_VAL;

  ans = -rate + val * log(rate) - log_fact(val);

  return ans;
}

