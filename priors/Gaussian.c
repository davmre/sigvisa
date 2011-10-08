#include "../sigvisa.h"

#include <math.h>

#ifndef M_2_SQRTPI
 #define M_2_SQRTPI	((double) 1.12837916709551257390)
#endif


static double std_cdf(double x);
static double std_GL(double a, double b);
static double std_f(double x);


double Gaussian_prob(double val, double m, double s)
{
  return exp(- (val - m) * (val - m) / (2.0 * s * s)) / sqrt(2.0 * PI * s * s);
}

double Gaussian_logprob(double val, double m, double s)
{
  return - (val - m) * (val - m) / (2.0 * s * s) - .5 * log( 2.0 * PI * s * s);
}


double Gaussian_sample(double m, double s)
/* ========================================================================
 * Returns a normal (Gaussian) distributed real number.
 * NOTE: use s > 0.0
 *
 * Uses a very accurate approximation of the normal idf due to Odeh & Evans, 
 * J. Applied Statistics, 1974, vol 23, pp 96-97.
 * ========================================================================
 * copied from http://www.cs.wm.edu/~va/software/park/
 */
{ 
  const double p0 = 0.322232431088;     const double q0 = 0.099348462606;
  const double p1 = 1.0;                const double q1 = 0.588581570495;
  const double p2 = 0.342242088547;     const double q2 = 0.531103462366;
  const double p3 = 0.204231210245e-1;  const double q3 = 0.103537752850;
  const double p4 = 0.453642210148e-4;  const double q4 = 0.385607006340e-2;
  double u, t, p, q, z;

  u = RAND_DOUBLE;
  if (u < 0.5)
    t = sqrt(-2.0 * log(u));
  else
    t = sqrt(-2.0 * log(1.0 - u));
  p   = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)));
  q   = q0 + t * (q1 + t * (q2 + t * (q3 + t * q4)));
  if (u < 0.5)
    z = (p / q) - t;
  else
    z = t - (p / q);

  return (m + s * z);
}

double Gaussian_cdf(double val, double m, double s)
{
  return std_cdf((val - m) / s);
}

/* http://129.3.20.41/eps/comp/papers/0212/0212001.pdf */

/* cdf of a standard normal */
static double std_cdf(double x)
{
  if(x>=0.)
  {
    return (1. + std_GL(0, x / sqrt(2.)))/2.;
  }
  else 
  {
    return (1. - std_GL(0, -x / sqrt(2.)))/2.;
  }
}
/* Integration on a closed interval */
static double std_GL(double a, double b)
{
  double y1=0, y2=0, y3=0, y4=0, y5=0;
  double x1=-sqrt(245.+14.*sqrt(70.))/21., x2=-sqrt(245. -
                                                         14.*sqrt(70.))/21.;
  double x3=0, x4=-x2, x5=-x1;
  double w1=(322.-13.*sqrt(70.))/900., w2=(322.+13.*sqrt(70.))/900.;
  double w3=128./225.,w4=w2,w5=w1;
  int n=4800;
  double i=0, s=0, h=(b-a)/n;
   for (i=0;i<=n;i++)
   {
     y1=h*x1/2.+(h+2.*(a+i*h))/2.;
     y2=h*x2/2.+(h+2.*(a+i*h))/2.;
     y3=h*x3/2.+(h+2.*(a+i*h))/2.;
     y4=h*x4/2.+(h+2.*(a+i*h))/2.;
     y5=h*x5/2.+(h+2.*(a+i*h))/2.;
     s=s+h*(w1*std_f(y1)+w2*std_f(y2)+w3*std_f(y3)
            + w4*std_f(y4) + w5*std_f(y5))/2.;
   }
   return s;
}

/* Function f, to integrate */
static double std_f(double x)
{
  return (M_2_SQRTPI) * exp(-x * x);
}
