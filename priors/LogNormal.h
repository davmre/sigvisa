
#define LogNormal_ldensity(mean, sigma, val)\
 (- .5 * (log(val) - (mean)) * (log(val) - (mean)) / ((sigma) * (sigma)) \
 - .5 * log(2 * PI * (sigma) * (sigma)) - log(val))
