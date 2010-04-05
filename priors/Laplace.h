#define ABS(x) ((x) > 0 ? (x) : -(x))

#define Laplace_ldensity(location, scale, val) \
  (- log(2 * (scale)) - ABS((val) - (location)) / (scale))

