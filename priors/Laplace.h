#define ABS(x) ((x) > 0 ? (x) : -(x))

#define Laplace_ldensity(location, scale, val) \
  (- log(2 * (scale)) - ABS((val) - (location)) / (scale))

#define Laplace_trunc_ldensity(location, scale, minval, maxval, val)     \
  (- log((scale) * (2 - exp(-((location)-(minval))/(scale))             \
                    - exp(-((maxval)-(location))/(scale))))             \
   - ABS((val) - (location)) / (scale))

