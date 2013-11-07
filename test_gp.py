import numpy as np

from sigvisa.models.spatial_regression.SparseGP import SparseGP


# for a geographic kernel:
# input points are of the form (lon, lat, depth, dist, azi),
# where dist and azi optionally refer to the distance and azimuth
# between this location and some reference location (generally the
# site of the seismic station detecting an event).
# all distances/depths are in km.
# note the GP really only needs (lon, lat, depth), dist and azi are
# used as part of a parametric model that we combine with the GP
# (as in section 2.7 of R&W).
X = np.array([
    [120, 30, 0, ],
    [118, 31, 0, ],
    [120, 29, 40,],
    [110, 30, 20,],
], dtype=float)
y = np.array([
    -0.02,
    -0.01,
    -0.015,
    -0.005,
])

testX = np.array([[120, 30, 0],[119, 31, 0] ], dtype=float)

# our kernel is of the form
# w( d(x, x')  )
# where d is a distance function, default 'lld' is
# geographic distance for x represented as (lon, lat, depth),
# and w is a weight function, default 'se' is the squared
# exponential kernel exp(-d^2).

# the kernel is described by some hyperparameters:
# hparams[0] is the noise variance
# hparams[1] is the signal variance
# hparams[2:] are parameters of the distance function,
#             specifically the lengthscale of typical variation
#             wrt horizontal distance and depth, respectively.
# so the full kernel looks like
# hparams[0] * I(x, x') + hparams[1]*exp(-d(x, x')^2)
# where d(x, x') = sqrt( horiz_distance(lat1, lon1, lat2, lon2)^2/hparams[2]^2 + (depth2-depth1)^2/hparams[3]^2    )
# there's also a dfn "euclidean", which computes standard
# euclidean distances; for a d-dimensional space it uses
# hparams[2:(d+2)] as the lengthscale in each dimension.

hparams = [.01, .03, 200.0, 400.0]

# train a Gaussian process on data (X,y) with given hyperparams
gp = SparseGP(X=X, y=y, hyperparams=hparams, dfn_str="lld", wfn_str="se")

predictions = gp.predict(testX)
cov = gp.covariance(testX, include_obs=True)

print predictions
print cov
# should print
# predictions = [-0.01617128 -0.0114027 ]
# cov = [[ 0.01628304  0.00360742]
#        [ 0.00360742  0.02163215]]
