import numpy as np
import os

from sigvisa import Sigvisa
from sigvisa.utils.geog import dist_km, deg_to_km


def read_GA_boundaries(fname=None):
    if fname is None:
        s = Sigvisa()
        fname = os.path.join(s.homedir, "parameters", "GA_dist_depth_ranges")

    boundaries = {}
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split()
            phase = cols[0]
            min_dist, max_dist, min_depth, max_depth = cols[1:5]
            min_dist = deg_to_km(float(min_dist))
            max_dist = deg_to_km(float(max_dist))
            min_depth = float(min_depth)
            max_depth = float(max_depth)
            if phase not in boundaries:
                boundaries[phase] = [min_dist, max_dist, min_depth, max_depth, None, None, None]
            else:
                dist1, dist2, depth1, depth2, _, _, _ = boundaries[phase]
                dists = sorted(list(set((dist1, dist2, min_dist, max_dist))))
                assert ( len(dists) == 3)
                global_min_dist, split_dist, global_max_dist = dists
                if global_min_dist == min_dist:
                    min_depth_1, max_depth_1  = min_depth, max_depth
                    min_depth_2, max_depth_2 = depth1, depth2
                else:
                    min_depth_1, max_depth_1 = depth1, depth2
                    min_depth_2, max_depth_2  = min_depth, max_depth
                boundaries[phase] = [global_min_dist, global_max_dist, min_depth_1, max_depth_1, split_dist, min_depth_2, max_depth_2]
                    
    return boundaries

GA_boundaries = read_GA_boundaries()

class PhaseExistenceModel(object):


    def __init__(self, phase, boundaries=None, 
                 boundary_prob=0.01, half_lives=None,
                 max_prob = 0.99):
        if boundaries is None:
            boundaries = GA_boundaries[phase]
        if half_lives is None:
            half_lives = (40, 60, 5, 5)

        # semantics:
        # if split dist is None, then region is a rectangle defined by first four params
        # else, region is union of two rectangles defined by
        #   [min_dist, split_dist] x [min_depth, max_depth] and
        #   [split_dist, max_dist] x [min_depth2, max_depth2]
        # this is a total hack but it serves to represent the GA phase boundary list
        self.min_dist, self.max_dist, self.min_depth, self.max_depth, self.split_dist, self.min_depth2, self.max_depth2 = boundaries
        
        self.half_lives = half_lives
        self.boundary_prob = boundary_prob
        self.log_max_prob = np.log(max_prob)

        # locations of 50% probability
        origins = [self.min_dist + self.half_lives[0],
                   self.max_dist - self.half_lives[1],
                   self.min_depth + self.half_lives[2],
                   self.max_depth - self.half_lives[3]]
        if self.split_dist is not None:
            origins +=  [self.min_depth2 + self.half_lives[2], self.max_depth2 - self.half_lives[3]]

        z = np.log(1.0/self.boundary_prob -1)
        scales = (z / self.half_lives[0],
                  z / self.half_lives[1],
                  z / self.half_lives[2],
                  z / self.half_lives[3],
                  z / self.half_lives[2],
                  z / self.half_lives[3])
        self.scales, self.origins = scales, origins

    def _get_distance_depth(self, distance=None, depth=None, ev=None, site=None):
        if distance is None:
            depth = ev.depth
            s = Sigvisa()
            site_loc = s.earthmodel.site_info(site, ev.time)[:2]
            distance = dist_km((ev.lon, ev.lat), site_loc)

        return distance, depth

    def _in_bounds(self, distance, depth):

        if self.split_dist is None:
            if distance > self.min_dist and distance < self.max_depth and depth > self.min_depth and depth < self.max_depth:
                return True
        elif distance < self.split_dist:
            if distance > self.min_dist and depth > self.min_depth and depth < self.max_depth:
                return True
        else:
            if distance < self.max_dist and depth > self.min_depth2 and depth < self.max_depth2:
                return True
        return False
        
    
    def log_p(self, exists=True, **kwargs):
        distance, depth = self._get_distance_depth(**kwargs)


        #if not self._in_bounds(distance, depth):
        #    return -np.inf


        logit_min_dist = 0.0
        if self.min_dist > 0:
            logit_min_dist = -np.log(1.0 + np.exp(-self.scales[0] * (distance - self.origins[0])))
        logit_max_dist = -np.log(1+ np.exp(-self.scales[1] * (self.origins[1] - distance)))

        logit_min_depth = 0.0
        if self.min_depth > 0:
            logit_min_depth = -np.log(1+ np.exp(-self.scales[2] * (depth - self.origins[2])))
        logit_max_depth = -np.log(1+ np.exp(-self.scales[3] * (self.origins[3] - depth)))

        if self.split_dist is None:
            logit1 = logit_min_dist
            logit2 = logit_max_dist
            logit3 = logit_min_depth
            logit4 = logit_max_depth
        elif distance < self.split_dist:
            logit1 = logit_min_dist
            logit2 = 0.0
            logit3 = logit_min_depth
            logit4 = logit_max_depth
        else:
            logit_min_depth2 = 0.0
            if self.min_depth2 > 0:
                logit_min_depth2 = -np.log(1+ np.exp(-self.scales[4] * (depth - self.origins[4])))
            logit_max_depth2 = -np.log(1+ np.exp(-self.scales[5] * (self.origins[5] - depth)))

            logit1 = 0.0
            logit2 = logit_max_dist
            logit3 = logit_min_depth2
            logit4 = logit_max_depth2

        lp = logit1 + logit2 + logit3 + logit4 + self.log_max_prob        

        if not exists:
            lp = np.log(1.0 - np.exp(lp))

        return lp

    def sample(self, fix_result=None, **kwargs):
        p = np.exp(self.log_p(True, **kwargs))
        if fix_result is not None:
            r = fix_result
        else:
            r = np.random.rand() < p
        rp = p if r else 1-p
        return r, np.log(rp)

"""
m = PhaseExistenceModel(phase="P")

for depth in np.linspace(0, 100, 20):
    print depth,
    for dist in np.linspace(1.0, 3000, 40):
        print "%.2f " % np.exp(m.log_p(True, distance=dist, depth=depth)),
    print 

print GA_boundaries["P"]
print GA_boundaries["S"]
print GA_boundaries["pP"]
print GA_boundaries["PcP"]
print GA_boundaries["Pg"]
"""
