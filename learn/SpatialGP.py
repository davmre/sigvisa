import os
import numpy as np

from gpr import munge, kernels, evaluate, learn, distributions, plot
from gpr.gp import GaussianProcess


def dist_azi_depth_distfn_log(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(dad1[0]+1)-np.log(dad2[0]+1)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * np.log(avg_dist)
    depth = np.log(dad1[2]+1)- np.log(dad2[2]+1)

    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)
    return r

def dist_azi_depth_distfn_linear(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0]-dad2[0]
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist
    depth = dad1[2]- dad2[2]

    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)
    return r

def dist_azi_depth_distfn_cuberoot(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0]**(1.0/3)-dad2[0]**(1.0/3)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist**(1.0/3)
    depth = dad1[2]**(1.0/3)- dad2[2]**(1.0/3)

    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)
    return r

def dist_distfn(lldda1, lldda2, params=None):
    return lldda1[3]**(1.0/3)-lldda2[3]**(1.0/3)

def depth_distfn(lldda1, lldda2, params=None):
    return lldda1[2]**(1.0/3)-lldda2[2]**(1.0/3)

def azi_distfn(lldda1, lldda2, params=None):
    avg_dist = (lldda1[2]+lldda2[2])/2
    return utils.geog.degdiff(lldda1[4], lldda2[4]) * avg_dist**(1.0/3)

def ll_distfn(lldda1, lldda2, params=None):
    return utils.geog.dist_km(lldda1[0:2], lldda2[0:2])

def dist_azi_depth_distfn_deriv_cuberoot(i, dad1, dad2, params):
    azi_scale = params[0]

    depth_scale = params[1]
    dist = dad1[0]**(1.0/3)-dad2[0]**(1.0/3)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist**(1.0/3)
    depth = dad1[2]**(1.0/3)- dad2[2]**(1.0/3)
    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)

    if i==0: # deriv wrt azi_scale                                                                                                                                                                  
        deriv = azi_scale * azi**2 / r if r != 0 else 0
    elif i==1: # deriv wrt depth_scale                                                                                                                                                              
        deriv = depth_scale * depth**2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv

def dist_azi_depth_distfn_deriv_linear(i, dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0]-dad2[0]
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist
    depth = dad1[2]- dad2[2]
    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)

    if i==0: # deriv wrt azi_scale                                                                                                                                                                  
        deriv = azi_scale * azi**2 / r if r != 0 else 0
    elif i==1: # deriv wrt depth_scale                                                                                                                                                              
        deriv = depth_scale * depth**2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv

def dist_azi_depth_distfn_deriv_log(i, dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(dad1[0]+1)-np.log(dad2[0]+1)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * np.log(avg_dist+1)
    depth = np.log(dad1[2]+1)- np.log(dad2[2]+1)
    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)

    if i==0: # deriv wrt azi_scale                                                                                                                                                                  
        deriv = azi_scale * azi**2 / r if r != 0 else 0
    elif i==1: # deriv wrt depth_scale                                                                                                                                                              
        deriv = depth_scale * depth**2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)


def lon_lat_depth_distfn(lld1, lld2, params=None):
    ll = utils.geog.dist_km(lld1[0:2], lld2[0:2])
    depth = lld1[2] - lld2[2]
    r = np.sqrt(ll**2 + depth**2)
    return r


distfns = {
"dad_cuberoot": [dist_azi_depth_distfn_cuberoot, dist_azi_depth_distfn_deriv_cuberoot]
"dad_linear": [dist_azi_depth_distfn_linear, dist_azi_depth_distfn_deriv_linear]
"dad_log": [dist_azi_depth_distfn_log, dist_azi_depth_distfn_deriv_log]
"lld": lon_lat_depth_distfn,
}

class SpatialGP(GaussianProcess):

    def __init__(self, *args, distfn_str="dad_log", **kwargs):
        self.distfn_str = distfn_str
        kwargs['kernel_extra'] = distfns[distfn_str]
        GaussianProcess.__init__(self, *args, **kwargs)
        
        
    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        kname = np.array((self.kernel_name,))
        mname = np.array((self.mean,))
        np.savez(filename, X = self.X, y=self.y, mu = np.array((self.mu,)), kernel_name=kname, kernel_params=self.kernel_params, mname = mname, alpha=self.alpha, Kinv=self.Kinv, K=self.K, L=self.L, distfn_str = self.distfn_str)


    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.__unpack_npz(npzfile)
        self.distfn_str = npzfile['distfn_str']
        self.kernel_extra = distfns[distfn_str]
        self.kernel = kernels.setup_kernel(self.kernel_name, self.kernel_params, extra=self.kernel_extra)
        
