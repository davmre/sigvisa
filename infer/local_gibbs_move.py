import numpy as np

from sigvisa.models.distributions import PiecewiseLinear

"""

Methods to propose a param value from a piecewise linear approximation
to the Gibbs conditional. This can be based on either the full model
posterior (proxylp_full) or a cheap proxy based on the noise model
applied to a local region of the unexplained envelope (proxylp_localenv).

"""

def proxylp_full(sg, wn, node):
    def proxylp(candidate):
        node.set_value(candidate)
        lp = node.log_p() + wn.log_p()    
        return float(lp)
    return proxylp

def proxylp_localenv(sg, wn, eid, phase, param):
    tmnodes = sg.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)
    k, node = tmnodes[param]
    
    tmvals = dict([(p, n.get_value(key=k)) for (p, (k, n)) in tmnodes.items()])
    
    atime = tmvals['arrival_time']
    peak_time = tmvals['arrival_time'] + np.exp(tmvals['peak_offset'])
    unexplained = wn.unexplained_env(eid, phase)

    peak_idx = int((peak_time - wn.st) * wn.srate)
    start_idx_true = int((atime - wn.st) * wn.srate)
    end_idx_true = int(peak_idx + 60*wn.srate)
    start_idx = max(0, start_idx_true)
    end_idx = min(wn.npts, end_idx_true)
    start_offset = start_idx - start_idx_true
    if end_idx-start_idx < wn.srate:
        # if less than 1s of available signal, don't even bother
        return None
    
    unexplained_local = unexplained[start_idx:end_idx]
    n = len(unexplained_local)  
    
    def proxylp(candidate):
        tmvals[param] = candidate
        
        l = tg.abstract_logenv_raw(tmvals, srate=wn.srate, fixedlen=n+start_offset)
        diff = unexplained_local - np.exp(l[start_offset:])
        lp = wn.nm_env.log_p(diff) + node.model.log_p(candidate, cond=node._pv_cache)
        return float(lp)
    
    return proxylp
        
def approximate_scalar_gibbs_distribution(sg, wn, eid, phase, param, 
                                          node, proxylp, prior_weight = 0.0):

    class priormodel(object):
        def __init__(self, node):
            self.model = node.model
            self.pv = node._pv_cache
        
        def log_p(self, x, **kwargs):
            return float(self.model.log_p(x, cond=self.pv, **kwargs))

        def sample(self, **kwargs):
            return float(self.model.sample(cond=self.pv, **kwargs))

    assert (not node.deterministic())
    tg = sg.template_generator(phase)
    lbounds, hbounds = tg.low_bounds(), tg.high_bounds()
    
    # generate a range of plausible values based on the prior,
    # and on the current value v (which should already be adapted
    # somewhat to the data).
    pv = node._pv_cache
    v = float(node.get_value())
    pred = node.model.predict(cond=pv)
    std = np.sqrt(node.model.variance(cond=pv, include_obs=True))
    if param=="tt_residual":
        prior_min, prior_max = -25, 25
    elif param=="mult_wiggle_std":
        prior_min = 0.1
        prior_max = 0.99
    else:
        prior_min, prior_max = pred-4*std, pred+4*std
        prior_min = min(prior_min, v-4*std)
        prior_max = max(prior_max, v + 4*std)
        if param in lbounds:
            prior_min = max(prior_min, lbounds[param])
            prior_max = min(prior_max, hbounds[param])
    candidates = np.linspace(prior_min,  prior_max, 20)
    candidates = np.array(sorted(list(candidates) + [v,]))
    
    # compute the logp at each of these candidates
    lps = np.array([proxylp(candidate) for candidate in candidates])
    
    # now refine the approximation in regions of high probability
    def bad_indices(lps, candidates):
        best_idx = np.argmax(lps)
        best_lp = np.max(lps)
        lp_diff = np.abs(np.diff(lps))

        # an lp is "significant" if it or its neighbor is above the threshold
        thresh = best_lp - 3
        significant_lps = ( lps[:-1] > thresh ) +  ( lps[1:] > thresh )

        # a "bad step" is where we have a sharp boundary next to a significant lp.
        # that is, the significant lps are the areas where it's important to
        # approximate the posterior well, and a large difference in lp between adjacent
        # candidates means we're not doing that.
        badsteps = significant_lps * (lp_diff > 1)
        bad_idxs = np.arange(len(lps)-1)[badsteps]

        # if we've already refined a lot at a particular bad idx,
        # just give up since there's probably a genuine discontinuity there
        c_diff = np.abs(np.diff(candidates))
        hopeless = c_diff < 1e-3
        bad_idxs = [idx for idx in bad_idxs if not hopeless[idx]]

        return bad_idxs

    bad_idxs = bad_indices(lps, candidates)
    while len(bad_idxs) > 0:
        new_candidates = []
        new_lps = []
        for idx in bad_idxs:
            c1 = candidates[idx]
            c2 = candidates[idx+1]
            c = c1 + (c2-c1)/2.0
            new_candidates.append(c)
            new_lps.append( proxylp(c))
            
        # merge the new candidates into their sorted positions in
        # the existing list
        full_c = np.concatenate((candidates, new_candidates))
        full_lps = np.concatenate((lps, new_lps))
        perm = sorted(np.arange(len(full_c)), key = lambda i : full_c[i])
        candidates = np.array(full_c[perm])
        lps = np.array(full_lps[perm])
        bad_idxs = bad_indices(lps, candidates)
    
    node.set_value(v)

    p = PiecewiseLinear(candidates, np.array(lps), mix_weight = prior_weight, mix_dist = priormodel(node))
    return p
    
    
