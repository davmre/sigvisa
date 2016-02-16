# recompute training messages to fix the fact that I used the wrong prior

import numpy as np
from sigvisa import Sigvisa
from collections import defaultdict
from sigvisa.graph.sigvisa_graph import get_param_model_id, ModelNotFoundError
from sigvisa.learn.train_param_common import load_modelid 
from sigvisa.learn.fit_shape_params_mcmc import multiply_scalar_gaussian
from sigvisa.source.event import get_event
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential, TruncatedGaussian, LogNormal, InvGamma, Beta

fix_runid = 2
prior_runid=1


dummyPriorModel = {
"tt_residual": TruncatedGaussian(mean=0.0, std=1.0, a=-25.0, b=25.0),
"amp_transfer": Gaussian(mean=0.0, std=2.0),
"peak_offset": TruncatedGaussian(mean=-0.5, std=1.0, b=4.0),
"mult_wiggle_std": Beta(4.0, 1.0),
"coda_decay": Gaussian(mean=0.0, std=1.0),
"peak_decay": Gaussian(mean=0.0, std=1.0)
}

model_type = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb', 'peak_decay': 'param_linear_distmb', 'mult_wiggle_std': 'dummyPrior'}

def get_prior(sta, phase, param, chan, band):
    try:
        modelid = get_param_model_id((prior_runid,), sta, phase, model_type[param], param, "lin_polyexp", chan=chan, band=band)
        model = load_modelid(modelid)
    except ModelNotFoundError:
        model = dummyPriorModel[param]
    return model

s = Sigvisa()
cursor = s.dbconn.cursor()
sql_query = "select sta, chan, band, evid, fitid from sigvisa_coda_fit where runid=%d" % fix_runid
cursor.execute(sql_query)
fitids = cursor.fetchall()

for sta, chan, band, evid, fitid in fitids:
    ev = get_event(evid=evid)

    sql_query = "select fpid, phase, message_fname from sigvisa_coda_fit_phase where fitid=%d" % fitid
    cursor.execute(sql_query)
    fpids = cursor.fetchall()
    for fpid, phase, message_fname in fpids:

        if phase not in ("P", "Lg"): continue

        fname = "training_messages/runid_%d/%s" % (fix_runid, message_fname)
        with open(fname, 'r') as f:
            messages = eval(f.read(), {'array': np.array})
        params = [k for k in messages.keys() if "posterior" not in k]
    
        
        for param in params:
            prior = get_prior(sta, phase, param, chan, band)
            prior_mean = prior.predict(cond=ev)
            prior_var = prior.variance(cond=ev, include_obs=True)
            mm, mv = messages[param]
            pm, pv = messages[param+"_posterior"]
            fixed_mm, fixed_mv = multiply_scalar_gaussian(pm, pv, prior_mean, -prior_var)
            fixed_mv = np.abs(fixed_mv)
            messages[param] = (float(fixed_mm), float(fixed_mv))

        with open(fname, 'w') as f:
            f.write(repr(messages))
        print "wrote", fname
    

cursor.close()
