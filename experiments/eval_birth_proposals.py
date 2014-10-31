import numpy as np

from sigvisa.experiments.one_station_templates_test import sample_template
from sigvisa.infer.template_mcmc import *


def birth_proposal_likelihood(birth_proposal, nm_type='l1'):
    lps = []
    for seed in range(50):
        wave, templates = sample_template(seed=seed, hardcoded=False, nm_type=nm_type, n_templates=1, srate=1.0)
        tvals = dict([(k, n.get_value()) for (k, n) in templates[0].items()])

        sg = SigvisaGraph(template_model_type="dummy", template_shape="lin_polyexp",
                              wiggle_model_type="dummy", wiggle_family="dummy",
                              phases="leb", nm_type = nm_type, wiggle_len_s = 60.0)
        wn = sg.add_wave(wave)

        tg = sg.template_generator('UA')
        tg.hack_force_mean = np.log(wn.nm.c * 10)

        vals, lp = birth_proposal(sg, wn, fix_result=tvals)
        print "%.2f, " % lp,
        lps.append(lp)
    print
    return np.array(lps)

lp1 = birth_proposal_likelihood(optimizing_birth_proposal)
lp2 =  birth_proposal_likelihood(optimizing_birth_proposal, nm_type='ar')

lp3 = birth_proposal_likelihood(birth_proposal)
lp4 = birth_proposal_likelihood(birth_proposal, nm_type='ar')

n = float(len(lp1))

print np.arange(50)[lp1>lp3]
print np.arange(50)[lp2>lp4]

print "l1: optimizing %f vs dumb %f, optimizing wins %.1f%% of the time" % (np.mean(lp1), np.mean(lp3), 100/n*np.sum(lp1>lp3))
print "ar: optimizing %f vs dumb %f, optimizing wins %.1f%% of the time" % (np.mean(lp2), np.mean(lp4), 100/n*np.sum(lp2>lp4))
