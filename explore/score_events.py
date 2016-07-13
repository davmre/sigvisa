import numpy as np
import cPickle as pickle
import sys
import os

from sigvisa.infer.event_birthdeath import prior_location_proposal, ev_death_executor

def final_mcmc_state(ev_dir):
    sorted_steps = sorted([int(d[5:]) for d in os.listdir(ev_dir) if d.startswith('step')])


    idx = -1
    max_step = sorted_steps[idx]

    loaded = False
    while not loaded:
        try:
            with open(os.path.join(ev_dir, "step_%06d" % max_step, 'pickle.sg'), 'rb') as f:
                sg = pickle.load(f)
            loaded = True
        except EOFError:
            idx -= 1
            print "exception loading %s step %d, trying step %d" % (ev_dir, max_step, sorted_steps[idx])
            max_step = sorted_steps[idx]
        
    return sg



def score_sgfolder(folderpath, force_repeat=False):
    
    print "processing", folderpath

    fname_repr = os.path.join(folderpath, "ev_scores.repr")
    if os.path.exists(fname_repr) and not force_repeat:
        print "result file %s already exists, skipping..." % fname_repr
        return

    sg = final_mcmc_state(folderpath)
    print "  loaded sg"
    sg.current_log_p()

    print "starting death scoring"
    scoredict = {}
    scorestrs = []
    scores = []

    for eid in sg.evnodes.keys():
        ev = sg.get_event(eid)
        r = ev_death_executor(sg, prior_location_proposal, 
                              proposal_includes_mb=True,
                              use_correlation=False,
                              repropose_uatemplates=False,
                              birth_type="dumb",
                              inference_step=-1,
                              force_kill_eid=eid,
                              propose_map=True)
        lp_new, lp_old, log_qforward, log_qbackward, redeath, rebirth, proposal_extra = r
        score = (lp_old + log_qforward) - (lp_new + log_qbackward)

        evdict = ev.to_dict()
        
        scoredict[eid] = (score, evdict)
        scorestr = "eid %d score %.2f\nev %s\n" % (eid, score, ev)
        print scorestr.split("\n")[0]
        scorestrs.append(scorestr)
        scores.append(score)

    with open(fname_repr, 'w') as f:
        f.write(repr(scoredict))
    print "wrote", fname_repr

    p = sorted(range(len(scorestrs)), key = lambda i : -scores[i])
    sorted_scorestrs = [scorestrs[i] for i in p]
    fname_txt = os.path.join(folderpath, "ev_scores.txt")
    with open(fname_txt, 'w') as f:
        f.write("\n".join(sorted_scorestrs))
    print "wrote", fname_txt

def main():

    job_folder = sys.argv[1]
    for j in os.listdir(job_folder):
        folderpath = os.path.join(job_folder, j)

        if not os.path.isdir(folderpath): continue

        try:
            score_sgfolder(folderpath)
        except Exception as e:
            print "error on folder %s: %s" % (folderpath, e)
            continue

if __name__ == "__main__":
    main()
