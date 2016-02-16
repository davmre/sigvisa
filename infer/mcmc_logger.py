import numpy as np
import sys
import os
import time
import shutil
import logging

from sigvisa import Sigvisa
from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir, make_next_unused_dir

class MCMCLogger(object):

    def __init__(self, run_dir=None,  write_template_vals=False, dump_interval_s=10, template_move_step=True, print_interval_s=10, transient=False, write_gp_hparams=False, serialize_interval_s=10, max_dumps=None):

        s = Sigvisa()

        if run_dir is None:
            if os.getenv("LOGDIR") is None:
                base_path = os.path.join(s.homedir, "logs", "mcmc")
                formatter = lambda i : "%05d" % i
                run_dir = make_next_unused_dir(base_path, formatter)
            else:
                run_dir = os.getenv("LOGDIR")

        mkdir_p(run_dir)

        self.run_dir = run_dir

        cmd_file = os.path.join(run_dir, "cmd.txt")

        if not (os.path.exists(cmd_file)):
            with open(cmd_file, 'w') as f:
                f.write(" ".join(sys.argv))

        self.log_handles = dict()

        self.write_template_vals=write_template_vals
        self.write_gp_hparams = write_gp_hparams
        self.dump_interval_s = dump_interval_s
        self.print_interval_s = print_interval_s
        self.serialize_interval_s = serialize_interval_s
        self.max_dumps = max_dumps

        self.last_dump_time = time.time()
        self.last_serialize_time = time.time()
        self.last_print_time = 0.0

        self.start_time = None

        # whether to delete the log directoryat the end of the run
        self.transient = transient

        self.lps = []
        self.last_step = 0


    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def log(self, sg, step, n_accepted, n_attempted, move_times):

        self.last_step = step
        if 'lp' not in self.log_handles:
            self.log_handles['lp'] = open(os.path.join(self.run_dir, 'lp.txt'), 'a')
        lp = sg.current_log_p()
        self.lps.append(lp)
        self.log_handles['lp'].write('%f\n' % lp)


        if 'times' not in self.log_handles:
            self.log_handles['times'] = open(os.path.join(self.run_dir, 'times.txt'), 'a')
        if self.start_time is None:
            raise Exception("must call logger.start() before calling logger.log()")
        elapsed = time.time() - self.start_time
        self.log_handles['times'].write('%f\n' % elapsed)


        t = time.time()
        if t - self.last_dump_time > self.dump_interval_s:
            self.last_dump_time = t
            self.dump(sg)

        if t - self.last_serialize_time > self.serialize_interval_s:
            self.last_serialize_time = t
            self.serialize(sg)

        for (eid, evnodes) in sg.evnodes.items():

            handle = open(os.path.join(self.run_dir, 'ev_%05d.txt' % eid), 'a')
            evlon = evnodes['loc'].get_local_value('lon')
            evlat = evnodes['loc'].get_local_value('lat')
            evdepth = evnodes['loc'].get_local_value('depth')
            evtime = evnodes['time'].get_local_value('time')
            evmb = evnodes['mb'].get_local_value('mb')
            evsource = evnodes['natural_source'].get_local_value('natural_source')
            handle.write('%06d\t%3.4f\t%3.4f\t%4.4f\t%10.2f\t%2.3f\t%d\n' % (step, evlon, evlat, evdepth, evtime, evmb, evsource))

            if self.write_template_vals:
                for (sta,wns) in sg.station_waves.items():
                    for wn in wns:
                        ev_phases = [phase for (eeid,phase) in wn.arrivals() if eeid==eid]
                        for phase in ev_phases:
                            try:
                                tmvals = sg.get_template_vals(eid, sta, phase, wn.band, wn.chan)
                            except KeyError:
                                # if this event does not generate this phase at this station
                                continue

                            lbl = "%d_%s_%s" % (eid, wn.label, phase)
                            mkdir_p(os.path.join(self.run_dir, 'ev_%05d' % eid))
                            lbl_handle = open(os.path.join(self.run_dir, 'ev_%05d' % eid, "tmpl_%s" % lbl), 'a')
                            try:
                                mws = tmvals['mult_wiggle_std']
                            except:
                                mws = -1.0
                            lbl_handle.write('%06d %f %f %f %f %f %f\n' % (step,
                                                                     tmvals['arrival_time'],
                                                                     tmvals['peak_offset'],
                                                                     tmvals['coda_height'],
                                                                     tmvals['peak_decay'],
                                                                     tmvals['coda_decay'],
                                                                     mws))
                            lbl_handle.close()

            handle.close()

        for sta, wns in sg.station_waves.items():
            for wn in wns:
                lbl = '%s.txt' % wn.nm_node.label
                if lbl not in self.log_handles:
                    self.log_handles[lbl] = open(os.path.join(self.run_dir, lbl), 'a')
                handle = self.log_handles[lbl]
                nm = wn.nm_node.get_value()
                nm_params = np.concatenate(((nm.c, nm.em.std), nm.params))
                handle.write("%06d " % step + " ".join(["%.4f" % x for x in nm_params])  +"\n")

        if sg.jointgp and self.write_gp_hparams:
            gpdir = os.path.join(self.run_dir, 'gp_hparams')
            mkdir_p(gpdir)
            for hparam_key, hparam_nodes in sg._jointgp_hparam_nodes.items():
                lbl = hparam_key

            #for (sta, pdicts) in sg._joint_gpmodels.items():
            #    for hparam in sg.jointgp_hparam_prior.keys():
            #        
                if lbl not in self.log_handles:
                    self.log_handles[lbl] = open(os.path.join(gpdir, lbl), 'a')

                s = ""

                for hparam in sorted(hparam_nodes.keys()):

                    try:
                        s += "%.4f " % hparam_nodes[hparam].get_value()
                    except KeyError:
                        continue

                handle = self.log_handles[lbl]
                handle.write(s + "\n")
                handle.flush()


        for move_name in move_times.keys():
            if move_name not in self.log_handles:
                self.log_handles[move_name] = open(os.path.join(self.run_dir, 'move_%s_times.txt' % move_name), 'a')
            for (step, t) in move_times[move_name]:
                self.log_handles[move_name].write('%d %f\n' % (step, t));
            del move_times[move_name]

        if step > 0 and time.time() - self.last_print_time > self.print_interval_s:
            s = self.acceptance_string(sg, lp, step, n_accepted, n_attempted)
            print s
            self.last_print_time = time.time()

            if "acceptance_rates" not in self.log_handles:
                self.log_handles["acceptance_rates"] = open(os.path.join(self.run_dir, 'acceptance_rates.txt'), 'a')
            self.log_handles["acceptance_rates"].write(s + "\n\n")

    def dump(self, sg):
        step = self.last_step
        sg.debug_dump(dump_path = os.path.join(self.run_dir, 'step_%06d' % step), pickle_only=True)
        for f in self.log_handles.values():
            if type(f) == file:
                f.flush()
        self.gc_dumps()

    def gc_dumps(self):
        if self.max_dumps is None:
            return

        dump_dirs = sorted([dname for dname in os.listdir(self.run_dir) if fname.startswith("step_")])
        if len(dump_dirs)  > self.max_dumps:
            to_delete = dump_dirs[1: -self.max_dumps]
            for ddir in to_delete:
                shutil.rmtree(os.path.join(self.run_dir, ddir))

    def serialize(self, sg):
        step = self.last_step
        serial_dir = os.path.join(self.run_dir, "serialized")
        mkdir_p(serial_dir)

        sg.serialize_to_file(os.path.join(serial_dir, 'step_%06d' % step))
        for f in self.log_handles.values():
            if type(f) == file:
                f.flush()

    def log_event_move(self, sg, eid, old_ev, new_ev, details, lp_old, lp_new, lqf, lqb, accepted):
        s = "\n\nstep %d: proposed to move eid %d to %s\n" % (self.last_step, eid, new_ev)
        s += "  additional proposal details: %s\n" % repr(details)
        s += "  lp_new %.2f lp_old %.2f log_qforward %.2f log_qbackward %.2f\n" % (lp_new, lp_old, lqf, lqb)
        ratio = lp_new + lqb - lp_old - lqf
        if accepted:
            s += "  move ACCEPTED with ratio %.2f\n" % ratio
        else:
            s += "  move REJECTED with ratio %.2f\n" % ratio

        handle = open(os.path.join(self.run_dir, 'ev_%05d_proposals.txt' % eid), 'a')
        handle.write(s)
        handle.close()

    def load_template_vals(self, eid, phase, wn):
        """
        Return a numpy array containing the template vals saved for this run, along with
        the logprob of each state.
        """

        if not self.write_template_vals:
            raise Exception("MCMC logger did not save template vals, cannot load")

        lbl = "%d_%s_%s" % (eid, wn.label, phase)
        lbl_fname = os.path.join(self.run_dir, 'ev_%05d' % eid, "tmpl_%s" % lbl)
        vals = np.loadtxt(lbl_fname)
        labels = ('step', 'arrival_time', 'peak_offset', 'coda_height', 'peak_decay', 'coda_decay', 'mult_wiggle_std')


        lp_fname = os.path.join(self.run_dir, 'lp.txt')
        try:
            lps = np.loadtxt(lp_fname)
        except:
            lps = np.genfromtxt(lp_fname, skip_footer=2)
            n = len(lps)
            vals = vals[:n]
            
        return vals, labels, lps

    def acceptance_string(self, sg, lp, step, n_accepted, n_attempted):

        s = "step %d: lp %.2f, accepted " % (step, lp)
        for key in sorted(n_accepted.keys()):
            s += "%s: %.3f%%, " % (key, float(n_accepted[key])/n_attempted[key])
        s += ", uatemplates: %d " % len(sg.uatemplates)
        s += ", events: %d " % (len(sg.evnodes) - len(sg.fixed_events))
        return s

    def __del__(self):
        for v in self.log_handles.values():
            if type(v) == file:
                v.close()

        if self.transient:
            shutil.rmtree(self.run_dir)
            print "run finished, deleted log dir %s" % (self.run_dir)
