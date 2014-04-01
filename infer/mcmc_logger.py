import numpy as np
import sys
import os


from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir

class MCMCLogger(object):

    def __init__(self, run_dir=None, dumpsteps=False, write_template_vals=False, dump_interval=500, template_move_step=True, print_interval=20):

        if run_dir is None:
            base_path = os.path.join("logs", "mcmc")
            mkdir_p(base_path)
            run_dir = os.path.join(base_path, "%05d" % next_unused_int_in_dir(base_path))
        mkdir_p(run_dir)
        self.run_dir = run_dir

        self.log_handles = dict()

        self.dumpsteps = dumpsteps
        self.write_template_vals=write_template_vals
        self.dump_interval = dump_interval
        self.print_interval = print_interval

    def log(self, sg, step, n_accepted, n_attempted, move_times):

        if 'lp' not in self.log_handles:
            self.log_handles['lp'] = open(os.path.join(self.run_dir, 'lp.txt'), 'a')
        lp = sg.current_log_p()
        self.log_handles['lp'].write('%f\n' % lp)

        if (step % self.dump_interval == self.dump_interval-1):
            sg.debug_dump(dump_path = os.path.join(self.run_dir, 'step_%06d' % step), pickle_only=True)
            for f in self.log_handles.values():
                if type(f) == file:
                    f.flush()

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
                        for phase in sg.phases:
                            lbl = "%d_%s_%s" % (eid, wn.label, phase)
                            mkdir_p(os.path.join(self.run_dir, 'ev_%05d' % eid))
                            lbl_handle = open(os.path.join(self.run_dir, 'ev_%05d' % eid, "tmpl_%s" % lbl), 'a')
                            tmvals = sg.get_template_vals(eid, sta, phase, wn.band, wn.chan)
                            lbl_handle.write('%06d %f %f %f %f\n' % (step,
                                                                     tmvals['arrival_time'],
                                                                     tmvals['peak_offset'],
                                                                     tmvals['coda_height'],
                                                                     tmvals['coda_decay']))
                            lbl_handle.close()

            handle.close()


        for move_name in move_times.keys():
            if move_name not in self.log_handles:
                self.log_handles[move_name] = open(os.path.join(self.run_dir, 'move_%s_times.txt' % move_name), 'a')
            for (step, t) in move_times[move_name]:
                self.log_handles[move_name].write('%d %f\n' % (step, t));
            del move_times[move_name]

        if self.dumpsteps:
            # dump images for each station at each step
            self.print_mcmc_acceptances(sg, lp, step, n_accepted, n_attempted)
            for (sta, waves) in sg.station_waves.items():
                for wn in waves:
                    plot_with_fit_shapes(os.path.join(self.run_dir, "%s_step%06d.png" % (wn.label, step)), wn)

        if step > 0 and ((step % self.print_interval == 0) or (step < 5)):
            self.print_mcmc_acceptances(sg, lp, step, n_accepted, n_attempted)

    def print_mcmc_acceptances(self, sg, lp, step, n_accepted, n_attempted):

        print "step %d: lp %.2f, accepted " % (step, lp),
        for key in sorted(n_accepted.keys()):
            print "%s: %.3f%%, " % (key, float(n_accepted[key])/n_attempted[key]),
        print ", uatemplates: ", len(sg.uatemplates),
        print ", events: ", len(sg.evnodes)


    def __del__(self):
        for v in self.log_handles.values():
            if type(v) == file:
                v.close()