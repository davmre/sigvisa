import zmq


from sigvisa.infer.swap_rpc.serialization import serialize, deserialize
from sigvisa.infer.coarse_to_fine_init import initialize_sg, do_inference
from sigvisa import Sigvisa
from sigvisa.source.event import Event

class SwapClient(object):

    def __init__(self, sg, name, socket, swap_socket):
        self.sg = sg
        self.name = name
        self.socket = socket
        self.swap_socket = swap_socket

        self.n_attempted = None
        self.n_accepted = None

    def register_counters(self, n_attempted, n_accepted):
        self.n_attempted = n_attempted
        self.n_accepted = n_accepted
        
    def try_sync(self, step, checkpoint):
        
        self.socket.send("SYNC %d %s" % (step, checkpoint))
        msg = self.socket.recv()
        if msg == "SWAP":
            print "client SWAPPING"
            
            self.swap_socket.send("SWAPPING")
            cmd, kwargs = self._parse_msg(self.swap_socket.recv())
            while not cmd.startswith("DONE"):
                response = self._dispatch_command(cmd, kwargs)
                self.swap_socket.send(serialize(response))
                cmd, kwargs = self._parse_msg(self.swap_socket.recv())
        elif msg == "CONTINUE":
            pass
        else:
            raise Exception("unrecognized message %s" % msg)
            

    def _parse_msg(self, msg):
        try:
            cmd, kwargs = msg.split(" ", 1)
            kwargs = deserialize(kwargs)
        except:
            cmd = msg
            kwargs = {}

        return cmd, kwargs

    def _dispatch_command(self, cmd, kwargs):

        if cmd == "GET_SCBS":
            response = self.get_scbs(**kwargs)
        elif cmd == "BIRTH_TEMPLATE":
            response = self.birth_template(**kwargs)
        elif cmd == "KILL_TEMPLATE":
            response = self.kill_template(**kwargs)
        elif cmd == "SET_TEMPLATE":
            response = self.set_template(**kwargs)
        elif cmd == "UATEMPLATES_AT_SCB":
            response = self.uatemplates_at_sta(**kwargs)
        elif cmd == "LOGP_AT_SCB":
            response = self.logp_at_sta(**kwargs)
        elif cmd == "CURRENT_LOG_P":
            response = self.current_log_p(**kwargs)
        elif cmd == "GET_EVENT_LOCS":
            response = self.get_event_locs(**kwargs)
        elif cmd == "GET_EVENT_TEMPLATES":
            response = self.get_event_templates(**kwargs)
        elif cmd == "BIRTH_EVENT":
            response = self.birth_event(**kwargs)
        elif cmd == "KILL_EVENT":
            response = self.kill_event(**kwargs)
        elif cmd == "GET_RAW":
            response = self.get_raw(**kwargs)
        elif cmd == "DUMP_IMG_SCB":
            response = self.dump_img_scb(**kwargs)
        elif cmd == "TOPO_SORT":
            self.sg._topo_sort()
            response = ""
        elif cmd == "COUNT":
            move_name = kwargs["move_name"]
            accepted = kwargs["accepted"]
            print "count", move_name, "accepted", accepted
            if self.n_attempted is not None:
                self.n_attempted[move_name] += 1
            if self.n_accepted is not None:
                self.n_accepted[move_name] += accepted
            response = ""
        elif cmd == "DEBUG":
            import pdb; pdb.set_trace()
            response = ""
        elif cmd == "BREAKDOWN":
            response = self.sg.current_log_p_breakdown()
        else:
            raise Exception("unrecognized command %s" % cmd)

        return response

    def get_raw(self):
        return self.sg.raw_signals

    def get_event_locs(self, include_fixed=False):
        evdicts = {}
        for eid in self.sg.evnodes.keys():
            if eid in self.sg.fixed_events and not include_fixed:
                continue
            ev = self.sg.get_event(eid)
            evdicts[eid] = ev.to_dict() 
        return evdicts

    def get_event_templates(self, eid):
        # return a nested dict:
        # (sta, chan, band) -> phase -> param_name -> val
        r = {}
        
        for sta, wns in self.sg.station_waves.items():
            for phase in self.sg.ev_arriving_phases(eid, sta=sta):
                for wn in wns:
                    key = (sta, wn.chan, wn.band)
                    try:
                        tmvals, _ = wn.get_template_params_for_arrival(eid, phase)
                        if key not in r:
                            r[key] = dict()
                        r[key][phase] = tmvals
                    except:
                        continue

        return r
        
    def birth_event(self, evdict, tmvals, force_id=None):
        # create the event itself
        ev = Event(**evdict)
        evnodes = self.sg.add_event(ev, eid=force_id, phases="none")
        eid = evnodes["loc"].eid

        s = Sigvisa()
        for (sta, chan, band), sta_tmvals in tmvals.items():
            site= s.get_array_site(sta)
            for phase, phase_tmvals in sta_tmvals.items():

                # add all the relevant phases
                self.sg.phases_used.add(phase)
                tg = self.sg.template_generator(phase)
                self.sg.add_event_site_phase(tg, site, phase, evnodes, sample_templates=False)

                # and set their templates to the given values
                tmnodes = self.sg.get_template_nodes(eid, sta, phase, band, chan)
                for param, (k, n) in tmnodes.items():
                    try:
                        n.set_value(key=k, value=phase_tmvals[param])
                    except KeyError as e:
                        if param == "tt_residual" or param == "amp_transfer":
                            # redundant with  arrival_time and coda_height respectively
                            pass
                        else:
                            raise e

        self.sg._topo_sort()

        return eid

    def kill_event(self, eid):
        self.sg.remove_event(eid)

    def get_scbs(self):
        scbs = set()
        for sta, wns in self.sg.station_waves.items():
            for wn in wns:
                scbs.add((sta, wn.chan, wn.band))
        return scbs

    def dump_img_scb(self, scb, label):
        sta, chan, band = scb
        wns = self.sg.station_waves[sta]
        wn = wns[0]

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(20, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        wn.plot(ax=ax)
        ax.set_title("sta %s logp %.1f" % (scb, self.logp_at_sta(scb)))
        ax.set_xlim([wn.st, wn.et])
        canvas.print_figure(label + ".png")
        return None

    def birth_template(self, scb, tmvals, force_id=None, nosort=False):
        sta, chan, band = scb
        wns = self.sg.station_waves[sta]
        # api mismatch here between wns and scb, will have to do
        # something more complicated if this ever becomes an issue
        assert(len(wns)==1)
        wn = wns[0]

        atime = tmvals["arrival_time"]
        nodes = self.sg.create_unassociated_template(wn, atime, 
                                                     nosort=nosort,
                                                     tmid=force_id, 
                                                     initial_vals=tmvals)
        tmid = nodes["arrival_time"].tmid
        return tmid

    def kill_template(self, tmid, nosort=False):
        self.sg.destroy_unassociated_template(tmid=tmid, nosort=nosort)

    def set_template(self, tmid, tmvals):
        tmnodes = self.sg.uatemplates[tmid]
        for k, v in tmvals.items():
            tmnodes[k].set_value(v)

    def current_log_p(self):
        return self.sg.current_log_p()

    def logp_at_sta(self, scb):
        sta, chan, band = scb
        lp = 0.0
        wns = [ wn for wn in self.sg.station_waves[sta] if wn.band == band and wn.chan == chan]
        for wn in wns:
            lp += self.sg.ntemplates_sta_log_p(wn)
        
        tmids = self.sg.uatemplate_ids[scb]
        relevant_nodes = wns
        for tmid in tmids:
            nodes = [n for n in self.sg.uatemplates[tmid].values() if not n.deterministic()]
            relevant_nodes += nodes
        lp += self.sg.joint_logprob_keys(relevant_nodes)

        return lp
        
    def uatemplates_at_sta(self, scb):
        tmids = self.sg.uatemplate_ids[scb]
        r = {}
        for tmid in tmids:
            r[tmid] = dict([(p, n.get_value()) for (p, n) in self.sg.uatemplates[tmid].items()])
        return r
        

def run_client(name, modelspec, runspec, port):

    print "RUNNING CLIENT with", name
    print name, "modelspec", modelspec
    print name, "runspec", runspec
    print name, "port", port
 
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:%d' % port)

    socket.send("NAME %s" % name)
    port_msg = socket.recv()
    private_port = int(port_msg.split()[1])
    swap_socket = context.socket(zmq.REQ)
    swap_socket.connect('tcp://127.0.0.1:%d' % private_port)

    sg = runspec.build_sg(modelspec)

    swapper = SwapClient(sg, name, socket, swap_socket)

    do_inference(sg, modelspec, runspec, 
                 dump_interval = 10, print_interval = 10,
                 swapper=swapper, model_switch_lp_threshold=None)
