import numpy as np
import zmq
import logging
import time

from multiprocessing import Process

from sigvisa.infer.swap_rpc.sg_client import run_client
from sigvisa.infer.swap_rpc.swap_server import SwapServer
from sigvisa.infer.swap_rpc.swap_moves import crossover_uatemplates, crossover_event_region_move, swap_events_move
from sigvisa.infer.swap_rpc.serialization import serialize, deserialize

class SgSwapServer(SwapServer):

    def __init__(self, *args, **kwargs):
        super(SgSwapServer, self).__init__(*args, **kwargs)

        self.scbs = {}
        self.raw_signals = {}

    def do_swap_helper(self, client1, client2):

        socket1 = self.swap_sockets[client1]
        socket2 = self.swap_sockets[client2]
        
        # both clients should check in to be ready to receive commands
        ack1 = socket1.recv()
        ack2 = socket2.recv()
        assert(ack1 == "SWAPPING")
        assert(ack2 == "SWAPPING")

        sg1 = SgRpcShim(socket1)
        sg2 = SgRpcShim(socket2)

        if client1 not in self.scbs:
            self.scbs[client1] = sg1.get_scbs()
            self.raw_signals[client1] = sg1.get_raw()
        if client2 not in self.scbs:
            self.scbs[client2] = sg2.get_scbs()
            self.raw_signals[client2] = sg2.get_raw()
        assert(self.scbs[client1] == self.scbs[client2])
        scbs = list(self.scbs[client1])

        raw1 = self.raw_signals[client1]
        raw2 = self.raw_signals[client2]

        """
        for scb in scbs:
            accepted = crossover_uatemplates(sg1, sg2, scb, raw1, raw2, 
                                             crossover_period_s=5.0,
                                             crossover_period_pre_s = 2.0)
            logging.info("crossover at %s: accepted %s" % (str(scb), str(accepted)))
            sg1.move_finished("crossover_uatemplates_short", accepted)
            sg2.move_finished("crossover_uatemplates_short", accepted)

        for scb in scbs:
            accepted = crossover_uatemplates(sg1, sg2, scb, raw1, raw2)
            logging.info("crossover at %s: accepted %s" % (str(scb), str(accepted)))
            sg1.move_finished("crossover_uatemplates", accepted)
            sg2.move_finished("crossover_uatemplates", accepted)
        """

        accepted = crossover_event_region_move(sg1, sg2, raw1, raw2,
                                               crossover_radius_km=1000,
                                               crossover_radius_s=2000)
        logging.info("event region crossover accepted %s" % (str(accepted)))
        sg1.move_finished("crossover_event_region", accepted)
        sg2.move_finished("crossover_event_region", accepted)

        accepted = swap_events_move(sg1, sg2, raw1, raw2)
        logging.info("event swap accepted %s" % (str(accepted)))
        sg1.move_finished("crossover_event_swap", accepted)
        sg2.move_finished("crossover_event_swap", accepted)

        sg1.done()
        sg2.done()

        
        self._cleanup_swap(client1, client2)

class SgRpcShim(object):

    def __init__(self, socket):
        self.socket = socket

    def get_raw(self):
        return self._send_cmd("GET_RAW", kwargs={})

    def get_scbs(self):
        return self._send_cmd("GET_SCBS", kwargs={})

    def get_event_locs(self):
        cmd = "GET_EVENT_LOCS"
        return self._send_cmd(cmd, kwargs={})

    def get_event_templates(self, eid):
        kwargs = {"eid": eid}
        cmd = "GET_EVENT_TEMPLATES"
        return self._send_cmd(cmd, kwargs=kwargs)

    def kill_event(self, eid):
        kwargs = {"eid": eid}
        cmd = "KILL_EVENT"
        return self._send_cmd(cmd, kwargs=kwargs)

    def birth_event(self, evdict, tmvals, force_id=None):
        kwargs = {"evdict": evdict, "tmvals": tmvals, "force_id": force_id}
        cmd = "BIRTH_EVENT"
        return self._send_cmd(cmd, kwargs=kwargs)

    def current_log_p(self):
        cmd = "CURRENT_LOG_P"
        return self._send_cmd(cmd, kwargs={})

    def current_log_p_breakdown(self):
        cmd = "BREAKDOWN"
        return self._send_cmd(cmd, kwargs={})

    def birth_template(self, scb, tmvals, force_id=None):
        kwargs = {"scb": scb, "tmvals": tmvals, "force_id": force_id}
        cmd = "BIRTH_TEMPLATE"
        return self._send_cmd(cmd, kwargs)

    def kill_template(self, tmid):
        kwargs = {"tmid": tmid}
        cmd = "KILL_TEMPLATE"
        return self._send_cmd(cmd, kwargs)
    
    def set_template(self, tmid, tmvals):
        kwargs = {"tmvals": tmvals, "tmid": tmid}
        cmd = "SET_TEMPLATE"
        return self._send_cmd(cmd, kwargs)

    def logp_at_scb(self, scb):
        kwargs = {"scb": scb}
        cmd = "LOGP_AT_SCB"
        return self._send_cmd(cmd, kwargs)

    def uatemplates_at_scb(self, scb):
        kwargs = {"scb": scb}
        cmd = "UATEMPLATES_AT_SCB"
        return self._send_cmd(cmd, kwargs)

    def dump_img_scb(self, scb, label):
        cmd = "DUMP_IMG_SCB"
        kwargs = {"scb": scb, "label": label}
        return self._send_cmd(cmd, kwargs)

    def debug(self):
        msg = "DEBUG"
        self.socket.send(msg)

    def move_finished(self, move_name, accepted):
        cmd = "COUNT"
        kwargs = {"move_name": move_name, "accepted": accepted}
        return self._send_cmd(cmd, kwargs)

    def done(self):
        msg = "DONE"
        self.socket.send(msg)

    def _send_cmd(self, cmd, kwargs):
        argstr = serialize(kwargs)
        msg = cmd + " " + argstr
        self.socket.send(msg)
        rstr = self.socket.recv()
        resp = deserialize(rstr)
        #print "cmd", cmd, "response", rstr, "deserialized", resp
        return resp


def run_parallel_coarse_to_fine(names, specs, 
                                server_only=False, client_only=None,
                                min_swap_s = 20.0, 
                                max_swap_s = 45.0,
                                allowable_wait_s = 0.5):
    # names is an ordered list of strings naming each thread.
    #   - we will only ever run swap moves between adjacent threads
    # specs is a dict mapping name:(modelspec, runspec)

    processes = {}

    def chain_neighbors(a):
        # given a list, return a dict encoding the graph where each
        # entry is connected to its predecessor and successor.

        d = {}
        for i, x in enumerate(a):
            d[x] = []
            if i > 0:
                d[x].append(a[i-1])
            if i < len(a)-1:
                d[x].append(a[i+1])
        return d

    control_port=5555
    neighbors = chain_neighbors(names)

    if client_only is not None:
        name = client_only
        ms, rs = specs[name]
        run_client(name, ms, rs, control_port)
        return

    if not server_only:
        for name in names:
            ms, rs = specs[name]
            processes[name] = Process(target=run_client, kwargs={"name": name, 
                                                                 "modelspec": ms, 
                                                                 "runspec": rs, 
                                                                 "port": control_port})
            processes[name].start()


    serv = SgSwapServer(neighbors=neighbors, 
                        min_swap_s = min_swap_s,
                        allowable_wait_s = allowable_wait_s,
                        port=control_port)
    rootLogger = logging.getLogger()
    rootLogger.setLevel("INFO")

    def any_children_alive():
        if server_only:
            return True

        for name in names:
            if processes[name].is_alive():
                return True
        return False

    while any_children_alive():
        serv.process()
        logging.debug( "state dump: %s " % serv.client_state)
