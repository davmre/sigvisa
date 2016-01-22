import numpy as np
import zmq
from optparse import OptionParser
import time
from collections import defaultdict
import threading
import sys
import logging

class SwapServer(object):

    """General purpose class for managing interactions between parallel subprocesses.
    Each subprocess does its own computations but also needs to occasionally 
    exchange information with its "neighbors", which can be specified as an arbitrary
    graph topology (though in practice will be a chain). 

    Before starting a new work unit, each subprocess checks in with the estimated
    time requirement for that work unit. If appropriate, the subprocess is told to 
    pause and wait for a neighboring process to become ready, so that they can 
    exchange information. The goal is to minimizing time spent waiting while 
    maintaining consistent communication between processes. 

    Currently this is implemented as:
     - if the elapsed time since last swap with this neighbor is 
       greater than max_swap_s, always force a swap
     - otherwise, if the elapsed time is greater than min_swap,
       and we expect the neighbor to be ready in less than
       allowable_wait_s, then wait to swap
     - otherwise, continue.

    Interprocess communication is implemented using ZMQ, with a naive
    design that nonetheless seems to work. The server runs a main ROUTER socket 
    that handles routine checkin requests from clients. It also maintains a 
    private REQ/REP channel with each client used for coordinating the actual
    swap move. Since responses on this channel can cause clients to do arbitrary 
    computation and block for long periods, the private swap_sockets are only 
    ever accessed from child threads that cannot block the main server. 

    The state of each client is either:
      - "WORKING", extra_state:(checkpoint, start_time) where start_time is the time of last checkin, 
        when it started working on the current work unit. Given this info and the 
        checkpoint, we can estimate the expected finish time. 
      - "WAITING", extra_state:(<name>, start_time): the client has checked in and the server intends for it 
        to swap with neighbor <name>, so the client is blocking while we wait for 
        <name> to check in. 
      - "SWAPPING": the client is currently swapping and being controlled
        by a child thread. 
      - "SWAPPED": client has finished a swap and is proceeding with its original 
        work unit, but we don't have an expected finish time.

    The dictionary state_locks contains a lock for the state of each client. The state of
    the client can only be changed by the thread holding the lock. The main thread 
    will acquire this lock whenever it receives a checkin from a client, and will either 
    release it after processing the checkin, or hand the lock to a child thread performing
    a swap move.

    The swap move itself is specified by overriding the do_swap_helper method
        do_swap_helper(name1, name2):
    where name1 and name2 are names for the clients.


    """

    def __init__(self, 
                 neighbors,
                 port=5555, 
                 min_swap_s = 20.0, 
                 max_swap_s=50.0,
                 allowable_wait_s = 0.5):

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind('tcp://127.0.0.1:%d' % port)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.next_port = port+1

        self.neighbors = neighbors

        self.client_state = {}
        self.client_state_extra = {}
        self.swap_sockets = {}
        self.state_lock = {}
        
        # nested dict: for each client, we have a dict mapping
        # checkpoint names to the time taken for the last cycle of
        # work from that checkpoint
        self.checkpoint_times = defaultdict(dict)

        # keys are tuples (name1, name2) with client names in sorted order
        self.last_swap = defaultdict(float)

        self.min_swap_s = min_swap_s
        self.max_swap_s = max_swap_s
        self.allowable_wait_s = allowable_wait_s

        # mappings to connect client names with ZMQ addresses
        self.name_map = {}
        self.name_backmap = {}

    def _cleanup_swap(self, client1, client2):
        """
        Enforce invariants once a swap move is finished.
        To be called by each child swap thread before it terminates. 
        """
         
        self.client_state[client1] = "SWAPPED"
        self.client_state[client2] = "SWAPPED"

        pair_key = tuple(sorted((client1, client2)))
        self.last_swap[pair_key] = time.time()

        self.state_lock[client1].release()
        self.state_lock[client2].release()

    def do_swap_helper(self, client1, client2):        
        """
        Perform the actual swap move. This is a toy version that just exchanges messages.

        Preconditions/invariants:
         - lock is held for the state of both clients
         - state of both clients is "SWAPPING", which implies that clients have been sent 
           a "SWAP" message on the main channel, and we are waiting for them to respond on 
           the private channel. 

        Outcome invariants: 
         - no outstanding requests blocked on either swap_socket (i.e., the final 
           interaction was a response sent from the server)
         - calling self._cleanup_swap() as last action.
        """


        socket1, socket2 = self.swap_sockets[client1], self.swap_sockets[client2]

        msg1 = socket1.recv()
        msg2 = socket2.recv()

        socket1.send(msg2)
        socket2.send(msg1)

        self._cleanup_swap(client1, client2)

    def do_swap(self, client1, client2):
        addr1 = self.name_map[client1]
        addr2 = self.name_map[client2]

        self.client_state[client1] = "SWAPPING"
        self.client_state[client2] = "SWAPPING"

        self.socket.send_multipart([addr1, b'', 'SWAP'])
        self.socket.send_multipart([addr2, b'', 'SWAP'])

        t = threading.Thread(target=self.do_swap_helper, args=(client1, client2))
        t.start()

    def start_swap_if_others_waiting(self, client):
        others = self.neighbors[client]

        swap_done = False
        for other in others:
            if other not in self.client_state: 
                # if this neighbor hasn't yet registered itself 
                # with the server, skip it
                continue

            locked = self.state_lock[other].acquire(False)
            # if we can't get the lock on this neighbor because it's 
            # doing something else, no point trying to swap with it...
            if not locked: 
                continue

            # if the other client is waiting for us, do the swap
            if self.client_state[other] == ("WAITING") and self.client_state_extra[other][0] == client:
                logging.info( "starting swap between %s %s" % ( client, other))
                swap_done = True
                # lock will be released by the do_swap method
                self.do_swap(client, other)
                break
            else:
                self.state_lock[other].release()

        return swap_done

    def wait_if_appropriate(self, client):

        others = self.neighbors[client]
        current_time = time.time()

        waiting = False
        for other in others:
            et = self.estimate_done_time(other)
            if et is None:
                continue

            pair_key = tuple(sorted((client, other)))
            elapsed = current_time - self.last_swap[pair_key] 
            if elapsed > self.min_swap_s:
                try:
                    predicted_wait = et - current_time
                except:
                    # if current state is not a number, e.g., currently swapping
                    continue

                logging.debug("predicted wait %f" % predicted_wait)
                if predicted_wait < self.allowable_wait_s or elapsed > self.max_swap_s:
                    self.client_state[client] = "WAITING"
                    self.client_state_extra[client] = (other, current_time)
                    logging.info("setting client state %s to WAITING on %s, elapsed %.1f predicted wait %.1f" % (client, other, elapsed, predicted_wait))
                    waiting = True
                    break

        return waiting

    def update_timing_from_sync(self, client_name, checkpoint):
        """
        Bookkeeping to update time estimates when a client checks in. 
        """

        t = time.time()
        try:
            state = self.client_state[client_name]
            assert( state == "WORKING" )
            old_checkpoint, last_time = self.client_state_extra[client_name]


            # record the time elapsed since the last checkpoint as 
            # our new time estimate for that checkpoint
            elapsed = t - last_time
            self.checkpoint_times[client_name][old_checkpoint] = elapsed
        except:
            pass

        # set state to the new checkpoint and current time
        self.client_state[client_name] = "WORKING"
        self.client_state_extra[client_name] = (checkpoint, t)


    def estimate_done_time(self, client_name):
        """
        Return a unix timestamp (float) giving the estimated time of next checkin from 
        client_name. May return None if an estimate is not available. 
        """

        state = None
        try:
            # will fail if client not yet registered
            state = self.client_state[client_name]

            # will fail if this client is not currently working (e.g., is already 
            # in a swap move)
            assert( state== "WORKING" )
            checkpoint, last_time = self.client_state_extra[client_name]

            # will fail if we don't yet have a time estimate for this checkpoint
            estimate = self.checkpoint_times[client_name][checkpoint]

            return last_time + estimate
        except:
            logging.debug("cannot estimate time for '%s' in state %s" % (client_name, state))
            return None

    def initial_checkin(self, client_name, address):
        # handle registration of new clients and 
        # establishing a private socket
        self.name_map[client_name] = address
        self.name_backmap[address] = client_name

        swap_socket = self.context.socket(zmq.REP)
        swap_socket.bind('tcp://127.0.0.1:%d' % self.next_port)
        self.swap_sockets[client_name] = swap_socket
        self.state_lock[client_name] = threading.Lock()
        response = "PORT %d" % self.next_port
        self.next_port += 1

        logging.info( "registered %s on %s response %s" % (client_name, address, response))
        self.socket.send_multipart([address,b'',response])        

    def process(self):

        # block for a checkin from some client
        msg = None
        while msg is None:
            try:
                address, empty, msg = self.socket.recv_multipart()
            except zmq.error.Again:
                # we set a timeout of 1000ms on this socket so we 
                # don't block forever waiting for one slow client 
                # to check in
                self.timeout_long_waits()

        if msg.startswith("NAME"):
            client_name = msg.split()[1]
            self.initial_checkin(client_name, address)
        elif msg.startswith("SYNC"):
            parts = msg.split()
            stepno = parts[1]
            checkpoint = parts[2]

            client_name = self.name_backmap[address]

            self.state_lock[client_name].acquire()

            self.update_timing_from_sync(client_name, checkpoint)
            started_swap = self.start_swap_if_others_waiting(client_name)        
            if not started_swap:
                waiting = self.wait_if_appropriate(client_name)        

            if not started_swap and not waiting:
                self.socket.send_multipart([
                    address,
                    b'',
                    'CONTINUE',
                ])        

            if not started_swap:
                self.state_lock[client_name].release()

            logging.debug("processed %s state %s" % (client_name, self.client_state[client_name]))



        else:
            raise Exception("unrecognized command %s from client %s" % (msg, address))


    def timeout_long_waits(self):
        # clear any clients that have waited too long
        for client in self.client_state.keys():
            if self.client_state[client] == "WAITING":
                other, started_waiting = self.client_state_extra[client]
                tt = time.time()
                elapsed = tt - started_waiting
                if elapsed > self.allowable_wait_s * 4:
                    with self.state_lock[client]:
                        # this is essentially a cancelled swap
                        self.client_state[client] = "SWAPPED"
                        address = self.name_map[client]
                        self.socket.send_multipart([
                            address,
                            b'',
                            'CONTINUE',
                        ])

                        # if we timed out waiting for a particular unit, 
                        # update our estimate of times for that work unit
                        # so we don't keep waiting on it. 
                        if self.client_state[other] == "WORKING":
                            checkpoint, start_time = self.client_state_extra[other]
                            other_elapsed = tt - start_time
                            self.checkpoint_times[other][checkpoint] = max(other_elapsed, 
                                                                           self.checkpoint_times[other][checkpoint])
                        else:
                            logging.warning("client %s was waiting on %s but %s is not working?????" % (client, other, other))

                    logging.info("cancelled swap for %s after waiting %.1fs" % (client, elapsed))



def main():
    rootLogger = logging.getLogger()
    rootLogger.setLevel("INFO")
    neighbors = {"c1": ("c2",),
                 "c2": ("c1", "c3"),
                 "c3": ("c2", "c4"),
                 "c4": ("c3",)}
    ev = SwapServer(neighbors=neighbors)
    while True:
        ev.process()

if __name__ == "__main__":
    main()
