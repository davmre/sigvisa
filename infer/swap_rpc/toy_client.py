import numpy as np
import scipy.stats
import zmq
from optparse import OptionParser
import time
from time import sleep

parser = OptionParser()
parser.add_option("--name", dest="name", default="", type="str")
(options, args) = parser.parse_args()

# client
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5555' )

client_name = options.name

socket.send("NAME %s" % client_name)
port_msg = socket.recv()
port = int(port_msg.split()[1])

swap_socket = context.socket(zmq.REQ)
swap_socket.connect('tcp://127.0.0.1:%d' % int(port))

t0 = time.time()
overhead = 0.0
i = 0
while True:
    tasklen = np.random.rand() 
    t1 = time.time()
    socket.send("SYNC %f" % tasklen)
    resp = socket.recv()
    if resp.startswith("SWAP"):
        print "initiating swap"
        swap_socket.send("hi from client %s" % client_name)
        msg = swap_socket.recv()
        print "received message", msg
    t2 = time.time()
    overhead += t2-t1
    
    sleep(tasklen)
    i += 1

    if i % 20 == 0:
        elapsed = time.time() - t0
        print "elapsed", elapsed, "overhead", overhead, "fraction", overhead/elapsed
