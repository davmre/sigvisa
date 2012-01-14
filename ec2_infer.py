# master process for distributed inference
#
# This will first farm out the job of birth proposals to be run by each
# node.
# The results will be collected from all the nodes and then a final pass will
# be run with the collected events.
#
# The only options that are interpreted are -k or --skip, -r or --hours,
# and -l or --label. The other options are passed to infer.py as is.
import sys, os, time, subprocess, shlex
import numpy as np

from database.dataset import *
import MySQLdb
from ec2_start import locate_keyname

# unbuffer stdout
class Unbuffered:
  def __init__(self, stream):
    self.stream = stream
  def write(self, data):
    self.stream.write(data)
    self.stream.flush()
  def __getattr__(self, attr):
    return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def main():

  args = sys.argv[1:]

  ec2keyname = locate_keyname()

  numsamples = int(parse_remove(args, "-n", "--numsamples", 10))
  
  hours = parse_remove(args, "-r", "--hours", None)
  if hours is not None:
    hours = float(hours)

  skip = float(parse_remove(args, "-k", "--skip", 0))

  label = parse_remove(args, "-l", "--label", "validation")

  # remove any description and replace it with the key name
  parse_remove(args, "-d", "--descrip", None)
  args.extend(["-d", ec2keyname])

  email = parse_remove(args, None, "--email", None)
  
  print "key", ec2keyname
  print "numsamples", numsamples
  print "label", label
  print "hours", hours
  print "skip", skip
  print "rest:", " ".join(args)
  
  # read all the instance names
  hostnames = []
  for row in open(os.path.join(os.getenv("HOME"), ec2keyname+".instid")):
    hostnames.append(row.split()[2])
  print "hosts:", hostnames
  
  # now deduce the starting and ending time of the data
  localconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", host="127.0.0.1",
                              port=3306)
  cursor = localconn.cursor()
  data_start, data_end = read_timerange(cursor, label, hours, skip)

  # get the current max runid
  cursor.execute("select max(runid) from visa_run")
  prevmaxrunid, = cursor.fetchone()

  # compute the starting and ending times of all the slaves
  data_splits = np.linspace(data_start, data_end, len(hostnames)+1)
  
  print "Splits:", data_splits

  # now divide the job among all the hosts, we give a little extra at the end
  # time to allow for detections past the end point.
  # we will also open a tunnel in the background to each of the hosts so that
  # we can monitor their progress using MySQL
  hostconns = []
  for hostidx, hostname in enumerate(hostnames):
    # open a tunnel to the host in the background
    while ssh_tunnel(hostname, ec2keyname, 3307 + hostidx, 3306):
      print "Retrying ssh_tunnel to hostidx %d host %s" % (hostidx, hostname)
    # start inference on the host also in the background
    ssh_cmd(hostname, ec2keyname, "cd netvisa; utils/bgjob python -u "
            "infer.py %s -n 0 -l %s -k %f -r %f"
            % (" ".join(args), label, data_splits[hostidx],
               min(data_splits[hostidx+1] + MAX_TRAVEL_TIME, data_end)))
    # create a connection to the host's MySQL database
    conn = MySQLdb.connect(user="ctbt", db="ctbt3mos", host="127.0.0.1",
                           port=3307 + hostidx)
    hostconns.append(conn)
  
  # first wait for the local host to get started with its run to avoid any
  # confusion in the runids
  print "Waiting for local run to start...",
  cursor = localconn.cursor()
  runid = prevmaxrunid
  while runid == prevmaxrunid:
    cursor.execute("select max(runid) from visa_run")
    runid, = cursor.fetchone()
    time.sleep(5)
  print "done"

  # NOTE: variable runid now contains the runid which is being used by
  # all the hosts to run the above propose job

  # create a new run for storing the proposal
  cursor.execute ("insert into visa_run(run_start, run_end, "
                  "data_start, data_end, descrip, numsamples, window, step) "
                  "values (now(), now(), %f, %f, '%s-propose', 0, 0, 0)" %
                  (data_start, data_end, ec2keyname))
  localconn.commit()
  cursor.execute("select max(runid) from visa_run")
  proprunid, = cursor.fetchone()

  host_cpus = []
  # for each host check if it has completed and if so copy over the events
  # and shutdown the host, also record the cpu time used
  pending_hostidcs = set(range(len(hostnames)))
  num_prop_origin = 0
  propose_start = time.time()
  while len(pending_hostidcs):
    for hostidx, hostconn in enumerate(hostconns):
      if hostidx in pending_hostidcs:
        time.sleep(5)                   # don't query too often
        hostcursor = hostconn.cursor()
        # if the run has not even started then skip it
        if not hostcursor.execute("select f1 from visa_run where runid=%d"
                                  % runid):
          continue
        f1, = hostcursor.fetchone()
        # when the run is over it populates the f1 column
        if f1 is not None:
          # we will copy over the results to the current machine
          cursor = localconn.cursor()

          # note: we don't want to query events which were proposed in the
          # extra time
          hostcursor.execute("select lon, lat, depth, time, mb from visa_origin"
                             " where runid=%d and time <= %f"
                             % (runid, data_splits[hostidx+1]))
          for row in hostcursor.fetchall():
            num_prop_origin += 1
            cursor.execute("insert into visa_origin(lon, lat, depth, time, mb,"
                           "orid, runid) values (%f, %f, %f, %f, %f, %d, %d)"
                           % (row[0], row[1], row[2], row[3], row[4],
                              num_prop_origin, proprunid))
          
          # compute the amount of time spent by this host
          host_cpu = time.time() - propose_start
          print "Host %d used %.1f cpu seconds" % (hostidx, host_cpu)
          host_cpus.append(host_cpu)
          
          cursor.execute("update visa_run set run_end = now() where runid=%d"
                         % proprunid)
          localconn.commit()
          
          # we don't need a connection to this host anymore
          hostconn.close()
          
          # shutdown the host, but don't shutdown if it's the master!!
          if hostidx > 0:
            print "shutting down hostidx", hostidx
            ssh_cmd(hostnames[hostidx], ec2keyname,
                    "sudo /sbin/shutdown -h now")
          
          # and forget about this host for future iterations
          pending_hostidcs.remove(hostidx)

  # now we will do a run on the local host using the aggregated events as the
  # proposal run
  exec_cmd("python -u infer.py %s -n %d -l %s -k %f -r %f -p %d"
           " -d '%s (#nodes=%d node-hrs: sum=%.1f max=%.1f)'"
           % (" ".join(args), numsamples, label, data_start, data_end,
              proprunid, ec2keyname, len(hostnames), sum(host_cpus) / 3600.,
              max(host_cpus)/3600.))

  # next we will save the results in a tar file
  exec_cmd("python -m utils.saverun %d %s.tar" %(proprunid+1, ec2keyname))

  # if the user has requested email then send it out and shutdown
  if email is not None:
    exec_cmd("echo | mutt -s %s -a %s.tar -- %s" % (ec2keyname, ec2keyname,
                                                    email))
    exec_cmd("sudo /sbin/shutdown -h now")

def ssh_tunnel(host, keyname, localport, remoteport):
  keypath = os.path.join(os.getenv("HOME"), keyname)
  t1 = time.time()
  retcode = exec_cmd("ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=%s.hostkey -i %s.pem -fNg -L %d:127.0.0.1:%d ubuntu@%s"
                     % (keypath, keypath, localport, remoteport, host))
  print "ssh_tunnel took %.1f s" % (time.time() - t1)
  return retcode
  
def ssh_cmd(host, keyname, cmd):
  keypath = os.path.join(os.getenv("HOME"), keyname)
  t1 = time.time()
  retcode = exec_cmd("ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=%s.hostkey -i %s.pem ubuntu@%s '%s'"
                     % (keypath, keypath, host, cmd))
  print "ssh_cmd took %.1f s" % (time.time() - t1)
  return retcode

def exec_cmd(cmd):
  proc = subprocess.Popen(shlex.split(cmd))
  proc.wait()
  return proc.returncode

def parse_remove(args, shortname, longname, default):
  """
  parse the argument and return the value
  """
  retval = default

  remove_indices = []

  longname_eq = longname + "="
  
  # scan the arguments
  for idx, val in enumerate(args):
    # skip this index if it has already been read as part of the previous one
    if len(remove_indices) and idx == remove_indices[-1]:
      continue
    
    if val == shortname or val == longname and (idx+1) < len(args):
      retval = args[idx+1]
      remove_indices.append(idx)
      remove_indices.append(idx+1)
    
    elif val.startswith(longname_eq):
      retval = val[len(longname_eq):]
      remove_indices.append(idx)

  for idx in remove_indices[::-1]:
    args.pop(idx)
  
  return retval

if __name__ == "__main__":
  main()

