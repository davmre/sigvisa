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
import learn

# we will divide the work with the following time granularity
TIME_QUANT = 900

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

def main(param_dirname):

  args = sys.argv[1:]

  ec2keyname = locate_keyname()

  hours = parse_remove(args, "-r", "--hours", None)
  if hours is not None:
    hours = float(hours)

  skip = float(parse_remove(args, "-k", "--skip", 0))

  label = parse_remove(args, "-l", "--label", "validation")

  # remove any description and replace it with the key name
  descrip = parse_remove(args, "-d", "--descrip", "")
  args.extend(["-d", ec2keyname])

  email = parse_remove(args, None, "--email", None)

  # look for any datafile directive
  datafile = parse_remove(args, None, "--datafile", None)
  if datafile is not None:
    args.extend(["--datafile", datafile])
  
  print "key", ec2keyname
  print "descrip", descrip
  print "label", label
  print "datafile", datafile
  print "hours", hours
  print "skip", skip
  print "rest:", " ".join(args)
  
  # read all the instance names
  hostnames = []
  for row in open(os.path.join(os.getenv("HOME"), ec2keyname+".instid")):
    hostnames.append(row.split()[2])
  print "hosts:", hostnames

  # now read the data to deduce the starting and ending time
  if datafile is not None:
    start_time, end_time, detections, leb_events, leb_evlist,\
      sel3_events, sel3_evlist, site_up, sites, phasenames, \
      phasetimedef, sitenames \
      = learn.read_datafile_and_sitephase(datafile, param_dirname,
                                          hours = hours, skip = skip)
    
  else:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef \
                = read_data(label, hours=hours, skip=skip)
  
  # get the current max runid
  localconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", host="127.0.0.1",
                              port=3306)
  cursor = localconn.cursor()
  cursor.execute("select max(runid) from visa_run")
  prevmaxrunid, = cursor.fetchone()

  # count the number of detections in each hour and divide the hours between
  # the slaves so that each slave has approximately the same total number of
  # the square of the number of detections
  hr_cnt = np.zeros(np.ceil((end_time - start_time) / TIME_QUANT))
  for det in detections:
    if det[DET_TIME_COL] < end_time:
      hr_cnt[(det[DET_TIME_COL] - start_time) // TIME_QUANT] += 1
  hr_cnt = hr_cnt ** 2

  # the target share for each host
  tgtcnt = float(hr_cnt.sum()) / len(hostnames)

  host_ranges = []
  curr_skip = 0
  curr_cnt = 0
  for idx, cnt in enumerate(hr_cnt):
    curr_cnt += cnt
    if curr_cnt >= tgtcnt or idx == len(hr_cnt)-1:
      host_ranges.append((start_time + curr_skip * TIME_QUANT,
                          min(start_time + (idx + 1) * TIME_QUANT,
                              end_time)))
      curr_skip = idx + 1
      curr_cnt = 0

  # complete the host ranges for all the other hosts with dummy values
  for i in range(len(hostnames) - len(host_ranges)):
    host_ranges.append((end_time-50, end_time))
  
  print "Host Ranges:", host_ranges
  
  # now divide the job among all the hosts, we give each host 50 seconds
  # extra to ensure that events on the border between the two hosts are
  # not squeezed
  # we will also open a tunnel in the background to each of the hosts so that
  # we can monitor their progress using MySQL
  for hostidx, (hostname, (host_start, host_end))\
          in enumerate(zip(hostnames, host_ranges)):
    # open a tunnel to the host in the background
    while ssh_tunnel(hostname, ec2keyname, 3307 + hostidx, 3306):
      print "Retrying ssh_tunnel to hostidx %d host %s" % (hostidx, hostname)
    # start inference on the host also in the background
    ssh_cmd(hostname, ec2keyname, "cd netvisa; utils/bgjob python -u "
            "infer.py %s -n 0 -l %s -k %f -r %f"
            % (" ".join(args), label, host_start, min(host_end+50, end_time)))
  
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

  # NOTE: the variable 'runid' now contains the runid which is being used by
  # all the hosts to run the above propose job

  # create a new run for storing the proposal
  cursor.execute ("insert into visa_run(run_start, "
                  "data_start, descrip, numsamples, window, step) "
                  "values (now(), %f, '%s-propose (%s)', 0, 0, 0)" %
                  (start_time, descrip, ec2keyname))
  localconn.commit()
  cursor.execute("select max(runid) from visa_run")
  proprunid, = cursor.fetchone()

  # we are closing and reopening the connections to avoid errors caused by
  # the MySQL server going away
  localconn.close()

  host_cpus = []
  # for each host check if it has completed and if so copy over the events
  # and shutdown the host, also record the cpu time used
  pending_hostidcs = set(range(len(hostnames)))
  num_prop_origin = 0
  connect_errors = 0
  propose_start = time.time()
  while len(pending_hostidcs):
    for hostidx in range(len(hostnames)):
      if hostidx in pending_hostidcs:
        time.sleep(5)                   # don't query too often
        # connect to the MySQL database on the host
        try:
          # a connect timeout is needed because the host can be overloaded
          # due to heavy activity
          hostconn = MySQLdb.connect(user="ctbt", db="ctbt3mos",
                                     host="127.0.0.1", port=3307 + hostidx,
                                     connect_timeout=300)
        except (AttributeError, MySQLdb.OperationalError):
          print "Error connecting to MySQL db of hostidx %d" % hostidx
          connect_errors += 1
          if connect_errors > 100:
            raise
        hostcursor = hostconn.cursor()
        # if the run has not even started then skip it
        if not hostcursor.execute("select f1, data_end from visa_run "
                                  "where runid=%d" % runid):
          hostconn.close()
          continue
        host_f1, host_data_end = hostcursor.fetchone()
        # when the run is over it populates the f1 column
        if host_f1 is not None:
          # we will copy over the results to the current machine
          localconn = MySQLdb.connect(user="ctbt", db="ctbt3mos",
                                      host="127.0.0.1", port=3306)
          cursor = localconn.cursor()
    
          # we will pull over all the events (even those in the extra
          # 50 second margin)
          hostcursor.execute("select lon, lat, depth, time, mb from visa_origin"
                             " where runid=%d"
                             % (runid,))
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

          # find the minimum pending host index, its start time is the
          # maximum time that the propose run has covered
          for otheridx, (othermin, othermax) in enumerate(host_ranges):
            if otheridx in pending_hostidcs:
              curr_max_time = othermin
              break
          else:
            curr_max_time = end_time

          cursor.execute("update visa_run set run_end = now(), data_end = %f"
                         " where runid=%d" % (curr_max_time, proprunid))
          localconn.commit()
          localconn.close()
          
          # we don't need a connection to this host anymore
          hostconn.close()
          
          # shutdown the host, but don't shut it down if it's the master!!
          if hostidx > 0:
            print "shutting down hostidx", hostidx
            ssh_cmd(hostnames[hostidx], ec2keyname,
                    "sudo /sbin/shutdown -h now")
          
          # and forget about this host for future iterations
          pending_hostidcs.remove(hostidx)

  # save the results of the propose run
  exec_cmd("python -m utils.saverun %d results-%s-propose.tar"
           % (proprunid, ec2keyname))
  
  # now we will do a run on the local host using the aggregated events as the
  # proposal run
  exec_cmd("python -u infer.py %s -l %s -k %f -r %f -p %d"
           " -d '%s (%s) (#nodes=%d node-hrs: sum=%.1f max=%.1f)'"
           % (" ".join(args), label, start_time, end_time,
              proprunid, descrip, ec2keyname, len(hostnames),
              sum(host_cpus) / 3600., max(host_cpus)/3600.))

  # next we will save the results in a tar file
  exec_cmd("python -m utils.saverun %d results-%s.tar"
           % (proprunid+1, ec2keyname))

  # if the user has requested email then send it out and shutdown
  if email is not None:
    exec_cmd("echo | mutt -s %s-propose -a results-%s-propose.tar -- %s"
             % (ec2keyname, ec2keyname, email))
    exec_cmd("echo | mutt -s %s -a results-%s.tar -- %s"
             % (ec2keyname, ec2keyname, email))
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
  main("parameters")

