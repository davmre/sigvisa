# Launches infer.py on an EC2 instance
# Expects a visa-credentials.csv file in the current directory
# Expects a revision-nnn image where nnn is the current SVN revision
# packages all the modified files and all parameters into a tarball and uploads
# to the EC2 instance.
# Compiles all the files.
# Runs inference.
# Monitors for progress
# Downloads the results when done and loads into the local database.
#
import subprocess, StringIO, sys, time, tarfile, os

from utils import EC2
from ec2_start import *
from ec2_infer import parse_remove

CRED_FNAME = "visa-credentials.csv"

t = time.localtime()
TIMESTR = "%04d%02d%02d%02d%02d" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                                    t.tm_min)

def main():
  tarname, dellist = package_repository()
  revision = get_revision()
  keyname = TIMESTR
  ec2conn = EC2.connect_with_credfile(CRED_FNAME)

  par = extract_par(["-z", "--threads"])
  if par is None or par <= 1:
    insttype = "m1.large"
  else:
    insttype = "c1.xlarge"

  datafile = extract_par(["--datafile"])
  if datafile is None:
    datafilearg = ""
  else:
    datafilearg = "--datafile %s" % datafile
  
  numnodes = parse_remove(sys.argv, None, "--nodes", 1)

  imgname = parse_remove(sys.argv, None, "--image", "Revision-" + revision)
  
  print "Image:", imgname
  print "Instance Type:", insttype
  print "Number of Nodes:", numnodes
  
  inst_names, master = create_instances(ec2conn, imgname, keyname, numnodes,
                                        insttype)

  if len(dellist):
    print "Deleting old files",
    for pubname in inst_names.itervalues():
      ssh_cmd(pubname, keyname, "cd netvisa; rm %s" % " ".join(dellist))
    print "done."
  
  print "Uploading tar file",
  for pubname in inst_names.itervalues():
    scp_to(pubname, keyname, tarname, "netvisa/"+tarname)
    print ".",
  print "done."

  print "Untarring and compiling:"
  for pubname in inst_names.itervalues():
    ssh_cmd(pubname, keyname, "cd netvisa; svn update -r %s; tar xf %s;"
            " rm -fr build; rm -f netvisa.so; "
            "python setup.py build_ext --inplace" % (revision, tarname))

  print "Starting run on master"
  if numnodes == 1:
    ssh_cmd(inst_names[master], keyname, "cd netvisa; nohup python -u infer.py %s &> trace-%s.out &" % (" ".join(sys.argv[1:]), keyname))
  else:
    ssh_cmd(inst_names[master], keyname, "cd netvisa; nohup python -u ec2_infer.py %s &> trace-%s.out &" % (" ".join(sys.argv[1:]), keyname))

  # wait for the run to start before calling analyze
  print "Sleeping 60 seconds"
  time.sleep(60)

  sleepcnt = 0
  try:
    while True:
      print keyname, inst_names[master], master
      ssh_cmd(inst_names[master], keyname, "cd netvisa; python analyze.py %s"
              % datafilearg)
      time.sleep(60)
      sleepcnt += 1
      if sleepcnt % 10 == 0:
        if not ssh_py_infer(inst_names[master], keyname):
          print "No python *infer.py process on master"
          raise KeyboardInterrupt
      
  except KeyboardInterrupt:
    pass

  print "Copying results:"
  
  if numnodes == 1:
    ssh_cmd(inst_names[master], keyname,
            "cd netvisa; python -m utils.saverun results-%s.tar" % keyname)
  
  if numnodes > 1:
    scp_from(inst_names[master], keyname,
             "netvisa/results-%s-propose.tar" % keyname, ".")
  
  scp_from(inst_names[master], keyname, "netvisa/results-%s.tar" % keyname, ".")
  scp_from(inst_names[master], keyname, "netvisa/trace-%s.out" % keyname, ".")
  
  os.system("python -m utils.loadrun results-%s.tar" % keyname)
  
  os.system("python ec2_kill.py %s" % keyname)
  
  os.system("python analyze.py %s -g" % datafilearg)

  os.system("less trace-%s.out" % keyname)
  print "trace-%s.out" % keyname
  
def package_repository():
  tarname = "changes-" + TIMESTR + ".tar"
  tar = tarfile.open(name = tarname, mode="w")
  dellist = []
  
  print "Tarring:", tarname
  
  for line in exec_cmd(["svn", "status"]):
    status, fname = line.split()
    if status in ("A", "M"):
      print " ", fname
      tar.add(name = fname, arcname=fname)
    elif status == "D":
      dellist.append(fname)
      
  for fname in os.listdir("parameters"):
    if fname.endswith("Prior.txt"):
      tar.add(name = os.path.join("parameters", fname),
              arcname = "parameters/" + fname)
      #print " ", "parameters/" + fname
  
  tar.close()

  return tarname, dellist

def get_revision():
  for line in exec_cmd(["svn", "info"]):
    vals = line.split()
    if vals[0] == "Revision:":
      return vals[1]
  else:
    print "Error: svn info doesn't contain revision!"
    sys.exit(1)

def exec_cmd(cmds):
  proc = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
  out, err = proc.communicate(None)
  if proc.returncode:
    print err
    sys.exit(1)
  for line in out.split("\n"):
    if not len(line):
      continue
    yield line

def extract_par(forms):
  val = None
  for idx, par in enumerate(sys.argv):
    if par in forms:
      val = sys.argv[idx+1]
  return val

def ssh_uptime(host, keyname):
  ssh_cmd(host, keyname, "uptime > uptime.out")
  scp_from(host, keyname, "uptime.out", "uptime-%s.out" % keyname)
  fp = open("uptime-%s.out" % keyname)
  row = fp.readline()
  fp.close()

  idx = row.find("load average:")
  if idx < 0:
    return 100.

  load = float(row[idx+13:].split(", ")[0])
  return load

def ssh_py_infer(host, keyname):
  ssh_cmd(host, keyname, "ps -aef  > py-infer.out")
  scp_from(host, keyname, "py-infer.out", "py-infer-%s.out" % keyname)
  fp = open("py-infer-%s.out" % keyname)
  cnt = 0
  for line in fp:
    if line.find("python") >= 0 and line.find("infer.py") >= 0:
      cnt += 1
      #print line
  return cnt

if __name__ == "__main__":
  main()
  
