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

CRED_FNAME = "visa-credentials.csv"

t = time.localtime()
TIMESTR = "%04d%02d%02d%02d%02d" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                                    t.tm_min)

def main():
  tarname, dellist = package_repository()
  imgname = get_image_name()
  keyname = TIMESTR
  ec2conn = EC2.connect_with_credfile(CRED_FNAME)

  print "Image:", imgname
  
  inst_names, master = create_instances(ec2conn, imgname, keyname, 1, "m1.large")

  if len(dellist):
    print "Deleting old files",
    ssh_cmd(inst_names[master], keyname, "cd netvisa; rm %s" % " ".join(dellist))
    print "done."
  
  print "Uploading tar file",
  scp_to(inst_names[master], keyname, tarname, "netvisa/"+tarname)
  print "done."

  print "Untarring and compiling:"
  ssh_cmd(inst_names[master], keyname, "cd netvisa; tar xf %s; python setup.py build_ext --inplace; nohup python -u infer.py %s &> trace-%s.out &" % (tarname, " ".join(sys.argv[1:]), keyname))

  # wait for the run to start before calling analyze
  print "Sleeping 60 seconds"
  time.sleep(60)
  
  try:
    while True:
      print keyname, inst_names[master], master
      ssh_cmd(inst_names[master], keyname, "cd netvisa; python analyze.py")
      time.sleep(60)
  except KeyboardInterrupt:
    pass

  print "Copying results:"
  
  ssh_cmd(inst_names[master], keyname, "cd netvisa; python -m utils.saverun results-%s.tar" % keyname)
  
  scp_from(inst_names[master], keyname, "netvisa/results-%s.tar" % keyname, ".")
  scp_from(inst_names[master], keyname, "netvisa/trace-%s.out" % keyname, ".")
  
  os.system("python -m utils.loadrun results-%s.tar" % keyname)
  
  os.system("python ec2_kill.py %s" % keyname)
  
  os.system("python analyze.py -g")

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

def get_image_name():
  for line in exec_cmd(["svn", "info"]):
    vals = line.split()
    if vals[0] == "Revision:":
      return "Revision-" + vals[1]
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

if __name__ == "__main__":
  main()
  
