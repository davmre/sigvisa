# Run a command on one or more EC2 instances.
#
# PREREQUISITES:
# - create access keys (under the IAM tab) and download the credentials.csv file
# - create a disk image and give it a descriptive name
#   The image must, at a minimum, have ~/netvisa/utils/bgjob compiled (from
#   bgjob.c)
#
# USAGE
# python ec2_start.py <cred> <key> <image> <num-instances> <instance-type> <file-copy> <command>
#
# This creates a number of instances, copies the files to them, and
# then runs the command on the master.
#
# The master can expect ~/<key>.pem ~/<key>.hostkey ~/<key>.instid
# to be available to get started.
#
#
# EXAMPLE:
#
# python -u ec2_start.py visa-credentials.csv baseline-194-clus30 revision-194 30 c1.xlarge 'ec2_infer.py,netvisa/' 'cd netvisa; python -u ec2_infer.py -z 8'
#
# if you need to kill the instances try:
#   python ec2-kill.py <credentials-file> <key-name>
#
# Common instance types
#  c1.xlarge -> 8 core 64-bit  $.68/hr
#  m1.large  -> 2 core 64-bit  $.34/hr
#  m1.small  -> 1 core 32-bit  $.085/hr
#  t1.micro  -> 1-2 core 32/64 but $.02/hr
import warnings
warnings.filterwarnings("ignore")

import csv, sys, time, subprocess, os, shlex, socket

from utils import EC2

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
  # parse command line arguments
  if len(sys.argv) != 8:
    print "Usage: python ec2_start.py <credentials-file> <key-name> "\
          "<image-name> <number-of-instances> <instance-type> "\
          "<file-copy-list> <command>"
    sys.exit(1)
  
  cred_fname, ec2keyname, ec2imgname, numinst, insttype, file_list, command \
              = sys.argv[1:]

  # read the credentials file
  ec2conn = EC2.connect_with_credfile(cred_fname)
  
  # create the instances
  inst_names, master = create_instances(ec2conn, ec2imgname, ec2keyname,
                                        numinst, insttype)

  # check if any files have to be uploaded
  if len(file_list):
    filenames = file_list.split(",")
    if len(filenames) % 2:
      print "illegal list of filenames to copy", file_list
      sys.exit(1)
    for idx in range(0, len(filenames), 2):
      print "Uploading %s -> %s " % (filenames[idx], filenames[idx+1]),
      for instid, pubname in inst_names.iteritems():
        scp_to(pubname, ec2keyname, filenames[idx], filenames[idx+1])
        print ".",
      print "done"
      
  # now send the commands to the master
  if len(command):
    print "Sending commands to master:"
    ssh_cmd(inst_names[master], ec2keyname,
            'netvisa/utils/bgjob "%s"' % (command,))
  else:
    print "Warning: no commands to send to master"
  
  # reconnect since our original connection has probably timed out
  #ec2conn = EC2.AWSAuthConnection(ec2accesskey, ec2secretkey)
  
  #destroy_instances(ec2conn, ec2keyname, inst_names)
  
def create_instances(ec2conn, ec2imgname, ec2keyname, numinst, insttype):
  # download the images
  ec2images = ec2conn.describe_images(owners=["self"])

  for _,imageid, imageloc, imageownerid, imagestate,imageispub,imgarch, imgtype, imgkernel, imageram, imagename in ec2images.structure:
    if imagestate == "available" and imagename == ec2imgname:
      print "Image:", imageid, imgarch, imgtype, imgkernel, imageram
      ec2imageid = imageid
      break
  else:
    print "Image not found"
    sys.exit(1)
  
  # now create the key pair for this run
  ec2keys = ec2conn.create_keypair(ec2keyname).structure
  if ec2keys[0][0] == "KEYPAIR":
    ec2keytext = ec2keys[1][0]
  else:
    print "Error:", str(ec2keys)
    sys.exit(1)
  
  # write the key pair to a file and protect it from other users otherwise
  # ssh will complain
  fp = open(ec2keyname+".pem", "w")
  fp.write(ec2keytext)
  fp.close()
  exec_cmd("chmod 600 %s.pem" % ec2keyname)
  print "Created key file"
  
  # create a file for storing the host keys
  fp = open(ec2keyname + ".hostkey", "w")
  fp.close()
  print "Created host key file"
  
  # create the instances and get the instance-ids
  ec2insts = ec2conn.run_instances(ec2imageid, numinst, numinst, ec2keyname,
                                   instanceType=insttype)
  pending_instids = []
  for inst in ec2insts.structure:
    if inst[0] == "INSTANCE":
      pending_instids.append(inst[1])

  if not(len(pending_instids)):
    print "No instances launched"
    print ec2conn.delete_keypair(ec2keyname)
    print ec2insts
    sys.exit(1)
  
  print "Launched Instances:", pending_instids

  # wait for the instances to start running
  print "Waiting for instances ",
  inst_names = {}
  priv_names = {}
  master = None
  while len(pending_instids):
    time.sleep(5)
    for inst in ec2conn.describe_instances(pending_instids).structure:
      if inst[0] == "INSTANCE":
        if inst[1] in pending_instids and inst[5]=="running":
          pending_instids.remove(inst[1])
          inst_names[inst[1]] = inst[3]
          priv_names[inst[1]] = inst[4]
        if inst[7] == "0":              # amiLaunchIndex
          master = inst[1]
    print ".",
  print "done"
  
  print "Master's name:", inst_names[master]

  # create a file with the master and slave instance ids and names
  fp = open(ec2keyname + ".instid", "w")
  # the master will be the first entry, so it can be recognized
  print >> fp, master, inst_names[master], priv_names[master]
  for instid in inst_names.iterkeys():
    if instid != master:
      print >> fp, instid, inst_names[instid], priv_names[instid]
  fp.close()
  
  # name the instances so that the ec2keyname shows up as the name in the
  # web console
  if ec2conn.create_tags(inst_names.keys(), {"Name": ec2keyname}).structure\
     != [["true"]]:
    print "Can't set tag"
  else:
    print "Changed name of instances to", ec2keyname
  
  # the ssh server is not up right away so we can't really send any
  # requests yet
  print "Waiting for ssh servers ",
  for pubname in inst_names.values():
    while not is_ssh_up(pubname):
      pass
    print ".",
  print "done"
  
  # upload the key file to all the instances so that they can communicate
  # with each other, this will give warnings as the host keys for all the
  # servers are downloaded and stored in the .hostkey file
  print "Uploading key file ",
  scp_to(inst_names[master], ec2keyname, ec2keyname+".pem",
         "/home/ubuntu/%s.pem" % ec2keyname)
  print "done"

  # now our hostkeys file has been populated we can give it to all the
  # instances so they can talk to each other without any concern
  print "Uploading hostkey/instid files",
  scp_to(inst_names[master], ec2keyname, ec2keyname+".hostkey",
         "/home/ubuntu/%s.hostkey" % ec2keyname)
  scp_to(inst_names[master], ec2keyname, ec2keyname+".instid",
         "/home/ubuntu/%s.instid" % ec2keyname)
  print "done"
  
  return inst_names, master

def destroy_instances(ec2conn, ec2keyname, inst_names):
  # finally, we need to terminate the instances
  print "Terminating instances...",
  insts = ec2conn.terminate_instances(inst_names.keys())
  if len(insts.structure) == len(inst_names):
    print "done"
  else:
    print "failure %d instances not shutdown" \
          % (len(inst_names) - len(insts.structure))
  
  # describe the state of all these instances
  print "Final state of instances"
  print ec2conn.describe_instances(inst_names.keys())
  
  # delete the key pair
  ec2keys = ec2conn.delete_keypair(ec2keyname)
  print ec2keys

def scp_to(host, keyname, srcfile, destfile):
  retcode = exec_cmd("scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=%s.hostkey -i %s.pem %s ubuntu@%s:%s"
                     % (keyname, keyname, srcfile, host, destfile))
  return retcode

def scp_from(host, keyname, srcfile, destfile):
  retcode = exec_cmd("scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=%s.hostkey -i %s.pem ubuntu@%s:%s %s"
                     % (keyname, keyname, host, srcfile, destfile))
  return retcode

def is_ssh_up(host):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    sock.connect((host, 22))
  except:
    return False
  sock.shutdown(socket.SHUT_RDWR)
  sock.close()
  return True
  
def ssh_cmd(host, keyname, cmd):
  cmd = "ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=%s.hostkey "\
        "-i %s.pem ubuntu@%s '%s'" % (keyname, keyname, host, cmd)
  #print cmd
  retcode = exec_cmd(cmd)
  return retcode

def exec_cmd(cmd):
  proc = subprocess.Popen(shlex.split(cmd))
  proc.wait()
  return proc.returncode


def locate_keyname():
  for fname in os.listdir(os.getenv("HOME")):
    name, ext = os.path.splitext(fname)
    if ext == ".pem":
      return name
  
if __name__ == "__main__":
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
