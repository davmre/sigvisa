# terminates all the ec2 instances which share a given key name
import warnings
warnings.filterwarnings("ignore")

import sys

from utils import EC2

CRED_FNAME="visa-credentials.csv"

def main():
  if len(sys.argv) != 2:
    print "Usage: python ec2_kill.py <key-name>"
    sys.exit(1)

  ec2keyname = sys.argv[1]

  # connect to EC2
  ec2conn = EC2.connect_with_credfile(CRED_FNAME)
  
  # find all the running instances with the keypair ..
  instids = []
  for desc in ec2conn.describe_instances().structure:
    if desc[0] == "INSTANCE" and desc[6] == ec2keyname \
           and desc[14] != "terminated":
      instids.append(desc[1])

  # .. and kill them
  if not len(instids):
    print "Error: No running instance has key", ec2keyname
  else:
    descs = ec2conn.terminate_instances(instids)
    for desc in descs.structure:
      if desc[0] == "INSTANCE":
        print "%s: %s -> %s" % (desc[1], desc[2], desc[3])

  # now delete the key pair
  print ec2conn.delete_keypair(ec2keyname)
  
  # all done!
  
if __name__ == "__main__":
  main()
  
