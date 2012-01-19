# EC2 image manipulation commands
import warnings
warnings.filterwarnings("ignore")

import sys

from utils import EC2

CRED_FNAME="visa-credentials.csv"

def main():
  if (len(sys.argv) < 2 or
      (sys.argv[1] not in ('create', 'delete', 'list'))
      or (sys.argv[1] == 'create' and len(sys.argv) != 4)
      or (sys.argv[1] == 'delete' and len(sys.argv) != 3)
      or (sys.argv[1] == 'list' and len(sys.argv) != 2) ):
    print """\
Usage: python ec2_image.py command
  where valid commands are:
  create <image-name> <instance-id> 
  delete <image-name
  list
  """
    sys.exit(1)

  cmd_prefix = sys.argv[1]

  # connect to EC2
  ec2conn = EC2.connect_with_credfile(CRED_FNAME)

  if cmd_prefix == 'create':
    create_image(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'delete':
    delete_image(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'list':
    list_images(ec2conn)
  else:
    assert false

def create_image(ec2conn, img_name, instid):
  print "Creating image", img_name, "from instance", instid
  print ec2conn.create_image(img_name, instid)

def delete_image(ec2conn, img_name):
  print "Deleting image", img_name
  ec2images = ec2conn.describe_images(owners=["self"])

  for _,imageid, imageloc, imageownerid, imagestate,imageispub,imgarch, imgtype, imgkernel, imageram, imagename in ec2images.structure:
    if imagename == img_name:
      print ec2conn.deregister_image(imageid)
      break
  else:
    print "Error: Image %s not found" % img_name

def list_images(ec2conn):
  ec2images = ec2conn.describe_images(owners=["self"])

  for _,imageid, imageloc, imageownerid, imagestate,imageispub,imgarch, imgtype, imgkernel, imageram, imagename in ec2images.structure:
    print imageid, imagestate, imgarch, "'%s'" % imagename
  

if __name__ == "__main__":
  main()
