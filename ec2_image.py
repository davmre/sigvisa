# EC2 image manipulation commands
import warnings
warnings.filterwarnings("ignore")

import sys

from utils import EC2

def main():
  if (len(sys.argv) < 3 or
      (sys.argv[2] not in ('create', 'delete', 'list'))
      or (sys.argv[2] == 'create' and len(sys.argv) != 5)
      or (sys.argv[2] == 'delete' and len(sys.argv) != 4)
      or (sys.argv[2] == 'list' and len(sys.argv) != 3) ):
    print """\
Usage: python ec2_image.py <credentials-file> command
  where valid commands are:
  create <image-name> <instance-id> 
  delete <image-name
  list
  """
    sys.exit(1)

  cred_fname, cmd_prefix = sys.argv[1:3]

  # connect to EC2
  ec2conn = EC2.connect_with_credfile(cred_fname)

  if cmd_prefix == 'create':
    create_image(ec2conn, *sys.argv[3:])
  elif cmd_prefix == 'delete':
    delete_image(ec2conn, *sys.argv[3:])
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
