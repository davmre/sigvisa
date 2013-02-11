# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# EC2 image manipulation commands
import warnings
warnings.filterwarnings("ignore")

import sys

from sigvisa.utils import EC2

CRED_FNAME = "visa-credentials.csv"


def main():
    if (len(sys.argv) < 2 or
        (sys.argv[1] not in ('create', 'delete', 'list'))
        or (sys.argv[1] == 'create' and len(sys.argv) != 4)
        or (sys.argv[1] == 'delete' and len(sys.argv) != 3)
            or (sys.argv[1] == 'list' and len(sys.argv) != 2)):
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

    for _, imageid, imageloc, imageownerid, imagestate, imageispub, imgarch, imgtype, imgkernel, imageram, imagename in ec2images.structure:
        if imagename == img_name:
            print ec2conn.deregister_image(imageid)
            break
    else:
        print "Error: Image %s not found" % img_name


def list_images(ec2conn):
    ec2images = ec2conn.describe_images(owners=["self"])

    for _, imageid, imageloc, imageownerid, imagestate, imageispub, imgarch, imgtype, imgkernel, imageram, imagename in ec2images.structure:
        print imageid, imagestate, imgarch, "'%s'" % imagename


if __name__ == "__main__":
    main()
