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
# terminates all the ec2 instances which share a given key name
import warnings
warnings.filterwarnings("ignore")

import sys

from sigvisa.utils import EC2

CRED_FNAME = "visa-credentials.csv"


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
