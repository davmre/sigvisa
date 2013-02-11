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
# run a command on all the EC2 instances and copy over files
#
# Examples:
# python ec2_run.py visa-credentials.csv nimarclus 'cd netvisa; python analyze.pys -s'
# python ec2_run.py visa-credentials.csv nimarclus 'cd netvisa; python -m utils.saverun 32 temp.tar' 'netvisa/temp.tar,.'
#
# one can run this in an infinite loop
# WINDOWS:
# for /L %i in (0,0,0) do @python ec2_run.py key "cd netvisa; python analyze.py" "" & ping -n 60 127.0.0.1>NUL
# BASH:
# for ((;;)) do python ec2_run.py key 'cd netvisa; python analyze.py' '' ; sleep 60; done
#
import warnings
warnings.filterwarnings("ignore")

import sys

from sigvisa.utils import EC2

from ec2_start import ssh_cmd, scp_from

CRED_FNAME = "visa-credentials.csv"


def main():
    if len(sys.argv) != 4:
        print "Usage: python ec2_run.py <key-name> <command>"\
              " <file-list>"
        sys.exit(1)

    ec2keyname, command, file_list = sys.argv[1:]

    # connect to EC2
    ec2conn = EC2.connect_with_credfile(CRED_FNAME)

    # find all the running instances with the keypair ..
    hostnames = []
    instids = []
    for desc in ec2conn.describe_instances().structure:
        if desc[0] == "INSTANCE" and desc[6] == ec2keyname \
                and desc[14] == "running":
            instids.append(desc[1])
            hostnames.append(desc[3])

    if not len(hostnames):
        print "Error: no running host found for key", ec2keyname

    for hostidx, hostname in enumerate(hostnames):
        print "%d/%d %s %s" % (hostidx + 1, len(hostnames), hostname,
                               instids[hostidx])
        ssh_cmd(hostname, ec2keyname, command)

    # check if any files have to be copied
    if len(file_list):
        filenames = file_list.split(",")
        if len(filenames) % 2:
            print "illegal list of filenames to copy", file_list
            sys.exit(1)
        for idx in range(0, len(filenames), 2):
            print "Downloading %s -> %s " % (filenames[idx], filenames[idx + 1]),
            for hostname in hostnames:
                scp_from(hostname, ec2keyname, filenames[idx], filenames[idx + 1])
                print ".",
            print "done"


if __name__ == "__main__":
    main()
