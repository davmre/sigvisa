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
# EC2 instance manipulation commands
import warnings
warnings.filterwarnings("ignore")

import sys

from sigvisa.utils import EC2
from ec2_start import ssh_cmd, scp_to, scp_from

CRED_FNAME = "visa-credentials.csv"

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

def main():
  if (len(sys.argv) < 2 or
      (sys.argv[1] not in ('start', 'stop', 'terminate', 'list', 'ssh',
                           'scpto', 'scpfrom'))
      or (sys.argv[1] == 'start' and len(sys.argv) != 3)
      or (sys.argv[1] == 'stop' and len(sys.argv) != 3)
      or (sys.argv[1] == 'terminate' and len(sys.argv) != 3)
      or (sys.argv[1] == 'ssh' and len(sys.argv) < 3)
      or (sys.argv[1] == 'scpto' and len(sys.argv) != 4)
      or (sys.argv[1] == 'scpfrom' and len(sys.argv) != 4)
      or (sys.argv[1] == 'list' and len(sys.argv) != 2)
      ):
    print """\
Usage: python ec2_inst.py command
  where valid commands are:
  start inst-id
  stop inst-id
  terminate inst-id
  ssh inst-id [commands]
  scpto inst-id local-file1,remote-file1,local-file2,remote-file2,...
  scpfrom inst-id remote-file1,local-file1,remote-file2,local-file1,...
  list
  """
    sys.exit(1)

  cmd_prefix = sys.argv[1]

  # connect to EC2
  ec2conn = EC2.connect_with_credfile(CRED_FNAME)

  if cmd_prefix == 'start':
    start_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'stop':
    stop_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'terminate':
    terminate_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'ssh':
    ssh_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'scpto':
    scpto_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'scpfrom':
    scpfrom_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'ssh':
    scpfrom_inst(ec2conn, *sys.argv[2:])
  elif cmd_prefix == 'list':
    list_insts(ec2conn)
  else:
    assert False

def start_inst(ec2conn, instid):
  print ec2conn.start_instances([instid])

def stop_inst(ec2conn, instid):
  print ec2conn.stop_instances([instid])

def terminate_inst(ec2conn, instid):
  print ec2conn.terminate_instances([instid])

def ssh_inst(ec2conn, instid, *cmds):
  # determine the key name and hostname
  inst = EC2.find_instid(ec2conn, instid)
  host, state, key = inst[3], inst[5], inst[6]

  if state != "running" or host is None:
    print "Error: instance is not running"
    sys.exit(1)

  ssh_cmd(host, key, " ".join(cmds))

def scpto_inst(ec2conn, instid, filelist):
  # determine the key name and hostname
  inst = EC2.find_instid(ec2conn, instid)
  host, state, key = inst[3], inst[5], inst[6]

  if state != "running" or host is None:
    print "Error: instance is not running"
    sys.exit(1)

  filenames = filelist.split(",")
  if len(filenames) % 2:
    print "illegal list of filenames to copy", file_list
    sys.exit(1)
  for idx in range(0, len(filenames), 2):
    print "Uploading %s -> %s " % (filenames[idx], filenames[idx+1]),
    scp_to(host, key, filenames[idx], filenames[idx+1])
    print "done"

def scpfrom_inst(ec2conn, instid, filelist):
  # determine the key name and hostname
  inst = EC2.find_instid(ec2conn, instid)
  host, state, key = inst[3], inst[5], inst[6]

  if state != "running" or host is None:
    print "Error: instance is not running"
    sys.exit(1)

  filenames = filelist.split(",")
  if len(filenames) % 2:
    print "illegal list of filenames to copy", file_list
    sys.exit(1)
  for idx in range(0, len(filenames), 2):
    print "Downloading %s -> %s " % (filenames[idx], filenames[idx+1]),
    scp_from(host, key, filenames[idx], filenames[idx+1])
    print "done"

def list_insts(ec2conn):
  ec2insts = ec2conn.describe_instances()

  for inst in ec2insts.structure:
    if inst[0] == 'INSTANCE' and inst[5] != 'terminated':
      for tag in ec2conn.describe_tags(inst[1]).structure:
        if tag[0] == "tagSet" and tag[1]=="Name":
          name = tag[2]
          break
      else:
        name = '<empty>'
      print inst[1], inst[5], name


if __name__ == "__main__":
  main()
