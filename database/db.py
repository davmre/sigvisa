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
import os

def connect(unix_socket=None):
  if os.getenv("VISA_ORA_USER") is not None:
    import cx_Oracle
    dbconn = cx_Oracle.connect(user=os.getenv("VISA_ORA_USER"), password=os.getenv("VISA_ORA_PASS"), threaded=True)
  else:
    import MySQLdb
    if os.name in ['posix']:
      # on linux we don't use named pipes
      if unix_socket is not None:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", unix_socket=unix_socket)
      elif "VISA_SOCKET" in os.environ:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", unix_socket=os.environ["VISA_SOCKET"])
      else:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos")

  return dbconn
