import MySQLdb

def connect():
  import os
  if os.name in ['posix']:
    # on linux we don't use named pipes
    dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos")
  elif os.name in ['nt']:
    # on windows we need named pipes
    dbconn = MySQLdb.connect(named_pipe=True, user="ctbt", db="ctbt3mos")

  return dbconn

