import os



def connect(unix_socket=None):
  if os.getenv("VISA_ORA_USER") is not None:
    import cx_Oracle
    dbconn = cx_Oracle.connect(user=os.getenv("VISA_ORA_USER"), password=os.getenv("VISA_ORA_PASS"))
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
    elif os.name in ['nt']:
      # on windows we can use named pipes
      try:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos")
      except:
        dbconn = MySQLdb.connect(named_pipe=True, user="ctbt", db="ctbt3mos")
  
  return dbconn

