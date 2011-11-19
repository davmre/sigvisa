import os


def connect(unix_socket=None):
  if os.getenv("VISA_ORA_USER") is not None:
    import cx_Oracle
    dbconn = cx_Oracle.connect(os.getenv("VISA_ORA_USER"))
  else:
    import MySQLdb
    if os.name in ['posix']:
      # on linux we don't use named pipes
      if unix_socket is not None:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", unix_socket=unix_socket)
      else:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos", unix_socket="/home/dmoore/mysql/tmp/mysql.sock")
    elif os.name in ['nt']:
      # on windows we can use named pipes
      try:
        dbconn = MySQLdb.connect(user="ctbt", db="ctbt3mos")
      except:
        dbconn = MySQLdb.connect(named_pipe=True, user="ctbt", db="ctbt3mos")
  
  return dbconn

