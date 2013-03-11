"""

Remove template param models from the database if they no longer exist on disk.

"""


from sigvisa import Sigvisa
import os

cursor = Sigvisa().dbconn.cursor()

sql_query = "select modelid, model_fname from sigvisa_param_model"
cursor.execute(sql_query)
models = cursor.fetchall()

basedir = os.getenv("SIGVISA_HOME")

print models
for (modelid, fname) in models:

    if not os.path.exists(os.path.join(basedir, fname)):
        print 'deleting modelid %d' % modelid
        cursor.execute('delete from sigvisa_param_model where modelid=%d' % modelid)
        Sigvisa().dbconn.commit()
    else:
        print "file %s exists!" % fname
