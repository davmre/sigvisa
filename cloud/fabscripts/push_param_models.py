from fabric.api import *

# push param models from local install/mysql to remote install. 

from sigvisa.database.db import connect
import csv
import tarfile
import os
import sys

import logging ; logging.basicConfig(level=logging.DEBUG)


#env.hosts = open('fabric_hosts', 'r').readlines()
env.hosts = ['sigvisa%d.cloudapp.net' % i for i in range(1, 2)]

#env.use_ssh_config = True
env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'

remote_sigvisa_home = "/home/sigvisa/python/sigvisa"

def dump_local_models(runid, dump_fname, shrinkage_iter=5):
    runid = int(runid)
    shrinkage_iter = int(shrinkage_iter)
    if shrinkage_iter < 0:
        # noise models
        cond = "(shrinkage_iter=0 and phase like 'noise%%')" 
    else:
        cond = ("shrinkage_iter=%d" % shrinkage_iter)
    sql_query = "select * from sigvisa_param_model where fitting_runid=%d and %s;" % (runid, cond)

    cursor = None
    try:
        dbconn = connect()
        cursor = dbconn.cursor()

        cursor.execute("select run_name,iter from sigvisa_coda_fitting_run where runid=%d;" % runid)
        r = cursor.fetchall()
        run_name, run_iter = r[0][0], r[0][1]
        reverse_code = "insert into sigvisa_coda_fitting_run (runid, run_name, iter) values (%d, '%s', %d) on duplicate key update runid=runid;" % (runid, run_name, run_iter)

        with open(dump_fname+".csv", "wb") as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            cursor.execute(sql_query)
            rows = cursor.fetchall()
            fnames = []
            for row in rows:
                fnames.append(row[12])
                # explicitly represent None elements as NULL
                rr = [x if x is not None else "\\N" for x in row ]
                spamwriter.writerow(rr)
        reverse_code += "load data local infile \'/home/sigvisa/python/sigvisa/%s.csv\' into table sigvisa_param_model fields terminated by ',' optionally enclosed by '|' lines terminated by '\r\n';" % dump_fname
    except Exception as e:
        print e
    finally:
        if cursor is not None:
            cursor.close()
            dbconn.close()
        else:
            print "ERROR: couldn't connect to database. make sure you're in the sigvisa virtualenv?"
            sys.exit(-1)

    tgz_fname = dump_fname+".tgz"
    if os.path.exists(tgz_fname):
        print "not creating tarfile", tgz_fname, "because it already exists"
    else:
        tf = tarfile.TarFile.open(tgz_fname, mode="w:gz")
        sigvisa_home = os.getenv("SIGVISA_HOME")
        for fname in fnames:
            tf.add(os.path.join(sigvisa_home, fname), arcname=fname)
        tf.close()

    return reverse_code

def push_models(dump_fname):
    put(dump_fname+".tgz", remote_sigvisa_home, use_sudo=True)
    put(dump_fname+".csv", remote_sigvisa_home, use_sudo=True)

def load_models_remote(dump_fname, reverse_code):
    sudo("cd %s && tar xvfz %s.tgz" % (remote_sigvisa_home, dump_fname), user="sigvisa")
    sudo("source /home/sigvisa/.bash_profile && cd %s && mysql -u $VISA_MYSQL_USER -p$VISA_MYSQL_PASS -e \"%s\" $VISA_MYSQL_DB --local-infile" % (remote_sigvisa_home, reverse_code), user="sigvisa")


# this is the "main" method to drive all the rest
def deploy_models(runid, dump_fname, **kwargs):
    rc = dump_local_models(runid, dump_fname, **kwargs)
    push_models(dump_fname)
    load_models_remote(dump_fname, rc)

