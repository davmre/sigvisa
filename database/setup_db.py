import os
import sys
import string
import random
from subprocess import call


def pw_generator(size=8, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def prompt(text, default=""):
    print text + " [%s]: " % default,
    v = raw_input()
    if len(v) == 0:
        if default == "":
            raise Exception("nothing entered; no default.")
        return default
    else:
        return v

required_files = ('leb_arrival.csv', 'idcx_arrival.csv', 'leb_assoc.csv', 'leb_origin.csv', 'static_site.csv', 'static_sitechan.csv', 'idcx_wfdisc.csv', 'static_siteid.csv', 'static_phaseid.csv')

SIGVISA_HOME = os.getenv("SIGVISA_HOME")
if SIGVISA_HOME is None:
    raise Exception("environmental variable SIGVISA_HOME is not defined!")
db_dir = os.path.join(SIGVISA_HOME, "database")
os.chdir(db_dir)

csv_dir = prompt("full path to directory with CSV files", db_dir)

for fname in required_files:
    print "checking for %s ... " %(fname),
    if not os.path.exists(os.path.join(csv_dir, fname)):
        raise Exception("required file '%s' not found in '%s'" % (fname, csv_dir))
    print "found."

mysql_admin = prompt("MySQL admin user", "root")
mysql_root = prompt("MySQL root password")
db_name = prompt("name of DB to create for sigvisa", "ctbt3mos")
db_user = prompt ("name of DB user to create for sigvisa", "ctbt")
db_pass = prompt ("password for new DB user '%s'" % db_user, pw_generator())
waveform_data_dir = prompt("full pathname of directory storing waveform data (i.e. having seismic/ as a subdir)")

with open('setup_db.sql', 'r') as f:
    setup_sql = f.read()

setup_sql = setup_sql.replace("load data local infile '", "load data local infile '%s/" % csv_dir)
setup_sql = setup_sql.replace("$VISA_MYSQL_DB", db_name)
setup_sql = setup_sql.replace("$VISA_MYSQL_USER", db_user)
setup_sql = setup_sql.replace("$VISA_MYSQL_PASS", db_pass)
setup_sql = setup_sql.replace("$VISA_SIGNAL_BASEDIR", waveform_data_dir)

with open('setup_db_custom.sql', 'w') as f:
    f.write(setup_sql)


with open('setup_db_custom.sql', 'r') as f:
    cmd = ["mysql", "-u", "root", "-p%s" % mysql_root, "--local-infile", ]
    print "running %s" % " ".join(cmd) +  " < setup_db_custom.sql"
#    call(cmd, stdin=f)
print "removing setup_db_custom.sql..."
os.remove('setup_db_custom.sql')

with open('sigvisa.sql', 'r') as f:
    cmd = ["mysql", "-u", "root", "-p%s" % mysql_root, db_name]
    print "running %s" % " ".join(cmd) +  " < sigvisa.sql"
#    call(cmd, stdin=f)

print "add the following to your login script (i.e. .bashrc, or virtualenv postactivate script):"
print "export VISA_MYSQL_USER=%s" % db_user
print "export VISA_MYSQL_DB=%s" % db_name
print "export VISA_MYSQL_PASS=%s" % db_pass

print "done!"
