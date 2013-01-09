import sys
from sigvisa import *

def main():
    s= Sigvisa()
    cursor = s.dbconn.cursor()

    evcount = 0
    lebcount = 0

    f = open(sys.argv[1], 'r')
    for line in f:
        ev = [st.strip() for st in line.split()]
        lat = float(ev[0])
        lon = float(ev[1])
        tim = float(ev[3])

        sql_query = "SELECT * from leb_origin where lon between %f and %f and lat between %f and %f and time between %f and %f" % (lon-.5, lon+.5, lat-.5, lat+.5, tim -20, tim+20)
        cursor.execute(sql_query)
        r = cursor.fetchall()
        evcount += 1
        if len(r) > 0:
            lebcount += 1

    print "evcount", evcount
    print "lebcount", lebcount

if __name__ == "__main__":
    main()
