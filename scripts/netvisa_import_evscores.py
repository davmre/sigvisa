import numpy as np
from sigvisa import Sigvisa
import csv

visa_origin_csv='visa_origin_semi.txt'

def main():

    s = Sigvisa()

    with open(visa_origin_csv, 'rb') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for row in reader:
            orid = int(row[4])
            evscore = float(row[6])
            assert(row[7] == str(int(row[7])))
            print orid, evscore

            s.sql("update visa_origin_new set evscore=%f where orid=%d" % (evscore, orid))
    s.dbconn.commit()

if __name__=="__main__":
    main()
