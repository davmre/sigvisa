import sigvisa_util
from database import db
import numpy as np
import utils.geog
from priors.coda_decay.coda_decay_common import *
import csv

cursor = db.connect().cursor()

rows = load_shape_data(cursor, chan="BHZ", runid=3, phaseids=P_PHASEIDS)
csvWriter = csv.writer(open('fit_data.csv', 'wb'), delimiter=',',
                        quotechar="'", quoting=csv.QUOTE_MINIMAL)
for r in rows:
   csvWriter.writerow(r)



