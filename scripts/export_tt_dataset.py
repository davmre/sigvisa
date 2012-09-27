import matplotlib
matplotlib.use('PDF')
import database.db
from database.dataset import *
import learn, netvisa, sigvisa
#from multiprocessing import Process
import utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time

import csv

AVG_EARTH_RADIUS_KM = 6371

start_time = 1238889600
end_time = 1245456000

cursor = database.db.connect().cursor()
sites = read_sites(cursor)
phasenames, phasetimedef = read_phases(cursor)
earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)

detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)
events, orid2num = read_events(cursor, start_time, end_time, "leb")
evlist = read_assoc(cursor, start_time, end_time, orid2num, arid2num, "leb")

csvWriter = csv.writer(open('tt_data.csv', 'wb'), delimiter=',')

c = 0

for evnum, event in enumerate(events):
	if evnum % 100 == 0:
		print "saving event %d of %d" % (evnum, len(events))

	for phaseid, detnum in evlist[evnum]:
		siteid = int(detections[detnum, DET_SITE_COL])
		arrtime = detections[detnum, DET_TIME_COL]

		pred_arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
									 event[EV_DEPTH_COL],
									 event[EV_TIME_COL], phaseid,
									 siteid)
		if pred_arrtime < 0:
			c += 1
			continue

		res = arrtime - pred_arrtime

		if res > 1000:
			raise ValueError("residual too large")

		site = sites[siteid-1]
		csvWriter.writerow([siteid, phaseid, event[EV_LON_COL], event[EV_LAT_COL], event[EV_DEPTH_COL], site[0], site[1], site[2], arrtime - event[EV_TIME_COL], res ])

print "skipped", c, "arrivals"
