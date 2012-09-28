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

import utils.gp_regression as gpr
import priors.ArrivalAmplitudePrior as ArrivalAmplitudePrior

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

data = None

for evnum, event in enumerate(events):
	for phaseid, detnum in evlist[evnum]:
		siteid = int(detections[detnum, DET_SITE_COL])
		arrtime = detections[detnum, DET_TIME_COL]

		if siteid != 2 or phaseid != 0:
			continue

		row = np.array([event[EV_LON_COL], event[EV_LAT_COL], event[EV_DEPTH_COL], event[EV_MB_COL], detections[detnum, DET_TIME_COL] - event[EV_TIME_COL], np.log(detections[detnum, DET_AMP_COL]),  np.log(detections[detnum, DET_AMP_COL]) - event[EV_MB_COL]])

		if data is None:
			data = row
		else:
			data = np.vstack([data, row])


y = data[:, 5]
mbs = data[:, 3]
n = len(y)
print n


nimar_data = data[:, [3, 2, 4, 5]]
coeffs = ArrivalAmplitudePrior.learn_amp_model(nimar_data)
print "learned coeffs", coeffs
py = []
for i,x in enumerate(nimar_data):
	py.append(ArrivalAmplitudePrior.predict_amp_model(coeffs, x[0], x[1], x[2]) )
print "rms loss sqrt(%f / %d) = sqrt(%f) = %f" % (gpr.sq_loss(y, py), n, gpr.sq_loss(y, py)/n, np.sqrt(gpr.sq_loss(y, py)/n))
print "abs loss", gpr.abs_loss(y, py)/n

my = np.ones(y.shape) * np.mean(y-mbs)
print "mean", np.mean(y)
print "mean rms loss sqrt(%f/%d) = sqrt(%f) = %f" % (gpr.sq_loss(y-mbs, my), n, gpr.sq_loss(y-mbs, my)/n, np.sqrt(gpr.sq_loss(y-mbs, my)/n))
print "mean abs loss", gpr.abs_loss(y-mbs, my)/n


X = data[:, 0:2]
y = data[:, 6]
distfn = lambda ll1, ll2: utils.geog.dist_km(ll1, ll2)
best_params = [1, 1, 100]
#gp = GaussianProcess(X, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn)
print "running rms CV..."
#r = gpr.test_kfold(X, y, 5, "distfn", best_params, distfn, None, loss_fn = gpr.sq_loss, train_loss=True)
r1, r2, r3, predA = gpr.test_kfold(data[:, [2, 4, 5]], y, 5, "se_iso", best_params, distfn, None, loss_fn = gpr.sq_loss, train_loss=True)
print "results", np.sqrt([r1, r2, r3])

print "running rms CV with mb..."
#r = gpr.test_kfold(X, y, 5, "distfn", best_params, distfn, None, loss_fn = gpr.sq_loss, train_loss=True)
r1, r2, r3, predB = gpr.test_kfold(data[:, [3, 2, 4]], data[:, 5], 5, "linear", best_params, None, None, loss_fn = gpr.sq_loss, train_loss=True)
print "results", np.sqrt([r1, r2, r3])

print "py", py
print "predA", predA
print "predB", predB
print "y", y
print "y+mb", data[:,5]

print predB.shape, data[:,5].shape
pbl = np.abs(np.reshape(predB, (-1, 1)) - np.reshape(data[:,5], (-1, 1)))
print "predB losses", pbl
pyl = np.abs(np.reshape(py, (-1, 1)) - np.reshape(data[:,5], (-1, 1)))
print "py losses", pyl

print "compare", np.mean(pbl), np.mean(pyl)


#print "running abs CV..."
#r = gpr.test_kfold(X, y, 5, "distfn", best_params, distfn, None, loss_fn = gpr.abs_loss, train_loss=True)
#print "results", r
