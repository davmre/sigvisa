import numpy as np
import scipy as sp
import pywt
import matplotlib.pyplot as plt
from gaussian import *
from gridutil import *
from data import *
import models.noise.armodel.learner as lnr
from matplotlib.backends.backend_pdf import PdfPages
import database.db
from database.dataset import *
import utils.geog
from plotting.event_heatmap import get_eventHeatmap
import sys
import matplotlib.mlab as mlab
import wave
import os
import traceback, pdb


def mpd4test(fmodel, trnmodel, testdom, trndom, i, grid, n):
    margin = 0.2
    pd,fr = pd4test(fmodel,trnmodel,testdom,trndom,i,grid)
    pd += margin
    for temp in range(n-1):
        currpd,currfr = pd4test(fmodel,trnmodel,testdom,trndom,i,grid)
        pd *= (currpd + margin)
    return pd

def pd4test(fmodel, tmodel, testdom, trndom, i, grid):
    fullrange = fmodel.sample()
    testrange = fullrange[i]
    trnrange = np.append(fullrange[:i],fullrange[i+1:],axis=0)
    testlnr = GPLearner(tmodel, trnrange)
    pd = testlnr.lklhood(grid,testrange)
    return pd, fullrange



def gp_likelihood_fn(testevid, feature, index, params, events=None):
    """
    Given a test evid, train a GP on all remaining events, using
    representation "feature" and params "param", and targeting the
    specific feature specified by "index".

    Return a function f : (lon, lat) -> likelihood of testevid's feature value at that location.

    """
    if events is None:
        events = validevents2(ex=testevid,feature=feature)
    inputs = inputs2d(events)
    outputs = gpvals(events, index)
    gpm = GPModel(inputs,params=params)
    gpl = GPLearner(gpm, outputs)

    teste = Event(testevid,feature=feature)
    output = teste.gpval[index]

    fn = lambda lon, lat : gpl.lklhood(np.array([(lat, lon),]), output)

    return fn

def gp2d(testevid, feature, index, params, gridmin, gridmax, res=40):
    events = validevents2(ex=testevid,feature=feature)
    teste = Event(testevid,feature=feature)
    input = [teste.lat, teste.lon]
    output = teste.gpval[index]
    inputs = inputs2d(events)
    outputs = gpvals(events, index)

    gpm = GPModel(inputs,params=params)
    gpl = GPLearner(gpm, outputs)
    ingrid = makegrid(gridmin, gridmax, res=res)
    inlist = grid2list(ingrid,2)
    pdlist = gpl.lklhood(inlist, output)
    pdgrid = list2grid(pdlist, [res,res])
    return pdgrid

def draw2d(pdgrid, gridmin, gridmax, testevid):
    dr, dc = np.unravel_index(np.argmax(pdgrid),pdgrid.shape)
    res = len(pdgrid)
    targetlat = float(dr*(gridmax[0]-gridmin[0]))/(res-1)+gridmin[0]
    targetlon = float(dc*(gridmax[1]-gridmin[1]))/(res-1)+gridmin[1]
    plt.axvline(targetlon, color='red')
    plt.axhline(targetlat, color='red')
    event = Event(testevid)
    plt.axvline(event.lon, color='white')
    plt.axhline(event.lat, color='white')
    plt.imshow(pdgrid,extent=[gridmin[1],gridmax[1],gridmax[0],gridmin[0]], interpolation=None)
    predicted = Location(targetlat, targetlon, 0)
    actual = Location(event.lat, event.lon, 0)
    return predicted.dist(actual)

def princomp(A):
    """ performs principal components analysis
    (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables.

    Returns :
    coeff :
    is a p-by-p matrix, each column containing coefficients
    for one principal component.
    score :
    the principal component scores; that is, the representation
    of A in the principal component space. Rows of SCORE
    correspond to observations, columns to components.
    latent :
    a vector containing the eigenvalues
    of the covariance matrix of A.
    """
 # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted

    # sort the components
    idx = latent.argsort()[::-1]
    latent = latent[idx]
    coeff = coeff[:,idx]

    score = np.dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent


def testmega():

    evids = open("events_sorted.txt", 'r')
    out = open("outputs/locations.txt", 'a')
    for evid in evids:
        evid = int(evid)
        print "testing evid", evid
        dist1,dist0,dist5 = test_psd(evid)
        dist2, dist0, dist5 = testeigen(evid)
        out.write("%d %.2f %.2f %.2f %.2f\n" % (evid, dist1, 0, dist0, dist5))
    out.close()


def testmegap1():

    evids = open("events_sorted1.txt", 'r')
    out = open("outputs/locations_p1.txt", 'a')
    for evid in evids:
        evid = int(evid)
        print "testing evid", evid
        dist1,dist0,dist5 = test_psd(evid)
        out.write("%d %.2f %.2f %.2f\n" % (evid, dist1, dist0, dist5))
    out.close()

def testmegae1():

    evids = open("events_sorted1.txt", 'r')
    out = open("outputs/locations_e1.txt", 'a')
    for evid in evids:
        evid = int(evid)
        print "testing evid", evid
        dist1,dist0,dist5 = testeigen(evid)
        out.write("%d %.2f %.2f %.2f\n" % (evid, dist1, dist0, dist5))
    out.close()



def testmegap2():

    evids = open("events_sorted2.txt", 'r')
    out = open("outputs/locations_p2.txt", 'a')
    for evid in evids:
        evid = int(evid)
        print "testing evid", evid
        dist1,dist0,dist5 = test_psd(evid)
        out.write("%d %.2f %.2f %.2f\n" % (evid, dist1, dist0, dist5))
    out.close()

def testmegae2():

    evids = open("events_sorted2.txt", 'r')
    out = open("outputs/locations_e2.txt", 'a')
    for evid in evids:
        evid = int(evid)
        print "testing evid", evid
        dist1,dist0,dist5 = testeigen(evid)
        out.write("%d %.2f %.2f %.2f\n" % (evid, dist1, dist0, dist5))
    out.close()



def testgeteigenvalues():
    events = validevents2(ex=None)
    X = event_signal_matrix(events)
    coeff, score,latent = princomp(X)
    print "computed principle components"
    np.savetxt('coeff.txt', coeff)
    np.savetxt('score.txt', score)
    np.savetxt('latent.txt', latent)

def get_pca_events(testevid, ncomponents):

    try:
        data = np.load('%d.npz' % testevid)
        coeff =  data['coeff']
        feature = lambda data: wave.eigbasis(coeff[:, :ncomponents], data)
        print "loaded principle components from file"
    except:
        print "computing principle components"
        events = validevents2(ex=testevid,feature=wave.psd)
        X = event_signal_matrix(events)
        print X.shape
        coeff, score,latent = princomp(X)
        print "computed principle components"
        feature = lambda data: wave.eigbasis(coeff[:, :ncomponents], data)
        np.savez("%d.npz"%testevid, coeff=coeff)


    events = validevents2(ex=testevid,feature=feature)
    return events, feature

def test21():
    testevid = 4689462
    compevid = 4686108
    otherevid = 4657242
    ncomponents = 15

    events, feature = get_pca_events(testevid, ncomponents)
    te = Event(testevid, feature=feature)
    ce = Event(compevid, feature=feature)
    oe = Event(otherevid, feature=feature)

    plt.plot(te.gpval[0:ncomponents])
    plt.plot(ce.gpval[0:ncomponents])
    plt.plot(oe.gpval[0:ncomponents])
    plt.savefig("outputs/%d_eigensimple/features_sidebyside.png" % testevid)
    plt.close()

    simple_events = [ce, oe]

    params = (0.8, 1, 0.1) # length, vs, vn

    lonbounds = [80.5, 83]
    latbounds = [34.2, 36.8]
    res = 100

    plot_by_feature("eigensimple", simple_events, feature, testevid, params, res, lonbounds, latbounds, ncomponents)


def test20():
    testevid = 4689462 #4653310
    testeigen(testevid)

def plot_by_feature(label, events, feature, testevid, params, res, lonbounds, latbounds, ncomponents):
    te = Event(testevid, feature=feature)

    total_hm = EventHeatmap(f = lambda lon, lat : 1, n=res, lonbounds=lonbounds, latbounds =latbounds)
    total_hm.add_stations("MKAR")
    total_hm.add_events(event_locations(events))
    total_hm.set_true_event(te.lon, te.lat)

    dist0 =  999
    dist5 = 999
    os.system("mkdir outputs/%d_%s" %(testevid,label)) # saves results to outputs folder
    for index in range(ncomponents): # 'index' indexes values computed by feature function
        gp_lklhood = gp_likelihood_fn(testevid,feature,index,params, events=events)
        hm = EventHeatmap(gp_lklhood, n=res, lonbounds=lonbounds, latbounds = latbounds)
        hm.add_stations("MKAR")
        hm.add_events(event_locations(events))
        dist = hm.set_true_event(te.lon, te.lat)

        total_hm = hm * total_hm

        hm.savefig("outputs/%d_%s/%d.png" %(testevid,label,index), event_alpha=1, title="", colorbar=False)
        total_hm.savefig("outputs/%d_%s/total_%d_components.png" % (testevid,label, index), event_alpha=1, title="", colorbar=False)
        print "saved component %d, dist %.2f" % (index, dist)

        if index==0:
            dist0= total_hm.set_true_event(te.lon, te.lat)
        if index==4:
            dist5= total_hm.set_true_event(te.lon, te.lat)

    total_hm.savefig("outputs/%d_%s/total.png" %(testevid, label), title="", colorbar=False)

    dist = total_hm.set_true_event(te.lon, te.lat)
    return dist, dist0, dist5


def testeigen(testevid):
    ncomponents = 10

    events, feature = get_pca_events(testevid, ncomponents)


    params = (0.2, 2, 0.2) # length, vs, vn


    locations = event_locations(events)
    print locations
    min_locations = np.min(locations, axis=0)
    max_locations = np.max(locations, axis=0)
    lonbounds = [min_locations[0], max_locations[0]]
    latbounds = [min_locations[1], max_locations[1]]

#    lonbounds = [75, 88]
#    latbounds = [32.5, 45]
    res = 50

    return plot_by_feature("eigen", events, feature, testevid, params, res, lonbounds, latbounds, ncomponents)

def test19():

    testevid = 4689462
    compevid = 4686108

    gridmin = [32.5,80] # lat, lon min
    gridmax = [37.5,85] # lat, lon max

    ncomponents = 15

    pdgrid = np.ones([40,40]) # cumulative prob distribution grid

    feature = wave.psd
    events = validevents2(feature=feature)
    X = event_signal_matrix(events)
    print X.shape
    coeff, score,latent = princomp(X)
#    coeff = np.loadtxt("coeff.np") #(principle components)
#    score = np.loadtxt("score.np") #(coefficients for each training example
#    latent = np.loadtxt("latent.np") #eigenvalues for the components

    feature = lambda data: wave.eigbasis(coeff[:, 0:ncomponents], data)
    events = validevents2(feature=feature)

    te = Event(testevid, feature=feature)
    ce = Event(compevid, feature=feature)

    plt.plot(te.gpval)
    plt.plot(ce.gpval)
    plt.savefig("outputs/test19/features_sidebyside.png")
    plt.close()

    for e in events:
#    for i in range(1):
#        e = te

        plt.plot(e.gpval)
        plt.savefig("outputs/test19/%d_features.png" % e.evid)
        plt.close()

        plt.plot(e.x)
        plt.savefig("outputs/test19/%d_wave.png" % e.evid)
        plt.close()

        plt.plot(wave.reconstruct(coeff, e.gpval))
        plt.savefig("outputs/test19/%d_reconstruct.png" % e.evid)
        plt.close()

    for i in range(ncomponents):
        plt.plot(coeff[:, i])
        plt.savefig("outputs/test19/component_%d.png" % i)
        plt.close()



# for quickly viewing how feature function transform the real data
def test18():
    testevid = 4689462
    compevid = 4686108
    te = Event(testevid, feature=wave.psd)
    ce = Event(compevid, feature=wave.psd)
    x = np.linspace(1.25, 2.5, len(te.gpval))
    plt.plot(x, te.gpval)
    plt.plot(x, ce.gpval)
    print te.dist(ce)
    plt.savefig("outputs/test18.png")

def test17():
#    testevid = 4653310
    testevid = 4689462
    test_psd(testevid)

def test_psd(testevid):

    print "loading events...",
    feature = wave.psd # feature function used
    events = validevents2(ex=testevid,feature=feature)
#    compevid = 4686108
#    otherevid = 4653310
#    otherevids = [4649838, 4755120, 4695731, 4706242, 4649673, 4677818]
    te = Event(testevid, feature=feature)
#    ce = Event(compevid, feature=feature)
#    oe = Event(otherevid, feature=feature)
#    events = [ce, oe]
#    for oevid in otherevids:
#        oe = Event(oevid, feature=feature)
#        events.append(oe)


#    plt.clf()
#    plt.plot(te.gpval)
#    plt.plot(ce.gpval)
#    plt.plot(oe.gpval)
#    plt.savefig("outputs/%d_psd/features_sidebyside.png" % testevid)


    locations = event_locations(events)
    print locations
    min_locations = np.min(locations, axis=0)
    max_locations = np.max(locations, axis=0)
    lonbounds = [min_locations[0], max_locations[0]]
    latbounds = [min_locations[1], max_locations[1]]

    params = (.2, 1, 0.1) # length, vs, vn
    print "done"

    lonbounds = [80.5, 83]
    latbounds = [33.5, 36.5]
    res = 50
#    ncomponents=len(te.gpval)
    ncomponents=20

    return plot_by_feature("psd", events, feature, testevid, params, res, lonbounds, latbounds, ncomponents)

def test16():
    evid1 = 4686108
    evid2 = 4689462
    evid3 = 4650871
    e1 = get_event(evid1)
    e2 = get_event(evid2)
    e3 = get_event(evid3)
    print e1.dist(e3)
    def filter(x):
        cA1, cD1 = pywt.dwt(x,'db1') #cA1 0-10 Hz
        cA2, cD2 = pywt.dwt(cA1,'db1') #cA2 0-5 Hz
        cA3, cD3 = pywt.dwt(cA2, 'db1') #cA3 0-2.5Hz
        cA4, cD4 = pywt.dwt(cA3, 'db1') #cD4 1.25-2.5 Hz
        cA5, cD5 = pywt.dwt(cD4, 'db1') #cA5 1.25-1.875 Hz
        return x

    f1 = filter(e1.x)
    f2 = filter(e2.x)
    f3 = filter(e3.x)



    plt.plot(f1)
    plt.plot(f2)
    plt.plot(f3)
    plt.show()

    y1 = np.fft.rfft(filter(e1.x))
    y2 = np.fft.rfft(filter(e2.x))
    y3 = np.fft.rfft(filter(e3.x))


    def convert(x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 2*np.pi+x[i]
        return x

    plt.plot(convert(np.angle(y1))[:100])
    plt.plot(convert(np.angle(y2))[:100])
    plt.plot(convert(np.angle(y3))[:100])
    plt.show()

    plt.plot(np.abs(y1)[:100])
    plt.plot(np.abs(y2)[:100])
    plt.plot(np.abs(y3)[:100])
    plt.show()

def test15():
    evid1 = 4686108
    evid2 = 4689462
    list = closest2(evid=evid1)
    evid3 = list[-3][1]
    evid3 = 4657242
    e1 = get_event(evid1)
    e2 = get_event(evid2)
    e3 = get_event(evid3)
    y1 = np.fft.rfft(e1.x)
    y2 = np.fft.rfft(e2.x)
    y3 = np.fft.rfft(e3.x)


    t = 150
    x = np.linspace(0,20,len(y1))
    plt.plot(x[:t],y1[:t])
    plt.plot(x[:t],y2[:t])
    plt.plot(x[:t],y3[:t])
    plt.show()

    y1p = []
    x1p = []
    y1n = []
    x1n = []

    y2p = []
    x2p = []
    y2n = []
    x2n = []

    def divide(y, xp, yp, xn, yn, limit=None):
        if limit == None:
            limit = len(y)
        for i in range(limit):
            if y[i] >= 0:
                yp.append(y[i])
                xp.append(i)
            else:
                yn.append(-y[i])
                xn.append(i)

    divide(y1,x1p,y1p,x1n,y1n,100)
    divide(y2,x2p,y2p,x2n,y2n,100)

    plt.plot(x1p,y1p)
    plt.plot(x2p,y2p)
    plt.show()

def test14():
    evid1 = 4686108
    evid2 = 4689462
    list = closest2(evid=evid1)
    evid3 = list[-4][1]
#    evid3 = 4657242
    e1 = get_event(evid1)
    e2 = get_event(evid2)
    e3 = get_event(evid3)

    plt.plot(e3.x)

    plt.show()

    nf = 512
    y1, x1= mlab.psd(e1.x,NFFT=nf,Fs=40)
    y2, x2 = mlab.psd(e2.x,NFFT=nf,Fs=40)
    y3, x3 = mlab.psd(e3.x,NFFT=nf,Fs=40)
    t = 100
    plt.plot(x1,np.log(y1))
    plt.plot(x2,np.log(y2))
    plt.plot(x3,np.log(y3))
    plt.show()

def test13():
    l = closest2()
    k = 5
    fl = l[:k]
    sl = l[-k:]
    fl.extend(sl)
    for x in fl:
        dist, evid1, evid2 = x
        print "creating pdf for %d-%d..." %(evid1,evid2)
        createpdf(evid1, evid2)

def test12():
    f = open('closest.txt', 'w')
    list = closest2()
    for x in list:
        f.write("%d, %d, %f\n" %(x[1],x[2], x[0]))
    f.close()

def createpdf(evid1, evid2, N=8):

    e1 = get_event(evid1)
    e2 = get_event(evid2)
    name = 'outputs/%f_%d-%d.pdf' %(e1.dist(e2), evid1, evid2)
    print "creating %s..." %name
    pdf = PdfPages(name)
    Fs=40.0

    plt.subplot(411)
    plt.plot(e1.x)
    plt.title("Origianl")
    plt.subplot(412)
    plt.plot(e2.x)
    plt.subplot(413)
    plt.psd(e1.x, Fs=Fs)
    plt.psd(e2.x, Fs=Fs)
    plt.subplot(414)
    plt.plot(np.correlate(e1.x,e2.x,mode='full'))
    plt.savefig(pdf, format='pdf')
    plt.close()


    cA1, cD1 = pywt.dwt(e1.x,'db1')
    cA2, cD2 = pywt.dwt(e2.x,'db1')
    max = 20.0
    for i in range(N):
        mid = max / 2.0
        Fs /= 2.0
        plt.subplot(411)
        plt.plot(cA1)
        plt.title("0-%f Hz" %(mid))
        plt.subplot(412)
        plt.plot(cA2)
        plt.subplot(413)
        plt.psd(cA1, Fs=Fs)
        plt.psd(cA2, Fs=Fs)
        plt.subplot(414)
        plt.plot(np.correlate(cA1,cA2,mode='full'))


        plt.savefig(pdf, format='pdf')
        plt.close()


        plt.subplot(411)
        plt.plot(cD1)
        plt.title("%f-%f Hz" %(mid, max))
        plt.subplot(412)
        plt.plot(cD2)
        plt.subplot(413)
        plt.psd(cD1, Fs=Fs)
        plt.psd(cD2, Fs=Fs)
        plt.subplot(414)
        plt.plot(np.correlate(cD1,cD2,mode='full'))
        plt.savefig(pdf, format='pdf')
        plt.close()

        max /= 2.0

        cA1, cD1 = pywt.dwt(cA1, 'db1')
        cA2, cD2 = pywt.dwt(cA2, 'db1')

    pdf.close()

def test10():
    evid1 = 4686108
    evid2 = 4689462
    evid3 = 4650871
    e1 = get_event(evid1)
    e2 = get_event(evid2)
    e3 = get_event(evid3)
    x1 = e1.data-np.mean(e1.data)
    x2 = e2.data-np.mean(e2.data)
    x3 = e3.data-np.mean(e3.data)
    plt.plot(e1.data)
    plt.plot(e2.data)
    plt.plot(e3.data)
    plt.show()
    a=1
    FS = 40
    FC = 1.5/(0.5*FS)
    N = 41
    b=signal.firwin(N,FC)
    y1 = signal.lfilter(b,a,x1)
    y2 = signal.lfilter(b,a,x2)
    y3 = signal.lfilter(b,a,x3)
    plt.plot(y1, 'b')
    plt.plot(y2, 'r')
    plt.plot(y3, 'g')
    plt.show()

# returns a res by res matrix containing the probability values

def test9():
    evid = 4733535
    events = validevents2(ex=evid)
    lons = np.zeros(len(events))
    lats = np.zeros(len(events))
    p = 5
    inputs = np.zeros([len(events), 2])
    outputs_t = np.zeros([p,len(events)])
    for i in range(len(events)):
        e = events[i]
        lons[i] = e.lon
        lats[i] = e.lat
        inputs[i] = [e.lat, e.lon]
        l = lnr.ARLearner(e.data)
        param,std = l.yulewalker(p)
        for ind in range(p):
            outputs_t[ind][i] = param[ind]
    e = get_event(evid, phase=None)
    input = [e.lat, e.lon]
    l = lnr.ARLearner(e.data)
    param, std = l.yulewalker(5)
    output_t = param

    l = 0.4
    vs = 1
    vn = 0.1
    params = (l,vs,vn)
    gpm = GPModel(inputs,params=params)
    ingrid = makegrid([32.5,80], [37.5,85], res=101)
    inlist = grid2list(ingrid,2)

    def pd(outputs, output):
        gpl = GPLearner(gpm, outputs)
        return gpl.lklhood(inlist, output)

    margin = 0.3
    pdlist = pd(outputs_t[0], output_t[0]) + margin
    for i in range(p-1):
        pdlist *= pd(outputs_t[i+1], output_t[i+1]) + margin

    pdgrid = list2grid(pdlist, [101,101])
    print np.unravel_index(np.argmax(pdgrid),pdgrid.shape)
    plt.axvline(e.lon, color='white')
    plt.axhline(e.lat, color='white')
    plt.imshow(pdgrid,extent=[80,85,37.5,32.5], interpolation='none')
    plt.show()

def test8():
    k = int(sys.argv[2])
    evid = 4653310
    events = validevents2(ex=evid)
    lons = np.zeros(len(events))
    lats = np.zeros(len(events))
    inputs = np.zeros([len(events), 2])
    outputs = np.zeros(len(events))
    for i in range(len(events)):
        e = events[i]
        lons[i] = e.lon
        lats[i] = e.lat
        inputs[i] = [e.lat, e.lon]
        l = lnr.ARLearner(e.data)
        param,std = l.yulewalker(5)
        outputs[i] = param[k]

    e = get_event(evid, phase=None)
    input = [e.lat, e.lon]
    l = lnr.ARLearner(e.data)
    param, std = l.yulewalker(5)
    output = param[k]

    l = 0.2
    vs = 0.5
    vn = 0.1
    params = (l,vs,vn)
    gpm = GPModel(inputs,params=params)
    gpl = GPLearner(gpm, outputs)
    ingrid = makegrid([32.5,80], [37.5,85], res=101)
    inlist = grid2list(ingrid,2)
    outlist = gpl.lklhood(inlist, output)
    outgrid = list2grid(outlist, [101,101])
    plt.axvline(e.lon, color='white')
    plt.axhline(e.lat, color='white')
    plt.imshow(outgrid,extent=[80,85,37.5,32.5],interpolation='none')
    print outputs
    print e.lon, e.lat

    plt.show()

def test7():
    events = validevents("GNI")
    print len(events)
    inputs = np.zeros([len(events),2])
    outputs = np.zeros(len(events))
    for i in range(len(events)):
        e = events[i]
        inputs[i] = [e.lat, e.lon]
        outputs[i] = e.y
    ingrid = makegrid([-90,-180],[90,180],res=101)
    inlist = grid2list(ingrid, 2)
    gpm = GPModel(inputs)
    lnr = GPLearner(gpm, outputs)
    outlist, vars, logp = lnr.predict(inlist)
    outgrid = list2grid(outlist, [101,101])
    plt.imshow(outgrid,interpolation='none')

    plt.show()

def test6():
    evids = validevids("GNI")
    for evid in evids:
        e = get_event(evid)
        plt.plot(e.data)
        plt.show()

def test5():
    evids = validevids("LPAZ")
    print len(evids)
    for evid in evids:
        try:
            e = get_event(evid)
            plt.plot(e.lat, e.lon, "o")
        except:
            pass
    plt.show()

def test4():
    if len(sys.argv) < 3:
        print "need two evid to compare their wiggles."
    siteid = "2"
    phase = "0.70_1.00"
    evid1 = sys.argv[1]
    evid2 = sys.argv[2]
    evid3 = sys.argv[3]
    evid4 = sys.argv[4]
    prefix = "../sigvisa_data/wiggles/3/%s/1/%s/" %(siteid, phase)
    suffix = "_BHZ.dat"
    addr1 = "%s%s%s" %(prefix, evid1, suffix)
    addr2 = "%s%s%s" %(prefix, evid2, suffix)
    addr3 = "%s%s%s" %(prefix, evid3, suffix)
    addr4 = "%s%s%s" %(prefix, evid4, suffix)

    wiggle1 = np.loadtxt(addr1)
    wiggle2 = np.loadtxt(addr2)
    wiggle3 = np.loadtxt(addr3)
    wiggle4 = np.loadtxt(addr4)

    def plot4(d1, d2, d3, d4):
        ax1 = plt.subplot(221)
        plt.plot(d1)
        plt.title(evid1)
        ax2 = plt.subplot(223, sharex=ax1, sharey=ax1)
        plt.plot(d2)
        plt.title(evid2)
        ax3 = plt.subplot(222, sharey=ax1, sharex=ax1)
        plt.plot(d3)
        plt.title(evid3)
        ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
        plt.plot(d4)
        plt.title(evid4)
        plt.show()

    plot4(wiggle1,wiggle2,wiggle3,wiggle4)

def test3():
    l = 0.8
    vs = 1
    vn = 0.01
    params = (l,vs,vn)
    fullgrid = makegrid([0.1,0.1],[0.9,0.9],res=5)
    fulldomain = grid2list(fullgrid,2)
    grid = makegrid([0,0],[1,1],res=101)
    gd = grid2list(grid,2)
    testi = int(len(fulldomain)*np.random.rand())

    testdomain = fulldomain[testi]
    trndomain = np.append(fulldomain[:testi],fulldomain[testi+1:],axis=0)
    fullmodel = GPModel(fulldomain,params)
    trnmodel = GPModel(trndomain,params)

    pd = mpd4test(fullmodel, trnmodel, testdomain, trndomain, testi, gd, 5)
    pdgrid = list2grid(pd,[101,101])
    y,x = testdomain
    print testdomain
    print np.unravel_index(np.argmax(pdgrid),pdgrid.shape)
    plt.imshow(pdgrid, extent=[0,1,1,0],interpolation='none')
    plt.axvline(x, color='white')
    plt.axhline(y, color='white')
    plt.show()

def test2():
    l = 0.8
    vs = 1
    vn = 0.01
    params = (l,vs,vn)
    fulldomain = np.linspace(0.1,0.9,17)
    gd = np.linspace(0,1,201)
    testi = int(len(fulldomain)*np.random.rand())

    testdomain = fulldomain[testi]
    trndomain = np.append(fulldomain[:testi],fulldomain[testi+1:])
    fullmodel = GPModel(fulldomain,params)
    trnmodel = GPModel(trndomain,params)

    pd = mpd4test(fullmodel,trnmodel,testdomain,trndomain,testi,gd,10)
    maxi = np.argmax(pd)
    plt.plot(gd,pd)
    plt.axvline(testdomain,color='red')
    plt.axvline(gd[maxi],color='green')
    plt.show()

def test1():
    l = 0.5
    vs = 1
    vn = 0.01
    params = (l,vs,vn)
    fulldomain = np.linspace(0.1,0.9,17)
    gd = np.linspace(0,1,201)
    testi = int(len(fulldomain)*np.random.rand())

    testdomain = fulldomain[testi]
    trndomain = np.append(fulldomain[:testi],fulldomain[testi+1:])
    fullmodel = GPModel(fulldomain,params)
    trnmodel = GPModel(trndomain,params)

    pd1,fr1 = pd4test(fullmodel, trnmodel, testdomain, trndomain, testi, gd)
    pd2,fr2 = pd4test(fullmodel, trnmodel, testdomain, trndomain, testi, gd)
    pd3,fr3 = pd4test(fullmodel, trnmodel, testdomain, trndomain, testi, gd)

    pd = pd1*pd2*pd3

    ax1 = plt.subplot(241)
    plt.plot(fulldomain,fr1)
    plt.subplot(245, sharex=ax1)
    plt.plot(gd,pd1)
    plt.axvline(testdomain,color='red')

    ax2=plt.subplot(242)
    plt.plot(fulldomain,fr2)
    plt.subplot(246,sharex=ax2)
    plt.plot(gd,pd2)
    plt.axvline(testdomain,color='red')

    ax3=plt.subplot(243)
    plt.plot(fulldomain,fr3)
    plt.subplot(247,sharex=ax3)
    plt.plot(gd,pd3)
    plt.axvline(testdomain,color='red')

    plt.subplot(248)
    plt.plot(gd,pd)
    plt.axvline(testdomain,color='red')

    plt.show()


def main():
    num = sys.argv[1]
    method = "test%s" %num
    getattr(sys.modules[__name__], method)()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
