import numpy as np
import scipy as sp
import pywt
import matplotlib.pyplot as plt
from gaussian import *
from gridutil import *
from data import *
import signals.armodel.learner as lnr
from matplotlib.backends.backend_pdf import PdfPages
import database.db
from database.dataset import *
import utils.geog
import sys
import matplotlib.mlab as mlab
import wave
import os

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

def gp2d(testevid, feature, index, params, gridmin, gridmax, res=101):
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
    plt.imshow(pdgrid,extent=[gridmin[1],gridmax[1],gridmax[0],gridmin[0]], interpolation='none')
    predicted = Location(targetlat, targetlon, 0)
    actual = Location(event.lat, event.lon, 0)
    return predicted.dist(actual)

# for quickly viewing how feature function transform the real data
def test18(): 
    testevid = 4689462
    compevid = 4686108
    te = Event(testevid, feature=wave.psd)
    ce = Event(compevid, feature=wave.psd)
    plt.plot(te.gpval)
    plt.plot(ce.gpval)
    print te.dist(ce)
    plt.show()
    

def test17():
    testevid = 4653310
    feature = wave.psd # feature function used
    params = (0.4, 1, 0.1) # length, vs, vn
    gridmin = [32.5,80] # lat, lon min
    gridmax = [37.5,85] # lat, lon max
    os.system("mkdir outputs/%d" %testevid) # saves results to outputs folder
    
    pdgrid = np.ones([101,101]) # cumulative prob distribution grid
    
    for index in range(20): # 'index' indexes values computed by feature function
        currpdgrid = gp2d(testevid,feature,index,params,gridmin,gridmax)
        dist = draw2d(currpdgrid,gridmin,gridmax,testevid) # distance between maxima and actual
        plt.title("distance = %f" %dist)
        plt.savefig("outputs/%d/%d.png" %(testevid,index))
        plt.close()
        pdgrid *= currpdgrid
        
    dist = draw2d(pdgrid,gridmin,gridmax,testevid)
    plt.title("distance = %f" %dist)
    plt.savefig("outputs/%d/total.png" %testevid)
    plt.close()


def test16():
    evid1 = 4686108
    evid2 = 4689462
    evid3 = 4650871
    e1 = Event(evid1)
    e2 = Event(evid2)
    e3 = Event(evid3)
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
    e1 = Event(evid1)
    e2 = Event(evid2)
    e3 = Event(evid3)
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
    e1 = Event(evid1)
    e2 = Event(evid2)
    e3 = Event(evid3)
    
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

    e1 = Event(evid1)
    e2 = Event(evid2)
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
    e1 = Event(evid1)
    e2 = Event(evid2)
    e3 = Event(evid3)
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
    e = Event(evid, phase=None)
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
        
    e = Event(evid, phase=None)
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
        e = Event(evid)
        plt.plot(e.data)
        plt.show()

def test5():
    evids = validevids("LPAZ")
    print len(evids)
    for evid in evids:
        try:
            e = Event(evid)
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
    
num = sys.argv[1]
method = "test%s" %num
getattr(sys.modules[__name__], method)()