from database import db, wave
import learner, model

import numpy as np

from pylab import * #for matplotlib


class Data():
    def __init__(self, start_time, end_time, data, samprate):
        self.start_time = start_time
        self.end_time = end_time
        self.data = data
        self.samprate = samprate

def tconv(start_time, desired_time, samprate):
    return int((desired_time - start_time) * samprate)

class NoiseFetcher:
    def __init__(self, sta, posttime=200):
        self.cursor = db.connect().cursor()
        self.sta = 'DAV'
        self.chan = 'BHZ'
        self.time_start = 1237680000
        self.time_end = 1237766400
        self.cursor.execute("select time from idcx_arrival where sta='%s' and time > %f and time < %f" %(sta, self.time_start, self.time_end))
        self.arrival = [x[0] for x in self.cursor.fetchall()] 
        self.arrival_cursor = 0
        self.cursor.execute("select time, endtime from idcx_wfdisc where sta='%s' and time > %f and endtime < %f" %(sta, self.time_start, self.time_end))
        startend = self.cursor.fetchall()
        self.raw_start = [x[0] for x in startend[1:]]
        self.start_cursor = 0
        self.raw_end = [x[1] for x in startend[1:]]
        self.end_cursor = 0
        self.posttime = posttime
    def nextseg(self):
        def helper():
            if self.start_cursor >= len(self.raw_start):
                return None
            if self.end_cursor >= len(self.raw_end):
                return None
            if self.arrival_cursor >= len(self.arrival):
                return None
            
            s = self.raw_start[self.start_cursor]
            e = self.raw_end[self.end_cursor]
            a = self.arrival[self.arrival_cursor]
            if s < e and e < a:
                self.start_cursor += 1
                self.end_cursor += 1
                return (s, e)
            elif s < a and a < e:
                self.start_cursor += 1
                return (s, a)
            elif e < a and a < s:
                self.end_cursor += 1
            elif e < s and s < a:
                self.end_cursor += 1
            elif a < s and s < e:
                if a + self.posttime < s:
                    self.arrival_cursor += 1                    
                else:
                    self.start_cursor += 1
                    
            else:
                if a + self.posttime < e:
                    self.arrival_cursor += 1
                else:
                    self.end_cursor += 1
            return helper()
        s, e = helper()
 
        if s == None or e == None:
            return None
        
        data, samprate = wave.fetch_waveform(self.sta, self.chan, int(s+1), int(e))
        d = Data(s, e, data, samprate)
        return d
    
"""
cursor = db.connect().cursor()
cursor.execute("select distinct sta from idcx_arrival")
stas = [x[0] for x in cursor.fetchall()]

for sta in stas:
    nf = NoiseFetcher(sta)
    try:
        d = nf.nextseg()
        print sta
    except:
        x = 1
"""

nf1 = NoiseFetcher('MBAR')
d1 = nf1.nextseg()
while len(d1.data) < 1000:
    d1 = nf1.nextseg()

nf2 = NoiseFetcher('DAV')
d2 = nf2.nextseg()
while len(d2.data) < 1000:
    d2 = nf2.nextseg()
    
shortd1 = d1.data[2000:4000] - np.mean(d1.data)
shortd2 = d2.data[:1024] - np.mean(d2.data)
longd1 = d1.data[:10000] - np.mean(d1.data)


def ploterror(data, p, loc, test, numbin=100):
    lnr = learner.ARLearner(data)
    params, std = lnr.yulewalker(p)
    em = model.ErrorModel(0,std)
    arm = model.ARModel(params,em)
    realerror = arm.errors(test)
    sample = arm.sample(len(test))
    sampleerror = arm.errors(sample)
    bins = linspace(-3*std,3*std,numbin)
    hist(realerror, bins, histtype='step', label='Actual')
    hist(sampleerror, bins, histtype='step', label='Sample')
    xlabel('Error')
    ylabel('Frequency')
    title('Yule-Walker Fit AR Model Errors for p=%d at %s' %(p,loc))
    legend()
    show()

def plotcompare(data, p, loc, l=200, sf=40.0):
    lnr = learner.ARLearner(data)
    params, std = lnr.yulewalker(p)
    em = model.ErrorModel(0,std)
    arm = model.ARModel(params,em)
    initdata = longd1[:p]
    s = arm.sample(l, initdata=initdata)
    x = np.array(range(l))/float(sf)
    plot(x, longd1[:l], label='Actual')
    plot(x, s, label='Sampled')
    xlabel('Time (s)')
    ylabel('magnitude')
    title('Comparison btwn Real and Sampled Data from AR Model for p=%d' %p)
    legend()
    show()

def plotpsd(data, p, loc):
    lnr = learner.ARLearner(data)
    params, std= lnr.yulewalker(p)
    
    em = model.ErrorModel(0,std)
    arm = model.ARModel(params,em)
    x,y = arm.psd(size=len(data))
    plot(x,y,label='ideal AR PSD')
    x2, y2 = lnr.psd()
    plot(x2,y2,label='actual PSD')
    xlabel('Frequency (Hz)')
    ylabel('Power (natural log scale)')
    title('PSD of Ideal AR Model and Actual Data @ %s for p=%d'%(loc,p))
    legend()
    show()


def plotcv(data, d, dtype, loc):
    x = np.array(range(d))+1
    lnr = learner.ARLearner(data)
    if dtype == 'raw':
        y = [lnr.crossval(p) for p in x]
        plot(x,y)
        xlabel('degree of model')
        ylabel('lklhood (sum of log prob)')
        title('Cross-validation of Raw Data @ MBAR')
        show()
    elif dtype == 'psd':
        y = [lnr.psdcrossval(p) for p in x]
        plot(x,y)
        xlabel('degree of model')
        ylabel('avg deviation')
        title('Cross-validation of PSD @ MBAR')
        show()

def plotaic(data, d, loc):
    x = np.array(range(d))+1
    lnr = learner.ARLearner(data)
    y = [lnr.aic(p) for p in x]
    plot(x,y)
    xlabel('degree of model')
    ylabel('AIC value')
    title('AIC vs Degree of AR Model via Yule-Wlaker @ %s' %loc)
    show()

#plotaic(longd1, 50, 'MBAR')
#ploterror(longd1, 20, 'MBAR', longd1)
plotcompare(longd1, 20, 'MBAR')



#plotpsd(shortd1,3,'MBAR')
#plotcv(shortd1,50,'psd','MBAR')
#plotcv(shortd1,50,'raw','MBAR')


