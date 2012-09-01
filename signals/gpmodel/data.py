import database.db
from database.dataset import *
import utils.geog
import sys
import numpy as np
import wave

"""
phase = 0.70_1.00, 1.00_1.50, 
"""

class Location(object):
    def __init__(self,lat=None,lon=None,depth=None,pos=None):
        if pos==None:
            self.lat = lat
            self.lon = lon
            self.depth = depth
            self.pos = self.getpos()
        else:
            self.pos = pos
            
    def dist(self, e):
        return np.sqrt(sum(np.square(self.pos-e.pos)))
    
    def getpos(self):
        R = 6356.8
        r = R-self.depth
        x = r*np.cos(np.radians(self.lat))*np.cos(np.radians(self.lon))
        y = r*np.cos(np.radians(self.lat))*np.sin(np.radians(self.lon))
        z = r*np.sin(np.radians(self.lat))
        return np.array([x,y,z])
        
            
class Event(Location):
    def __init__(self, evid, feature=None, phase=None, normalize=True):
        evid = int(evid)
        self.evid = evid
        if feature == None:
            self.feature = wave.psd
        else:
            self.feature = feature
        cursor = database.db.connect().cursor()
        command = "select runid, siteid, phaseid from sigvisa_wiggle_wfdisc \
        where evid=%d" %evid
        cursor.execute(command)
        runid, siteid, phaseid = cursor.fetchall()[0]
        prefix="../sigvisa_data/wiggles"
        suffix="_BHZ.dat"
        if phase==None:
            suffix = "_BHZ_raw.dat"
            addr="%s/%s/%s/%s/%s%s"%(prefix,runid,siteid,phaseid,evid,suffix)
        else:
            addr="%s/%s/%s/%s/%s/%s%s"%(prefix,runid,siteid,phaseid,phase,evid,suffix)
        try:
            self.data=np.loadtxt(addr)
            if normalize:
                self.x=(self.data-np.mean(self.data))/np.std(self.data) # normalized
            else:
                self.x = self.data
        except:
            self.data=None
            self.x=None
        
        if self.data != None:
            self.y = self.feature(self.data)
        else:
            self.y = None
        command = "select lon, lat, depth from leb_origin where evid=%d" %evid
        cursor.execute(command)
        self.lon, self.lat, self.depth = cursor.fetchall()[0]
        self.pos = self.getpos()
        
        self.gpval = self.feature(self.x)
"""
        R = 6356.8 # radius of earth, in km
        r = R-self.depth
        x = r*np.cos(np.radians(self.lat))*np.cos(np.radians(self.lon))
        y = r*np.cos(np.radians(self.lat))*np.sin(np.radians(self.lon))
        z = r*np.sin(np.radians(self.lat))
        self.pos = np.array([x,y,z])
"""     
        

# return evid numbers which meet criteria as in utils.closest_event_pairs
def validevids(sta):
    command ="select distinct lebo.evid from leb_origin lebo, leb_assoc leba, \
    leb_arrival l, sigvisa_coda_fits fit, sigvisa_wiggle_wfdisc wf where \
    leba.arid=fit.arid and leba.orid=lebo.orid and l.arid=leba.arid and \
    l.sta='%s' and fit.acost<10 and leba.phase='P' and \
    (fit.runid=3 or fit.runid=4) and wf.arid=fit.arid and wf.band='0.70_1.00' \
    and wf.chan='BHZ'" % (sta)
    cursor = database.db.connect().cursor()
    cursor.execute(command)
    return np.array(cursor.fetchall())[:,0]

def validevents(sta):
    evids = validevids(sta)
    events = []
    for evid in evids:
        e = Event(evid)
        if e.data != None:
            events.append(e)
    return events

def validevids2():
    cursor = database.db.connect().cursor()
    command = "select distinct evid from sigvisa_wiggle_wfdisc where runid=6 \
    and siteid=66 and phaseid=2"
    cursor.execute(command)
    evid = np.array(cursor.fetchall())[:,0]
    return evid


#ex excludes a certain event
def validevents2(ex=None,feature=None):
    evids = validevids2()
    events = []
    for evid in evids:
        e = Event(evid, feature=feature, phase=None)
        if e.data != None:
            if ex != evid:
                events.append(e)
    return events
    

#LPAZ, GNI, AFI, JNU

def validstations():
    command ="select distinct l.sta from leb_origin lebo, leb_assoc leba, \
    leb_arrival l, sigvisa_coda_fits fit, sigvisa_wiggle_wfdisc wf where \
    leba.arid=fit.arid and leba.orid=lebo.orid and l.arid=leba.arid and \
    fit.acost<10 and leba.phase='P' and (fit.runid=3 or fit.runid=4) and \
    wf.arid=fit.arid and wf.band='1.00_1.50' and wf.chan='BHZ'"
    cursor = database.db.connect().cursor()
    cursor.execute(command)
    return np.array(cursor.fetchall())[:,0]

def closest2(evid=None):
    events = validevents2(ex=evid)
    list = []
    if evid == None:
        for i in range(len(events)):
            ei = events[i]
            for j in range(i+1,len(events)):
                ej = events[j]
                list.append((ei.dist(ej),ei.evid, ej.evid))
    else:
        e = Event(evid)
        for ei in events:
            list.append((e.dist(ei),ei.evid))
            
    list.sort()
    return list

# returns inputs for 2D
def inputs2d(events):
    inputs = np.zeros([len(events), 2])
    for i in range(len(events)):
        inputs[i] = [events[i].lat, events[i].lon]
    return np.array(inputs)

def gpvals(events, index):
    outputs = np.zeros(len(events))
    for i in range(len(events)):
        outputs[i] = events[i].gpval[index]
    return outputs