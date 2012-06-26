import numpy as np
import matplotlib.pyplot as plt
from signals.gpmodel.gaussian import *
from signals.gpmodel.gridutil import *

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
    
    pd = mpd4test(fullmodel, trnmodel, testdomain, trndomain, testi, gd, 1)
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
    
