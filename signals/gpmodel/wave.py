import numpy as np
import matplotlib.pyplot as plt
import pywt
import gaussian as gs

def psd(data):
    
    cA1, cD1 = pywt.dwt(data,'db1') #cA1 0-10 Hz
    cA2, cD2 = pywt.dwt(cA1,'db1') #cA2 0-5 Hz
    cA3, cD3 = pywt.dwt(cA2, 'db1') #cA3 0-2.5Hz
    cA4, cD4 = pywt.dwt(cA3, 'db1') #cD4 1.25-2.5 Hz
#    cA5, cD5 = pywt.dwt(cD4, 'db1') #cA5 1.25-1.875 Hz
    y = np.abs(np.fft.rfft(cD4))[5:-5]
    x = range(len(y))
    params = (1.8,5,0.7)
    gpm = gs.GPModel(x,params=params)
    gpl = gs.GPLearner(gpm, y)
    yp,vars,logp = gpl.predict(x)
    return yp/15
#    return np.log(np.abs(np.fft.rfft(data)))[20:100]

def test1():
    N = 20
    a = np.random.rand(N)-0.5
    
    M = 1000
    x = np.zeros(M,dtype=complex)
    xpos = np.linspace(0,60,M)
    for n in range(M):
        x[n] = np.sum([a[k]*np.exp(1j*k*(2*np.pi/N)*xpos[n]) for k in range(N)])
    plt.plot(np.abs(x))
    plt.show()
    
    aa = np.zeros(N,dtype=complex)
    for k in range(N):
        aa[k] = (1.0/N)*np.sum([x[n]*np.exp(-1j*k*(2*np.pi/N)*xpos[n]) for n in range(M)])
    
    print a
    print aa