import numpy as np

#autocorrelation function for wide-sense stationary process
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]/len(x)

#the probability density function of normal distribution
def normaldist(x, mean, std):
    t1 = 1.0/(std*np.sqrt(2*np.pi))
    t2 = -0.5 * np.square((x-mean)/std)
    return t1*np.exp(t2)
