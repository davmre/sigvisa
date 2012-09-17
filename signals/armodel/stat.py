import numpy as np

#autocorrelation function for wide-sense stationary process
def autocorr(exes, p):

    results = []
    total_len = 0
    for x in exes:
        result = np.correlate(x, x, mode='full')
        results.append(result[result.size/2:result.size/2+p+1])
        total_len = total_len + len(x)
    results = np.array(results)
    results = np.sum(results, axis=0) / total_len
    return results

#the probability density function of normal distribution
def normaldist(x, mean, std):
    t1 = 1.0/(std*np.sqrt(2*np.pi))
    t2 = -0.5 * np.square((x-mean)/std)
    return t1*np.exp(t2)
