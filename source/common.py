import numpy as np



def band_to_hz(short_band):
    if short_band=="broadband":
        raise Exception("source model: cannot assign a frequency band to broadband signal")
    return np.median([float(x) for x in short_band.split('_')[1:]])
