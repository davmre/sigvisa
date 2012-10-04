import numpy as np


def band_to_hz(short_band):
    return np.median([float(x) for x in short_band.split('_')[1:]])


