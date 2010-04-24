from distutils.core import setup, Extension
import numpy as np
import os

priors_sources = ['NumEventPrior.c', 'EventLocationPrior.c',
                  'EventMagPrior.c', 'EventDetectionPrior.c',
                  'EarthModel.c', 'ArrivalTimePrior.c', 'NumFalseDetPrior.c',
                  'ArrivalAzimuthPrior.c', 'ArrivalSlownessPrior.c',
                  'ArrivalPhasePrior.c', 'ArrivalSNRPrior.c',
                  'Poisson.c', 'score.c', 'Gaussian.c']

infer_sources = ['infer.c', 'propose.c']

netvisa_module = Extension('netvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + ["netvisa.c"])
                           )
setup (name = 'netvisa',
       version = '1.0',
       description = 'Network Vertically Integrated Seismological Processing',
       include_dirs = [np.get_include()],
       ext_modules = [netvisa_module])
