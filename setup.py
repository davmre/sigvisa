from distutils.core import setup, Extension
import numpy as np
import os

extra_compile_args = ['-std=c99', '-D_GNU_SOURCE']

priors_sources = ['NumEventPrior.c', 'EventLocationPrior.c',
                  'EventMagPrior.c', 'EventDetectionPrior.c',
                  'EarthModel.c', 'ArrivalTimePrior.c', 'NumFalseDetPrior.c',
                  'ArrivalAzimuthPrior.c', 'ArrivalSlownessPrior.c',
                  'ArrivalPhasePrior.c', 'ArrivalSNRPrior.c',
                  'ArrivalAmplitudePrior.c',
                  'Poisson.c', 'score.c', 'Gaussian.c',
                  'NumSecDetPrior.c']

infer_sources = ['infer.c', 'propose.c', 'quickselect.c']

netvisa_module = Extension('netvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + ["netvisa.c"]),
                           extra_compile_args = extra_compile_args
                           )
setup (name = 'netvisa',
       version = '1.0',
       description = 'Network Vertically Integrated Seismological Processing',
       include_dirs = [np.get_include()],
       ext_modules = [netvisa_module])
