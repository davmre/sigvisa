from distutils.core import setup, Extension
import numpy as np
import os

liblogger_include = '/home/dmoore/local/include/'
liblogger_library = '/home/dmoore/local/lib/'

extra_compile_args = ['-std=c99', '-g', '-O0']
extra_link_args = []

priors_sources = ['NumEventPrior.c', 'EventLocationPrior.c',
                  'EventMagPrior.c', 'EventDetectionPrior.c',
                  'EarthModel.c', 'ArrivalTimePrior.c', 'ArrivalTimeJointPrior.c', 
                  'NumFalseDetPrior.c',
                  'ArrivalAzimuthPrior.c', 'ArrivalSlownessPrior.c',
                  'ArrivalPhasePrior.c', 'ArrivalSNRPrior.c',
                  'ArrivalAmplitudePrior.c',
                  'Poisson.c', 'score.c', 'score_sig.c', 
                  'Gaussian.c', 'Gamma.c',
                  'NumSecDetPrior.c', 'SignalPrior.c', 'SignalModelCommon.c', 'SignalModelUtil.c', 'SpectralEnvelopeModel.c']

infer_sources = ['infer.c', 'propose.c', 'quickselect.c']
misc_sources = ['logging.c']

sigvisa_module = Extension('sigvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + [f for f in misc_sources]
                                      + ["netvisa.c"]
                                      + ["sigvisa.c"]),
                           libraries = ['logger', 'gsl', 'gslcblas'],
                           library_dirs = [liblogger_library],
                           runtime_library_dirs = [liblogger_library],
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args,
                           )

netvisa_module = Extension('netvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + [f for f in misc_sources]
                                      + ["netvisa.c"]
                                      + ["sigvisa.c"]),
                           libraries = ['logger', 'gsl', 'gslcblas'],
                           library_dirs = [liblogger_library],
                           runtime_library_dirs = [liblogger_library],
                           extra_compile_args = extra_compile_args
                           )

setup (name = 'sigvisa',
       version = '1.0',
       description = 'Signal-Based Vertically Integrated Seismological Processing',
       include_dirs = [np.get_include(), liblogger_include],
       ext_modules = [sigvisa_module, netvisa_module])

