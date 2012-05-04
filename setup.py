from distutils.core import setup, Extension
import numpy as np
import os


nonempty = lambda x : len(x) > 0

if os.getenv("C_INCLUDE_PATH") is not None:
    sys_includes = filter(nonempty, os.getenv("C_INCLUDE_PATH").split(':'))
else:
    sys_includes = []
if os.getenv("LIBRARY_PATH") is not None:
    sys_libraries = filter(nonempty, os.getenv("LIBRARY_PATH").split(':'))
else:
    sys_libraries = []

print sys_includes
print sys_libraries

extra_compile_args = ['-std=c99', '-g', '-O0']
extra_link_args = []

priors_sources = ['NumEventPrior.c', 'EventLocationPrior.c',
                  'EventMagPrior.c', 'EventDetectionPrior.c',
                  'EarthModel.c', 'ArrivalTimePrior.c', 'ArrivalTimeJointPrior.c',
                  'NumFalseDetPrior.c',
                  'ArrivalAzimuthPrior.c', 'ArrivalSlownessPrior.c',
                  'ArrivalPhasePrior.c', 'ArrivalSNRPrior.c',
                  'ArrivalAmplitudePrior.c',
                  'Poisson.c', 'score.c',
                  'Gaussian.c', 'Gamma.c',
                  'NumSecDetPrior.c']

signals_sources = ['SignalPrior.c', 'SignalModelCommon.c', 'SignalModelUtil.c', 'SpectralEnvelopeModel.c', 'envelope.c', 'score_sig.c']

infer_sources = ['infer.c', 'propose.c', 'quickselect.c']
misc_sources = ['logging.c']

sigvisa_module = Extension('sigvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("signals", f)
                                       for f in signals_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + [f for f in misc_sources]
                                      + ["netvisa.c"]
                                      + ["sigvisa.c"]),
                           libraries = ['logger', 'gsl', 'gslcblas'],
                           library_dirs = sys_libraries,
                           runtime_library_dirs = sys_libraries,
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args,
                           )

netvisa_module = Extension('netvisa',
                           sources = ([os.path.join("priors", f)
                                       for f in priors_sources]
                                      + [os.path.join("signals", f)
                                       for f in signals_sources]
                                      + [os.path.join("infer", f)
                                         for f in infer_sources]
                                      + [f for f in misc_sources]
                                      + ["netvisa.c"]
                                      + ["sigvisa.c"]),
                           libraries = ['logger', 'gsl', 'gslcblas'],
                           library_dirs = sys_libraries,
                           runtime_library_dirs = sys_libraries,
                           extra_compile_args = extra_compile_args
                           )

setup (name = 'sigvisa',
       version = '1.0',
       description = 'Signal-Based Vertically Integrated Seismological Processing',
       include_dirs = [np.get_include()] +  sys_includes,
       ext_modules = [sigvisa_module, netvisa_module])

