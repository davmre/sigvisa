# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 

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
                  'SecDetPrior.c']

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

