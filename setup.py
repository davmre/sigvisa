3# Copyright (c) 2012, Bayesian Logic, Inc.
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

nonempty = lambda x: len(x) > 0

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
#extra_compile_args = ['-std=c99','-O3']
#extra_link_args = ['-Wl,--strip-all']
extra_link_args = ['-lrt', ]

priors_sources = ['NumEventPrior.c', 'EventLocationPrior.c',
                  'EventMagPrior.c',
                  'EarthModel.c', 'ArrivalTimePrior.c', 'ArrivalTimeJointPrior.c',
                  'ArrivalAzimuthPrior.c', 'ArrivalSlownessPrior.c',
                  'ArrivalAmplitudePrior.c',
                  'Poisson.c',
                  'Gaussian.c', 'Gamma.c']

main_sources = ['sigvisa.c',]



sigvisa_module = Extension('sigvisa_c',
                           sources=([os.path.join("priors", f)
                                     for f in priors_sources]
                                    + [f for f in main_sources]),
                           library_dirs = sys_libraries,
                           runtime_library_dirs = sys_libraries,
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args,
                           )


ssm_sources = ['python_wrappers.cc', 'transient_combined.cc',
               'statespace.cc', 'compact_support.cc',
               'autoregression.cc']
from imp import find_module
f, pathname, descr = find_module("pyublas")
CTREE_INCLUDE_DIRS = [os.path.join(pathname, "include"),]

statespacemodel_module = Extension('ssms_c',
                           sources=([os.path.join("models", "statespace", "fast_c", f)
                                     for f in ssm_sources]
                                    ),
                           include_dirs=CTREE_INCLUDE_DIRS,
                           library_dirs = sys_libraries,
                           libraries=['boost_python'],
                           runtime_library_dirs = sys_libraries,
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args,
                           )

setup(name='sigvisa',
      version='1.0',
      description='Signal-Based Vertically Integrated Seismological Processing',
      include_dirs=[np.get_include()] + sys_includes,
      ext_modules=[sigvisa_module, statespacemodel_module])
