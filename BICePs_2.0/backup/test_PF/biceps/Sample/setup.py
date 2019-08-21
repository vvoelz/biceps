#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("Sample",
                             sources=["Sampler.pyx","sample.cpp"],
                             language="c++",
                             include_dirs=[numpy.get_include()])],
    )





