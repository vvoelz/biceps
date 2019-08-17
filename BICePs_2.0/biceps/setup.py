#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(name="c_convergence",
                             sources=["c_convergence.pyx","convergence.cpp"],
                             language="c++",
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=["--std=c++11"]
                             #extra_compile_args=["-std=c++11"],
                             #extra_link_args=["-std=c++11"]
                             )]
    )





