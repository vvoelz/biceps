
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os,glob
import subprocess
import json


#compiler = "g++"
compiler = "clang++"

#profile = 0
#
#if profile:
#    compiler = "/opt/homebrew/opt/llvm/bin/clang++"
#    os.environ['CC'] = compiler
#    os.environ['CXX'] = compiler


if compiler == "g++":
    #os.environ['CC'] = "g++-12"
    #os.environ['CXX'] = "g++-12"
    pass

if compiler == "clang++":
    pass



#NOTE: compiler options
"""
Optimization:
https://caiorss.github.io/C-Cpp-Notes/compiler-flags-options.html
https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
"""

if compiler == "g++":
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("PosteriorSampler",
            sources=["PosteriorSampler.pyx","cppPosteriorSampler.cpp"],
            language="c++",
            #include_dirs=[numpy.get_include(), os.path.join(libtorch_path, 'include')],
            #include_dirs=[numpy.get_include(), os.path.join(libtorch_path, 'include'), os.path.join(libtorch_path, 'include', 'torch', 'csrc', 'api', 'include')],
            #libraries=['torch', 'torch_cpu', 'c10'],

            include_dirs=[numpy.get_include()],
            extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O3", "-march=native", '-fopenmp'],
            #extra_link_args=["-O3", "-march=native", '-lomp','-Wl,-rpath,' + os.path.join(libtorch_path, 'lib')],
            extra_link_args=["-O3", "-march=native", '-lomp'],
            )],
        )

if compiler == "clang++":
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("PosteriorSampler",
            sources=["PosteriorSampler.pyx","cppPosteriorSampler.cpp"],
            language="c++",
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O3", "-g", '-stdlib=libc++', '-std=c++20', '-Xclang',  '-fopenmp'],
            #extra_link_args=["-O3","-g", '-stdlib=libc++', '-lomp','-Wl,-rpath,' + os.path.join(libtorch_path, 'lib')],
            extra_link_args=["-O3","-g", '-stdlib=libc++', '-lomp'],
            )],
        )



