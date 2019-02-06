#from __future__ import print_function
#import numpy as np
#cimport numpy as np
from libcpp.vector cimport vector
import cython

cdef extern from "sample.h":
    cdef void c_sample(int nsteps, vector[double] array)

# Don't allow negative index in array, and check the bounds of the array
@cython.boundscheck(False)
@cython.wraparound(False)
def sample(int nsteps, vector[double] array):
    """Takes a numpy array as input and outputs a new array that can be passed
    back to python.
    """

    cdef int nstpes
    c_sample(<int> nsteps, array)



