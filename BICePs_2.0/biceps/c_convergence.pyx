from libcpp.vector cimport vector
from libcpp cimport bool
import cython

cdef extern from "convergence.h":
    cdef vector[vector[float]] c_autocorrelation(
            vector[vector[float]] sampled_parameters,
            int maxtau, bool normalize)

@cython.boundscheck(False)
@cython.wraparound(False)
def autocorrelation(vector[vector[float]] sampled_parameters,
        int maxtau=10000, bool normalize=True):

    return c_autocorrelation(sampled_parameters, maxtau, normalize)


