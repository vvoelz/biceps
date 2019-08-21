from libcpp.vector cimport vector
from libcpp cimport bool
import cython

cdef extern from "convergence.h":
    cdef vector[vector[float]] c_autocorrelation(
            vector[vector[float]] sampled_parameters,
            int maxtau, bool normalize)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def autocorrelation(vector[vector[float]] sampled_parameters,
        int maxtau=10000, bool normalize=True):
    """Calculate the autocorrelaton function for a time-series f(t).

    :param np.array sampled_parameters: a 2D numpy array containing the time series f(t) for each nuisance parameterurn c_autocorrelation(sampled_parameters, maxtau, normalize)
    :param int max_tau: the maximum autocorrelation time to consider.
    :param bool normalize: if True, return g(tau)/g[0]
    :return np.array: array of size (len(sampled_parameters), max_tau+1) containing g(tau) for each nuisance parameter
    """

    return c_autocorrelation(sampled_parameters, maxtau, normalize)


