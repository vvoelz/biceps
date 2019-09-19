from libcpp.vector cimport vector
from libcpp cimport bool
import cython

cdef extern from "convergence.h":
    cdef vector[vector[float]] c_autocorrelation(
            vector[vector[float]] sampled_parameters,
            int maxtau, bool normalize)

    cdef vector[float] c_autocorrelation_time(
            vector[vector[float]] autocorr, bool normalize)


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def autocorrelation_time(vector[vector[float]] autocorr, bool normalize=True):
    """Calculate the autocorrelation time, tau_c for a time-series f(t).  The autocorrelation
    time tau_c quantifies the amount of time necessary for simulation data to become
    decorrelated or "lose their memory".

    :math: \tau_{c} = \int_{0}^{\infty} g(\tau) d \tau

    :param np.array autocorr: a 2D numpy array containing the autocorrelation for each nuisance parameter
    :param bool normalize: if True, return g(tau)/g[0]
    :return np.array: a numpy array of size (3, max_tau+1) containing g(tau) for each nuisance parameter
    """

    return c_autocorrelation_time(autocorr, normalize)




