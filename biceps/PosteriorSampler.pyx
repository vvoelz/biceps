# distutils: language = c++
# cython: language_level=3, boundscheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-
# libraries:{{{
import time, copy, re
import pandas as pd
import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython
import binascii
import warnings
import inspect
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libc.string cimport strdup
from libc.math cimport cos
from libcpp cimport bool
from libcpp cimport list
from libc.math cimport sqrt as c_sqrt
from libcpp cimport tuple as c_tuple
from libcpp.utility cimport pair
import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.operator cimport dereference as deref
#from tqdm import tqdm # progress bar
from .toolbox import save_object as saveObj
from .toolbox import compute_f0
from .Restraint import ExpandedEnsemble
from pymbar import MBAR
from pymbar.utils import kln_to_kn
from pymbar import mbar_solvers
from itertools import groupby
from scipy import interpolate, integrate
from cython cimport boundscheck, wraparound
#import torch.nn as nn
#import torch
#from torch.autograd import grad
cimport numpy as cnp
from scipy.stats import mode


# }}}

# external:{{{
cdef extern from "cppPosteriorSampler.h" namespace "PS":
    cdef struct GFE:
        vector[vector[vector[double]]] u_kln
        vector[vector[vector[double]]] states_kn
        vector[vector[int]] Nr_array
        vector[vector[vector[double]]] diff_u_kln
        vector[vector[vector[double]]] diff2_u_kln

    cdef GFE get_u_kln_and_states_kn(object ensembles,
            vector[vector[vector[int]]] state_traces, vector[vector[double]] energy_traces,
            vector[vector[vector[double]]] parameter_traces, vector[vector[vector[double]]] expanded_traces,
            vector[float] logZs, bool progress, bool capture_stdout,
            bool scale_energies, bool compute_derivative, bool multiprocess,
            object sampler)

    cdef double get_data_restraint_energy(int, string, vector[double], vector[double],
            vector[double], double, vector[double], string)

    cdef cppclass cppHREPosteriorSampler:
        cppHREPosteriorSampler(object, object, int, int, int, int, double, double, bool, bool)
        void cpp_sample(int, int, int, bool, bool, int, int, bool, bool, bool, bool, bool, bool)
        void dealloc();
        object get_traj();
        object get_fmp_traj();
#        object get_pmp_traj();
        object get_convergence_metrics();
        #object get_pm_prior_sigma_traj();
        object get_fm_prior_sigma_traj();
        object get_ti_info();
        object get_exchange_info();
        object get_N_replicas();
        vector[vector[double]] get_populations();
        vector[double] get_entropy();
        vector[double] get_chi2();

# }}}

# python methods:{{{
def change_xi_every(nsteps, dxi=0.1, verbose=False):
    xi_initial = 1.0
    xi_final = 0.0
    n_ratchets = int((xi_initial - xi_final) / dxi)
    if verbose: print("n_ratchets = ",n_ratchets)
    change_xi_every = nsteps // (n_ratchets + 1)  # Adding 1 to ensure xi=0 has the same duration
    if verbose: print("change_xi_every = ",change_xi_every)
    return change_xi_every


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def u_kln_and_states_kn(ensembles, trajs, nstates, logZs,
        capture_stdout=False, scale_energies=False, compute_derivative=False,
        multiprocess=True, progress=True, verbose=False, sampler=None):
    """Wrapper function that points to the C++ code for building the u_kln matrix.
       Returns the energy matrix u_kln to be passed to MBAR. The construction
       of this matrix is as follows: Suppose the energies sampled from each
       simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
       of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at
       reduced potential for state l. Initialize MBAR with reduced energies
       u_kln and number of uncorrelated configurations from each state N_k.
       u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where
       U_l(x) is the potential energy function for state l,
       beta is the inverse temperature, and and x_kn denotes uncorrelated
       configuration n from state k.

    Args:
        ensembles(object): the
        trajs(list): list of MCMC trajectories for each thermodynamic ensemble
        nstates(int): number of conformational states
        logZs(np.ndarray): numpy array containing logZ for each ensemble
        capture_stdout(bool=False): if in Jupyter notebook and using progress bar, you will want this turned on
        scale_energies(bool=False): scale energies by the number of replicas
        compute_derivative(bool=False):
        multiprocess(bool=True): run in parallel
        progress(bool=True): progress bar
        verbose(bool=False): verbosity
    """

    stime = time.time()
    cdef vector[double] expanded_values
    cdef vector[vector[double]] expanded_trace
    cdef vector[vector[vector[double]]] expanded_traces
    cdef vector[double] parameters
    cdef vector[vector[double]] parameter_trace
    cdef vector[vector[vector[double]]] parameter_traces
    cdef vector[int] states
    cdef vector[vector[int]] state_trace
    cdef vector[vector[vector[int]]] state_traces
    cdef vector[double] energy_trace
    cdef vector[vector[double]] energy_traces
    for l in range(len(trajs)):
        traj_objects = np.array(trajs[l]['trajectory'], dtype=object).T
        expanded_trace.clear()
        _expanded_trace = traj_objects[7]
        state_trace.clear()
        parameter_trace.clear()
        energy_trace.clear()
        _states = traj_objects[3]
        _parameters = traj_objects[5]
        for i,energy in enumerate(traj_objects[1]):
            expanded_values.clear()
            for j in range(len(_expanded_trace[i])):
                expanded_values.push_back(_expanded_trace[i][j])
            expanded_trace.push_back(expanded_values)
            energy_trace.push_back(energy)
            states.clear()
            parameters.clear()
            for j in range(len(_states[i])):
                states.push_back(_states[i][j])
            for j in range(len(_parameters[i])):
                for k in range(len(_parameters[i][j])):
                    parameters.push_back(_parameters[i][j][k])
            state_trace.push_back(states)
            parameter_trace.push_back(parameters)
        state_traces.push_back(state_trace)
        energy_traces.push_back(energy_trace)
        parameter_traces.push_back(parameter_trace)
        expanded_traces.push_back(expanded_trace)
    cdef vector[float] _logZs
    for logZ in logZs: _logZs.push_back(logZ)
    #replica_trace = np.array([len(trajs[0]['trajectory'][i][3]) for i in range(len(trajs[0]['trajectory']))], dtype=object)
    #cdef bool const_Nr = all_equal(replica_trace)
    cdef GFE u
    u = get_u_kln_and_states_kn(ensembles, state_traces, energy_traces, parameter_traces,
            expanded_traces, _logZs, progress, capture_stdout,
            scale_energies, compute_derivative, multiprocess, sampler)

    total_time = time.time() - stime
#    if verbose: print(f"Time to get_u_kln: {total_time:.2f}s")
    if compute_derivative:
        return np.array(u.u_kln), np.array(u.states_kn), np.array(u.Nr_array),\
                np.array(u.diff_u_kln), np.array(u.diff2_u_kln)
    else:
        return np.array(u.u_kln), np.array(u.states_kn), np.array(u.Nr_array)



# }}}

# PosteriorSampler:{{{
class PosteriorSampler(object):
    def __init__(self, ensemble, nreplicas=1, change_Nr_every=0, write_every=100,
            move_ftilde_every=0, continuous_space=False, dsigma=0.01, move_sigma_std=1.0,
            fwd_model_mixture=False, fwd_model_weights=None, pmo=False, fmo=False, fmo_method="SGD", pmo_method="SGD",
            fmo_model_idx=2,
            xi_integration=False, dXi=0.1, change_xi_every=0, num_xi_values=10, xi_schedule=None,
            dftilde=0.1, ftilde_sigma=1.0, scale_and_offset=False, verbose=False):

        stime = time.time()
        if isinstance(ensemble, ExpandedEnsemble):
            # NOTE: list of thermodynamic ensembles (ensemble for each lambda)
            self.lambda_values = ensemble.lambda_values
            self.xi_values = ensemble.xi_values
            self.pmo = pmo
            self.fmo = fmo
            self.fmo_method = fmo_method
            self.pmo_method = pmo_method
            self.sem_method = "sem"

            if self.pmo:
                self.prior_model = ensemble.prior_model
                self.prior_model_parameters = ensemble.prior_model_parameters

                if hasattr(ensemble, "prior_model_attrs"):
                    self.__dict__.update(ensemble.prior_model_attrs)
                else:
                    ensemble.initialize_prior_model(self.prior_model_parameters, min_max_paras=None, parameter_priors=None)
                    self.__dict__.update(ensemble.prior_model_attrs)



            if self.fmo:
                # NOTE: TODO: FIXME: you might want to construct a class
                # from this class after you have initialized it inside the ensemble object,
                # you can then simply pass all of the local class attributes to the ensemble class here...
                # NOTE: However, I think that this initialize_fwd_model() function
                # will be particularly useful for passing a fwd_model_obj (would be nice if it was a C++ compiled object)

                self.fmo_model_idx = fmo_model_idx
                if hasattr(ensemble, "fwd_model_attrs"):
                    self.__dict__.update(ensemble.fwd_model_attrs)
                else:
                    self.phi_angles = ensemble.phi_angles
                    self.phase_shifts = ensemble.phase_shifts
                    self.fwd_model_parameters = ensemble.fwd_model_parameters
                    self.fmo_restraint_indices = ensemble.fmo_restraint_indices
                    ensemble.initialize_fwd_model(self.fwd_model_parameters, self.phi_angles,
                                    self.fmo_restraint_indices, min_max_paras=None,
                                    parameter_priors=None, **{"phi0": self.phase_shifts})
                    self.__dict__.update(ensemble.fwd_model_attrs)


            self.expanded_values = ensemble.expanded_values
            self.expanded_values = [tuple(val) for val in self.expanded_values]
            self.ensembles = ensemble.to_list()
            # if True, we can control the mixing of forward models
            self.fwd_model_mixture = fwd_model_mixture
            if isinstance(fwd_model_weights, (np.ndarray, list, tuple)):
                if len(fwd_model_weights) != len(self.ensembles):
                    raise ValueError("len(fwd_model_weights) != len(ensembles).")
                else: self.fwd_model_weights = fwd_model_weights
            else:
                self.fwd_model_weights = [(np.ones(len(self.expanded_values), dtype=float)/len(self.expanded_values)).tolist() for i in range(len(self.expanded_values))]
            #print(self.fwd_model_weights)
            #exit()
            self.compute_logZ() # creates self.logZs
            self.prior_populations = compute_prior_populations(self.ensembles[-1])
            if np.count_nonzero(self.lambda_values[1:] == 1.0) < 1:
                self.f_k = []
                for l in range(len(self.lambda_values)-1):
                    self.f_k.append(np.array([s[0].energy for s in self.ensembles[l+1]], dtype=np.float64))
            else:
                self.f_k = np.array([s[0].energy for s in self.ensembles[-1]], dtype=np.float64)
            self.model = get_restraint_attr(self.ensembles[0], str.encode("model"))
            #self.Nd = int(np.concatenate(self.model[0]).shape[0])
            self.Nd = 0
            for data in self.model[0]: self.Nd += len(data)
            #print(self.Nd)
            self.nstates = len(self.ensembles[0]) # ensemble is a list of Restraint objects
            self.nreplicas = nreplicas
            self.change_Nr_every = change_Nr_every
            self.states = [np.random.randint(low=0, high=self.nstates, size=self.nreplicas) for _ in range(len(self.ensembles))]
            if self.nreplicas == self.nstates:
                self.states = [list(range(self.nstates))]*len(self.ensembles)

            self.E = [1.0e99 for _ in range(len(self.ensembles))]
            self.indices = []
            allowed_parameters = compile_nuisance_parameters(self.ensembles[0])
            self.parameters = []
            for k in range(len(self.states)):
                _indices = []
                for i,R in enumerate(self.ensembles[0][self.states[k][0]]):
                    keys = R.__dict__.keys() # all attributes of the Child Restraint class
                    index_keys = R.ind_order
                    [index_keys.append(key) for key in keys if (key not in index_keys) and ("_index" in key)]
                    for j in index_keys: # get the parameter indices
                        _indices.append(getattr(R, j))
                self.indices.append(_indices)
                self.parameters.append([allowed_parameters[i][j] for i,j in enumerate(_indices)])
            self.write_every = write_every
            self.move_ftilde_every = move_ftilde_every
            self.dftilde = dftilde
            self.continuous_space = continuous_space
            self.dsigma = [dsigma for p in range(len(self.parameters[0]))]
            self.move_sigma_std = move_sigma_std
            self.ftilde_sigma = ftilde_sigma
            self.scale_and_offset = scale_and_offset
            self.verbose = verbose
            self.xi_integration = xi_integration
            self.change_xi_every = change_xi_every
            #self.num_xi_values = num_xi_values
            #self.dXi = 1 / (self.num_xi_values - 1)
            if xi_schedule is None:
                self.xi_schedule = np.linspace(0, 1, num=num_xi_values, dtype=np.float64)[::-1].tolist()
            else:
                self.xi_schedule = list(xi_schedule)

            self.num_xi_values = len(self.xi_schedule)
            self.dXi = 1 / (self.num_xi_values - 1)

            self.traj = []
            self.rest_type = get_restraint_labels(self.ensembles[0])
            self.data_types = [re.sub(r'\d+', '', dtype).replace("sigma_","")
                               for dtype in self.rest_type if "sigma" in dtype]
        else:
            raise ValueError("What did you give me? `ensemble` must be a class object.")

        total_time = time.time() - stime
        if verbose: print(f"Time to initialize PosteriorSampler: {total_time:.2f}s")

    def compute_logZ(self):
        """Compute reference state logZ for the free energies to normalize."""
        if hasattr(self, 'ensemble'):
            Z = 0.0
            for s in self.ensemble:
                Z +=  np.exp( -np.array(s[0].energy, dtype=np.float64) )
            self.logZ = np.log(Z)


        if hasattr(self, 'ensembles'):
            Z = np.zeros(len(self.ensembles))
            for i,ensemble in enumerate(self.ensembles):
                for s in ensemble:
                    Z[i] +=  np.exp( -np.array(s[0].energy, dtype=np.float64) )
            self.logZs = np.log(Z)

    def sample(self, int nsteps, int attempt_lambda_swap_every=0, int burn=0,
            bool swap_sigmas=False, bool swap_forward_model=False, int print_freq=1000,
            bool walk_in_all_dim=False, int attempt_move_state_every=1, int attempt_move_sigma_every=1,
            int attempt_move_fmp_every=0, int attempt_move_pmp_every=0, int pmp_batch_size=1, int fmp_batch_size=1,
            int attempt_move_fm_prior_sigma_every=0, int attempt_move_pm_prior_sigma_every=0, int attempt_move_pm_extern_loss_sigma_every=0,
            int attempt_move_DB_sigma_every=0, int attempt_move_PC_sigma_every=0, int attempt_move_lambda_every=0, int attempt_move_xi_every=0, int attempt_move_rho_every=0,
            int sigma_batch_size=1, bool verbose=False, bool progress=True, bool multiprocess=True,
            bool capture_stdout=False, bool find_optimal_nreplicas=False):
        """Perform n number of steps (nsteps) of posterior sampling, where Monte
        Carlo moves are accepted or rejected according to Metroplis criterion.
        Energies are computed via :class:`neglogP`.

        Args:
            nsteps(int): number of steps of sampling.
            print_freq(int): frequency of printing to the screen
        """

        self.pmp_batch_size = pmp_batch_size
        self.fmp_batch_size = fmp_batch_size
        #if self.change_xi_every == 0: self.change_xi_every = round(nsteps/11)
        if self.xi_integration:
            if self.change_xi_every == 0:
                self.change_xi_every = change_xi_every(nsteps, dxi=self.dXi, verbose=False)

        if attempt_move_rho_every == 0:
            attempt_move_rho_every = nsteps
        self.attempt_move_rho_every = attempt_move_rho_every


        if attempt_move_fmp_every == 0:
            attempt_move_fmp_every = nsteps
        self.attempt_move_fmp_every = attempt_move_fmp_every

        if attempt_move_pmp_every == 0:
            attempt_move_pmp_every = nsteps
        self.attempt_move_pmp_every = attempt_move_pmp_every


        if attempt_move_fm_prior_sigma_every == 0:
            attempt_move_fm_prior_sigma_every = nsteps
        self.attempt_move_fm_prior_sigma_every = attempt_move_fm_prior_sigma_every

        if attempt_move_pm_prior_sigma_every == 0:
            attempt_move_pm_prior_sigma_every = nsteps
        self.attempt_move_pm_prior_sigma_every = attempt_move_pm_prior_sigma_every

        if attempt_move_pm_extern_loss_sigma_every == 0:
            attempt_move_pm_extern_loss_sigma_every = nsteps
        self.attempt_move_pm_extern_loss_sigma_every = attempt_move_pm_extern_loss_sigma_every


        if attempt_move_DB_sigma_every == 0:
            attempt_move_DB_sigma_every = nsteps
        self.attempt_move_DB_sigma_every = attempt_move_DB_sigma_every


        if attempt_move_PC_sigma_every == 0:
            attempt_move_PC_sigma_every = nsteps
        self.attempt_move_PC_sigma_every = attempt_move_PC_sigma_every

        if attempt_move_lambda_every == 0:
            attempt_move_lambda_every = nsteps
        self.attempt_move_lambda_every = attempt_move_lambda_every

        if attempt_move_xi_every == 0:
            attempt_move_xi_every = nsteps
        self.attempt_move_xi_every = attempt_move_xi_every





        self.attempt_move_state_every = attempt_move_state_every
        self.attempt_move_sigma_every = attempt_move_sigma_every
        self.nsteps = nsteps
        self.attempt_lambda_swap_every = attempt_lambda_swap_every
        #TODO: Conditional statement from input argument (swap_every > 0)
        if attempt_lambda_swap_every == 0:
            attempt_lambda_swap_every = nsteps
        if self.move_ftilde_every == 0:
            self.move_ftilde_every = nsteps
        if self.change_Nr_every == 0:
            self.change_Nr_every = nsteps

        stime = time.time()
        cppHREPS = new cppHREPosteriorSampler(self.ensembles, self, self.nreplicas,  self.change_Nr_every,
                self.write_every, self.move_ftilde_every, self.dftilde, self.ftilde_sigma, self.scale_and_offset, self.verbose)
        cppHREPS.cpp_sample(nsteps, attempt_lambda_swap_every, burn,
                swap_sigmas, swap_forward_model, print_freq, sigma_batch_size,
                walk_in_all_dim, find_optimal_nreplicas, verbose,
                progress, multiprocess, capture_stdout)
        self.nreplicas = cppHREPS.get_N_replicas()
        self.populations = np.array(cppHREPS.get_populations())
        self.entropy = np.array(cppHREPS.get_entropy())
        self.chi2 = np.array(cppHREPS.get_chi2())

        self.fmp_traj = np.array(cppHREPS.get_fmp_traj())
        self.fm_prior_sigma_traj = np.array(cppHREPS.get_fm_prior_sigma_traj())

#        self.pmp_traj = np.array(cppHREPS.get_pmp_traj())
#        self.pm_prior_sigma_traj = np.array(cppHREPS.get_pm_prior_sigma_traj())


        self.convergence_metrics = [pd.DataFrame(data) for data in cppHREPS.get_convergence_metrics()]


        if verbose: print(f"Time for sampling: %.3f s" % (time.time() - stime));
        if self.traj == []:
            # NOTE: self.traj is a list of trajectories
            self.traj = cppHREPS.get_traj()
        else:
            self.traj = self.append_trajectories(cppHREPS.get_traj())
        self.ti_info = cppHREPS.get_ti_info()
        #self.xi_schedule = self.ti_info[0]

#        if np.array(self.ti_info[0]).shape[0] == 0:
#            self.xi_schedule = self.ti_info[0]
#        else:
#            self.xi_schedule = np.array(self.ti_info[0])[:,1]

        # Get HREX information from C++ PosteriorSampler object
        self.exchange_info = pd.DataFrame(cppHREPS.get_exchange_info())
        if (self.attempt_lambda_swap_every != 0) or (attempt_lambda_swap_every != nsteps):
            self.exchange_info["energies"] = [np.around(np.array(energy_pair), decimals=2) for energy_pair in self.exchange_info["energies"].to_numpy()]
            self.exchange_info = self.exchange_info.round({"exchange %":2})
        # Save acceptance information
        acceptance_info = []
        self.E = []
        self.indices, self.states = [],[]
        self.xi_traces = [[] for l in range(len(self.expanded_values))]
        for l,e_vals in enumerate(self.expanded_values):
            lam,xi = e_vals
            accept = self.traj[l].sep_accept[0]
            accept_dict = {"lambda": lam, "xi": xi}
            [accept_dict.update({f"{self.rest_type[i]}": np.around(accept[i], decimals=1)}) for i in range(len(self.rest_type))]
            accept_dict.update({"state": np.around(accept[-1], decimals=1)})
            acceptance_info.append(accept_dict)
            trajectory = np.array(self.traj[l].__dict__["trajectory"], dtype=object)
            self.E.append(trajectory[-1][1])
            self.indices.append(np.concatenate(trajectory[-1][4]))
            self.states.append(trajectory[-1][3])
            self.xi_traces[l] = trajectory.T[7]
        self.acceptance_info = pd.DataFrame(acceptance_info)


    def update_prior(self, energies, diff_energies=None, diff2_energies=None):
        """function for FF optimization"""

        # Loop over the energies for each s and r value
        for l in range(1,len(self.lambda_values)):
            for p,energy in enumerate(energies):
                for r in range(len(self.ensembles[l][p])):
                    self.ensembles[l][p][r].energy = energy*self.lambda_values[l]
                    if diff_energies is not None:
                        self.ensembles[l][p][r].diff_energy = diff_energies[p]*self.lambda_values[l]
                    if diff2_energies is not None:
                        self.ensembles[l][p][r].diff2_energy = diff2_energies[p]*self.lambda_values[l]
        self.compute_logZ()


     # FIXME:
    #def get_approximate_score_using_TI(self):
    def get_score_using_TI(self):
        x,y = self.ti_info
        x = np.array(x)[:,1]
        # Perform interpolation
        f = interpolate.interp1d(x, y, kind='cubic')
        # Define the integral function
        integrand = lambda x: f(x)
        # Calculate the integral using the trapezoidal rule
        integral = integrate.quad(integrand, np.min(x), 1)[0]
        return integral/self.nreplicas

    def get_energy_mode(self):
        values = []
        for c in range(len(self.traj)):
            traj = self.traj[c].__dict__["trajectory"]
            energies = np.array([traj[i][1] for i in range(len(traj))])
            values.append(mode(energies)[0][0]/self.nreplicas)
        values = np.array(values)
        return values#.mean()


    def integrate_xi_ensembles(self, multiprocess=True, progress=True, scale_energies=False,
            compute_derivative=False, capture_stdout=False, verbose=False, plot_overlap=False,
            return_u_kln=False, filename="./contour.png"):
        """Compute the u_kln matrix for all of the intermediates bridging xi = 0 -> 1.
        Build and return an MBAR object containing all of the thermodyanmic states.
        """

        if self.fmo or self.pmo: _sampler = self
        else: _sampler = None

        if self.xi_integration == False:
            #raise(AttributeError, "Thermodynamic integration was not used. xi_integration==False")
            raise(ValueError, "Thermodynamic integration was not used. xi_integration==False")

        trajs = [traj.__dict__ for traj in self.traj]

        # IMPORTANT: only going to work with the zeroith trajectory
        traj_objects = np.array(trajs[0]['trajectory'], dtype=object).T
        if self.change_xi_every < self.write_every:
            raise(ValueError, "`change_xi_every` cannot be smaller than `write_every`")

        cdef int count = 0
        cdef int _step = 0
        N_k,ensembles = [],[]

        for i,step in enumerate(traj_objects[0]):
            count += 1
            if (step-_step) > self.change_xi_every:
                _step = step
                N_k.append(count)
                ensembles.append(self.ensembles[0])
                count = 0
            elif i == (len(traj_objects[0])-1):
                N_k.append(count)
                ensembles.append(self.ensembles[0])

        N_k = np.array(N_k)
        logZs = [self.logZs[0]]*len(ensembles)
        logZs = np.array(logZs)

        cdef vector[double] expanded_values
        cdef vector[vector[double]] expanded_trace
        cdef vector[vector[vector[double]]] expanded_traces

        #cdef vector[double] xi_trace
        #cdef vector[vector[double]] xi_traces
        cdef vector[double] parameters
        cdef vector[vector[double]] parameter_trace
        cdef vector[vector[vector[double]]] parameter_traces
        cdef vector[int] states
        cdef vector[vector[int]] state_trace
        cdef vector[vector[vector[int]]] state_traces
        cdef vector[double] energy_trace
        cdef vector[vector[double]] energy_traces

        cdef int c = 0
        cdef int val = 0
        for l,num in enumerate(N_k):
            val += num
            #print(f"Iteration: {l}, c: {c}, val: {val}, num: {num}, len: {len(traj_objects[0])}")
            if c < len(traj_objects[0]) and val <= len(traj_objects[0]): pass
            else: print("Index out of bounds or empty object")
            expanded_trace.clear()
            _expanded_trace = traj_objects[7][c:val]

            state_trace.clear()
            parameter_trace.clear()
            energy_trace.clear()
            _states = traj_objects[3][c:val]
            _parameters = traj_objects[5][c:val]
            _energies = traj_objects[1][c:val]

            if len(traj_objects) > 0 and c < len(traj_objects[0]) and val <= len(traj_objects[0]): pass
            else: print("Index out of bounds or empty object")

            for i,energy in enumerate(_energies):
                expanded_values.clear()
                for j in range(len(_expanded_trace[i])):
                    expanded_values.push_back(_expanded_trace[i][j])
                expanded_trace.push_back(expanded_values)

                energy_trace.push_back(energy)
                states.clear()
                parameters.clear()
                for j in range(len(_states[i])):
                    states.push_back(_states[i][j])
                for j in range(len(_parameters[i])):
                    for k in range(len(_parameters[i][j])):
                        parameters.push_back(_parameters[i][j][k])
                state_trace.push_back(states)
                parameter_trace.push_back(parameters)
            state_traces.push_back(state_trace)
            energy_traces.push_back(energy_trace)
            parameter_traces.push_back(parameter_trace)
            expanded_traces.push_back(expanded_trace)
            c = val

        cdef bool _compute_derivative = compute_derivative
        cdef vector[float] _logZs
        for logZ in logZs: _logZs.push_back(logZ)
        cdef GFE u
        # FIXME: Segfaults here from time to time
        u = get_u_kln_and_states_kn(ensembles, state_traces, energy_traces, parameter_traces,
                expanded_traces, _logZs, progress, capture_stdout,
                scale_energies, _compute_derivative, multiprocess, sampler=_sampler)


        u_kln = np.array(u.u_kln)
        self.u_kln = u_kln

        if return_u_kln: return u_kln

        x_kindices = np.array(list(range(len(N_k)))[::-1])
        mbar = MBAR(u_kln, N_k, verbose=self.verbose)
        #self.mbar = mbar

        if plot_overlap:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from mpl_toolkits.axes_grid1 import make_axes_locatable


            overlap = mbar.compute_overlap()
            ti_info = self.ti_info
            #print("ti_info = ",ti_info)
            xi_trace = np.array(self.xi_schedule)
            overlap_matrix = overlap["matrix"]
            force_constants = [r"$\xi=%0.2f$"%float(value) for value in xi_trace]
            #print("force_constants = ",force_constants)

            # Mask zero values so they are set to 'bad' in the colormap and appear white
            masked_overlap_matrix = np.ma.masked_array(overlap_matrix, mask=(overlap_matrix == 0))

            cmap = plt.cm.viridis_r.copy()

            fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the figsize as desired
            im = ax.pcolor(masked_overlap_matrix, edgecolors='k', linewidths=2, cmap=cmap)

            # Add annotations
            for i in range(len(overlap_matrix)):
                for j in range(len(overlap_matrix[i])):
                    value = overlap_matrix[i][j]

                    if value >= 0.01:
                        #text_color = 'white' if value < 0.5 else 'black'
                        text_color = 'white' if value > 0.5 else 'black'
                        ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color=text_color,
                                fontsize=12)  # Adjust fontsize as desired
                    else:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white'))

            # Set tick positions and labels
            ax.set_xticks(np.arange(len(force_constants)) + 0.5, minor=False)
            ax.set_yticks(np.arange(len(force_constants)) + 0.5, minor=False)
            ax.set_xticklabels(force_constants, rotation=90, size=16)
            ax.set_yticklabels(force_constants, size=16)
            ax.tick_params(axis='x', direction='inout')
            ax.tick_params(axis='y', direction='inout')
            ax.set_xticklabels(ax.get_xticklabels(), ha='left')
            ax.set_yticklabels(ax.get_yticklabels(), va='bottom')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label("Overlap probability between states", size=16)  # Set the colorbar label
            fig.tight_layout()
            fig.savefig(f"{filename}", dpi=400)


###################################
        if compute_derivative:
            #u_kln,states_kn,Nr_array,diff_u_kln,diff2_u_kln =
            diff_mbar = MBAR(np.array(u.diff_u_kln), N_k, verbose=verbose)
            diff2_mbar = MBAR(np.array(u.diff2_u_kln), N_k, verbose=verbose)
            #return mbar.f_k, diff_mbar.f_k, diff2_mbar.f_k

            diff_mbar2 = MBAR(np.array(u.diff_u_kln)**2, N_k, verbose=verbose)
            return mbar, diff_mbar, diff2_mbar, diff_mbar2
        else:
            return mbar


    def get_mbar_obj_for_TI(self, multiprocess=True, progress=True, scale_energies=False,
            compute_derivative=False, capture_stdout=False, verbose=False, plot_overlap=False,
            return_u_kln=False, filename="./contour.png"):
        warnings.warn("`get_mbar_obj_for_TI` is deprecated and will be removed in a future version. Use `integrate_xi_ensembles` instead.", DeprecationWarning, stacklevel=2)
        return self.integrate_xi_ensembles(multiprocess, progress, scale_energies,
            compute_derivative, capture_stdout, verbose, plot_overlap,
            return_u_kln, filename)





    def save_trajectories(self, outdir, save_object=False):

        for l,e_vals in enumerate(self.expanded_values):
            lam,xi = e_vals
            self.traj[l].process_results(f"{outdir}/traj_lambda{(lam,xi)}.npz", save_object=save_object)

    def append_trajectories(self, trajs):

        for l,e_vals in enumerate(self.expanded_values):
            lam,xi = e_vals
            last_idx = len(self.traj[l].trajectory)
            last_step = self.traj[l].trajectory[last_idx-1][0]
            for step in range(len(trajs[l].trajectory[1:])):
                trajs[l].trajectory[step][0] += last_step
                self.traj[l].trajectory.append(trajs[l].trajectory[step])
            self.traj[l].state_trace = self.traj[l].state_trace + trajs[l].state_trace
            self.traj[l].sem_trace = self.traj[l].sem_trace + trajs[l].sem_trace
            self.traj[l].sse_trace = self.traj[l].sse_trace + trajs[l].sse_trace
            self.traj[l].sseB_trace = self.traj[l].sseB_trace + trajs[l].sseB_trace
            self.traj[l].sseSEM_trace = self.traj[l].sseSEM_trace + trajs[l].sseSEM_trace
            self.traj[l].traces = self.traj[l].traces + trajs[l].traces
            self.traj[l].sampled_parameters = [self.traj[l].sampled_parameters[i] + trajs[l].sampled_parameters[i] for i in range(len(trajs[l].sampled_parameters))]
            self.traj[l].state_counts = [self.traj[l].state_counts[i] + trajs[l].state_counts[i] for i in range(len(trajs[l].state_counts))]
        return self.traj


    def plot_exchange_info(self, xlim=(-100, 10000), figname=None, figsize=(10,10)):

        if self.exchange_info.empty:
            print(f"Empty DataFrame. Unable to create plot due to zero lambda exchanges.\n")
            return None

        exc = self.exchange_info

        # Extract the lambdas and xis data
        steps = np.sort(np.array(list(set(exc["step"].to_numpy()))))
        steps = [0]+steps

        # NOTE: set initial starting point for 0 steps
        lam_traj = {f"{i}": [i] for i,val in enumerate(self.expanded_values)}
        lam_traj["x"] = [0]

        # Loop over the steps and extract the (lambda, xi) pairs that swapped
        for step in steps:
            df = exc.iloc[np.where(exc["step"]==step)[0]]
            accept_loc = []
            for row in df.iterrows():
                row = pd.DataFrame(row[1]).transpose()
                accept,indices = int(row["accepted"]),row["indices"].to_list()[0]
                index0 = indices[0]
                index1 = indices[1]

                if accept:
                    accept_loc.append(index0)
                    accept_loc.append(index1)
                    lam_traj[f"{index0}"].append(index1)
                    lam_traj[f"{index1}"].append(index0)
            accept_loc = list(set(accept_loc))
            for i,val in enumerate(self.expanded_values):
                if i not in accept_loc:
                    lam_traj[f"{i}"].append(i)
            lam_traj["x"].append(step)

        lam_traj = pd.DataFrame(lam_traj)
        lam_traj.columns = [str(val) for val in self.expanded_values]+["x"]
        # Plot
        ax = lam_traj.iloc[0:].plot.line(x="x", figsize=figsize, legend=False)
        ax.set_ylabel(r"$(\lambda, \xi)$", fontsize=18)
        ax.set_xlabel("steps", fontsize=18)
        ax.set_xlim(xlim[0], xlim[1])
        ticks = [str(val) for val in self.expanded_values]
        nticks = len(ticks)
        ax.set_yticks(range(nticks))
        ax.set_yticklabels(ticks)
        # Setting the ticks and tick marks
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks(),]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels(),]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                mark.set_rotation(s=15)
        fig = ax.get_figure()
        fig.tight_layout()
        if figname: fig.savefig(figname)

    def get_sem_trace_as_df(self):
        """
        Will only grab the sigma for each restraint....

        cols = [c for c in sigma_SEM.columns if "gamma" not in c]
        """

        rests = self.rest_type.copy()
        rests = [c for c in rests if "sigma" in c]
        result = []

        for l,e_vals in enumerate(self.expanded_values):
            lam,xi = e_vals
            sem_trace = self.traj[l].sem_trace

            restraints = []
            nth_rest = 0
            Nd = 0
            for i,R in enumerate(self.ensembles[l][0]):
                if f"{rests[i]}" == f"{rests[i-1]}":
                    restraints.append(f"{rests[i]}{Nd+1}") #rest+str(d) for rest in rests[i]])
                    Nd += 1
                else:
                    Nd = 0
                    restraints.append(f"{rests[i]}{Nd}") #rest+str(d) for rest in rests[i]])

            sem_trace = pd.DataFrame(sem_trace, columns=restraints)
            sem_trace = sem_trace.loc[:,~sem_trace.columns.duplicated(keep='last')]
            result.append(sem_trace)
        return result


    def get_score(self):

        trajs = [traj.__dict__ for traj in self.traj]
        u_trajs = [np.array(trajs[i]['trajectory']).T[1] for i in range(len(self.expanded_values))]
        try:
            Z0, df0_model1 = compute_f0(u_trajs[0])
            Z1, df0_model2 = compute_f0(u_trajs[-1])
            f = -np.log(Z1/Z0)
        except(Exception) as e:
            f = np.nan
        return f




    def get_results(self, f_k=False, progress=True, capture_stdout=False,
            scores_only=False, compute_derivative=False, k_indices=None,
            return_sigma=False, verbose=True):

        results = {}
        expanded_values = self.expanded_values.copy()
        _expanded_values = []
        if k_indices:
            # NOTE: order the ensembles, trajectories, logZs according to the reference
            _trajs, _ensembles, _logZs = [], [], []
            for index in k_indices:
                _trajs.append(self.traj[index].__dict__)
                _ensembles.append(self.ensembles[index])
                _logZs.append(self.logZs[index])
                _expanded_values.append(self.expanded_values[index])
            self.expanded_values = _expanded_values
        else:
            _ensembles = self.ensembles
            _trajs = [traj.__dict__ for traj in self.traj]
            _logZs = self.logZs
            _expanded_values = self.expanded_values.copy()

        N_k = np.array( [len(_trajs[i]['trajectory']) for i in range(len(_expanded_values))] )
        if compute_derivative:
            u_kln,states_kn,Nr_array,diff_u_kln,diff2_u_kln = u_kln_and_states_kn(
                    ensembles=_ensembles, trajs=_trajs, nstates=self.nstates,
                    logZs=_logZs, progress=progress, capture_stdout=capture_stdout, compute_derivative=compute_derivative)
        else:
            u_kln,states_kn,Nr_array = u_kln_and_states_kn(
                    ensembles=_ensembles, trajs=_trajs, nstates=self.nstates, logZs=_logZs,
                    progress=progress, capture_stdout=capture_stdout, compute_derivative=compute_derivative)
        self.u_kln,self.states_kn,self.Nr_array = np.array(u_kln),np.array(states_kn),np.array(Nr_array)
        mbar = MBAR(u_kln, N_k, verbose=verbose)
        self.N_eff = mbar.compute_effective_sample_number()
        self.expanded_values = expanded_values


        if compute_derivative:

            diff_mbar = MBAR(np.array(diff_u_kln), N_k, verbose=verbose)

            diff2_mbar = MBAR(np.array(diff2_u_kln), N_k, verbose=verbose)
            diff_mbar2 = MBAR(np.array(diff_u_kln)**2, N_k, verbose=verbose)

            diff2 = np.array([np.nansum([diff2_mbar.f_k[j], -diff_mbar2.f_k[j]/(self.nreplicas), diff_mbar.f_k[j]**2/(self.nreplicas)]) for j in range(len(diff_mbar.f_k))])

            #return mbar.f_k, diff_mbar.f_k, diff2_mbar.f_k
            if return_sigma:
                diff_mbar = diff_mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=False)
                return mbar.f_k, diff_mbar, diff2
            else: return mbar.f_k, diff_mbar.f_k, diff2

        else:
            if scores_only: return mbar.f_k

        _results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
        Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
        output = mbar.compute_overlap()
        results["overlap"] = output["scalar"] # output["eigenvalues"], output["matrix"]
        u_kn = kln_to_kn(u_kln, N_k=N_k)
        if type(f_k) == np.ndarray:
            df_df = mbar_solvers.mbar_objective_and_gradient(u_kn, N_k, f_k)
        else:
            df_df = mbar_solvers.mbar_objective_and_gradient(u_kn, N_k, mbar.f_k)
        H = mbar_solvers.mbar_hessian(u_kn, N_k, mbar.f_k)
        results["objective"] = df_df[0]
        results["gradient"] = df_df[1]
        results["hessian"] = H

        beta = 1.0 # keep in units kT
        f_df = np.zeros( (len(self.lambda_values), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
        f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
        f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std

        # Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
        # Here, A_kn[k,n] = A(x_{kn})
        #(A_k, dA_k) = mbar.computeExpectations(A_kn)
        P_dP = np.zeros( (self.nstates, 2*len(self.lambda_values)) )  # left columns are P, right columns are dP
        nreplicas = len(self.states_kn[-1,-1])
        # NOTE: Get populations for each state
        for i in range(self.nstates):
            sampled = np.array([np.where(states_kn[:,:,r]==i,1,0) for r in range(nreplicas)])
            #A_kn = sampled.sum(axis=0)/nreplicas
            A_kn = sampled.sum(axis=0)/Nr_array #/nreplicas
            output = mbar.compute_expectations(A_kn, uncertainty_method='approximate')
            p_i, dp_i = output["mu"],output["sigma"]
            P_dP[i,0:len(self.lambda_values)] = p_i
            P_dP[i,len(self.lambda_values):2*len(self.lambda_values)] = dp_i
        pops, dpops = P_dP[:, 0:len(self.lambda_values)], P_dP[:, len(self.lambda_values):2*len(self.lambda_values)]

        results["scores"] = f_df[:,0]
        results["scores_std"] = f_df[:,1]
        results["pops"] = pops
        results["N_eff"] = mbar.compute_effective_sample_number()
        return results
# }}}

# Restraint/Sampler methods:{{{


# NOTE: These are helper functions to be used in C++ sampling. Also can be used as Python functions

@cython.boundscheck(False)
cdef public vector[vector[double]] get_fwd_model_weights(object sampler):
    """
    """

    cdef vector[vector[double]] weights
    cdef vector[double] _weights
    for vec in sampler.fwd_model_weights:
        _weights.clear()
        for val in vec:
            _weights.push_back(val)
        weights.push_back(_weights)
    return weights

@cython.boundscheck(False)
cdef public vector[double] compute_prior_populations(object ensemble):
    """Compute reference state logZ for the free energies to normalize."""

    cdef double Z = 0.0
    for s in ensemble:
        Z +=  np.exp( -np.array(s[0].energy, dtype=np.float64) )
    return np.array([np.exp( -np.array(s[0].energy, dtype=np.float64) ) for s in ensemble])/Z

@cython.boundscheck(False)
cdef public vector[vector[float]] compile_nuisance_parameters(object ensemble):
    """Compiles numpy arrays of allowed parameters for each nuisance parameter.
    """

    # Generate empty lists for each restraint to fill with nuisance parameters
    cdef vector[vector[float]] nuisance_parameters
    cdef vector[float] parameters
    cdef object R,
    for R in ensemble[0]:
        keys = R.__dict__.keys() # all attributes of the Child Restraint class
        allowed_keys = R.allow_order
        [allowed_keys.append(key) for key in keys if (key not in allowed_keys) and ("allowed_" in key)]
        for j in allowed_keys: # get the allowed parameters
            parameters.clear()
            [parameters.push_back(parameter) for parameter in getattr(R,j)]
            nuisance_parameters.push_back(parameters)
    return nuisance_parameters


@cython.boundscheck(False)
cdef public vector[vector[vector[double]]] get_restraint_attr(object ensemble, string key):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[vector[double]]] data
    cdef vector[vector[double]] _data
    cdef vector[double] m
    cdef object R
    cdef Py_ssize_t s,j
    for s in range(len(ensemble)):
        _data.clear()
        for R in ensemble[s]:
            m.clear()
            R.n = len(R.restraints) # NOTE: added this 07-10-23 in the case that the ensemble has been changed to remove specific restraints
            for j in range(R.n):
                m.push_back(R.restraints[j][key.decode()])
            _data.push_back(m)
        data.push_back(_data)
    #print(f"{key.decode()} = {data}")
    #print()
    return data


@cython.boundscheck(False)
cdef public vector[vector[vector[double]]] get_phi_angles(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[vector[double]]] data
    cdef vector[vector[double]] _data
    cdef vector[double] d
    cdef Py_ssize_t s,j,k
    for s in range(len(sampler.phi_angles)):
        _data.clear()
        for j in range(len(sampler.phi_angles[s])):
            d.clear()
            for k in range(len(sampler.phi_angles[s][j])):
                d.push_back(sampler.phi_angles[s][j][k])
            _data.push_back(d)
        data.push_back(_data)
    return data

@cython.boundscheck(False)
cdef public vector[double] get_phase_shifts(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[double] data
    cdef Py_ssize_t s
    for s in range(len(sampler.phase_shifts)):
        data.push_back(sampler.phase_shifts[s])
    return data

@cython.boundscheck(False)
cdef public vector[int] get_fmo_restraint_indices(object sampler):
    """Returns the fmo restraint indices"""

    cdef vector[int] data
    cdef Py_ssize_t s
    for s in range(len(sampler.fmo_restraint_indices)):
        data.push_back(sampler.fmo_restraint_indices[s])
    return data





@cython.boundscheck(False)
cdef public vector[vector[double]] get_fwd_model_parameters(object sampler, int l):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[double]] parameters
    cdef vector[double] _parameters
    cdef Py_ssize_t s,j
    for s in range(len(sampler.fwd_model_parameters[l])):
        _parameters.clear()
        for j in range(len(sampler.fwd_model_parameters[l][s])):
            _parameters.push_back(sampler.fwd_model_parameters[l][s][j])
        parameters.push_back(_parameters)
    return parameters




@cython.boundscheck(False)
cdef public vector[double] get_prior_model_parameter_attr(object sampler, string key):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[double] parameters
    cdef Py_ssize_t s
    obj = getattr(sampler, key.decode())
    for s in range(len(obj)):
        parameters.push_back(obj[s])
    return parameters


@cython.boundscheck(False)
cdef public vector[string] get_pmp_prior_models(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[string] parameters
    cdef Py_ssize_t s
    for s in range(len(sampler.pmp_prior_models)):
        parameters.push_back(strdup(str(sampler.pmp_prior_models[s]).encode('utf-8')))
    return parameters





@cython.boundscheck(False)
cdef public vector[vector[double]] get_fwd_model_parameter_attr(object sampler, string key):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[double]] parameters
    cdef vector[double] _parameters
    cdef Py_ssize_t s,j
    obj = getattr(sampler, key.decode())
    for s in range(len(obj)):
        _parameters.clear()
        for j in range(len(obj[s])):
            _parameters.push_back(obj[s][j])
        parameters.push_back(_parameters)
    return parameters


@cython.boundscheck(False)
cdef public vector[vector[double]] get_min_max_fwd_model_parameters(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[double]] parameters
    cdef vector[double] _parameters
    cdef Py_ssize_t s,j
    for s in range(len(sampler.min_max_fwd_model_parameters)):
        _parameters.clear()
        for j in range(len(sampler.min_max_fwd_model_parameters[s])):
            _parameters.push_back(sampler.min_max_fwd_model_parameters[s][j])
        parameters.push_back(_parameters)
    return parameters

@cython.boundscheck(False)
cdef public vector[vector[double]] get_fmp_prior_mus(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[double]] parameters
    cdef vector[double] _parameters
    cdef Py_ssize_t s,j
    for s in range(len(sampler.fmp_prior_mus)):
        _parameters.clear()
        for j in range(len(sampler.fmp_prior_mus[s])):
            _parameters.push_back(sampler.fmp_prior_mus[s][j])
        parameters.push_back(_parameters)
    return parameters

@cython.boundscheck(False)
cdef public vector[vector[double]] get_fmp_prior_sigmas(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[double]] parameters
    cdef vector[double] _parameters
    cdef Py_ssize_t s,j
    for s in range(len(sampler.fmp_prior_sigmas)):
        _parameters.clear()
        for j in range(len(sampler.fmp_prior_sigmas[s])):
            _parameters.push_back(sampler.fmp_prior_sigmas[s][j])
        parameters.push_back(_parameters)
    return parameters





@cython.boundscheck(False)
cdef public vector[vector[string]] get_fmp_prior_models(object sampler):
    """Returns the data for a particular restraint attribute when providing the key"""

    cdef vector[vector[string]] parameters
    cdef vector[string] _parameters
    cdef Py_ssize_t s,j
    #sampler_fmp_prior_models = sampler.fmp_prior_models.tolist() if isinstance(sampler.fmp_prior_models, np.ndarray) else sampler.fmp_prior_models

    for s in range(len(sampler.fmp_prior_models)):
        _parameters.clear()
        for j in range(len(sampler.fmp_prior_models[s])):
            #_parameters.push_back(sampler.fmp_prior_models[s][j])
            _parameters.push_back(strdup(str(sampler.fmp_prior_models[s][j]).encode('utf-8')))
        parameters.push_back(_parameters)
    return parameters


@cython.boundscheck(False)
cdef public vector[vector[double]] get_d_fmp(object sampler):
    """
    """

    cdef vector[vector[double]] d_fmp
    cdef vector[double] d
    cdef Py_ssize_t s,j
    for s in range(len(sampler.d_fmp)):
        d.clear()
        for j in range(len(sampler.d_fmp[s])):
            d.push_back(sampler.d_fmp[s][j])
        d_fmp.push_back(d)
    return d_fmp



def get_restraint_labels(ensemble):

    rest_type = []
    n,k = 0,str()
    for i,R in enumerate(ensemble[0]):
        keys = R.__dict__.keys() # all attributes of the Child Restraint class
        allowed_keys = R.allow_order
        [allowed_keys.append(key) for key in keys if (key not in allowed_keys) and ("allowed_" in key)]
        for j in [key.split("_")[-1] for key in allowed_keys]: #
            rest_type.append(str(j)+"_"+str(R.__repr__).split("_")[-1].split()[0].split(">")[0])
            val = rest_type[-1].split("_")[-1]
            if k != val: n,k = 0,val
            rest_type[-1] += f":{n}" # NOTE: added ":" 12-21-23
        n += 1
    return rest_type

cdef public vector[int] get_rest_index(object ensemble, vector[int] vec):
    cdef int i
    cdef object R
    for i,R in enumerate(ensemble[0]):
        keys = R.__dict__.keys() # all attributes of the Child Restraint class
        allowed_keys = R.allow_order
        [allowed_keys.append(key) for key in keys if (key not in allowed_keys) and ("allowed_" in key)]
        for j in allowed_keys: # get the allowed parameters
            vec.push_back(i)
    return vec


cdef public vector[int] get_para_indices(object ensemble, vector[int] vec):
    cdef int i
    cdef object R
    for i,R in enumerate(ensemble[0]):
        keys = R.__dict__.keys() # all attributes of the Child Restraint class
        index_keys = R.ind_order
        [index_keys.append(key) for key in keys if (key not in index_keys) and ("_index" in key)]
        for j in index_keys: # get the parameter indices
            vec.push_back(int(getattr(R, j)))
    return vec


cdef public void build_exp_ref(object ensemble, int rest_index):
    """Looks at each structure to find the average observables
    :math:`<r_{j}>`, then stores the reference potential info for each
    :attr:`Restraint` of this type for each structure.

    ``beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)``

    Args:
        rest_index(int): restraint index
    """

    cdef object s
    # collect distributions of observables r_j across all structures
    n_observables  = ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
    distributions = [[] for j in range(n_observables)]
    for s in ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
        for j in range(len(s[rest_index].restraints)):
            distributions[j].append( s[rest_index].restraints[j]['model'] )
    # Find the MLE average (i.e. beta_j) for each noe
    # calculate beta[j] for every observable r_j
    betas = np.zeros(n_observables)
    for j in range(n_observables):
        # the maximum likelihood exponential distribution fitting the data
        betas[j] =  np.array(distributions[j]).sum()/(len(distributions[j])+1.0)
    # store the beta information in each structure and compute/store the -log P_potential
    for s in ensemble:
        s[rest_index].betas = betas
        s[rest_index].compute_neglog_exp_ref()


cdef public build_gaussian_ref(object ensemble, int rest_index,
        bool use_global_ref_sigma):
    """Looks at all the structures to find the mean (:math:`\\mu`) and std
    (:math:`\\sigma`) of observables r_j then store this reference potential
    info for all restraints of this type for each structure.

    Args:
        rest_index(int): restraint index
        use_global_ref_sigma(bool):
    """

    cdef object s
    # collect distributions of observables r_j across all structures
    n_observables  = ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
    distributions = [[] for j in range(n_observables)]
    for s in ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
        for j in range(len(s[rest_index].restraints)):
            distributions[j].append( s[rest_index].restraints[j]['model'] )
    # Find the MLE mean (ref_mu_j) and std (ref_sigma_j) for each observable
    ref_mean  = np.zeros(n_observables)
    ref_sigma = np.zeros(n_observables)
    for j in range(n_observables):
        ref_mean[j] =  np.array(distributions[j]).mean()
        squared_diffs = [ (d - ref_mean[j])**2.0 for d in distributions[j] ]
        ref_sigma[j] = np.sqrt( np.array(squared_diffs).sum() / (len(distributions[j])+1.0))
    if use_global_ref_sigma == True:
        # Use the variance across all ref_sigma[j] values to calculate a single value of ref_sigma for all observables
        global_ref_sigma = ( np.array([ref_sigma[j]**(-2.0) for j in range(n_observables)]).mean() )**-0.5
        for j in range(n_observables):
            ref_sigma[j] = global_ref_sigma
    # store the ref_mean and ref_sigma information in each structure and compute/store the -log P_potential
    for s in ensemble:
        s[rest_index].ref_mean = ref_mean
        s[rest_index].ref_sigma = ref_sigma
        s[rest_index].compute_neglog_gaussian_ref()


cdef public void build_exp_ref_pf(object ensemble, int rest_index):
    """Calculate the MLE average PF values for restraint j across all structures,

    ``beta_PF_j = np.array(protectionfactor_distributions[j]).sum()/(len(protectionfactor_distributions[j])+1.0)``

    then use this information to compute the reference prior for each structures.

    .. tip:: **(not required)** an additional method specific for protection factor
    """

    cdef object s
    # collect distributions of observables r_j across all structures
    n_observables  = ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
    for s in ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
        s[rest_index].betas = []
    # for each restraint, find the average model_protectionfactor (a 6-dim array in parameter space) across all structures
    for j in range(len(s[rest_index].restraints)):
        running_total = np.zeros(ensemble[0][rest_index].restraints[j]['model'].shape)
        for s in ensemble:
            running_total += s[rest_index].restraints[j]['model']
        beta_pf_j = running_total/(len(s[rest_index].restraints)+1.0)
        for s in ensemble:
            s[rest_index].betas.append(beta_pf_j)
    # With the beta_PF_j values computed (and stored in each structure), now we can calculate the neglog reference potentials
    for s in ensemble:
        s[rest_index].compute_neglog_exp_ref_pf()


cdef public void build_gaussian_ref_pf(object ensemble, int rest_index):
    """Calculate the mean and std PF values for restraint j across all structures,
    then use this information to compute a gaussian reference prior for each structure.

    .. tip:: **(not required)** an additional method specific for protection factor
    """

    cdef object s
    # collect distributions of observables r_j across all structures
    n_observables  = ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
    # Find the MLE mean (ref_mu_j) and std (ref_sigma_j) for each observable
    for s in ensemble:
        s[rest_index].ref_mean = []
        s[rest_index].ref_sigma = []
    # for each restraint, find the average model_protectionfactor (a 6-dim array in parameter space) across all structures
    for j in range(len(s[rest_index].restraints)):
        mean_PF_j  = np.zeros( ensemble[0][rest_index].restraints[j]['model'].shape )
        sigma_PF_j = np.zeros( ensemble[0][rest_index].restraints[j]['model'].shape )
        for s in ensemble:
            mean_PF_j += s[rest_index].restraints[j]['model']   # a 6-dim array
        mean_PF_j = mean_PF_j/(len(s[rest_index].restraints)+1.0)
        for s in ensemble:
            sigma_PF_j += (s[rest_index].restraints[j]['model'] - mean_PF_j)**2.0
        sigma_PF_j = np.sqrt(sigma_PF_j/(len(s[rest_index].restraints)+1.0))
        for s in ensemble:
            s[rest_index].ref_mean.append(mean_PF_j)
            s[rest_index].ref_sigma.append(sigma_PF_j)
    for s in ensemble:
        s[rest_index].compute_neglog_gaussian_ref_pf()

cdef public vector[vector[double]] build_reference_potentials(object ensemble):

    cdef vector[vector[double]] ref_potentials
    cdef vector[double] ref
    cdef object R
    cdef Py_ssize_t s,i
    for s in range(len(ensemble)):
        ref.clear()
        for R in ensemble[s]:
            if R.ref.lower() == "uniform":
                ref.push_back(0.0)
            if R.ref.lower() == "exponential":
                ref.push_back(R.sum_neglog_exp_ref)
            if R.ref.lower() == "gaussian":
                ref.push_back(R.sum_neglog_gaussian_ref)
        ref_potentials.push_back(ref)
    return ref_potentials
# }}}

# Trajectory:{{{
cdef public Trajectory(object sampler, int ensemble_index):
    return PosteriorTrajectory(sampler, ensemble_index)

# NOTE: IMPORTANT: This is the new class for HRE only
class PosteriorTrajectory(object):
    def __init__(self, object sampler, int ensemble_index, verbose=False):
        """A container class to store and perform operations on the trajectories of
        sampling runs.

        Args:
            ensemble(list): ensemble of :attr:`Restraint` objects
            nreplicas(int): number of replicas
        """

        self.verbose = verbose
        self.model = sampler.model
        self.ensemble_index = ensemble_index
        ensemble = sampler.ensembles[self.ensemble_index]
        self.xi = sampler.expanded_values[self.ensemble_index][1]
        self.nreplicas = sampler.nreplicas
        self.nstates = len(ensemble)
        self.state_counts = np.ones(self.nstates)  # add a pseudo-count to avoid log(0) errors
        # Lists for each restraint inside a list
        self.sampled_parameters = []
        self.ref = [ []  for i in range(len(ensemble[0]))]  # parameters of reference potentials
        self.sep_accept = []     # separate accepted ratio
        self.state_trace = []
        self.sem_trace = []
        self.sse_trace = []
        self.sseB_trace = []
        self.sseSEM_trace = []
        # Generate a list of the names of the parameter indices for the traj header
        parameter_indices = []
        self.rest_type = sampler.rest_type
        self.allowed_parameters = compile_nuisance_parameters(ensemble)
        for j in self.allowed_parameters:
            self.sampled_parameters.append(np.zeros(len(j)))
        parameter_indices = sampler.indices
        self.restraints = [R.__class__.__name__ for R in ensemble[0]]
        self.unique_restraints = list(set(self.restraints))
        self.N_unique_restraints = len(self.unique_restraints)
        self.trajectory_headers = ["step", "E", "accept", "state",
                "para_index = %s"%parameter_indices]
        self.trajectory = []
#        self.model_optimization = []
        self.traces = []
        self.results = {}


    def process_results(self, filename='traj.npz', save_object=False):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables.

        Args:
            filename(str): path and filename of output MCMC trajectory

        .. tip:: [Future] Returns: Pandas DataFrame
        """

        stime = time.time()
        # Saving sse and sem traces as pkl
        outdir = filename.replace(filename.split("/")[-1],"")
        out = f"{outdir}/Sigma_SEM_trace_{filename.split('/')[-1].split('.npz')[0]}.pkl"
        # Saving results as npz
        try:
            self.results['model'] = np.array(self.model, dtype=object)
            self.results['rest_type'] = self.rest_type
            self.results['trajectory_headers'] = self.trajectory_headers
            self.results['trajectory'] = self.trajectory
#            self.results['model_optimization'] = self.model_optimization
            self.results['sep_accept'] = self.sep_accept
            self.results['allowed_parameters'] = self.allowed_parameters
            self.results['sampled_parameters'] = self.sampled_parameters
            self.results['ref'] = np.array(self.ref)
            self.results['traces'] = np.array(self.traces)
            self.results['state_trace'] = np.array(self.state_trace, dtype=object)
            self.results['sse_trace'] = np.array(self.sse_trace, dtype=object)
            self.write(filename, self.results)
            # Save Sampler object
        except(MemoryError) as e: print(e)
        if self.verbose: print(f"Time for processing: %.3f s" % (time.time() - stime));


    def write(self, file='traj.npz', *args, **kwds):
        """Writes a compact file of several arrays into binary format.
        Standardized: Yes ; Binary: Yes; Human Readable: No;

        Args:
            filename(str): path and filename of output MCMC trajectory

        :rtype: npz (numpy compressed file)

        https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
        """

        #if compress:
        np.savez_compressed(file, *args, **kwds)
        #else: np.savez(file, *args, **kwds)

# }}}

# analysis methods:{{{
# NOTE: helper functions for Analysis

@cython.boundscheck(False)
cpdef find_all_state_sampled_time(trace, Py_ssize_t nstates, bool verbose=True):
    """Determine which states were sampled and the states with zero counts.

    Args:
        trace(np.ndarray): trajectory trace
        nstates(int): number of states
    """

    cdef list frac = []
    cdef ndarray[double, ndim=1] all_states = np.zeros(nstates)
    cdef Py_ssize_t init = 0
    while 0 in all_states:
        if init == len(trace):
            if verbose: print('These states have not been sampled:\n', np.where(all_states == 0)[0])
            return 'null', frac
        else:
            all_states[trace[init]] += 1
            frac.append(float(len(np.where(all_states!=0)[0]))/float(nstates))
            init += 1
    return init, frac


# FIXME: this will not work because we removed sse and put in
# ndarray[double, ndim=1] fX and ndarray[double, ndim=1] d

@cython.boundscheck(False)
cpdef get_negloglikelihood(int nreplicas, string model, ndarray[double, ndim=1] sse,
        ndarray[double, ndim=1] sigmaSEM, ndarray[double, ndim=1] sigmaB, double scale,
        double Ndof, ndarray[double, ndim=1] sseB, ndarray[double, ndim=1] sseSEM,
        string data_uncertainty, double xi):
    """

    Args:
        trace(np.ndarray): trajectory trace
        nstates(int): number of states
    """

    cdef double result
    result = xi*get_data_restraint_energy(nreplicas, model, sse, sigmaSEM, sigmaB,
                        Ndof, sseB, data_uncertainty)
    return result
# }}}





