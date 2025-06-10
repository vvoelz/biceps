# libraries:{{{
import os, glob, time, re, math
import numpy as np
import pickle
import pandas as pd
from pymbar import MBAR
from pymbar.utils import kln_to_kn
from pymbar import mbar_solvers
from .Restraint import *
from .PosteriorSampler import *
from .PosteriorSampler import u_kln_and_states_kn
from .PosteriorSampler import find_all_state_sampled_time
from .PosteriorSampler import get_negloglikelihood
from .convergence import get_autocorrelation_time, exp_average
from .toolbox import get_files
from .toolbox import format_label
from .toolbox import bic,aic,hqic,aicc
import string
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# }}}

def matprint(mat, fmt="g"):
    """
    https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")



def get_trajs(files):
    trajs = []
    # Load in npz trajectories
    trajectory_files = files
    for filename in trajectory_files:
        print('Loading %s ...'%filename)
        traj = np.load(filename, allow_pickle=True )['arr_0'].item()
        trajs.append(traj)
    return trajs




class Analysis(object):
    def __init__(self, sampler=None, trajs=None, nstates=0, precheck=True, expanded_values=None,
            BSdir='BS.dat', popdir='populations.dat', MBAR=True, scale_energies=False,
            verbose=True, outdir=None, progress=True, multiprocess=True, capture_stdout=False,
            only_unique_values=False):
        """A class to perform analysis and plot figures.

        Args:
            sampler(object): PosteriorSampler object
            nstates(int): number of conformational states
            trajs(str): relative path to glob '*.npz' trajectories (analysis files and figures will be placed inside this directory)
            precheck(bool): find the all the states that haven't been sampled if any
            BSdir(str): relative path for BICePs score file name
            popdir(str): relative path for BICePs reweighted populations file name
            picfile(str): relative path for BICePs figure
        """

        if (sampler == None) and (trajs == None):
            raise ValueError("Must provide PosteriorSampler oject (sampler) OR relative path to glob '*.npz' trajectories (trajs).")

        self.multiprocess = multiprocess
        self.capture_stdout = capture_stdout
        self.trajs = trajs
        self.scale_energies = scale_energies
        self.progress = progress
        if outdir:  self.resultdir = outdir
        elif trajs: self.resultdir = os.path.dirname(trajs)
        else:       self.resultdir = os.path.dirname("./")
        self.verbose = verbose
        self.BSdir = os.path.join(self.resultdir,BSdir)
        self.popdir = os.path.join(self.resultdir,popdir)
        self.scheme = None
        self.traj = []

        self.lam = None
        self.f_df = None
        self.P_dp = None
        self.precheck = precheck
        # load necessary data first
        load_pickle_files = True
        if sampler != None:
            load_pickle_files = False
            if expanded_values: self.expanded_values = expanded_values
            else: self.expanded_values = sampler.expanded_values

            # FIXME: why the fuck is there indices?
            if only_unique_values: indices = [sampler.expanded_values.index(vals) for vals in self.expanded_values]
            else: indices = list(range(len(self.expanded_values)))

            self.ensembles = [ensemble for i,ensemble in enumerate(sampler.ensembles) if i in indices]
            self.model = np.array(sampler.model, dtype=object)
            self.Nd = 0
            for i in range(len(self.model[0])): self.Nd += len(self.model[0][i])
            self.nstates = self.model.shape[0]
            self.logZs = [logZ for i,logZ in enumerate(sampler.logZs) if i in indices]
            self.traj = [traj.__dict__ for i,traj in enumerate(sampler.traj) if i in indices]
            self.nreplicas = len(self.traj[0]["trajectory"][-1][3])
            self.lam = [vals[0] for vals in self.expanded_values]
            self.sem_traces = sampler.get_sem_trace_as_df()
            self.continuous_space = sampler.continuous_space
            self.sse_traces = np.array([traj.sse_trace for traj in sampler.traj], dtype=object) # shape: (#lam, # saved frames, # restraints, # data points)
            self.sseB_traces = np.array([traj.sseB_trace for traj in sampler.traj], dtype=object) # shape: (#lam, # saved frames, # restraints, # data points)
            self.sseSEM_traces = np.array([traj.sseSEM_trace for traj in sampler.traj], dtype=object) # shape: (#lam, # saved frames, # restraints, # data points)
            self.rest_type = sampler.rest_type
            self.f_k = sampler.f_k
            self.fwd_model_mixture = sampler.fwd_model_mixture
            self.fwd_model_weights = sampler.fwd_model_weights
        else:
            if sampler: load_pickle_files = False
            self.nstates = nstates
            if self.nstates == 0:
                raise ValueError("State number cannot be zero.")
            self.load_data(load_pickle_files)
            trajectory_files = get_files(self.trajs)
            self.lam = [ float( (s.split('lambda')[1]).replace('.npz','') ) for s in trajectory_files ]

        if self.precheck: self.get_total_fractions()

        # parse the lambda* filenames to get the full list of lambdas
        self.nlambda = len(self.traj)
        #if self.verbose: print('lam =', self.lam)
        self.scheme = self.traj[0]['rest_type']
        self.K = self.nlambda   # number of thermodynamic ensembles
        # next get MABR sampling done
        if MBAR: self.MBAR_analysis()


    def load_data(self, load_pickle_files=True):
        """Load input data from BICePs sampling (*npz and *pkl files)."""

        # Load in npz trajectories
        trajectory_files = get_files(self.trajs)
        for filename in trajectory_files:
            if self.verbose: print('Loading %s ...'%filename)
            traj = np.load(filename, allow_pickle=True )['arr_0'].item()
            self.traj.append(traj)
        self.nreplicas = len(traj['trajectory'][-1][3])

        # Load in cpickled sampler objects
        if load_pickle_files:
            self.sampler = []
            sampler_files = get_files(self.trajs.replace('.npz','.pkl'))
            if sampler_files == []:
                print(f"\n\nValueError:\nCheck 1) Analysis() is given a Sampler object or\
 2) *.pkl files (Sampler Ojbs) are being saved with npz trajectories.\n\n")
                raise(ValueError)

            for pkl_filename in sampler_files:
                if self.verbose: print('Loading %s ...'%pkl_filename)
                pkl_file = open(pkl_filename, 'rb')
                self.sampler.append( pickle.load(pkl_file) )

            try:
                self.ensembles = [sampler.ensemble for sampler in self.sampler]
                self.logZs = [sampler.logZ for sampler in self.sampler]
                self.model = np.array([np.array(sampler.model) for sampler in self.sampler])
            except(Exception) as e:
                try:
                    self.ensembles = self.sampler[-1].ensembles
                    self.logZs = self.sampler[-1].logZ
                    self.model = self.sampler[-1].model
                except(Exception) as e:
                    print(e)
                    exit()


    def get_total_fractions(self):
        steps = []
        fractions = []
        for i in range(len(self.traj)):
            s,f = find_all_state_sampled_time(np.array(self.traj[i]["trajectory"], dtype=object).T[3], self.nstates, self.verbose)
            steps.append(s)
            fractions.append(f)
        total_fractions = np.concatenate(fractions)


    #def get_forward_model(self, traj_index=-1, nth_restraint=0, stride=1):
    #    npz = self.traj[traj_index]
    #    state_trace = pd.DataFrame([npz['trajectory'][i][3] for i in range(len(npz["trajectory"]))])
    #    state_trace_npy = state_trace.to_numpy()
    #    forward_model = []
    #    restraint = nth_restraint # TODO: FIXME: This should be an argument. Otherwise you will only be getting a single restraint that will
    #    # always be the first restraint type
    #    n_rest = len(self.model[0])
    #    k = nth_restraint
    #    #for k in range(n_rest): # loop through Restraints
    #    Nd = len(self.model[0][k])
    #    for i in range(Nd): # loop through Restraints
    #        for states in state_trace_npy[::stride]:
    #           avg_r = np.mean([self.model[state][k][i] for state in states], axis=0)
    #           R = self.ensembles[traj_index][0][k]
    #           #NOTE: forward model is an average over the states
    #           # Create list of dictionaries that matches Restraints
    #           f = [dict(restraint=k, data_point=i,
    #               model=avg_r, exp=R.restraints[i]["exp"],
    #               weight=R.restraints[i]["weight"])]
    #           df = pd.DataFrame(f)
    #           forward_model.append(df)
    #    return forward_model


    def get_sigmaB_trace(self, traj_index=-1):

        # get trajectory for lambda = 1.0
        npz = self.traj[traj_index]
        traj = npz["trajectory"]
        allowed_parameters = npz["allowed_parameters"]
        columns = self.rest_type #self.get_restraint_labels()
        indices,rests = np.array([[i,rest] for i,rest in enumerate(columns) if "sigma" in rest]).T
        parameters = []
        for i in range(len(traj)): # total number of steps
            para_step = np.array(traj[i][4])
            para_dict = {}
            for k,index in enumerate(indices):
                ind = para_step[k][0]
                para_dict[rests[k]] = allowed_parameters[int(index)][int(ind)]
            parameters.append(para_dict)
        sigma_B = pd.DataFrame(parameters)#.drop(["gamma_noe"], axis=1)
        return sigma_B


    def get_traces(self, traj_index=-1):

        npz = self.traj[traj_index]
        columns = self.rest_type #self.get_restraint_labels()
        df = pd.DataFrame(np.array(npz["traces"]).transpose(), columns)
        df = df.transpose()
        return df



    def get_SEM_trace(self, traj_index=-1):
        """Return the trajectory trace for the standard error of the mean:"""

        if self.sem_traces: return self.sem_traces[traj_index]

        if self.trajs:
            file = get_files(self.resultdir+f"/Sigma_SEM_*_lambda{self.lambda_values[traj_index]}*.pkl")[0]
            sigma_SEM = pd.read_pickle(file)
            return sigma_SEM


    def get_restraint_intensity(self, traj_index=-1):
        """Compute and return the restraint intensity over all data restraints
        """

        out = os.path.join(self.resultdir,f"restraint_intensity.txt")
        sigma_SEM = self.get_SEM_trace(traj_index)
        # TODO: FIXME:
        """
        A value is trying to be set on a copy of a slice from a DataFrame.
        Try using .loc[row_indexer,col_indexer] = value instead

        See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
          sigma_SEM[col] = [np.max(vals) for vals in sigma_SEM[col].to_numpy()]
        """
        for i,col in enumerate(sigma_SEM.columns):
            if type(sigma_SEM[col][0]) == list:
                #sigma_SEM[col] = [np.mean(vals) for vals in sigma_SEM[col].to_numpy()]
                sigma_SEM[col] = [np.max(vals) for vals in sigma_SEM[col].to_numpy()]

        sigma_B = self.get_sigmaB_trace(traj_index)
        # columns to drop
        cols = [c for c in sigma_B.columns if "sigma" in c]
        sigma_B = sigma_B[cols]
        for col in sigma_B.columns:
            if type(sigma_B[col][0]) == list:
                sigma_B[col] = [np.mean(vals) for vals in sigma_B[col].to_numpy()]
        k = 1.0/(sigma_B.mean().to_numpy()**2 + sigma_SEM.mean().to_numpy()**2) * self.nreplicas**2
        return k

    def get_restraint_intensity_distributions(self):
        """Compute and return the restraint intensity over all data restraints
        """

        sigma_SEM = self.get_SEM_trace()
        # TODO: FIXME:
        """
        A value is trying to be set on a copy of a slice from a DataFrame.
        Try using .loc[row_indexer,col_indexer] = value instead

        See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
          sigma_SEM[col] = [np.max(vals) for vals in sigma_SEM[col].to_numpy()]
        """
        for i,col in enumerate(sigma_SEM.columns):
            if type(sigma_SEM[col][0]) == list:
                sigma_SEM[col] = [np.max(vals) for vals in sigma_SEM[col].to_numpy()]
        sigma_B = self.get_sigmaB_trace()
        cols = [c for c in sigma_B.columns if "sigma" in c]
        sigma_B = sigma_B[cols]
        for col in sigma_B.columns:
            if type(sigma_B[col][0]) == list:
                sigma_B[col] = [np.mean(vals) for vals in sigma_B[col].to_numpy()]
        k = 1.0/(sigma_B.to_numpy()**2 + sigma_SEM.to_numpy()**2) * self.nreplicas**2
        columns = sigma_SEM.columns.to_list()
        k = pd.DataFrame(k, columns=columns)
        return k


    def plot_restraint_intensity(self, plottype="hist",
            figname="restraint_intensity.pdf", figsize=None,
            label_fontsize=12, legend_fontsize=10):
        """
        """

        k = self.get_restraint_intensity_distributions()
        cols = k.columns.to_list()

        labels = []
        for i,col in enumerate(cols):
            df_array = k[col].to_numpy()
            if all(df_array == np.ones(df_array.shape)): continue
            if all(df_array == np.zeros(df_array.shape)): continue
            labels.append(col)


        n_rows = 20
        if len(labels) > n_rows:
            if self.verbose:
                print(f"Number of nuisance parameter traces: {len(labels)}\n\
Too many plots for a single figure... only plotting the first {n_rows}.")
        else: n_rows = len(labels)

        n_columns = 1
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_columns)

        for i,label in enumerate(labels):
            ax = plt.subplot(gs[(i,0)])
            df_array = k[label].to_numpy()
            if plottype == "step": pass

            if plottype == "hist": ax = k.plot.hist(bins=40, alpha=0.5, edgecolor="k", ax=ax)

            ax.legend(loc='best')
            ax.set_ylabel("PDF", fontsize=16)
            ax.set_xlabel(r"$k$ [$k_{b}T$]", fontsize=16)
            if i == (n_rows-1): break
        plt.tight_layout()
        plt.savefig(os.path.join(self.resultdir,figname))


    def get_ML_parameters(self, model=0, sigma_only=False):
        #nParameters = self.nstates + len(self.scheme)
        if sigma_only:
            indices = [i for i,rest in enumerate(self.scheme) if "sigma" in rest]
        else:
            indices = [i for i,rest in enumerate(self.scheme)]

        t1 = self.traj[model]
        max_likelihood = []
        for k in indices:
            x, y = t1['allowed_parameters'][k], t1['sampled_parameters'][k]
            max_likelihood.append(x[np.argmax(y)])
        return max_likelihood


    def get_max_likelihood(self, model=0):

        nParameters = self.get_number_of_parameters() #self.nstates + len(self.scheme)
        t1 = self.traj[model]
        max_likelihood = []
        for k in range(len(self.scheme)):
            x, y = t1['allowed_parameters'][k], t1['sampled_parameters'][k]
            max_likelihood.append(x[np.argmax(y)])
        return max_likelihood



    def get_max_likelihood_parameters(self, model=0, sigma_only=False):

        #columns = self.get_restraint_labels()
        columns = self.rest_type #self.get_restraint_labels()
        if sigma_only:
            indices = [i for i,rest in enumerate(columns) if "sigma" in rest]
        else:
            indices = [i for i,rest in enumerate(columns)]

        t1 = self.traj[model]

        if self.continuous_space:
            hist = self.get_counts_and_bins_for_continuous_space(model=model, return_all=1)

        max_likelihood = {}
        for k in indices:
            if self.continuous_space:
                counts, bins = hist[columns[k]]
                max_likelihood[columns[k]] = [bins[np.argmax(counts)]]
            else:
                x, y = t1['allowed_parameters'][k], t1['sampled_parameters'][k]
                max_likelihood[columns[k]] = [x[np.argmax(y)]]
        max_likelihood = pd.DataFrame(max_likelihood)
        return max_likelihood



    def MBAR_analysis(self, debug=False):
        """MBAR analysis for populations and BICePs score"""

        results = {}
        # Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
        #   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
        self.K = len(self.expanded_values)   # number of thermodynamic ensembles
        N_k = np.array( [len(self.traj[i]['trajectory']) for i in range(len(self.expanded_values))] )
        self.u_kln, self.states_kn, self.Nr_array = u_kln_and_states_kn(ensembles=self.ensembles,
                trajs=self.traj, nstates=self.nstates, logZs=self.logZs, #fwd_model_weights=self.fwd_model_weights,
                progress=self.progress, scale_energies=self.scale_energies,
                capture_stdout=self.capture_stdout, #fwd_model_mixture=self.fwd_model_mixture,
                multiprocess=self.multiprocess, verbose=self.verbose)
        if debug:
            u_kln = self.u_kln.T
            print()
            for i in range(10):
                matprint(u_kln[i]);print()


        stime = time.time()
        # NOTE: Store these as public variables in case we want to use them to
        # calculate the biceps score with BIC in the future.
        #self.u_kln, self.N_k = u_kln, N_k
        self.N_k = N_k
        # Initialize MBAR with reduced energies u_kln and number of uncorrelated configurations from each state N_k.
        # u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where U_l(x) is the potential energy function for state l,
        # beta is the inverse temperature, and and x_kn denotes uncorrelated configuration n from state k.
        # N_k[k] is the number of configurations from state k stored in u_knm
        # Note that this step may take some time, as the relative dimensionless free energies f_k are determined at this point.
        mbar = MBAR(self.u_kln, self.N_k)

        # Extract dimensionless free energy differences and their statistical uncertainties.
#       (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
        #(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
        # NOTE: Get biceps score
        # NOTE: This function may need to be altered to gather more information about individual models
        #(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='approximate')
        try:
            _results = mbar.getFreeEnergyDifferences(uncertainty_method='approximate', return_theta=True, return_dict=True)
        except:
            _results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
        #print(_results.keys())
        Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
        self.Deltaf_ij = Deltaf_ij
        self.dDeltaf_ij = dDeltaf_ij
        try:
            self.N_eff = mbar.computeEffectiveSampleNumber()
        except:
            self.N_eff = mbar.compute_effective_sample_number()
        beta = 1.0 # keep in units kT
        #print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])
        self.f_df = np.zeros( (len(self.expanded_values), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
        self.f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
        self.f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std

        # Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
        # Here, A_kn[k,n] = A(x_{kn})
        #(A_k, dA_k) = mbar.computeExpectations(A_kn)
        self.P_dP = np.zeros( (self.nstates, 2*self.K) )  # left columns are P, right columns are dP
        if debug: print('state\tP\tdP')
        #states_kn = np.array(states_kn) # shape: (nlambdas, traj_steps_saved, nreplicas)
        self.u_kn = kln_to_kn(self.u_kln, N_k=self.N_k)
        try:
            self.computePerturbedFreeEnergies = mbar.computePerturbedFreeEnergies #(u_kn, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1e-10)
        except:
            self.compute_perturbed_free_energies = mbar.compute_perturbed_free_energies
        # NOTE: FIXME: changing self.nreplicas = len(states_kn[0,0])
        # this might override the nreplicas specified during __init__
        # ultimately, contradicts another self.nreplicas variable...
        #print("resetting the number of replicas..")
        self.nreplicas = len(self.states_kn[-1,-1])
        # NOTE: Get populations for each state
        for i in range(self.nstates):
            sampled = np.array([np.where(self.states_kn[:,:,r]==i,1,0) for r in range(self.nreplicas)])
            A_kn = sampled.sum(axis=0)/self.Nr_array #/self.nreplicas
            try:
                p_i, dp_i = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
            except:
                output = mbar.compute_expectations(A_kn, uncertainty_method='approximate')
                p_i, dp_i = output["mu"],output["sigma"]
            self.P_dP[i,0:self.K] = p_i
            self.P_dP[i,self.K:2*self.K] = dp_i
        pops, dpops = self.P_dP[:, 0:self.K], self.P_dP[:, self.K:2*self.K]
        if self.verbose: print(f"Time for MBAR: %.3f s" % (time.time() - stime));
        # save results
        self.save_MBAR()

    def save_MBAR(self):
        """save results (BICePs score and populations) from MBAR analysis"""

        if self.verbose: print('Writing %s...'%self.BSdir)
        np.savetxt(self.BSdir, self.f_df)

        if self.verbose: print('Writing %s...'%self.popdir)
        np.savetxt(self.popdir, self.P_dP)


    def get_model_scores(self, verbose=False):
        """Get BIC using -2loglik + logN * d
        """

        results = []
        for model in range(len(self.lam)):
            t = self.traj[model] # trajectory npz object
            self.nreplicas = len(t["trajectory"][0][3])
            # Using states and sigmas as parameters
            nParameters = self.get_number_of_parameters() # 1 + len(self.scheme)
            if verbose: print(f"Number of parameters: {nParameters}")
            nobs = self.get_number_of_observations()
            if verbose: print(f"Number of observations: {nobs}")

            if nobs <= nParameters: print("WARNING: BIC is only valid for N Observations >> N parameters.")

            pops = self.P_dP[:, model]

            # NOTE: Get experiment, stat_model, Ndof, etc
            experiment = []
            Ndof = 0
            data_likelihoods = []
            data_uncertainty = []
            stat_model = []
            for i in range(len(self.ensembles[0][0])):
                R = self.ensembles[0][0][i]
                _exp = []
                for k in range(len(R.restraints)):
                    _exp.append(R.restraints[k]["exp"])
                    Ndof += R.restraints[k]["weight"]
                    data_likelihoods.append(R.data_likelihood)
                    data_uncertainty.append(R.data_uncertainty)
                    stat_model.append(R.stat_model)
                experiment.append(_exp)
            experiment = np.array(experiment)

            # NOTE: compute the ensemble average observables
            try:
                ensemble_avg_obs = np.array([self.model[i]*w for i,w in enumerate(pops)])
                ensemble_avg_obs = np.concatenate(ensemble_avg_obs,axis=0)
                if len(ensemble_avg_obs) != 1:
                    ensemble_avg_obs = ensemble_avg_obs.sum(axis=0)

                ensemble_avg_obs = ensemble_avg_obs.reshape(experiment.shape)
            except(Exception) as e:

                ensemble_avg_obs = np.array([self.model[i]*w for i,w in enumerate(pops)])
                ensemble_avg_obs = np.concatenate(ensemble_avg_obs,axis=1)
                if len(ensemble_avg_obs) != 1:
                    ensemble_avg_obs = ensemble_avg_obs.sum(axis=1)

                ensemble_avg_obs = ensemble_avg_obs.reshape(experiment.shape)
            # NOTE: calculate SSE, sigmaB and sigmaSEM
            sse = np.concatenate((experiment-ensemble_avg_obs)**2)
            sigma_SEM = self.get_SEM_trace(model)
            cols = [c for c in sigma_SEM.columns if "sigma" in c]
            for i,col in enumerate(cols):
                if type(sigma_SEM[col][0]) == list:
                    sigma_SEM[col] = [np.max(vals) for vals in sigma_SEM[col].to_numpy()]
            sigma_SEM = sigma_SEM.mean().to_numpy()#[0]

            sigma_B = self.get_sigmaB_trace(model)
            cols = [c for c in sigma_B.columns if "sigma" in c]
            sigma_B = sigma_B[cols]
            for col in sigma_B.columns:
                if type(sigma_B[col][0]) == list:
                    sigma_B[col] = [np.mean(vals) for vals in sigma_B[col].to_numpy()]
            sigma_B = sigma_B.mean().to_numpy()#[0]

            max_likelihood_paras = self.get_ML_parameters(model, sigma_only=True)
            loglik = 0
            Cauchy_beta_issue_is_fixed = False
            if Cauchy_beta_issue_is_fixed:
                if (sigma_SEM.shape[0] > 1):

                    for i in range(len(sse)):
                        loglik += get_negloglikelihood(nreplicas=self.nreplicas, model=str.encode(stat_model[0]),
                                sse=np.array([sse[i]]), sigmaSEM=np.array([sigma_SEM[i]]), sigmaB=np.array([np.array(max_likelihood_paras)[i]]), scale=1.0,
                                #sse=sse, sigmaSEM=np.array([0.0]), sigmaB=np.array(max_likelihood_paras), scale=1.0,
                                Ndof=Ndof, sseB=np.array([0.0]), sseSEM=np.array([0.0]), data_uncertainty=str.encode(data_uncertainty[i]))/self.nreplicas
                        #loglik /=self.nreplicas
                else:
                   loglik += get_negloglikelihood(nreplicas=self.nreplicas, model=str.encode(stat_model[0]),
                           sse=sse, sigmaSEM=sigma_SEM, sigmaB=np.array(max_likelihood_paras), scale=1.0,
                           Ndof=Ndof, sseB=np.array([0.0]), sseSEM=np.array([0.0]), data_uncertainty=str.encode(data_uncertainty[i]))

            #print(loglik)
            BIC = bic(llf=-loglik, nobs=nobs, df_modelwc=nParameters)
            #BIC = bic(llf=, nobs=nobs, df_modelwc=nParameters)

            result = {"model": model, "nObs": nobs, "nParameters":nParameters, "loglik":loglik, "SSE":sse, "BIC":BIC}
            results.append(result)


        BICs = []
        for model in range(len(self.lam)):
            nobs = results[0]["nObs"]
            k0,k1 = results[0]["nParameters"],results[model]["nParameters"]
            loglik0,loglik = results[0]["loglik"],results[model]["loglik"]
            SSE0,SSE1 = results[0]["SSE"],results[model]["SSE"]
            #ΔBIC = nobs*np.log(np.mean(SSE1)/np.mean(SSE0)) + np.log(nobs) * (k1-k0)
            ΔBIC = nobs*np.log(np.sum(SSE1)/np.sum(SSE0)) + np.log(nobs) * (k1-k0)
            #print(ΔBIC)
#            #ΔBIC = np.log(np.mean(SSE1)/np.mean(SSE0)) + np.log(nobs) * (k1-k0)
#            print(0.5*ΔBIC)
#            ΔBIC = loglik - loglik0# + np.log(nobs) * (k1-k0)
#            print(0.5*ΔBIC)
#            print("\n")


            #BIC0,BIC = results[0]["BIC"],results[model]["BIC"]
            #ΔBIC = BIC - BIC0
#            #print(ΔBIC)
            result = {"model": model, "nObs": nobs,
                    "k":k1, "k0":k0, "loglik0":loglik0, "loglik":loglik,
#                    "BIC0":BIC0, "BIC":BIC,
                    "SSE":SSE1, "SSE0":SSE0,
                    #"ΔBIC": ΔBIC, "BIC score": -np.log(np.exp(-0.5*ΔBIC))}
                    "ΔBIC": ΔBIC, "BIC score": 0.5*ΔBIC}
            if verbose: print(result)
            #print(result)
            BICs.append(result)
        return BICs


    def get_number_of_parameters(self):
        """
        """

        scheme = []
        df1 = self.get_traces(traj_index=-1)
        for k in range(len(self.scheme)):
            df1_array = df1["%s"%(df1.columns.to_list()[k])].to_numpy()
            if all(df1_array == np.ones(df1_array.shape)):  continue
            if all(df1_array == np.zeros(df1_array.shape)):  continue
            else: scheme.append(self.scheme[k])

        # Using states and sigmas as parameters
        #self.nParameters = self.nstates + len(scheme)

        # NOTE: states should be considered as only a single parameter
        # and each sigma should be considered as a parameter.
        self.nParameters = 1. + len(scheme)
        return self.nParameters


    def get_number_of_observations(self):
        """Is the number of observables the number of model data points?
        Or is the the number of experimental data points?
        """
        # Number of observations (nobs) is the number of data points
        nobs = 0
        for i in range(len(self.model)):
            for j in range(len(self.model[i])):
                nobs += len(self.model[i][j])
        return nobs


    def get_state_counts(self, trace):
        """Determine which states were sampled and the states with zero counts.

        Args:
            trace(np.ndarray): trajectory trace
            nstates(int): number of states
        """

        all_states = np.zeros((len(trace), self.nstates))
        for i in range(len(trace)):
            for state in trace[i]:
                all_states[i,state] += 1
        return all_states

    def plot_population_evolution(self, figsize=None,
                        label_fontsize=12, legend_fontsize=10):

        for i in range(len(self.expanded_values)):
            fig = plt.figure(figsize=(12,8))
            gs = gridspec.GridSpec(1, 1)
            ax1 = plt.subplot(gs[0,0])
            traj_steps = np.array(self.traj[i]['trajectory']).T[0]
            state_trace = np.array(self.traj[i]["trajectory"]).T[3]
            state_counts = self.get_state_counts(state_trace).T
            state_populations = state_counts/state_counts.T.sum(axis=1)
            #c = ax1.pcolor(state_counts, cmap='gray_r') #cmap='RdBu')
            c = ax1.pcolor(state_populations, cmap='gray_r') #cmap='RdBu')
            xticks = np.array(ax1.get_xticks())
            xticks = np.linspace(0, traj_steps[-1], len(xticks), dtype=int)
            ax1.set_xticklabels([f"{tick}" for tick in xticks])
            ax1.set_ylabel("State index", fontsize=16)
            ax1.set_xlabel("Number of Steps", fontsize=16)
            fig.colorbar(c, ax=ax1)
            fig.tight_layout()
            fig.savefig(os.path.join(self.resultdir,f"populations_lam_{self.expanded_values[i]}.png"))

    def plot_acceptance_trace(self, figsize=(8,4), label_fontsize=12, legend_fontsize=10):

        for i in range(len(self.expanded_values)):
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 1)
            ax1 = plt.subplot(gs[0,0])
            traj_steps = np.array(self.traj[i]['trajectory'], dtype=object).T[0]
            acceptance = np.array(self.traj[i]['trajectory'], dtype=object).T[6]
            c = ax1.plot(traj_steps, acceptance, color='k')
            ax1.set_ylabel("MCMC Acceptance ", fontsize=16)
            ax1.set_xlabel("Number of Steps", fontsize=16)
            ax1.tick_params(which="major", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax1.tick_params(which="minor", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.grid()
            ticks = [ax1.xaxis.get_minor_ticks(),
                     ax1.xaxis.get_major_ticks()]
            marks = [ax1.get_xticklabels(),
                    ax1.get_yticklabels()]
            for k in range(0,len(ticks)):
                for tick in ticks[k]:
                    tick.label.set_fontsize(label_fontsize)
            for k in range(0,len(marks)):
                for mark in marks[k]:
                    mark.set_size(fontsize=label_fontsize-2)

            fig.tight_layout()
            fig.savefig(os.path.join(self.resultdir,f"mcmc_acceptance_lam_{self.expanded_values[i]}.png"))

    def approximate_scores(self, burn=1000, maxtau=500, ref_values=None, verbose=False):
        if ref_values:
            ref_lam, ref_xi = ref_values
        else:
            ref_lam,ref_xi=self.expanded_values[0][0],self.expanded_values[0][1]
        results = []
        for i in range(len(self.expanded_values)):
            x = np.array(self.traj[i]['trajectory'], dtype=object).T[0]
            energy = np.array(self.traj[i]['trajectory'], dtype=object).T[1]
            energy_dist = energy.copy()
            minima = np.array(energy_dist.min())
            #minima = 0.0
            # NOTE: burn-in a certain number of samples
#            _energy_dist = energy_dist[burn:]
            _energy_dist = energy_dist[burn:] - minima
            try:
                tau_c = get_autocorrelation_time([_energy_dist], method="auto", maxtau=maxtau)
                tau_c = float(tau_c)
                if np.isnan(tau_c):
                    tau_c = 0
                elif math.isnan(tau_c):
                    tau_c = 0
                else:
                    tau_c = int(round(tau_c))
            except(Exception) as e:
                print(e)
                tau_c = 0
            if tau_c < 0.0:
                print("\n\n\n\n Negative autocorrelation_time! \n\n\n")
                exit()
            if verbose: print("tau_c: ",tau_c)
            #value = exp_average(_energy_dist[tau_c:])
            value = minima -np.log(exp_average(_energy_dist[tau_c:]))
            results.append({"lambda": self.expanded_values[i][0], "xi": self.expanded_values[i][1], "exp_avg": value})
        df = pd.DataFrame(results)
        #Z0 = df.iloc[np.where(df["lambda"] == 0.0)[0]]["exp_avg"].to_numpy()
        Z0 = df.iloc[np.where((df["lambda"] == ref_lam) & (df["xi"] == ref_xi))[0]]["exp_avg"].to_numpy()
        scores = []
        for i in range(len(self.expanded_values)):
            Z = df.iloc[[i]]["exp_avg"].to_numpy()
            #scores.append(float(-np.log(Z/Z0)))
            try:
                scores.append(float((Z - Z0)))
            except(Exception) as e:
                scores.append(np.nan)
        df["approx_score"] = np.array(scores)/self.nreplicas
        return df




    def plot_energy_trace(self, figsize=(8,4), label_fontsize=12, legend_fontsize=10, bins="auto", show=False):

        figures = []
        steps,dists = [],[]
        for i in range(len(self.expanded_values)):
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=(4, 1))
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
            traj_steps = np.array(self.traj[i]['trajectory'], dtype=object).T[0]
            energy = np.array(self.traj[i]['trajectory'], dtype=object).T[1]
            c = ax1.plot(traj_steps, energy/self.nreplicas, color='k')
            #ax2.hist(energy, bins=bins, alpha=0.5, edgecolor="k", orientation='horizontal')
            ax2.hist(energy/self.nreplicas, bins=bins, alpha=0.5, edgecolor="k", orientation='horizontal')
            ax1.set_ylabel("Energy (kT)", fontsize=16)
            ax1.set_xlabel("Number of Steps", fontsize=16)
            ax1.tick_params(which="major", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax1.tick_params(which="minor", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.grid()
            ticks = [ax1.xaxis.get_minor_ticks(),
                     ax1.xaxis.get_major_ticks()]
            marks = [ax1.get_xticklabels(),
                    ax1.get_yticklabels()]
            for k in range(0,len(ticks)):
                for tick in ticks[k]:
                    tick.label.set_fontsize(label_fontsize)
            for k in range(0,len(marks)):
                for mark in marks[k]:
                    mark.set_size(fontsize=label_fontsize-2)

            fig.tight_layout()
            fig.savefig(os.path.join(self.resultdir,f"mcmc_energy_{self.expanded_values[i]}__{i}.png"))
            figures.append(fig)
            dists.append(energy)
            steps.append(traj_steps)
        if not show: plt.close('all')
        return figures,steps,dists



    def get_restraint_intensity_trace(self, model_index=-1):
        return np.array(np.array(self.traj[model_index]['trajectory']).T[6], dtype=float)

    def get_entropy_trace(self, model_index=-1):
        return np.array(self.traj[model_index]['trajectory']).T[7]

    def get_chi2_trace(self, model_index=-1):
        return np.array(self.traj[model_index]['trajectory']).T[8]

    def plot_convergence(self, figsize=None, label_fontsize=12, legend_fontsize=10):

        for i in range(len(self.expanded_values)):
            fig = plt.figure(figsize=(12,14))
            gs = gridspec.GridSpec(3, 2)
            ax1,ax2,ax3 = plt.subplot(gs[0,0]),plt.subplot(gs[1,0]),plt.subplot(gs[2,0])
            ax4,ax5,ax6 = plt.subplot(gs[0,1]),plt.subplot(gs[1,1]),plt.subplot(gs[2,1])

            traj_steps = np.array(self.traj[i]['trajectory'], dtype=object).T[0]
            traj_states = np.array(self.traj[i]['trajectory'], dtype=object).T[3]
            Nr_array = np.array([len(states) for states in traj_states], dtype=float)
            #Nr_array = np.array(self.Nr_array[i], dtype=float)
            restraint_intensity_trace = np.array(np.array(self.traj[i]['trajectory'], dtype=object).T[6], dtype=float)
            fitting = np.poly1d(np.polyfit(Nr_array**2, restraint_intensity_trace, 1))
            R2 = np.corrcoef(restraint_intensity_trace, fitting(Nr_array**2))[0][1]**2
            entropy_trace = np.array(self.traj[i]['trajectory'], dtype=object).T[7]
            chi2_trace = np.array(self.traj[i]['trajectory'], dtype=object).T[8]
            dchi2_trace = np.array(self.traj[i]['trajectory'], dtype=object).T[9]
            #state_counts = get_state_counts(state_trace, nStates).T

            #c = ax1.pcolor(state_counts, cmap='gray_r') #cmap='RdBu')
            ax1.plot(traj_steps, Nr_array)
            ax1.set_ylabel("$N_{r}$", fontsize=16)
            ax1.set_xlabel("Number of Steps", fontsize=16)
            ax2.plot(traj_steps, entropy_trace)
            ax2.set_ylabel("$S(p_{1}\mid p_{0})$", fontsize=16)
            ax2.set_xlabel("Number of Steps", fontsize=16)

            ax3.plot(traj_steps, chi2_trace)
            ax3.set_ylabel("$\chi^{2}$", fontsize=16)
            ax3.set_xlabel("Number of Steps", fontsize=16)
            ax4.plot(Nr_array**2, restraint_intensity_trace)
            ax4.set_ylabel(r"k ($k_{b}$T)", fontsize=16)
            ax4.set_xlabel("$N_{r}^{2}$", fontsize=16)
            ax4.annotate(r'$R^{2}$ = %0.6g'%R2, #(np.mean(Rs)),
                        xy=(8, restraint_intensity_trace.max()*0.9),
                        xycoords='data', fontsize=16)
            ax5.plot(traj_steps, dchi2_trace)
            ax5.set_ylabel(r"$\partial \chi^{2}$", fontsize=16)
            ax5.set_xlabel("Number of Steps", fontsize=16)
            ax6.plot(traj_steps, 0.5*chi2_trace+entropy_trace)
            ax6.set_ylabel(r"$\frac{\chi^{2}}{2} + S(p_{1}\mid p_{0})$", fontsize=16)
            ax6.set_xlabel("Number of Steps", fontsize=16)
            fig.tight_layout()
            fig.savefig(os.path.join(self.resultdir,f"optimize_replicas_entropy_trace_{self.expanded_values[i]}__{i}.png"))


    def get_counts_and_bins_for_continuous_space(self, model=0, return_all=False):
        result = {}
        indices = []
        df = self.get_traces(traj_index=model)
        for k in range(len(self.scheme)):
            col = df.columns.to_list()[k]
            df_array = df["%s"%(col)].to_numpy()
            if return_all == False:
                if all(df_array == np.ones(df_array.shape)): continue
                if all(df_array == np.zeros(df_array.shape)): continue
            counts, bins = np.histogram(df_array, bins="auto")
            result[col] = np.array([counts, bins])
        return result


    def plot(self, plottype="step", figname="BICePs.pdf", figsize=None,
            label_fontsize=12, legend_fontsize=10, pad=0.25, wspace=0.20, hspace=0.75,
             ref=None, model=-1, plot_all_distributions=False):
        """Plot figures for population and sampled nuisance parameters.

        Args:
        """

        if ref == None: ref = self.expanded_values.index((0.0, 1.0))

        df0 = self.get_traces(traj_index=ref)
        df1 = self.get_traces(traj_index=model)
        if self.continuous_space:
            hists0 = self.get_counts_and_bins_for_continuous_space(model=ref)
            hists1 = self.get_counts_and_bins_for_continuous_space(model=model)

        indices = []
        for k in range(len(self.scheme)):
            df0_array = df0["%s"%(df0.columns.to_list()[k])].to_numpy()
            if all(df0_array == np.ones(df0_array.shape)): continue
            if all(df0_array == np.zeros(df0_array.shape)): continue
            indices.append(k)

        cap = 20
        if plot_all_distributions: N = len(indices)
        else: N = 20

        if len(indices) < N: N = len(indices)

        # load in precomputed P and dP from MBAR analysis
        pops0, pops1   = self.P_dP[:,ref], self.P_dP[:,self.K-1]
        dpops0, dpops1 = self.P_dP[:,self.K], self.P_dP[:,2*self.K-1]
        t0 = self.traj[ref]
        t1 = self.traj[self.K-1]

        # Figure Plot SETTINGS
        fontfamily={'family':'sans-serif','sans-serif':['Arial']}
        plt.rc('font', **fontfamily)

        # determine number of row and column
        if N >= cap:
            if (cap+1)%2 != 0:
                c,r = 2, round((cap+2)/2)
            else:
                c,r = 2, round((cap+1)/2)
        else:
            c,r = 2, math.ceil((N+2)/2)

        if figsize:
            figsize = figsize
        else:
            figsize = (4*c,5*r)

        fig = plt.figure( figsize=figsize )
        # Make a subplot in the upper left
        plt.subplot(int(r),int(c),1)
        plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')

        #plt.hold(True)
        limit = 1e-6
        plt.plot([limit, 1], [limit, 1], color='k', linestyle='-', linewidth=1)
        plt.xlim(limit, 1.)
        plt.ylim(limit, 1.)
        plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
        plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)

        #label key states
        for i in range(len(pops1)):
            if (i==0) or (pops1[i] > 0.05):
                plt.text( pops0[i], pops1[i], str(i), color='g' , fontsize=legend_fontsize)

        ntop = int(int(self.nstates)/10.)
        if self.verbose:
            topN = pops1[np.argsort(pops1)[-ntop:]]
            topN_labels = [np.where(topN[i]==pops1)[0][0] for i in range(len(topN))]
            print(f"Top {ntop} states: {topN_labels}")
            print(f"Top {ntop} populations: {topN}")
        axs = []
        ax = plt.gca()
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.5, linewidth=0.5)
        axs.append(ax)
        nplots = math.ceil(N/20)
        print("nplots = ",nplots)
        Idx = -1
        for nfig in range(nplots):
            if nplots > 1:
                figname = figname.replace(".",f"{nfig}.")
            if nfig > 0:
                fig.clear()
                axs = []
            s = 0
            if nfig > 0: pos = 0
            else: pos = 1
            for k in range(len(self.scheme)):
                if k <= Idx: continue
                Idx += 1
                col = df0.columns.to_list()[k]
                df0_array = df0["%s"%(col)].to_numpy()
                if all(df0_array == np.ones(df0_array.shape)): continue
                if all(df0_array == np.zeros(df0_array.shape)): continue
                df1_array = df1["%s"%(df1.columns.to_list()[k])].to_numpy()
                if all(df1_array == np.ones(len(df1_array))): continue
                if all(df1_array == np.zeros(len(df1_array))): continue
                s += 2
                plt.subplot(r,c,pos+1)
                pos += 1
                ax = plt.gca()
                axs.append(ax)

                if plottype == "step":
                    if self.continuous_space:
                        counts, bins = hists0[col]
                        plt.step(bins[:-1], counts, 'b-', label='exp')
                        plt.fill_between(bins[:-1], counts, color='b', step="pre", alpha=0.4, label=None)
                        if "sigma" in df1.columns.to_list()[k]:
                            ax.set_xlim(left=0, right=df0["%s"%(df1.columns.to_list()[k])].max()*1.1)
                        else:
                            ax.set_xlim(left=df0["%s"%(df1.columns.to_list()[k])].min(), right=df0["%s"%(df1.columns.to_list()[k])].max()*1.1)

                    else:
                        plt.step(t0['allowed_parameters'][k], t0['sampled_parameters'][k], 'b-', label='exp')
                        plt.fill_between(t0['allowed_parameters'][k], t0['sampled_parameters'][k], color='b', step="pre", alpha=0.4, label=None)
                if plottype == "hist":
                    df0["%s"%(df0.columns.to_list()[k])].hist(bins='auto', facecolor='b', alpha=0.5, edgecolor="k", ax=ax, label='exp')

                xmax0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][-1]
                xmin0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][0]
                xmax1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][-1]
                xmin1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][0]
                xmax = max(xmax0,xmax1)
                xmin = min(xmin0, xmin1)
                plt.xlim(t0['allowed_parameters'][k][xmin], t0['allowed_parameters'][k][xmax])
                if plottype == "step":
                    if self.continuous_space:
                        counts, bins = hists1[col]
                        plt.step(bins[:-1], counts, 'r-', label='sim+exp')
                        plt.fill_between(bins[:-1], counts, color='r', step="pre", alpha=0.4, label=None)
                        if "sigma" in df1.columns.to_list()[k]:
                            ax.set_xlim(left=0, right=df1["%s"%(df1.columns.to_list()[k])].max()*1.1)
                        else:
                            ax.set_xlim(left=df1["%s"%(df1.columns.to_list()[k])].min(), right=df1["%s"%(df1.columns.to_list()[k])].max()*1.1)

                    else:
                        plt.step(t1['allowed_parameters'][k], t1['sampled_parameters'][k], 'r-', label='sim+exp')
                        plt.fill_between(t1['allowed_parameters'][k], t1['sampled_parameters'][k], color='r', step="pre", alpha=0.4, label=None)
                if plottype == "hist":
                    df1["%s"%(df1.columns.to_list()[k])].hist(bins='auto', facecolor='r', alpha=0.5, edgecolor="k", ax=ax, label='sim+exp')
                    if "sigma" in df1.columns.to_list()[k]:
                        ax.set_xlim(left=0, right=df1["%s"%(df1.columns.to_list()[k])].max()*1.1)
                    else:
                        ax.set_xlim(left=df1["%s"%(df1.columns.to_list()[k])].min(), right=df1["%s"%(df1.columns.to_list()[k])].max()*1.1)

                # NOTE: https://e2eml.school/matplotlib_ticks.html
                ax.tick_params(which="major", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
                ax.tick_params(which="minor", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
                ax.xaxis.set_minor_locator(AutoMinorLocator())

                #plt.legend(['exp', 'sim+exp'], loc='best',fontsize=legend_fontsize)
                label = format_label(self.scheme[k])
                plt.xlabel(r"%s"%label, fontsize=label_fontsize)
                plt.ylabel(r"$P$(%s)"%label, fontsize=label_fontsize)
                plt.yticks([])

                if pos == (cap+1): break

            for n, ax in enumerate(axs):
                if (nfig == 0) and (n == 1):
                    ax.legend(loc='best',fontsize=legend_fontsize)
                if nplots > 1:
                    ax.text(-0.12, 1.02, string.ascii_lowercase[n]+f"{nfig}",
                            transform=ax.transAxes,size=20, weight='bold')
                    if n == 0: ax.legend(loc='best',fontsize=legend_fontsize)
                else:
                    ax.text(-0.12, 1.02, string.ascii_lowercase[n],
                            transform=ax.transAxes,size=20, weight='bold')
                # Setting the ticks and tick marks
                ticks = [ax.xaxis.get_minor_ticks(),
                         ax.xaxis.get_major_ticks()]
                marks = [ax.get_xticklabels(),
                        ax.get_yticklabels()]
                for k in range(0,len(ticks)):
                    for tick in ticks[k]:
                        tick.label.set_fontsize(label_fontsize)
                for k in range(0,len(marks)):
                    for mark in marks[k]:
                        mark.set_size(fontsize=label_fontsize-2)
                        #mark.set_rotation(s=65)

            fig.tight_layout(pad=pad)#, w_pad=0.5, h_pad=2.0)
            fig.subplots_adjust(left=0.075, bottom=0.075, right=0.99, top=0.95,
                                wspace=wspace, hspace=hspace)
            fig.savefig(os.path.join(self.resultdir,figname))
        return fig









