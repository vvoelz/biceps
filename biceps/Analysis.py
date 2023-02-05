# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pickle
from pymbar import MBAR
from pymbar.utils import kln_to_kn
from .Restraint import *
from .PosteriorSampler import *
from .toolbox import get_files
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import string


def find_all_state_sampled_time(trace, nstates, verbose=True):
    """Determine which states were sampled and the states with zero counts.

    Args:
        trace(np.ndarray): trajectory trace
        nstates(int): number of states
    """

    frac = []
    all_states = np.zeros(nstates)
    init = 0
    while 0 in all_states:
        if init == len(trace):
            if verbose: print('not all state sampled, these states', np.where(all_states == 0)[0],'are not sampled')
            return 'null', frac
        else:
            all_states[trace[init]] += 1
            frac.append(float(len(np.where(all_states!=0)[0]))/float(nstates))
            init += 1
    return init, frac


class Analysis(object):
    def __init__(self, outdir, nstates=0, precheck=True, BSdir='BS.dat',
            popdir='populations.dat', picfile='BICePs.pdf', verbose=False):
        """A class to perform analysis and plot figures.

        Args:
            nstates(int): number of conformational states
            trajs(str): relative path to glob '*.npz' trajectories (analysis files and figures will be placed inside this directory)
            precheck(bool): find the all the states that haven't been sampled if any
            BSdir(str): relative path for BICePs score file name
            popdir(str): relative path for BICePs reweighted populations file name
            picfile(str): relative path for BICePs figure
        """

        self.verbose = verbose
        self.resultdir = outdir
        self.trajs = os.path.join(outdir,"*.npz")
        self.BSdir = os.path.join(self.resultdir,BSdir)
        self.popdir = os.path.join(self.resultdir,popdir)
        self.picfile = os.path.join(self.resultdir,picfile)
        self.scheme = None
        self.traj = []
        self.sampler = []
        self.lam = None
        self.f_df = None
        self.P_dp = None
        self.precheck = precheck
        self.nstates = nstates
        if self.nstates == 0:
            raise ValueError("State number cannot be zero.")

        # next get MABR sampling done
        self.MBAR_analysis()



    def load_data(self):
        """Load input data from BICePs sampling (*npz and *pkl files)."""

        # Load in npz trajectories
        files = get_files(self.trajs)
        for file in files:
            if self.verbose: print('Loading %s ...'%file)
            traj = np.load(file, allow_pickle=True)['arr_0'].item()
            self.traj.append(traj)
        self.nreplicas = len(traj['trajectory'][0][3])
        if self.precheck:
            steps = []
            fractions = []
            for i in range(len(self.traj)):
                s,f = find_all_state_sampled_time(self.traj[i]['state_trace'],self.nstates)
                steps.append(s)
                fractions.append(f)
            total_fractions = np.concatenate(fractions)
            if 1. in total_fractions:
                plt.figure()
                for i in range(len(fractions)):
                    plt.plot(list(range(len(fractions[i]))),fractions[i],label = r'$\lambda_{%s}$'%i)
                    plt.xlabel('steps')
                    plt.ylabel('fractions')
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(self.resultdir,'fractions.pdf'))
            #else:
            #    print('Error: Not all states are sampled in any of the lambda values')
            #    exit()

        # Load in cpickled sampler objects
        #sampler_files = get_files(self.trajs.replace('.npz','.pkl'))
        sampler_files = [file.replace('.npz','.pkl') for file in files]
        for pkl_filename in sampler_files:
            if self.verbose: print('Loading %s ...'%pkl_filename)
            pkl_file = open(pkl_filename, 'rb')
            self.sampler.append( pickle.load(pkl_file) )

        # parse the lambda* filenames to get the full list of lambdas
        self.nlambda = len(files)
        self.lam = [float( (s.split('lambda')[1]).replace('.npz','') ) for s in files ]
        if self.verbose: print('lam =', self.lam)
        self.scheme = self.traj[0]['rest_type']


    def get_max_likelihood_parameters(self, model=0, sigma_only=False):
        #nParameters = self.nstates + len(self.scheme)
        if sigma_only:
            indices = [i for i,rest in enumerate(self.scheme) if "sigma" in rest]
        else:
            indices = [i for i,rest in enumerate(self.scheme)]

        t1 = self.traj[model]
        max_likelihood = {}
        for k in indices:
            x, y = t1['allowed_parameters'][k], t1['sampled_parameters'][k]
            max_likelihood[self.scheme[k]] = [x[np.argmax(y)]]
        max_likelihood = pd.DataFrame(max_likelihood)
        return max_likelihood




    def MBAR_analysis(self, debug=False):
        """MBAR analysis for populations and BICePs score"""

        # load necessary data first
        self.load_data()

        # Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
        #   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
        self.K = self.nlambda   # number of thermodynamic ensembles
        # N_k[k] will denote the number of correlated snapshots from state k
        N_k = np.array( [len(self.traj[i]['trajectory']) for i in range(self.nlambda)] )
        nsnaps = N_k.max()
        u_kln = np.zeros( (self.K, self.K, nsnaps) )
        nstates = self.nstates
        if self.verbose: print('nstates', nstates)

        states_kn = np.zeros( (self.K, nsnaps, self.nreplicas) )

        # special treatment for neglogP function
        temp_parameters_indices = self.traj[0]['trajectory'][0][4:][0]
        #print temp_parameters_indices
        original_index =[]   # keep tracking the original index of the parameters
        for ind in range(len(temp_parameters_indices)):
            for in_ind in temp_parameters_indices[ind]:
                original_index.append(ind)

        # Get snapshot energies rescored in the different ensembles
        """['step', 'E', 'accept', 'state', [nuisance parameters]]"""
        for n in range(nsnaps):
            for k in range(self.K):
                for l in range(self.K):
                    if debug: print('step', self.traj[k]['trajectory'][n][0], end=' ')
                    if k==l:
                        u_kln[k,k,n] = self.traj[k]['trajectory'][n][1]
                    else:
                        state, sigma_index = self.traj[k]['trajectory'][n][3:]
                        for r in range(self.nreplicas):
                            states_kn[k,n,r] = state[r]
                        temp_parameters = []
                        new_parameters=[[] for i in range(len(temp_parameters_indices))]
                        temp_parameter_indices = np.concatenate(sigma_index)
                        for ind in range(len(temp_parameter_indices)):
                            temp_parameters.append(self.traj[k]['allowed_parameters'][ind][temp_parameter_indices[ind]])
                        for m in range(len(original_index)):
                            new_parameters[original_index[m]].append(temp_parameters[m])
                        u_kln[k,l,n] = self.sampler[l].neglogP(state, new_parameters, sigma_index)
                        if debug: print('E_%d evaluated in model_%d'%(k,l), u_kln[k,l,n])
#
        self.u_kln, self.N_k, self.states_kn = u_kln, N_k, states_kn
        stime = time.time()
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
        #_results = mbar.getFreeEnergyDifferences(uncertainty_method='approximate', return_theta=True, return_dict=True)
        _results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
        #print(_results.keys())
        Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
        self.Deltaf_ij = Deltaf_ij
        self.dDeltaf_ij = dDeltaf_ij
        beta = 1.0 # keep in units kT
        #print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])
        self.f_df = np.zeros( (self.nlambda, 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
        self.f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
        self.f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std

        self.P_dP = np.zeros( (self.nstates, 2*self.K) )  # left columns are P, right columns are dP
        if debug: print('state\tP\tdP')
        self.u_kn = kln_to_kn(self.u_kln, N_k=self.N_k)
        self.compute_perturbed_free_energies = mbar.compute_perturbed_free_energies
        self.nreplicas = len(self.states_kn[-1,-1])
        for i in range(self.nstates):
            sampled = np.array([np.where(self.states_kn[:,:,r]==i,1,0) for r in range(self.nreplicas)])
            A_kn = sampled.sum(axis=0)/self.nreplicas
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


    def get_traces(self, traj_index=-1):
        npz = self.traj[traj_index]
        traj = npz["trajectory"]
        rest_type = []
        n = 0
        rests = np.array(npz["rest_type"])
        for g in np.array(traj[0][4]):
            rest_type.append(rests[n:n+len(g)].tolist())
            n += len(g)
            if n >= len(np.array(npz["rest_type"])): break
        rest_type = np.array(rest_type)
        # Find the unique restraints
        unique = []
        r = ""
        for restraint in rests:
            if restraint not in unique:
                if restraint != r:
                    r = restraint
                    unique.append(restraint)

        n_rests_dict = {f"{r}":0 for r in unique}
        columns = []
        for k, rest in enumerate(npz['rest_type']):
            columns.append(str(rest)+f"{n_rests_dict[rest]}")
            n_rests_dict[rests[k]] += 1
        df = pd.DataFrame(np.array(npz["traces"]).transpose(), columns)
        df = df.transpose()
        return df




    def plot(self, plottype="hist", figname="BICePs.pdf", figsize=None,
            label_fontsize=12, legend_fontsize=10):
        """Plot figures for population and sampled nuisance parameters.

        Args:
            show(bool): show the plot in Jupyter Notebook.
        """

        df0 = self.get_traces(traj_index=0)
        df1 = self.get_traces(traj_index=-1)
        N = 20
        if df0.shape[1] > N:
            if self.verbose:
                print(f"Number of posterior distributions of \
nuisance parameters: {df0.shape[1]}\n\
Too many distributions for a single figure... only plotting the first {N}.")
            df0 = df0[df0.columns.to_list()[:N]]
            df1 = df1[df1.columns.to_list()[:N]]
        else:
            N = len(df0.columns.to_list())

        #df0 = self.get_sigmaB_trace(traj_index=0)
        #df1 = self.get_sigmaB_trace(traj_index=-1)

        ## next get MABR sampling done
        #self.MBAR_analysis()

        # load in precomputed P and dP from MBAR analysis
        pops0, pops1   = self.P_dP[:,0], self.P_dP[:,self.K-1]
        dpops0, dpops1 = self.P_dP[:,self.K], self.P_dP[:,2*self.K-1]
        t0 = self.traj[0]
        t1 = self.traj[self.K-1]

        # Figure Plot SETTINGS
        #label_fontsize = 12
        #legend_fontsize = 10
        fontfamily={'family':'sans-serif','sans-serif':['Arial']}
        plt.rc('font', **fontfamily)

        # determine number of row and column
        if (len(self.scheme[:N])+1)%2 != 0:
            c,r = 2, (len(self.scheme[:N])+2)/2
        else:
            c,r = 2, (len(self.scheme[:N])+1)/2
        if figsize:
            fig = plt.figure( figsize=figsize )
        else:
            fig = plt.figure( figsize=(4*c,5*r) )
        # Make a subplot in the upper left
        plt.subplot(int(r),int(c),1)
        plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')

        #plt.hold(True)
        limit = 1e-6
        plt.plot([limit, 1], [limit, 1], color='k', linestyle='-', linewidth=2)
        plt.xlim(limit, 1.)
        plt.ylim(limit, 1.)

        plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
        plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
        plt.xscale('log')
        plt.yscale('log')

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
        axs.append(ax)

        s = 0
        for k in range(len(self.scheme[:N])):
            df0_array = df0["%s"%(df0.columns.to_list()[k])].to_numpy()
            if all(df0_array == np.ones(df0_array.shape)): continue
            df1_array = df1["%s"%(df1.columns.to_list()[k])].to_numpy()
            if all(df1_array == np.ones(len(df1_array))): continue
            s += 2
            #plt.subplot(r,c,s)
            plt.subplot(r,c,k+2)
            ax = plt.gca()
            axs.append(ax)
            if plottype == "step":
                plt.step(t0['allowed_parameters'][k], t0['sampled_parameters'][k], 'b-', label='exp')
                plt.fill_between(t0['allowed_parameters'][k], t0['sampled_parameters'][k], color='b', step="pre", alpha=0.4, label=None)
            if plottype == "hist":
                #counts, bins = np.histogram(t0['sampled_parameters'][k])
                #width = 0.1
                #plt.bar(x=t0['allowed_parameters'][k], height=t0['sampled_parameters'][k],
                #        width=width,
                #        facecolor='b', alpha=0.5, edgecolor="k", linewidth=1.2)
                df0["%s"%(df0.columns.to_list()[k])].hist(bins='auto', facecolor='b', alpha=0.5, edgecolor="k", ax=ax, label='exp')

            xmax0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][-1]
            xmin0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][0]
            xmax1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][-1]
            xmin1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][0]
            xmax = max(xmax0,xmax1)
            xmin = min(xmin0, xmin1)
            plt.xlim(t0['allowed_parameters'][k][xmin], t0['allowed_parameters'][k][xmax])
            if plottype == "step":
                plt.step(t1['allowed_parameters'][k], t1['sampled_parameters'][k], 'r-', label='sim+exp')
                plt.fill_between(t1['allowed_parameters'][k], t1['sampled_parameters'][k], color='r', step="pre", alpha=0.4, label=None)
            if plottype == "hist":
                #counts, bins = np.histogram(t1['sampled_parameters'][k])
                #plt.bar(x=t1['allowed_parameters'][k], height=t1['sampled_parameters'][k],
                #        width=width,
                #        facecolor='r', alpha=0.5, edgecolor="k", linewidth=1.2)
                df1["%s"%(df1.columns.to_list()[k])].hist(bins='auto', facecolor='r', alpha=0.5, edgecolor="k", ax=ax, label='sim+exp')
                ax.set_xlim(left=0, right=df1["%s"%(df1.columns.to_list()[k])].max()*1.1)

            #plt.legend(['exp', 'sim+exp'], loc='best',fontsize=legend_fontsize)
            plt.legend(loc='best',fontsize=legend_fontsize)
            if self.scheme[k].count('_') == 0:
                if self.scheme[k] == 'gamma':
                    plt.xlabel("$\%s$"%self.scheme[k],fontsize=label_fontsize)
                    plt.ylabel("$P(\%s)$"%self.scheme[k],fontsize=label_fontsize)
                else:
                    plt.xlabel("$%s$"%self.scheme[k],fontsize=label_fontsize)
                    plt.ylabel("$P(%s)$"%self.scheme[k],fontsize=label_fontsize)
                plt.yticks([])

            elif self.scheme[k].count('_') == 1:
                if 'gamma' in self.scheme[k]:
                    plt.xlabel("${\%s_{%s}}^{-1/6}$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]),fontsize=label_fontsize)
                    plt.ylabel("${P(\%s_{%s}}^{-1/6})$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]), fontsize=label_fontsize)
                    plt.yticks([])
                else:
                    plt.xlabel("$\%s_{%s}$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]),fontsize=label_fontsize)
                    plt.ylabel("$P(\%s_{%s})$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]), fontsize=label_fontsize)
                    plt.yticks([])
            elif self.scheme[k].count('_') == 2:
                plt.xlabel("$\%s_{{%s}_{%s}}$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1],self.scheme[k].split('_')[2]),fontsize=label_fontsize)
                plt.ylabel("$P(\%s_{{%s}_{%s}})$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1],self.scheme[k].split('_')[2]),fontsize=label_fontsize)
                plt.yticks([])


        for n, ax in enumerate(axs):
            #ax.imshow(np.random.randn(10,10), interpolation='none')
            ax.text(-0.12, 1.02, string.ascii_lowercase[n], transform=ax.transAxes,
                    size=20, weight='bold')
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

        plt.tight_layout()
        plt.savefig(os.path.join(self.resultdir,figname))
        return fig




