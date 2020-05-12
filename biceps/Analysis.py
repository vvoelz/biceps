# -*- coding: utf-8 -*-
import os, glob
from .Restraint import *
from .PosteriorSampler import *
from . import toolbox as d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
from pymbar import MBAR

class Analysis(object):

    def __init__(self, outdir, nstates=0, precheck=True, BSdir='BS.dat',
            popdir='populations.dat', picfile='BICePs.pdf'):
        """A class to perform analysis and plot figures.

        :param int default=0 nstates: number of conformational states
        :param str outdir: output files directory
        :param str default='BS.dat' BSdir: output BICePs score file name
        :param str default='populations.dat' popdir: output BICePs reweighted populations file name
        :param str default='BICePs.pdf' picfile: output figure name
        """

        self.states = nstates
        self.resultdir = outdir
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
        if self.states == 0:
            raise ValueError("State number cannot be zero.")
        # next get MABR sampling done
        self.MBAR_analysis()


    def load_data(self, debug=True):
        """load input data from BICePs sampling (*npz and *pkl files)"""

        # Load in npz trajectories
        exp_files = glob.glob( os.path.join(self.resultdir,'traj_lambda*.npz') )
        exp_files.sort()
        for filename in exp_files:
            if debug:
                print('Loading %s ...'%filename)
            traj = np.load(filename, allow_pickle=True )['arr_0'].item()
            self.traj.append(traj)
        self.nreplicas = len(traj['trajectory'][0][3])
        if self.precheck:
            steps = []
            fractions = []
            for i in range(len(self.traj)):
                s,f = d.find_all_state_sampled_time(self.traj[i]['state_trace'],self.states)
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
        sampler_files = glob.glob( os.path.join(self.resultdir,'sampler_lambda*.pkl') )
        sampler_files.sort()
        for pkl_filename in sampler_files:
            if debug:
                print('Loading %s ...'%pkl_filename)
            pkl_file = open(pkl_filename, 'rb')
            self.sampler.append( pickle.load(pkl_file) )

        # parse the lambda* filenames to get the full list of lambdas
        self.nlambda = len(exp_files)
        self.lam = [float( (s.split('lambda')[1]).replace('.npz','') ) for s in exp_files ]
        if debug:
            print('lam =', self.lam)
        self.scheme = self.traj[0]['rest_type']


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
        nstates = int(self.states)
        print('nstates', nstates)
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
                    if debug:
                        print('step', self.traj[k]['trajectory'][n][0], end=' ')
                    if k==l:
                        u_kln[k,k,n] = self.traj[k]['trajectory'][n][1]
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
                    if debug:
                        print('E_%d evaluated in model_%d'%(k,l), u_kln[k,l,n])

        # Initialize MBAR with reduced energies u_kln and number of uncorrelated configurations from each state N_k.
        # u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where U_l(x) is the potential energy function for state l,
        # beta is the inverse temperature, and and x_kn denotes uncorrelated configuration n from state k.
        # N_k[k] is the number of configurations from state k stored in u_knm
        # Note that this step may take some time, as the relative dimensionless free energies f_k are determined at this point.
        mbar = MBAR(u_kln, N_k)

        # Extract dimensionless free energy differences and their statistical uncertainties.
#       (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
        #(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
        (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='approximate')
        #print 'Deltaf_ij', Deltaf_ij
        #print 'dDeltaf_ij', dDeltaf_ij
        beta = 1.0 # keep in units kT
        #print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])
        self.f_df = np.zeros( (self.nlambda, 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
        self.f_df[:,0] = Deltaf_ij[0,:]
        self.f_df[:,1] = dDeltaf_ij[0,:]

        # Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
        # Here, A_kn[k,n] = A(x_{kn})
        #(A_k, dA_k) = mbar.computeExpectations(A_kn)
        self.P_dP = np.zeros( (nstates, 2*self.K) )  # left columns are P, right columns are dP
        if debug:
            print('state\tP\tdP')
        states_kn = np.array(states_kn)
        for i in range(nstates):
            sampled = np.array([np.where(states_kn[:,:,r]==i,1,0) for r in range(self.nreplicas)])
            A_kn = sampled.sum(axis=0)
            (p_i, dp_i) = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
            self.P_dP[i,0:self.K] = p_i
            self.P_dP[i,self.K:2*self.K] = dp_i
        pops, dpops = self.P_dP[:, 0:self.K], self.P_dP[:, self.K:2*self.K]

        # save results
        self.save_MBAR()

    def save_MBAR(self):
        """save results (BICePs score and populations) from MBAR analysis"""

        print('Writing %s...'%self.BSdir)
        np.savetxt(self.BSdir, self.f_df)
        print('...Done.')

        print('Writing %s...'%self.popdir)
        np.savetxt(self.popdir, self.P_dP)
        print('...Done.')

    def plot(self, show=False, debug=False):
        """plot figures for population, nuisance parameters"""

        # first figure out what scheme is used
        #self.list_scheme()

        ## next get MABR sampling done
        #self.MBAR_analysis()

        # load in precomputed P and dP from MBAR analysis
        pops0, pops1   = self.P_dP[:,0], self.P_dP[:,self.K-1]
        dpops0, dpops1 = self.P_dP[:,self.K], self.P_dP[:,2*self.K-1]
        t0 = self.traj[0]
        t1 = self.traj[self.K-1]

        # Figure Plot SETTINGS
        label_fontsize = 12
        legend_fontsize = 10
        fontfamily={'family':'sans-serif','sans-serif':['Arial']}
        plt.rc('font', **fontfamily)

        # determine number of row and column
        if (len(self.scheme)+1)%2 != 0:
            c,r = 2, (len(self.scheme)+2)/2
        else:
            c,r = 2, (len(self.scheme)+1)/2
        plt.figure( figsize=(4*c,5*r) )
        # Make a subplot in the upper left
        plt.subplot(r,c,1)
        plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')

        #plt.hold(True)
        limit = 1e-6
        plt.plot([limit, 1], [limit, 1], color='k', linestyle='-', linewidth=2)
        plt.xlim(limit, 1.)
        plt.ylim(limit, 1.)

        #x0,x1 = np.min(pops0), np.max(pops0)
        #y0,y1 = np.min(pops1), np.max(pops1)
        #spacing = 0.0  # not sure yet...
        #low, high = np.min([x0, y0])-spacing, np.max([x1,y1])+spacing
        #plt.plot([low, high], [low, high], color='k', linestyle='-', linewidth=2)
        #plt.xlim(low, high)
        #plt.ylim(low, high)

        plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
        plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
        plt.xscale('log')
        plt.yscale('log')

        #label key states
        for i in range(len(pops1)):
            if (i==0) or (pops1[i] > 0.05):
                plt.text( pops0[i], pops1[i], str(i), color='g' )

        ntop = int(int(self.states)/10.)
        topN = pops1[np.argsort(pops1)[-ntop:]]
        topN_labels = [np.where(topN[i]==pops1)[0][0] for i in range(len(topN))]
        print(f"Top {ntop} states: {topN_labels}")
        print(f"Top {ntop} populations: {topN}")
        #for i in range(len(pops1)):
        #    if (i==0) or (pops1[i] in topN):
        #        plt.text( pops0[i], pops1[i], str(i), color='g' )
        for k in range(len(self.scheme)):
            plt.subplot(r,c,k+2)
            plt.step(t0['allowed_parameters'][k], t0['sampled_parameters'][k], 'b-')
            xmax0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][-1]
            xmin0 = [l for l,e in enumerate(t0['sampled_parameters'][k]) if e != 0.][0]
            xmax1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][-1]
            xmin1 = [l for l,e in enumerate(t1['sampled_parameters'][k]) if e != 0.][0]
            xmax = max(xmax0,xmax1)
            xmin = min(xmin0, xmin1)
            plt.xlim(t0['allowed_parameters'][k][xmin], t0['allowed_parameters'][k][xmax])
            plt.step(t1['allowed_parameters'][k], t1['sampled_parameters'][k], 'r-')
            plt.legend(['exp', 'sim+exp'], loc='best',fontsize=legend_fontsize)
            if self.scheme[k].count('_') == 0:
                if self.scheme[k] == 'gamma':
                    plt.xlabel("$\%s$"%self.scheme[k],fontsize=label_fontsize)
                    plt.ylabel("$P(\%s)$"%self.scheme[k],fontsize=label_fontsize)
                else:
                    plt.xlabel("$%s$"%self.scheme[k],fontsize=label_fontsize)
                    plt.ylabel("$P(%s)$"%self.scheme[k],fontsize=label_fontsize)
                plt.yticks([])

            elif self.scheme[k].count('_') == 1:
                plt.xlabel("$\%s_{%s}$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]),fontsize=label_fontsize)
                plt.ylabel("$P(\%s_{%s})$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1]), fontsize=label_fontsize)
                plt.yticks([])
            elif self.scheme[k].count('_') == 2:
                plt.xlabel("$\%s_{{%s}_{%s}}$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1],self.scheme[k].split('_')[2]),fontsize=label_fontsize)
                plt.ylabel("$P(\%s_{{%s}_{%s}})$"%(self.scheme[k].split('_')[0],self.scheme[k].split('_')[1],self.scheme[k].split('_')[2]),fontsize=label_fontsize)
                plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.picfile)
        if show:
            plt.show()




