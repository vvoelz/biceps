##############################################################################
# Authors: Yunhui Ge, Vincent Voelz
# Contributors: Rob Raddi
# This file is used to do posterior analysis using MBAR and plot figures. 
##############################################################################


##############################################################################
# Imports
##############################################################################

import sys, os, glob

sys.path.append('src')

from Restraint import *
from PosteriorSampler import *
import data as d
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR
import re

##############################################################################
# Code
##############################################################################

class Analysis(object):
    """A class to perform analysis and plot figures"""
    def __init__(self, states = 0, data = None, resultdir = None, BSdir = 'BS.dat', popdir = 'populations.dat', picfile = 'BICePs.pdf'):
	self.states = states
	self.data = data
	self.resultdir = resultdir
	self.BSdir = BSdir
	self.popdir = popdir
	self.picfile = picfile
	self.scheme = None
	self.traj = []
	self.sampler = []
	self.lam = None
	self.f_df = None
	self.P_dp = None
	if self.states == 0:
		raise ValueError("State number cannot be zero")
	if self.data == None:
		raise ValueError("Input data file is not specified")
	if self.resultdir == None:
		raise ValueError("Result directory is missing")

    def list_scheme(self):
	"""figure out what scheme is included in sampling"""
	input_data = d.sort_data(self.data)
	d_l=[]
	for i in input_data[0]:
	        if i.endswith('.noe'):
            	    d_l.append('sigma_noe')
	            d_l.append('gamma')
                elif i.endswith('.cs_H'):
                    d_l.append('sigma_cs_H')
                elif i.endswith('.cs_Ha'):
                    d_l.append('sigma_cs_Ha')
                elif i.endswith('.cs_N'):
                    d_l.append('sigma_cs_N')
                elif i.endswith('.cs_Ca'):
                    d_l.append('sigma_cs_Ca')
                elif i.endswith('.J'):
                    d_l.append('sigma_J')
		elif i.endswith('.pf'):
                    d_l.append('sigma_pf')
                else:
                    raise ValueError("Incompatible File extension. Use:{*.noe, *.J, *.cs_H, *.cs_Ha, *.cs_N, *.cs_Ca, *.pf}")
	self.scheme = d_l
	
    def load_data(self, debug = True):
	"""load input data from BICePs sampling (*yaml and *pkl files)"""
	# Load in yaml trajectories
	exp_files = glob.glob( os.path.join(self.resultdir,'traj_lambda*.yaml') )
	exp_files.sort()
	for filename in exp_files:
		if debug: 
   			print 'Loading %s ...'%filename
    		self.traj.append( yaml.load( file(filename, 'r') ) )

	# Load in cpickled sampler objects
	sampler_files = glob.glob( os.path.join(self.resultdir,'sampler_lambda*.pkl') )
	sampler_files.sort()
	for pkl_filename in sampler_files:
		if debug:
    			print 'Loading %s ...'%pkl_filename
    		pkl_file = open(pkl_filename, 'rb')
    		self.sampler.append( cPickle.load(pkl_file) )

	# parse the lambda* filenames to get the full list of lambdas
	self.nlambda = len(exp_files)
	self.lam = [float( (s.split('lambda')[1]).replace('.yaml','') ) for s in exp_files ]
	if debug:
		print 'lam =', self.lam

    def MBAR_analysis(self, debug = False):
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
	print 'nstates', nstates
	states_kn = np.zeros( (self.K, nsnaps) )

	# Get snapshot energies rescored in the different ensembles
	"""['step', 'E', 'accept', 'state', 'sigma_noe', 'sigma_J', 'sigma_cs', 'sigma_pf''gamma']
	[int(step), float(self.E), int(accept), int(self.state), int(self.sigma_noe_index), int(self.sigma_J_index), int(self.sigma_cs_H_index), int(self.sigma_cs_Ha_index), int(self.sigma_cs_N_index), int(self.sigma_cs_Ca_index), int(self.sigma_pf_index), int(self.gamma_index)]               	"""

	for n in range(nsnaps):

  		for k in range(self.K):
    			for l in range(self.K):
				if debug:
      					print 'step', self.traj[k]['trajectory'][n][0],
      				if k==l:
          				print 'E%d evaluated in model %d'%(k,k), self.traj[k]['trajectory'][n][1],
          				u_kln[k,k,n] = self.traj[k]['trajectory'][n][1]
          			state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_pf_index, gamma_index = self.traj[k]['trajectory'][n][3:] 	# IMPORTANT: make sure the order of these parameters is the same as the way they are saved in PosteriorSampler
          			print 'state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_pf_index, gamma_index', state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_pf_index, gamma_index 
          			states_kn[k,n] = state
          			sigma_noe = self.traj[k]['allowed_sigma_noe'][sigma_noe_index]
          			sigma_J = self.traj[k]['allowed_sigma_J'][sigma_J_index]
          			sigma_cs_H = self.traj[k]['allowed_sigma_cs_H'][sigma_cs_H_index]
          			sigma_cs_Ha = self.traj[k]['allowed_sigma_cs_Ha'][sigma_cs_Ha_index]
          			sigma_cs_N = self.traj[k]['allowed_sigma_cs_N'][sigma_cs_N_index]
          			sigma_cs_Ca = self.traj[k]['allowed_sigma_cs_Ca'][sigma_cs_Ca_index]
          			sigma_pf = self.traj[k]['allowed_sigma_pf'][sigma_pf_index]
          			u_kln[k,l,n] = self.sampler[l].neglogP(0, state, sigma_noe, sigma_J, sigma_cs_H, sigma_cs_Ha, sigma_cs_N, sigma_cs_Ca, sigma_pf, gamma_index)
				if debug:
					print 'E_%d evaluated in model_%d'%(k,l), u_kln[k,l,n]


	# Initialize MBAR with reduced energies u_kln and number of uncorrelated configurations from each state N_k.
 	# u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where U_l(x) is the potential energy function for state l,
	# beta is the inverse temperature, and and x_kn denotes uncorrelated configuration n from state k.
	# N_k[k] is the number of configurations from state k stored in u_knm
	# Note that this step may take some time, as the relative dimensionless free energies f_k are determined at this point.
	mbar = MBAR(u_kln, N_k)

	# Extract dimensionless free energy differences and their statistical uncertainties.
	(Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
	#(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
	#(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='approximate')
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
        	print 'state\tP\tdP'
    	for i in range(nstates):
        	A_kn = np.where(states_kn==i,1,0)
        	(p_i, dp_i) = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
        	self.P_dP[i,0:self.K] = p_i
        	self.P_dP[i,self.K:2*self.K] = dp_i
		print i
        	for p in p_i: print p,
        	for dp in dp_i: print dp,
		print 
	pops, dpops = self.P_dP[:,0:self.K], self.P_dP[:,self.K:2*self.K]

	# save results
	self.save_MBAR()

    def save_MBAR(self):
	"""save results (BICePs score and population) from MBAR analysis"""
	print 'Writing %s...'%self.BSdir
	savetxt(self.BSdir, self.f_df)
	print '...Done.'

	print 'Writing %s...'%self.popdir
	savetxt(self.popdir, self.P_dP)
	print '...Done.'

    def plot(self, debug = False):
	"""plot figures for population, nuisance parameters"""
	# first figure out what scheme is used
	self.list_scheme()

	# next get MABR sampling done
	self.MBAR_analysis()

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
    	plt.hold(True)
    	plt.plot([1e-6, 1], [1e-6, 1], color='k', linestyle='-', linewidth=2)
    	plt.xlim(1e-6, 1.)
    	plt.ylim(1e-6, 1.)
    	plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
    	plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
    	plt.xscale('log')
    	plt.yscale('log')
    	# label key states
    	plt.hold(True)
    	for i in range(len(pops1)):
        	if (i==0) or (pops1[i] > 0.05):
            		plt.text( pops0[i], pops1[i], str(i), color='g' )
    	for k in range(len(self.scheme)):
        	plt.subplot(r,c,k+2)
        	plt.step(t0['allowed_'+self.scheme[k]], t0['sampled_'+self.scheme[k]], 'b-')
        	plt.hold(True)
        	plt.xlim(0,5)
        	plt.step(t1['allowed_'+self.scheme[k]], t1['sampled_'+self.scheme[k]], 'r-')
        	plt.legend(['exp', 'sim+exp'], fontsize=legend_fontsize)
        	if self.scheme[k].find('cs') == -1:
            		plt.xlabel("$\%s$"%self.scheme[k], fontsize=label_fontsize)
            		plt.ylabel("$P(\%s)$"%self.scheme[k], fontsize=label_fontsize)
            		plt.yticks([])
        	else:
            		plt.xlabel("$\sigma_{%s}$"%self.scheme[k][6:],fontsize=label_fontsize)
            		plt.ylabel("$P(\sigma_{%s})$"%self.scheme[k][6:],fontsize=label_fontsize)
            		plt.yticks([])

    	plt.tight_layout()
    	plt.savefig(self.picfile)




