##############################################################################
# Authors: Vincent Voelz, Yunhui Ge, Rob Raddi
# This file is used to do posterior sampling of BICePs calculation.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob, copy
import numpy as np
from scipy  import loadtxt, savetxt
from matplotlib import pylab as plt
from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from Restraint import *   # Import the Restraint Parent Class as R
from toolbox import *

##############################################################################
# Main
##############################################################################

class PosteriorSampler(object):
    """A class to perform posterior sampling of conformational populations"""

    def __init__(self, ensemble, no_ref=False, use_exp_ref=True, use_gau_ref=False,
            freq_write_traj=1000, freq_print=1000,
            freq_save_traj=100):
        """Initialize PosteriorSampler Class."""

        # Step frequencies to write trajectory info
        self.write_traj = freq_write_traj

        # Frequency of printing to the screen
        self.print_every = freq_print # debug

        # Frequency of storing trajectory samples
        self.traj_every = freq_save_traj

        # Ensemble is a list of Restraint objects
        self.ensembles = [ ensemble ]
        self.nstates = len(ensemble)
        self.nensembles = len(self.ensembles)
        self.ensemble_index = 0

        # the initial state of the structural ensemble we're sampling from
        self.state = 0    # index in the ensemble
        self.E = 1.0e99   # initial energy
        self.accepted = 0
        self.total = 0

        # keep track of what we sampled in a trajectory
        self.traj = PosteriorSamplingTrajectory(
                self.ensembles[0],self.allowed_sigma,self.allowed_gamma)

        # compile reference potential of noe from the uniform distribution of noe
        self.no_ref = no_ref
        self.use_exp_ref = use_exp_ref
        self.use_gau_ref = use_gau_ref

        if not self.no_ref:
            s = self.ensembles[0][0]
            print '\n\n\n',s.sse,'\n\n\n'
            if s.sse != 0:
                if self.use_exp_ref == True and self.use_gau_ref == True:
                    self.build_gau_ref()
                if self.use_exp_ref == True and self.use_gau_ref == False:
                    self.build_exp_ref()

        # VERY IMPORTANT: compute reference state self.logZ  for the free energies, so they are properly normalized #
        Z = 0.0
        for ensemble_index in range(self.nensembles):
            for s in self.ensembles[ensemble_index]:
                Z +=  np.exp(-s.free_energy)
        self.logZ = np.log(Z)

        # store this constant so we're not recalculating it all the time in neglogP
        self.ln2pi = np.log(2.0*np.pi)

    def build_exp_ref(self):
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            n = len(ensemble[0].restraints)
     #       print "n", n
            All = []
            print 'n = ',n
            distributions = [[] for j in range(n)]
            for s in ensemble:
                for j in range(len(s.restraints)):
                    print s.restraints[j].model
                    distributions[j].append( s.restraints[j].model )
                    All.append( s.restraints[j].model )

            # Find the MLE average (i.e. beta_j) for each noe
            betas = np.zeros(n)
            for j in range(n):
                # plot the maximum likelihood exponential distribution fitting the data
                betas[j] =  np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas = betas
    #            print "s.betas", s.betas
                s.compute_neglog_exp_ref()

    def build_gau_ref(self):

        for k in range(self.nensembles):

            print 'Computing Gaussian reference potentials for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across All structures
            n = len(ensemble[0].restraints)
            All = []
            distributions = [[] for j in range(n)]
            for s in ensemble:
                for j in range(len(s.restraints)):
                    distributions[j].append( s.restraints[j].model )
                    All.append( s.restraints[j].model )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean = np.zeros(n)
            ref_sigma = np.zeros(n)
            for j in range(n):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean[j] =  np.array(distributions[j]).mean()
                squared_diffs = [ (d - ref_mean[j])**2.0 for d in distributions[j] ]
                ref_sigma[j] = np.sqrt( np.array(squared_diffs).sum() / (len(distributions[j])+1.0))
            global_ref_sigma = ( np.array([ref_sigma[j]**-2.0 for j in range(n)]).mean() )**-0.5
            for j in range(n):
                ref_sigma[j] = global_ref_sigma
#               ref_sigma[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean = ref_mean
                s.ref_sigma = ref_sigma
                s.compute_neglog_gau_ref()



    def neglogP(self, new_ensemble_index, new_state, new_sigma,
            new_gamma_index, verbose=True):
        """Return -ln P of the current configuration."""

        # The current structure being sampled
        s = self.ensembles[new_ensemble_index][new_state]

        print 's = ',s
        # model terms
        result = s.free_energy + self.logZ
        print 'Result =',result
        # noe terms
        #result += (Nj+1.0)*np.log(self.sigma)
        #if s.sse is not None:        # trying to fix a future warning:"comparison to `None` will result in an elementwise object comparison in the future."
        if s.sse != 0:
            result += (s.Ndof)*np.log(new_sigma)  # for use with log-spaced sigma values
            #result += s.sse[new_gamma_index] / (2.0*new_sigma**2.0)
            result += s.sse / (2.0*new_sigma**2.0)
            result += (s.Ndof)/2.0*self.ln2pi  # for normalization
            #if self.use_exp_ref == True and self.use_gau_ref == True:
            #    result -= s.sum_neglog_gau_ref
            #if self.use_exp_ref == True and self.use_gau_ref == False:
            #    result -= s.sum_neglog_exp_ref

        if verbose:
            print 'state, f_sim', new_state, s.free_energy,
            print 's.sse', s.sse, 's.Ndof', s.Ndof
            print 's.sum_neglog_exp_ref', s.sum_neglog_exp_ref
            print 's.sum_neglog_gau_ref', s.sum_neglog_gau_ref
        return result


#NOTE: sample(self, nsteps) will be rewritten in cpp

    def sample(self, nsteps):
        "Perform nsteps of posterior sampling."

        for step in range(nsteps):

            new_sigma = self.sigma
            new_sigma_index = self.sigma_index
            new_gamma = self.gamma
            new_gamma_index = self.gamma_index
            new_state = self.state
            new_ensemble_index = self.ensemble_index

            if np.random.random() < 0.16:
                # take a step in array of allowed sigma
                new_sigma_index += (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(self.allowed_sigma)) # don't go out of bounds
                new_sigma = self.allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.32:
                # take a step in array of allowed sigma_J
                new_sigma_index += (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(self.allowed_sigma)) # don't go out of bounds
                new_sigma = self.allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.48 :
                # take a step in array of allowed sigma_cs
                new_sigma_index += (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(self.allowed_sigma)) # don't go out of bounds
                new_sigma = self.allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.60 :
                # take a step in array of allowed sigma_pf
                new_sigma_index += (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(self.allowed_sigma)) # don't go out of bounds
                new_sigma = self.allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.78:
                # take a step in array of allowed gamma
                new_sigma_index += (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(self.allowed_sigma)) # don't go out of bounds
                new_sigma = self.allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.99:
                # take a random step in state space
                new_state = np.random.randint(self.nstates)

            else:
                # pick a random pair of ambiguous groups to switch
                new_ensemble_index = np.random.randint(self.nensembles)

            # compute new "energy"
            verbose = True
#            if step%self.print_every == 0:
#                verbose = True
            new_E = self.neglogP(new_ensemble_index, new_state, new_sigma,
                    new_gamma_index, verbose=verbose)

            # accept or reject the MC move according to Metroplis criterion
            accept = False
            if new_E < self.E:
                accept = True
            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

            # Store trajectory counts
            self.traj.sampled_sigma[self.sigma_index] += 1
            self.traj.sampled_gamma[self.gamma_index] += 1
            self.traj.state_counts[self.state] += 1

            # update parameters
            if accept:
                self.E = new_E
                self.sigma = new_sigma
                self.sigma_index = new_sigma_index
                self.gamma = new_gamma
                self.gamma_index = new_gamma_index
                self.state = new_state
                self.ensemble_index = new_ensemble_index
                self.accepted += 1.0
            self.total += 1.0

            # store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step), float(self.E),
                    int(accept), int(self.state), int(self.sigma_index),
                    int(self.gamma_index)] )


class PosteriorSamplingTrajectory(object):
    "A class to store and perform operations on the trajectories of sampling runs."

    def __init__(self, ensemble, allowed_sigma, allowed_gamma):
        "Initialize the PosteriorSamplingTrajectory."

        self.nstates = len(ensemble)
        #print 'self.nstates = ',self.nstates
        self.ensemble = ensemble

        self.n = len(ensemble[0].restraints)

        print 'self.ensemble[0] = ',self.ensemble[0]
        self.restraints = len(self.ensemble[0].restraints)

        self.allowed_sigma = allowed_sigma
        self.sampled_sigma = np.zeros(len(allowed_sigma))

        self.allowed_gamma = allowed_gamma
        self.sampled_gamma = np.zeros(len(allowed_gamma))

        self.state_counts = np.ones(self.nstates)  # add a pseudocount to avoid log(0) errors

        self.f_sim = np.array([e.free_energy for e in ensemble])
        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        # stores samples [step, self.E, accept, state, sigma, sigma_J, sigma_cs, gamma]
        self.trajectory_headers = ['step', 'E', 'accept', 'state', 'sigma_index', 'sigma_J_index', 'sigma_index', 'sigmaa_index', 'sigma_cs_N_index', 'sigma_cs_Ca_index', 'sigma_pf_index', 'gamma_index']
        self.trajectory = []

        # a dictionary to store results for YAML file
        self.results = {}



    def process(self):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables.

        NOTE: Where possible, we convert to lists, because the YAML output
        is more readable"""

        # Store the trajectory in rsults
        self.results['trajectory_headers'] = self.trajectory_headers
        self.results['trajectory'] = self.trajectory

        # Store the nuisance parameter distributions
        self.results['allowed_sigma'] = self.allowed_sigma.tolist()
        self.results['allowed_gamma'] = self.allowed_gamma.tolist()
        self.results['sampled_sigma'] = self.sampled_sigma.tolist()
        self.results['sampled_gamma'] = self.sampled_gamma.tolist()

        # Calculate the modes of the nuisance parameter marginal distributions
        self.results['sigma_mode'] = float(self.allowed_sigma[ np.argmax(self.sampled_sigma) ])
        self.results['gamma_mode'] = float(self.allowed_gamma[ np.argmax(self.sampled_gamma) ])

        # copy over the purely computational free energies f_i
        self.results['comp_f'] = self.f_sim.tolist()

        # Estimate the populations of each state
        self.results['state_pops'] = (self.state_counts/self.state_counts.sum()).tolist()

        # Estimate uncertainty in the populations by bootstrap
        self.nbootstraps = 1000
        self.bootstrapped_state_pops = np.random.multinomial(self.state_counts.sum(), self.results['state_pops'], size=self.nbootstraps)
        self.results['state_pops_std'] = self.bootstrapped_state_pops.std(axis=0).tolist()

        # Estimate the free energies of each state
        self.results['state_f'] = (-np.log(self.results['state_pops'])).tolist()
        state_f = -np.log(self.results['state_pops'])
        ref_f = state_f.min()
        state_f -=  ref_f
        self.results['state_f'] = state_f.tolist()
        self.bootstrapped_state_f = -np.log(self.bootstrapped_state_pops+1e-10) - ref_f  # add pseudocount to avoid log(0)s in the bootstrap
        self.results['state_f_std'] = self.bootstrapped_state_f.std(axis=0).tolist()
#################################################################################
        # Estimate the ensemble-<r**-6>averaged noe
        mean = np.zeros(self.n)
        Z = np.zeros(self.n)
        for i in range(self.nstates):
            for j in range(self.n):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].restraints[j].weight
                r = self.ensemble[i].restraints[j].model
                mean[j] += pop*weight*(r**(-6.0))
                Z[j] += pop*weight
        mean = (mean/Z)**(-1.0/6.0)
        self.results['mean'] = mean.tolist()

        # compute the experimental noe, using the most likely gamma'
        exp = np.array([self.results['gamma_mode']*self.ensemble[0].restraints[j].exp \
                                      for j in range(self.n)])
        self.results['exp'] = exp.tolist()

#        self.results['pairs'] = []
#        for j in range(self.n):
#            pair = [int(self.ensemble[0].restraints[j].i), int(self.ensemble[0].restraints[j].j)]
#            self.results['pairs'].append(pair)
#        abs_diffs = np.abs( exp - mean )
#        self.results['disagreement_mean'] = float(abs_diffs.mean())
#        self.results['disagreement_std'] = float(abs_diffs.std())

#        # Estimate the ensemble-averaged J-coupling values
#        mean_Jcoupling = np.zeros(self.ndihedrals)
#        Z = np.zeros(self.ndihedrals)
#        for i in range(self.nstates):
#            for j in range(self.ndihedrals):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].dihedral_restraints[j].weight
#                r = self.ensemble[i].dihedral_restraints[j].model_Jcoupling
#                mean_Jcoupling[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_Jcoupling = (mean_Jcoupling/Z)
#        self.results['mean_Jcoupling'] = mean_Jcoupling.tolist()
#
#        # Compute the experiment Jcouplings
#        exp_Jcoupling = np.array([self.ensemble[0].dihedral_restraints[j].exp_Jcoupling for j in range(self.ndihedrals)])
#        self.results['exp_Jcoupling'] = exp_Jcoupling.tolist()
#        abs_Jdiffs = np.abs( exp_Jcoupling - mean_Jcoupling )
#        self.results['disagreement_Jcoupling_mean'] = float(abs_Jdiffs.mean())
#        self.results['disagreement_Jcoupling_std'] = float(abs_Jdiffs.std())
#
        # Estimate the ensemble-averaged chemical shift values
        mean = np.zeros(self.n)
        Z = np.zeros(self.n)
        for i in range(self.nstates):
            for j in range(self.n):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].restraints[j].weight
                r = self.ensemble[i].restraints[j].model
                mean[j] += pop*weight*r
                Z[j] += pop*weight
        mean = (mean/Z)
        self.results['mean'] = mean.tolist()

        # Compute the experiment chemical shift
        exp = np.array([self.ensemble[0].restraints[j].exp for j in range(self.n)])
        self.results['exp'] = exp.tolist()
        abs_diffs = np.abs( exp - mean )
        self.results['disagreement_mean'] = float(abs_diffs.mean())
        self.results['disagreement_std'] = float(abs_diffs.std())
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_Ha = np.zeros(self.ncs_Ha)
#        Z = np.zeros(self.ncs_Ha)
#        for i in range(self.nstates):
#            for j in range(self.ncs_Ha):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_Ha_restraints[j].weight
#                r = self.ensemble[i].cs_Ha_restraints[j].model_cs_Ha
#                mean_cs_Ha[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_Ha = (mean_cs_Ha/Z)
#        self.results['mean_cs_Ha'] = mean_cs_Ha.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_Ha = np.array([self.ensemble[0].cs_Ha_restraints[j].exp_cs_Ha for j in range(self.ncs_Ha)])
#        self.results['exp_cs_Ha'] = exp_cs_Ha.tolist()
#        abs_cs_Ha_diffs = np.abs( exp_cs_Ha - mean_cs_Ha )
#        self.results['disagreement_cs_Ha_mean'] = float(abs_cs_Ha_diffs.mean())
#        self.results['disagreement_cs_Ha_std'] = float(abs_cs_Ha_diffs.std())
#
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_N = np.zeros(self.ncs_N)
#        Z = np.zeros(self.ncs_N)
#        for i in range(self.nstates):
#            for j in range(self.ncs_N):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_N_restraints[j].weight
#                r = self.ensemble[i].cs_N_restraints[j].model_cs_N
#                mean_cs_N[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_N = (mean_cs_N/Z)
#        self.results['mean_cs_N'] = mean_cs_N.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_N = np.array([self.ensemble[0].cs_N_restraints[j].exp_cs_N for j in range(self.ncs_N)])
#        self.results['exp_cs_N'] = exp_cs_N.tolist()
#        abs_cs_N_diffs = np.abs( exp_cs_N - mean_cs_N )
#        self.results['disagreement_cs_N_mean'] = float(abs_cs_N_diffs.mean())
#        self.results['disagreement_cs_N_std'] = float(abs_cs_N_diffs.std())
#
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_Ca = np.zeros(self.ncs_Ca)
#        Z = np.zeros(self.ncs_Ca)
#        for i in range(self.nstates):
#            for j in range(self.ncs_Ca):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_Ca_restraints[j].weight
#                r = self.ensemble[i].cs_Ca_restraints[j].model_cs_Ca
#                mean_cs_Ca[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_Ca = (mean_cs_Ca/Z)
#        self.results['mean_cs_Ca'] = mean_cs_Ca.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_Ca = np.array([self.ensemble[0].cs_Ca_restraints[j].exp_cs_Ca for j in range(self.ncs_Ca)])
#        self.results['exp_cs_Ca'] = exp_cs_Ca.tolist()
#        abs_cs_Ca_diffs = np.abs( exp_cs_Ca - mean_cs_Ca )
#        self.results['disagreement_cs_Ca_mean'] = float(abs_cs_Ca_diffs.mean())
#        self.results['disagreement_cs_Ca_std'] = float(abs_cs_Ca_diffs.std())
#
#
#
#        # Estimate the ensemble-averaged protection factor values
#        mean_pf = np.zeros(self.npf)
#        Z = np.zeros(self.npf)
#        for i in range(self.nstates):
#            for j in range(self.npf):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].pf_restraints[j].weight
#                r = self.ensemble[i].pf_restraints[j].model_pf
#                mean_pf[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_pf = (mean_pf/Z)
#        self.results['mean_pf'] = mean_pf.tolist()
#
#        # Compute the experiment protection factor
#
#        exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
##        exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
#        self.results['exp_pf'] = exp_pf.tolist()
#        abs_pfdiffs = np.abs( exp_pf - mean_pf )
#        self.results['disagreement_pf_mean'] = float(abs_pfdiffs.mean())
#        self.results['disagreement_pf_std'] = float(abs_pfdiffs.std())


    def logspaced_array(self, xmin, xmax, nsteps):
        ymin, ymax = np.log(xmin), np.log(xmax)
        dy = (ymax-ymin)/nsteps
        return np.exp(np.arange(ymin, ymax, dy))


    #NOTE: This will work well with Cython if we go that route.
    # Standardized: Yes ; Binary: Yes; Human Readable: No;

    def write_results(self, outfilename='traj.npz'):
        """Writes a compact file of several arrays into binary format."""

        np.savez_compressed(outfilename, self.results)

    def read_results(self,filename):
        """Reads a npz file"""

        loaded = np.load(filename)
        print loaded.items()







