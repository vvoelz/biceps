##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for protection factors in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################


##############################################################################
# Imports
##############################################################################

import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_pf import *    # Class - prepare functions for protection factors restraint file

##############################################################################
# Code
##############################################################################

class restraint_pf(object):

    def __init__(self):

        # Store chemical shift restraint info   #GYH
        self.pf_restraints = []
        self.npf = 0
        self.sse_pf = 0
        self.Ndof_pf = None
        self.betas_pf = None
        self.ref_sigma_pf = None
        self.ref_mean_pf = None
        self.neglog_reference_potentials_pf = None
        self.gaussian_neglog_reference_potentials_pf = None
        self.sum_neglog_reference_potentials_pf = 0.0   #GYH
        self.sum_gaussian_neglog_reference_potentials_pf = 0.0      #GYH

    def load_data_pf(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the protection factors data file
        b = prep_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_pf(line) ) 

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        # add the protection factors restraints
        for entry in data:
            restraint_index, i, exp_pf, model_pf  = entry[0], entry[1], entry[4], entry[5]
            self.add_pf_restraint(i, exp_pf, model_pf)

        self.compute_sse_pf()



    def add_pf_restraint(self, i, exp_pf, model_pf=None):
        """Add a NMR_Protectionfactor() object to the list."""

        self.pf_restraints.append( NMR_Protectionfactor(i, model_pf, exp_pf))

        self.npf += 1

    def compute_sse_pf(self, debug=True):    #GYH
        """Returns the (weighted) sum of squared errors for protection factor values"""

        sse_pf = 0.0
        N_pf = 0.0
        for i in range(self.npf):
		if debug:
               		print '---->', i, '%d'%self.pf_restraints[i].i,
               		print '      exp', self.pf_restraints[i].exp_pf, 'model', self.pf_restraints[i].model_pf

                err_pf=self.pf_restraints[i].model_pf - self.pf_restraints[i].exp_pf
                sse_pf += (self.pf_restraints[i].weight*err_pf**2.0)
                N_pf += self.pf_restraints[i].weight
        self.sse_pf = sse_pf
        self.Ndof_pf = N_pf
        if debug:
            print 'self.sse_pf', self.sse_pf

    def compute_neglog_reference_potentials_pf(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""


        self.neglog_reference_potentials_pf= np.zeros(self.nprotectionfactor)
        self.sum_neglog_reference_potentials_pf = 0.
        for j in range(self.nprotectionfactor):
            self.neglog_reference_potentials_pf[j] = np.log(self.betas_pf[j]) + self.protectionfactor_restraints[j].model_protectionfactor/self.betas_pf[j]
            self.sum_neglog_reference_potentials_pf  += self.protectionfactor_restraints[j].weight * self.neglog_reference_potentials_pf[j]


    def compute_gaussian_neglog_reference_potentials_pf(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_pf = np.zeros(self.nprotectionfactor)
        self.sum_gaussian_neglog_reference_potentials_pf = 0.
        for j in range(self.nprotectionfactor):
            self.gaussian_neglog_reference_potentials_pf[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_pf[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_pf[j])**2.0/(2*self.ref_sigma_pf[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_pf += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_potentials_pf[j]


class NMR_Protectionfactor(object):        #GYH
    """A class to store NMR protection factor information."""

    def __init__(self, i, model_pf, exp_pf):
        # Atom indices from the Conformation() defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model_pf = model_pf

        # the experimental protection factor
        self.exp_pf = exp_pf


        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1



