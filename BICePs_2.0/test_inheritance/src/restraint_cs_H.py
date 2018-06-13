##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (NH) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################


##############################################################################
# Imports
##############################################################################


import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_cs import *    # Class - creates Chemical shift restraint file
###############################################################################
# Code
###############################################################################

class restraint_cs_H(object):
    def __init__(self):

        # Store chemical shift restraint info   #GYH
        self.cs_H_restraints = []
        self.ncs_H = 0
        self.sse_cs_H = 0 
        self.Ndof_cs_H = None
        self.betas_H = None
        self.ref_sigma_H = None
        self.ref_mean_H = None
        self.neglog_reference_potentials_H = None
	self.gaussian_neglog_reference_potentials_H = None
        self.sum_neglog_reference_potentials_H = 0.0    #GYH
        self.sum_gaussian_neglog_reference_potentials_H = 0.0      #GYH

    def load_data_cs_H(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format.
        """

        # Read in the lines of the cs data file
        b = prep_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_cs_H, model_cs_H  = entry[0], entry[1], entry[4], entry[5]
            self.add_cs_H_restraint(i, exp_cs_H, model_cs_H)


        self.compute_sse_cs_H()



    def add_cs_H_restraint(self, i, exp_cs_H, model_cs_H=None):
        """Add a cs NMR_Chemicalshift() object to the list."""

        self.cs_H_restraints.append( NMR_Chemicalshift_H(i, model_cs_H, exp_cs_H))

        self.ncs_H += 1

    def compute_sse_cs_H(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_H = 0.0
        N_H = 0.0
        for i in range(self.ncs_H):
		if debug:
               		print '---->', i, '%d'%self.cs_H_restraints[i].i,
               		print '      exp', self.cs_H_restraints[i].exp_cs_H, 'model', self.cs_H_restraints[i].model_cs_H

                err_H=self.cs_H_restraints[i].model_cs_H - self.cs_H_restraints[i].exp_cs_H
                sse_H += (self.cs_H_restraints[i].weight*err_H**2.0)
                N_H += self.cs_H_restraints[i].weight
        self.sse_cs_H = sse_H
        self.Ndof_cs_H = N_H
        if debug:
            print 'self.sse_cs_H', self.sse_cs_H

    def compute_neglog_reference_potentials_H(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_H = np.zeros(self.ncs_H)
        self.sum_neglog_reference_potentials_H = 0.
        for j in range(self.ncs_H):
            self.neglog_reference_potentials_H[j] = np.log(self.betas_H[j]) + self.cs_H_restraints[j].model_cs_H/self.betas_H[j]
            self.sum_neglog_reference_potentials_H  += self.cs_H_restraints[j].weight * self.neglog_reference_potentials_H[j]
            print "self.sum_neglog_reference_potentials_H", self.sum_neglog_reference_potentials_H
    def compute_gaussian_neglog_reference_potentials_H(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_H = np.zeros(self.ncs_H)
        self.sum_gaussian_neglog_reference_potentials_H = 0.
        for j in range(self.ncs_H):
            self.gaussian_neglog_reference_potentials_H[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_H[j]) + (self.cs_H_restraints[j].model_cs_H - self.ref_mean_H[j])**2.0/(2*self.ref_sigma_H[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_H += self.cs_H_restraints[j].weight * self.gaussian_neglog_reference_potentials_H[j]


class NMR_Chemicalshift_H(object):        #GYH
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_cs_H, exp_cs_H):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_cs_H = model_cs_H

        # the experimental chemical shift
        self.exp_cs_H = exp_cs_H

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




