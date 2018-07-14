##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (CA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
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

##############################################################################
# Code
##############################################################################


class restraint_cs_Ca(object):
    def __init__(self):

        # Store chemical shift restraint info   
        self.cs_Ca_restraints = []
        self.ncs_Ca = 0
        self.sse_cs_Ca = 0 
        self.Ndof_cs_Ca = None 
        self.betas_Ca = None
        self.ref_sigma_Ca = None
        self.ref_mean_Ca = None
        self.neglog_reference_potentials_Ca = None
        self.gaussian_neglog_reference_potentials_Ca = None
        self.sum_neglog_reference_potentials_Ca = 0.0   #GYH
        self.sum_gaussian_neglog_reference_potentials_Ca = 0.0      #GYH


    def load_data_cs_Ca(self, filename, verbose=False):
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
            restraint_index, i, exp_cs_Ca, model_cs_Ca  = entry[0], entry[1], entry[4], entry[5]
            self.add_cs_Ca_restraint(i, exp_cs_Ca, model_cs_Ca)


        self.compute_sse_cs_Ca()



    def add_cs_Ca_restraint(self, i, exp_cs_Ca, model_cs_Ca=None):
        """Add a cs NMR_Chemicalshift() object to the list."""

        self.cs_Ca_restraints.append( NMR_Chemicalshift_Ca(i, model_cs_Ca, exp_cs_Ca))

        self.ncs_Ca += 1

    def compute_sse_cs_Ca(self, debug=False):    #GYCa
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_Ca = 0.0
        N_Ca = 0.0
        for i in range(self.ncs_Ca):
		if debug:
               		print '---->', i, '%d'%self.cs_Ca_restraints[i].i,
               		print '      exp', self.cs_Ca_restraints[i].exp_cs_Ca, 'model', self.cs_Ca_restraints[i].model_cs_Ca

                err_Ca=self.cs_Ca_restraints[i].model_cs_Ca - self.cs_Ca_restraints[i].exp_cs_Ca
                sse_Ca += (self.cs_Ca_restraints[i].weight*err_Ca**2.0)
                N_Ca += self.cs_Ca_restraints[i].weight
        self.sse_cs_Ca = sse_Ca
        self.Ndof_cs_Ca = N_Ca
        if debug:
            print 'self.sse_cs_Ca', self.sse_cs_Ca

    def compute_neglog_reference_potentials_Ca(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
        self.sum_neglog_reference_potentials_Ca = 0.
        for j in range(self.ncs_Ca):
            self.neglog_reference_potentials_Ca[j] = np.log(self.betas_Ca[j]) + self.cs_Ca_restraints[j].model_cs_Ca/self.betas_Ca[j]
            self.sum_neglog_reference_potentials_Ca  += self.cs_Ca_restraints[j].weight * self.neglog_reference_potentials_Ca[j]

    def compute_gaussian_neglog_reference_potentials_Ca(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
        self.sum_gaussian_neglog_reference_potentials_Ca = 0.
        for j in range(self.ncs_Ca):
            self.gaussian_neglog_reference_potentials_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.cs_Ca_restraints[j].model_cs_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_Ca += self.cs_Ca_restraints[j].weight * self.gaussian_neglog_reference_potentials_Ca[j]



class NMR_Chemicalshift_Ca(object):        #GYH
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_cs_Ca, exp_cs_Ca):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_cs_Ca = model_cs_Ca

        # the experimental chemical shift
        self.exp_cs_Ca = exp_cs_Ca

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




