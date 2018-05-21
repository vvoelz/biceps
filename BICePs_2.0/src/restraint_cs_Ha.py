##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (HA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
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


# Class Restraint:{{{
class restraint_cs_Ha(object):
    def __init__(self):

        # Store chemical shift restraint info   #GYHa
        self.chemicalshift_Ha_restraints = []
        self.nchemicalshift_Ha = 0


    def load_data_cs_Ha(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = prep_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_Ha, model_chemicalshift_Ha  = entry[0], entry[1], entry[4], entry[5]
            self.add_chemicalshift_Ha_restraint(i, exp_chemicalshift_Ha, model_chemicalshift_Ha)


        self.compute_sse_chemicalshift_Ha()



    def add_chemicalshift_Ha_restraint(self, i, exp_chemicalshift_Ha, model_chemicalshift_Ha=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""

        self.chemicalshift_Ha_restraints.append( NMR_Chemicalshift_Ha(i, model_chemicalshift_Ha, exp_chemicalshift_Ha))

        self.nchemicalshift_Ha += 1

    def compute_sse_chemicalshift_Ha(self, debug=True):    #GYHa
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_Ha = 0.0
        N_Ha = 0.0
        for i in range(self.nchemicalshift_Ha):
		if debug:
               		print '---->', i, '%d'%self.chemicalshift_Ha_restraints[i].i,
               		print '      exp', self.chemicalshift_Ha_restraints[i].exp_chemicalshift_Ha, 'model', self.chemicalshift_Ha_restraints[i].model_chemicalshift_Ha

                err_Ha=self.chemicalshift_Ha_restraints[i].model_chemicalshift_Ha - self.chemicalshift_Ha_restraints[i].exp_chemicalshift_Ha
                sse_Ha += (self.chemicalshift_Ha_restraints[i].weight*err_Ha**2.0)
                N_Ha += self.chemicalshift_Ha_restraints[i].weight
        self.sse_chemicalshift_Ha = sse_Ha
        self.Ndof_chemicalshift_Ha = N_Ha
        if debug:
            print 'self.sse_chemicalshift_Ha', self.sse_chemicalshift_Ha



class NMR_Chemicalshift_Ha(object):        #GYHa
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_chemicalshift_Ha, exp_chemicalshift_Ha):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_Ha = model_chemicalshift_Ha

        # the experimental chemical shift
        self.exp_chemicalshift_Ha = exp_chemicalshift_Ha

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




