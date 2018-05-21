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
        self.chemicalshift_H_restraints = []
        self.nchemicalshift_H = 0


    def load_data_cs_H(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = prep_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_H, model_chemicalshift_H  = entry[0], entry[1], entry[4], entry[5]
            self.add_chemicalshift_H_restraint(i, exp_chemicalshift_H, model_chemicalshift_H)


        self.compute_sse_chemicalshift_H()



    def add_chemicalshift_H_restraint(self, i, exp_chemicalshift_H, model_chemicalshift_H=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""

        self.chemicalshift_H_restraints.append( NMR_Chemicalshift_H(i, model_chemicalshift_H, exp_chemicalshift_H))

        self.nchemicalshift_H += 1

    def compute_sse_chemicalshift_H(self, debug=True):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_H = 0.0
        N_H = 0.0
        for i in range(self.nchemicalshift_H):
		if debug:
               		print '---->', i, '%d'%self.chemicalshift_H_restraints[i].i,
               		print '      exp', self.chemicalshift_H_restraints[i].exp_chemicalshift_H, 'model', self.chemicalshift_H_restraints[i].model_chemicalshift_H

                err_H=self.chemicalshift_H_restraints[i].model_chemicalshift_H - self.chemicalshift_H_restraints[i].exp_chemicalshift_H
                sse_H += (self.chemicalshift_H_restraints[i].weight*err_H**2.0)
                N_H += self.chemicalshift_H_restraints[i].weight
        self.sse_chemicalshift_H = sse_H
        self.Ndof_chemicalshift_H = N_H
        if debug:
            print 'self.sse_chemicalshift_H', self.sse_chemicalshift_H



class NMR_Chemicalshift_H(object):        #GYH
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_chemicalshift_H, exp_chemicalshift_H):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_H = model_chemicalshift_H

        # the experimental chemical shift
        self.exp_chemicalshift_H = exp_chemicalshift_H

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




