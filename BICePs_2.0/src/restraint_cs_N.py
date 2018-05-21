##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (N) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
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

class restraint_cs_N(object):

    def __init__(self):

        # Store chemical shift restraint info   #GYN
        self.cs_N_restraints = []
        self.ncs_N = 0


    def load_data_cs_N(self, filename, verbose=False):
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
            restraint_index, i, exp_cs_N, model_cs_N  = entry[0], entry[1], entry[4], entry[5]
            self.add_cs_N_restraint(i, exp_cs_N, model_cs_N)


        self.compute_sse_cs_N()



    def add_cs_N_restraint(self, i, exp_cs_N, model_cs_N=None):
        """Add a cs NMR_Chemicalshift() object to the list."""

        self.cs_N_restraints.append( NMR_Chemicalshift_N(i, model_cs_N, exp_cs_N))

        self.ncs_N += 1

    def compute_sse_cs_N(self, debug=False):    #GYN
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_N = 0.0
        N_N = 0.0
        for i in range(self.ncs_N):
		if debug:
               		print '---->', i, '%d'%self.cs_N_restraints[i].i,
               		print '      exp', self.cs_N_restraints[i].exp_cs_N, 'model', self.cs_N_restraints[i].model_cs_N

                err_N=self.cs_N_restraints[i].model_cs_N - self.cs_N_restraints[i].exp_cs_N
                sse_N += (self.cs_N_restraints[i].weight*err_N**2.0)
                N_N += self.cs_N_restraints[i].weight
        self.sse_cs_N = sse_N
        self.Ndof_cs_N = N_N
        if debug:
            print 'self.sse_cs_N', self.sse_cs_N



class NMR_Chemicalshift_N(object):        #GYN
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_cs_N, exp_cs_N):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_cs_N = model_cs_N

        # the experimental chemical shift
        self.exp_cs_N = exp_cs_N

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




