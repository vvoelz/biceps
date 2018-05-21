#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj
# Can we get rid of yaml and substitute for another multicolumn layout?
# Ideas:{{{

# }}}

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_pf import *    # Class - creates Chemical shift restraint file

# }}}

# Class Restraint:{{{
class restraint_pf(object):

    def __init__(self):

        # Store chemical shift restraint info   #GYH
        self.pf_restraints = []
        self.npf = 0

    def load_data_pf(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = prep_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]

        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_pf, model_pf  = entry[0], entry[0], entry[3]
            self.add_pf_restraint(i, exp_pf, model_pf)

        # build groups of equivalency group indices, ambiguous group etc.

        self.compute_sse_pf()



    def add_pf_restraint(self, i, exp_pf, model_pf=None):
        """Add a NMR_Protectionfactor() object to the list."""

        self.pf_restraints.append( NMR_Protectionfactor(i, model_pf, exp_pf))

        self.npf += 1

    def compute_sse_pf(self, debug=True):    #GYH
        """Returns the (weighted) sum of squared errors for protection factor values"""
#       for g in range(len(self.allowed_gamma)):

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



