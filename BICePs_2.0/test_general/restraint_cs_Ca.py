##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (CA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
from prep_cs import *    # Class - creates Chemical shift restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################

class Restraint_cs_Ca(RestraintClass):
    """A derived class of RestraintClass() for C_alpha chemical shift restraints."""

    def load_data(self, filename, verbose=False):
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
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Chemicalshift_Ca(i, model, exp))
        self.compute_sse()

class NMR_Chemicalshift_Ca(object):
    """A data containter class to store a datum for NMR chemical shift information."""

    def __init__(self, i, model, exp):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i
        # the model chemical shift in this structure (in ppm)
        self.model = model
        # the experimental chemical shift
        self.exp = exp
        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1






