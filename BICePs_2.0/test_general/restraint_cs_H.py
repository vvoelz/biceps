##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (NH) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
from prep_cs import *             # Class - creates Chemical shift restraint file
from Restraint import *

###############################################################################
# Code
###############################################################################

class Restraint_cs_H(RestraintClass):
    """A derived class of RestraintClass() for H chemical shift restraints."""

    #def load_data_cs_H(self, filename, verbose=True):
    def load_data(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format."""

        # Read in the lines of the cs data file

        #loading = load_data(filename=filename)
        read = prep_cs(filename=filename)
        if verbose:
                print read.lines
        data = []
        for line in read.lines:
                data.append( read.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]

        # add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            #self.add_restraint(i, exp, model)
            self.add_restraint(NMR_Chemicalshift_H(i, exp, model))
            if verbose:
                print entry

        self.compute_sse()

class NMR_Chemicalshift_H(object):
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model, exp):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i
        # the model chemical shift in this structure (in ppm)
        self.model = model
        # the experimental chemical shift
        self.exp = exp
        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




