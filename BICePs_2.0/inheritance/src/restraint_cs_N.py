##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (N) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
import mdtraj
from prep_cs import *   # Class - creates Chemical shift restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################

class Restraint_cs_N(Restraint):
    """A derived class of Restraint() for N chemical shift restraints."""

    def load_data_cs_N(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format."""

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
            self.add_restraint(NMR_Chemicalshift(i, exp, model))
        self.compute_sse()
      #  self.n += 1


