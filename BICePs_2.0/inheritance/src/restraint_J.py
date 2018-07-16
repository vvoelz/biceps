##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for J coupling constants in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
from KarplusRelation import *       # Returns J-coupling values from dihedral angles
from prep_J import *   # Creates J-coupling const. restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################
#NOTE: this derived class is not yet completed and is more difficult

class Restraint_J(Restraint):
    """A derived class of RestraintClass() for J coupling constant."""

    def load_data(self, filename, verbose=False):
        """Load in the experimental Jcoupling constant restraints from a .Jcoupling file format."""

        # Read in the lines of the biceps data file
        b = prep_J(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, atom_index3, res3, atom_name3, atom_index4, res4, atom_name4, J_coupling(Hz)]
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        ### Jcoupling ###
        # the equivalency indices for Jcoupling are in the first column of the *.Jcoupling file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'distance equivalency_indices', equivalency_indices
        # add the Jcoupling restraints
        for entry in data:
            restraint_index, i, j, k, l, exp, karplus  = entry[0], entry[1], entry[4], entry[7], entry[10], entry[13], entry[14]

            # if the modeled Jcoupling value is not specified, compute it from the
            # angle corresponding to the conformation, and the Karplus relation
            ri, rj, rk, rl = [self.conf.xyz[0,x,:] for x in [i, j, k, l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)
            model = self.karplus.J(model_angle, "Karplus_HH")

        self.add_restraint(NMR_Dihedral(i,j,k,l,model,exp,model_angle,
            equivalency_index=equivalency_index))
        self.n += 1
        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()


    def build_groups(self, verbose=False):
        """Build equivalency and ambiguity groups for distances and dihedrals,
        and store pre-computed SSE and d.o.f for distances and dihedrals"""

        # compile dihedral_equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            # print 'd', d, 'd.equivalency_index', d.equivalency_index
            if d.equivalency_index != None:
                if not self.dihedral_equivalency_groups.has_key(d.equivalency_index):
                    self.dihedral_equivalency_groups[d.equivalency_index] = []
                self.dihedral_equivalency_groups[d.equivalency_index].append(i)
        if verbose:
            print 'self.dihedral_equivalency_groups', self.dihedral_equivalency_groups
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        # precompute SSE and Ndof for dihedrals
        self.compute_sse()

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.dihedral_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.restraints[i].weight = 1.0/n


    def compute_sse(self, debug=False):
        """Returns the (weighted) sum of squared errors for J-coupling values,
        and the *effective* number of distances (i.e. the sum of the weights)"""

        sse = 0.0
        N =  0.0

        remaining_dihedral_indices = range(len(self.dihedral_restraints))

        # First, find all the equivalent groups, and average them.
        # (We assume that all of the experimental values have set to the same J value)
        for equivalency_index, group in self.dihedral_equivalency_groups.iteritems():
            avgJ = np.array([self.dihedral_restraints[i].model for i in group]).mean()
            err = self.dihedral_restraints[group[0]].exp - avgJ
            if debug:
                print group, 'avgJ_model',avgJ, 'expJ',self.dihedral_restraints[group[0]].exp, 'err', err
            sse += err**2.0
            N += 1
            # remove group indices from remaining_dihedral_indices
            for i in group:
                remaining_dihedral_indices.remove(i)

        for i in remaining_dihedral_indices:
            err = self.dihedral_restraints[i].exp - self.dihedral_restraints[i].model
            if debug:
                print 'J_model', self.dihedral_restraints[i].model, 'exp', self.dihedral_restraints[i].exp, 'err', err
            sse += (self.dihedral_restraints[i].weight * err**2.0)
            N += self.dihedral_restraints[i].weight

        if debug:
            print 'total sse', sse
        self.sse = sse
        self.Ndof = N




