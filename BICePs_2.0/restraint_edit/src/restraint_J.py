#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj
# Can we get rid of yaml and substitute for another multicolumn layout?

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_J import *     # Class - creates J-coupling const. restraint file

# }}}

# Class Restraint:{{{
class restraint_J(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self,dlogsigma_J=np.log(1.02),sigma_J_min=0.05,
            sigma_J_max=20.0):

        # Store dihedral restraint info
        self.dihedral_restraints = []
        self.dihedral_equivalency_groups = {}
        self.dihedral_ambiguity_groups = {}
        self.ndihedrals = 0

        # pick initial values for sigma_J (std of experimental uncertainty in J-coupling constant)
        self.dlogsigma_J = dlogsigma_J  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_J_min = sigma_J_min
        self.sigma_J_max = sigma_J_max
        self.allowed_sigma_J = np.exp(np.arange(np.log(self.sigma_J_min), np.log(self.sigma_J_max), self.dlogsigma_J))
#        print 'self.allowed_sigma_J', self.allowed_sigma_J
#        print 'len(self.allowed_sigma_J) =', len(self.allowed_sigma_J)

        self.sigma_J_index = len(self.allowed_sigma_J)/2   # pick an intermediate value to start with
        self.sigma_J = self.allowed_sigma_J[self.sigma_J_index]



    def load_data_J(self, filename, verbose=False):
        """Load in the experimental Jcoupling constant restraints from a .Jcoupling file format."""


        # Read in the lines of the biceps data file
        b = RestraintFile_J(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line_J(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, atom_index3, res3, atom_name3, atom_index4, res4, atom_name4, J_coupling(Hz)]

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
            restraint_index, i, j, k, l, exp_Jcoupling, karplus  = entry[0], entry[1], entry[4], entry[7], entry[10], entry[13], entry[14]
            self.add_dihedral_restraint(i, j, k, l, exp_Jcoupling, model_Jcoupling=None, equivalency_index=None, karplus_key=karplus)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_J()


    def add_dihedral_restraint(self, i, j, k, l, exp_Jcoupling, model_Jcoupling=None,
                               equivalency_index=None, karplus_key="Karplus_HH"):
        """Add a J-coupling NMR_Dihedral() object to the list."""

        # if the modeled Jcoupling value is not specified, compute it from the
        # angle corresponding to the conformation, and the Karplus relation
        if model_Jcoupling == None:
            ri, rj, rk, rl = [self.conf.xyz[0,x,:] for x in [i, j, k, l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)

            ###########################
            # NOTE: In the future, this function can be more sophisticated, parsing atom types
            # or karplus relation types on a case-by-case basis
            model_Jcoupling = self.karplus.J(model_angle, karplus_key)

        self.dihedral_restraints.append( NMR_Dihedral(i, j, k, l, model_Jcoupling, exp_Jcoupling, model_angle,
                                                      equivalency_index=equivalency_index) )
        self.ndihedrals += 1

    def build_groups_J(self, verbose=False):
        """Build equivalency and ambiguity groups for distances and dihedrals,
        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile dihedral_equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.dihedral_restraints)):
            d = self.dihedral_restraints[i]
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
        self.compute_sse_dihedrals()

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.dihedral_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.dihedral_restraints[i].weight = 1.0/n


    def compute_sse_dihedrals(self, debug=False):
        """Returns the (weighted) sum of squared errors for J-coupling values,
        and the *effective* number of distances (i.e. the sum of the weights)"""

        sse = 0.0
        N =  0.0

        remaining_dihedral_indices = range(len(self.dihedral_restraints))

        # First, find all the equivalent groups, and average them.
        # (We assume that all of the experimental values have set to the same J value)
        for equivalency_index, group in self.dihedral_equivalency_groups.iteritems():
            avgJ = np.array([self.dihedral_restraints[i].model_Jcoupling for i in group]).mean()
            err = self.dihedral_restraints[group[0]].exp_Jcoupling - avgJ
            if debug:
                print group, 'avgJ_model',avgJ, 'expJ',self.dihedral_restraints[group[0]].exp_Jcoupling, 'err', err
            sse += err**2.0
            N += 1
            # remove group indices from remaining_dihedral_indices
            for i in group:
                remaining_dihedral_indices.remove(i)

        for i in remaining_dihedral_indices:
            err = self.dihedral_restraints[i].exp_Jcoupling - self.dihedral_restraints[i].model_Jcoupling
            if debug:
                print 'J_model', self.dihedral_restraints[i].model_Jcoupling, 'exp', self.dihedral_restraints[i].exp_Jcoupling, 'err', err
            sse += (self.dihedral_restraints[i].weight * err**2.0)
            N += self.dihedral_restraints[i].weight

        if debug:
            print 'total sse', sse
        self.sse_dihedrals = sse
        self.Ndof_dihedrals = N

class NMR_Dihedral(object):
    """A class to store NMR J-coupling dihedral information."""

    # __init__:{{{
    def __init__(self, i, j, k, l, model_Jcoupling, exp_Jcoupling, model_angle, equivalency_index=None, ambiguity_index=None):

        # Atom indices from the Conformation() defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l

        # the model distance in this structure (in Angstroms)
        self.model_Jcoupling = model_Jcoupling

        # the experimental J-coupling constant
        self.exp_Jcoupling = exp_Jcoupling

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index




