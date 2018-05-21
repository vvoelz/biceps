##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for protection factors in BICePs and prepare fuctions for compute necessary quantities for posterior sampling. This file is only used in a special case (apoMb) and unlikely will be used in any other cases. 
##############################################################################


##############################################################################
# Imports
##############################################################################

import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_pf import *    # Class - creates Chemical shift restraint file

##############################################################################
# Code
##############################################################################


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
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

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
            self.add_pf_restraint(i, exp_pf, model_pf=None)


        self.compute_sse_pf()



    def add_pf_restraint(self, i, exp_pf, model_pf=None):
        """Add a NMR_Protectionfactor() object to the list."""

        if model_pf == None:
            # model_protectionfactor is a 6-dimensional array!
            # model_protectionfactor = self.compute_PF_multi(self.Ncs[:,:,i], self.Nhs[:,:,i])
            model_pf = self.compute_PF_multi(self.Ncs[:,:,i], self.Nhs[:,:,i], debug=True)
        self.pf_restraint.append(NMR_Protectionfactor(i, model_pf, exp_pf))  #???
        self.npf += 1


    def compute_PF(self, beta_c, beta_h, beta_0, Nc, Nh):
        """Calculate predicted (ln PF)
        INPUT    (nres, 2) array with columns <N_c> and <N_h> for each residue, 
        OUTPUT   array of <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
        """
        return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 '''

    def compute_PF_multi(self, Ncs_i, Nhs_i, debug=False):
        """Calculate predicted (ln PF)
        INPUT    (nres, 2) array with columns <N_c> and <N_h> for each residue, 
        OUTPUT   array of <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
        """

        N_beta_c = len(self.allowed_beta_c)
        N_beta_h = len(self.allowed_beta_h)
        N_beta_0 = len(self.allowed_beta_0)
        N_xc     = len(self.allowed_xcs)
        N_xh     = len(self.allowed_xhs)
        N_b      = len(self.allowed_bs)

        nuisance_shape = (N_beta_c, N_beta_h, N_beta_0, N_xc, N_xh, N_b)

        beta_c = self.tile_multiaxis(self.allowed_beta_c, nuisance_shape, axis=0)
        beta_h = self.tile_multiaxis(self.allowed_beta_h, nuisance_shape, axis=1)
        beta_0 = self.tile_multiaxis(self.allowed_beta_0, nuisance_shape, axis=2)
        Nc     = self.tile_2D_multiaxis(Ncs_i, nuisance_shape, axes=[3,5])
        Nh     = self.tile_2D_multiaxis(Nhs_i, nuisance_shape, axes=[4,5])

        if debug:
            print 'nuisance_shape', nuisance_shape
            print 'beta_c.shape', beta_c.shape
            print 'beta_h.shape', beta_h.shape
            print 'beta_0.shape', beta_0.shape
            print 'Nc.shape', Nc.shape
            print 'Nh.shape', Nh.shape

        return beta_c * Nc + beta_h * Nh + beta_0

    def tile_multiaxis(self, p, shape, axis=None):
        """Returns a multi-dimensional array of shape (tuple), with the 1D vector p along the specified axis,
           and tiled in all other dimenions.

        INPUT
        p       a 1D array to tile
        shape   a tuple describing the shape of the returned array
        axis    the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """


        print 'shape', shape , 'axis', axis, 'p.shape', p.shape
        assert shape[axis] == len(p), "len(p) must be equal to shape[axis]!"

        otherdims = [shape[i] for i in range(len(shape)) if i!=axis]
        result = np.tile(p, tuple(otherdims+[1]))
        print 'result.shape', result.shape
        last_axis = len(result.shape)-1
        print 'last_axis, axis', last_axis, axis
        result2 = np.rollaxis(result, last_axis, axis)
        print 'result2.shape', result2.shape
        return result2

    def tile_2D_multiaxis(self, q, shape, axes=None):
        """Returns a multi-dimensional array of shape (tuple), with the 2D vector p along the specified axis
           and tiled in all other dimenions.

        INPUT
        q       a 2D array to tile
        shape   a tuple describing the shape of the returned array
        axes    a list of two specified axes   NOTE: q.shape must be equal to (shape[axes[0]],shape[axes[1]])
        """

        print 'shape', shape , 'axes', axes, 'q.shape', q.shape
        assert (shape[axes[0]],shape[axes[1]]) == q.shape, "q.shape must be equal to (shape[axes[0]],shape[axes[1]])"

        otherdims = [shape[i] for i in range(len(shape)) if i not in axes]
        result = np.tile(q, tuple(otherdims+[1,1]))
        last_axis = len(result.shape)-1
        next_last_axis = len(result.shape)-2
        result2 = np.rollaxis(result, next_last_axis, axes[0])
        return np.rollaxis( result2, last_axis, axes[1])





    def compute_sse_pf(self, debug=True):    #GYH
    """ new defined sse based on computed PF"""
        self.sse_protectionfactor = np.zeros(  (len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                                                len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))
        self.Ndof_protectionfactor = 0.
        for i in range(self.nprotectionfactor):
#           print err
            err=self.pf_restraint[i].model_pf - self.pf_restraint[i].exp_pf
#           print err
            self.sse_protectionfactor += (self.protectionfactor_restraints[i].weight * err**2.0)
            self.Ndof_protectionfactor += self.protectionfactor_restraints[i].weight

        if debug:
            print 'self.sse_protectionfactor', self.sse_protectionfactor



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



