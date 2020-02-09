##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob, string
import numpy as np
import mdtraj as md
import re
from .prep_cs import *
from .prep_J import *
from .prep_noe import *
from .prep_pf import *

##############################################################################
# Code
##############################################################################

class Preparation(object):
    """A parent class to prepare input files for BICePs calculation.

    :param str scheme: type of experimental observables {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}

    :param int default=0 state: number of states

    :param str default=None indices: experimental observable index (*.txt file)

    :param str default=None exp_data: experimental measuremnets (*.txt file)

    :param str default=None top: topology file (*.pdb)

    :param str default=None data_dir: precomputed data directory (should have *txt file inside)
    """

#    def __init__(self,scheme=None,states=0.0,indices=None, exp_data=None, top=None, data_dir=None, Karplus=None):
    def __init__(self,scheme=None,states=0,indices=None, exp_data=None, top=None, data_dir=None, precomputed_pf=False):

        if scheme not in ['noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf']:
            raise ValueError("scheme must be one of ['noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf']")
        elif scheme == 'pf' and not precomputed_pf:
            if states==0.0 or indices == None or exp_data == None or top==None:
                raise ValueError("make sure you have actual input for states, indices, exp_data, topology file or data directory ")
        else:
            if states==0.0 or indices == None or exp_data == None or top==None or data_dir==None:
                raise ValueError("make sure you have actual input for states, indices, exp_data, topology file or data directory ")
#    if scheme == 'J' and Karplus ==None:
#        raise ValueError("Karplus relation information is missing but it's required for J_coupling constants")
#    elif scheme == 'J' and Karplus is not None:
#        with open(Karplus) as f:
#        lines=f.readlines()
#        line=''.join(lines)
#        self.karplus = line.strip().split('\n')
#    elif scheme != 'J' and Karplus == None:
#        continue

        self.ind=np.loadtxt(indices)
        self.restraint_data = np.loadtxt(exp_data)
        self.scheme = scheme
        if self.ind.shape[0] != self.restraint_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.')%(self.ind.shape[0],self.restraint_data.shape[0])

        self.topology = md.load(top).topology
        self.convert = lambda txt: int(txt) if txt.isdigit() else txt
        if scheme == 'pf' and not precomputed_pf:
            self.data  = self.restraint_data
        else:
            self.data = sorted(glob.glob(data_dir),key=lambda x: [self.convert(s) for s in re.split("([0-9]+)",x)])
            if int(len(self.data)) != int(states):
                raise ValueError("number of states doesn't equal to file numbers")

    def write(self,out_dir=None):
        """write BICePs format input files.

        :param str default=None out_dir: output directory
        """

        if out_dir == None:
            self.out = 'BICePs_'+self.scheme
        else:
            self.out = out_dir
        if not os.path.exists(self.out):
            os.mkdir(self.out)

        if self.scheme in ['cs_H','cs_Ha','cs_Ca','cs_N']:
            self.write_cs_input()
        elif self.scheme == 'noe':
            self.write_noe_input()
        elif self.scheme == 'J':
            self.write_J_input()
        elif self.scheme == 'pf':
            self.write_pf_input()
        else:
            raise ValueError("scheme must be one of ['noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf']")


    def write_cs_input(self):
        #print self.data
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])
            r = prep_cs()
            all_atom_indices = [atom.index for atom in self.topology.atoms]
            all_atom_residues = [atom.residue for atom in self.topology.atoms]
            all_atom_names = [atom.name for atom in self.topology.atoms]
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i])
                restraint_index = self.restraint_data[i,0]
                exp_chemical_shift        = self.restraint_data[i,1]
                model_chemical_shift      = model_data[i]
                r.add_line(restraint_index, a1, self.topology, exp_chemical_shift, model_chemical_shift)
            r.write('%s/%d.%s'%(self.out,j,self.scheme))

    def write_noe_input(self):
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])
            r = prep_noe()
            all_atom_indices = [atom.index for atom in self.topology.atoms]
            all_atom_residues = [atom.residue for atom in self.topology.atoms]
            all_atom_names = [atom.name for atom in self.topology.atoms]
            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                restraint_index = self.restraint_data[i,0]
                exp_noe        = self.restraint_data[i,1]
                model_noe      = model_data[i]
                r.add_line(restraint_index, a1, a2, self.topology, exp_noe, model_noe)
            r.write('%s/%d.%s'%(self.out,j,self.scheme))

    def write_J_input(self):
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])
            r = prep_J()
            all_atom_indices = [atom.index for atom in self.topology.atoms]
            all_atom_residues = [atom.residue for atom in self.topology.atoms]
            all_atom_names = [atom.name for atom in self.topology.atoms]
            for i in range(self.ind.shape[0]):
                a1, a2, a3, a4 = int(self.ind[i,0]), int(self.ind[i,1]), int(self.ind[i,2]), int(self.ind[i,3])
                restraint_index = self.restraint_data[i,0]
                exp_J_coupling      = self.restraint_data[i,1]
                model_J_coupling      = model_data[i]
                r.add_line(restraint_index, a1, a2, a3, a4, self.topology, exp_J_coupling, model_J_coupling)
#            r.add_line(restraint_index, a1, a2, a3, a4, self.topology, J_coupling, self.karplus)
            r.write('%s/%d.%s'%(self.out,j,self.scheme))

    def write_pf_input(self):
        if precomputed_pf:
            for j in range(len(self.data)):
                model_data = np.loadtxt(self.data[j])
                r = prep_pf()
                all_atom_indices = [atom.index for atom in self.topology.atoms]
                all_atom_residues = [atom.residue for atom in self.topology.atoms]
                all_atom_names = [atom.name for atom in self.topology.atoms]
                for i in range(self.ind.shape[0]):
                    a1 = int(self.ind[i])
                    restraint_index = self.restraint_data[i,0]
                    exp_pf          = self.restraint_data[i,1]
                    protectionfactor        = model_data[i]
                    r.add_line(restraint_index, a1, self.topology, exp_pf, protectionfactor)
                r.write('%s/%d.%s'%(self.out,j,self.scheme))
        else:
            for j in range(len(self.data)):
                r = prep_pf()
                all_atom_indices = [atom.index for atom in self.topology.atoms]
                all_atom_residues = [atom.residue for atom in self.topology.atoms]
                all_atom_names = [atom.name for atom in self.topology.atoms]
                for i in range(self.ind.shape[0]):
                    a1 = int(self.ind[i])
                    restraint_index = self.restraint_data[i,0]
                    exp_pf          = self.restraint_data[i,1]
                    r.add_line(restraint_index, a1, self.topology, exp_pf)
                r.write('%s/%d.%s'%(self.out,j,self.scheme))


#__all__ = [
#    'Preparation'

#]
