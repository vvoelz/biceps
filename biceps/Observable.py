# -*- coding: utf-8 -*-
import os, sys
import numpy as np

import os, sys, glob, string
import numpy as np
import mdtraj as md
import re
from biceps.prep_cs import prep_cs
from biceps.prep_J import prep_J
from biceps.prep_noe import prep_noe
from biceps.prep_pf import prep_pf
import biceps.toolbox


class NMR_Chemicalshift(object):
    """A data containter class to store a datum for NMR chemical shift information."""

    def __init__(self, i, exp, model):
        """Initialize the derived NMR_Chemicalshift class.

        :param int i: atom indices from the conformation defining this chemical shift
        :var exp: the experimental chemical shift
        :var model: the model chemical shift in this structure (in ppm)

        >>> biceps.Observable.NMR_Chemicalshift(i, exp, model)
        """

        # Atom indices from the conformation defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model = model

        # the experimental chemical shift
        self.exp = exp

        # N equivalent chemical shift should only get 1/N f the weight when
    #... computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0  #1.0/3.0 used in JCTC 2020 paper  # default is N=1



class NMR_Dihedral(object):
    """A data containter class to store a datum for NMR dihedral information."""

    def __init__(self, i, j, k, l, exp, model,
            equivalency_index=None, ambiguity_index=None):
        """Initialize NMR_Dihedral container class

        :param int i,j,k,l: atom indices from the conformation defining this dihedral
        :var exp: the experimental J-coupling constant
        :var model:  the model distance in this structure (in Angstroms)
        :var equivalency_index: the index of the equivalency group (i.e. a tag for equivalent H's)
        :var ambiguity_index: the index of the ambiguity group \n (i.e. some groups distances\
                have distant values, but ambiguous assignments.  Posterior sampling can be performed over these values)

        >>> biceps.Observable.NMR_Dihedral(i, j, k, l, exp,  model)
        """

        # Atom indices from the conformation defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l

        # the model distance in this structure (in Angstroms)
        self.model = model

        # the experimental J-coupling constant
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index


class NMR_Distance(object):
    """A class to store NMR noe information."""

    def __init__(self, i, j, exp, model, equivalency_index=None):
        """Initialize NMR_Distance container class

        :param int i,j: atom indices from the conformation defining this noe
        :var exp: the experimental NOE noe (in Angstroms)
        :var model: the model noe in this structure (in Angstroms)
        :var equivalency_index: the index of the equivalency group (i.e. a tag for equivalent H's)

        >>> biceps.Observable.NMR_Distance(i, j, exp,  model)
        """

        # Atom indices from the conformation defining this noe
        self.i = i
        self.j = j

        # the model noe in this structure (in Angstroms)
        self.model = model

        # the experimental NOE noe (in Angstroms)
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent noe should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1


class NMR_Protectionfactor(object):
    """A class to store NMR protection factor information."""

    def __init__(self, i, exp, model):
        """Initialize NMR_Protectionfactor container class

        :param int i: atom indices from the conformation defining this protection factor
        :var exp: the experimental protection factor
        :var model: the model protection factor in this structure


        >>> biceps.Observable.NMR_Protectionfactor(i, exp,  model)
        """

        # Atom indices from the conformation defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model = model

        # the experimental protection factor
        self.exp = exp

        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




class Preparation(object):
    """A parent class to prepare input files for BICePs calculation.

    :param str scheme: type of experimental observables {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
    :param int default=0 state: number of states
    :param str default=None indices: experimental observable index (*.txt file)
    :param str default=None exp_data: experimental measuremnets (*.txt file)
    :param str default=None top: topology file (*.pdb)
    :param str default=None data_dir: precomputed data directory (should have *txt file inside)
    """

    def __init__(self, scheme=None, states=0, indices=None, exp_data=None,
            top=None, data_dir=None, precomputed_pf=False):

        if scheme not in biceps.toolbox.list_possible_extensions():
            raise ValueError(f"Scheme must be one of {biceps.toolbox.list_possible_extensions()}")
        elif scheme == 'pf' and not precomputed_pf:
            if states==0.0 or indices == None or exp_data == None or top==None:
                raise ValueError("make sure you have actual input for states, indices, exp_data, topology file or data directory ")
        else:
            if states==0.0 or indices == None or exp_data == None or top==None or data_dir==None:
                raise ValueError("make sure you have actual input for states, indices, exp_data, topology file or data directory ")

        self.ind=np.loadtxt(indices)
        self.restraint_data = np.loadtxt(exp_data)
        self.scheme = scheme
        if self.ind.shape[0] != self.restraint_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.')%(self.ind.shape[0],self.restraint_data.shape[0])

        self.topology = md.load(top).topology
        self.convert = lambda txt: int(txt) if txt.isdigit() else txt
#        if scheme == 'pf' and not precomputed_pf:
#            self.data  = self.restraint_data
#        else:
        self.data = sorted(glob.glob(data_dir),key=lambda x: [self.convert(s) for s in re.split("([0-9]+)",x)])
        if int(len(self.data)) != int(states):
            raise ValueError("number of states doesn't equal to file numbers")


    def write(self, out_dir=None):
        """Write BICePs format input files for available experimental observables.

        :param str default=None out_dir: output directory
        """

        if self.scheme not in biceps.toolbox.list_possible_extensions():
            raise ValueError(f"scheme must be one of {biceps.toolbox.list_possible_extensions()}")

        if out_dir == None:
            self.out = 'BICePs_'+self.scheme
        else:
            self.out = out_dir

        biceps.toolbox.mkdir(self.out)
        write = getattr(self, "write_%s_input"%self.scheme)
        write()



    def write_cs_input(self):
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



'''
# Load in experimental restraint data
self.restraint_data = np.loadtxt(exp_data)

# Header for Pandas DataFrame
self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
        'atom_index2', 'res2', 'atom_name2', 'exp(A)', 'model(A)')
self.keys = [ self.header[i].split("(")[0] for i in range(len(self.header)) ]
self.data = { self.header[i]: [] for i in range(len(self.header)) }
self.comments = 0

if precomputed_data is not None:
    self.computed_distances = self.compute_distances()
else:
    self.load_raw_data()


## Load
#self.data = sorted(glob.glob(data_dir),key=lambda x: [self.convert(s) for s in re.split("([0-9]+)",x)])
#if int(len(self.data)) != int(len(states)):
#    raise ValueError("The number of states doesn't equal to number of files of input data.")

#self.precomputed = precomputed

'''




if __name__ == "__main__":

    import doctest
    doctest.testmod()




