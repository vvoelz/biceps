##############################################################################
# Authors: Vincent Voelz, Yunhui Ge, Rob Raddi
# This file stores experimental observables with derived container classes.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys
import numpy as np

##############################################################################
# Main
##############################################################################
class Observable(object):
    """A Parent class of observables """

    def __init__(self, i, model, exp,
            model_angle=None, j=None, k=None, l=None,
            equivalency_index=None, ambiguity_index=None):
        """Initialize the parent observable class"""

        # Atom indices from the Conformation() defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l

        # the model distance in this structure (in Angstroms)
        self.model = model
        self.model_angle = model_angle

        # the experimental J-coupling constant
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index



class NMR_Chemicalshift(Observable):
    """A data containter class to store a datum for NMR chemical shift information."""

    def __init__(self, i, model, exp):
        """Initialize the derived NMR_Chemicalshift class."""

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model = model

        # the experimental chemical shift
        self.exp = exp

        # N equivalent chemical shift should only get 1/N f the weight when
    #... computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1



class NMR_Dihedral(Observable):
    """A data containter class to store a datum for NMR dihedral information."""

    def __init__(self, i, j, k, l, model, exp, model_angle,
            equivalency_index=None, ambiguity_index=None):
        """Initialize NMR_Dihedral container class"""

        # Atom indices from the Conformation() defining this dihedral
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


class NMR_Distance(Observable):
    """A class to store NMR noe information."""

    def __init__(self, i, j, model, exp, equivalency_index=None):
        """Initialize NMR_Distance container class"""

        # Atom indices from the Conformation() defining this noe
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


class NMR_Protectionfactor(Observable):
    """A class to store NMR protection factor information."""

    def __init__(self, i, model, exp):
        """Initialize NMR_Protectionfactor container class"""

        # Atom indices from the Conformation() defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model = model

        # the experimental protection factor
        self.exp = exp

        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1


