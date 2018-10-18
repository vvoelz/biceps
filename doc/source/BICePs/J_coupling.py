# coding=utf-8
##############################################################################
# Authors: Rob Raddi
# Contributors: Yunhui Ge
# This file includes functions of computing J3 couplings and is modified based
# on the source code from MDTraj. Cite the original MDTraj paper to use this
# function.
##############################################################################

##############################################################################
# Imports
##############################################################################

import numpy as np

from mdtraj.geometry import compute_phi

##############################################################################
# Globals
##############################################################################


J3_HN_HA_coefficients = {  # See full citations below in docstring references.
    "Ruterjans1999": dict(phi0=-60 * np.pi/180., A=7.90, B=-1.05, C=0.65),  # From Table 1. in paper.
    "Bax2007": dict(phi0=-60 * np.pi/180., A=8.4, B=-1.36, C=0.33),         # From Table 1. in paper
    "Bax1997": dict(phi0=-60 * np.pi/180., A=7.09, B=-1.42, C=1.55),        # From Table 2. in paper
    "Habeck" :  dict(phi0=-60 * np.pi/180., A=7.13, B=1.31, C=1.56),        # From Table 1. in paper
    "Vuister" : dict(phi0=-60 * np.pi/180., A=6.51, B=-1.76, C=1.60),       # From Figure 4. in paper
    "Pardi"   : dict(phi0=-60 * np.pi/180., A=6.40, B=-1.40, C=1.90),       # From Figure 3. in paper
    }

J3_HN_HA_uncertainties = {
    # Values in [Hz]
    "Ruterjans1999": 0.25,
    "Bax2007": 0.36,
    "Bax1997": 0.39,
    "Habeck" : 0.34,
    "Vuister": 0.73,
    "Pardi"  : 0.76
}

##############################################################################
# Functions
##############################################################################

def _J3_function(phi, A, B, C, phi0):
    """Return a scalar couplings with a given choice of karplus coefficients.  USES RADIANS!"""
    return A * np.cos(phi + phi0) ** 2. + B * np.cos(phi + phi0) + C


def compute_J3_HN_HA(traj, model="Bax2007"):
    """Calculate the scalar coupling between HN and H_alpha.

    This function does not take into account periodic boundary conditions (it
    will give spurious results if the three atoms which make up any angle jump
    across a PBC (are not "wholed"))

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to compute J3_HN_HA for
    model : string, optional, default="Bax2007"
        Which scalar coupling model to use.  Must be one of Bax2007, Bax1999,
        or Ruterjans1999

    Returns
    -------
    indices : np.ndarray, shape=(n_phi, 4), dtype=int
        Atom indices (zero-based) of the phi dihedrals
    J : np.ndarray, shape=(n_frames, n_phi)
        Scalar couplings (J3_HN_HA, in [Hz]) of this trajectory.
        `J[k]` corresponds to the phi dihedral associated with
        atoms `indices[k]`

    Notes
    -----
    The coefficients are taken from the references below--please cite them.

    References
    ----------
    """
    indices, phi = compute_phi(traj)

    if model not in J3_HN_HA_coefficients:
        raise(KeyError("model must be one of %s" % J3_HN_HA_coefficients.keys()))

    J = _J3_function(phi, **J3_HN_HA_coefficients[model])
    return indices, J



