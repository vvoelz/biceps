# -*- coding: utf-8 -*-
import numpy as np
from mdtraj.geometry import compute_phi

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


def _J3_function(phi, A, B, C, phi0):
    """Return a scalar couplings with a given choice of karplus coefficients.

    :param float phi:
    :param float A:
    :param float B:
    :param float C:
    :param float phi0:

    .. warning:: in radians"""

    return A * np.cos(phi + phi0) ** 2. + B * np.cos(phi + phi0) + C


def compute_J3_HN_HA(traj, model="Bax2007"):
    """Calculate the scalar coupling between HN and H_alpha.
    This function does not take into account periodic boundary conditions (it
    will give spurious results if the three atoms which make up any angle jump
    across a PBC (are not "wholed"))

    :param  mdtraj.Trajectory traj: Trajectory to compute J3_HN_HA for
    :param  string, optional, default="Bax2007" model :
      Which scalar coupling model to use.  Must be one of Bax2007, Bax1999, or Ruterjans1999

    :return np.ndarray, shape=(n_phi, 4), dtype=int indices :
        Atom indices (zero-based) of the phi dihedrals
    :return  np.ndarray, shape=(n_frames, n_phi) J:
        Scalar couplings (J3_HN_HA, in [Hz]) of this trajectory.
        `J[k]` corresponds to the phi dihedral associated with
        atoms `indices[k]`

    Notes
    -----
    The coefficients are taken from the references below--please cite them.

    References
    ----------
    .. [1] Schmidt, J. M., Blümel, M., Löhr, F., & Rüterjans, H.
       "Self-consistent 3J coupling analysis for the joint calibration
       of Karplus coefficients and evaluation of torsion angles."
       J. Biomol. NMR, 14, 1 1-12 (1999)

    .. [2] Vögeli, B., Ying, J., Grishaev, A., & Bax, A.
       "Limits on variations in protein backbone dynamics from precise
       measurements of scalar couplings."
       J. Am. Chem. Soc., 129(30), 9377-9385 (2007)

    .. [3] Hu, J. S., & Bax, A.
       "Determination of ϕ and ξ1 Angles in Proteins from 13C-13C
       Three-Bond J Couplings Measured by Three-Dimensional Heteronuclear NMR.
       How Planar Is the Peptide Bond?."
       J. Am. Chem. Soc., 119(27), 6360-6368 (1997)

    .. [4] Habeck, M.; Rieping, W.; Nilges, M. Bayesian Estimation of Karplus
    Parameters and Torsion Angles from Three-Bond Scalar Couplings Constants.
    J. Magn. Reson. 2005, 177 (1), 160–165.

    .. [5] Vuister, G. W.; Bax, A. Quantitative J Correlation: A New Approach for
    Measuring Homonuclear Three-Bond JHN,Hα Coupling Constants in 15N-Enriched
    Proteins. J. Am. Chem. Soc. 1993, 115 (17), 7772–7777.

    .. [6] Pardi, A.; Billeter, M.; Wüthrich, K. Calibration of the Angular
    Dependence of the Amide Proton-C Alpha Proton Coupling Constants,
    3JHN Alpha, in a Globular Protein. Use of 3JHN Alpha for Identification
    of Helical Secondary Structure. J. Mol. Biol. 1984, 180 (3), 741–751.


    """
    indices, phi = compute_phi(traj)

    if model not in J3_HN_HA_coefficients:
        raise KeyError

    J = _J3_function(phi, **J3_HN_HA_coefficients[model])
    return indices, J



