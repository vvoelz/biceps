# -*- coding: utf-8 -*-
import numpy as np
import string
from mdtraj.geometry import compute_phi


J3_HN_HA_coefficients = {  # See full citations below in docstring references.
    "Ruterjans1999": dict(phi0=-60 * np.pi/180., A=7.90, B=-1.05, C=0.65),  # From Table 1. in paper.
    "Bax2007": dict(phi0=-60 * np.pi/180., A=8.4, B=-1.36, C=0.33),         # From Table 1. in paper
    "Bax1997": dict(phi0=-60 * np.pi/180., A=7.09, B=-1.42, C=1.55),        # From 1d3z
    "Bax1995": dict(phi0=-60 * np.pi/180., A=6.98, B=-1.38, C=1.72),        # From Table 1. in paper
    "Habeck" :  dict(phi0=-60 * np.pi/180., A=7.13, B=-1.31, C=1.56),        # From Table 1. in paper
    "Vuister" : dict(phi0=-60 * np.pi/180., A=6.51, B=-1.76, C=1.60),       # From Figure 4. in paper
    "Pardi"   : dict(phi0=-60 * np.pi/180., A=6.40, B=-1.40, C=1.90),       # From Figure 3. in paper
    "Raddi2024": dict(phi0=-60 * np.pi/180., A=6.97, B=-1.49, C=1.63),
    "Kessler1998": dict(phi0=-60 * np.pi/180., A=9.4, B=-1.1, C=0.4),
    "NRV2024": dict(phi0=-60 * np.pi/180., A=7.3, B=-2.3, C=1.6),
    }


J3_HN_HA_uncertainties = {
    # Values in [Hz]
    "Ruterjans1999": 0.25,
    "Bax2007": 0.36,
    "Bax1997": 0.39,
    "Bax1995": np.sqrt((0.04**2 + 0.04**2 + 0.03**2)),
    "Habeck" : 0.34,
    "Vuister": 0.73,
    "Pardi"  : 0.76,
    "Raddi2024": np.sqrt((0.07**2 + 0.04**2 + 0.05**2)),
    "Kessler1998": 0.0,
    "NRV2024": np.sqrt((0.39**2 + 0.18**2 + 0.25**2)),
}


J3_HN_CB_coefficients = {  # See full citations below in docstring references.
    "Habeck" : dict(phi0=+60 * np.pi/180., A=3.26, B=-0.87, C=0.10),  # From Table 1. in paper
    "Bax2007": dict(phi0=+60 * np.pi/180., A=3.71, B=-0.59, C=0.08),  # From Table 1. in paper
    "Bax1995": dict(phi0=+60 * np.pi/180., A=3.39, B=-0.94, C=0.07),  # From Table 1. in paper
    "Bax1997": dict(phi0=+60 * np.pi/180., A=3.06, B=-0.74, C=0.13),  # From 1d3z
    }

#coefficients 3.06 -0.74 0.13 60

J3_HN_CB_uncertainties = {
    # Values in [Hz]
    "Habeck": np.sqrt((0.23**2 + 0.24**2 + 0.08**2)),
    "Bax2007": 0.22,
    "Bax1995": np.sqrt((0.07**2 + 0.08**2 + 0.03**2)),
    "Bax1997": 0.21,
}

J3_HN_C_coefficients = {  # See full citations below in docstring references.
                        # NOTE: (phi0 + pi, A, -B, C) = (phi0, A, B, C)
    #"Habeck" : dict(phi0=0 * np.pi/180., A=4.19, B=0.99, C=0.03),  # From Table 1. in paper
    #"Bax2007": dict(phi0=0 * np.pi/180., A=4.36, B=1.08, C=-0.01),  # From Table 1. in paper
    #"Bax1995": dict(phi0=0 * np.pi/180., A=4.32, B=0.84, C=0.00),  # From Table 1. in paper

    "Habeck" : dict(phi0=+180 * np.pi/180., A=4.19, B=-0.99, C=0.03),  # From Table 1. in paper
    "Bax2007": dict(phi0=+180 * np.pi/180., A=4.36, B=-1.08, C=-0.01),  # From Table 1. in paper
    "Bax1995": dict(phi0=+180 * np.pi/180., A=4.32, B=-0.84, C=0.00),  # From Table 1. in paper
    "Bax1997": dict(phi0=+180 * np.pi/180., A=4.29, B=-1.01, C=0.00),  # From 1d3z
    }

J3_HN_C_uncertainties = {
    # Values in [Hz]
    "Habeck": np.sqrt((0.30**2 + 0.18**2 + 0.05**2)),
    "Bax2007": 0.30,
    "Bax1995": np.sqrt((0.08**2 + 0.03**2 + 0.02**2)),
    "Bax1997": 0.32,
}

J3_HA_C_coefficients = {  # See full citations below in docstring references.
    #"Habeck" : dict(phi0=-60 * np.pi/180., A=3.84, B=2.19, C=1.20),  # From Table 1. in paper
    #"Bax1995": dict(phi0=-60 * np.pi/180., A=3.75, B=2.19, C=1.28),  # From Table 1. in paper
    "Habeck" : dict(phi0=120 * np.pi/180., A=3.84, B=-2.19, C=1.20),  # From Table 1. in paper
    "Bax1995": dict(phi0=120 * np.pi/180., A=3.75, B=-2.19, C=1.28),  # From Table 1. in paper
    "Bax1997": dict(phi0=120 * np.pi/180., A=3.72, B=-2.18, C=1.28),  # From 1d3z
    }

J3_HA_C_uncertainties = {
    # Values in [Hz]
    "Habeck": np.sqrt((0.14**2 + 0.10**2 + 0.11**2)),
    "Bax1995": np.sqrt((0.05**2 + 0.06**2 + 0.03**2)),
    "Bax1997": 0.24,
}

J3_C_C_coefficients = {  # See full citations below in docstring references.
    "Habeck" : dict(phi0=0 * np.pi/180., A=1.30, B=-0.93, C=0.64),  # From Table 1. in paper
    "Bax1997": dict(phi0=0 * np.pi/180., A=1.36, B=-0.93, C=0.60),  # From 1d3z
    }

J3_C_C_uncertainties = {
    # Values in [Hz]
    "Habeck": np.sqrt((0.12**2 + 0.06**2 + 0.03**2)),
    "Bax1997": 0.13,
}

J3_C_CB_coefficients = {  # See full citations below in docstring references.
    #"Habeck" : dict(phi0=60 * np.pi/180., A=2.52, B=-0.49, C=0.51),  # From Table 1. in paper
    "Habeck" : dict(phi0=60 * np.pi/180., A=2.52, B=0.49, C=0.51),  # From Table 1. in paper
    #"Bax1997": dict(phi0=240 * np.pi/180., A=1.74, B=-0.57, C=0.25),  # From 1d3z
    "Bax1997": dict(phi0=60 * np.pi/180., A=1.74, B=0.57, C=0.25),  # From 1d3z
    #"Habeck" : dict(phi0=240 * np.pi/180., A=2.52, B=-0.49, C=0.51),  # From Table 1. in paper
    }

J3_C_CB_uncertainties = {
    # Values in [Hz]
    "Habeck": np.sqrt((0.14**2 + 0.10**2 + 0.11**2)),
    "Bax1997": 0.16,
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


# $3^J(H^{N} H^{\alpha})$,
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




#$3^J(H^{N} C^{\prime})$
def compute_J3_HN_C(traj, model="Bax2007"):
    """Calculate the scalar coupling between HN and C_prime.

    This function does not take into account periodic boundary conditions (it
    will give spurious results if the three atoms which make up any angle jump
    across a PBC (are not "wholed"))

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to compute J3_HN_C for
    model : string, optional, default="Bax2007"
        Which scalar coupling model to use.  Must be one of Bax2007

    Returns
    -------
    indices : np.ndarray, shape=(n_phi, 4), dtype=int
        Atom indices (zero-based) of the phi dihedrals
    J : np.ndarray, shape=(n_frames, n_phi)
        Scalar couplings (J3_HN_C, in [Hz]) of this trajectory.
        `J[k]` corresponds to the phi dihedral associated with
        atoms `indices[k]`

    Notes
    -----
    The coefficients are taken from the references below--please cite them.

    References
    ----------
    .. [1] Hu, J. S., & Bax, A.
       "Determination of ϕ and ξ1 Angles in Proteins from 13C-13C
       Three-Bond J Couplings Measured by Three-Dimensional Heteronuclear NMR.
       How Planar Is the Peptide Bond?."
       J. Am. Chem. Soc., 119(27), 6360-6368 (1997)

    """
    indices, phi = compute_phi(traj)

    if model not in J3_HN_C_coefficients:
        raise(KeyError("model must be one of %s" % J3_HN_C_coefficients.keys()))

    J = _J3_function(phi, **J3_HN_C_coefficients[model])
    return indices, J



#  $3^J(H^{N} C^{\beta})$,
def compute_J3_HN_CB(traj, model="Bax2007"):
    """Calculate the scalar coupling between HN and C_beta.

    This function does not take into account periodic boundary conditions (it
    will give spurious results if the three atoms which make up any angle jump
    across a PBC (are not "wholed"))

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to compute J3_HN_CB for
    model : string, optional, default="Bax2007"
        Which scalar coupling model to use.  Must be one of Bax2007

    Returns
    -------
    indices : np.ndarray, shape=(n_phi, 4), dtype=int
        Atom indices (zero-based) of the phi dihedrals
    J : np.ndarray, shape=(n_frames, n_phi)
        Scalar couplings (J3_HN_CB, in [Hz]) of this trajectory.
        `J[k]` corresponds to the phi dihedral associated with
        atoms `indices[k]`

    Notes
    -----
    The coefficients are taken from the references below--please cite them.

    References
    ----------
    .. [1] Hu, J. S., & Bax, A.
       "Determination of ϕ and ξ1 Angles in Proteins from 13C-13C
       Three-Bond J Couplings Measured by Three-Dimensional Heteronuclear NMR.
       How Planar Is the Peptide Bond?."
       J. Am. Chem. Soc., 119(27), 6360-6368 (1997)

    """
    indices, phi = compute_phi(traj)

    if model not in J3_HN_CB_coefficients:
        raise(KeyError("model must be one of %s" % J3_HN_CB_coefficients.keys()))

    J = _J3_function(phi, **J3_HN_CB_coefficients[model])
    return indices, J


#$3^J(H^{\alpha} C^{\prime})$
def compute_J3_HA_C(traj, model="Habeck"):
    """
    """
    indices, phi = compute_phi(traj)

    if model not in J3_HA_C_coefficients:
        raise(KeyError("model must be one of %s" % J3_HA_C_coefficients.keys()))

    J = _J3_function(phi, **J3_HA_C_coefficients[model])
    return indices, J

#$3^J(C^{\prime} C^{\prime})$
def compute_J3_C_C(traj, model="Habeck"):
    """
    """
    indices, phi = compute_phi(traj)

    if model not in J3_C_C_coefficients:
        raise(KeyError("model must be one of %s" % J3_C_C_coefficients.keys()))

    J = _J3_function(phi, **J3_C_C_coefficients[model])
    return indices, J

#$3^J(C^{\prime} C^{\beta})$
def compute_J3_C_CB(traj, model="Habeck"):
    """
    """
    indices, phi = compute_phi(traj)

    if model not in J3_C_CB_coefficients:
        raise(KeyError("model must be one of %s" % J3_C_CB_coefficients.keys()))

    J = _J3_function(phi, **J3_C_CB_coefficients[model])
    return indices, J




# $3^J(H^{N} H^{\alpha})$, $3^J(H^{\alpha} C^{\prime})$, $3^J(H^{N} C^{\beta})$,  $3^J(H^{N} C^{\prime})$


if __name__ == '__main__':


    from matplotlib import pyplot as plt
    import pandas as pd
    import string
    # Perform a scan of all the Karplus relations
    models = ["Ruterjans1999","Bax2007","Bax1997","Habeck","Vuister" ,"Pardi"]
    phis = np.linspace(-180, 180, 1000)
    phi0 = J3_HN_HA_coefficients[models[0]]["phi0"]
    #phi0 = 0.0

    results = []
    for model in models:

        items = {key:J3_HN_HA_coefficients[model][key] for key in ["A", "B", "C"]}
        #Calculate the scalar coupling between HN and H_alpha.
        #scalar_couplings = _J3_function(np.deg2rad(phis), **J3_HN_HA_coefficients[model])
        scalar_couplings = _J3_function(np.deg2rad(phis), **items, phi0=phi0)
        results.append({"model":model, "scalar_couplings":scalar_couplings, "phis":phis})


    df = pd.DataFrame(results)
    print(df)

    colors = ["blue", "purple", "red", "green", "grey", "orange", "yellow"]
    markers = ["o", "v", "s", "p", "+", "*", "x"]
    fig, ax = plt.subplots()
    for i in range(len(df["model"].to_numpy())):
        row = df.iloc[[i]]
        ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0],
                label=row["model"].to_numpy()[0], color=colors[i], marker=markers[i], ms=4, markevery=slice(0, None, 25))#, ls="dotted", lw=4)
        #ax.scatter(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label=row["model"].to_numpy(), marker="o")

    fig.set_size_inches(8, 4)  # Set the figure size to 8x4 inches
    ax.set_xlim(phis[0], phis[-1])
    yticks = list(range(14)[::2])
    ax.set_yticks(yticks)
    ax.set_ylim(yticks[0]-1, yticks[-1]+1)
    ax.set_xticks([phis[0], -90, 0, 90, phis[-1]])
    label_fontsize = 14
    #ax.legend(loc="best", fontsize=8)
    ax.legend(loc='center left', bbox_to_anchor=(1.025, 0.5), fontsize=label_fontsize)
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=18)
    ax.set_ylabel('${^{3}\!J}_{H^{N}H^{\\alpha}}$ (Hz)', fontsize=18)

    ## Add content to the subplots (example)
    for i, ax in enumerate([ax]):
        x,y = -0.1, 1.02
        #ax.text(x,y, string.ascii_lowercase[i], transform=ax.transAxes,
        #        size=14, weight='bold')

        # Setting the ticks and tick marks
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(label_fontsize)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=label_fontsize)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)

    #fig.tight_layout()
    plt.gcf().subplots_adjust(left=0.125, bottom=0.175, top=0.95, right=0.675, wspace=0.20, hspace=0.5)
    fig.savefig("karplus_relation.png")








