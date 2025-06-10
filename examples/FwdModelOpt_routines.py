
# Python libraries:{{{
import gc, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy, scipy.stats, copy, matplotlib
import string, re, sys, pickle
import warnings
import biceps
#from biceps.J_coupling import *
#from biceps.KarplusRelation import KarplusRelation

import mdtraj as md
from mdtraj.geometry import compute_phi

import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('axes', labelsize=16)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.colors import Normalize

#from biceps.PosteriorSampler import get_negloglikelihood
from sklearn import metrics
import warnings, time
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

from scipy.optimize import fmin
from scipy.linalg import lu
from numpy.linalg import inv

import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath

import matplotlib.gridspec as gridspec

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from itertools import combinations
import jax
import jax.numpy as jnp
from matplotlib import ticker
try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_dihedrals
except(Exception) as e:
    print("WARNING: You don't have MDAnalysis installed. Some functions will throw errors")


#:}}}

# Methods:{{{
def get_scalar_couplings_with_derivatives(phi, A, B, C, phi0):
    """Return a scalar couplings with a given choice of karplus coefficients.  USES RADIANS!"""

    #angle_in_rad = np.deg2rad(phi + phi0)
    J = A * np.cos(phi + phi0) ** 2. + B * np.cos(phi + phi0) + C
    dJ = np.array([np.cos(phi + phi0) ** 2., np.cos(phi + phi0), np.ones(J.shape)])
    d2J = np.zeros(dJ.shape)
    return J, dJ, d2J

def get_forward_model_derivatives(phi, A, B, C, phi0):
    """Return a scalar couplings with a given choice of karplus coefficients.  USES RADIANS!"""
    #dgdphi = -2*A*np.sin(phi + phi0)*np.cos(phi + phi0) - B*np.sin(phi + phi0)
    dgAdphi = np.array(-2*(np.ones(phi.shape)*A)*np.sin(phi + np.ones(phi.shape)*phi0)*np.cos(phi + np.ones(phi.shape)*phi0))
    dgBdphi = np.array(- B*np.sin(phi + phi0))
    dgCdphi = np.zeros(phi.shape)
    dgdphi = np.array([dgAdphi, dgBdphi, dgCdphi])
    return dgdphi



def generate_mixed_phi_angles(angles=(-90, 60), sigmas=(25, 5), weights=(0.75, 0.25),
                              nsamples=10000, dist_size=10000):

    # Validate that the weights sum to 1
    assert np.isclose(sum(weights), 1.0), "Weights should sum to 1"

    # Create empty list to store mixed angles
    mixed_angles = []

    # Generate the distributions
    distributions = []
    for i, angle in enumerate(angles):
        mean, std_dev = np.deg2rad(angle), np.deg2rad(sigmas[i])
        distributions.append(np.random.normal(mean, std_dev, dist_size))

    # Determine the number of samples to draw from each distribution
    samples_per_distribution = [int(weight * nsamples) for weight in weights]

    # Draw the samples
    for i, dist in enumerate(distributions):
        sampled_angles = np.random.choice(dist, samples_per_distribution[i], replace=False)
        mixed_angles.extend(sampled_angles)
    return np.array(mixed_angles)

def generate_phi_angles(nstates, Nd, angles=(-90, 0), sigmas=(0.1, 0.25),
                        dist_size=10000, weights=(0.75, 0.25), nsamples=1000):

    mixed_angles = generate_mixed_phi_angles(angles, sigmas,
                     dist_size=dist_size, weights=weights, nsamples=nsamples)

    # Extract Nd[0] random phi angles for each state
    phi_angles = [np.random.choice(mixed_angles, Nd, replace=True) for state in range(nstates)]
    return np.array(phi_angles) # in radians





## von Mises toy model:{{{
#def generate_toy_model(nstates, Nd, A=8.4, B=-1.3, C=0.4, phi0_deg=-60.0, verbose=False):
#    """
#    Generate a toy model to test J-coupling constants.
#
#    :param nstates: Number of states.
#    :param Nd: Number of J-coupling values for each state.
#    :param A: Karplus parameter A.
#    :param B: Karplus parameter B.
#    :param C: Karplus parameter C.
#    :param phi0_deg: phi0 angle in degrees.
#    :return: Tuple containing the true experimental J-couplings and a list of dictionaries representing the states.
#    """
#    from scipy.stats import vonmises
#
#    # Convert phi0 to radians
#    phi0 = np.deg2rad(phi0_deg)
#
#    states = [] # Initialize states list
#
#    # NOTE: Creating distribution of phi angles to sample from
#    sigmas=(20, 10, 5)
#    angles=(-110, -60, 60)
#    weights=(0.425, 0.55, 0.025)
#    phi_angles = generate_phi_angles(nstates, Nd,
#                    angles=angles, sigmas=sigmas, weights=weights,
#                    dist_size=10000,  nsamples=1000)
#
#    kappa = 1.  # Forcing all states to have the same kappa
##    kappa = 5.  # Forcing all states to have the same kappa
#    mu = np.deg2rad(-60.) # saying that alpha helix is lowest energy
#    # For each state, sample angles using the Von Mises distribution and calculate energies
#    for state in range(nstates):
#        # Extract the phi angles for the current state
#        state_phi_angles = phi_angles[state]
#        # Define the mu and kappa for the Von Mises distribution based on the state
##        mu = state_phi_angles.mean()
#
#        # Calculate energies using the Von Mises distribution
#        P = vonmises.pdf(state_phi_angles, kappa, loc=mu) #np.deg2rad(phi0_deg))
#        energy = -np.log(np.prod(P))
#        # Add the most stable state to the states list
#        states.append({'phi': state_phi_angles, 'energy': energy, 'mu':np.rad2deg(state_phi_angles.mean())})
#
#    min_energy = min(state['energy'] for state in states)
#    for state in states:
#        # Normalize energies and compute populations
#        total_population = sum(np.exp(-(state['energy'] - min_energy)) for state in states)
#        for state in states:
#            state['population'] = np.exp(-(state['energy'] - min_energy)) / total_population
#
#    # Compute J-couplings for each state using the Karplus relation
#    for state in states:
#        state['J'], state['diff_J'], state['diff2_J'] = get_scalar_couplings_with_derivatives(state['phi'], A, B, C, phi0)
#
#    # Calculate the "True" experimental J-couplings by weighting each state's J-couplings by its population
#    true_J = np.array([state["population"] * state["J"] for state in states]).sum(axis=0)
#    if verbose: print(pd.DataFrame(states))
#    return true_J, states
## }}}
#

## generate_toy_model_old:{{{
#def generate_toy_model(nstates, Nd, A=8.4, B=-1.3, C=0.4, phi0_deg=-60.0):
#    """
#    Generate a toy model to test J-coupling constants.
#
#    :param nstates: Number of states.
#    :param Nd: Number of J-coupling values for each state.
#    :param A: Karplus parameter A.
#    :param B: Karplus parameter B.
#    :param C: Karplus parameter C.
#    :param phi0_deg: phi0 angle in degrees.
#    :return: Tuple containing the true experimental J-couplings and a list of dictionaries representing the states.
#    """
#    from scipy.stats import vonmises
#
#    # Convert phi0 to radians
#    phi0 = np.deg2rad(phi0_deg)
#
#    # Initialize states list
#    states = []
#
#    # Define the range for phi angles
#    #left, right = np.deg2rad(-120), np.deg2rad(0)
#    left, right = np.deg2rad(-140), np.deg2rad(-20)
#
#    # For the most stable state, set mu to -60° and choose a high kappa value
#    mu = np.deg2rad(-60)
#    #kappa = np.random.uniform(5, 10)  # You may need to adjust the range
#    kappa = 20
#
#    # Generate phi angles from a Von Mises distribution for the most stable state
#    phi_angles = vonmises.rvs(kappa, loc=mu, size=Nd)
#    #phi_angles = np.clip(phi_angles, left, right)
#
#    # Compute energies based on the Von Mises distribution PDF for the most stable state
#    energy = -np.log(vonmises.pdf(phi_angles, kappa, loc=mu)).sum()
#
#    # Add the most stable state to the states list
#    states.append({'phi': phi_angles, 'energy': energy})
#
#    # Generate the rest of the states with random mu in the range [left, right] and random kappa
#    for _ in range(nstates - 1):
#        #mu_ = np.random.uniform(left, right)
#        mu_ = np.random.normal(loc=np.deg2rad(-90), scale=np.deg2rad(20), size=1)
#        kappa = np.random.uniform(18, 19)
#
#        phi_angles = vonmises.rvs(kappa, loc=mu_, size=Nd)
#        #phi_angles = np.clip(phi_angles, left, right)
#
#        # Compute energies based on the Von Mises distribution PDF
#        energy = -np.log(vonmises.pdf(phi_angles, kappa, loc=mu)).sum()
#
#        states.append({'phi': phi_angles, 'energy': energy})
#
#
#    # Normalize energies and compute populations
#    min_energy = min(state['energy'] for state in states)
#    total_population = sum(np.exp(-(state['energy'] - min_energy)) for state in states)
#    for state in states:
#        state['population'] = np.exp(-(state['energy'] - min_energy)) / total_population
#
#    # Compute J-couplings for each state using the Karplus relation
#    for state in states:
#        state['J'], state['diff_J'], state['diff2_J'] = get_scalar_couplings_with_derivatives(state['phi'], A, B, C, phi0)
#
#    # Calculate the "True" experimental J-couplings by weighting each state's J-couplings by its population
#    true_J = np.array([state["population"] * state["J"] for state in states]).sum(axis=0)
#
#    return true_J, states
## }}}
#


def get_boltzmann_weighted_states(nstates, σ=0.16, loc=0.06,
                scale=1.0, verbose=False, plot=True, domain=None):
    """Get Boltzmann weighted states and perturbed states with error equal to σ.

    Args:
        nstates(int): number of states
        σ(float): sigma is the error in the prior

    Returns:
        botlzmann weighted states and perturbed states inside tuple \
                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
    """
    import scipy.stats as stats
    iter = True

    kT = 1#0.5959 # kcal/mol
    pops = np.random.random(nstates)
    pops /= pops.sum()
    energies = -kT*np.log(pops)
    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
    df = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
    while iter == True:
        samples = df.copy()
        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
        #perturbed_w = w+σ*np.random.randn(len(w))
        perturbed_w = w+np.random.normal(loc=loc, scale=σ, size=len(w))
        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
        perturbed_w /= perturbed_w.sum()
        perturbed_E = -kT*np.log(perturbed_w)
        if all(i >= 0.0 for i in perturbed_w): iter = False
        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
        except(Exception) as e: iter = True
    if σ == 0.0:
        perturbed_E, perturbed_w = E.copy(),w.copy()
        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
        except(Exception) as e: iter = True
    if verbose:
        print(f"Populations:           {w}")
        print(f"Perturbed Populations: {perturbed_w}")
        print(f"Energies:              {E}")
        print(f"Perturbed Energies:    {perturbed_E}")
        print(f"RMSE in populations:   {RMSE_pops}")
        print(f"RMSE in energies:      {RMSE_E}")
        #print(f"RMSE in prior:      {RMSE_E}")

    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]


def get_noe_data(weights, Nd, μ_data, σ_data, verbose=True):

    x = np.random.uniform(low=1.0, high=10.0, size=(len(weights),Nd))
    exp = np.array([w*x[i] for i,w in enumerate(weights)]).sum(axis=0)
    _exp = exp.copy()
    if (μ_data or σ_data) != 0.0:
        if verbose: print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
        if σ_data > 0.0:
            exp += σ_data*np.random.randn(len(exp))
        if μ_data > 0.0:
            offset = np.random.uniform(3.0, 5.0, len(exp))
            for i in range(len(exp)):
                if np.random.random() <= 0.30:
                    exp[i] = exp[i] + offset[i]
    diff = abs(_exp - exp)
    return x,exp,diff


def write_inputs_to_log(outfile="inputs.log"):
    glo = globals()
    keys = np.array(list(glo.keys()))
    index = int(np.where("generate_new_configs" == keys)[0])
    keys = np.array(list(glo.keys()))[index:]
    inputs = []
    for key in keys:
        inputs.append(f"{key} = {glo[key]}")
    inputs = np.array(inputs, dtype=np.str)
    np.savetxt(outfile, inputs, fmt='%s')

def matprint(mat, fmt="g"):
    """
    https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


# }}}

# add_error_to_data:{{{
def add_error_to_data(x, μ_data=dict(domain=(3.0,5.0),frac_of_data=0.0),
            σ_data=0.0, indices=None, verbose=True):
    """Add random and/or systemtic error to experimental data points
    """
    if indices == None: indices = list(range(len(x)))
    exp = x.copy()
    _exp = exp.copy()
    if (μ_data["frac_of_data"] or σ_data) != 0.0:
        if verbose: print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
        if σ_data > 0.0:
            for i in range(len(exp)):
                if i in indices:
                    exp[i] += σ_data*np.random.randn(1)
        if μ_data["frac_of_data"] > 0.0:
            offset = np.random.uniform(μ_data["domain"][0], μ_data["domain"][1], len(exp))
            for i in range(len(exp)):
                if i in indices:
                    if np.random.random() <= μ_data["frac_of_data"]:
                        exp[i] = exp[i] + offset[i]
                        #exp[i] = exp[i] - offset[i]
    diff = abs(_exp - exp)
    sse = 0
    for dev in diff: sse += dev*dev
    sigma = np.sqrt(sse/len(diff))
    return x,exp,diff,sigma

# }}}

# Convergence_check:{{{
def convergence(parameters_old, parameters_new, threshold=0.05, method="relative-change", verbose=True):
    """
    Checks convergence based on given method: "L2-norm", "relative-change", or "avg-change".
    """
    parameters_old = np.array(parameters_old)
    parameters_new = np.array(parameters_new)
    diff = parameters_old - parameters_new
    relative_change = np.abs(diff / parameters_old)

    if method == "L2-norm":
        measure = np.sqrt(np.sum(relative_change**2))
    elif method == "relative-change":
        measure = np.sum(relative_change)
    elif method == "avg-change":
        measure = np.mean(relative_change)
    else:
        raise ValueError("Invalid method. Choose from 'L2-norm', 'relative-change', or 'avg-change'.")

    if measure > threshold:
        if verbose:
            print(f"Convergence check ({method}): {measure:.5f} > (tol={threshold:.5f})")
        return False
    if verbose:
        print(f"Convergence check ({method}): {measure:.5f} < (tol={threshold:.5f})")
    return True
#:}}}

# Run BICePs:{{{
def run_biceps(ensemble, PSkwargs, sample_kwargs):
    sampler = biceps.PosteriorSampler(ensemble, **PSkwargs)
    sampler.sample(**sample_kwargs)
    return sampler
# }}}

# biceps_TI:{{{
def _xi_integration(ensemble, PSkwargs, sample_kwargs, plot_overlap=0, figname="BS_overlap.png", max_attempts=2, progress=True, scale_energies=False, save_sampler_obj=0):

    debug = 1
    global sampler
    good_overlap = True
    threshold = 0.02
    rerun = True
    trial = 0
    #max_attempts = 2
    while rerun == True:
        good_overlap = True
        trial += 1

        ## We need to tell the PosteriorSampler class to expect the forward model optimization protocol
        #sampler = biceps.PosteriorSampler(ensemble, int(nreplicas), bool(xi_integration))
        #sampler.sample(int(nsteps), int(burn))
        #mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=0, plot_overlap=plot_overlap, filename=figname)

        sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
        #xi_values,_ = sampler.ti_info
        xi_values = sampler.xi_schedule
        #print(xi_values)
        #exit()

        #print("done sampling.")
        #BS = sampler.get_results(scores_only=1, progress=0, compute_derivative=0, k_indices=[1,2])
        try:
            mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=progress, compute_derivative=0, plot_overlap=plot_overlap, filename=figname, scale_energies=scale_energies)
        #    print("done getting mbar obj")
            overlap = mbar.compute_overlap()
            overlap_matrix = overlap["matrix"]
        except(Exception) as e:
            sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
            mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=progress, compute_derivative=0, plot_overlap=plot_overlap, filename=figname, scale_energies=scale_energies)
            overlap = mbar.compute_overlap()
            overlap_matrix = overlap["matrix"]

        # overlap_matrix is a square matrix
        n = len(overlap_matrix)
        for i in range(n - 1):  # Iterate through rows, except the last row
            # Check the element to the right of the diagonal
            if overlap_matrix[i][i + 1] <= threshold:
                good_overlap = False
            # Check the element below the diagonal
            if overlap_matrix[i + 1][i] <= threshold:
                good_overlap = False
            if debug: print(f"O_({i},{i+1}), O_({i+1},{i}) = {overlap_matrix[i][i + 1]}, {overlap_matrix[i + 1][i]}")

        if not good_overlap: print("WARNING: Overlap matrix issues...trying again")
        else:
            rerun = False
            #BS = (BS + -mbar.f_k[-1])
            BS = np.array([-mbar.f_k[-1]])
            nreplicas = sampler.nreplicas
            BS = BS/nreplicas
            if BS[-1] == 0.0:
                BS = jnp.array(BS)
                BS.at[-1].set(jnp.nan)

        if trial == max_attempts:
            print("WARNING: Still overlap matrix issues...\nNo more attempts...\nYou might want to consider optimizing xi-values.")
            #BS = np.array([-mbar.f_k[-1]])
            BS = jnp.array([jnp.nan])
            rerun = False
    if save_sampler_obj:
        append_name, ext = os.path.splitext(os.path.basename(figname))
        dir = os.path.abspath(figname)
        name = dir.replace(os.path.basename(figname), "")
        biceps.toolbox.save_object(sampler, os.path.join(name,f"sampler_{append_name}.pkl"))
    return BS[-1]


def xi_integration(ensemble, PSkwargs, sample_kwargs, plot_overlap=False, outdir=None,
                   optimize_xi_values=True, optimize_nXis=True, xi_opt_steps=2000000, tol=1e-7, alpha=1e-5,
                   max_attempts=2, print_every=1000, progress=True, scale_energies=False, save_sampler_obj=False, verbose=False):

    PSkwargs["xi_integration"] = True
    if plot_overlap and optimize_xi_values: make_plots = 1
    else: make_plots = 0

    figname=f"{outdir}/BS_overlap.png"
    score = _xi_integration(ensemble=ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs,
                    plot_overlap=plot_overlap, figname=figname, max_attempts=1, progress=progress,
                            scale_energies=scale_energies, save_sampler_obj=save_sampler_obj)
    if verbose: print(score)
    if optimize_xi_values:
        #xi_values,_ = sampler.ti_info
        xi_values = sampler.xi_schedule
        u_kln = sampler.u_kln
        np.save("/Users/RR/github/pylambdaopt/biceps_xi_optimization/u_kln.npy", u_kln)

#
#        orig_shape = u_kln.shape
#        #print(u_kln.shape)
#        new = []
#        for i in range(u_kln.shape[-1]):
#            n_rows, n_cols = u_kln[:,:,i].shape
#            # Create a mask to keep only diagonal and adjacent elements
#            mask = np.zeros((n_rows, n_cols), dtype=bool)
#            # Set the mask to True for diagonal and off-diagonal elements
#            for i in range(n_rows):
#                # Diagonal element
#                mask[i, i] = True
#                # Element above the diagonal, if within bounds
#                if i > 0:
#                    mask[i, i - 1] = True
#                # Element below the diagonal, if within bounds
#                if i < n_cols - 1:
#                    mask[i, i + 1] = True
#            # Apply the mask to the matrix: keep only desired elements, set others to zero
#            new.append(np.where(mask, u_kln[:,:,i], 0))
#        u_kln = np.array(new).reshape(orig_shape)
#        print(u_kln)
#


        if verbose: print(xi_values)
        if optimize_xi_values:
            new_xi_values,_ = biceps.XiOpt.optimize_xi_values(xi_values, u_kln, outdir,
                            nsteps=xi_opt_steps, tol=tol, alpha=alpha, print_every=print_every,
                            make_plots=make_plots, optimize_nXis=optimize_nXis, verbose=verbose)

        if save_sampler_obj:
            biceps.toolbox.save_object(_, os.path.join(outdir,f"opt_xis.pkl"))

        if verbose: print(new_xi_values)
        PSkwargs["xi_schedule"] = new_xi_values
        ####################################################################
        PSkwargs["change_xi_every"] = round(sampler.nsteps/len(new_xi_values))
        PSkwargs["dXi"] = 1 / (len(new_xi_values) - 1)
        ####################################################################


        figname=f"{outdir}/BS_overlap_after.png"
        score = _xi_integration(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs,
                            plot_overlap=plot_overlap, figname=figname, max_attempts=max_attempts,
                                progress=progress, scale_energies=scale_energies, save_sampler_obj=save_sampler_obj)


#    A = biceps.Analysis(sampler, outdir=outdir, MBAR=False)
#    figures,steps,dists = A.plot_energy_trace()



    return score



#
#def xi_integration(ensemble, PSkwargs, sample_kwargs, plot_overlap=False, outdir=None,
#                   optimize_xi_values=True, optimize_nXis=True, xi_opt_steps=2000000, tol=1e-7, alpha=1e-5,
#                   max_attempts=2, print_every=1000, progress=True, scale_energies=False, verbose=False):
#
#    PSkwargs["xi_integration"] = True
#    if plot_overlap and optimize_xi_values: make_plots = 1
#    else: make_plots = 0
#
#    figname=f"{outdir}/BS_overlap.png"
#    score = _xi_integration(ensemble=ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs,
#                    plot_overlap=plot_overlap, figname=figname, max_attempts=1, progress=progress, scale_energies=scale_energies)
#
#    if verbose: print(score)
#    #xi_values,_ = sampler.ti_info
#    xi_values = sampler.xi_schedule
#    u_kln = sampler.u_kln
##    print(u_kln)
#    if verbose: print(xi_values)
#
#    passed = 0
#    if optimize_nXis:
#        try:
#            new_xi_values = biceps.XiOpt.optimize_xi_values(xi_values, u_kln, outdir,
#                            nsteps=xi_opt_steps, tol=tol, alpha=alpha, print_every=print_every,
#                            make_plots=make_plots, optimize_nXis=optimize_nXis, verbose=verbose)
#            passed = 1
#        except(Exception) as e:
#            new_xi_values = biceps.XiOpt.optimize_xi_values(xi_values, u_kln, outdir,
#                            nsteps=xi_opt_steps, tol=tol, alpha=alpha, print_every=print_every,
#                            make_plots=make_plots, optimize_nXis=False, verbose=verbose)
#            passed = 0
#
#    else:
#        new_xi_values = biceps.XiOpt.optimize_xi_values(xi_values, u_kln, outdir,
#                        nsteps=xi_opt_steps, tol=tol, alpha=alpha, print_every=print_every,
#                        make_plots=make_plots, optimize_nXis=optimize_nXis, verbose=verbose)
#        passed = 1
#
#
#    if verbose: print(new_xi_values)
#    PSkwargs["xi_schedule"] = new_xi_values
#    ####################################################################
#    PSkwargs["change_xi_every"] = round(sampler.nsteps/len(new_xi_values))
#    PSkwargs["dXi"] = 1 / (len(new_xi_values) - 1)
#    ####################################################################
#
#    figname=f"{outdir}/BS_overlap_after.png"
#    score = _xi_integration(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs,
#                        plot_overlap=plot_overlap, figname=figname, max_attempts=max_attempts, progress=progress, scale_energies=scale_energies)
#
#    if passed == 0:
#
#        xi_values = sampler.xi_schedule
#        u_kln = sampler.u_kln
#        print(u_kln)
#        new_xi_values = biceps.XiOpt.optimize_xi_values(xi_values, u_kln, outdir,
#                        nsteps=xi_opt_steps, tol=tol, alpha=alpha, print_every=print_every,
#                        make_plots=make_plots, optimize_nXis=optimize_nXis, verbose=verbose)
#
#        PSkwargs["xi_schedule"] = new_xi_values
#        ####################################################################
#        PSkwargs["change_xi_every"] = round(sampler.nsteps/len(new_xi_values))
#        PSkwargs["dXi"] = 1 / (len(new_xi_values) - 1)
#        ####################################################################
#
#        figname=f"{outdir}/BS_overlap_after.png"
#        score = _xi_integration(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs,
#                            plot_overlap=plot_overlap, figname=figname, max_attempts=max_attempts, progress=progress, scale_energies=scale_energies)
#
#
#
#
#
#    return score
#

def biceps_TI(ensemble, PSkwargs, sample_kwargs, plot_overlap=0, figname="BS_overlap.png", max_attempts=2):
    warnings.warn("`biceps_TI` is deprecated and will be removed in a future version. Use `xi_integration` instead.", DeprecationWarning, stacklevel=2)
    return xi_integration(ensemble, PSkwargs, sample_kwargs, plot_overlap, figname, max_attempts)

# }}}

# Objective Function:{{{
def objective_function(parameters, init_parameters, ensemble, fwd_model_function, phi0, phi_angles, restraint_index, verbose, PSkwargs, sample_kwargs, max_attempts=2):
    if not np.isfinite(init_parameters).all():
        print("Initial parameters contain NaNs or infs!")
        return np.nan
    if np.any(parameters) == np.nan: return np.nan

#    A, B, C = init_parameters
#    init_result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
#    init_model_J, init_diff_model_J, init_diff2_model_J = init_result

#    stime = time.time()
    try:
        A, B, C = parameters
        result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
    except(Exception) as e:
        result = fwd_model_function(phi_angles, parameters, phi0=phi0)

    model_J, diff_model_J, diff2_model_J = result
    #print(model_J)
    #exit()
#    print(f"Time for getting the data: {time.time() - stime}s")
#    stime = time.time()
    for l in range(len(ensemble.expanded_values)): # thermodynamic ensembles
        for s in range(len(ensemble.ensembles[l].ensemble)): # conformational states
            for r in range(len(ensemble.ensembles[l].ensemble[s])): # data restraint types
                if r != restraint_index: continue
                for j in range(len(ensemble.ensembles[l].ensemble[s][r].restraints)): # data points (observables)
                    ensemble.ensembles[l].ensemble[s][r].restraints[j]["model"] = model_J[s][j]

    global sampler
    good_overlap = True
    threshold = 0.02
    rerun = True
    trial = 0
    #max_attempts = 2
    while rerun == True:
        trial += 1
        sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
        print("done sampling.")
        #BS = sampler.get_results(scores_only=1, progress=0, compute_derivative=0, k_indices=[1,2])
        try:
            mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=0, plot_overlap=0, filename="BS_overlap.png")
            print("done getting mbar obj")
            overlap = mbar.compute_overlap()
            overlap_matrix = overlap["matrix"]
        except(Exception) as e:
            sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
            mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=0, plot_overlap=0, filename="BS_overlap.png")
            overlap = mbar.compute_overlap()
            overlap_matrix = overlap["matrix"]

        # overlap_matrix is a square matrix
        n = len(overlap_matrix)
        for i in range(n - 1):  # Iterate through rows, except the last row
            # Check the element to the right of the diagonal
            if overlap_matrix[i][i + 1] <= threshold:
                good_overlap = False
            # Check the element below the diagonal
            if overlap_matrix[i + 1][i] <= threshold:
                good_overlap = False

        if not good_overlap: print("WARNING: Overlap matrix issues...trying again")
        else:
            rerun = False
            #BS = (BS + -mbar.f_k[-1])
            BS = np.array([-mbar.f_k[-1]])
            nreplicas = sampler.nreplicas
            BS = BS/nreplicas
            if BS[-1] == 0.0:
                BS = jnp.array(BS)
                BS.at[-1].set(jnp.nan)
            if verbose: print(f"(parameters, score) = ({parameters}, {float(BS[-1])})")

        if trial == max_attempts:
            print("WARNING: Still overlap matrix issues...\nNo more attempts...\nYou might want to consider changing PSkwargs['change_xi_every'] and PSkwargs['dXi']")
            BS = jnp.array([jnp.nan])
            rerun = False

    return BS[-1]



def Jac(parameters, init_parameters, ensemble, fwd_model_function, phi0, phi_angles, restraint_index, verbose, *args):
    if not np.isfinite(init_parameters).all():
        print("Initial parameters contain NaNs or infs!")
        return jnp.ones(len(init_parameters))*np.nan
    if np.any(parameters) == np.nan: return jnp.ones(len(init_parameters))*np.nan

    nreplicas = sampler.nreplicas
    jac = []
    try:
        A, B, C = parameters
        result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
    except(Exception) as e:
        result = fwd_model_function(phi_angles, parameters, phi0=phi0)
    model_J, diff_model_J, diff2_model_J = result
    l = 0
    for i in range(len(parameters)):
        for l in range(len(sampler.expanded_values)): # thermodynamic ensembles
            for s in range(len(sampler.ensembles[l])): # conformational states
                for r in range(len(sampler.ensembles[l][s])): # data restraint types
                    if r != restraint_index: continue
                    for j in range(len(sampler.ensembles[l][s][r].restraints)): # data points (observables)
                        sampler.ensembles[l][s][r].restraints[j]["model"] = model_J[s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff model"] = diff_model_J[i][s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff2 model"] = diff2_model_J[i][s][j]

        #dBS = sampler.get_results(scores_only=1, progress=0, compute_derivative=1, k_indices=[1,2])[1][-1]
        mbar, diff_mbar, diff2_mbar, diff_mbar2 = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=1,
                                                                              plot_overlap=0, filename="jac_overlap.png")
        dBS = -diff_mbar.f_k[-1]
        #jac.append( (dBS + -diff_mbar.f_k[-1]) )
        jac.append( (dBS) )
#    print(f"Time for initializing the ensemble (JAC): {time.time() - stime}s")
    jac = jnp.array(jac)
    for i,val in enumerate(jac):
       if jac[i] == 0.0:
           jac.at[i].set(jnp.nan)
    if verbose: print(jac/nreplicas)
    jac = jac/nreplicas
    # Check for NaN values
    if jnp.all(jnp.isnan(jac)):
        return jnp.ones_like(jac) * 1e8  # Return a large number for NaN values
    else:
        return jac


def Hessian(parameters, init_parameters, ensemble, fwd_model_function, phi0, phi_angles, restraint_index, verbose, *args):
    if not np.isfinite(init_parameters).all():
        print("Initial parameters contain NaNs or infs!")
        return jnp.ones((len(init_parameters), len(init_parameters)))*np.nan
    if np.any(parameters) == np.nan: return jnp.ones((len(init_parameters), len(init_parameters)))*np.nan

    nreplicas = sampler.nreplicas
    hess_diag = []
    try:
        A, B, C = parameters
        result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
    except(Exception) as e:
        result = fwd_model_function(phi_angles, parameters, phi0=phi0)
    model_J, diff_model_J, diff2_model_J = result

    for i in range(len(parameters)):
        for l in range(len(sampler.expanded_values)): # thermodynamic ensembles
            for s in range(len(sampler.ensembles[l])): # conformational states
                for r in range(len(sampler.ensembles[l][s])): # data restraint types
                    if r != restraint_index: continue
                    for j in range(len(sampler.ensembles[l][s][r].restraints)): # data points (observables)
                        sampler.ensembles[l][s][r].restraints[j]["model"] = model_J[s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff model"] = diff_model_J[i][s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff2 model"] = diff2_model_J[i][s][j]

#        hess.append(sampler.get_results(scores_only=1, progress=0, compute_derivative=1)[2][-1])

        #d2BS = sampler.get_results(scores_only=1, progress=0, compute_derivative=1, k_indices=[1,2])[2][-1]
        # NOTE: Should be positive
#        mbar, diff_mbar, diff2_mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=1)
#        d2BS = diff2_mbar.f_k[-1]

        mbar, diff_mbar, diff2_mbar, diff_mbar2 = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=1,
                                                                              plot_overlap=0, filename="hess_overlap.png")

        dBS2 = diff_mbar2.f_k
        dBS = diff_mbar.f_k
        _d2BS = diff2_mbar.f_k
        d2BS = np.array([np.nansum([_d2BS[j], -dBS2[j]/(sampler.nreplicas), dBS[j]**2/(sampler.nreplicas)]) for j in range(len(dBS))])
        hess_diag.append( (-d2BS[-1]) )

    hess_diag = np.array(hess_diag)
    H = jnp.ones((len(hess_diag), len(hess_diag)))*d2BS[-1]

    for i, val in enumerate(hess_diag):
        H = H.at[i, i].set(val)

    # Calculate the off-diagonal terms
    for combo in combinations(range(len(init_parameters)), 2):

        i, j = combo
        if i == j: continue
        # Create a new list of parameters for perturbation
        perturbed_parameters = list(init_parameters)
        for idx in combo:
            perturbed_parameters[idx] = parameters[idx]

        # Now, unpack perturbed_parameters maintaining order
        try:
            A, B, C = perturbed_parameters
            result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
        except(Exception) as e:
            result = fwd_model_function(phi_angles, perturbed_parameters, phi0=phi0)
#        model_J, diff_model_J, diff2_model_J = result
#        diff_model_J, diff2_model_J = diff_model_J.sum(axis=0), diff2_model_J.sum(axis=0)
#        print(diff_model_J[0])

        model_J, diff_model_J, diff2_model_J = result
        diff_model_J, diff2_model_J = diff_model_J[[i,j]].sum(axis=0), diff2_model_J[[i,j]].sum(axis=0)
#        print(diff_model_J[0])
#        #print(diff_model_J.shape)
#        exit()

        for l in range(len(sampler.expanded_values)): # thermodynamic ensembles
            for s in range(len(sampler.ensembles[l])): # conformational states
                for r in range(len(sampler.ensembles[l][s])): # data restraint types
                    if r != restraint_index: continue
                    for j in range(len(sampler.ensembles[l][s][r].restraints)): # data points (observables)
                        sampler.ensembles[l][s][r].restraints[j]["model"] = model_J[s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff model"] = diff_model_J[s][j]
                        sampler.ensembles[l][s][r].restraints[j]["diff2 model"] = diff2_model_J[s][j]
        mbar, diff_mbar, diff2_mbar, diff_mbar2 = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=1)

        dBS2 = diff_mbar2.f_k
        dBS = diff_mbar.f_k
        _d2BS = diff2_mbar.f_k
        d2BS = np.array([np.nansum([_d2BS[j], -dBS2[j]/(sampler.nreplicas), dBS[j]**2/(sampler.nreplicas)]) for j in range(len(dBS))])
        off_diag_val = -d2BS[-1]
        i, j = combo
        H = H.at[i, j].set(off_diag_val)
        H = H.at[j, i].set(off_diag_val)  # Hessian matrix is symmetric
    if verbose: print(H/nreplicas)
    #print(H/nreplicas)

    debug = 1
    if debug:
        #print(H)
        eigenvalues = np.linalg.eigvals(H)
        if np.all(eigenvalues > 0):
            covariance_matrix = np.linalg.inv(H)
        else:
            print("Hessian is not positive definite")
        # Calculate the determinant of the Hessian matrix
        determinant = np.linalg.det(H)
        # Define a small threshold value
        threshold = 1e-10
        # Check if the absolute value of the determinant is close to zero
        if np.abs(determinant) < threshold:
            print("The determinant is close to zero.")
        else:
            print("The determinant is not close to zero.")

    return H/nreplicas



def get_uncertainty_from_covariance_matrix(parameters, init_parameters, ensemble,
            fwd_model_function, phi0, phi_angles, restraint_index, verbose,
                                           PSkwargs, sample_kwargs, sampler):
    """
    """
    try:
        hessian_matrix = Hessian(parameters, init_parameters, ensemble,
                                  fwd_model_function, phi0, phi_angles, restraint_index, False,
                                  PSkwargs, sample_kwargs, sampler)
        hessian_matrix *= sampler.nreplicas

        # Check the positive definiteness of the Hessian
        eigenvalues = np.linalg.eigvals(hessian_matrix)
        if np.all(eigenvalues > 0):
            # Calculate the covariance matrix using the inverse
            covariance_matrix = np.linalg.inv(hessian_matrix)
        else:
            if verbose:
                print("Hessian is not positive definite. Using pseudo-inverse.")
            # Use pseudo-inverse if Hessian is not positive definite
            covariance_matrix = np.linalg.pinv(hessian_matrix)

        # Extract parameter uncertainties (standard errors)
        parameter_uncertainties = np.sqrt(np.diagonal(covariance_matrix))

    except Exception as e:
        if verbose:
            print(f"Exception occurred: {e}")
        parameter_uncertainties = np.array([np.nan] * len(parameters))

    return parameter_uncertainties







# Define a callback function to track the number of iterations
def callback(xk):
    global num_iterations
    num_iterations += 1
    print("Callback")
    if num_iterations >= maxiters:
        print("Reached maximum number of iterations. Terminating optimization.")
        return True


# }}}

# plot karplus curves:{{{
def get_karplus_curves(ax, optimized_parameters, initial_parameters, exp_parameters,
            phi0=0.0, ref_color="r", opt_color="blue", karplus_label="H^{N}H^{\alpha}",
            opt_model=f"Opt."):

    phis = np.linspace(-180, 180, 1000)
    results = []

    A, B, C = initial_parameters
    scalar_couplings = get_scalar_couplings_with_derivatives(np.deg2rad(phis), A=A, B=B, C=C, phi0=phi0)[0]
    #results.append({"model":f"Initial: {initial_parameters}", "scalar_couplings":scalar_couplings, "phis":phis})
    results.append({"model":f"Initial", "scalar_couplings":scalar_couplings, "phis":phis})

    A, B, C = optimized_parameters
    scalar_couplings = get_scalar_couplings_with_derivatives(np.deg2rad(phis), A=A, B=B, C=C, phi0=phi0)[0]
    results.append({"model":opt_model, "scalar_couplings":scalar_couplings, "phis":phis})

    if not all(np.isnan(exp_parameters)):
        #print("jsdnvjkasdncjksnajlksdnjkl")
        A, B, C = exp_parameters
        scalar_couplings = get_scalar_couplings_with_derivatives(np.deg2rad(phis), A=A, B=B, C=C, phi0=phi0)[0]
        results.append({"model":"Exp.", "scalar_couplings":scalar_couplings, "phis":phis})

    df = pd.DataFrame(results)
    #print(df)

    for i in range(len(df["model"].to_numpy())):
        row = df.iloc[[i]]
        if row["model"].to_numpy()[0] == "Exp.":
            ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label=row["model"].to_numpy()[0], ls="dotted", color="k", lw=2)
        elif row["model"].to_numpy()[0] == opt_model:
            ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label=row["model"].to_numpy()[0], color=opt_color, lw=2)
        elif "Initial" in row["model"].to_numpy()[0]:
            if np.isnan(row["scalar_couplings"].to_numpy()[0]).all():
                ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label="__no_legend__", color=ref_color, lw=2)
            else:
                ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label=row["model"].to_numpy()[0], color=ref_color, lw=2)
        else:
            ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], label=row["model"].to_numpy()[0])
    ax.set_xlim(phis[0], phis[-1])
    yticks = list(range(14)[::2])
    ax.set_yticks(yticks)
    ax.set_ylim(yticks[0]-1, yticks[-1]+2)
    ax.set_xticks([phis[0], 0, phis[-1]])
    ax.legend(loc="best", fontsize=10)
    #ax.set_xlabel(r'$\phi$ (degrees)', fontsize=14)
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=14)
    ax.set_ylabel(r'$^3 J_{%s}$ (Hz)'%karplus_label, fontsize=14)
    return ax


def plot_karplus_curves(results, initial_parameters, exp_parameters, phi0=0.0, figname=None, karplus_label=r"H^{N}H^{\alpha}"):

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (8, 4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100
    ax4 = plt.subplot(gs[0, 0])              # subplot for karplus curves
    axs = [ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.grid(alpha=0.5, linewidth=0.5)

    final_parameters = [float("%0.3g"%results["A"].to_numpy()[-1]), float("%0.3g"%results["B"].to_numpy()[-1]), float("%0.3g"%results["C"].to_numpy()[-1])]
    optimized_parameters = final_parameters
    ax4 = get_karplus_curves(ax4, optimized_parameters, initial_parameters, exp_parameters, phi0, karplus_label=karplus_label)
    ax4.set_title(r"$J(\phi+\varphi) = Acos^{2}(\phi+\varphi) + Bcos(\phi+\varphi) + C$", fontsize=12)

    # Save the figure
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig


# }}}

# plot:{{{
def plot_results(results, initial_parameters, exp_parameters,
    exp_parameters_sigma=[], figname=None, use_uncertainties=True, phi0=0.0, karplus_label=r"H^{N}H^{\alpha}"):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (12, 6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100


    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)  # Subplot for A vs C
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # Subplot for B vs C
    ax4 = plt.subplot(gs[1, 2])              # subplot for karplus curves
    axTable = plt.subplot(gs[0, 1:])         # subplot for karplus curves

    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.grid(alpha=0.5, linewidth=0.5)

    final_parameters = [float("%0.3g"%results["A"].to_numpy()[-1]), float("%0.3g"%results["B"].to_numpy()[-1]), float("%0.3g"%results["C"].to_numpy()[-1])]

    # Get the y-axis limits
    y_min, y_max = ax1.get_ylim()
    # Calculate the range of the y-axis
    y_range = y_max - y_min
    # Define the offset as a fraction of the y-axis range
    offset_fraction = 0.0025  # Adjust this value as needed
    offset = offset_fraction * y_range

    import matplotlib.patches as patches
    if exp_parameters_sigma != []:
        lower_left_corner = (exp_parameters[0] - exp_parameters_sigma[0], exp_parameters[1] - exp_parameters_sigma[1])
        square = patches.Rectangle(lower_left_corner, 2*exp_parameters_sigma[0], 2*exp_parameters_sigma[1], linewidth=2, edgecolor='k', facecolor='none')
        ax1.add_patch(square)
        ax1.text(exp_parameters[0], exp_parameters[1]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)

        lower_left_corner = (exp_parameters[0] - exp_parameters_sigma[0], exp_parameters[2] - exp_parameters_sigma[2])
        square = patches.Rectangle(lower_left_corner, 2*exp_parameters_sigma[0], 2*exp_parameters_sigma[2], linewidth=2, edgecolor='k', facecolor='none')
        ax2.add_patch(square)
        ax2.text(exp_parameters[0], exp_parameters[2]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)

        lower_left_corner = (exp_parameters[1] - exp_parameters_sigma[1], exp_parameters[2] - exp_parameters_sigma[2])
        square = patches.Rectangle(lower_left_corner, 2*exp_parameters_sigma[1], 2*exp_parameters_sigma[2], linewidth=2, edgecolor='k', facecolor='none')
        ax3.add_patch(square)
        ax3.text(exp_parameters[1], exp_parameters[2]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)

    else:
        if not all(np.isnan(exp_parameters)):
            ax1.scatter(exp_parameters[0], exp_parameters[1], color="k", marker="s", facecolors="w", linewidths=2, s=main_marker_size)
            ax1.text(exp_parameters[0], exp_parameters[1]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)
            ax2.scatter(exp_parameters[0], exp_parameters[2], color="k", marker="s", facecolors="w", linewidths=2, s=main_marker_size)
            ax2.text(exp_parameters[0], exp_parameters[2]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)
            ax3.scatter(exp_parameters[1], exp_parameters[2], color="k", marker="s", facecolors="w", linewidths=2, s=main_marker_size)
            ax3.text(exp_parameters[1], exp_parameters[2]+offset, f'Exp.', color='k', ha='left', va='bottom', rotation=0)

###############################################################################

    # Plot A vs B
    if "A sigma" in results.columns.to_list():
        ax1.errorbar(results["A"], results["B"], xerr=results["A sigma"].to_numpy(), yerr=results["B sigma"].to_numpy(), color="k", capsize=4, ls="None")
    ax1.scatter(results["A"], results["B"], s=marker_size, edgecolors='black', linewidth=1, facecolors=facecolors)
    ax1.set_xlabel(r"$A$", fontsize=16)
    ax1.set_ylabel(r"$B$", fontsize=16)

    # Plot A vs C
    if "A sigma" in results.columns.to_list():
         ax2.errorbar(results["A"], results["C"], xerr=results["A sigma"].to_numpy(), yerr=results["C sigma"].to_numpy(), color="k", capsize=4, ls="None")
    ax2.scatter(results["A"], results["C"], s=marker_size, edgecolors='black', linewidth=1, facecolors=facecolors)
    ax2.set_xlabel(r"$A$", fontsize=16)
    ax2.set_ylabel(r"$C$", fontsize=16)

    # Plot B vs C
    if "B sigma" in results.columns.to_list():
        ax3.errorbar(results["B"], results["C"], xerr=results["B sigma"].to_numpy(), yerr=results["C sigma"].to_numpy(), color="k", capsize=4, ls="None")
    ax3.scatter(results["B"], results["C"], s=marker_size, edgecolors='black', linewidth=1, facecolors=facecolors)
    ax3.set_xlabel(r"$B$", fontsize=16)
    ax3.set_ylabel(r"$C$", fontsize=16)


###############################################################################

    ax1.scatter(initial_parameters[0], initial_parameters[1], color="red", marker="*", s=main_marker_size)
    ax1.text(initial_parameters[0], initial_parameters[1], f'Initial', color='red', ha='left', va='bottom', rotation=0)

    ax2.scatter(initial_parameters[0], initial_parameters[2], color="red", marker="*", s=main_marker_size)
    ax2.text(initial_parameters[0], initial_parameters[2], f'Initial', color='red', ha='left', va='bottom', rotation=0)

    ax3.scatter(initial_parameters[1], initial_parameters[2], color="red", marker="*", s=main_marker_size)
    ax3.text(initial_parameters[1], initial_parameters[2], f'Initial', color='red', ha='left', va='bottom', rotation=0)

###############################################################################
    if len(results) <= 1:
        final_parameters = [np.nan]*len(initial_parameters)
    optimized_parameters = final_parameters
    ax4 = get_karplus_curves(ax4, optimized_parameters, initial_parameters, exp_parameters, phi0=phi0, karplus_label=karplus_label)



    parameter_sets = biceps.J_coupling.J3_HN_HA_coefficients
    phis = np.linspace(-180, 180, 1000)
    #phi0 = parameter_sets["Habeck"]["phi0"]

    y1 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=optimized_parameters[0]+np.nan_to_num(results["A sigma"].to_numpy()[-1]),
            B=optimized_parameters[1]+np.nan_to_num(results["B sigma"].to_numpy()[-1]),
            C=optimized_parameters[2]+np.nan_to_num(results["C sigma"].to_numpy()[-1]), phi0=phi0)[0]
    y2 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=optimized_parameters[0]-np.nan_to_num(results["A sigma"].to_numpy()[-1]),
            B=optimized_parameters[1]-np.nan_to_num(results["B sigma"].to_numpy()[-1]),
            C=optimized_parameters[2]-np.nan_to_num(results["C sigma"].to_numpy()[-1]), phi0=phi0)[0]
    ax4.fill_between(phis, y1, y2, color="blue", alpha=0.4)


    if exp_parameters_sigma != []:
        y1 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
                A=exp_parameters[0]+np.nan_to_num(exp_parameters_sigma[0]),
                B=exp_parameters[1]+np.nan_to_num(exp_parameters_sigma[1]),
                C=exp_parameters[2]+np.nan_to_num(exp_parameters_sigma[2]), phi0=phi0)[0]
        y2 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
                A=exp_parameters[0]-np.nan_to_num(exp_parameters_sigma[0]),
                B=exp_parameters[1]-np.nan_to_num(exp_parameters_sigma[1]),
                C=exp_parameters[2]-np.nan_to_num(exp_parameters_sigma[2]), phi0=phi0)[0]
        ax4.fill_between(phis, y1, y2, color="gray", alpha=0.6)


    ax4.set_title(r"$J(\phi) = Acos^{2}(\phi) + Bcos(\phi) + C$", fontsize=12)

###############################################################################

    # Define the normalization for the arrow lengths
    arrow_norm = Normalize(vmin=0, vmax=1)

    # Add arrows to the next epsilon location
    for i in range(len(results["A"].to_numpy()) - 1):
        dx = results["A"].to_numpy()[i + 1] - results["A"].to_numpy()[i]
        dy = results["B"].to_numpy()[i + 1] - results["B"].to_numpy()[i]
        arrow_len = np.sqrt(dx ** 2 + dy ** 2)
        x_pos = results["A"].to_numpy()[i]
        y_pos = results["B"].to_numpy()[i]
        arrow_length = arrow_norm(arrow_len)
        ax1.quiver(x_pos, y_pos, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.01, color='k')

        # Add arrows for A vs C plot
        dx = results["A"].to_numpy()[i + 1] - results["A"].to_numpy()[i]
        dy = results["C"].to_numpy()[i + 1] - results["C"].to_numpy()[i]
        x_pos = results["A"].to_numpy()[i]
        y_pos = results["C"].to_numpy()[i]
        ax2.quiver(x_pos, y_pos, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.01, color='k')

        # Add arrows for B vs C plot
        dx = results["B"].to_numpy()[i + 1] - results["B"].to_numpy()[i]
        dy = results["C"].to_numpy()[i + 1] - results["C"].to_numpy()[i]
        x_pos = results["B"].to_numpy()[i]
        y_pos = results["C"].to_numpy()[i]
        ax3.quiver(x_pos, y_pos, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.01, color='k')

    # Create a color map based on the score
    cmap = plt.cm.get_cmap('coolwarm')
    norm = matplotlib.colors.Normalize(vmin=results['score'].min(), vmax=results['score'].max())



    # no labels
    axTable.xaxis.set_major_formatter(nullfmt)
    axTable.yaxis.set_major_formatter(nullfmt)

# FIXME: IMPORTANT: Remove "true" and make it something else
   # Table information
    rows = ["","Initial","","Exp.","","Optimized"]
    columns= [r"$A$",r"$B$",r"$C$"]
    empty_row = ["" for c in range(len(columns))]

    _final_parameters = []
    for par in final_parameters:
        if np.isnan(par): _final_parameters.append("")
        else: _final_parameters.append("%0.3g" % par)

    cell_text = [empty_row,
            ["%0.3g" % par for par in initial_parameters],\
            empty_row,
            ["%0.3g" % par for par in exp_parameters],\
            empty_row,
            _final_parameters,\
            ]
    colors = [None,"red",None,"k",None,"blue",]

    rowColours=colors
    the_table = axTable.table(cellText=cell_text,\
                             rowLabels=rows,\
                             colLabels=columns,\
                             loc='best', cellLoc='center',
                             rowLoc='center', rowColours=rowColours,
                             #cellColours=cellColours,
                             #colWidths=[0.14 for c in columns],
                             colWidths=[0.16 for c in columns],
                             in_layout=True, zorder=1)#, edges="open")
    the_table.zorder=1
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    axTable.axis('off')
    # iterate through cells of a table
    table_props = the_table.properties()
    table_cells = table_props['celld']


    _colors_ = [c for c in colors if c is not None]
    alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,]
    c = 0
    for cell in table_cells.items():
        cell[1].set_edgecolor('none')
        cell[1].set_facecolor('white')
        if cell[0][-1] == -1:
            if cell[0][0]%2 == 0:
                cell[1].set_text_props(color=_colors_[c])
                cell[1].set_facecolor('white')
        #cell[1].set_height(.071)
        cell[1].set_height(.12)
        if cell[0][-1] == -1:
            if cell[0][0]%2 == 0:
                cell[1].set_edgecolor('black')
                the_color = (*matplotlib.colors.to_rgb(_colors_[c]), alphas[c])
                #if (_colors_[c] == "k") or (_colors_[c] == "black"):
                #    cell[1].set_text_props(color="white")
                #    cell[1].set_facecolor('black')
                #    c += 1
                #    continue
                #cell[1].set_facecolor(the_color)
                c += 1

#    the_table.get_celld()[(0,0)].set_text_props(color=colors[1])
#    the_table.get_celld()[(1,0)].set_text_props(color=colors[3])
#    the_table.get_celld()[(2,0)].set_text_props(color=colors[5])




    # Save the figure
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# create_gif:{{{
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def create_gif(figures, figname, duration=500, pause_duration=1000):
    import imageio
    from PIL import Image

    images = []
    for fig in figures:
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(image)

    # Duplicate the last frame to create a pause
    pause_frames = int(pause_duration / duration)
    last_frame = images[-1]
    for _ in range(pause_frames):
        images.append(last_frame)

    # Save the images as frames of a GIF
    gif_path = os.path.splitext(figname)[0] + '.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    # Display the GIF
    if isnotebook():
        from IPython.display import display, Image
        display(Image(filename=gif_path, format='png'))


# }}}

# plot_overlap_matrix:{{{
def plot_overlap_matrix(sampler, outdir="./"):

    #ti_info = sampler.ti_info
    xi_values = sampler.xi_schedule
    mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=0, compute_derivative=0)
    overlap = mbar.compute_overlap()
    overlap_matrix = overlap["matrix"]
    #matrices.append(overlap_matrix)
    _results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
    Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
    #print(Deltaf_ij)
    f_df = np.zeros( (len(overlap_matrix), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
    f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
    f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std
    BS = -f_df[:,0]/sampler.nreplicas
    #print(BS)
    print("Integral from MBAR:", BS[-1])

    force_constants = [r"$\xi$=%0.2g"%val for val in xi_values]
    fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the figsize as desired
    im = ax.pcolor(overlap_matrix, edgecolors='k', linewidths=2)

    # Add annotations
    for i in range(len(overlap_matrix)):
        for j in range(len(overlap_matrix[i])):
            value = overlap_matrix[i][j]
            if value > 0.01:
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color=text_color,
                        fontsize=12)  # Adjust fontsize as desired
    ax.set_xticks(np.array(list(range(len(force_constants)))))
    ax.set_yticks(np.array(list(range(len(force_constants)))))
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.grid()

    try:
        ax.set_xticklabels([str(tick) for tick in force_constants], rotation=90, size=16)
        ax.set_yticklabels([str(tick) for tick in force_constants], size=16)
    except Exception as e:
        print(e)

    ax.set_xticklabels(ax.get_xticklabels(), ha='left')
    ax.set_yticklabels(ax.get_yticklabels(), va='bottom')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Overlap probability between states", size=16)  # Set the colorbar label
    fig.tight_layout()
    fig.savefig(f"{outdir}/contour.png")
# }}}

# plot_hist_of_phi_angles:{{{
def plot_hist_of_phi_angles(phi_angles, parameters=None, phi0_deg=0, label=None, show_hist=False):
    import matplotlib
    from biceps.J_coupling import _J3_function, J3_HN_HA_coefficients

    mpl_colors = matplotlib.colors.get_named_colors_mapping()
    mpl_colors = list(mpl_colors.values())[::5]
    extra_colors = mpl_colors.copy()

    mpl_colors = ['k',"blue","purple","brown","green","grey",
                  "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
            '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
                '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]

    markers = ['--','-o','-s','-^','-v','-<','->','-d','-*','-p','-h']

    # Perform a scan of all the Karplus relations
    #models = ["Ruterjans1999","Bax2007","Bax1997","Habeck","Vuister" ,"Pardi"]
    #models = ["Ruterjans1999","Bax2007","Habeck","Vuister" ,"Pardi"]
    #labels = ["Ruterjans (1999)","Bax (2007)","Habeck (2005)","Vuister (1993)","Pardi (1984)"]
    models = ["Ruterjans1999"]
    if label: labels = [label]
    else: labels = ["Ruterjans (1999)"]

    phis = np.linspace(-180, 180, 50)

    _J3_function = biceps.J_coupling._J3_function

    results = []
    for i,model in enumerate(models):
        if parameters != []:
            items = {key:parameters[k] for k,key in enumerate(["A", "B", "C"])}
        else:
            items = {key:J3_HN_HA_coefficients[model][key] for key in ["A", "B", "C"]}

        #Calculate the scalar coupling between HN and H_alpha.
        #scalar_couplings = _J3_function(np.deg2rad(phis), **J3_HN_HA_coefficients[model])
        scalar_couplings = _J3_function(np.deg2rad(phis), **items, phi0=np.deg2rad(phi0_deg))
        results.append({"model":model, "scalar_couplings":scalar_couplings, "phis":phis, "label":labels[i]})

    df = pd.DataFrame(results)
    #print(df)


    fig, ax = plt.subplots()
    for i in range(len(df["model"].to_numpy())):
        row = df.iloc[[i]]
        color = mpl_colors[i]
        if row["model"].to_numpy()[0].startswith("Opt"):
            ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], markers[i], ms=4, label=row["model"].to_numpy()[0], ls="dotted", lw=4, color=color, zorder=999)
        else:
            ax.plot(row["phis"].to_numpy()[0], row["scalar_couplings"].to_numpy()[0], markers[i], ms=4, label=row["label"].to_numpy()[0], color=color, lw=2, zorder=999)

    fig.set_size_inches(5.5, 3.5)  # Set the figure size to 8x4 inches
    xlim = (phis[0], phis[-1])
    ax.set_xlim(xlim)
    ax.set_xticks([phis[0], -90, 0, 90, phis[-1]])

    yticks = list(range(14)[::2])
    ax.set_yticks(yticks)
    ylim = (yticks[0]-1, yticks[-1])
    ax.set_ylim(ylim)
    label_fontsize = 12
    #legend = ax.legend(loc="best", fontsize=10)
    #legend = ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.6), fontsize=label_fontsize)


    if show_hist:
        hist_values, bin_edges = np.histogram(np.rad2deg(phi_angles), bins=25)
        line_y_max = np.max(row["scalar_couplings"].to_numpy()[0])  # Assuming y is the array of y-values from your line plot
        scale_factor = 0.7  # for example, 20% larger
        hist_values_normalized = hist_values * (line_y_max / max(hist_values)) * scale_factor
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist_values_normalized, width=bin_width, facecolor="blue", edgecolor='black')
        ax.set_yticks([])

    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=16)

    #ax.set_ylabel('Counts', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    ax.set_ylim(bottom=0)

    ## Add content to the subplots (example)
    for i, ax in enumerate([ax]):
        x,y = -0.1, 1.02

        # Setting the ticks and tick marks
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(14)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=14)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)

    fig.tight_layout()
    #plt.gcf().subplots_adjust(left=0.105, bottom=0.15, top=0.93, right=0.62, wspace=0.20, hspace=0.5)
    #fig.savefig("karplus_relation.png", dpi=400)
    return fig
#:}}}


# plot_score_with_error:{{{
def plot_score_with_error(traces, traces_std, ylim=(0,4.5)):

    # Plotting
    figsize=(6,4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0,0])
    colors = ['r','b','g']

    for g,group in enumerate(traces):
        print(group)
        color = (["gold", "m", "g", "cyan", "silver"]+list(colors))[g]
        y=group[f"score"].to_numpy()
        y_std = traces_std[g][f"score"].to_numpy()
        x = list(range(len(y)))
        #ax.plot(x, y, color=color)
        ax.plot(x, y, color="k")
        ax.errorbar(x, y, yerr=y_std, color="k", zorder=3, capsize=3, ls="None")
        #ax.scatter(X, y, facecolor=color, edgecolor="k", zorder=4)


    label_fontsize = 12
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel(r"$f$", fontsize=20, rotation=0, labelpad=20)

    ax.tick_params(which="major", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
    #ax.tick_params(which="minor", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
    ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=0, top=0)
    ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=0)

    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ticks = [#ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax.yaxis.get_minor_ticks(),
             ax.yaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(label_fontsize)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=label_fontsize-2)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

#    ax.set_ylim(ylim)
#    ax.set_xlim((0, len(Y_mean)-1))
    # Save the figure
    fig.tight_layout()
    #fig.savefig(figname, dpi=600)
    return fig

# }}}


# plot:{{{
def plot_final_karplus_curve(results, initial_parameters, exp_parameters,
    exp_parameters_sigma=None, figname=None, use_uncertainties=True,
    phi0=0.0, ref_color="r", karplus_label="H^{N}H^{\alpha}", show_initial=False,
    show_opt=True):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    figsize = (8, 4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100

    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B

    axs = [ax1]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.grid(alpha=0.5, linewidth=0.5)

    final_parameters = [float("%0.3g"%results["A"].to_numpy()[-1]), float("%0.3g"%results["B"].to_numpy()[-1]), float("%0.3g"%results["C"].to_numpy()[-1])]

###############################################################################
    if len(results) <= 1:
        final_parameters = [np.nan]*len(initial_parameters)
    optimized_parameters = final_parameters
    if show_opt == False: opt_label = "__no_legend__"
    else: opt_label = "Opt."
    ax1 = get_karplus_curves(ax1, optimized_parameters, initial_parameters, exp_parameters, phi0=phi0, ref_color=ref_color, karplus_label=karplus_label, opt_model=opt_label)

    parameter_sets = biceps.J_coupling.J3_HN_HA_coefficients
    phis = np.linspace(-180, 180, 1000)
    #phi0 = parameter_sets["Habeck"]["phi0"]

    y1 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=optimized_parameters[0]+np.nan_to_num(results["A sigma"].to_numpy()[-1]),
            B=optimized_parameters[1]+np.nan_to_num(results["B sigma"].to_numpy()[-1]),
            C=optimized_parameters[2]+np.nan_to_num(results["C sigma"].to_numpy()[-1]), phi0=phi0)[0]
    y2 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=optimized_parameters[0]-np.nan_to_num(results["A sigma"].to_numpy()[-1]),
            B=optimized_parameters[1]-np.nan_to_num(results["B sigma"].to_numpy()[-1]),
            C=optimized_parameters[2]-np.nan_to_num(results["C sigma"].to_numpy()[-1]), phi0=phi0)[0]
    ax1.fill_between(phis, y1, y2, color="blue", alpha=0.4)


    y1 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=exp_parameters[0]+np.nan_to_num(exp_parameters_sigma[0]),
            B=exp_parameters[1]+np.nan_to_num(exp_parameters_sigma[1]),
            C=exp_parameters[2]+np.nan_to_num(exp_parameters_sigma[2]), phi0=phi0)[0]
    y2 = get_scalar_couplings_with_derivatives(np.deg2rad(phis),
            A=exp_parameters[0]-np.nan_to_num(exp_parameters_sigma[0]),
            B=exp_parameters[1]-np.nan_to_num(exp_parameters_sigma[1]),
            C=exp_parameters[2]-np.nan_to_num(exp_parameters_sigma[2]), phi0=phi0)[0]
    ax1.fill_between(phis, y1, y2, color="gray", alpha=0.6)

    yticks = list(range(14)[::2])
    ax1.set_yticks(yticks)
    #ax1.set_ylim(yticks[0]-1, np.max(exp_parameters)+3.5)
    ax1.set_ylim(yticks[0]-1, np.max(y1)+4.5)

    ax1.set_title(r"$J(\phi) = Acos^{2}(\phi+\phi_{0}) + Bcos(\phi+\phi_{0}) + C$", fontsize=12)

    # Save the figure
    fig.tight_layout()
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# generate_toy_model:{{{
def add_prior_error(states, σ_prior):
    """
    states():
    σ_prior(float): prior error in degrees
    """
    for state in states:
        state["true phi"] = copy.deepcopy(state["phi"])
        if σ_prior > 0.0:
            new_phi = []
            for phi in state["phi"]:
                if phi > 0.0:
                    new_phi.append(phi)
                else:
                    if σ_prior > 0.0:
                        noise = np.deg2rad(σ_prior) * np.random.randn()  # Generate Gaussian noise in radians
                        new_phi_rad = phi + noise  # Add noise to phi, converting phi to radians first
                        # Normalize the angle to be within -pi and pi (equivalent to -180° to 180°)
                        #print(np.rad2deg(new_phi_rad), " -> ", np.rad2deg((new_phi_rad + np.pi) % (2 * np.pi) - np.pi))
                        new_phi_rad = (new_phi_rad + np.pi) % (2 * np.pi) - np.pi
                        new_phi.append(new_phi_rad)
                    else:
                        new_phi.append(phi)
            state["phi"] = new_phi #state["phi"] + np.deg2rad(σ_prior)*np.random.randn()
    return states

def generate_toy_model(nstates, Nd, initial_parameters, exp_parameters, phi0_deg=-60.0, σ_prior=0.0, verbose=False):
    """
    Generate a toy model to test J-coupling constants.

    :param nstates: Number of states.
    :param Nd: Number of J-coupling values for each state.
    :param A: Karplus parameter A.
    :param B: Karplus parameter B.
    :param C: Karplus parameter C.
    :param phi0_deg: phi0 angle in degrees.
    :return: Tuple containing the true experimental J-couplings and a list of dictionaries representing the states.
    """
    A, B, C = exp_parameters
    A0, B0, C0 = initial_parameters
    # Convert phi0 to radians
    phi0 = np.deg2rad(phi0_deg)
    states = [] # Initialize states list
    # NOTE: Creating distribution of phi angles to sample from
    sigmas=(20, 10, 5)
    angles=(-110, -60, 60)
    weights=(0.35, 0.5, 0.15)
    dist_size = 10000
    nsamples = 1000
    # Validate that the weights sum to 1
    assert np.isclose(sum(weights), 1.0), "Weights should sum to 1"
    # Create empty list to store mixed angles
    mixed_angles = []
    # Generate the distributions
    distributions = []
    for i, angle in enumerate(angles):
        mean, std_dev = np.deg2rad(angle), np.deg2rad(sigmas[i])
        distributions.append(np.random.normal(mean, std_dev, dist_size))
    # Determine the number of samples to draw from each distribution
    samples_per_distribution = [int(weight * nsamples) for weight in weights]



    # Draw the samples
    for l, dist in enumerate(distributions):
        sampled_angles = np.random.choice(dist, samples_per_distribution[l], replace=False)
        mixed_angles.extend(sampled_angles)
    mixed_angles = np.array(mixed_angles)

    kT = 1.0
    phi = np.arange(-180.0, 180.0, 0.01)  # degrees
    probability = 0.0
    for l, dist in enumerate(distributions):
        probability += weights[l] * 1/np.sqrt(2*np.pi*sigmas[l]**2) *np.exp(-(phi - angles[l])**2/(2*sigmas[l]**2))
    u = -kT*np.log(probability)
################################################################################
#    fig, ax = plt.subplots()
#    ax.scatter(phi, u, s=1, color="k")
#    ax.set_ylabel(r"$E (k_{B}T)$", fontsize=16)#, labelpad=20)#, rotation=0)
#    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=16)
#    ax.set_ylim(0, 30)
#    fig.tight_layout()
#    fig.savefig("toy_model_figure_energies.png")
################################################################################

#    fig, ax = plt.subplots()
#    ax.scatter(phi, u, s=1, color="k")
#    ax.set_ylabel(r"$E (k_{B}T)$", fontsize=16)#, labelpad=20)#, rotation=0)
#    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=16)
#    ylim = [-12, 30]
#    ax.set_ylim(ylim)
#    xlim = [-180, 180]
#    ax.set_xlim(xlim)
#    colors = ["r", "b", "g"]
#    kT = 1.0
#    phi = np.arange(-180.0, 180.0, 0.01)  # degrees
#    for l, dist in enumerate(distributions):
#        color = colors[l]
#        probability = 0.0
#        probability += weights[l] * 1/np.sqrt(2*np.pi*sigmas[l]**2) *np.exp(-(phi - angles[l])**2/(2*sigmas[l]**2))
#        u1 = -kT*np.log(probability)
#        ax.plot(phi, u1, color=color, ls="--")
#
#    nl = "\n"
#    #ax.text(xlim[0]*1.00, ylim[1]*0.36, r'$\mu = %0.3g°$%s$\sigma=%0.2g°$%s$w = %s$'%(angles[0],nl,sigmas[0],nl,weights[0]), color=colors[0], ha='left', va='bottom', fontsize=14)
#    ax.text(xlim[0]*0.95, ylim[1]*-0.13, r'$\mu = %0.3g°$%s$\sigma=%0.2g°$%s$w = %s$'%(angles[0],nl,sigmas[0],nl,weights[0]), color=colors[0], ha='left', va='bottom', fontsize=14)
#    ax.text(xlim[0]+xlim[1]*0.55, ylim[1]*0.55, r'$\mu = %0.2g°$%s$\sigma=%0.2g°$%s$w = %s$'%(angles[1],nl,sigmas[1],nl,weights[1]), color=colors[1], ha='left', va='bottom', fontsize=14)
#    ax.text(xlim[1]*0.85, ylim[1]*0.55, r'$\mu = %0.2g°$%s$\sigma=%0.2g°$%s$w = %s$'%(angles[2],nl,sigmas[2],nl,weights[2]), color=colors[2], ha='right', va='bottom', fontsize=14)
#    fig.tight_layout()
#    fig.savefig("toy_model_figure_energies.png", dpi=400)
#
#    exit()
################################################################################

    # Draw the samples for each data point
    indices = np.random.choice(list(range(len(sigmas))), size=Nd, p=weights)
    # Draw Nstates samples from a particular distribution. Do this for each Nd
    # ***All conformations have angles sampled from the same mode***.
    state_angles = []
    for l in indices:
        sampled_angles = np.random.choice(distributions[l], nstates, replace=False)
        state_angles.append(sampled_angles + np.deg2rad(sigmas[l])*np.random.randn())
    state_angles = np.array(state_angles).T
    for k in range(nstates):
        closest_indices = np.array([np.argmin(np.abs(phi - angle), axis=0) for angle in np.rad2deg(state_angles[k])])
        states.append({'phi': state_angles[k], 'energy': u[closest_indices].sum()})

    min_energy = min(state['energy'] for state in states)
    # Normalize energies and compute populations
    total_population = sum(np.exp(-(state['energy'] - min_energy)) for state in states)
    for state in states:
        state['population'] = np.exp(-(state['energy'] - min_energy)) / total_population

    # Compute J-couplings for each state using the Karplus relation
    for state in states:
        state['J'], state['diff_J'], state['diff2_J'] = get_scalar_couplings_with_derivatives(state['phi'], A0, B0, C0, phi0)

    # Compute True J-couplings
    pops = np.array([state["population"] for state in states])
    phi_angles = np.array([state["phi"] for state in states])


    w_phi_angles = np.array([w*(phi_angles[i]) for i,w in enumerate(pops)]).sum(axis=0)
    #print(w_phi_angles)

    # no difference due to symmetry
    #w_phi_angles = circular_weighted_avg(phi_angles, pops, axis=0)
    #print(w_phi_angles)
    #exit()

#    true_J, _, _ = get_scalar_couplings_with_derivatives(w_phi_angles, A, B, C, phi0)


    true_J = np.array([w*get_scalar_couplings_with_derivatives(phi_angles[i], A, B, C, phi0)[0] for i,w in enumerate(pops)]).sum(axis=0)

    #print(true_J)
    #exit()
    states = add_prior_error(states, σ_prior)

    if verbose: print(pd.DataFrame(states))
    return true_J, states

# }}}

# generate_toy_model:{{{
def generate_difficult_toy_model(nstates, Nd, initial_parameters, exp_parameters,
        extra_Nd, extra_parameters, phi0_deg=-60.0, verbose=False):
    """
    Generate a toy model to test J-coupling constants.

    :param nstates: Number of states.
    :param Nd: Number of J-coupling values for each state.
    :param A: Karplus parameter A.
    :param B: Karplus parameter B.
    :param C: Karplus parameter C.
    :param phi0_deg: phi0 angle in degrees.
    :return: Tuple containing the true experimental J-couplings and a list of dictionaries representing the states.
    """
    A, B, C = exp_parameters
    A0, B0, C0 = initial_parameters
    # Convert phi0 to radians
    phi0 = np.deg2rad(phi0_deg)
    states = [] # Initialize states list
    new_states = [] # Initialize states list
    # NOTE: Creating distribution of phi angles to sample from
    sigmas=(20, 10, 5)
    angles=(-110, -60, 60)
    weights=(0.40, 0.6, 0.0)
    dist_size = 10000
    nsamples = 1000
    # Validate that the weights sum to 1
    assert np.isclose(sum(weights), 1.0), "Weights should sum to 1"
    # Create empty list to store mixed angles
    mixed_angles = []
    # Generate the distributions
    distributions = []
    for i, angle in enumerate(angles):
        mean, std_dev = np.deg2rad(angle), np.deg2rad(sigmas[i])
        distributions.append(np.random.normal(mean, std_dev, dist_size))
    # Determine the number of samples to draw from each distribution
    samples_per_distribution = [int(weight * nsamples) for weight in weights]

    # Draw the samples
    for l, dist in enumerate(distributions):
        sampled_angles = np.random.choice(dist, samples_per_distribution[l], replace=False)
        mixed_angles.extend(sampled_angles)
    mixed_angles = np.array(mixed_angles)


    kT = 1.0
    phi = np.arange(-180.0, 180.0, 0.01)  # degrees
    probability = 0.0
    for l, dist in enumerate(distributions):
        probability += weights[l] * 1/np.sqrt(2*np.pi*sigmas[l]**2) *np.exp(-(phi - angles[l])**2/(2*sigmas[l]**2))
    u = -kT*np.log(probability)

#    fig, ax = plt.subplots()
#    ax.scatter(phi, u, s=1, color="k")
#    ax.set_ylabel(r"$E$", fontsize=14, labelpad=20, rotation=0)
#    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=14)
#    ax.set_ylim(0, 30)
#    fig.savefig("toy_model_figure_energies.png")
#    exit()

    # Draw the samples for each data point
    indices = np.random.choice(list(range(len(sigmas))), size=Nd, p=weights)
    _state_angles = []
    for l in indices:
        sampled_angles = np.random.choice(distributions[l], nstates, replace=False)
        _state_angles.append(sampled_angles + np.deg2rad(sigmas[l])*np.random.randn())

    state_angles = np.array(_state_angles).T
    for k in range(nstates):
        closest_indices = np.array([np.argmin(np.abs(phi - angle), axis=0) for angle in np.rad2deg(state_angles[k])])
        states.append({'phi': state_angles[k], 'energy': u[closest_indices].sum()})

    min_energy = min(state['energy'] for state in states)
    # Normalize energies and compute populations
    total_population = sum(np.exp(-(state['energy'] - min_energy)) for state in states)
    for state in states:
        state['population'] = np.exp(-(state['energy'] - min_energy)) / total_population

    # Compute J-couplings for each state using the Karplus relation
    for state in states:
        state['J'], state['diff_J'], state['diff2_J'] = get_scalar_couplings_with_derivatives(state['phi'], A0, B0, C0, phi0)

    # Compute True J-couplings
    pops = np.array([state["population"] for state in states])
    phi_angles = np.array([state["phi"] for state in states])
    w_phi_angles = np.array([w*(phi_angles[i]) for i,w in enumerate(pops)]).sum(axis=0)

    true_J, _, _ = get_scalar_couplings_with_derivatives(w_phi_angles, A, B, C, phi0)
    ############################################################################
    ############################################################################
    ############################################################################

    extra_mixed_angles = []
    extra_weights=(0.00, 0.0, 1.0)
    extra_samples_per_distribution = [] #(0, 0, extra_Nd)
    extra_distributions = []
    for i, angle in enumerate(angles):
        mean, std_dev = np.deg2rad(angle), np.deg2rad(sigmas[i])
        extra_distributions.append(np.random.normal(mean, std_dev, dist_size))
    extra_samples_per_distribution = [int(weight * nsamples) for weight in extra_weights]


    _indices = np.random.choice(list(range(len(sigmas))), size=extra_Nd, p=extra_weights)
    print(_indices)

    #extra_angles = []
    for l in _indices:
        extra_sampled_angles = np.random.choice(np.random.choice(extra_distributions[l], 1), nstates, replace=True)
        _state_angles.append(extra_sampled_angles)# + np.deg2rad(sigmas[l])*np.random.randn())

    new_states = states.copy()
    state_angles = np.array(_state_angles).T
    for k in range(nstates):
        new_states[k]['phi'] = state_angles[k]

    A0, B0, C0 = extra_parameters
    # Compute J-couplings for each state using the Karplus relation
    for state in new_states:
        for i in range(extra_Nd):
            _res = get_scalar_couplings_with_derivatives([state['phi'][Nd]], A0, B0, C0, phi0)
            #_res = get_scalar_couplings_with_derivatives([state['phi'][Nd+i]], A0, B0, C0, phi0)
            state['J'] = np.concatenate([state['J'], _res[0]], axis=0)
            state['diff_J'] = np.concatenate([state['diff_J'], _res[1]], axis=1)
            state['diff2_J'] = np.concatenate([state['diff2_J'], _res[2]], axis=1)

    phi_angles = np.array([state["phi"] for state in new_states])
    w_phi_angles = np.array([w*(phi_angles[i]) for i,w in enumerate(pops)]).sum(axis=0)

    sig = 0.5
    sig = 0.0
    for i in range(extra_Nd):
        A0_, B0_, C0_ = A0+sig*np.random.randn(), B0+sig*np.random.randn(), C0+sig*np.random.randn()
        extra_true_J, _, _ = get_scalar_couplings_with_derivatives([w_phi_angles[Nd]], A0_, B0_, C0_, phi0)
        #extra_true_J, _, _ = get_scalar_couplings_with_derivatives([w_phi_angles[Nd+i]], A0, B0, C0, phi0)


#        extra_true_J, _, _ = get_scalar_couplings_with_derivatives([new_states[]["phi"][Nd+i]], A0, B0, C0, phi0)
        #print(extra_true_J)
        true_J = np.append(true_J, extra_true_J)
    if verbose: print(pd.DataFrame(new_states))
    return true_J, new_states

# }}}

# circular_mean:{{{
def circular_mean(angles, weights=None, axis=0):
    """
    Compute the circular mean of a set of angles, optionally weighted.
    :param angles: Array of angles in radians.
    :param weights: Optional array of weights for each angle.
    :return: Circular mean of angles in radians.
    """
    if weights is None:
        weights = np.ones_like(angles)

    sum_sin = np.sum(np.sin(angles) * weights, axis=axis)
    sum_cos = np.sum(np.cos(angles) * weights, axis=axis)
    mean_angle = np.arctan2(sum_sin, sum_cos)

    return mean_angle
# }}}

# circular_weighted_avg:{{{
def circular_weighted_avg(angles, weights=None, axis=0):
    """
    Computes the circular average of a set of angles, taking into account the cyclical nature of angular measurements.
    This is particularly important for angles close to the wrapping point, such as -180 degrees and 180 degrees
    which are, in practice, the same direction.

    Parameters:
    - angles: an array of angles to be averaged, in degrees.
    - weights: an optional array of weights for each angle. If None, a simple (unweighted) average is calculated.

    Steps for calculation:
    1. Convert angles to radians for trigonometric functions.
    2. Decompose each angle into its vector components (x = cos(angle), y = sin(angle)).
    3. Compute the weighted (or unweighted if weights are None) average of the x and y components.
    4. Calculate the average angle using the arctan2 function to obtain the correct quadrant for the average vector.

    :param angles: Array of angles in radians.
    :param weights: Optional array of weights for each angle.
    :return: Circular mean of angles in radians.

    Returns:
    - The circular average of the angles, in degrees.
    """

    if weights is None:
        weights = np.ones_like(angles)

    sum_sin = np.array([np.sin(angles[i]) * weights[i] for i in range(len(weights))]).sum(axis=axis)
    sum_cos = np.array([np.cos(angles[i]) * weights[i] for i in range(len(weights))]).sum(axis=axis)
    mean_angle = np.arctan2(sum_sin, sum_cos)

    return mean_angle
# }}}

# get_Karplus_parameters_from_SVD:{{{
def get_Karplus_parameters_from_SVD(phis, j_values, phi0=np.radians(-60)):
    """
    Compute Karplus parameters using SVD.
    The Karplus relation has the general form:
        J(\phi) = A*cos^{2}(\phi + \phi_{0}) + B*cos(\phi + \phi_{0}) + C

    :param phis: List of dihedral angles in radians.
    :param j_values: List of corresponding J-coupling values.
    :param phi0: Phase shift in radians (default is -60 degrees in radians).
    :return: A tuple containing the Karplus parameters (A, B, C).
    """

    # Construct the matrix M
    M = np.vstack([
        np.cos(phis + phi0) ** 2,
        np.cos(phis + phi0),
        np.ones_like(phis)
    ]).T

    # Decompose the matrix M using SVD
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)

    # Compute the parameters
    params = Vt.T @ np.linalg.inv(np.diag(sigma)) @ U.T @ j_values

    return params[0], params[1], params[2]

# }}}

# get_Karplus_parameters_from_weighted_SVD:{{{
def get_Karplus_parameters_from_weighted_SVD(phis, exp, weights, phi0=np.radians(-60)):
    """
    Compute Karplus parameters using SVD with weighted averaging over states.
    The Karplus relation has the general form:
        J(\phi) = A*cos^2(\phi + phi0) + B*cos(\phi + phi0) + C

    :param phis: Matrix of dihedral angles in radians (nstates, nobs).
    :param exp: Array of ensemble-averaged J-coupling values (nobs).
    :param weights: Array of weights for each state (nstates).
    :param phi0: Phase shift in radians (default is -60 degrees in radians).
    :return: A tuple containing the Karplus parameters (A, B, C).
    """
    nstates, nobs = phis.shape

    # Initialize matrix M with zeros for the weighted average calculations
    M = np.zeros((nobs, 3))

    # Expand weights to match the repeated structure of phis
    weights_expanded = weights[:, np.newaxis]  # Shape (nstates, 1) to broadcast over columns

    # Compute weighted components for each observation across states
    for j in range(nobs):
        cos_squared = np.cos(phis[:, j] + phi0) ** 2
        cos_value = np.cos(phis[:, j] + phi0)

        M[j, 0] = np.sum(weights * cos_squared) / np.sum(weights)  # Weighted average of cos^2
        M[j, 1] = np.sum(weights * cos_value) / np.sum(weights)    # Weighted average of cos
        M[j, 2] = 1  # Constant term does not depend on phi

    # SVD decomposition of the matrix M
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)

    # Compute pseudo-inverse of sigma to avoid division by zero
    #sigma_inv = np.diag([1/s if s > 1e-10 else 0 for s in sigma])
    #sigma_inv = np.diag([1/(s + 1e-10) for s in sigma])
    sigma_inv = np.diag([1/(s + 1e-6) for s in sigma])

    # Compute parameters
    params = Vt.T @ sigma_inv @ U.T @ exp

    return params[0], params[1], params[2]

# }}}

# fit_karplus_parameters with SSE: {{{
def sse(params, phis, exp, weights, phi0=np.deg2rad(60)):
    """
    Objective function to minimize: the sum of squared differences between
    predicted and experimental J-couplings.
    """
    A, B, C = params
    j_predicted = np.array([get_scalar_couplings_with_derivatives(angles, A, B, C, phi0)[0]*weights[i] for i,angles in enumerate(phis)]).sum(axis=0)
    result = np.sum((j_predicted - exp) ** 2)
    return result

def fit_karplus_parameters(initial_guess, phis, exp, weights, phi0=np.deg2rad(60)):
    """
    Fit Karplus parameters to experimental J-couplings using nonlinear optimization.

    :param initial_guess: Initial guess for the parameters
    :param phis: Array of dihedral angles in radians from all conformational states.
    :param exp: Ensemble-averaged experimental J-coupling values.
    :return: Optimized Karplus parameters (A, B, C).
    """
    #result = scipy.optimize.minimize(sse, initial_guess, args=(phis, exp, weights, phi0), method='L-BFGS-B')
    result = scipy.optimize.minimize(sse, initial_guess, args=(phis, exp, weights, phi0), method="Nelder-Mead")
    A_opt, B_opt, C_opt = np.array(result['x'])
    return np.array([A_opt, B_opt, C_opt])


# }}}


# chain_statistics:{{{
def chain_statistics(chains):
    """
    Compute MCMC statistics across chains. This includes averages, std,
    Gelman-Rubin statistic and other diagnostics for a set of chains.
    https://bookdown.org/rdpeng/advstatcomp/monitoring-convergence.html#gelman-rubin-statistic

    Parameters:
    - chains: A 3D numpy array of shape (num_chains, num_samples, num_parameters)

    Returns:
    - A dictionary with diagnostics including R_hat, means, SEM, and standard deviations.
    """
    n = chains.shape[1]  # Number of samples per chain
    m = chains.shape[0]  # Number of chains
    p = chains.shape[2]  # Number of parameters

    # Compute within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    avg_uncert = np.ones((p, m))
    for i in range(m):
        for k in range(p):
            avg_uncert[k,i] = chains[i,:,k].std()
    avg_uncert = avg_uncert.mean(axis=1)

    # Compute between-chain variance
    theta_bar = np.mean(chains, axis=1)  # Mean of each chain

#    # Reflect B if positive
#    for i in range(len(theta_bar)):
#        if theta_bar[i,1] > 0:
#            theta_bar[i,1] *= -1

    theta_double_bar = np.mean(theta_bar, axis=0)  # Mean over all chains
    B = n * np.var(theta_bar, axis=0, ddof=1)  # Variance of chain means, multiplied by n

    # Estimate of marginal posterior variance
    var_theta = (1 - 1.0 / n) * W + (1.0 / n) * B

    # Compute potential scale reduction factor (R_hat)
    R_hat = np.sqrt(var_theta / W)

    # Compute standard deviation over all chains
    std_dev = np.sqrt(var_theta)

    # Compute SEM over all chains
    SEM = std_dev / np.sqrt(n * m)

    # Compute mean over all chains
    mean_over_all_chains = theta_double_bar

    # Compute SEM over all chains
    SEM = std_dev / np.sqrt(n * m)

    diagnostics = {
        "mean_over_all_chains": mean_over_all_chains,
        "mean_of_each_chain": theta_bar,
        "R_hat": R_hat,
        "SEM_over_all_chains": SEM,
        "std_dev_over_all_chains": std_dev,
        "avg_uncert": avg_uncert
    }

    return diagnostics
# }}}



# biceps for SVD:{{{
def biceps_for_SVD(parameters, ensemble, fwd_model_function, phi0, phi_angles, restraint_index, verbose, PSkwargs, sample_kwargs):
    if not np.isfinite(parameters).all():
        print("Initial parameters contain NaNs or infs!")
        return np.nan
    if np.any(parameters) == np.nan: return np.nan

    print("parameters: ",parameters)
    A, B, C = parameters
    if restraint_index != None:
        result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
        model_J, diff_model_J, diff2_model_J = result
        exp = []
        for l in range(len(ensemble.expanded_values)): # thermodynamic ensembles
            #if l == 0: continue
            for s in range(len(ensemble.ensembles[l].ensemble)): # conformational states
                for r in range(len(ensemble.ensembles[l].ensemble[s])): # data restraint types
                    if r != restraint_index: continue
                    for j in range(len(ensemble.ensembles[l].ensemble[s][r].restraints)): # data points (observables)
                        ensemble.ensembles[l].ensemble[s][r].restraints[j]["model"] = model_J[s][j]
                        if (l == 0) and (s == 0):
                            exp.append(ensemble.ensembles[l].ensemble[s][r].restraints[j]["exp"])
        #print(exp)
    sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
    return sampler

#[ 8.31  9.27 10.04 10.    9.3   8.93  4.15  8.63  7.26  9.67  9.91  9.62
#   9.95  9.77  9.82  9.56  7.62  5.52  7.57  3.94  5.5   5.72  4.28  4.57
#   5.55  5.77  3.82  7.13  9.54  7.98  5.15  9.45  8.41  9.81  9.66 10.27
#   8.73  6.47  9.41  6.97  7.11  8.06  2.48  9.59  9.2   3.7   3.7   4.14
#   9.45  7.26  7.34  9.01  2.36  7.19  5.63  9.82  9.88  9.7   9.59  9.88
#   8.3   7.52  6.67]


# }}}


# biceps for test:{{{
def biceps_for_test(parameters, ensemble, fwd_model_function, phi0, phi_angles, restraint_index, verbose, PSkwargs, sample_kwargs, remove_indices=[]):
    if not np.isfinite(parameters).all():
        print("Initial parameters contain NaNs or infs!")
        return np.nan
    if np.any(parameters) == np.nan: return np.nan

    print("parameters: ")
    print(parameters)
    A, B, C = parameters
    if restraint_index != None:
        result = fwd_model_function(phi_angles, A=A, B=B, C=C, phi0=phi0)
        model_J, diff_model_J, diff2_model_J = result
        exp = []
        for l in range(len(ensemble.expanded_values)): # thermodynamic ensembles
            for s in range(len(ensemble.ensembles[l].ensemble)): # conformational states
                for r in range(len(ensemble.ensembles[l].ensemble[s])): # data restraint types
                    if r != restraint_index: continue
                    if remove_indices != []:
                        ensemble.ensembles[l].ensemble[s][r].restraints = copy.deepcopy(np.delete(ensemble.ensembles[l].ensemble[s][r].restraints, remove_indices, axis=0))

                    for j in range(len(ensemble.ensembles[l].ensemble[s][r].restraints)): # data points (observables)
                        ensemble.ensembles[l].ensemble[s][r].restraints[j]["model"] = model_J[s][j]
                        if (l == 0) and (s == 0):
                            exp.append(ensemble.ensembles[l].ensemble[s][r].restraints[j]["exp"])

    sampler = run_biceps(ensemble, PSkwargs=PSkwargs, sample_kwargs=sample_kwargs)
    return sampler

#[ 8.31  9.27 10.04 10.    9.3   8.93  4.15  8.63  7.26  9.67  9.91  9.62
#   9.95  9.77  9.82  9.56  7.62  5.52  7.57  3.94  5.5   5.72  4.28  4.57
#   5.55  5.77  3.82  7.13  9.54  7.98  5.15  9.45  8.41  9.81  9.66 10.27
#   8.73  6.47  9.41  6.97  7.11  8.06  2.48  9.59  9.2   3.7   3.7   4.14
#   9.45  7.26  7.34  9.01  2.36  7.19  5.63  9.82  9.88  9.7   9.59  9.88
#   8.3   7.52  6.67]


# }}}


# compute dihedrals from MDAnalysis:{{{
def calculate_phi_dihedral(phi_atoms):
    """Calculate dihedral angle in degrees given four atoms."""
    p0 = phi_atoms[0].position
    p1 = phi_atoms[1].position
    p2 = phi_atoms[2].position
    p3 = phi_atoms[3].position
    dihedral_rad = calc_dihedrals(p0, p1, p2, p3)
    dihedral_deg = np.degrees(dihedral_rad)
    return dihedral_deg

def set_phi(u, residue, phi_perturbation):
    """Set the phi angle of a given residue to new_phi degrees."""
    # Define atoms for the phi dihedral: C-N-CA-C
    # Note: You may need to adjust atom selection based on your structure
    prev_residue = residue.resid - 1

    if prev_residue < 0:
        print("Cannot set phi for the first residue.")
        return

    # Adjusted atom selection to ensure it forms a single AtomGroup
    phi_atoms = u.select_atoms(
        f"(resid {prev_residue} and name C) or "
        f"(resid {residue.resid} and name N) or "
        f"(resid {residue.resid} and name CA) or "
        f"(resid {residue.resid} and name C)"
    )
    current_phi = calculate_phi_dihedral(phi_atoms)

    new_phi = current_phi + phi_perturbation

    # Calculate the difference and convert to radians
    phi_diff = np.radians(new_phi) - np.radians(current_phi)

    # Select the atoms to rotate: All atoms in the current and subsequent residues
    #atoms_to_rotate = u.select_atoms(f"resid {residue.resid} and (name N or name CA or name C or name O or name CB) or resid {residue.resid+1} onwards")

    # Assuming 'u' is your MDAnalysis Universe object and 'target_resid' is the residue ID from which you want to select atoms onwards
    max_resid = max(residue.resid for residue in u.residues)
    atoms_to_rotate = u.select_atoms(f"resid {residue.resid}:{max_resid}")


    # Define rotation axis as the vector between CA and C atoms of the current residue
    axis = phi_atoms.positions[2] - phi_atoms.positions[3]  # CA-C vector
    point = phi_atoms.positions[3]  # C atom position

    # Apply rotation
    atoms_to_rotate.rotateby(phi_diff, axis, point=point)
    return u
# }}}


# plot_landscape code:{{{

def get_kernel(kernel_idx):

    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct

    if kernel_idx == -2:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.) + WhiteKernel(noise_level=1e-5)

    elif kernel_idx == -1:
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-5)

    elif kernel_idx == 0:

        # You can adjust the sigma_0 parameter which represents σ²_b, the initial variance or intercept term.
        linear_kernel = DotProduct(sigma_0=0.5)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-5)
        kernel = linear_kernel + kernel

    elif kernel_idx == 1:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)

    elif kernel_idx == 2:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) #+ Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0))

    elif kernel_idx == 3:
        #kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-2)
        #kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)

    elif kernel_idx == 4:
        #kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-2)
        #kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1) + Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2)
        kernel = RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.1, 10.0), alpha=0.1)*(Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1e-2))

    elif kernel_idx == 5:
        kernel = (Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-2))*RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))

    else:
        raise ValueError("Invalid kernel index")

    return kernel



def plot_landscape(results, figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=0):

    from scipy.interpolate import interp2d
    from scipy.interpolate import griddata
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100


    # heatmap_function:{{{
    def generate_heatmap(ax, x, y, score, gridpoints=100, lvls=50, upper_xy_lim=None, show_colorbar=1, kernel_idx=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from matplotlib import ticker
        from sklearn.gaussian_process.kernels import Matern

        if isinstance(upper_xy_lim, (list, tuple, np.ndarray)):
            max_x = upper_xy_lim[0]
            max_y = upper_xy_lim[1]
        else:
            max_x =max(x)
            max_y = max(y)

        #for gridpoints in range(2, 25):
        x_grid = np.linspace(min(x), max_x, gridpoints)
        y_grid = np.linspace(min(y), max_y, gridpoints)
        X, Y = np.meshgrid(x_grid, y_grid)

        kernel = get_kernel(kernel_idx)

        gp = GaussianProcessRegressor(kernel=kernel)

        X_train = np.vstack([x, y]).T
        gp.fit(X_train, score)

        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        Z, std = gp.predict(X_test, return_std=True)
        Z = Z.reshape(X.shape)
        print(gridpoints, Z.min(), Z.max())
        #exit()



        cmap = plt.cm.coolwarm
        #cmap = plt.cm.RdBu_r

        min_score = min(score)
        max_score = max(score)
        levels = np.linspace(min_score, max_score, lvls)
        norm = matplotlib.colors.Normalize(vmin=min_score, vmax=max_score)

        cont = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
        # Add contour lines with dark color and increased width
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.0, alpha=0.6)


        if show_colorbar:
            cbar = plt.colorbar(cont, ax=ax, extend='both')

            # Specify the tick locations
            tick_locator = ticker.MaxNLocator(nbins=10)
            cbar.locator = tick_locator

            # Format the tick labels
            tick_formatter = ticker.FormatStrFormatter("%.1f")
            cbar.formatter = tick_formatter
            cbar.ax.tick_params(labelsize=14)

            #cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)
            cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)
        return cont
    # }}}

    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)  # Subplot for A vs C
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # Subplot for B vs C
    #ax4 = plt.subplot(gs[1, 2])              # subplot for karplus curves

    axs = [ax1, ax2, ax3]#, ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        #ax.grid(alpha=0.5, linewidth=0.5)


    # Heatmap + Quiver for A vs B
    cont = generate_heatmap(ax1, results["A"], results["B"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for A vs C
    generate_heatmap(ax2, results["A"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for B vs C
    generate_heatmap(ax3, results["B"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)


    ax1_pos = ax1.get_position()
    cbar_ax = fig.add_axes([ax1_pos.width+0.10, ax1_pos.y0+0.025, 0.02, ax1_pos.height-0.025])
    #cbar_ax = fig.add_axes([0.5, 0.60, 0.02, 0.9])
#    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar = plt.colorbar(cont, cax=cbar_ax, orientation='vertical')

    # Specify the tick locations
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator

    # Format the tick labels
    tick_formatter = ticker.FormatStrFormatter("%.1f")
    cbar.formatter = tick_formatter
    cbar.ax.tick_params(labelsize=14)

    cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)

    res = results.iloc[np.where(results["score"].to_numpy() == results["score"].to_numpy().min())[0]]

    print(f"Score Min: {results['score'].min()}")
    print(f"Score Max: {results['score'].max()}")
    print(f"Lowest BICePs score at: {res}")

    final_parameters = [float("%0.3g"%res["A"].to_numpy()[-1]), float("%0.3g"%res["B"].to_numpy()[-1]), float("%0.3g"%res["C"].to_numpy()[-1])]

    # Get the y-axis limits
    y_min, y_max = ax1.get_ylim()
    # Calculate the range of the y-axis
    y_range = y_max - y_min
    # Define the offset as a fraction of the y-axis range
    offset_fraction = 0.0025  # Adjust this value as needed
    offset = offset_fraction * y_range

    ax1.set_xlabel(r"$A$", fontsize=16)
    ax1.set_ylabel(r"$B$", fontsize=16)
    ax2.set_xlabel(r"$A$", fontsize=16)
    ax2.set_ylabel(r"$C$", fontsize=16)
    ax3.set_xlabel(r"$B$", fontsize=16)
    ax3.set_ylabel(r"$C$", fontsize=16)

    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# plot_landscape_with_curve:{{{
def plot_landscape_with_curve(results, figname=None, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=0):

    from scipy.interpolate import interp2d
    from scipy.interpolate import griddata
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)

    #facecolors = ["white"] + colors[:len(results["score"].unique()) - 1]
    facecolors = ["white"] + colors[:len(results["A"]) - 1]

    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)  # Create a grid with 2 rows and 2 columns
    marker_size = 50
    main_marker_size = 100


    # heatmap_function:{{{
    def generate_heatmap(ax, x, y, score, gridpoints=100, lvls=50, upper_xy_lim=None, show_colorbar=1, kernel_idx=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from matplotlib import ticker
        from sklearn.gaussian_process.kernels import Matern

        if isinstance(upper_xy_lim, (list, tuple, np.ndarray)):
            max_x = upper_xy_lim[0]
            max_y = upper_xy_lim[1]
        else:
            max_x = max(x)
            #max_x = max_x - max_x*0.025
            max_y = max(y)
            #max_y = max_y - max_y*0.025

        #for gridpoints in range(2, 25):
        x_grid = np.linspace(min(x), max_x, gridpoints)
        y_grid = np.linspace(min(y), max_y, gridpoints)
        X, Y = np.meshgrid(x_grid, y_grid)

        kernel = get_kernel(kernel_idx)

        gp = GaussianProcessRegressor(kernel=kernel)

        X_train = np.vstack([x, y]).T
        gp.fit(X_train, score)

        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        Z, std = gp.predict(X_test, return_std=True)
        Z = Z.reshape(X.shape)
        print(gridpoints, Z.min(), Z.max())
        #exit()



        cmap = plt.cm.coolwarm
        #cmap = plt.cm.RdBu_r

        min_score = min(score)
        max_score = max(score)
        levels = np.linspace(min_score, max_score, lvls)
        norm = matplotlib.colors.Normalize(vmin=min_score, vmax=max_score)

        cont = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
        ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
        # Add contour lines with dark color and increased width
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.0, alpha=0.6)


        if show_colorbar:
            cbar = plt.colorbar(cont, ax=ax, extend='both')

            # Specify the tick locations
            tick_locator = ticker.MaxNLocator(nbins=10)
            cbar.locator = tick_locator

            # Format the tick labels
            tick_formatter = ticker.FormatStrFormatter("%.1f")
            cbar.formatter = tick_formatter
            cbar.ax.tick_params(labelsize=14)

            #cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)
            cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)
        return cont
    # }}}

    ax1 = plt.subplot(gs[0, 0])              # Subplot for A vs B
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)  # Subplot for A vs C
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # Subplot for B vs C
    ax4 = plt.subplot(gs[0, 1])              # subplot for karplus curves

    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        #ax.grid(alpha=0.5, linewidth=0.5)

    # Heatmap + Quiver for A vs B
    cont = generate_heatmap(ax1, results["A"], results["B"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for A vs C
    generate_heatmap(ax2, results["A"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)

    # Heatmap + Quiver for B vs C
    generate_heatmap(ax3, results["B"], results["C"], results["score"], gridpoints=gridpoints, lvls=lvls, upper_xy_lim=upper_xy_lim, show_colorbar=0, kernel_idx=kernel_idx)


    ax1_pos = ax1.get_position()
    #cbar_ax = fig.add_axes([ax1_pos.width+0.10, ax1_pos.y0+0.025, 0.02, ax1_pos.height-0.025])
    #cbar = plt.colorbar(cont, cax=cbar_ax, orientation='vertical')

    cbar_ax = fig.add_axes([ax1_pos.x0+0.025, ax1_pos.y0 + ax1_pos.height + 0.01, ax1_pos.width-0.05, 0.02])
    cbar = plt.colorbar(cont, cax=cbar_ax, orientation='horizontal')

    # Specify the tick locations
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator

    # Format the tick labels
    tick_formatter = ticker.FormatStrFormatter("%.1f")
    cbar.formatter = tick_formatter
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)

    cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)

    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')


    res = results.iloc[np.where(results["score"].to_numpy() == results["score"].to_numpy().min())[0]]

    print(f"Score Min: {results['score'].min()}")
    print(f"Score Max: {results['score'].max()}")
    print(f"Lowest BICePs score at: {res}")

    final_parameters = [float("%0.3g"%res["A"].to_numpy()[-1]), float("%0.3g"%res["B"].to_numpy()[-1]), float("%0.3g"%res["C"].to_numpy()[-1])]

    # Get the y-axis limits
    y_min, y_max = ax1.get_ylim()
    # Calculate the range of the y-axis
    y_range = y_max - y_min
    # Define the offset as a fraction of the y-axis range
    offset_fraction = 0.0025  # Adjust this value as needed
    offset = offset_fraction * y_range

    ax1.set_xlabel(r"$A$", fontsize=16)
    ax1.set_ylabel(r"$B$", fontsize=16)
    ax2.set_xlabel(r"$A$", fontsize=16)
    ax2.set_ylabel(r"$C$", fontsize=16)
    ax3.set_xlabel(r"$B$", fontsize=16)
    ax3.set_ylabel(r"$C$", fontsize=16)

    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    if figname != None: fig.savefig(figname, dpi=600)
    return fig
# }}}

# plot_landscapes:{{{
def plot_landscapes(sampler, J_type, chain=0):

    traj = sampler.traj[chain].__dict__["trajectory"]
    energies = [traj[i][1] for i in range(len(traj))]
    npz = sampler.traj[chain].traces[-1]
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()
    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])//3

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=nrows, ncols=3)
    gs = np.concatenate(gs)

    for r in range(len(data[0])):
        results = pd.DataFrame(data[:,r,:], columns=["A", "B", "C"])
        results["score"] = energies
        figname = f"{outdir}/contour_{J_type[r]}.png"
        plot_landscape(results, figname=figname, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=-1)

# }}}


# plot_fmp_posterior_histograms:{{{

def plot_fmp_posterior_histograms(sampler, k_labels=[], chain=0, outdir="./", figsize=(14,8), plot_type="step"):

    data = sampler.fmp_traj[chain]
    npz = sampler.traj[chain].traces[-1]
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()
    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])//3
    if nrows == 0: nrows = 1

    ncols=3

    if k_labels == []: k_labels = ["" for i in range(nrows*ncols)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    print(gs)

    for r in range(len(data[0])):
        row_idx = r // ncols
        col_idx = r % ncols

        results = pd.DataFrame(data[:,r,:], columns=[string.ascii_uppercase[q] for q in range(data.shape[-1])])
        #ax = fig.add_subplot(gs[row_idx, col_idx])
        ax = fig.add_subplot(gs[row_idx,col_idx])
        for i in range(data.shape[2]):
            labels = ["A", "B", "C"]+[string.ascii_uppercase[q] for q in range(3,26)]
            colors = ["r", "b", "g"]+[biceps.toolbox.mpl_colors[::2][q] for q in range(3,26)]
            if plot_type == "hist":
                ax.hist(data[:,r,i], alpha=0.5, label=labels[i], color=colors[i], edgecolor="k", bins="auto")

#            if i == 1:
#                ax.hist(-data[:,r,i], alpha=0.5, label="__no_legend__", color=colors[i], edgecolor="k", bins="auto")

            elif plot_type == "step":
                counts, bin_edges = np.histogram(data[:,r,i], bins="auto")
                try:
                    ax.step(bin_edges[:-1], counts, '%s-'%colors[i])
                except(Exception) as e:
                    ax.step(bin_edges[:-1], counts, 'k-')
                ax.fill_between(bin_edges[:-1], counts, color=colors[i], label=labels[i], step="pre", alpha=0.4)
            else:
                print("`plot_type` needs to be either 'hist' or 'step'")
                exit()


        ax.set_title(r"${^{3}\!J}_{%s}$"%k_labels[r], fontsize=18)
        if r >= 3: ax.set_xlabel("Karplus coefficients [Hz]", fontsize=16)
        if (r == 0) or (r==3): ax.set_ylabel('Density', fontsize=16)
        # Increase the number of x ticks and improve their readability
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Adjust number of x ticks
        if (r == 0): ax.legend()
        #ax.set_xlim(data.min(), data.max())
        ax.set_ylim(bottom=0)

        ax.text(-0.12, 1.02, string.ascii_lowercase[r],
            transform=ax.transAxes, size=18, weight='bold')


    fig.savefig(f"{outdir}/karplus_coefficients_histograms.png")
    return fig



# }}}

# plot_fmp_traces:{{{
def plot_fmp_traces(sampler, k_labels=[], chain=0, outdir="./", figsize=(14,8)):
    data = sampler.fmp_traj[chain]
    npz = sampler.traj[chain].traces[-1]
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()
    traj = sampler.traj[chain].__dict__
    x = np.array([traj["trajectory"][i][0] for i in range(len(traj["trajectory"]))])

    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])//3
    if nrows == 0: nrows = 1
    ncols=3

    if k_labels == []: k_labels = ["" for i in range(nrows*ncols)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for r in range(len(data[0])):
        row_idx = r // ncols
        col_idx = r % ncols


        results = pd.DataFrame(data[:,r,:], columns=[string.ascii_uppercase[q] for q in range(data.shape[-1])])
        #ax = fig.add_subplot(gs[row_idx, col_idx])
        ax = fig.add_subplot(gs[row_idx,col_idx])
        for i in range(data.shape[2]):
            labels = ["A", "B", "C"]+[string.ascii_uppercase[q] for q in range(3,26)]
            colors = ["r", "b", "g"]+[biceps.toolbox.mpl_colors[::2][q] for q in range(3,26)]
            ax.plot(x, data[:,r,i], label=labels[i], color=colors[i])

        ax.set_title(r"${^{3}\!J}_{%s}$"%k_labels[r], fontsize=18)
        if r >= 3: ax.set_xlabel("steps", fontsize=16)
        if (r == 0) or (r==3): ax.set_ylabel("Karplus coefficients [Hz]", fontsize=16)
        # Increase the number of x ticks and improve their readability
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Adjust number of x ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Adjust number of x ticks
        if (r == 0): ax.legend()
        #ax.set_ylim(data[:,r,i].min(), data[:,r,i].max())
        #ax.set_xlim(0, len(data[:,r,i]))
        ax.text(-0.12, 1.02, string.ascii_lowercase[r],
            transform=ax.transAxes, size=18, weight='bold')



    #fig.set_figwidth(8)
    #fig.set_figheight(12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/karplus_coefficients_traces.png")
    return fig
# }}}



# plot_data_errors_with_likelihood:{{{
def plot_data_errors_with_likelihood(sampler, outdir="./"):
    npz = sampler.traj[0].traces[-1]
    columns = sampler.rest_type
    df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
    df0 = df.transpose()

    nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=nrows, ncols=2)
    counter = 0
    r = 0
    for k in range(len(df0.columns.to_list())):
        col = df0.columns.to_list()[k]
        allowed = np.array(sampler.traj[0].allowed_parameters[k])
        sampled = np.array(sampler.traj[0].sampled_parameters[k])
        if all(allowed == np.ones(allowed.shape)): continue
        if all(allowed == np.zeros(allowed.shape)): continue

        row_idx, col_idx = divmod(counter, 2)
        _ = fig.add_subplot(gs[row_idx, col_idx])

        _.step(allowed, sampled, 'b-', label='exp')
        _.fill_between(allowed, sampled, color='b', step="pre", alpha=0.4, label=None)
        _.set_xlim(left=0, right=df0["%s"%(col)].max()*1.1)
        if "phi" in col:
            _.set_xlim(left=df0["%s"%(col)].min(), right=df0["%s"%(col)].max()*1.1)

        number = int("".join([char for char in col if char.isdigit()]))
        if "J" in col:
            col = col.replace(str(number), k_labels[r])
        label = biceps.toolbox.format_label(col)
        label_fontsize = 16
        _.set_xlabel(r"%s"%label, fontsize=label_fontsize)
        _.set_ylabel(r"$P$(%s)"%label, fontsize=label_fontsize)
        counter += 1
        if "phi" in col: r += 1

    fig.set_figwidth(8)
    fig.set_figheight(12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/marginal_distributions_.png")
# }}}


# check convergence:{{{
def check_convergence(sampler):

    data_types = np.unique([re.sub(r'\d+', '', dtype).replace("sigma_","")
                  for dtype in sampler.rest_type if "sigma" in dtype])
    traj = sampler.traj[-1].__dict__
    converge_dir = f"{outdir}/convergence_results"
    biceps.toolbox.mkdir(converge_dir)
    traj = sampler.traj[-1].__dict__
    C = biceps.Convergence(traj, outdir=converge_dir)
    #biceps.toolbox.save_object(C, filename=outdir+"/convergence.pkl")
    C.plot_traces(figname="traces.pdf", xlim=(0, traj["trajectory"][-1][0]), figsize=(8,30))
    #exit()

    #C.get_autocorrelation_curves(method="block-avg-auto", nblocks=3, maxtau=500, figsize=(8,30))
    C.get_autocorrelation_curves(method="block-avg-auto", nblocks=3, maxtau=100, figsize=(8,30))
    nblock=5
    nfold=5 #10
#    nfold=3 #10
    nround=100
    C.process(nblock=nblock, nfold=nfold, nround=nround, plot=1)


# }}}







if __name__ == "__main__":
#    avg_biceps_phi_angles = []
#    pops = [[0.1, 0.2, 0.2, 0.5], [0.2, 0.3, 0.1, 0.4]]
#    angles = np.deg2rad(np.linspace(-180, 180, 100))
#    b4_phi_angles = np.array([np.random.choice(angles, 50, replace=False) for i in range(len(pops[0]))])
#    for k in range(len(pops)):
#        biceps_phi_angles = circular_weighted_avg(b4_phi_angles, weights=pops[k], axis=0)
#        avg_biceps_phi_angles.append(biceps_phi_angles)
#    print(np.rad2deg(avg_biceps_phi_angles))
#    print(np.rad2deg(avg_biceps_phi_angles).shape)
#    avg_biceps_phi_angles = np.rad2deg(circular_weighted_avg(np.array(avg_biceps_phi_angles), axis=0))
#    print(avg_biceps_phi_angles)
#    print(avg_biceps_phi_angles.shape)
#    exit()


    generate_toy_model(nstates=5, Nd=20, initial_parameters=[4,-2,0], exp_parameters=[4,-2,0], phi0_deg=-60.0, σ_prior=0.0, verbose=False)
    exit()



    import scipy, gc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    outdir = "/Users/rr/github/FwdModelOpt/Ubq_results/ncluster_500/GB_single_sigma/50000_steps_1_replicas_0_lam_opt_J_and_NOE"
    outdir = "/Users/rr/github/FwdModelOpt/Ubq_results/ncluster_500/GB_single_sigma/100000_steps_1_replicas_2_lam_opt_J_and_NOE"
    outdir = "/Users/rr/github/FwdModelOpt/Ubq_results/ncluster_500/GB_single_sigma/30000_steps_2_replicas_16_lam_opt"
    outdir = "/Users/rr/github/FwdModelOpt/Ubq_results/ncluster_500/GB_single_sigma/30000_steps_1_replicas_15_lam_opt"
    outdir = "/Users/rr/github/FwdModelOpt/toy_model_test/_GB_single_sigma/100_states_50001_steps_100_replicas_1_lam_[7.13,-1.31,1.56]_as_ref_and_true_being_[6.51,-1.76,1.6]"
    #file = f"{outdir}/fmp_trajectory.npy"
    #data = np.load(file)
    #print(data.shape)

    sampler = biceps.toolbox.load_object(f"{outdir}/sampler_obj.pkl")
#    plot_fmp_posterior_histograms(sampler, outdir=outdir)
#    plot_fmp_traces(sampler, outdir=outdir)
    data = sampler.fmp_traj
    print(data.shape)
    check_convergence(sampler)
    exit()

    #marginal_parameters = sampler.traj[0].traces[-1]


## previous main:{{{
#
#if __name__ == "__main__":
#    import scipy, gc
#    import pandas as pd
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    outdir = "./"
#    xi_values = np.load("xi_values.npy")[::-1]
#    u_kln = np.load("u_kln.npy")
#    new_xi_values = optimize_xi_values(xi_values, u_kln, outdir, nsteps=100000, tol=1e-7, alpha=1e-5, print_every=1000, make_plots=True)
#    print(xi_values)
#    print(new_xi_values)
#    exit()
#
#
#
#    phi0 = -60
#    #phi0 = 0
#    exp_parameters = [8.4, -1.3, 0.4]
#    #A, B, C = exp_parameters
#    initial_parameters = [7.02, -2.3, 1.0]
#    true_J, states = generate_toy_model(nstates=100, Nd=20,
#                            initial_parameters=initial_parameters,
#                            exp_parameters=exp_parameters,
#                            phi0_deg=phi0, verbose=False)
#
#    phi_angles = np.array([state["phi"] for state in states])
#    pops = np.array([state["population"] for state in states])
#    energies = np.array([state["energy"] for state in states])
#    model_J = np.array([state["J"] for state in states])
#    diff_model_J = np.array([state["diff_J"] for state in states])
#    diff2_model_J = np.array([state["diff2_J"] for state in states])
#    x = model_J
#    exp = true_J
#
#    pd.DataFrame(states)
#
#    w_phi_angles = np.array([w*(np.rad2deg(phi_angles[i])) for i,w in enumerate(pops)]).sum(axis=0)
#    print(w_phi_angles)
#    print(exp)
#    fig = plot_hist_of_phi_angles(phi_angles, exp_parameters, phi0)
#    ax = fig.axes[0]
#    for i,w in enumerate(pops):
#        X = np.rad2deg(phi_angles[i])
#        Y = model_J[i]
#        ax.scatter(X, Y, c="orange", edgecolor="black")
#
#    ax.scatter(w_phi_angles, exp, c="red", edgecolor="black")
#    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=14)
#
#    fig = ax.get_figure()
#    fig.savefig("testing.png")
#    exit()
#
#
## }}}
#




















