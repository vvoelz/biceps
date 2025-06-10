"""
A toy model system for forward model optimization of Karplus parameters. In
this example, we simultaneously infer the joint posterior of 100 conformational
state populations as well as the optimal Karplus parameters using 60 J-coupling
observables. We run 4 independent chains in parallel, each starting from different
initial parameters, and see that they all converge to the same location in
parameter space. Using the average parameters over these 4 chains as our optimal
Karplus parameters, we quantify the BICePs score, $f_{xi=0 -> 1}$
represented as the free energy of "turning on" the data restraints. The energy
separation between end states is rather large, so we apply the pylambdaopt module
to determine the optimal positioning of our intermediates to increase the quality
of our free energy calculation.
"""

# libraries:{{{
import biceps
import FwdModelOpt_routines as fmo
import scipy, gc
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('axes', labelsize=16)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats, copy, matplotlib
#from matplotlib import gridspec
from scipy.optimize import fmin
import matplotlib.gridspec as gridspec
from matplotlib import ticker
# }}}

# Weight data:{{{
class WeightData(object):
    def __init__(self, Nd, extra_Nd, nstates, weight=1.0):
        self.Nd = Nd
        self.extra_Nd = extra_Nd
        self.weight = weight
        self.weight_J = np.concatenate([np.ones((nstates, self.Nd)), np.ones((nstates, extra_Nd))*self.weight], axis=1)
        self.weight_dJ = np.array([np.concatenate([np.ones((nstates, self.Nd)), np.ones((nstates, extra_Nd))*self.weight], axis=1),
                                   np.concatenate([np.ones((nstates, self.Nd)), np.ones((nstates, extra_Nd))*self.weight], axis=1),
                                   # NOTE: this next one is always 1... is this okay?
                                   np.concatenate([np.ones((nstates, self.Nd)), np.ones((nstates, extra_Nd))*self.weight], axis=1)
                                   ])
        self.weight_d2J = np.zeros(self.weight_dJ.shape)


    def get_scalar_couplings_with_derivatives(self, phi, A, B, C, phi0):
        """Return a scalar couplings with a given choice of karplus coefficients.  USES RADIANS!"""
        J, dJ, d2J = fmo.get_scalar_couplings_with_derivatives(phi, A=A, B=B, C=C, phi0=phi0)
        J, dJ, d2J = J*self.weight_J, dJ*self.weight_dJ, d2J*self.weight_d2J
        return J, dJ, d2J
# }}}

# Append to Database:{{{
def append_to_database(dbName="database.pkl", verbose=False, **kwargs):
    data = pd.DataFrame()
    for arg in kwargs:
        #print(arg)
        data[arg] = [kwargs.get(arg)]

    # NOTE: Saving results to database
    if os.path.isfile(dbName):
       db = pd.read_pickle(dbName)
    else:
        if verbose: print("Database not found...\nCreating database...")
        db = pd.DataFrame()
        db.to_pickle(dbName)
    db = db.append(data, ignore_index=True)
    db.to_pickle(dbName)
    #gc.collect()

# }}}

# plot_landscape:{{{

def get_kernel(kernel_idx):

    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct

    if kernel_idx == -2:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)#+ WhiteKernel(noise_level=1e-8)

    elif kernel_idx == -1:
        kernel = 2.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1e-5)

    elif kernel_idx == 0:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0))
        #kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)
#        kernel = RBF(length_scale=0.5, length_scale_bounds=(0.1, 10.0)) + Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=0.1)

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
        ax.grid(alpha=0.5, linewidth=0.5)


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

    #cbar.ax.set_ylabel(r'$u$', fontsize=22, rotation=0, labelpad=10)
    cbar.ax.set_ylabel(r'$f$', fontsize=22, rotation=0, labelpad=10)

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


generate_data = 1
run_sampling = 1
run_xi_optimization = 1


nreplicas, nsteps = 100, 50001
nreplicas, nsteps = 32, 50000


swap_every = 0

plot_landscapes = 1
landscape_stride = 5

lambda_values = np.array([0.0])
n_lambdas = len(lambda_values)
progress_bar = 0
multiprocess=1
verbose=True

stat_model, data_uncertainty="GB", "single"
#stat_model, data_uncertainty="Bayesian", "single"
#stat_model, data_uncertainty="Students", "single"
#stat_model, data_uncertainty="Gaussian", "multiple"
data_likelihood = "gaussian" #"log normal" # "gaussian"
#data_likelihood = "log normal" # "gaussian"

attempt_move_state_every = 1
attempt_move_sigma_every = 1
attempt_move_fmp_every = 1
write_every = 10

burn = 20000
#burn = 50000
#burn = 10000
#burn = 0


if stat_model == "GB":
    #phi,phi_index=(1.0, 2.0, 1),0
    phi,phi_index=(1.0, 10.0, 1000),0
    #phi,phi_index=(1.0, 10.0, 10000),0
    beta,beta_index=(1.0, 2.0, 1),0
elif stat_model == "Students":
    phi,phi_index=(1.0, 2.0, 1),0
    beta,beta_index=(1.0, 100.0, 1000),0
else:
    phi,phi_index=(1.0, 2.0, 1),0
    beta,beta_index=(1.0, 2.0, 1),0

#sigMin,sigMax,dsig = 0.001,50, 1.02
sigMin,sigMax,dsig = 0.001,100, 1.02

arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
l = len(arr)
sigma_index = round(l*0.73)
print(arr[sigma_index])

local_vars = locals()
PSkwargs = biceps.toolbox.get_PSkwargs(local_vars, exclude=["ensemble"])
PSkwargs["verbose"] = False
sample_kwargs = biceps.toolbox.get_sample_kwargs(local_vars)
sample_kwargs["verbose"] = False
sample_kwargs["progress"] = progress_bar


#PSkwargs["change_xi_every"] = round(nsteps/11)
#PSkwargs["dXi"] = 0.1
#PSkwargs["xi_integration"] = True

all_data = 0
J_only = 1

parameter_sets = biceps.J_coupling.J3_HN_HA_coefficients
models = list(parameter_sets.keys())
# ['Ruterjans1999', 'Bax2007', 'Bax1997', 'Habeck', 'Vuister', 'Pardi']
model_idx = 5
print(f"models: {models}")
ref_model = models[4]
initial_parameters = [val for key,val in parameter_sets[ref_model].items() if key != "phi0"]
#initial_parameters =  [1.0, -1.05, 0.10] # randomly chosem
phi0 = parameter_sets[ref_model]["phi0"] # in radians
#phi0 = 0
print(f"initial_parameters = {initial_parameters}; with phi0 = {phi0}")
# Habeck = {'A': 7.13, 'B': -1.31, 'C': 1.56}
# 'Bax2007'= {'A': 8.4, 'B': -1.36, 'C': 0.33}
exp_parameters = [val for key,val in parameter_sets[models[model_idx]].items() if key != "phi0"]
print(f"True parameters = {exp_parameters}; with phi0 = {phi0}")
#exit()

main_dir = f"toy_model_test"
#main_dir = f"toy_model_test_prior"
biceps.toolbox.mkdir(main_dir)

#nstates = 10
nstates = 100
#nstates = 500
#nstates = 5
σ_prior=0.0
σ_prior=10.0
# IMPORTANT: Adding experimental error
# random and systematic error in the data
μ_data=dict(domain=(0.25,2.25), frac_of_data=0.20)
σ_data=0.50

#μ_data=dict(domain=(0.25,2.25), frac_of_data=0.0)
#σ_data=0.50

# no error in the data
μ_data=dict(domain=(0.25,2.25), frac_of_data=0.0)
σ_data=0.00


Nd = [5, 0] # [J, noe]
Nd = [10, 0] # [J, noe]
Nd = [20, 10] # [J, noe]
Nd = [60, 0] # [J, noe]
#Nd = [40, 0] # [J, noe]
#Nd = [100, 0] # [J, noe]
#Nd = [40, 0] # [J, noe]
#Nd = [10, 20] # [J, noe]

extra_Nd=0
syn_data_weight = 1.0
wd = WeightData(Nd[0], extra_Nd, nstates, weight=syn_data_weight)

outdir = f'{main_dir}/_{stat_model}_{data_uncertainty}_sigma/{nstates}_states_{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam_{initial_parameters}_as_ref_and_true_being_{exp_parameters}'.replace(" ", "")
#outdir = f'{main_dir}/_{stat_model}_{data_uncertainty}_sigma'
data_dir = main_dir+f"/data"
data_dir = data_dir.replace(" ", "")
biceps.toolbox.mkdir(data_dir)
print(f"data_dir: {data_dir}")
print(f"outdir: {outdir}")
biceps.toolbox.mkdir(outdir)

dbName = f"{outdir}/database.pkl"


A, B, C = exp_parameters


if generate_data:
    use_difficult_toy_model = 0
    if use_difficult_toy_model:
        toy_model = fmo.generate_difficult_toy_model(nstates, Nd[0],
                        initial_parameters=initial_parameters,
                        exp_parameters=exp_parameters,
                        #extra_Nd=extra_Nd, extra_parameters=[6.40, -1.40, 1.9],
                        extra_Nd=extra_Nd, extra_parameters=[6.51,-1.76,1.6],
                        phi0_deg=np.rad2deg(phi0))
    else:
        toy_model = fmo.generate_toy_model(nstates, Nd[0],
                        initial_parameters=initial_parameters,
                        exp_parameters=exp_parameters, σ_prior=σ_prior,
                        phi0_deg=np.rad2deg(phi0))

if generate_data:
    biceps.toolbox.save_object(toy_model, f"{outdir}/toy_model_obj.pkl")

toy_model = biceps.toolbox.load_object(f"{outdir}/toy_model_obj.pkl")
true_J, states = toy_model

# NOTE: IMPORTANT: Adding experimental error (random & systematic)
if generate_data:
    data_obj = fmo.add_error_to_data(true_J,
            μ_data=μ_data, σ_data=σ_data, verbose=True)
    biceps.toolbox.save_object(data_obj, f"{outdir}/data_obj.pkl")

data_obj = biceps.toolbox.load_object(f"{outdir}/data_obj.pkl")
_,exp,diff,sigma = data_obj

try:
   true_phi_angles = np.array([state["true phi"] for state in states])
except(Exception) as e:
   true_phi_angles = np.array([state["phi"] for state in states])

phi_angles = np.array([state["phi"] for state in states])
pops = np.array([state["population"] for state in states])
energies = np.array([state["energy"] for state in states])
model_J = np.array([state["J"] for state in states])
diff_model_J = np.array([state["diff_J"] for state in states])
diff2_model_J = np.array([state["diff2_J"] for state in states])
x = model_J
#exit()


print(f"Experimental errors: {diff}")
print(f"N data points perturbed: {np.count_nonzero(diff)}")
print(f"Experimental couplings: {_}")
print(f"Experimental couplings: {exp}")
print(f"Experimental sigma: {sigma}")
#exit()

w_phi_angles = np.array([w*(np.rad2deg(phi_angles[i])) for i,w in enumerate(pops)]).sum(axis=0)
true_w_phi_angles = np.array([w*(np.rad2deg(true_phi_angles[i])) for i,w in enumerate(pops)]).sum(axis=0)
prior_sigma = np.sqrt(metrics.mean_squared_error(w_phi_angles, true_w_phi_angles))
print(f"Prior sigma: {prior_sigma}")

#print(w_phi_angles)
#print(exp)
fig = fmo.plot_hist_of_phi_angles(phi_angles, exp_parameters, np.rad2deg(phi0))
ax = fig.axes[0]
for i,w in enumerate(pops):
    #X = np.rad2deg(phi_angles[i][:Nd[0]])
    X = np.rad2deg(phi_angles[i])
    Y = model_J[i]
#    ax.scatter(X, Y, c="orange", edgecolor="black")

#print(w_phi_angles)
#print(exp)
ax.scatter(true_w_phi_angles[:Nd[0]], exp[:Nd[0]], c="red", edgecolor="black")
ax.scatter(true_w_phi_angles[Nd[0]:], exp[Nd[0]:], c="green", edgecolor="black")
ax.set_xlabel(r"$\phi$ (degrees)", fontsize=14)
fig.savefig(f"{outdir}/hist_of_phi_angles.png")
#exit()

################################################################################
#parameters_for_states = []
#for i,angles in enumerate(phi_angles[:,:Nd[0]]):
#    A, B, C = fmo.get_Karplus_parameters_from_SVD(angles, exp[:Nd[0]])
#    parameters_for_states.append(np.array([A,B,C]))
#parameters_for_states = np.array(parameters_for_states)

parameters_for_states = []
for i,angles in enumerate(phi_angles):
    A, B, C = fmo.get_Karplus_parameters_from_SVD(angles, exp)
    parameters_for_states.append(np.array([A,B,C]))
parameters_for_states = np.array(parameters_for_states)
################################################################################


# create J-coupling data
for i in range(len(pops)):
    model = pd.read_pickle("template.noe")
    _model = pd.DataFrame()
    for j in range(Nd[0]+extra_Nd):
        model["atom_index3"], model["atom_index4"] = i,j
        model["restraint_index"], model["model"], model["exp"] = j+1, x[i,j], exp[j]
        _model = pd.concat([_model,model], ignore_index=True)
    print(_model)
    _model.to_pickle(data_dir+"/%s.J"%i)
if all_data:
    # create NOE data
    noe_model, noe_exp, noe_diff = fmo.get_noe_data(pops, Nd[1], μ_data=0.0, σ_data=0.0, verbose=False)
    for i in range(len(pops)):
        model = pd.read_pickle("template.noe")
        _model = pd.DataFrame()
        for j in range(Nd[1]):
            model["restraint_index"], model["model"], model["exp"] = j+1, noe_model[i,j], noe_exp[j]
            _model = pd.concat([_model,model], ignore_index=True)
        _model.to_pickle(data_dir+"/%s.noe"%i)


optfile = f"{outdir}/optimization.csv"

print(data_dir)
if all_data: input_data = biceps.toolbox.sort_data(f'{data_dir}')
if J_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/*.J')]

options = biceps.get_restraint_options(input_data)
if all_data:
    options[0].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
    options[1].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
else:
    options[0].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))


# if there are more than 1 data type, we need to find the index of J-coupling
restraint_index = int(np.where(".J" == np.array(biceps.toolbox.list_extensions(input_data)))[0])
print(pd.DataFrame(options))

################################################################################

iteration = 0 # Counter.
n_converged = 0 # Counter. Increased if parameter values are within threshold.
is_converged = False

marginal_parameters_trace = []
sem_trace = []
populations_trace = []
reweighted_J = []

# Construct the initial ensemble
ensemble = biceps.ExpandedEnsemble(energies, lambda_values=lambda_values)
ensemble.initialize_restraints(input_data, options)

init_parameters = initial_parameters.copy()
A, B, C = init_parameters
parameter_optimization = [init_parameters]


results = []
results.append({
    "A": init_parameters[0],
    "B": init_parameters[1],
    "C": init_parameters[2],
    "score": np.nan,
    "dscore_A": np.nan,
    "dscore_B": np.nan,
    "dscore_C": np.nan,
    "A sigma": np.nan,
    "B sigma": np.nan,
    "C sigma": np.nan,
    })


#stat_model, data_uncertainty="Students", "single"
#phi,phi_index=(1.0, 2.0, 1),0
#beta,beta_index=(1.0, 100.0, 1000),0


options = biceps.get_restraint_options(input_data)
for i in range(len(options)):
    options[i].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))


###############################################################################
###############################################################################
###############################################################################
# Construct the initial ensemble

#fwd_model_paras = [[6, -1, 2]]
fwd_model_paras = [[9, -1, 1]]
#fwd_model_paras = [[6.51,-1.76,1.6]]
#fwd_model_paras = [[6.7,-1.8,1.7]]
#fwd_model_paras = [[7.1,-1.8,1.7]]

fwd_model_paras = [[[9, -1, 1]],
                   [[4, 0, 3]],
                   [[0, 0, 0]],
                   [[1, -1, 2]]]
fwd_model_paras = np.array(fwd_model_paras)

_lambda_values = [0.0]*len(fwd_model_paras)
print(_lambda_values)

_ensemble = biceps.ExpandedEnsemble(energies, lambda_values=_lambda_values)
_ensemble.initialize_restraints(input_data, options)

#_ensemble = change_weight_of_syn_data(_ensemble, Nd, extra_Nd, restraint_index, weight=syn_data_weight)

_PSkwargs = biceps.toolbox.get_PSkwargs(local_vars, exclude=["ensemble"])
_sample_kwargs = biceps.toolbox.get_sample_kwargs(local_vars)
_sample_kwargs["progress"] = 1#progress_bar
_sample_kwargs["verbose"] = 0
_sample_kwargs["nsteps"] = nsteps
_PSkwargs["xi_integration"] = 0
_PSkwargs["nreplicas"] = nreplicas


#_args = [init_parameters, _ensemble, fmo.get_scalar_couplings_with_derivatives,
_args = [init_parameters, _ensemble, wd.get_scalar_couplings_with_derivatives,
        phi0, phi_angles, restraint_index, verbose, _PSkwargs, _sample_kwargs]


#sampler = fmo.biceps_for_SVD(*_args)
phi0 = [phi0]
opts = pd.DataFrame(options)
restraint_indices = [i for i,val in enumerate(opts["extension"].to_numpy()) if f"J" in val]
#print(restraint_indices)


#print(phi0)
#exit()

#_ensemble.fmo_restraint_indices = restraint_indices
#_ensemble.phi_angles = [phi_angles]
#_ensemble.phase_shifts = phi0
#_ensemble.fwd_model_parameters = fwd_model_paras

parameter_priors = np.array([["Gaussian" for i in range(len(fwd_model_paras[0][0]))]
                    for k in range(len(fwd_model_paras[0]))])
#print(parameter_priors)
#print(parameter_priors.shape)
#exit()

kwargs = {"phi0": phi0}
_ensemble.initialize_fwd_model(init_paras=fwd_model_paras, x=[phi_angles], indices=restraint_indices,
                               min_max_paras=None, parameter_priors=parameter_priors, **kwargs)


_PSkwargs = biceps.toolbox.get_PSkwargs(local_vars, exclude=["ensemble"])
_sample_kwargs = biceps.toolbox.get_sample_kwargs(local_vars)
_sample_kwargs["progress"] = 1#progress_bar
_sample_kwargs["verbose"] = 0
_sample_kwargs["attempt_lambda_swap_every"] = swap_every
#_sample_kwargs["print_freq"] = 1000
_sample_kwargs["burn"] = burn
_sample_kwargs["nsteps"] = nsteps
_PSkwargs["xi_integration"] = 0
_PSkwargs["nreplicas"] = nreplicas
_PSkwargs["fmo"] = 1
_PSkwargs["fmo_method"] = "SGD"
#_PSkwargs["fmo_method"] = "gaussian"
#_PSkwargs["fmo_method"] = "adam"
#_PSkwargs["fmo_method"] = "uniform"

#_PSkwargs["sem_method"] = "sumDev"
##_PSkwargs["sem_method"] = "sem"

if run_sampling:
    sampler = fmo.run_biceps(_ensemble, PSkwargs=_PSkwargs, sample_kwargs=_sample_kwargs)
    biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")
else:
    sampler = biceps.toolbox.load_object(f"{outdir}/sampler_obj.pkl")

print(sampler.acceptance_info)
print(sampler.exchange_info)
sampler.plot_exchange_info(xlim=(-100, nsteps), figname=f"{outdir}/lambda_swaps.png")

a = biceps.Analysis(sampler, outdir=outdir, MBAR=False)
a.plot_acceptance_trace()
a.plot_energy_trace()

print("Nd = ",sampler.Nd)
print("You started with:")
print(fwd_model_paras)
chains = sampler.fmp_traj
new_parameters = []
new_parameters_sigma = []
for ct in range(chains.shape[2]):
    #gr_input = chains[:,:,ct,:].reshape((chains.shape[0], chains.shape[1], chains.shape[3]))
    gr_input = chains[:,:,ct,:]
    print(gr_input.shape)
    chain_stats = fmo.chain_statistics(gr_input)
    R_hat = chain_stats["R_hat"]
    print("R_hat: ",R_hat)
    print(chain_stats["mean_of_each_chain"])
    mean = chain_stats["mean_over_all_chains"]
    new_parameters.append(mean)
    std_dev = chain_stats["std_dev_over_all_chains"]
    new_parameters_sigma.append(std_dev)
    print(["%0.2g"%mean[i] + "±%0.2g"%std_dev[i] for i in range(len(mean))])
    print("################################")
    RMSE = np.sqrt(metrics.mean_squared_error(exp_parameters, new_parameters[ct]))
    print("sumDev RMSE: ", RMSE)
#exit()

#################################################################################
#################################################################################
#################################################################################





data = sampler.fmp_traj[0]
print(data.shape)


traj = sampler.traj[0].__dict__["trajectory"]
energies = [traj[i][1] for i in range(len(traj))]
#print(sampler.traj[0].__dict__['trajectory_headers'])
#exit()

_ = data.reshape((data.shape[1], data.shape[0], data.shape[2]))[0]
results = pd.DataFrame(_, columns=["A", "B", "C"])
print(results)
results["score"] = np.array(energies)/nreplicas
print(results)
figname = f"{outdir}/contour.png"
if plot_landscapes: plot_landscape(results[::landscape_stride], figname=figname, gridpoints=100, lvls=50, upper_xy_lim=None, kernel_idx=-1)


###############################################################################
from plot_marginal_likelihood_for_each_iteration import plot_marginal_likelihood
populations = sampler.populations[-1]
marginal_parameters = sampler.traj[0].traces[-1]
print(f"Marginal likelihood parameters: {marginal_parameters}")
sem_df = sampler.get_sem_trace_as_df()[0].iloc[[-1]]
#sem_trace = np.sum(sem_df[sem_df.keys()[restraint_index]].to_numpy()[0])

columns = sampler.rest_type
sigma_indices = [i for i,col in enumerate(columns) if "sigma" in col]
phi_indices = [i for i,col in enumerate(columns) if "phi" in col]
sem_trace = [np.max(sem_df[sem_df.keys()[i]].to_numpy()[0]) for i in range(len(sigma_indices))]

likelihood_parameters = [{"sigmaB":marginal_parameters[sigma_indices[i]], "sem":sem_trace[i], "phi":marginal_parameters[phi_indices[i]]} for i in range(len(sigma_indices))]
figures = []
for i in restraint_indices:
    A, B, C = np.mean(data[:,i], axis=0)
    result = fmo.get_scalar_couplings_with_derivatives(phi_angles, A=A, B=B, C=C, phi0=phi0[i])
    model_J, diff_model_J, diff2_model_J = result
    weights = populations
    reweighted_J = np.array([model_J[w]*weights[w] for w in range(len(weights))]).sum(axis=0)
    devs = np.abs(exp - reweighted_J)
    print(devs)
    print(likelihood_parameters[i])
    try:
        fig = plot_marginal_likelihood(likelihood_parameters[i], devs, xlim=(0,2))
    except(Exception) as e:
        print(e)
    figname=f"{outdir}/opt_marginal_likelihood_{i}.png"
    fig.savefig(figname)

###############################################################################

npz = sampler.traj[0].traces[-1]
columns = sampler.rest_type
df = pd.DataFrame(np.array(sampler.traj[0].traces).transpose(), columns)
df0 = df.transpose()



nrows = np.sum([1 for col in df0.columns.to_list() if "sigma" in col])

fig = plt.figure()
gs = gridspec.GridSpec(nrows=nrows, ncols=2)
counter = 0
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
    label = biceps.toolbox.format_label(col)
    label_fontsize = 16
    _.set_xlabel(r"%s"%label, fontsize=label_fontsize)
    _.set_ylabel(r"$P$(%s)"%label, fontsize=label_fontsize)
    counter += 1


fig.set_figwidth(8)
fig.set_figheight(6)
fig.tight_layout()
fig.savefig(f"{outdir}/marginal_distributions_.png")



plot_type = "step"
for r in range(len(data[0])):
    fig, ax = plt.subplots()
    for i in range(data.shape[2]):
        labels = ["A", "B", "C"]
        colors = ["r", "b", "g"]
        if plot_type == "hist":
            ax.hist(data[:,r,i], alpha=0.5, label=labels[i], color=colors[i], edgecolor="k", bins="auto")
        elif plot_type == "step":
            counts, bin_edges = np.histogram(data[:,r,i], bins="auto")
            ax.step(bin_edges[:-1], counts, '%s-'%colors[i])
            ax.fill_between(bin_edges[:-1], counts, color=colors[i], label=labels[i], step="pre", alpha=0.4)
        else:
            print("`plot_type` needs to be either 'hist' or 'step'")
            exit()


    ax.set_ylim(bottom=0)
    #ax.set_title(r"${^{3}\!J}$", fontsize=18)
    ax.set_xlabel("Karplus coefficents [Hz]", fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    # Increase the number of x ticks and improve their readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust number of x ticks
    ax.legend()
    #ax.set_xlim(data.min(), data.max())
    fig.tight_layout()
    fig.savefig(f"{outdir}/karplus_coefficents_.png")

    #new_parameters = np.mean(data[:,r], axis=0)
    #new_parameters_sigma = np.std(data[:,r], axis=0)
    #print(["%0.2g"%new_parameters[i] + "±%0.2g"%new_parameters_sigma[i] for i in range(len(new_parameters))])

#######################################################
    fig = fmo.plot_hist_of_phi_angles(phi_angles, exp_parameters, np.rad2deg(phi0[r]))
    ax = fig.axes[0]

    angles = np.deg2rad(np.linspace(-180, 180, 100))

    # NOTE:
    # randomly select 90% of the data without replacement
    # predict SVD 1000 times and take std to get error bars
    num = round(len(exp[:Nd[0]])*0.90)
    indices = np.arange(len(exp[:Nd[0]]))
    res = []
    for i in range(1000):
        selected_indices = np.random.choice(indices, size=num, replace=False)
        A_svd, B_svd, C_svd = fmo.get_Karplus_parameters_from_SVD(np.deg2rad(w_phi_angles[:Nd[0]][selected_indices]), exp[:Nd[0]][selected_indices], phi0=phi0[r])
        res.append(np.array([A_svd, B_svd, C_svd]))
    res = np.array(res)
    A_svd, B_svd, C_svd = np.mean(res, axis=0)
    svd_parameters_sigma = np.std(res, axis=0)
    RMSE = np.sqrt(metrics.mean_squared_error(exp_parameters, [A_svd, B_svd, C_svd]))
    print("SVD RMSE: ", RMSE)
    print("SVD parameters: ", [A_svd, B_svd, C_svd])
    print("SVD sigma: ", svd_parameters_sigma)

    y1 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=[A_svd, B_svd, C_svd][0]+np.nan_to_num(svd_parameters_sigma[0]),
            B=[A_svd, B_svd, C_svd][1]+np.nan_to_num(svd_parameters_sigma[1]),
            C=[A_svd, B_svd, C_svd][2]+np.nan_to_num(svd_parameters_sigma[2]), phi0=phi0[r])[0]
    y2 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=[A_svd, B_svd, C_svd][0]-np.nan_to_num(svd_parameters_sigma[0]),
            B=[A_svd, B_svd, C_svd][1]-np.nan_to_num(svd_parameters_sigma[1]),
            C=[A_svd, B_svd, C_svd][2]-np.nan_to_num(svd_parameters_sigma[2]), phi0=phi0[r])[0]
    #ax.fill_between(np.rad2deg(angles), y1, y2, color="gray", alpha=0.5, label="__no_legend__")
    ax.fill_between(np.rad2deg(angles), y1, y2, color="orange", alpha=0.75, label="SVD")

    ###########################################################################
    ###########################################################################
    y1 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=new_parameters[r][0]+np.nan_to_num(new_parameters_sigma[r][0]),
            B=new_parameters[r][1]+np.nan_to_num(new_parameters_sigma[r][1]),
            C=new_parameters[r][2]+np.nan_to_num(new_parameters_sigma[r][2]), phi0=phi0[r])[0]
    y2 = fmo.get_scalar_couplings_with_derivatives(angles,
            A=new_parameters[r][0]-np.nan_to_num(new_parameters_sigma[r][0]),
            B=new_parameters[r][1]-np.nan_to_num(new_parameters_sigma[r][1]),
            C=new_parameters[r][2]-np.nan_to_num(new_parameters_sigma[r][2]), phi0=phi0[r])[0]
    ax.fill_between(np.rad2deg(angles), y1, y2, color="blue", alpha=0.5, label="BICePs")
    ###########################################################################
    ###########################################################################



#    # NOTE:
#    # randomly select 90% of the data without replacement
#    # predict SVD 1000 times and take std to get error bars
#    num = round(len(exp[:Nd[0]])*0.90)
#    indices = np.arange(len(exp[:Nd[0]]))
#    res = []
#    for i in range(1000):
#        selected_indices = np.random.choice(indices, size=num, replace=False)
#        #A_svd, B_svd, C_svd = fmo.get_Karplus_parameters_from_SVD(np.deg2rad(w_phi_angles[:Nd[0]][selected_indices]), exp[:Nd[0]][selected_indices], phi0=phi0[r])
#        initial_guess = [1.0, -1.0, 1.0]
#        A_svd, B_svd, C_svd = fmo.fit_karplus_parameters(initial_guess, np.deg2rad(w_phi_angles[:Nd[0]][selected_indices]), exp[:Nd[0]][selected_indices], weights=populations, phi0=phi0[r])
#        res.append(np.array([A_svd, B_svd, C_svd]))
#    res = np.array(res)
#    A_svd, B_svd, C_svd = np.mean(res, axis=0)
#    svd_parameters_sigma = np.std(res, axis=0)
#    y1 = fmo.get_scalar_couplings_with_derivatives(angles,
#            A=[A_svd, B_svd, C_svd][0]+np.nan_to_num(svd_parameters_sigma[0]),
#            B=[A_svd, B_svd, C_svd][1]+np.nan_to_num(svd_parameters_sigma[1]),
#            C=[A_svd, B_svd, C_svd][2]+np.nan_to_num(svd_parameters_sigma[2]), phi0=phi0[r])[0]
#    y2 = fmo.get_scalar_couplings_with_derivatives(angles,
#            A=[A_svd, B_svd, C_svd][0]-np.nan_to_num(svd_parameters_sigma[0]),
#            B=[A_svd, B_svd, C_svd][1]-np.nan_to_num(svd_parameters_sigma[1]),
#            C=[A_svd, B_svd, C_svd][2]-np.nan_to_num(svd_parameters_sigma[2]), phi0=phi0[r])[0]
#    ax.fill_between(np.rad2deg(angles), y1, y2, color="gray", alpha=0.5, label="SSE")



    ax.scatter(true_w_phi_angles[:Nd[0]], exp[:Nd[0]], c="red", edgecolor="black")
    ax.scatter(true_w_phi_angles[Nd[0]:], exp[Nd[0]:], c="green", edgecolor="black")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels[0] = 'True'  # Change the (index) label
    ax.legend(handles, labels)


    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=14)
    ax.set_ylabel(r"$J$ (Hz)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{outdir}/opt_with_phi_angles.png")


print("angles = ",angles)


check_biceps_score = 1
if check_biceps_score:
    if run_xi_optimization:
        lambda_values = np.array([0.0])
        energies = np.ones(nstates)/nstates
        _ensemble = biceps.ExpandedEnsemble(energies, lambda_values=lambda_values)
        _ensemble.initialize_restraints(input_data, options)
        A0, B0, C0 = new_parameters[0]
        for state in states:
            state['J'], state['diff_J'], state['diff2_J'] = fmo.get_scalar_couplings_with_derivatives(np.array(state['phi']), A0, B0, C0, phi0=phi0[0])

        model_J = np.array([state["J"] for state in states])

        # replace the old model data with the optimized fwd model data
        for l in range(len(_ensemble.ensembles)):
            for s in range(len(_ensemble.ensembles[l].ensemble)): # conformational states
                for r in range(len(_ensemble.ensembles[l].ensemble[s])): # data restraint types
                    if r == restraint_index:
                        for i in range(len(_ensemble.ensembles[l].ensemble[s][r].restraints)):
                            _ensemble.ensembles[l].ensemble[s][r].restraints[i]['model'] = model_J[s][i]


        local_vars = locals()
        _PSkwargs = biceps.toolbox.get_PSkwargs(local_vars, exclude=["ensemble"])
        _sample_kwargs = biceps.toolbox.get_sample_kwargs(local_vars)
        _sample_kwargs["progress"] = 1#progress_bar
        _sample_kwargs["verbose"] = 0
        _sample_kwargs["nsteps"] = nsteps
        #_sample_kwargs["burn"] = 0
        _sample_kwargs["burn"] = 1000
    #    _sample_kwargs["burn"] = 5000
        _PSkwargs["xi_integration"] = 1
        _PSkwargs["nreplicas"] = nreplicas
        _PSkwargs["fmo"] = 0
        #_PSkwargs["change_xi_every"] = round(nsteps/11)
        #_PSkwargs["dXi"] = 0.1
        _PSkwargs["num_xi_values"] = 11

        score = fmo.xi_integration(_ensemble, _PSkwargs, _sample_kwargs, plot_overlap=True, outdir=f"{outdir}",
                       optimize_xi_values=1, optimize_nXis=0, xi_opt_steps=5000000, tol=1e-7, alpha=1e-5, progress=1,
                       #optimize_xi_values=0, optimize_nXis=0, xi_opt_steps=5000000, tol=1e-7, alpha=1e-5, progress=1,
                       max_attempts=5, print_every=1000, scale_energies=False, verbose=False, save_sampler_obj=1)
        print(fmo.sampler.xi_values)
        print(f"BICePs score: {score}")


    ###########################################################################
    ###########################################################################
    ###########################################################################
    # NOTE: make main figure;
    """
    - plot energy traces for both
    """
    figname = f"{outdir}/fmo_xi_optimization.png"
    files = [f"{outdir}/sampler_BS_overlap.pkl", f"{outdir}/sampler_BS_overlap_after.pkl"]
    opt_xis = biceps.toolbox.load_object(f"{outdir}/opt_xis.pkl")


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import AutoMinorLocator
    label_fontsize=14
    matrix_label_fontsize=9.0
    legend_fontsize=12
    # Create a figure
#    fig = plt.figure(figsize=(10, 6))
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1])
    # Create each subplot
    ax1 = fig.add_subplot(gs[0, 0])  # Top left (wider)
    ax2 = fig.add_subplot(gs[0, 1])  # Top right (narrower)
    # Create a nested GridSpec within the bottom left cell (gs[1, 0])
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,  # 2 rows, 1 column
        subplot_spec=gs[1, 0],
        hspace=0  # No horizontal space between inner subplots
    )

    # Create subplots within the nested GridSpec with shared x-axis
    ax3a = fig.add_subplot(inner_gs[0, 0])  # Top subplot in the nested grid
    ax3b = fig.add_subplot(inner_gs[1, 0], sharex=ax3a)  # Bottom subplot sh
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right (narrower)

    #ax1.set_title("Top Left (wider)")
    #ax2.set_title("Top Right (narrower)")
    #ax3.set_title("Bottom Left (wider)")
    #ax4.set_title("Bottom Right (narrower)")

    # Adjust aspect ratios to make the subplots have equal height and width
#    for ax in [ax3a, ax3b]:
#        ax.set_aspect('equal')


    for f,file in enumerate(files):
        sampler = biceps.toolbox.load_object(file)
        mbar = sampler.integrate_xi_ensembles(multiprocess=1, progress=1, compute_derivative=0, plot_overlap=0, filename=None, scale_energies=0)

        # NOTE: make overlap matrix
        overlap = mbar.compute_overlap()
        overlap_matrix = overlap["matrix"]

        ti_info = sampler.ti_info
        xi_trace = np.array(sampler.xi_schedule)
        overlap_matrix = overlap["matrix"]
        #force_constants = [r"$\xi=%0.2f$"%float(value) for value in xi_trace]
        force_constants = [r"$%0.2f$"%float(value) for value in xi_trace]
        masked_overlap_matrix = np.ma.masked_array(overlap_matrix, mask=(overlap_matrix == 0))
        cmap = plt.cm.viridis_r.copy()
        ax = [ax2,ax4][f]
        im = ax.pcolor(masked_overlap_matrix, edgecolors='k', linewidths=2, cmap=cmap)
        # Add annotations
        for i in range(len(overlap_matrix)):
            for j in range(len(overlap_matrix[i])):
                value = overlap_matrix[i][j]
                if value >= 0.01:
                    #text_color = 'white' if value < 0.5 else 'black'
                    text_color = 'white' if value > 0.5 else 'black'
                    ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color=text_color,
                            fontsize=matrix_label_fontsize)  # Adjust fontsize as desired
                elif abs(i - j) == 1:  # Check if the element is adjacent to the diagonal
                    # Add a white rectangle with a black outline
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black', lw=1.5)
                    ax.add_patch(rect)
                    # Add black text for values less than 0.01
                    ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color='black', fontsize=matrix_label_fontsize)
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white'))
        titles = [r'Initial $\xi$-values', r'Optimized $\xi$-values']
        ax.set_title(titles[f], y=0.9, fontsize=label_fontsize)
        # Set tick positions and labels
        ax.set_xticks(np.arange(len(force_constants)) + 0.5, minor=False)
        ax.set_yticks(np.arange(len(force_constants)) + 0.5, minor=False)
        ax.set_xlabel(r"$\xi$", fontsize=label_fontsize+2)
        ax.set_ylabel(r"$\xi$", fontsize=label_fontsize+2)
        ax.set_xticklabels(force_constants, rotation=90, size=label_fontsize)
        ax.set_yticklabels(force_constants, size=label_fontsize)
        ax.tick_params(axis='x', direction='inout')
        ax.tick_params(axis='y', direction='inout')
        #ax.set_xticklabels(ax.get_xticklabels(), ha='left')
        #ax.set_yticklabels(ax.get_yticklabels(), va='bottom')
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), va='center')
        if f ==0:
            divider = make_axes_locatable(ax)
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            #cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal', pad=0.1)
            cbar.ax.tick_params(labelsize=label_fontsize-2)
            cbar.ax.xaxis.set_ticks_position('top')  # Move ticks to the top
            cbar.ax.xaxis.set_label_position('top')  # Move the label to the top

    #        cbar.set_label("Overlap probability between states", size=16)  # Set the colorbar label


        # NOTE: Make energy traces
        #A = biceps.Analysis(sampler, outdir=outdir, MBAR=False)
        #figures,steps,dists = A.plot_energy_trace()

        ax = ax1 #[ax2,ax4][f]

        traj = [traj.__dict__ for traj in sampler.traj]
        for i in range(len(sampler.expanded_values)):
            traj_steps = np.array(traj[i]['trajectory'], dtype=object).T[0]
            energy = np.array(traj[i]['trajectory'], dtype=object).T[1]
            c = ax.plot(traj_steps, energy/sampler.nreplicas, color=["r", "g"][f], label=[r"Old $\xi$",r"New $\xi$"][f])
            ax.legend(fontsize=legend_fontsize, loc="best")
            ax.set_yscale('log')
            ax.set_ylabel("u (kT)", fontsize=label_fontsize+2)
            ax.set_xlabel("Number of Steps", fontsize=label_fontsize)
            ax.tick_params(which="major", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax.tick_params(which="minor", axis="x", direction="inout", left=0, bottom=1, right=0, top=0)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.grid()



    y_spl = opt_xis["y_spl"]
    x_range = opt_xis["x_range"]
    xi_values = opt_xis["old_xis"]
    new_Xis = opt_xis["new_Xis"]
#    print(x_range)
#    print(y_spl(x_range))

    ax3a.plot(x_range, y_spl(x_range), 'b-', label="spline")
    ax3a.plot(xi_values, y_spl(np.array(xi_values)), 'r.', label=r"Old $\xi$")
    for value in xi_values:
        ax3a.plot([value, value], [0, y_spl(value)], 'r-')
    ax3a.legend(fontsize=legend_fontsize, loc="best")
    ax3a.set_xlabel(r'$\xi$', fontsize=label_fontsize+2)
#    ax3a.set_ylabel('Thermodynamic\nlength, $l$', fontsize=label_fontsize)
    ax3a.set_title(r'Initial $\xi$-values', y=0.8, fontsize=label_fontsize)

    ax3b.plot(x_range, y_spl(x_range), 'b-', label="spline")
    ax3b.plot(new_Xis, y_spl(new_Xis), 'g.', label=r"New $\xi$")
    for value in new_Xis:
        ax3b.plot([value, value], [0, y_spl(value)], 'g-')
    ax3b.legend(fontsize=legend_fontsize, loc="best")
    ax3b.set_xlabel(r'$\xi$', fontsize=label_fontsize+2)
#    ax3b.set_ylabel('Thermodynamic\nlength, $l$', fontsize=label_fontsize)
    ax3b.set_title(r'Optimized $\xi$-values', y=0.8, fontsize=label_fontsize)
    ax3a.set_ylim(bottom=0)
    ax3b.set_ylim(bottom=0)

    # Add a shared y-axis label for ax3a and ax3b
    fig.text(0.005, 0.275, 'Thermodynamic length, $l$', va='center', rotation='vertical', fontsize=12)

    for ax in [ax1, ax2, ax3a, ax3b, ax4]:
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(label_fontsize)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=label_fontsize-2)

    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=400)
    ###########################################################################
    ###########################################################################
    ###########################################################################



print("Exp parameters: ",exp_parameters)
print(f"Experimental errors: {diff}")
print(f"N data points perturbed: {np.count_nonzero(diff)}")
print(f"Experimental sigma: {sigma}")



exit()























