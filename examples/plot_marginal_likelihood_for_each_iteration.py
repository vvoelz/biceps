import numpy as np
from scipy.special import gammainc, gamma
import string
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import FwdModelOpt_routines as fmo
from scipy.signal import find_peaks

mpl_colors = matplotlib.colors.get_named_colors_mapping()
mpl_colors = list(mpl_colors.values())[::5]
extra_colors = mpl_colors.copy()
mpl_colors = ["k","grey","purple","brown","c","green",
              "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
        '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
            '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]

linestyles = ['-', '--', ':']
#colors = ["k","g","gold","brown","orange","c"]

import matplotlib
colors = matplotlib.colors
colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::5]
colors = mpl_colors #["b", "g", "r", "m", "c", "orange", "purple"] + list(colors)




# Good-Bad plot: {{{

def marginal_likelihood(x, d, phi, sigma, sem):
    return 1/(2*np.sqrt(2*np.pi)*phi*sigma) * (np.exp(-(x-d)**2/(2*phi**2 * sigma**2))\
                *(1 - np.heaviside(-phi*sigma+sem, 0.5)) +\
                np.exp(-(x-d)**2/(2* sigma**2))*phi*(1 - np.heaviside(sem-sigma, 0.5)))


def first_derivative(x, d, phi, sigma, sem, f_prime):
    """This and the other function are equiv"""

    f = x
    dev = (d-f)
    numerator = f_prime*dev*(phi**3 * (1 - np.heaviside(sem - sigma, 0.5)) * np.exp(dev**2 / (2*phi**2 * sigma**2)) + (1 - (np.heaviside(-phi*sigma + sem, 0.5))) * np.exp(dev**2 / (2*sigma**2)) )
    denominator = phi**2 * sigma**2*(phi * (np.heaviside(sem - sigma, 0.5) - 1) * np.exp(dev**2 / (2*phi**2 * sigma**2)) + (np.heaviside(-phi*sigma + sem, 0.5) - 1) * np.exp(dev**2 / (2*sigma**2)) )
    return numerator/denominator

def second_derivative(x, d, phi, sigma, sem, f_prime, f_2prime):
    f = x
    expr1 = -(d - f)*(phi**3*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))*f_2prime/(phi**2*sigma**2*(phi*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))) - (d - f)*(-phi*(d - f)*np.exp((d - f)**2/(2*phi**2*sigma**2))*f_prime/sigma**2 - (d - f)*np.exp((d - f)**2/(2*sigma**2))*f_prime/sigma**2)*f_prime/(phi**2*sigma**2*(phi*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))) - (d - f)*(phi**3*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))*((d - f)*np.exp((d - f)**2/(2*sigma**2))*f_prime/sigma**2 + (d - f)*np.exp((d - f)**2/(2*phi**2*sigma**2))*f_prime/(phi*sigma**2))*f_prime/(phi**2*sigma**2*(phi*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))**2) + (phi**3*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2)))*f_prime**2/(phi**2*sigma**2*(phi*np.exp((d - f)**2/(2*phi**2*sigma**2)) + np.exp((d - f)**2/(2*sigma**2))))

    expr2 = (-d + f)*f_2prime/(phi**2*sigma**2) + f_prime**2/(phi**2*sigma**2)

    expr3 = (-d + f)*f_2prime/sigma**2 + f_prime**2/sigma**2

    expr4 = 0

    # Conditions
    cond1 = (sem < sigma) & (phi*sigma - sem > 0)
    cond2 = (sem > sigma) & (phi*sigma - sem > 0)
    cond3 = (sem < sigma) & (phi*sigma - sem < 0)
    cond4 = (sem > sigma) & (phi*sigma - sem < 0)

    # Return based on conditions
    if cond1:
        return expr1
    elif cond2:
        return expr2
    elif cond3:
        return expr3
    elif cond4:
        return expr4
    else:
        print("Condition not met")
        exit()

#:}}}


# plot_marginal_likelihood:{{{
def plot_marginal_likelihood(likelihood_parameters, devs, xlim=(0,2)):
    label_fontsize = 14
    figsize = (10, 6.5) # for poster
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 1, hspace=0.3)
    ax = fig.add_subplot(gs[(0,0)])
    ax1 = fig.add_subplot(gs[(1,0)], sharex=ax)
    ax2 = fig.add_subplot(gs[(2,0)], sharex=ax)
    ax3 = fig.add_subplot(gs[(3,0)], sharex=ax)

    sigmaB = likelihood_parameters["sigmaB"]
    sem = likelihood_parameters["sem"]
    phi = likelihood_parameters["phi"]


    dx = 0.00005
    x = np.arange(0, 20, dx)
    sigma = np.sqrt(sem**2 + sigmaB**2)
    f = np.mean(devs)
    f_prime = 1.0
    f_2prime = 0

    #print(f"forward model: {f}")
    #print(devs)
    #print(f"average of all D: {np.mean(devs)}")


    i = -1
    #values = list(np.arange(1, 7.5, 0.5))
    values = [1.0, phi]
    for phi in values:
        i += 1

        likelihood = []
        energy = []
        denergy = []
        d2energy = []

        for d in devs:

            GB_likelihood = marginal_likelihood(x, d, phi, sigma, sem)
            result = first_derivative(x, d, phi, sigma, sem, f_prime)
            denergy.append(result)

            ##########################################
            # second Derivative
            result = second_derivative(x, d, phi, sigma, sem, f_prime, f_2prime)
            d2energy.append(result)
            #print(d2energy[-1])

            likelihood.append(GB_likelihood)
        likelihood = np.array(likelihood)
        GB = likelihood.T.prod(axis=1)
        GB = GB/(np.nansum(GB)*dx)
        energy = -np.log(GB)

        denergy = np.array(denergy)
        denergy = denergy.T.sum(axis=1)

        d2energy = np.array(d2energy)
        d2energy = d2energy.T.sum(axis=1)


        label=r"$\phi=%.1f$"%phi

        color = mpl_colors[i]
        ls = "--"
        linewidth=1
        if i == len(values)-1: color = "blue"
        if i == 0: color = "red"
        if (i == 0) or (i == len(values)-1):
            linewidth = 2
            ls='-'

        #if (i+1)%2==0: label = '_nolegend_'
        ax.plot(x, GB, ls=ls, color=color, linewidth=linewidth,
                 label=label)
        ax1.plot(x, energy, ls=ls, color=color, linewidth=linewidth,
                 label=label)
        ax2.plot(x, denergy, ls=ls, color=color, linewidth=linewidth,
                 label=label)
        ax3.plot(x, d2energy, ls=ls, color=color, linewidth=linewidth,
                 label=label)
        if i == 1:
            std_devs = []
            outliers = []
            # Get initial peaks without distance constraint
            initial_peaks, _ = find_peaks(GB)
            # Calculate automatic distance based on average distance between these peaks
            auto_distance = np.mean(np.diff(x[initial_peaks])) if len(initial_peaks) > 1 else len(x)
            # Find peaks again with the calculated distance
            peaks, _ = find_peaks(GB, distance=auto_distance)
            try:
                peak = x[peaks[0]]
                relevant_data_indices = np.where( np.abs(devs-peak) > (2*sigma) )[0]
                outliers = np.array(devs[relevant_data_indices])
                for _,d in enumerate(devs):
                    c = colors[_]
                    ax.scatter(d, 10+2, color='k', marker='o', facecolor=c, edgecolor="k")
                    #if any(element in _outliers for element in d):
                    if len(outliers) > 0:
                        if np.any(np.isin(d, outliers)):
                        #if d in _outliers:
                            arrow_props = dict(arrowstyle='-', linewidth=2, color="black")
                            _x,_y = d, 10+2
                            ax.annotate('', xy=(_x-0.05, _y), xytext=(_x+0.05, _y), arrowprops=arrow_props)
            except(Exception) as e: print(e)

    for axes in [ax, ax1, ax2]:
        axes.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=0, top=1)
        axes.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=0, top=1)
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=False, left=True, right=False)

    for axes in [ax3]:
        axes.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=0, top=1)
        axes.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=0, top=1)
        axes.xaxis.set_minor_locator(AutoMinorLocator())

    axs = [ax,ax1,ax2,ax3]
    for n, axes in enumerate(axs):
        axes.grid(alpha=0.5, linewidth=0.5)
        #axes.text(-0.155, 1.02, string.ascii_lowercase[n], transform=axes.transAxes,
        axes.text(-0.165, 1.02, string.ascii_lowercase[n], transform=axes.transAxes,
                size=20, weight='bold')
        # Setting the ticks and tick marks
        ticks = [axes.xaxis.get_minor_ticks(),
                 axes.xaxis.get_major_ticks()]
        marks = [axes.get_xticklabels(),
                axes.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(label_fontsize)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=label_fontsize-2)

    #xlim = (0, 1.3)
    #xlim = (0, 8)
    ax.set_xlim(xlim)

    dev_latex = r"$(d_{j} - f_{j}(X))^{2}$"
    sem_latex = r"$\sigma^{SEM}$"
    stat_model = "Good-Bad"
    ax.set_title(f"{stat_model} model", fontsize=16)
    #ax.set_ylim(-5, 7)
    ax.set_ylim(0, 17)
    ax.set_ylabel(r'$p(D| \mathbf{X},\sigma_{0}, \phi)$', fontsize=14, labelpad=20, rotation=60)

    shift = 0.2

    ax1ylim = (-5, 7)
    ax1.text(xlim[1]*0.95, ax1ylim[0]*0.99, r'$u = -log(p(D| \mathbf{X},\sigma_{0}, \phi))$', color='k', ha='right', va='bottom', fontsize=14)
    ylabel = ax1.set_ylabel(r'$u$', fontsize=18, labelpad=20, rotation=0)
    xpos, ypos = ylabel.get_position()
    ylabel.set_position((xpos, ypos - shift))  # move down by shift
    ax1.set_ylim(ax1ylim) #(-5, 7)
    ax1.legend(loc='center left', bbox_to_anchor=(1.025, -0.2), fontsize=label_fontsize)


    ax2.set_ylim(-60, 60)
    ylabel = ax2.set_ylabel(r'$\frac{\partial u}{\partial \theta} $', fontsize=22, labelpad=20, rotation=0)
    xpos, ypos = ylabel.get_position()
    ylabel.set_position((xpos, ypos - shift))  # move down by shift

    #ax3.set_ylim(np.nanmin(d2energy), np.nanmax(d2energy))
    ylabel = ax3.set_ylabel(r'$\frac{\partial^{2} u}{\partial \theta^{2}} $', fontsize=22, labelpad=20, rotation=0)
    xpos, ypos = ylabel.get_position()
    ylabel.set_position((xpos, ypos - shift))  # move down by shift


    handles, labels = ax1.get_legend_handles_labels()
    #order = list(range(len(handles)))
    #order = [order[0]]+order[2:]
    #order.append(1)

    ax3.set_xlabel('$|d - f(\mathbf{X}|\\theta)|$', fontsize=16)
    #plt.gcf().subplots_adjust(left=0.105, bottom=0.125, top=0.92, right=0.785, wspace=0.20, hspace=0.5)
    plt.gcf().subplots_adjust(left=0.125, bottom=0.125, top=0.92, right=0.8, wspace=0.20, hspace=0.5)
    #fig.savefig("marginal_likelihood_iter.png", dpi=600)
    return fig

# }}}





















