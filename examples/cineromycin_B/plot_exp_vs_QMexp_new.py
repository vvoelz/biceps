import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np
import yaml


# Load in results from the exp-only sampling
t = yaml.load( file('traj_exp.yaml', 'r') )

# Load in results from the QM+exp sampling
t2 = yaml.load( file('traj_QMexp.yaml', 'r') )


# Plot the results
if (1):

        # Make a figure
        plt.figure()

        # plot histograms of the populations 
        #plt.subplot(3,2,1)

        #qm_pops = np.exp(-energies)
        #qm_pops = qm_pops/qm_pops.sum()
        #####plt.bar(np.arange(sampler.traj.nstates)-0.4, sampler.traj.sim_pops, width=0.4, color='r', align='edge')
        #plt.bar(np.arange(sampler.traj.nstates)-0.6, qm_pops, width=0.3, color='k', align='edge')
        #plt.bar(np.arange(sampler.traj.nstates)-0.3, sampler.traj.state_pops, width=0.3, color='b', align='edge')
        #plt.bar(np.arange(sampler2.traj.nstates), sampler2.traj.state_pops, width=0.3, color='r', align='edge')
        #plt.xticks(range(sampler.traj.nstates))
        #plt.xlim(0,sampler.traj.nstates)
        #plt.legend(['QM', 'exp', 'QM+exp'])
        #plt.xlabel('state')
        #plt.ylabel('population')

        # plot scatter plot of the f_i QM vs. f_i posterior with no modeling
        plt.subplot(2,2,1)

        # the data needs some massaging cause there might be nans or infs in there, which screws up plotting
        Ind = []  # state indices that are good
        for i in range(len(t["state_f"])):
            f, f2, f_std, f2_std = t["state_f"][i], t2["state_f"][i], t["state_f_std"][i], t2["state_f_std"][i]
            if (~np.isnan(f) and ~np.isnan(f2) and (f != np.inf) and (f2 != np.inf)):
                # if (~np.isnan(f_std) and ~np.isnan(f2_std) and (f_std != np.inf) and (f2_std != np.inf) ):
                Ind.append(i)

        f  = [t["state_f"][i] for i in Ind]
        f2 = [t2["state_f"][i] for i in Ind]
        f_std  = [t["state_f_std"][i] for i in Ind]
        f2_std = [t2["state_f_std"][i] for i in Ind]

        """### OLD WAY (with numpy arrays): ###
        Ind = ~np.isnan(sampler.traj.state_f)*(sampler.traj.state_f != np.inf)
        Ind *= ~np.isnan(sampler2.traj.state_f)*(sampler2.traj.state_f != np.inf)
        structure_Ind = np.arange(sampler.traj.nstates)[Ind]
        f = sampler.traj.state_f[Ind]
        f2 = sampler2.traj.state_f[Ind]
        f_std = sampler.traj.state_f_std[Ind]
        f2_std = sampler.traj.state_f_std[Ind]
        """
        plt.errorbar( f, f2, xerr=f_std, yerr=f2_std, fmt='k.')
        plt.hold(True)
        for i in np.arange(len(f)):
            if f2[i] < 4:
                plt.text( f[i], f2[i], str(Ind[i]) ) 
            elif f[i] < 3:
                plt.text( f[i], f2[i], str(Ind[i]) )
            else:
                pass 
        plt.plot([0, 6], [0, 6], color='k', linestyle='-', linewidth=2)
        plt.xlim(-2, 9)
        plt.ylim(-2, 12)
        plt.xlabel('$f_i$ exp (units kT)')
        plt.ylabel('$f_i$ QM+exp (units kT)')

        # plot histograms of sampled sigma 
        plt.subplot(2,2,2)
        plt.step(t['allowed_sigma_noe'], t['sampled_sigma_noe'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_sigma_noe'], t2['sampled_sigma_noe'], 'r-')
        plt.xlim(0,5)
        plt.legend(['exp', 'QM+exp'])
        plt.xlabel("$\sigma_d$")
        plt.ylabel("$P(\sigma_d)$")
        plt.yticks([])

        # plot histograms of sampled sigma_J
        plt.subplot(2,2,3)
        plt.step(t['allowed_sigma_J'], t['sampled_sigma_J'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_sigma_J'], t2['sampled_sigma_J'], 'r-')
        plt.legend(['exp', 'QM+exp'])
        plt.xlabel("$\sigma_J$")
        plt.ylabel("$P(\sigma_J)$")
        plt.yticks([])

        # plot histograms of sampled gamma 
        plt.subplot(2,2,4)
        plt.step(t['allowed_gamma'], t['sampled_gamma'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_gamma'], t2['sampled_gamma'], 'r-')
        plt.legend(['exp', 'QM+exp'])
        plt.xlim(0.8, 2.1)
        plt.xlabel("$\gamma^{-1/6}$")
        plt.ylabel("$P(\gamma^{-1/6})$")
        plt.yticks([])


        plt.show()




