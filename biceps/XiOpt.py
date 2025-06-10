
import scipy, gc, os, copy, time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# estimate_sigmas:{{{
def estimate_sigmas(u_kln, plot_data=True, verbose=0):
    """Using as input the Delta_U_ij energies from u_kln,
    estimate the standard deviations P(U_{i-->i+1}) for neighboring ensembles.

    RETURNS
    sigmas   - a np.array() of standard deviations P(U_{i-->i+1}).
    """

    nXis = u_kln.shape[1]
    thermo_states = np.array(list(range(nXis)))
    if verbose: print('nXis', nXis)

    if plot_data: plt.figure(figsize=(6, 80))

    Delta_uij_values = []
    sigmas = []

    for j in range(nXis-1):

        ## transitions from state 0 to 1 or 1 to 2, or 2 to 3 ....
        Ind = (thermo_states == j)
        delta_u_ij = u_kln[Ind, j+1]       # forward delta_u only for neighbored ensembles

        Ind2 = (thermo_states == (j+1))
        delta_u_ji = u_kln[Ind2, j]       # forward delta_u only for neighbored ensembles

        #print ('xi index=', j)
        #print ('delta_u_ij.shape=', delta_u_ij.shape)

        #Delta_uij_values.append(delta_u_ij)

        ### VAV debug
        if verbose: print('Are any delta_u_ij values nan?')
        if verbose: print(delta_u_ij)
        if verbose: print('Are any delta_u_ji values nan?')
        if verbose: print(delta_u_ji)

        mu_ij, sigma_ij = scipy.stats.norm.fit(delta_u_ij)
        mu_ji, sigma_ji = scipy.stats.norm.fit(delta_u_ji)

        sigma = ( sigma_ij + sigma_ji ) / 2.0
        sigmas.append(sigma)

        delta_u_bins = np.arange(-15., 15., 0.02)
        counts, bin_edges = np.histogram(delta_u_ij, bins=delta_u_bins)
        counts = counts/counts.sum() # normalize
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0

        if plot_data:
            plt.subplot(nXis-1, 1, j+1)
            plt.step(bin_centers, counts, label='$\Delta u_{%d \\rightarrow %d} \sigma$=%.2f'%(j,j+1,sigma))
            #plt.xlabel('$\Delta u_{%d \\rightarrow %d}$'%(j, j+1))
            plt.legend(loc='best')

    if plot_data:
        plt.tight_layout()
        plt.savefig("sigmas.png")

    ## VAV: hot fix for non-sampled xi (sigma = nan), or sampled only once (sigma = 0)
    max_sigma = max(sigmas)
    for i in range(len(sigmas)):
        if (sigmas[i] == 0) or np.isnan(sigmas[i]):
            sigmas[i] = max_sigma

    return np.array(sigmas)
# }}}

# optimize_xi_values:{{{
def optimize_xi_values(xi_values, u_kln, outdir=None, nsteps=100000, tol=1e-7,
    alpha=1e-5, print_every=1000, make_plots=True, optimize_nXis=False, verbose=True):
    """Optimize the xi values for all intermediates to minimize the total variance
    in P(\Delta u_ij) for neighboring thermodynamic ensembles. To do so, we run
    a steepest descent algorithm. We run for some fixed number of steps, or
    until some tolerance is reached and stop if the Xis dont change within this tolerance.

    Args:
        xi_values(list,np.ndarray): xi values before optimization
        u_kln(np.ndarray): u_kln[k,l,n] - the reduced potential energy of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
        nsteps(int): number of gradient descent steps
        outdir(str): out directory for plots
        alpha(float): gradient descent step size
        tol(float): gradient descent tolerance
        make_plots(bool): if True, `outdir` must not be None

    Return:
        optimized xi values
    """
    reverse_xi = False
    ### Check if the xi_values are in the correct order
    if not np.all(np.diff(xi_values) > 0):
        reverse_xi = True
        if verbose:
            print("xi_values are not in ascending order. Reversing xi_values and u_kln.")
        xi_values = xi_values[::-1]
        u_kln = u_kln[::-1]

    ### Lambda optimization
    dx = estimate_sigmas(u_kln, plot_data=0, verbose=verbose) # equal to sigmas
    x_values = np.cumsum(dx)    # convert to a list of x values of separated harmonic potentials
#    if xi_values[0] != 0.0:
#        x_values = np.array(np.concatenate([[0], x_values]))    # add a zero corresponding to xi0 = 0.0
    x_values = np.array(np.concatenate([[0], x_values]))    # add a zero corresponding to xi0 = 0.0

    print(xi_values)
    print(x_values)

    if verbose: print('x_values', x_values)

    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import interp1d

    x_observed = xi_values      #not inclduing the first one, xi_0
    y_observed = x_values

    if make_plots:
        save_plots = True
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(x_observed, y_observed, 'ro', label = 'data')
        #plt.semilogy(x_observed, y_observed, 'ro', label = 'data')
    #y_spl = CubicSpline(x_observed, y_observed)#, s=0,k=4)
    y_spl = UnivariateSpline(x_observed, y_observed, s=0, k=3)
    x_range = np.linspace(x_observed[0], x_observed[-1], 1000)

    if make_plots:
        plt.plot(x_range, y_spl(x_range), label="spline")   # for UnivariateSpline
        ## plt.plot(x_observed, y_spl(x_observed), label="spline") # for CubicSpline
        plt.legend()
        plt.xlabel('xi')
        plt.ylabel('x values')
        plt.subplot(1,2, 2)   #derivative plot
    y_spl_1d = y_spl.derivative(n=1)    #n=1 , means the first order derivative
    #print (y_spl_1d(x_observed))
    # y_spl_1d = y_spl(x_observed, 1)  # first derivative of Cubic spline

    if make_plots:
        plt.plot(x_range, y_spl_1d(x_range), '-')
        plt.plot(x_observed, y_spl_1d(x_observed), '.')
        plt.ylabel('dx/dxi')
        #plt.plot(x_observed, y_spl_1d, '.-', label='derivative')
        plt.legend()
        plt.xlabel('xi')

        if save_plots:
            spline_pngfile = os.path.join(outdir, f'splinefit.png')
            plt.savefig(spline_pngfile)
            if verbose: print(f'Wrote: {spline_pngfile}')


    max_del_xi = 0.0001   # the minimization step limited to this as a maximum change
    nXis = len(xi_values)
    if verbose: print('xi_values', xi_values)
    old_Xis = np.array(xi_values)
    traj_Xis = np.zeros( (nXis,nsteps) )
    for step in range(nsteps):

        # store the trajectory of Xis
        traj_Xis[:,step] = old_Xis
        if verbose: print('step', step, old_Xis)

        # perform a steepest descent step
        new_Xis = np.zeros( old_Xis.shape )
        del_Xis = np.zeros( old_Xis.shape )
        del_Xis[0] = 0.0   # fix the \xi = 0 endpoint
        del_Xis[nXis-1] = 0.0  # fix the \xi = 1 endpoint

        if False:  # do in a loop (SLOW!)
            for i in range(1, (nXis-1)):
                del_Xis[i] = -1.0*alpha*2.0*y_spl_1d(old_Xis[i])*( 2.0*y_spl(old_Xis[i]) - y_spl(old_Xis[i-1]) - y_spl(old_Xis[i+1]))
        else:   # do as a vector operation (FAST!)
            y_all = y_spl(old_Xis)
            yh, yi, yj = y_all[0:nXis-2], y_all[1:nXis-1], y_all[2:nXis]
            del_Xis[1:nXis-1] = -1.0*alpha*2.0*y_spl_1d(old_Xis[1:nXis-1])*( 2.0*yi - yh - yj)
        if abs(np.max(del_Xis)) > max_del_xi:
            del_Xis[1:nXis-1] = del_Xis[1:nXis-1]*max_del_xi/np.max(del_Xis)
        new_Xis = old_Xis + del_Xis

        # record the average change in the Xis
        del_Xis = np.abs(old_Xis - new_Xis).mean()
        if step % print_every == 0:
            if verbose: print('step', step, 'del_Xis', del_Xis)
        if del_Xis < tol:
            if verbose: print('Tolerance has been reached: del_Xis =', del_Xis, '< tol =', tol)
            break

        old_Xis = new_Xis

    if make_plots:

        # Plot the results
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        for i in range(nXis):
            plt.plot(range(step), traj_Xis[i,0:step], '-')
        plt.xlabel('step')
        plt.ylabel('xi values')

        plt.subplot(1,2,2)
        for i in range(nXis):
            plt.plot(range(step), y_spl(traj_Xis[i,0:step]), '-')
        plt.xlabel('step')
        plt.ylabel('x values')

        if save_plots:
            traces_pngfile = os.path.join(outdir, f'xi_optimization_traces.png')
            plt.savefig(traces_pngfile)
            if verbose: print(f'Wrote: {traces_pngfile}')

    results = {"x_range": x_range, "y_spl": y_spl,
               "old_xis": xi_values, "new_Xis": new_Xis
               }

    if make_plots:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(2,1,1)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(xi_values, y_spl(np.array(xi_values)), 'r.', label=r"Old $\xi$")
        for value in xi_values:
            plt.plot([value, value], [0, y_spl(value)], 'r-')
        plt.legend()
        plt.xlabel(r'$\xi$', fontsize=18)
        plt.ylabel('Thermodynamic\nlength, $l$', fontsize=16)
        plt.title(r'Initial $\xi$-values', fontsize=16)

        plt.subplot(2,1,2)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(new_Xis, y_spl(new_Xis), 'g.', label=r"New $\xi$")
        for value in new_Xis:
            plt.plot([value, value], [0, y_spl(value)], 'g-')
        plt.legend()
        plt.xlabel(r'$\xi$', fontsize=18)
        plt.ylabel('Thermodynamic\nlength, $l$', fontsize=16)
        plt.title(r'Optimized $\xi$-values', fontsize=16)

        if save_plots:
            old_vs_new_Xis_pngfile = os.path.join(outdir, f'old_vs_new_Xis.png')
            fig.tight_layout()
            fig.savefig(old_vs_new_Xis_pngfile)
            if verbose: print(f'Wrote: {old_vs_new_Xis_pngfile}')
    if np.any(new_Xis < 0.0): raise ValueError("Xi-values should all be positive. Optimization give at least 1 negative xi value.")

    if reverse_xi: new_Xis = new_Xis[::-1]
    return new_Xis, results
# }}}


if __name__ == "__main__":
    outdir = "/Users/rr/github/FwdModelOpt/"
    xi_values = np.concatenate([[0.0],np.load("/Users/rr/github/FwdModelOpt/xi_values.npy")[::-1]])
#    print(xi_values)
    print(len(xi_values))
    u_kln = np.load("/Users/rr/github/FwdModelOpt/u_kln.npy")
    print(len(u_kln))
#    exit()
    new_xi_values = optimize_xi_values(xi_values, u_kln, outdir, nsteps=100000, tol=1e-7, alpha=1e-5, print_every=1000, make_plots=0, verbose=0)
    print(xi_values)
    print(new_xi_values)
    exit()















