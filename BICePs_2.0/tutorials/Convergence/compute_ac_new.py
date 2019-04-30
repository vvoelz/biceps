import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



traj = np.load('traj_lambda1.00.npz')['arr_0'].item()['trajectory']
allowed_sigma_J = np.load('traj_lambda1.00.npz')['arr_0'].item()['allowed_sigma'][0]
allowed_sigma_noe = np.load('traj_lambda1.00.npz')['arr_0'].item()['allowed_sigma'][1]
allowed_gamma = np.load('traj_lambda1.00.npz')['arr_0'].item()['allowed_gamma']
sampled_sigma_J = []
sampled_sigma_noe = []
sampled_gamma = []
# append sampled nuisance paramters, this part is hard coded now, will be fixed in the future
for i in range(len(traj)):
    sampled_sigma_J.append(allowed_sigma_J[traj[i][4][0][0]])
    sampled_sigma_noe.append(allowed_sigma_noe[traj[i][4][1][0]])
    sampled_gamma.append(allowed_gamma[traj[i][4][1][1]])

time = np.arange(1,len(sampled_sigma_J)+0.1,1)
if (1):
	plt.figure(figsize=(10,15))
	plt.subplot(3,1,1)
	plt.plot(time,sampled_sigma_J,label='sigma_J',color='blue')
	plt.xlabel('steps')
	plt.ylabel('sigma_J')
	plt.legend(loc='best')
	plt.subplot(3,1,2)
	plt.plot(time,sampled_sigma_noe,label='sigma_noe',color='green')
	plt.xlabel('steps')
	plt.ylabel('sigma_noe')
	plt.legend(loc='best')
	plt.subplot(3,1,3)
	plt.plot(time,sampled_gamma,label='gamma',color='black')
	plt.xlabel('steps')
	plt.ylabel('gamma')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('sampled_parameters.png')
	#plt.show()

sampled_sigma_J = np.array(sampled_sigma_J)
sampled_sigma_noe = np.array(sampled_sigma_noe)
sampled_gamma = np.array(sampled_gamma)


def g(f, max_tau=10000, normalize=True):
    """Calculate the autocorrelaton function for a time-series f(t).
    INPUT
    f         - a 1D numpy array containing the time series f(t)
    
    PARAMETERS
    max_tau   - the maximum autocorrelation time to consider.
    normalize - if True, return g(tau)/g[0]
    
    RETURNS
    result    - a numpy array of size (max_tau+1,) containing g(tau).
    """
    
    f_zeroed = f-f.mean()
    T = f_zeroed.shape[0]
    result = np.zeros(max_tau+1)
    for tau in range(max_tau+1):
        result[tau] = np.dot(f_zeroed[0:-1-tau],f_zeroed[tau:-1])/(T-tau)
    
    if normalize:
        return result/result[0]
    else:
        return result

trace_labels = ['$\\sigma_J$','$\\sigma_{noe}$','$\\gamma$']
max_tau=10000
autocorrs = []
for timeseries in [sampled_sigma_J,sampled_sigma_noe,sampled_gamma]:
    autocorrs.append( g(timeseries, max_tau=max_tau) )

plt.figure( figsize=(10,10))
for i in range(len(autocorrs)):
    plt.subplot(len(autocorrs),2,i+1)
    plt.plot(np.arange(max_tau+1),autocorrs[i])
    plt.xlabel('$\\tau$')
    plt.ylabel('$g(\\tau)$ for %s'%trace_labels[i])
plt.tight_layout()
plt.savefig('auto_corr.png')



from scipy.optimize import curve_fit

def single_exp_decay(x, a0, a1, tau1):
    return a0 + a1*np.exp(-(x/tau1)) 

def double_exp_decay(x, a0, a1, a2, tau1, tau2):
    return a0 + a1*np.exp(-(x/tau1)) + a2*np.exp(-(x/tau2))

def exponential_fit(ac, use_function='single'):
    """Perform a single- or double- exponential fit on an autocorrelation curve.
    
    RETURNS
    yFit  - the y-values of the fit curve."""
    
    nsteps = ac.shape[0]
    if use_function == 'single':
        v0 = [0.0, 1.0 , 4000.]  # Initial guess [a0, a1, tau1] for a0 + a1*exp(-(x/tau1))
        popt, pcov = curve_fit(single_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = single_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2])
        # print 'best-fit a0 = ', popt[0], '+/-', pcov[0][0]
        # print 'best-fit a1 = ', popt[1], '+/-', pcov[1][1]
        print 'best-fit tau1 = ', popt[2], '+/-', pcov[2][2]
    else:
        v0 = [0.0, 0.9, 0.1, 4000., 200.0]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
        popt, pcov = curve_fit(double_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = double_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2], popt[3], popt[4])
        # print 'best-fit a0 = ', popt[0], '+/-', pcov[0][0]
        #print 'best-fit a1 = ', popt[1], '+/-', pcov[1][1]
        #print 'best-fit a2 = ', popt[2], '+/-', pcov[2][2]
        print 'best-fit tau1 = ', popt[3], '+/-', pcov[3][3]
        print 'best-fit tau2 = ', popt[4], '+/-', pcov[4][4]

    return yFit_data


plt.figure( figsize=(10,10))
for i in range(len(autocorrs)):
    yFit = exponential_fit(autocorrs[i])
    plt.subplot(len(autocorrs),2,i+1)
    plt.plot(np.arange(max_tau+1), autocorrs[i])
    plt.plot(np.arange(max_tau+1), yFit, 'r--')
    plt.xlabel('$\\tau$')
    plt.ylabel('$g(\\tau)$ for %s'%trace_labels[i])
plt.tight_layout()
plt.savefig('auto_corr_fit.png')




sys.exit()
# compute auto-correlation
def autocorr(x):
    tau = x.size
    g = np.correlate(x, x, mode='full')[tau:]
    n = np.arange(tau-1,0,-1)
    return g/n

#ac_sigma_J =  autocorr( np.array(sampled_sigma_J) )
#ac_sigma_noe =  autocorr( np.array(sampled_sigma_noe) )
#ac_gamma =  autocorr( np.array(sampled_gamma) )

var_sigma_J = np.var(np.array(sampled_sigma_J))
var_sigma_noe = np.var(np.array(sampled_sigma_noe))
var_gamma = np.var(np.array(sampled_gamma))


#time_in_steps = np.arange(1,len(ac_sigma_J)+1,1)

def autocorr_valid(x):
    tau = 1000
    y = x[:np.size(x)-tau]
    g = np.correlate(x, y, mode='valid')
    n = np.array([np.size(x)-tau]*len(g))
    return g/n

ac_sigma_J =  autocorr_valid( np.array(sampled_sigma_J) )
ac_sigma_noe =  autocorr_valid( np.array(sampled_sigma_noe) )
ac_gamma =  autocorr_valid( np.array(sampled_gamma) )

time_in_steps = np.arange(1,len(ac_sigma_J)+1,1)

plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
plt.plot(time_in_steps, ac_sigma_J, label='sigma_J', color='blue')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of sigma_J')
plt.subplot(3,1,2)
plt.plot(time_in_steps, ac_sigma_noe,label='sigma_noe', color='green')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of sigma_noe')
plt.subplot(3,1,3)
plt.plot(time_in_steps, ac_gamma, label='gamma',color='black')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of gamma')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('autocorrelation_valid.pdf')

sys.exit()

from scipy.optimize import curve_fit

def single_exp_decay(x, a0,a1, tau1):
    return a0+a1*np.exp(-(x/tau1))

v0_sigma_J = [0.1, 0.5,40.]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
popt_sigma_J, pcov_sigma_J = curve_fit(single_exp_decay, time_in_steps, ac_sigma_J, p0=v0_sigma_J, maxfev=10000)  # ignore last bin, which has 0 counts
yFit_data_sigma_J = single_exp_decay(time_in_steps, popt_sigma_J[0], popt_sigma_J[1],popt_sigma_J[2])


v0_sigma_noe = [0.1,0.5, 40.]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
popt_sigma_noe, pcov_sigma_noe = curve_fit(single_exp_decay, time_in_steps, ac_sigma_noe, p0=v0_sigma_noe, maxfev=10000)  # ignore last bin, which has 0 counts
yFit_data_sigma_noe = single_exp_decay(time_in_steps, popt_sigma_noe[0], popt_sigma_noe[1],popt_sigma_noe[2])

v0_gamma = [0.1,0.5, 40.]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
popt_gamma, pcov_gamma = curve_fit(single_exp_decay, time_in_steps, ac_gamma, p0=v0_gamma, maxfev=10000)  # ignore last bin, which has 0 counts
yFit_data_gamma = single_exp_decay(time_in_steps, popt_gamma[0], popt_gamma[1],popt_gamma[2])

plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
plt.plot(time_in_steps, ac_sigma_J, label='sigma_J', color='blue')
plt.plot(time_in_steps,yFit_data_sigma_J,'r--')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of sigma_J')
plt.subplot(3,1,2)
plt.plot(time_in_steps, ac_sigma_noe,label='sigma_noe', color='green')
plt.plot(time_in_steps,yFit_data_sigma_noe,'r--')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of sigma_noe')
plt.subplot(3,1,3)
plt.plot(time_in_steps, ac_gamma, label='gamma',color='black')
plt.plot(time_in_steps,yFit_data_gamma,'r--')
plt.xlabel(r'$\tau$ (steps)')
plt.ylabel('autocorrel of gamma')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('autocorrelation_curve_fitting.pdf')
#plt.show()





