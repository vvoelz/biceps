import sys,os
sys.path.append("/Users/tuc41004/github/biceps/BICePs_2.0/")
import biceps

if __name__ == "__main__":

    # NOTE: Code to plot autocorrelation curves for figure 5.
    # User specified paths
    traj = "/Volumes/WD_Passport_1TB/new_sampling/new_sampling/d_1.03/results_ref_normal_1000000/traj_lambda1.00.npz"
    #traj = "/Volumes/WD_Passport_1TB/new_sampling/new_sampling/d_1.03/results_ref_normal_100000/traj_lambda1.00.npz"
    #C = biceps.Convergence(trajfile=traj)
    C = biceps.Convergence(trajfile=traj)
    C.get_autocorrelation_curves(method="normal", maxtau=5000)
    C.process(nblock=5, nfold=10, nround=100, savefile=True,
        plot=True, block=True, normalize=True)
    #C.get_autocorrelation_curves(method="exp", maxtau=5000)



