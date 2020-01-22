import sys,os
sys.path.append("../../../")
import biceps


if __name__ == "__main__":

    testing = False
    if testing:
        trajfile = "/Volumes/WD_Passport_1TB/new_sampling/new_sampling/d_1.01/results_ref_normal_100000/traj_lambda1.00.npz"
    else:
        trajfile = "traj_lambda1.00.npz"
    C = biceps.Convergence(trajfile)
    C.get_autocorrelation_curves(method="normal", maxtau=50000)
#    C.get_autocorrelation_curves(method="normal", maxtau=1000)
    C.process(nblock=5, nfold=10, nround=100, savefile=True,
        plot=True, block=True, normalize=True)




