import sys
from matplotlib import pyplot as plt
sys.path.append("/Users/tuc41004/github/biceps/BICePs_2.0/")
import biceps

# User specified paths
traj = "/Volumes/RMR_4TB/new_sampling/d_1.01/results_ref_normal_1000000/traj_lambda1.00.npz"

if __name__ == "__main__":

    C = biceps.Convergence(trajfile=traj,
        maxtau=5000, nblock=5, nfold=10, nround=1000)
    C.get_autocorrelation_curves(method="block-avg", nblocks=1,
        plot_traces=True)
    C.process(nblock=5, nfold=10, nrounds=100, savefile=True,
        plot=True, block=False, normalize=True)









