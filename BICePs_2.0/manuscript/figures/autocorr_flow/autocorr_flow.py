import sys,os
sys.path.append("../../../")
import biceps

C = biceps.Convergence(trajfile="traj_lambda1.00.npz")
C.get_autocorrelation_curves(method="normal", maxtau=5000)
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)


