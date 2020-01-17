import sys,os
sys.path.append("../../../")
import biceps

C = biceps.Convergence(trajfile="./traj_lambda1.00.npz")
#C.plot_traces(fname="traces.pdf", xlim=(0, 1000000))
print(C.rest_type)
print(C.labels)
exit()
C.get_autocorrelation_curves(method="normal", maxtau=50000)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, 10000))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)


