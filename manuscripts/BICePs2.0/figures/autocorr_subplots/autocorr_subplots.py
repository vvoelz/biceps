import sys
#sys.path.append("/Users/tuc41004/github/biceps/BICePs_2.0/")
import biceps

file = "traj_lambda1.00.npz"
C = biceps.Convergence(trajfile=file)
C.plot_traces(fname="traces.pdf", xlim=(0, 1000000))
#print(C.rest_type)
#print(C.labels)

C.get_autocorrelation_curves(method="exp", maxtau=5000)
C.plot_auto_curve_with_exp_fitting(fname="autocorrelation_curve_with_exp_fitting.png")

C.get_autocorrelation_curves(method="normal", maxtau=5000)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, 10000))

C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)


