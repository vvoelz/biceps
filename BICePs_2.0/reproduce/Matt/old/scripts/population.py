import numpy as np
for i in range(10000):
	a=np.load("state/state%d.npy"%i)
	if len(a) != 0:
#		print 1
#	else:
		print len(a)
