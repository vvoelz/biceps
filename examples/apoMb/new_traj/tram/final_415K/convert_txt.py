import numpy as np
a=np.load('tram_stationary_distribution.npy')
for i in range(len(a)):
	np.savetxt('pop_model_%d.txt'%i, a[i])
