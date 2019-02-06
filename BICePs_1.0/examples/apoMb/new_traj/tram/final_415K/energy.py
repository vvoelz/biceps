import numpy as np
for i in range(6):
	a=np.loadtxt("pop_model_%d.txt"%i)
	b=[]
	for j in a:
		f=-np.log(j) # units in kT
		b.append(f)
	np.savetxt('energy_model_%d.txt'%i,b)
