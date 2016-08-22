import numpy as np
for i in range(250):
        a=np.load("state%d.npy"%i)
        print len(a)

