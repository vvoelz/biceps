import sys, os
import numpy as np
os.system('mkdir distance')
for i in range(15037):
    print i
    a=np.loadtxt("dis_state%d.txt"%i)*10.
    np.savetxt('distance/dis%d.txt'%i,a)

