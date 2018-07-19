from __future__ import print_function
import numpy as np
from Sample import *

name = 'new_array.txt'
x = np.array([0.00,0.00,0.00],dtype=np.float64)
#for i in [100,150,200,250]:
print('Initial values = ',x)
print('Start..')
#sample(i, x)
sample(100, x)
print('Done!')
new = np.loadtxt(name)
print('Loaded %s'%name)
print(new.shape)
print(new)

