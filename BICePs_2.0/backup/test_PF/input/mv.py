import sys, os
import numpy as np

a=np.arange(5.0,8.5,0.5)
b=np.arange(2.0,2.8,0.1)

for i in a:
	print i
	os.system('cp apo_mb_ph7_input/Nc_all/x%.1f/*npy Nc'%i)
for j in b:
	print j
	os.system('cp apo_mb_ph7_input/Nh_all/xh%.1f/*npy Nh'%j)

