import sys, os
sys.path.append('biceps')
from Preparation import *
from toolbox import *

os.system('mkdir PF_CS')
ind = np.loadtxt('cs_indices_H.txt')

path = 'H/*txt'

states = 25

exp_data = 'new_exp_H.txt'

top = 'state0.pdb'

out_dir = 'PF_CS'

p = Preparation('cs_H',states=states,indices=ind,exp_data=exp_data,top=top,data_dir=path)

p.write(out_dir=out_dir)


ind = np.loadtxt('cs_indices_Ca.txt')

path = 'Ca/*txt'

states = 25

exp_data = 'new_exp_Ca.txt'

top = 'state0.pdb'

out_dir = 'PF_CS'

p = Preparation('cs_Ca',states=states,indices=ind,exp_data=exp_data,top=top,data_dir=path)

p.write(out_dir=out_dir)


ind = np.loadtxt('cs_indices_N.txt')

path = 'N/*txt'

states = 25

exp_data = 'new_exp_N.txt'

top = 'state0.pdb'

out_dir = 'PF_CS'

p = Preparation('cs_N',states=states,indices=ind,exp_data=exp_data,top=top,data_dir=path)

p.write(out_dir=out_dir)






