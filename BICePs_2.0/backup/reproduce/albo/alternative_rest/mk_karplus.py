import sys, os
import numpy as np

a=['Allylic','Allylic','Allylic','Allylic','Allylic','Karplus_HH','Karplus_antiperiplanar_O','Karplus_antiperiplanar_O','Karplus_antiperiplanar_O','Karplus_antiperiplanar_O','Karplus_antiperiplanar_O','Karplus_antiperiplanar_O']
np.save('Karplus.npy',a)
b=[[45, 3, 4, 44],[21, 20, 19, 22],[22, 19, 13, 23],[28, 11, 10, 29],[28, 11, 10, 30],[37, 8, 7, 42],[42, 7, 38, 40],[42, 7, 38, 39],[42, 7, 38, 41],[37, 8, 33, 34],[37, 8, 33, 35],[37, 8, 33, 36]]
np.save('ind.npy',b)
