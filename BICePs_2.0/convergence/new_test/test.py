import sys ,os
import numpy as np
from matplotlib import pyplot as plt

sampled_parameters = np.load('test.npy')
if (1):
        total_steps = len(sampled_parameters[0])
        x = np.arange(1,total_steps+0.1,1)
        plt.figure(figsize=(3*3,15))

        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(x,sampled_parameters[i])
            plt.xlabel('steps')
            plt.legend(loc='best')
        plt.tight_layout()
        print 'Done!'
        plt.show()


