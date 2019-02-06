import sys, os
import numpy as np

allowed_gamma=np.load('allowed_gamma.npy')
exp=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.38,0.38,0.38,0.38,0.38,0.38]
model=[0.2859,0.3429,0.2351,0.3156,0.2768,0.2666,0.4390,0.2853,0.2567,0.4612,0.3321,0.3082,0.2487,0.2755,0.2619,0.2702,0.2703,0.2703]
model2=[0.2388,0.3951,0.2564,0.2787,0.3852,0.2809,0.2894,0.2234,0.2865,0.3096,0.3037,0.3423,0.2711,0.2588,0.2646,0.2643,0.2782,0.2514]
total_sse = np.array([[0.0 for gamma in allowed_gamma],[0.0 for gamma in allowed_gamma]])

for j in range(2):

    for g in range(len(allowed_gamma)):
        sse = 0.0

        for i in range(len(exp)):
            gamma = allowed_gamma[g]
            if j == 0:

                err = gamma*exp[i]*10. - model[i]*10.
                sse += (err**2.0)
            elif j == 1:
                err = gamma*exp[i]*10. - model2[i]*10.
                sse += (err**2.0)

        total_sse[j][g] = sse

np.save('sse.npy',total_sse)
