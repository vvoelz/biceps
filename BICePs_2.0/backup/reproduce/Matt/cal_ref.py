import sys, os
import numpy as np

model=[0.2859,0.3429,0.2351,0.3156,0.2768,0.2666,0.4390,0.2853,0.2567,0.4612,0.3321,0.3082,0.2487,0.2755,0.2619,0.2702,0.2703,0.2703]
model2=[0.2388,0.3951,0.2564,0.2787,0.3852,0.2809,0.2894,0.2234,0.2865,0.3096,0.3037,0.3423,0.2711,0.2588,0.2646,0.2643,0.2782,0.2514]
beta=[]
for i in range(len(model)):
    b=(model[i]*10.+model2[i]*10.)/3.
    beta.append(b)
print beta

ref=[]
ref1=[]
for i in range(len(model)):
    a=np.log(beta[i])+model[i]*10./beta[i]
    ref.append(a)
    a1=np.log(beta[i])+model2[i]*10./beta[i]
    ref1.append(a1)
print sum(ref)
print sum(ref1)

