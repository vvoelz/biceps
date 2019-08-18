# Results of Cython-Wrapped C++ Convergence Methods

-------------------------------------------------------
#### Sat Aug 17 09:36:49 EDT 2019
-------------------------------------------------------

## First, let's check to make sure that the results are the same from both versions... We can print out the vector of the first nussaince parameter.

Python: 
```python
autocorr[0] = [ 1.    0.97121878  0.94352928 ..., -0.01157458 -0.0115672
 -0.01140086]

autocorr[1] = [ 1.    0.93664384  0.87762239 ...,  0.02668202  0.02662936
  0.02656123]

autocorr[2] = [ 1.    0.88556567  0.78811024 ..., -0.0083355  -0.00892486
   -0.00957131]

```

C++:

```python
autocorr[0] = [ 1.    0.97122443  0.94353521 ..., -0.01158515 -0.01158234
 -0.01141563]

autocorr[1] = [ 1.    0.93664414  0.87762195 ...,  0.02667784  0.02662288
  0.02655921]

autocorr[2] =  [ 1.    0.88556552  0.7881102  ..., -0.00833518 -0.00892458
   -0.00957092]

```

#### Note: The plots of the autocorrelation curve also match

---------------------------

## Next, the performance...

#### NOTE: Using data from Cineromycin B

### Trajectory: 10M steps, save every 100 steps

Python:

```bash
Loading trajectory file...
Collecting rest_type...
Collecting allowed_parameters...
Collecting sampled parameters...
Sat Aug 17 10:10:37 2019
Calculating autocorrelation ...
Done!
Sat Aug 17 10:12:34 2019
python 'old_convergence.py'  121.45s user 0.84s system 93% cpu 2:10.80 total
```

###### Time of Autocorrs Method: 1 min 57 sec = 117 sec


C++:
```bash
Sat Aug 17 10:10:47 2019
Calculating autocorrelation...
Sat Aug 17 10:11:14 2019
python 'convergence.py'  35.73s user 0.59s system 88% cpu 41.270 total
```

###### Time of Autocorrs Method: 27 sec


---------------------------------

<br>
<br>
<br>
<br>

# Other Tests:


### Trajectory: 100k steps, save every 1 steps

Python:

```bash
 traj_lambda0.00_100k_steps_every_1_steps.npz

 Loading trajectory file...
 Collecting rest_type...
 Collecting allowed_parameters...
 Collecting sampled parameters...
 Sat Aug 17 10:39:21 2019
 Calculating autocorrelation ...
 Done!
 Sat Aug 17 10:41:09 2019
 python 'old_convergence.py'  116.28s user 0.67s system 99% cpu 1:58.02 total

```


C++:

```bash
 traj_lambda0.00_100k_steps_every_1_steps.npz

 Loading trajectory file...
 Collecting rest_type...
 Collecting allowed_parameters...
 Collecting sampled parameters...
 Sat Aug 17 10:37:54 2019
 Calculating autocorrelation...
 Sat Aug 17 10:38:16 2019
 python 'convergence.py'  30.94s user 0.58s system 100% cpu 31.428 total
```


---------------------------------


### Trajectory: 1M steps, save every 1 steps

Python:
```bash
 traj_lambda0.00_1M_steps_every_1_steps.npz

 Loading trajectory file...
 Collecting rest_type...
 Collecting allowed_parameters...
 Collecting sampled parameters...
 Sat Aug 17 10:45:26 2019
 Calculating autocorrelation ...
 Done!
 Sat Aug 17 11:04:01 2019
 python 'old_convergence.py'  1199.39s user 4.66s system 98% cpu 20:17.54 total

```

###### Time of Autocorrs Method: 18 min 35 sec = 1115 sec


C++:

```bash
 traj_lambda0.00_1M_steps_every_1_steps.npz

 Loading trajectory file...
 Collecting rest_type...
 Collecting allowed_parameters...
 Collecting sampled parameters...
 Sat Aug 17 10:41:28 2019
 Calculating autocorrelation...
 Sat Aug 17 10:47:51 2019
 python 'convergence.py'  392.96s user 80.73s system 98% cpu 7:59.24 total

```

###### Time of Autocorrs Method: 6 min 23 sec = 383 sec




---------------------------------




<br>
<br>
<br>
<br>

# Final Note:

### Trajectory: 10k steps, save every 1 steps

C++:

```bash
 traj_lambda0.00_10k_steps_every_1_steps.npz

 Loading trajectory file...
 Collecting rest_type...
 Collecting allowed_parameters...
 Collecting sampled parameters...
 Sat Aug 17 10:39:19 2019
 Calculating autocorrelation...
 libc++abi.dylib: terminating with uncaught exception of type std::invalid_argument: Slice Error: must have a valid range
 python 'convergence.py'  1.91s user 0.23s system 102% cpu 2.084 total
```
## NOTE: FIXME: This Slice Error is an error message that I added in the source code. Highly likely the error was thrown is due to the trajectory having to few steps... We should add something in the source code to suggest this.





