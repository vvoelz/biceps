# Vince's notes on Installation


### Create a conda environment

Create a conda environment for biceps and install the required packages:

```
conda create --name biceps
conda activate biceps             
conda install --file requirements.txt
```

NOTE: I was able to install all the packages if I commented out the `autoapi` package

### Compile the C++ code parts of the package

```
% cd biceps

% cat rebuild.sh 

python setup.py clean --all
python setup.py build_ext --inplace

% bash rebuild.sh
```

There were lots of warnings, but it appears to have worked:
```
(biceps) vv@MacBookPro biceps % bash rebuild.sh 
'build/lib.macosx-10.13-x86_64-cpython-312' does not exist -- can't clean it
'build/bdist.macosx-10.13-x86_64' does not exist -- can't clean it
'build/scripts-3.12' does not exist -- can't clean it
Compiling PosteriorSampler.pyx because it changed.
[1/1] Cythonizing PosteriorSampler.pyx
In file included from PosteriorSampler.cpp:1203:
./cppPosteriorSampler.h:357:31: warning: field 'sampler' will be initialized after field 'nreplicas' [-Wreorder-ctor]
  357 |         ensembles(ensembles), sampler(sampler), nreplicas(nreplicas), change_Nr_every(change_Nr_every), write_every(write_every),
      |         ~~~~~~~~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~
      |         nreplicas(nreplicas)  change_Nr_every(change_Nr_every) write_every(write_every) move_ftilde_every(move_ftilde_every) dftilde(dftilde)
  358 |         move_ftilde_every(move_ftilde_every), dftilde(dftilde), ftilde_sigma(ftilde_sigma), scale_and_offset(scale_and_offset)
      |         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ftilde_sigma(ftilde_sigma)            scale_and_offset(scale_and_offset) ensembles(ensembles) sampler(sampler)
PosteriorSampler.cpp:43316:17: warning: fallthrough annotation in unreachable code [-Wunreachable-code-fallthrough]
 43316 |                 CYTHON_FALLTHROUGH;
       |                 ^
PosteriorSampler.cpp:524:36: note: expanded from macro 'CYTHON_FALLTHROUGH'
  524 |         #define CYTHON_FALLTHROUGH [[fallthrough]]
      |                                    ^
PosteriorSampler.cpp:43327:17: warning: fallthrough annotation in unreachable code [-Wunreachable-code-fallthrough]
 43327 |                 CYTHON_FALLTHROUGH;
       |                 ^
PosteriorSampler.cpp:524:36: note: expanded from macro 'CYTHON_FALLTHROUGH'
  524 |         #define CYTHON_FALLTHROUGH [[fallthrough]]
      |                                    ^
PosteriorSampler.cpp:44145:17: warning: fallthrough annotation in unreachable code [-Wunreachable-code-fallthrough]
 44145 |                 CYTHON_FALLTHROUGH;
       |                 ^
PosteriorSampler.cpp:524:36: note: expanded from macro 'CYTHON_FALLTHROUGH'
  524 |         #define CYTHON_FALLTHROUGH [[fallthrough]]
      |                                    ^
PosteriorSampler.cpp:44156:17: warning: fallthrough annotation in unreachable code [-Wunreachable-code-fallthrough]
 44156 |                 CYTHON_FALLTHROUGH;
       |                 ^
PosteriorSampler.cpp:524:36: note: expanded from macro 'CYTHON_FALLTHROUGH'
  524 |         #define CYTHON_FALLTHROUGH [[fallthrough]]
      |                                    ^
5 warnings generated.
cppPosteriorSampler.cpp:175:20: warning: passing an object of reference type to 'va_start' has undefined behavior [-Wvarargs]
  175 |     va_start(args, format);
      |                    ^
cppPosteriorSampler.cpp:172:45: note: parameter of type 'const std::string &' (aka 'const basic_string<char> &') is declared here
  172 | std::string stringFormat(const std::string& format, ...)
      |                                             ^
cppPosteriorSampler.cpp:182:24: warning: passing an object of reference type to 'va_start' has undefined behavior [-Wvarargs]
  182 |         va_start(args, format);
      |                        ^
cppPosteriorSampler.cpp:172:45: note: parameter of type 'const std::string &' (aka 'const basic_string<char> &') is declared here
  172 | std::string stringFormat(const std::string& format, ...)
      |                                             ^
cppPosteriorSampler.cpp:302:39: warning: comparison of integers of different signs: 'int' and 'size_type' (aka 'unsigned long') [-Wsign-compare]
  302 |         if (i < subvector.size() && p < subvector[i].size()) {
      |                                     ~ ^ ~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:302:15: warning: comparison of integers of different signs: 'int' and 'size_type' (aka 'unsigned long') [-Wsign-compare]
  302 |         if (i < subvector.size() && p < subvector[i].size()) {
      |             ~ ^ ~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:3820:9: warning: variable 'fm_Np' set but not used [-Wunused-but-set-variable]
 3820 |     int fm_Np = 0;
      |         ^
cppPosteriorSampler.cpp:3821:15: warning: unused variable 'nstates' [-Wunused-variable]
 3821 |     const int nstates = state_energies.size();
      |               ^~~~~~~
cppPosteriorSampler.cpp:4006:18: warning: unused variable 'lambda' [-Wunused-variable]
 4006 |     const double lambda = expanded_values[0];
      |                  ^~~~~~
cppPosteriorSampler.cpp:4177:10: warning: unused variable 'use_pmo' [-Wunused-variable]
 4177 |     bool use_pmo = false;
      |          ^~~~~~~
cppPosteriorSampler.cpp:4321:31: warning: field 'sampler' will be initialized after field 'nreplicas' [-Wreorder-ctor]
 4321 |         ensembles(ensembles), sampler(sampler), nreplicas(nreplicas),
      |         ~~~~~~~~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~
      |         nreplicas(nreplicas)  change_Nr_every(change_Nr_every) write_every(write_every)
 4322 |         change_Nr_every(change_Nr_every), write_every(write_every),
      |         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~
      |         move_ftilde_every(move_ftilde_every) dftilde(dftilde)
 4323 |         move_ftilde_every(move_ftilde_every), dftilde(dftilde),
      |         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~
      |         ftilde_sigma(ftilde_sigma)            scale_and_offset(scale_and_offset)
 4324 |         ftilde_sigma(ftilde_sigma), scale_and_offset(scale_and_offset)
      |         ~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ensembles(ensembles)        sampler(sampler)
cppPosteriorSampler.cpp:5680:19: warning: unused variable 'switch_NN_every' [-Wunused-variable]
 5680 |               int switch_NN_every = 100;
      |                   ^~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4953:15: warning: unused variable 'attempt_move_pmp_every' [-Wunused-variable]
 4953 |     const int attempt_move_pmp_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pmp_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4955:15: warning: unused variable 'attempt_move_pm_prior_sigma_every' [-Wunused-variable]
 4955 |     const int attempt_move_pm_prior_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pm_prior_sigma_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4956:15: warning: unused variable 'attempt_move_pm_extern_loss_sigma_every' [-Wunused-variable]
 4956 |     const int attempt_move_pm_extern_loss_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pm_extern_loss_sigma_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4957:15: warning: unused variable 'attempt_move_DB_sigma_every' [-Wunused-variable]
 4957 |     const int attempt_move_DB_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_DB_sigma_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4958:15: warning: unused variable 'attempt_move_PC_sigma_every' [-Wunused-variable]
 4958 |     const int attempt_move_PC_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_PC_sigma_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4959:15: warning: unused variable 'attempt_move_lambda_every' [-Wunused-variable]
 4959 |     const int attempt_move_lambda_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_lambda_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4960:15: warning: unused variable 'attempt_move_xi_every' [-Wunused-variable]
 4960 |     const int attempt_move_xi_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_xi_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:4961:15: warning: unused variable 'attempt_move_rho_every' [-Wunused-variable]
 4961 |     const int attempt_move_rho_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_rho_every"));
      |               ^~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:5394:10: warning: unused variable 'allowed_early_stopping' [-Wunused-variable]
 5394 |     bool allowed_early_stopping = false;
      |          ^~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:5396:12: warning: unused variable 'gradientNormThreshold' [-Wunused-variable]
 5396 |     double gradientNormThreshold = 0.0;
      |            ^~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:5398:9: warning: unused variable 'print_training_every' [-Wunused-variable]
 5398 |     int print_training_every = 100; // 1000;
      |         ^~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:5419:19: warning: code will never be executed [-Wunreachable-code]
 5419 |                   #pragma omp cancel for
      |                   ^~~~~~~~~~~~~~~~~~~~~~
cppPosteriorSampler.cpp:7361:12: warning: unused variable 'lambda' [-Wunused-variable]
 7361 |     double lambda = expanded_values->at(l)[0];
      |            ^~~~~~
cppPosteriorSampler.cpp:7449:12: warning: unused variable 'lambda' [-Wunused-variable]
 7449 |     double lambda = expanded_values->at(l)[0];
      |            ^~~~~~
24 warnings generated.
ld: warning: duplicate -rpath '/Users/vv/anaconda3/envs/biceps/lib' ignored
(biceps) vv@MacBookPro biceps % ls -lasrt
total 8208
   0 drwxr-xr-x  20 vv  staff      640 Jan 16  2021 __pycache__
   8 -rw-r--r--   1 vv  staff     2372 Jun  5 09:50 decorators.py
 104 -rw-r--r--   1 vv  staff    50920 Jun 10 09:11 Analysis.py
  32 -rw-r--r--   1 vv  staff    15028 Jun 10 09:11 J_coupling.py
  16 -rw-r--r--   1 vv  staff     5285 Jun 10 09:11 KarplusRelation.py
 136 -rw-r--r--   1 vv  staff    66455 Jun 10 09:11 PosteriorSampler.pyx
 152 -rw-r--r--   1 vv  staff    76715 Jun 10 09:11 Restraint.py
  24 -rw-r--r--   1 vv  staff    10200 Jun 10 09:11 XiOpt.py
   8 -rw-r--r--   1 vv  staff      869 Jun 10 09:11 __init__.py
  72 -rw-r--r--   1 vv  staff    33907 Jun 10 09:11 convergence.py
 656 -rw-r--r--   1 vv  staff   332523 Jun 10 09:11 cppPosteriorSampler.cpp
  40 -rw-r--r--   1 vv  staff    18497 Jun 10 09:11 cppPosteriorSampler.h
   8 -rw-r--r--   1 vv  staff     2075 Jun 10 09:11 parse_star.py
  40 -rw-r--r--   1 vv  staff    18587 Jun 10 09:11 rdc.py
   8 -rw-r--r--   1 vv  staff       67 Jun 10 09:11 rebuild.sh
   8 -rw-r--r--   1 vv  staff     2209 Jun 10 09:11 setup.py
  80 -rw-r--r--   1 vv  staff    40594 Jun 10 09:11 toolbox.py
  16 -rw-r--r--   1 vv  staff     6894 Jun 10 09:11 tqdm.h
   0 drwxr-xr-x  22 vv  staff      704 Jun 10 12:52 ..
4640 -rw-r--r--   1 vv  staff  2371959 Jun 10 12:59 PosteriorSampler.cpp
   8 -rw-r--r--   1 vv  staff     4029 Jun 10 12:59 PosteriorSampler.h
   0 drwxr-xr-x   3 vv  staff       96 Jun 10 12:59 build
   0 drwxr-xr-x  24 vv  staff      768 Jun 10 12:59 .
2152 -rwxr-xr-x   1 vv  staff  1100216 Jun 10 12:59 PosteriorSampler.cpython-312-darwin.so
```

## Next, testing if I can run the scripts!

In `examples/simple_three_state_reweighting.py`,  I needed to install the following to get it to work:
```
conda install h5py Bio uncertainties
pip install pynmrstar
```



