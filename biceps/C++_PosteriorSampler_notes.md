
# Protocol for C++ sample function

accepts:

• compiled nuisance parameters

state, nsteps, parameters, parameter indices, allowed_parameters, new_state, parameters, parameter_indices


C++ methods needed:
• Trajectory class or method
• neglogP --> needs to be given the energy function to use for specific restraints in sim

```python
    def compute_logZ(self):
        """Compute reference state logZ for the free energies to normalize."""

        Z = 0.0
#        for rest_index in range(len(self.ensemble[0])):
#            for s in self.ensemble[rest_index]:
        for s in self.ensemble:
            Z +=  np.exp( -np.array(s[0].energy, dtype=np.float128) )
        self.logZ = np.log(Z)
        self.ln2pi = np.log(2.0*np.pi)
```


