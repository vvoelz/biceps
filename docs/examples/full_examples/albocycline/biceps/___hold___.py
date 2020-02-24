for rest_index in range(len(s)):
    nuisance_parameters = getattr(s[rest_index], "_nuisance_parameters")
    for para in nuisance_parameters:
        self.allowed_parameters.append(getattr(s[rest_index], para))
        self.sampled_parameters.append(np.zeros(len(getattr(s[rest_index], para))))


