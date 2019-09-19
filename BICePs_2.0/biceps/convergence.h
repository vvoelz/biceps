#include <iostream>
#include <vector>
using namespace std;

vector< vector<float>> c_autocorrelation(vector< vector<float>> sampled_parameters,
        int maxtau=10000, bool normalize=true);

vector<float> c_autocorrelation_time(vector< vector<float>> autocorr,
        bool normalize=true);



