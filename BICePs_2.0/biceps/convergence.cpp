//##### INCLUDE #####:{{{
#include <iostream> // basic input/output stream
#include <cmath>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cstdarg>
//#include <cstdlib>  // srand() - random number generation
//#include <omp.h>
#include <stdio.h>
#include <numeric>
using namespace std;
using std::vector;
//}}}

// ##### Templates #####: {{{
void print_2D_vec(vector< vector<float> > v) {
    int x = 0;
    for (int i=0; i<v.size(); ++i) {
        for (int j=0; j<v[i].size(); ++j) {
            if (x != i) {
                x += 1;
            }
            cout << v[i][j] << endl;
        }
    }
}

void print_vec(vector<float> v) {
    int x = 0;
    for (int i=0; i<v.size(); ++i) {
        if (x != i) {
            x += 1;
        }
        cout << v[i] << endl;
    }
}

template<class T=float, class U=int>
vector<T> slice_vector(vector<T> vec, U start, U stop, U step=1){
    /* This function will partition a vector. This is a very close equivalent to
     * the python example:
     * vec[start:stop]
     */

   if (step == float(0) || int(0)) {
      throw invalid_argument("step for range must be non-zero");
    }

    vector<T> v;
    // You can't slice nothing...
    if (start == stop) {
        throw invalid_argument("Slice Error: must have a valid range");
    }
    while(1) {
        if (start >= stop) {
            break;
        }
        v.push_back(vec[start]);
        start += step;
    }
    return v;
}

/*}}}*/

// Autocorrelation Method:{{{
vector< vector<float>> c_autocorrelation(vector< vector<float>> sampled_parameters,
        int maxtau=10000, bool normalize=true) {
    /* Calculate the autocorrelaton function for a time-series f(t).
     *
     * :param np.array sampled_parameters: a 2D numpy array containing the time series f(t) for each nuisance parameter
     * :param int max_tau: the maximum autocorrelation time to consider.
     * :param bool normalize: if True, return g(tau)/g[0]
     * :return np.array: a numpy array of size (3, max_tau+1) containing g(tau) for each nuisance parameter
     *
     */

    printf("Calculating autocorrelation...\n");

    vector< vector<float>> result(sampled_parameters.size());

    for (int k=0; k<sampled_parameters.size(); k++) {
        vector<float> f = sampled_parameters[k];
        if (f.size() <= maxtau) {
            printf("The time series is shorter than the\
                    tau values (%i), you need either sample more steps \
                    or change the tau value smaller.",maxtau);
            exit(1);
        }

        float f_mean = accumulate( f.begin(), f.end(), 0.0)/f.size();
        vector<float> f_zeroed;
        for (int i=0; i<f.size(); i++) {
            f_zeroed.push_back(f[i]-f_mean);
        }

        vector<float> _result;
        float norm;
        for (int tau=0; tau<(maxtau+1); tau++) {
            vector<float> series1 = slice_vector(f_zeroed, 0, int(f_zeroed.size()-tau));
            vector<float> series2 = slice_vector(f_zeroed, tau, int(f_zeroed.size()));

            if ((f_zeroed.size()-tau) != 0) {
                _result.push_back(inner_product(
                            series1.begin(),series1.end(),
                            series2.begin(),0.0)/(f_zeroed.size()-tau));

            // destructor: object.clear(); or better yet ~object;
            series1.clear(); //destructor
            series2.clear(); //destructor
            }

            if (normalize) {
                if (tau==0) {
                    norm = _result[0];
                }
                _result[tau] = _result[tau]/norm;
            }
        }
        result[k].swap(_result);
    }
    return result;
}
//}}}



