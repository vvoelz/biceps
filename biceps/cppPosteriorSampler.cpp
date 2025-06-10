
/*
 * Author: Rob Raddi
 * Email: rraddi@temple.edu
 */

/*
 * General description:
 * Python objects are converted into C++ data structures to perform
 * posterior sampling in C++ which allows faster function evaluations,
 * Hamiltonian replica exchange, and many more benefits. Once MCMC sampling
 * is completed, the trajectory data is converted back to Python objects.
 */

//##### INCLUDE #####:{{{
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "tqdm.h"
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include <iterator>
#include <functional>
#include <numeric>
#include <cmath>
#include <random>
#include <ctime>      // contains the time for the random number generation
#include <cstdlib>    // srand() - random number generation
#include <cstdarg>
#include <cstring>
#include <locale>     // std::locale, std::tolower
#include <string>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <stdio.h>
#include <exception>
#include <limits>
#include <type_traits>
#include <omp.h>
#include <thread>
#include <cassert>
//#include <mutex>
//#include <future>
//#include <thread>

//#include <boost/math/differentiation/autodiff.hpp>

//#include <torch/torch.h>
//#include <torch/script.h>

#include <csignal>
//#include <pybind11/pybind11.h>

#include "PosteriorSampler.h"
using namespace std;
using std::vector;


//}}}

// ##### Templates #####: {{{
bool random_bool() {
    // Initialize random number generator with a seed based on the current time
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(0.5); // 50% chance for true or false
    return d(gen);
}
void handleSignal(int signal) {
    if (signal == SIGINT) {
        std::cout << "SIGINT signal (Control-C) received. Exiting program..." << std::endl;
        std::exit(0);  // Gracefully exit the program
    }
}

double sqrt2 = sqrt(2.0);

template <typename T>
bool isIndexValid(const std::vector<T>& vec, size_t index) {
    return index >= 0 && index < vec.size();
}

bool compare(int i, int j) {
  return (i==j);
}

int py_mod(int a, int b)
{
  if (a < 0)
    if (b < 0)
      return -(-a % -b);
    else
      return -a % b - (-a % -b != 0 ? 1 : 0);
  else if (b < 0)
      return -(a % -b) + (-a % -b != 0 ? 1 : 0);
    else
      return a % b;
}

template<typename T>
std::ostream & operator<<(std::ostream & os, std::vector<T> vec)
{
    /* Print 1-D vectors to output stream */
    os<<"{ ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
    os<<"}";
    return os;
}


// Thread-local random engine for all generations of random numbers
thread_local std::mt19937 gen(std::random_device{}());
//thread_local std::mt19937_64 gen(std::random_device{}());  // no benefit from 19937

int get_random_int(int low, int high) {
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

double get_random_double(double low, double high) {
    std::uniform_real_distribution<> dist(low, high);
    return dist(gen);
}


double Gaussian(double mu, double sigma) {
  std::normal_distribution<double> dist(mu, sigma);
  return dist(gen);
}

template<typename T>
std::vector<T> rand_vec_of_ints(T max, T num)
{
    std::uniform_int_distribution<> dis(0, max - 1);
    std::vector<T> v(num);
    std::generate(v.begin(), v.end(), [&]() { return dis(gen); });
    return v;
}

template<typename T>
std::vector<T> rand_vec_of_unique_ints(T max, T num)
{
    std::vector<T> v = rand_vec_of_ints(max, num);
    // Sort and remove duplicates
    std::sort(v.begin(), v.end());
    auto it = std::unique(v.begin(), v.end());
    v.resize(std::distance(v.begin(), it));
    return v;
}

template <typename T>
std::vector<T> generate_sequence(T num) {
    std::vector<T> sequence(num);
    std::iota(sequence.begin(), sequence.end(), static_cast<T>(0));
    return sequence;
}

template<class T = float, class U = int>
std::vector<T> rand_vec_of_floats(T max, U num)
{
    std::uniform_real_distribution<> dis(0, max);
    std::vector<T> v(num);
    std::generate(v.begin(), v.end(), [&]() { return dis(gen); });
    return v;
}



std::string stringFormat(const std::string& format, ...)
{
    va_list args;
    va_start(args, format);

    try {
        size_t len = vsnprintf(NULL, 0, format.c_str(), args);
        vector<char> vec(len + 1);

        va_end(args); // We need to end before we start again
        va_start(args, format);

        vsnprintf(&vec[0], len + 1, format.c_str(), args);

        va_end(args); // Clean up the va_list

        return &vec[0];
    }
    catch (...) {
        va_end(args); // Clean up the va_list in case of an exception
        throw; // Re-throw the caught exception.
    }
}



std::string transformStringToLower(std::string& str)
{
    std::locale loc;
    std::string new_str;
    for (auto s:str) { new_str.push_back(std::tolower(s,loc)); }
    return new_str;
}




template<typename T>
class Combinations {
private:
    vector<vector<T>> res;
    void solve(T n, T k, vector<T>& temp, T start=0) {
        if(static_cast<T>(temp.size()) == k){
            res.push_back(temp);
            return;
        }
        for(T i = start; i <= n; i++){
            temp.push_back(i);
            solve(n, k, temp, i + 1);
            temp.pop_back();
        }
    }
public:
    vector<vector<T>> combinations(T n, T k) {
        res.clear();
        vector<T> temp;
        temp.reserve(k);  // reserve memory for k elements
        solve(n ,k, temp);
        return res;
    }
};

template<typename T>
std::vector<T> where(const std::vector<std::vector<T>>& values, const T& target) {
    std::vector<T> result;
    for (const auto& element : values) {
        if (element.size() > 1 && element[1] == target) {
            result.push_back(element[0]);
        }
    }
    return result;
}

template<typename T>
vector<T> find_index_from_2D_vec(vector<vector<T>> v, T K) {
    vector<T> result;
    for (T i=0; i<static_cast<T>(v.size()); ++i) {
        auto it = find(v[i].begin(), v[i].end(), K);
        if (it != v[i].end()) {
            result.push_back(i);
        }
    }
    return result;
}

template<typename T>
vector<T> find_indices_of_2D_vec(vector<vector<T>> &v, vector<T> &K) {
    vector<T> result;
    for (T i=0; i<static_cast<T>(v.size()); ++i) {
        auto it1 = find(v[i].begin(), v[i].end(), K[0]);
        auto it2 = find(v[i].begin(), v[i].end(), K[1]);
        if (it1 != v[i].end() || it2 != v[i].end()) {
            result.push_back(i);
        }
    }
    return result;
}



double sgn(double v) {
    if (v < 0.0)
        return -1.0;
    else if (v > 0.0)
        return 1.0;
    else
        return 0.5;
}

double heaviside(double value, double halfMax) {
    if (value < 0.0) return 0.0;
    else if (value == 0.0) return halfMax;
    else return 1.0;
}

double incgamma (double x, double a){
    double sum=0;
    double term=1.0/a;
    int n=1;
    while (term != 0){
        sum = sum + term;
        term = term*(x/(a+n));
        n++;
    }
    return pow(x,a)*exp(-1*x)*sum;
}

std::vector<double> collect_sigma_trajectory(const std::vector<std::vector<std::vector<double>>>& data, int i, int p) {
    std::vector<double> result;
    for (const auto& subvector : data) {
        if (i < subvector.size() && p < subvector[i].size()) {
            result.push_back(subvector[i][p]);
        }
    }
    return result;
}

double calculate_variance(const std::vector<double>& data) {
    if (data.size() <= 1) return 0.0;  // Variance is not defined for 0 or 1 element

    // Calculate the mean of the elements
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    // Calculate the variance
    double variance = std::accumulate(data.begin(), data.end(), 0.0,
        [mean](double acc, double x) {
            return acc + std::pow(x - mean, 2);
        }) / (data.size() - 1);  // Using sample variance formula (n-1)

    return variance;
}




tuple<vector<double>, vector<vector<double>>> cpp_get_scalar_couplings_with_derivatives(
    const vector<double>& phi, const vector<double> parameters, const double phi0, const int model_idx) {

    //double epsilon = 2e-1;
    //double shift = 0;
    size_t phi_size = phi.size();
    size_t nparameters = parameters.size();
    vector<double> J(phi_size);
    vector<vector<double>> dJ(nparameters, vector<double>(phi_size));
    vector<double> cos_angles(phi_size);
    vector<double> cos_para4angles(phi_size);
    vector<double> dcos_para4angles(phi_size);
    vector<double> cos_angle_squared(phi_size);
    vector<double> cos_angle_cubed(phi_size);

    // Compute cosines and their squares in a batch
    for (size_t i = 0; i < phi_size; ++i) {
        double angle = phi[i] + phi0;// + shift;
        cos_angles[i] = cos(angle);
        if ( model_idx == 3 ) {
            cos_para4angles[i] = cos(parameters[4]*angle);
            dcos_para4angles[i] = -angle*sin(parameters[4]*angle);
        }
        cos_angle_squared[i] = cos_angles[i] * cos_angles[i];
        cos_angle_cubed[i] = cos_angles[i] * cos_angles[i] * cos_angles[i];
    }

    // Compute J and its derivatives in a batch
    for (size_t i = 0; i < phi_size; ++i) {

        if ( model_idx == 0 ) {
            J[i] = parameters[0] * cos_angle_squared[i];
            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
        }

        else if ( model_idx == 1 ) {
            J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i];
            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
            dJ[1][i] = cos_angles[i]; // Derivative with respect to B
        }


        else if ( model_idx == 2 ) {
            J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i] + parameters[2];
            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
            dJ[1][i] = cos_angles[i]; // Derivative with respect to B
            dJ[2][i] = 1.0; // Derivative with respect to C
            if ( nparameters > 3 ) {
                for (size_t k = 3; k < nparameters; ++k) {
                    //J[i] += parameters[k];
                    //dJ[k][i] = 1.0; // Derivative with respect to D

                    J[i] += 0.0*parameters[k];
                    dJ[k][i] = 0.0; // Derivative with respect to D
                }
            }
        }

        else if ( model_idx == 3 ) {
            J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i] + parameters[2] + parameters[3] * cos_angle_cubed[i];
            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
            dJ[1][i] = cos_angles[i]; // Derivative with respect to B
            dJ[2][i] = 1.0; // Derivative with respect to C
            dJ[3][i] = cos_angle_cubed[i]; // Derivative with respect to D
            if ( nparameters > 4 ) {
                for (size_t k = 4; k < nparameters; ++k) {
                    J[i] += 0.0*parameters[k];
                    dJ[k][i] = 0.0; // Derivative with respect to D
                }
            }

        }

//        else if ( model_idx == 3 ) {
//            J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i] + parameters[2] + parameters[3] * cos_para4angles[i];
//            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
//            dJ[1][i] = cos_angles[i]; // Derivative with respect to B
//            dJ[2][i] = 1.0; // Derivative with respect to C
//            dJ[3][i] = cos_para4angles[i]; // Derivative with respect to D
//            dJ[4][i] = parameters[3] * dcos_para4angles[i]; // Derivative with respect to E
//        }

        //else if ( model_idx == 3 ) {
        //    J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i] + parameters[2] + parameters[3] * cos_angles[i] + parameters[4] * cos_angles[i];
        //    dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
        //    dJ[1][i] = cos_angles[i]; // Derivative with respect to B
        //    dJ[2][i] = 1.0; // Derivative with respect to C
        //    dJ[3][i] = cos_angles[i]; // Derivative with respect to D
        //    dJ[4][i] = cos_angles[i]; // Derivative with respect to E
        //}


//        else if ( model_idx == 3 ) {
//            J[i] = parameters[0] * cos_angle_squared[i] + parameters[1] * cos_angles[i] + parameters[2] + parameters[3] * 0.0 + parameters[4] * 0.0;
//            dJ[0][i] = cos_angle_squared[i]; // Derivative with respect to A
//            dJ[1][i] = cos_angles[i]; // Derivative with respect to B
//            dJ[2][i] = 1.0; // Derivative with respect to C
//            dJ[3][i] = 0.0; // Derivative with respect to D
//            dJ[4][i] = 0.0; // Derivative with respect to E
//        }




        else {
            cout << "Weird" << endl;

        }
    }

    //return {J, dJ, d2J};
    return {J, dJ};
}



double gamma_series(double a, double x, double epsilon = 1e-15, int max_iter = 1000) {
    double sum = 1.0 / a;
    double term = 1.0 / a;

    for (int n = 1; n < max_iter; ++n) {
        term *= x / (a + n);
        sum += term;

        if (std::abs(term) < epsilon * std::abs(sum)) {
            break;
        }
    }

    return sum * std::exp(-x + a * std::log(x));
}

double gamma_continued_fraction(double a, double x, double epsilon = 1e-15, int max_iter = 1000) {
    double b = x + 1 - a;
    double c = 1.0 / std::numeric_limits<double>::min();
    double d = 1.0 / b;
    double h = d;

    for (int n = 1; n < max_iter; ++n) {
        double an = -n * (n - a);
        b += 2;
        d = an * d + b;
        c = b + an / c;

        if (std::abs(d) < std::numeric_limits<double>::min()) {
            d = std::numeric_limits<double>::min();
        }
        if (std::abs(c) < std::numeric_limits<double>::min()) {
            c = std::numeric_limits<double>::min();
        }

        d = 1.0 / d;
        double delta = c * d;
        h *= delta;

        if (std::abs(delta - 1.0) < epsilon) {
            break;
        }
    }

    return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

double tgamma_lower(double a, double x) {
    /* P(a, x) = 1 / gamma(a) * integral from 0 to x of t^(a-1) * exp(-t) dt
     *
     * The tgamma_lower function uses the series expansion method
     * (implemented by gamma_series) for small x values (x < a + 1), and the
     * continued fraction method (implemented by gamma_continued_fraction) for
     * larger x values. These methods ensure good convergence and precision
     * for a wide range of input values. Note that this implementation assumes
     * that the input values are valid and the series converges.
     *
     */
     // For the same function in boost C++ library:
     //https://home.cc.umanitoba.ca/~psgendb/doc/local/pkg/CASAVA_v1.8.2-build/opt/bootstrap/build/boost_1_44_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_gamma/igamma.html

    if (a <= 0 || x < 0) {
        throw std::domain_error("Invalid arguments for tgamma_lower");
    }

    double epsilon = 1e-15;
    int max_iter = 1000;

    if (x < a + 1) {
        return gamma_series(a, x, epsilon, max_iter);
    } else {
        return std::tgamma(a) - gamma_continued_fraction(a, x, epsilon, max_iter);
    }
}

double clipGradient(double gradient, double max_gradient) {
    if (gradient > max_gradient) {
        return max_gradient;
    } else if (gradient < -max_gradient) {
        return -max_gradient;
    }
    return gradient;
}


double adjustGaussianNoiseScale(int iteration, int max_iterations, double initial_scale,
        double min_scale, double convergence_metric, double target_acceptance_rate) {
    /**
     * Adaptively adjusts the Gaussian noise scale based on the iteration number and/or other metrics.
     *
     * iteration - Current iteration number.
     * max_iterations - Maximum number of iterations.
     * initial_scale -  Initial scale of the Gaussian noise.
     * min_scale Minimum -  allowable scale of the Gaussian noise.
     * convergence_metric -  A metric indicating the convergence status (e.g., acceptance rate).
     * returns The adjusted scale for the Gaussian noise.
     */

    // Reduce noise scale linearly with iterations
    double scale_factor = 1.0 - (double(iteration) / max_iterations);
    double adjusted_scale = initial_scale * scale_factor;

    // Ensure scale does not fall below a minimum threshold
    adjusted_scale = std::max(adjusted_scale, min_scale);

    // Adjust based on acceptance rate
    if (convergence_metric < target_acceptance_rate) {
        adjusted_scale *= 1.1; // Increase noise if acceptance rate is below target
    } else {
        adjusted_scale *= 0.9; // Decrease noise if acceptance rate is above target
    }

    return adjusted_scale;
}

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> linspaced;

    if (num == 0) {
        return linspaced;
    }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i = 0; i < num-1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    // Ensure that the end value is exactly as specified to avoid floating-point arithmetic errors
    linspaced.push_back(end);

    return linspaced;
}

/* Flatten, load and print PyTorch stuff: {{{*/
//
//std::vector<double> flatten_parameters(const torch::jit::script::Module& module) {
//    std::vector<double> flat_params;
//    // Iterate through each parameter in the module
//    for (const auto& param : module.parameters()) {
//        // Check if the parameter is defined; this will skip any non-existing bias terms
//        if (!param.defined()) continue;
//        auto param_data = param.contiguous().view(-1).to(torch::kCPU, torch::kDouble);
//        flat_params.insert(flat_params.end(), param_data.data_ptr<double>(), param_data.data_ptr<double>() + param_data.numel());
//    }
//    return flat_params;
//}
//
//std::vector<std::string> flatten_parameter_labels(const torch::jit::script::Module& module) {
//    std::vector<std::string> labels;
//
//    // Iterate through each named parameter in the module
//    for (const auto& named_param : module.named_parameters()) {
//        const std::string& name = named_param.name;
//        const auto& param = named_param.value;
//
//        // Determine the parameter type
//        std::string param_type = "unknown";
//        if (name.find("weight") != std::string::npos) {
//            param_type = "weight";
//        } else if (name.find("bias") != std::string::npos) {
//            param_type = "bias";
//        } else if (name.find("kernel") != std::string::npos) {
//            param_type = "kernel";
//        } else if (name.find("offset") != std::string::npos) {
//            param_type = "offset";
//        }
//
//        // Add the parameter type label for each element in the parameter tensor
//        labels.insert(labels.end(), param.numel(), param_type);
//    }
//
//    return labels;
//}
//
//std::vector<int> get_parameter_indices(const torch::jit::script::Module& module, const std::string& key) {
//    std::vector<int> indices;
//    int current_index = 0;
//
//    // Get all parameters in the module as a flat vector (useful for indexing)
//    auto all_params = module.parameters();
//
//    // Access the specific parameters by attribute key
//    auto specific_params = module.attr(key).toTensorList();
//
//    // Iterate over all_params to find where each specific_param is located
//    for (const auto& param : all_params) {
//        int param_size = param.numel();
//
//        // Check if this param is part of specific_params
//        bool is_specific_param = false;
//        for (const auto& specific_param : specific_params) {
//            if (param.is_same(specific_param)) {
//                is_specific_param = true;
//                break;
//            }
//        }
//
//        // If the parameter is in specific_params, add its indices to the list
//        if (is_specific_param) {
//            for (int i = 0; i < param_size; ++i) {
//                indices.push_back(current_index + i);
//            }
//        }
//
//        // Update current index to the next parameter start
//        current_index += param_size;
//    }
//
//    return indices;
//}
//
//
//void print_parameters(const torch::jit::script::Module& module) {
//    std::cout << "Listing parameters:" << std::endl;
//    for (const auto& param : module.named_parameters()) {
//        std::cout << "Parameter name: " << param.name << ", Defined: " << param.value.defined() << ", Size: ";
//        for (const auto& size : param.value.sizes()) {
//            std::cout << size << " ";
//        }
//        std::cout << std::endl;
//    }
//    cout << flatten_parameters(module) << endl;
//}
//
//
////void load_parameters(torch::jit::script::Module& module, const std::vector<double>& flat_params) {
////    //print_parameters(module);
////    auto params = module.parameters(true); // Retrieve all parameters recursively
////    auto flat_param_iter = flat_params.begin();
////
////    try {
////        torch::NoGradGuard no_grad; // Disable gradient tracking during parameter update
////        for (const auto& param : params) { // Iterate over each parameter
////            if (flat_param_iter == flat_params.end()) {
////                throw std::runtime_error("More parameters in the module than values in flat_params.");
////            }
////
////            if (!param.is_floating_point()) {
////                throw std::runtime_error("Non-floating point parameter encountered, expecting all double types.");
////            }
////
////            // Create a tensor with the same shape as param filled with the value from flat_params
////            auto new_value = torch::full_like(param, *flat_param_iter);
////            param.copy_(new_value); // Update the parameter
////
////            ++flat_param_iter; // Advance to the next element in flat_params
////        }
////
////        if (flat_param_iter != flat_params.end()) {
////            throw std::runtime_error("Not all values in flat_params were used: mismatch in the number of parameters.");
////        }
////    } catch (const std::exception& e) {
////        std::cerr << "Error updating module parameters: " << e.what() << std::endl;
////        throw; // Re-throw the exception after logging
////    }
////    //print_parameters(module);
////    //exit(1);
////
////}
//
//
//void load_parameters(torch::jit::script::Module& module, const std::vector<double>& flat_params) {
//    std::vector<torch::Tensor> params; // Vector to store parameters
//
//    for (const auto& param : module.parameters()) {
//        params.push_back(param);
//    }
//
//    int64_t current_index = 0;
//    for (auto& param : params) {
//        int64_t param_length = param.numel();
//        auto param_data = torch::from_blob(
//            const_cast<double*>(flat_params.data()) + current_index,
//            param.sizes(),
//            torch::TensorOptions().dtype(torch::kDouble)
//        );
//        param.data().copy_(param_data);
//        current_index += param_length;
//    }
//}
//
//


/*}}}*/

///* compute gradients w/ finite diff: {{{*/
//std::tuple<std::vector<double>, std::vector<std::vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module, const std::vector<torch::jit::IValue>& states, double scaling = 1.0, double lambda = 1.0, bool compute_dprior = true) {
//
//    double epsilon = 1e-8;
//    bool debug = 0;
//
//    //PyGILState_STATE gstate = PyGILState_Ensure();
//
//    std::vector<double> parameters = flatten_parameters(module);
//
//    module.train(); // Ensure the module is in training mode to track gradienu
//         //toTensor().squeeze();
//    torch::Tensor prior = module.forward(states).toTensor();
//    std::vector<double> prior_vector(prior.data_ptr<double>(), prior.data_ptr<double>() + prior.numel());
//    //cout << prior_vector << endl;
//    std::vector<std::vector<double>> dprior(parameters.size(), std::vector<double>(prior.numel(), 0.0));
//
//    std::vector<torch::Tensor> params;
//    for (const auto& param : module.parameters()) {
//        params.push_back(param);
//        //param.set_requires_grad(true);
//    }
//    //PyGILState_Release(gstate);
//
//    if (compute_dprior) {
//        std::vector<torch::Tensor> grad_params(params.size());
//
//        for (size_t i = 0; i < params.size(); ++i) {
//            torch::Tensor original_param = params[i].clone();
//            // Perturb the parameter slightly upward
//            params[i] = original_param + epsilon;
//            torch::Tensor output_up = module.forward(states).toTensor().squeeze().to(torch::kDouble);
//
//            // Perturb the parameter slightly downward
//            params[i] = original_param - epsilon;
//            torch::Tensor output_down = module.forward(states).toTensor().squeeze().to(torch::kDouble);
//
//            // Compute the finite difference and restore the parameter
//            grad_params[i] = (output_up - output_down) / (2 * epsilon);
//            params[i] = original_param;  // Restore the original parameter
//        }
//
//        for (size_t j = 0; j < grad_params.size(); ++j) {
//            auto grad_flat = grad_params[j].contiguous().view(-1).to(torch::kDouble);
//            std::copy(grad_flat.data_ptr<double>(), grad_flat.data_ptr<double>() + grad_flat.numel(), dprior[j].begin());
//        }
//    }
//
//    for (auto& grad : dprior) {
//        for (auto& value : grad) { value *= scaling * lambda; }
//    }
//    for (auto& val : prior_vector) { val *= lambda; }
//
//    return {prior_vector, dprior};
//}
///*}}}*/
//

///* compute gradients w/ finite diff with indices: {{{*/
//
//std::tuple<std::vector<double>, std::vector<std::vector<double>>>
//compute_energies_and_derivatives_fd(torch::jit::script::Module& module,
//                                    const std::vector<torch::jit::IValue>& states,
//                                    const std::vector<size_t>& indices,
//                                    double scaling = 1.0, double lambda = 1.0, bool compute_dprior = true) {
//
//    double base_epsilon = 1e-8; // Minimum epsilon
//    torch::NoGradGuard no_grad; // Prevent autograd from tracking these operations
//    module.eval();
//
//    torch::Tensor prior = module.forward(states).toTensor().detach();
//    std::vector<double> prior_vector(prior.data_ptr<double>(), prior.data_ptr<double>() + prior.numel());
//    std::vector<std::vector<double>> dprior(prior.numel(), std::vector<double>(indices.size(), 0.0));
//
//
//    // Finding the minimum value in prior_vector and subtracting it
//    double min_value = *std::min_element(prior_vector.begin(), prior_vector.end());
//    for (double& value : prior_vector) { value -= min_value; }
//
//
//    if (!compute_dprior) { return {prior_vector, dprior}; }
//
//    std::vector<double> parameters = flatten_parameters(module);
//
//    for (size_t idx = 0; idx < indices.size(); ++idx) {
//        size_t i = indices[idx];
//        double original_param = parameters[i];
//        double epsilon = std::max(base_epsilon, std::abs(original_param) * 1e-2); // Adaptive epsilon
//
//        // Perturb parameter upwards
//        parameters[i] = original_param + epsilon;
//        load_parameters(module, parameters);
//        torch::Tensor output_up = module.forward(states).toTensor().detach();
//        std::vector<double> prior_up(output_up.data_ptr<double>(), output_up.data_ptr<double>() + output_up.numel());
//        double min_up = *std::min_element(prior_up.begin(), prior_up.end());
//        for (double& value : prior_up) { value -= min_value; }
//
//        // Perturb parameter downwards
//        parameters[i] = original_param - epsilon;
//        load_parameters(module, parameters);
//        torch::Tensor output_down = module.forward(states).toTensor().detach();
//        std::vector<double> prior_down(output_down.data_ptr<double>(), output_down.data_ptr<double>() + output_down.numel());
//        double min_down = *std::min_element(prior_down.begin(), prior_down.end());
//        for (double& value : prior_down) { value -= min_value; }
//
//        // Compute finite differences
//        for (size_t j = 0; j < prior_vector.size(); ++j) {
//            dprior[j][idx] = (prior_up[j] - prior_down[j]) / (2 * epsilon);
//        }
//        parameters[i] = original_param;  // Restore the original parameter
//    }
//
//    for (auto& grad : dprior) {
//        for (auto& value : grad) value *= scaling * lambda;
//    }
//    for (auto& val : prior_vector) val *= lambda;
//
//    return {prior_vector, dprior};
//}
//
//
///*}}}*/
//

///* compute gradients w/ finite diff with indices: {{{*/
//
//std::tuple<std::vector<double>, std::vector<std::vector<double>>>
//compute_energies_and_derivatives_fd(torch::jit::script::Module& module,
//                                    const std::vector<torch::jit::IValue>& states,
//                                    const std::vector<size_t>& indices,
//                                    double scaling = 1.0, double lambda = 1.0, bool compute_dprior = true) {
//
//    //double base_epsilon = 1e-8; // Minimum epsilon
//    double base_epsilon = 1e-6; // Minimum epsilon
//    torch::NoGradGuard no_grad; // Prevent autograd from tracking these operations
//    module.eval();
//
//    torch::Tensor prior = module.forward(states).toTensor().detach();
//    std::vector<double> prior_vector(prior.data_ptr<double>(), prior.data_ptr<double>() + prior.numel());
//    std::vector<double> parameters = flatten_parameters(module);
//    std::vector<std::vector<double>> dprior(parameters.size(), std::vector<double>(prior.numel(), 0.0));
//
//    if (!compute_dprior) { return {prior_vector, dprior}; }
//
//
//    for (size_t idx = 0; idx < indices.size(); ++idx) {
//        size_t i = indices[idx];
//        const double original_param = parameters[i];
//        double epsilon = std::max(base_epsilon, std::abs(original_param) * 1e-2); // Adaptive epsilon
//
//        // Perturb parameter upwards
//        parameters[i] = original_param + epsilon;
//        load_parameters(module, parameters);
//        torch::Tensor output_up = module.forward(states).toTensor().detach();
//        //printTensor<double>(output_up);
//        vector<double> prior_up(output_up.data_ptr<double>(), output_up.data_ptr<double>() + output_up.numel());
//
//        // Perturb parameter downwards
//        parameters[i] = original_param - epsilon;
//        load_parameters(module, parameters);
//        torch::Tensor output_down = module.forward(states).toTensor().detach();
//        vector<double> prior_down(output_down.data_ptr<double>(), output_down.data_ptr<double>() + output_down.numel());
//
//        // Compute finite differences
//        for (size_t j = 0; j < dprior[0].size(); ++j) {
//            dprior[i][j] = (prior_up[j] - prior_down[j]) / (2 * epsilon);
//        }
//        parameters[i] = original_param;  // Restore the original parameter
//    }
//
////    for (auto& grad : dprior) {
////        for (auto& value : grad) value *= scaling * lambda;
////    }
////    for (auto& val : prior_vector) val *= lambda;
//
//    return {prior_vector, dprior};
//}
//
//
///*}}}*/
//

///* compute_energies_and_derivatives: {{{*/
//
//std::tuple<std::vector<double>, std::vector<std::vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                 std::vector<torch::jit::IValue>& inputs,
//                                 double scaling = 1.0, double lambda = 1.0,
//                                 bool compute_gradients = true) {
//    module.train();
//
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        std::cerr << "Output tensor is not of floating point type." << std::endl;
//        return {};
//    }
//
//    std::vector<double> output_vector(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    std::transform(output_vector.begin(), output_vector.end(), output_vector.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.sizes()[0];
//    }
//    //cout << nstates << endl;
//    size_t num_parameter_elements = 0; // Count number of elements in each parameter
//    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
//    std::vector<std::vector<double>> all_gradients(num_parameter_elements, std::vector<double>(nstates, 0.0));
//
//    if (!compute_gradients) { return {output_vector, all_gradients}; }
//    // Release Python GIL once before the loop
//    PyThreadState* thread_state = PyEval_SaveThread();
//    ////////////////////////////////////////////////////////////////////////////
//    ////////////////////////////////////////////////////////////////////////////
//
//    size_t state_index = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        //for (int k = 0; k < tensor.numel(); ++k) {
//        for (int k = 0; k < nstates; ++k) {
//            auto output = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
//            output.backward(torch::ones_like(output));
//
//            size_t param_index = 0;
//            for (const auto& param : module.parameters()) {
//                if (param.grad().defined()) {
//                    auto grad_data = param.grad().view(-1);
//                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                        all_gradients[param_index][state_index] = grad_data[i].item<double>();
//                    }
//                    param.grad().zero_();
//                }
//            }
//            ++state_index;
//        }
//    }
//
//    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
//
//    // Apply scaling to gradients
//    for (auto& gradients : all_gradients) {
//        std::transform(gradients.begin(), gradients.end(), gradients.begin(), [scaling](double v) { return v * scaling; });
//    }
//
//    return {output_vector, all_gradients};
//}
//
///*}}}*/
//


/* NOTE:  good */
///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                 vector<torch::jit::IValue>& inputs,
//                                 double scaling = 1.0, double lambda = 1.0,
//                                 bool return_jac = true) {
//    bool debug = 1;
//
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.sizes()[0];
//    }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
//
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
//    vector<vector<vector<double>>> hessian;
//
//    if (!return_jac) { return {result, jac}; }
//
//    //py::gil_scoped_release no_gil;
////    py::gil_scoped_release release;
//
//
//    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
//
//
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        for (int k = 0; k < nstates; ++k) {
//            auto result = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
//            result.backward(torch::ones_like(result), true);  // Enable gradient accumulation
//            size_t param_index = 0;
//            for (const auto& param : module.parameters()) {
//                if (param.grad().defined()) {
//                    auto grad_data = param.grad().view(-1);
//                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                        jac[param_index][k] = grad_data[i].item<double>();
//                    }
//                    param.grad().zero_();
//                }
//            }
//        }
//    }
//
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//
//
//    return {result, jac};
//}
//
///*}}}*/
//

///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                 vector<torch::jit::IValue>& inputs,
//                                 double scaling = 1.0, double lambda = 1.0,
//                                 bool return_jac = true) {
//    bool debug = 1;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.sizes()[0];
//    }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
//
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
//    vector<vector<vector<double>>> hessian;
//
//    if (!return_jac) { return {result, jac}; }
//
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        for (int k = 0; k < nstates; ++k) {
//            auto result = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
//            result.backward(torch::ones_like(result), true);  // Enable gradient accumulation
//            size_t param_index = 0;
//            for (const auto& param : module.parameters()) {
//                if (param.grad().defined()) {
//                    auto grad_data = param.grad().view(-1);
//                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                        jac[param_index][k] = grad_data[i].item<double>();
//                    }
//                    param.grad().zero_();
//                }
//            }
//        }
//    }
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//
//    return {result, jac};
//}
//
///*}}}*/
//


//
///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                          vector<torch::jit::IValue>& inputs,
//                                          double scaling = 1.0, double lambda = 1.0,
//                                          bool return_jac = true) {
//    bool debug = 1;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    // Forward pass to get energies
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.size(0);
//    }
//
//    // Flatten all parameters into a single vector
//    vector<torch::Tensor> parameters;
//    for (const auto& param : module.parameters()) { parameters.push_back(param); }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : parameters) { num_parameter_elements += param.numel(); }
//
//    // Initialize Jacobian matrix
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(nstates, 0.0));
//
//    if (!return_jac) {
//        auto end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed_time = end_time - start_time;
//        if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//        return {result, jac};
//    }
//
//    // Prepare batched input
//    // Assuming all input tensors have the same shape except the first dimension
//    // Concatenate all input tensors along the first dimension
//    vector<torch::Tensor> tensors = {inputs[0].toTensor()};
//    torch::Tensor batched_input = torch::cat(tensors, 0).detach().requires_grad_(true);
//
//    // Perform forward pass with batched input
//    torch::Tensor batched_output = module.forward({batched_input}).toTensor();
//
//    // Check if batched_output has shape [nstates, ...]
//    // Assuming the output per state is a scalar
//    // If not, additional handling is needed
//    assert(batched_output.size(0) == nstates);
//
//    // Create a gradient tensor with shape [nstates]
//    // Each element in the gradient corresponds to the gradient for one state
//    // To compute the Jacobian, we need to compute gradients for each output separately
//    // This can be done using torch::autograd::grad with create_graph=true
//
//    // Pre-flatten all parameters for easier indexing
//    vector<torch::Tensor> flat_parameters;
//    for (auto& param : parameters) { flat_parameters.push_back(param.view(-1)); }
//
//    // Concatenate all flat parameters
//    torch::Tensor all_params = torch::cat(flat_parameters, 0);
//
//    // Compute gradients for all states in a batched manner
//    // This leverages the fact that torch::autograd::grad can compute multiple gradients at once
//    // However, in libtorch, it's not as straightforward as in Python
//    // As an alternative, perform a single backward pass with a gradient vector of ones
//    // and manually extract per-sample gradients if possible
//
//    // This approach computes the sum of gradients, which is not what we want
//    // Therefore, we need to compute gradients per sample
//    // To optimize, we can use torch::autograd::grad with retain_graph=true
//
//    // However, libtorch might not support advanced autograd features like functorch
//    // So, we might still need to loop, but with optimized operations
//
//    // Loop over all states and compute gradients
//    // To optimize, minimize operations inside the loop
//
//    size_t param_index = 0;
//    // Iterate over all states
//    for (int k = 0; k < nstates; ++k) {
//        // Compute gradient for the k-th state
//        // Select the k-th output
//        torch::Tensor output_k = batched_output[k];
//        output_k.backward(torch::ones_like(output_k), /*retain_graph=*/true);
//
//        // Iterate over all parameters
//        for (const auto& param : module.parameters()) {
//            if (param.grad().defined()) {
//                auto grad_data = param.grad().view(-1);
//                for (int i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                    jac[param_index][k] = grad_data[i].item<double>();
//                }
//                param.grad().zero_();
//            }
//        }
//        param_index = 0; // Reset for next state
//    }
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//
//    return {result, jac};
//}
//
///*}}}*/
//



/* hold onto this: */
///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                          vector<torch::jit::IValue>& inputs,
//                                          double scaling = 1.0, double lambda = 1.0,
//                                          bool return_jac = true) {
//    bool debug = 1;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    // Forward pass to get energies
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.size(0);
//    }
//
//    // Flatten all parameters into a single vector
//    vector<torch::Tensor> parameters;
//    for (const auto& param : module.parameters()) { parameters.push_back(param); }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : parameters) { num_parameter_elements += param.numel(); }
//
//    // Initialize Jacobian matrix
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(nstates, 0.0));
//
//    if (!return_jac) {
//        auto end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed_time = end_time - start_time;
//        if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//        return {result, jac};
//    }
//
//    // Prepare batched input
//    // Assuming all input tensors have the same shape except the first dimension
//    // Concatenate all input tensors along the first dimension
//    vector<torch::Tensor> tensors = {inputs[0].toTensor()};
//    torch::Tensor batched_input = torch::cat(tensors, 0).detach().requires_grad_(true);
//
//    // Perform forward pass with batched input
//    torch::Tensor batched_output = module.forward({batched_input}).toTensor();
//
//    // Check if batched_output has shape [nstates, ...]
//    // Assuming the output per state is a scalar
//    // If not, additional handling is needed
//    assert(batched_output.size(0) == nstates);
//
//    // Create a gradient tensor with shape [nstates]
//    // Each element in the gradient corresponds to the gradient for one state
//    // To compute the Jacobian, we need to compute gradients for each output separately
//    // This can be done using torch::autograd::grad with create_graph=true
//
//    // Pre-flatten all parameters for easier indexing
//    vector<torch::Tensor> flat_parameters;
//    for (auto& param : parameters) { flat_parameters.push_back(param.view(-1)); }
//
//    // Concatenate all flat parameters
//    torch::Tensor all_params = torch::cat(flat_parameters, 0);
//
//    // Compute gradients for all states in a batched manner
//    // This leverages the fact that torch::autograd::grad can compute multiple gradients at once
//    // However, in libtorch, it's not as straightforward as in Python
//    // As an alternative, perform a single backward pass with a gradient vector of ones
//    // and manually extract per-sample gradients if possible
//
//    // This approach computes the sum of gradients, which is not what we want
//    // Therefore, we need to compute gradients per sample
//    // To optimize, we can use torch::autograd::grad with retain_graph=true
//
//    // However, libtorch might not support advanced autograd features like functorch
//    // So, we might still need to loop, but with optimized operations
//
//    // Loop over all states and compute gradients
//    // To optimize, minimize operations inside the loop
//
//
//// Pre-allocate Jacobian tensor with the same data type as parameters
//torch::Tensor jacobian = torch::zeros({static_cast<long>(num_parameter_elements), static_cast<long>(nstates)}, parameters[0].options());
//
//// Iterate over all states
//for (int k = 0; k < nstates; ++k) {
//    // Compute gradient for the k-th state
//    torch::Tensor output_k = batched_output[k];
//    output_k.backward(torch::ones_like(output_k), /*retain_graph=*/true);
//
//    // Flatten all gradients into a single vector
//    std::vector<torch::Tensor> grad_tensors;
//    for (const auto& param : parameters) {
//        if (param.grad().defined()) {
//            grad_tensors.push_back(param.grad().view(-1).detach());
//            param.grad().zero_();
//        } else {
//            grad_tensors.push_back(torch::zeros({param.numel()}, param.options()));
//        }
//    }
//    torch::Tensor grad_vec = torch::cat(grad_tensors);
//
//    // Store grad_vec in jacobian
//    jacobian.slice(/*dim=*/1, /*start=*/k, /*end=*/k+1).copy_(grad_vec.unsqueeze(1));
//}
//
//// Convert jacobian tensor to vector<vector<double>>
//if (jacobian.scalar_type() == torch::kDouble) {
//    // If jacobian is Double, directly access and store values
//    auto jacobian_accessor = jacobian.accessor<double, 2>();
//    for (size_t i = 0; i < num_parameter_elements; ++i) {
//        for (int k = 0; k < nstates; ++k) {
//            jac[i][k] = jacobian_accessor[i][k];
//        }
//    }
//} else if (jacobian.scalar_type() == torch::kFloat) {
//    // If jacobian is Float, access as float and cast to double
//    auto jacobian_accessor = jacobian.accessor<float, 2>();
//    for (size_t i = 0; i < num_parameter_elements; ++i) {
//        for (int k = 0; k < nstates; ++k) {
//            jac[i][k] = static_cast<double>(jacobian_accessor[i][k]);
//        }
//    }
//} else {
//    throw std::runtime_error("Unsupported tensor scalar type for jacobian.");
//}
//
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//
//    return {result, jac};
//}
//
///*}}}*/
//


///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                          vector<torch::jit::IValue>& inputs,
//                                          double scaling = 1.0, double lambda = 1.0,
//                                          bool return_jac = true) {
//    bool debug = 1;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    // Forward pass to get energies
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.size(0);
//    }
//
//    // Flatten all parameters into a single vector
//    vector<torch::Tensor> parameters;
//    for (const auto& param : module.parameters()) { parameters.push_back(param); }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : parameters) { num_parameter_elements += param.numel(); }
//
//    // Initialize Jacobian matrix
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(nstates, 0.0));
//
//    if (!return_jac) {
//        auto end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed_time = end_time - start_time;
//        if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//        return {result, jac};
//    }
//
//
//
//
//
//    // Pre-flatten all parameters for easier indexing
//    vector<torch::Tensor> flat_parameters;
//    for (auto& param : parameters) { flat_parameters.push_back(param.view(-1)); }
//
//    // Concatenate all flat parameters
//    torch::Tensor all_params = torch::cat(flat_parameters, 0);
//
//    torch::Tensor batched_output = module.forward({torch::jit::IValue(inputs[0].toTensor().detach().requires_grad_(true))}).toTensor();
//
//// Pre-allocate Jacobian tensor with the same data type as parameters
//torch::Tensor jacobian = torch::zeros({static_cast<long>(num_parameter_elements), static_cast<long>(nstates)}, parameters[0].options());
//
//// Iterate over all states
//for (int k = 0; k < nstates; ++k) {
//    // Compute gradient for the k-th state
//    torch::Tensor output_k = batched_output[k];
//    output_k.backward(torch::ones_like(output_k), /*retain_graph=*/true);
//
//    // Flatten all gradients into a single vector
//    std::vector<torch::Tensor> grad_tensors;
//    for (const auto& param : module.parameters()) {
//        if (param.grad().defined()) {
//            grad_tensors.push_back(param.grad().view(-1).detach());
//            param.grad().zero_();
//        } else {
//            grad_tensors.push_back(torch::zeros({param.numel()}, param.options()));
//        }
//    }
//    for (const auto& param : module.parameters()) {
//        if (param.grad().defined()) {
//            auto grad_data = param.grad().view(-1);
//            for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                jac[param_index][k] = grad_data[i].item<double>();
//            }
//            param.grad().zero_();
//        } else {
//            grad_tensors.push_back(torch::zeros({param.numel()}, param.options()));
//        }
//
//    }
//
//
//
//
//    torch::Tensor grad_vec = torch::cat(grad_tensors);
//
//    // Store grad_vec in jacobian
//    jacobian.slice(/*dim=*/1, /*start=*/k, /*end=*/k+1).copy_(grad_vec.unsqueeze(1));
//}
//
//    printTensor<double>(jacobian);
//    exit(1);
//
//
//
//
///*
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        for (int k = 0; k < nstates; ++k) {
//            auto result = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
//            result.backward(torch::ones_like(result), true);  // Enable gradient accumulation
//            size_t param_index = 0;
//            for (const auto& param : module.parameters()) {
//                if (param.grad().defined()) {
//                    auto grad_data = param.grad().view(-1);
//                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
//                        jac[param_index][k] = grad_data[i].item<double>();
//                    }
//                    param.grad().zero_();
//*/
//
//
//
//// Convert jacobian tensor to vector<vector<double>>
//if (jacobian.scalar_type() == torch::kDouble) {
//    // If jacobian is Double, directly access and store values
//    auto jacobian_accessor = jacobian.accessor<double, 2>();
//    for (size_t i = 0; i < num_parameter_elements; ++i) {
//        for (int k = 0; k < nstates; ++k) {
//            jac[i][k] = jacobian_accessor[i][k];
//        }
//    }
//} else if (jacobian.scalar_type() == torch::kFloat) {
//    // If jacobian is Float, access as float and cast to double
//    auto jacobian_accessor = jacobian.accessor<float, 2>();
//    for (size_t i = 0; i < num_parameter_elements; ++i) {
//        for (int k = 0; k < nstates; ++k) {
//            jac[i][k] = static_cast<double>(jacobian_accessor[i][k]);
//        }
//    }
//} else {
//    throw std::runtime_error("Unsupported tensor scalar type for jacobian.");
//}
//
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) { cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" <<  endl; }
//
//    return {result, jac};
//}
//
///*}}}*/
//

///* compute_energies_and_derivatives: {{{*/
//tuple<vector<double>, vector<vector<double>>>
//compute_energies_and_derivatives(torch::jit::script::Module& module,
//                                 vector<torch::jit::IValue>& inputs,
//                                 double scaling = 1.0, double lambda = 1.0,
//                                 bool return_jac = true) {
//    bool debug = 0;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    module.train();
//
//    // Forward pass to get energies
//    auto outputs = module.forward(inputs).toTensor();
//    if (!outputs.is_floating_point()) {
//        cerr << "Output tensor is not of floating point type." << endl;
//        return {};
//    }
//    if (outputs.scalar_type() != torch::kDouble) { outputs = outputs.to(torch::kDouble); }
//
////    cout << "outputs.size(0) = "<< outputs.size(0) << endl;
//
//
//    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
//    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
//
////    cout << result << endl;
//
//    size_t nstates = 0;
//    for (auto& state : inputs) {
//        torch::Tensor tensor = state.toTensor();
//        nstates += tensor.size(0);
//    }
//
//    // Flatten all parameters into a single vector
//    vector<torch::Tensor> parameters;
//    for (const auto& param : module.parameters()) {
//        parameters.push_back(param);
//    }
//
//    size_t num_parameter_elements = 0;
//    for (const auto& param : parameters) {
//        num_parameter_elements += param.numel();
//    }
//
//    // Initialize Jacobian matrix
//    vector<vector<double>> jac(num_parameter_elements, vector<double>(nstates, 0.0));
//
//    if (!return_jac) {
//        auto end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed_time = end_time - start_time;
//        if (debug) {
//            cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" << endl;
//        }
//        return {result, jac};
//    }
//
//    // Prepare batched input
//    vector<torch::Tensor> tensors;
//    for (auto& input : inputs) {
//        tensors.push_back(input.toTensor());
//    }
//    torch::Tensor batched_input = torch::cat(tensors, 0).detach().requires_grad_(true);
//
//    // Perform forward pass with batched input
//    torch::Tensor batched_output = module.forward({batched_input}).toTensor();
//
//    if (batched_output.scalar_type() != torch::kDouble) { batched_output = batched_output.to(torch::kDouble); }
//
//
////    cout << "batched_output.size(0) = "<< batched_output.size(0) << endl;
////    exit(1);
//
//    // Check if batched_output has shape [nstates, ...]
//    assert(batched_output.size(0) == nstates);
//
//    // Pre-allocate Jacobian tensor with the same data type and device as parameters
//    torch::Tensor jacobian = torch::zeros({static_cast<long>(num_parameter_elements), static_cast<long>(nstates)}, parameters[0].options());
//
//    // Iterate over all states
////    #pragma omp parallel for
//    for (int k = 0; k < nstates; ++k) {
//        // Compute gradient for the k-th state
//        torch::Tensor output_k = batched_output[k];
//        output_k.backward(torch::ones_like(output_k), /*retain_graph=*/true);
//
//        // Flatten all gradients into a single vector
//        std::vector<torch::Tensor> grad_tensors;
//        for (const auto& param : module.parameters()) {
//            if (param.grad().defined()) {
//                grad_tensors.push_back(param.grad().view(-1).detach().clone());
//                param.grad().zero_();
//            } else {
//                grad_tensors.push_back(torch::zeros({param.numel()}, param.options()));
//            }
//        }
//        torch::Tensor grad_vec = torch::cat(grad_tensors);
//
//        // Store grad_vec in jacobian
//        jacobian.slice(/*dim=*/1, /*start=*/k, /*end=*/k+1).copy_(grad_vec.unsqueeze(1));
//    }
//    //printTensor<double>(jacobian);
//    //exit(1);
//
//    // Convert jacobian tensor to vector<vector<double>>
//    if (jacobian.scalar_type() == torch::kDouble) {
//        auto jacobian_accessor = jacobian.accessor<double, 2>();
//        for (size_t i = 0; i < num_parameter_elements; ++i) {
//            for (int k = 0; k < nstates; ++k) {
//                jac[i][k] = jacobian_accessor[i][k];
//            }
//        }
//    } else if (jacobian.scalar_type() == torch::kFloat) {
//        auto jacobian_accessor = jacobian.accessor<float, 2>();
//        for (size_t i = 0; i < num_parameter_elements; ++i) {
//            for (int k = 0; k < nstates; ++k) {
//                jac[i][k] = static_cast<double>(jacobian_accessor[i][k]);
//            }
//        }
//    } else {
//        throw std::runtime_error("Unsupported tensor scalar type for jacobian.");
//    }
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed_time = end_time - start_time;
//    if (debug) {
//        cout << "compute_energies_and_derivatives time: " << elapsed_time.count() << "s" << endl;
//    }
//
//    return {result, jac};
//}
///*}}}*/
//







///* old functions of compute_energies_and_derivatives: {{{*/
//
//
/////* compute_energies_and_derivatives: {{{*/
////tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>>
////compute_energies_and_derivatives(torch::jit::script::Module& module,
////                                 vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool return_jac = true,
////                                 bool return_hessian = false) {
////    module.train();
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        cerr << "Output tensor is not of floating point type." << endl;
////        return {};
////    }
////
////    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.sizes()[0];
////    }
////
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////
////    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
////    //vector<vector<vector<double>>> hessian(num_parameter_elements,
////    //        vector<vector<double>>(num_parameter_elements, vector<double>(result.size(), 0.0)));
////
////    std::vector<std::vector<std::vector<double>>> hessian;
////    if (return_hessian) {
////        hessian.resize(num_parameter_elements,
////            std::vector<std::vector<double>>(num_parameter_elements,
////                std::vector<double>(result.size(), 0.0)));
////    }
////
////
////    if (!return_jac) { return {result, jac, hessian}; }
////    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
////
////    size_t state_index = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        for (int k = 0; k < nstates; ++k) {
////            auto result = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
////            result.backward(torch::ones_like(result));
////
////            size_t param_index = 0;
////            for (const auto& param : module.parameters()) {
////                if (param.grad().defined()) {
////                    auto grad_data = param.grad().view(-1);
////                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
////                        jac[param_index][state_index] = grad_data[i].item<double>();
////                    }
////                    param.grad().zero_();
////                }
////            }
////            if (return_hessian) {
////                size_t param_index_1 = 0;
////                for (const auto& param1 : module.parameters()) {
////                    if (!param1.grad().defined()) continue;
////                    auto grad1 = param1.grad().detach().requires_grad_(true);
////                    for (size_t k = 0; k < grad1.size(0); ++k) {
////                        // Zero gradients for all parameters
////                        for (const auto& param : module.parameters()) {
////                            if (param.grad().defined()) {
////                                param.grad().zero_();
////                            }
////                        }
////                        // Compute gradients of the second order
////                        grad1[k].backward(torch::ones_like(grad1[k]), true);
////
////                        size_t param_index_2 = 0;
////                        for (const auto& param2 : module.parameters()) {
////                            if (param2.grad().defined()) {
////                                auto second_order_grad = param2.grad().view(-1);
////                                for (size_t l = 0; l < second_order_grad.size(0); ++l) {
////                                    hessian[param_index_1 + k][param_index_2 + l][state_index] = second_order_grad[l].item<double>();
////                                }
////                            }
////                            param_index_2 += param2.numel();
////                        }
////                    }
////                    param_index_1 += param1.numel();
////                }
////            }
////
////            ++state_index;
////        }
////    }
////
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
//////    // Apply scaling to gradients and hessian
//////    for (auto& gradients : jac) {
//////        transform(gradients.begin(), gradients.end(), gradients.begin(), [scaling](double v) { return v * scaling; });
//////    }
//////
//////    if (return_hessian) {
//////        for (auto& matrix : hessian) {
//////            for (auto& row : matrix) {
//////                transform(row.begin(), row.end(), row.begin(), [scaling](double v) { return v * scaling; });
//////            }
//////        }
//////    }
////
////    return {result, jac, hessian};
////}
////
/////*}}}*/
////
//
//
//
//
/////* NOTE:  good */
/////* compute_energies_and_derivatives: {{{*/
////tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>>
////compute_energies_and_derivatives(torch::jit::script::Module& module,
////                                 vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool return_jac = true,
////                                 bool return_hessian = false) {
////    module.train();
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        cerr << "Output tensor is not of floating point type." << endl;
////        return {};
////    }
////
////    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.sizes()[0];
////    }
////
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////
////    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
////    vector<vector<vector<double>>> hessian;
////    if (return_hessian) {
////        hessian.resize(num_parameter_elements,
////            vector<vector<double>>(num_parameter_elements,
////                vector<double>(result.size(), 0.0)));
////    }
////
////    if (!return_jac) { return {result, jac, hessian}; }
////
////    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
////
////    size_t state_index = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        for (int k = 0; k < nstates; ++k) {
////            auto result = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
////            result.backward(torch::ones_like(result), true);  // Enable gradient accumulation
////
////            size_t param_index = 0;
////            for (const auto& param : module.parameters()) {
////                if (param.grad().defined()) {
////                    auto grad_data = param.grad().view(-1);
////                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
////                        jac[param_index][state_index] = grad_data[i].item<double>();
////                    }
////                    param.grad().zero_();
////                }
////            }
////
////            if (return_hessian) {
////                size_t param_index_1 = 0;
////                for (const auto& param1 : module.parameters()) {
////                    if (!param1.grad().defined()) continue;
////                    auto grad1 = param1.grad().detach().requires_grad_(true);
////                    for (size_t k = 0; k < grad1.size(0); ++k) {
////                        // Zero gradients for all parameters
////                        for (const auto& param : module.parameters()) {
////                            if (param.grad().defined()) {
////                                param.grad().zero_();
////                            }
////                        }
////                        // Compute gradients of the second order
////                        grad1[k].backward(torch::ones_like(grad1[k]), true);
////
////                        size_t param_index_2 = 0;
////                        for (const auto& param2 : module.parameters()) {
////                            if (param2.grad().defined()) {
////                                auto second_order_grad = param2.grad().view(-1);
////                                for (size_t l = 0; l < second_order_grad.size(0); ++l) {
////                                    hessian[param_index_1 + k][param_index_2 + l][state_index] = second_order_grad[l].item<double>();
////                                }
////                            }
////                            param_index_2 += param2.numel();
////                        }
////                    }
////                    param_index_1 += param1.numel();
////                }
////            }
////
////            ++state_index;
////        }
////    }
////
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
////    return {result, jac, hessian};
////}
////
/////*}}}*/
////
////
/////* compute_energies_and_derivatives: {{{*/
////tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>>
////compute_energies_and_derivatives(torch::jit::script::Module& module,
////                                 vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool return_jac = true,
////                                 bool return_hessian = false) {
////    module.train();
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        cerr << "Output tensor is not of floating point type." << endl;
////        return {};
////    }
////
////    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.sizes()[0];
////    }
////
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////
////    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
////    vector<vector<vector<double>>> hessian;
////    if (return_hessian) {
////        hessian.resize(num_parameter_elements,
////            vector<vector<double>>(num_parameter_elements,
////                vector<double>(result.size(), 0.0)));
////    }
////
////    if (!return_jac) { return {result, jac, hessian}; }
////
////    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
////
////    // Batch process states
////    vector<torch::Tensor> tensors;
////    for (auto& input : inputs) {
////        auto tensor = input.toTensor();
////        if (tensor.dim() == 1) {
////            tensor = tensor.unsqueeze(-1);
////        }
////        tensors.push_back(tensor);
////    }
////
////    torch::Tensor input_tensor = torch::cat(tensors, 0).detach().requires_grad_(true);
////    auto batch_outputs = module.forward({input_tensor}).toTensor();
////    batch_outputs.backward(torch::ones_like(batch_outputs));
////
////    size_t param_index = 0;
////    for (const auto& param : module.parameters()) {
////        if (param.grad().defined()) {
////            auto grad_data = param.grad().view(-1);
////            for (size_t i = 0; i < grad_data.size(0); ++i) {
////                jac[param_index + i][i / nstates] = grad_data[i].item<double>();
////            }
////            param_index += grad_data.size(0);
////            param.grad().zero_();
////        }
////    }
////
////
//////    if (return_hessian) {
//////        auto grad1 = input_tensor.grad().detach().requires_grad_(true);
//////        for (size_t i = 0; i < grad1.size(0); ++i) {
//////            // Zero gradients for all parameters
//////            for (const auto& param : module.parameters()) {
//////                if (param.grad().defined()) {
//////                    param.grad().zero_();
//////                }
//////            }
//////            // Compute gradients of the second order
//////            grad1[i].backward(torch::ones_like(grad1[i]), true);
//////
//////            size_t param_index_1 = 0;
//////            for (const auto& param1 : module.parameters()) {
//////                if (param1.grad().defined()) {
//////                    auto second_order_grad = param1.grad().view(-1);
//////                    for (size_t k = 0; k < second_order_grad.size(0); ++k) {
//////                        hessian[param_index_1 + k][i].assign(second_order_grad[k].data_ptr<double>(), second_order_grad[k].data_ptr<double>() + result.size());
//////                    }
//////                }
//////                param_index_1 += param1.numel();
//////            }
//////        }
//////    }
////
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
////    return {result, jac, hessian};
////}
////
/////*}}}*/
////
//
//
//
//
//
/////* compute_forward_and_backward: {{{*/
////tuple<vector<double>, vector<vector<double>>>
////compute_forward_and_backward(torch::jit::script::Module& module,
////                                 vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool return_jac = true) {
////    module.train();
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        cerr << "Output tensor is not of floating point type." << endl;
////        return {};
////    }
////
////    vector<double> result(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    transform(result.begin(), result.end(), result.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.sizes()[0];
////    }
////
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////
////    vector<vector<double>> jac(num_parameter_elements, vector<double>(result.size(), 0.0));
////
////
////    if (!return_jac) { return {result, jac}; }
////    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
////
////    // Ensure gradients are computed
////    outputs.backward(torch::ones_like(outputs));
////
////    // Collect the gradients for each parameter
////    size_t param_idx = 0;
////    for (const auto& param : module.parameters()) {
////        auto grad = param.grad().view(-1);  // Flatten the gradient tensor
////        for (size_t i = 0; i < result.size(); ++i) {
////            for (size_t j = 0; j < grad.numel(); ++j) {
////                jac[i][param_idx + j] = grad[j].item<double>();
////            }
////        }
////        param_idx += grad.numel();
////    }
////
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
////    return {result, jac};
////}
////
/////*}}}*/
////
//
//
//
////
/////* compute_energies_and_derivatives_batched: {{{*/
////
////std::tuple<std::vector<double>, std::vector<double>>
////compute_energies_and_derivatives_batched(torch::jit::script::Module& module,
////                                       std::vector<torch::jit::IValue>& inputs,
////                                       double scaling = 1.0, double lambda = 1.0,
////                                       bool compute_gradients = true) {
////    module.train();  // Set module to training mode
////
////    // Prepare batch from inputs
////    std::vector<torch::Tensor> tensors;
////    for (const auto& input : inputs) {
////        tensors.push_back(input.toTensor());
////    }
////    auto batched_inputs = torch::stack(tensors); // Create a batched tensor from the vector of tensors
////
////    auto outputs = module.forward({batched_inputs}).toTensor();
////
////
////    if (!outputs.is_floating_point()) {
////        std::cerr << "Output tensor is not of floating point type." << std::endl;
////        return {}; // Error handling
////    }
////
////    size_t num_parameter_elements = 0; // Count number of elements in each parameter
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////
////
////    std::vector<double> output_vector(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    std::transform(output_vector.begin(), output_vector.end(), output_vector.begin(), [lambda](double v) { return v * lambda; });
////
////    if (!compute_gradients) {
////        return {output_vector, std::vector<double>(num_parameter_elements, 0.0)};
////    }
////
////    // Release Python GIL before heavy computation
////    PyThreadState* thread_state = PyEval_SaveThread();
////
////    outputs.backward(torch::ones_like(outputs));
////
////    outputs = module.forward({batched_inputs.detach().requires_grad_(true)}).toTensor();
////
////    std::vector<double> parameter_gradients;
////    for (const auto& param : module.parameters()) {
////        if (param.grad().defined()) {
////            auto grad_data = param.grad().view(-1);
////            double sum_gradients = 0.0;
////            for (int i = 0; i < grad_data.size(0); ++i) {
////                sum_gradients += grad_data[i].item<double>();
////            }
////            parameter_gradients.push_back(sum_gradients * scaling); // Apply scaling directly
////        }
////        param.grad().zero_(); // Reset gradients after processing
////    }
////    // Re-acquire Python GIL after computation
////    PyEval_RestoreThread(thread_state);
////
////
////    return {output_vector, parameter_gradients};
////}
////
////
/////*}}}*/
////
//
//
/////* compute_energies_and_derivatives_: {{{*/
////
////std::tuple<std::vector<double>, std::vector<std::vector<double>>>
////compute_energies_and_derivatives_(torch::jit::script::Module& module,
////                                 std::vector<torch::jit::IValue>& inputs,
////                                 const vector<size_t>& indices,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool compute_gradients = true) {
////    module.train();  // Set module to training mode
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        std::cerr << "Output tensor is not of floating point type." << std::endl;
////        return {}; // Error handling
////    }
////
////    std::vector<double> output_vector(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    std::transform(output_vector.begin(), output_vector.end(), output_vector.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.numel();
////    }
////    size_t num_parameter_elements = 0; // Count number of elements in each parameter
////    for (const auto& param : module.parameters()) { num_parameter_elements += param.numel(); }
////    std::vector<std::vector<double>> all_gradients(num_parameter_elements, std::vector<double>(nstates, 0.0));
////
////    if (!compute_gradients) { return {output_vector, all_gradients}; }
////
////
////
////    // Create a set for quick lookup to see if gradients are needed for a parameter
////    std::unordered_set<size_t> gradient_indices(indices.begin(), indices.end());
////
////
////    // Release Python GIL once before the loop
////    PyThreadState* thread_state = PyEval_SaveThread();
////    ////////////////////////////////////////////////////////////////////////////
////    ////////////////////////////////////////////////////////////////////////////
////
////
////
////    size_t state_index = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        for (int k = 0; k < tensor.numel(); ++k) {
////            auto single_input = torch::jit::IValue(tensor[k].detach().requires_grad_(true));
////            auto output = module.forward({single_input}).toTensor();
////            output.backward(torch::ones_like(output));
////
////            size_t param_index = 0;
////            for (const auto& param : module.parameters()) {
////                if (param.grad().defined() && gradient_indices.count(param_index)) {
////                    auto grad_data = param.grad().view(-1);
////                    for (size_t i = 0; i < grad_data.size(0); ++i, ++param_index) {
////                        all_gradients[param_index][state_index] = grad_data[i].item<double>();
////                    }
////                    param.grad().zero_();
////                } else {
////                    param_index += param.numel(); // Increment param_index by the number of elements in the parameter
////                }
////            }
////            ++state_index;
////        }
////    }
////
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
////    // Apply scaling to gradients
////    for (auto& gradients : all_gradients) {
////        std::transform(gradients.begin(), gradients.end(), gradients.begin(), [scaling](double v) { return v * scaling; });
////    }
////
////    return {output_vector, all_gradients};
////}
////
/////*}}}*/
////
//
//
//
//
//
//
//
//
/////* compute_energies_and_derivatives: {{{*/
////
////std::tuple<std::vector<double>, std::vector<std::vector<double>>>
////compute_energies_and_derivatives(torch::jit::script::Module& module,
////                                 std::vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool compute_gradients = true) {
////    module.train();  // Set module to training mode
////
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        std::cerr << "Output tensor is not of floating point type." << std::endl;
////        return {}; // Error handling
////    }
////
////    std::vector<double> output_vector(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////    std::transform(output_vector.begin(), output_vector.end(), output_vector.begin(), [lambda](double v) { return v * lambda; });
////
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) {
////        num_parameter_elements += param.numel();  // Count number of elements in each parameter
////    }
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.numel();
////    }
////
////    std::vector<std::vector<double>> all_gradients(num_parameter_elements, std::vector<double>(nstates, 0.0));
////
////    if (!compute_gradients) {
////        return {output_vector, all_gradients};
////    }
////
////    PyThreadState* thread_state = PyEval_SaveThread(); // Release Python GIL once before the loop
////    outputs.backward(torch::ones_like(outputs)); // Compute gradients for all outputs at once if possible
////
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        size_t s = 0;
////        for (int k=0; k < tensor.numel(); ++k) {
////            auto output = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
////            output.backward(torch::ones_like(output));
////
////            size_t param_index = 0;
////            size_t global_index = 0;
////            for (const auto& param : module.parameters()) {
////                if (param.grad().defined()) {
////                    auto grad_data = param.grad().view(-1);
////                    for (int i = 0; i < grad_data.size(0); ++i) {
////                        all_gradients[global_index][s] = grad_data[i].item<double>();
////                        global_index++;
////                    }
////                    param.grad().zero_();
////                }
////                param_index++;
////            }
////            s++;
////        }
////    }
////    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop
////
////    for (auto& gradients : all_gradients) {
////        std::transform(gradients.begin(), gradients.end(), gradients.begin(), [scaling](double v) { return v * scaling; });
////    }
////
////    return {output_vector, all_gradients};
////}
////
////
/////*}}}*/
////
//
//
//
///* NOTE: This function works, but is slow */
//
/////* compute_energies_and_derivatives: {{{*/
////
////std::tuple<std::vector<double>, std::vector<std::vector<double>>>
////compute_energies_and_derivatives(torch::jit::script::Module& module,
////                                 std::vector<torch::jit::IValue>& inputs,
////                                 double scaling = 1.0, double lambda = 1.0,
////                                 bool compute_gradients = true) {
////    module.train();  // Set module to training mode
////
////    // Forward pass for all inputs at once
////    auto outputs = module.forward(inputs).toTensor();
////    if (!outputs.is_floating_point()) {
////        std::cerr << "Output tensor is not of floating point type." << std::endl;
////        return {}; // Error handling
////    }
////
////    // Convert outputs to a vector of doubles
////    std::vector<double> output_vector(outputs.data_ptr<double>(), outputs.data_ptr<double>() + outputs.numel());
////
////    // Apply lambda scaling
////    std::transform(output_vector.begin(), output_vector.end(), output_vector.begin(), [lambda](double v) { return v * lambda; });
////
////    // Prepare gradients data structure
////    size_t num_parameter_elements = 0;
////    for (const auto& param : module.parameters()) {
////        num_parameter_elements += param.numel();  // Count number of elements in each parameter
////    }
////
////    size_t nstates = 0;
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        nstates += tensor.numel();
////    }
////
////    std::vector<std::vector<double>> all_gradients(num_parameter_elements, std::vector<double>(nstates, 0.0));
////
////    if (!compute_gradients) {
////        // Return zero-initialized gradients if not computing them
////        return {output_vector, all_gradients};
////    }
////
////    for (auto& state : inputs) {
////        torch::Tensor tensor = state.toTensor();
////        size_t s = 0;
////        for (int k=0; k < tensor.numel(); ++k) {
////            PyThreadState* thread_state = PyEval_SaveThread();
////            auto output = module.forward({torch::jit::IValue(tensor[k].detach().requires_grad_(true))}).toTensor();
////            output.backward(torch::ones_like(output));
////            PyEval_RestoreThread(thread_state);
////
////            size_t param_index = 0;
////            size_t global_index = 0;
////            for (const auto& param : module.parameters()) {
////                if (param.grad().defined()) {
////                    auto grad_data = param.grad().view(-1);
////                    for (int i = 0; i < grad_data.size(0); ++i) {
////                        all_gradients[global_index][s] = grad_data[i].item<double>();
////                        global_index++;
////                    }
////                    param.grad().zero_();
////                }
////                param_index++;
////            }
////            s++;
////        }
////    }
////
////    // Apply scaling to gradients
////    for (auto& gradients : all_gradients) {
////        std::transform(gradients.begin(), gradients.end(), gradients.begin(), [scaling](double v) { return v * scaling; });
////    }
////
////    return {output_vector, all_gradients};
////}
////
/////*}}}*/
////
//
///*}}}*/
//


/*}}}*/


// Functions to evaluate convergence / stopping criteria:{{{
double calculateMaxWeightChange(const std::vector<double>& old_weights, const std::vector<double>& new_weights) {
    double max_change = 0.0;
    for (size_t i = 0; i < old_weights.size(); ++i) {
        double change = std::abs(new_weights[i] - old_weights[i]);
        if (change > max_change) {
            max_change = change;
        }
    }
    return max_change;
}
double calculateAverageAbsDerivative(const std::vector<double>& derivatives) {
// Function to calculate the average absolute derivative
    double sum = 0.0;
    for (const auto& derivative : derivatives) {
        sum += std::abs(derivative);
    }
    return sum / derivatives.size();
}

double calculateGradientNorm(const std::vector<double>& derivatives) {
// Function to calculate the norm of the gradient
    double sum = 0.0;
    for (const auto& derivative : derivatives) {
        sum += derivative * derivative;
    }
    return std::sqrt(sum);
}


double calculateCumulativeMovingAverage(const std::vector<double>& losses) {
    if (losses.empty()) {
        throw std::invalid_argument("The losses vector is empty.");
    }

    double cumulative_sum = 0.0;
    for (size_t i = 0; i < losses.size(); ++i) {
        cumulative_sum += losses[i];
    }

    return cumulative_sum / static_cast<double>(losses.size());
}
/*}}}*/


struct PriorLosses {
    vector<double> stationary_distribution;
    vector<double> prior_energies_vector;
    vector<vector<double>> grad_prior_energies_vector;
    vector<vector<vector<double>>> forward_model;
    double L_DB;
    vector<double> grad_L_DB;
    double L_PC;
    vector<double> grad_L_PC;
    double L_ll;
    vector<double> grad_L_ll;
};


namespace PS {

// Data Structures:{{{

struct GFE
{
    // Get Free Energy
    vector<vector<vector<double>>> u_kln;         // (K, K, nsnaps)
    vector<vector<vector<double>>> states_kn;     // (K, nsnaps, nreplicas)
    vector<vector<int>> Nr_array;                 // (K, nsnaps)
    vector<vector<vector<double>>> diff_u_kln;    // (K, K, nsnaps)
    vector<vector<vector<double>>> diff2_u_kln;   // (K, K, nsnaps)
};

struct SEP
{
    // Seperated parameters and indices for each restraint
    vector<vector<int>> sep_indices;
    vector<vector<double>> sep_parameters;
};

struct GRI
{
    // get restraint information
    vector<string> data_uncertainty;
    vector<string> models;
    vector<double> Ndofs;
    vector<double> reference_potentials;
};

struct HRE
{
    // Hamiltonian replica exchange
    vector<double> energies;
    vector<vector<int>> lam_indices;
    vector<vector<int>> lam_states;
};

struct SCALE_OFFSET
{
vector<double> scales_f_exp;
vector<double> scales_f;
vector<double> offsets;
};

struct ENTROPY_CHI2
{
vector<double> S;
vector<double> dS;
vector<double> Chi2;
vector<double> dChi2;
};

struct FMO
{
    size_t batch_size;
    double fmp_accepted;
    double fmp_attempted;
    double fmp_acceptance;
    double fmp_scale_factor;

    vector<vector<vector<double>>> new_forward_model;
    vector<vector<double>> fwd_model_parameters; // (nrestraint, nparameters)
//    vector<vector<string>> fmp_prior_models; // (nrestraint, nparameters)
//    vector<vector<double>> fmp_prior_sigmas; // (nrestraint, nparameters)
//    vector<vector<double>> fmp_prior_mus; // (nrestraint, nparameters)

    vector<vector<vector<double>>> phi;                  // (nrestraint, nstates, nangles)
    vector<double> phi0;                                 // (nshifts)
    double decay_rate;
    vector<vector<double>> velocity;                     // (nrestraint, nparameters)
    double base_epsilon;
    double epsilon; // velocity scaling parameter
    double eta;     // noise scaling parameter
    double initial_eta_scale;
    double min_eta_scale;
    double max_eta_scale;

    vector<vector<double>> min_max_parameters;          //(nparameters, vector<double>(2, 0.0))
    double replica_scale_factor;
    double target_acceptance;
    double acceptance_adjustment_factor;
    vector<double> max_gradients;
    vector<int> restraint_indices;

    vector<vector<double>> du_dtheta; //(fwd_model_parameters.size());
    vector<vector<vector<vector<double>>>> fwd_model_derivatives;//(fwd_model_parameters.size());

    double beta1;
    double beta2;
    double alpha;
    double epsilon_adam;
    int t;
    vector<double> first_moment;
    vector<double> second_moment;

    double max_temperature;
    double temperature;
    double cooling_rate;


};


struct MO
{
    double rho;
    double reg_rho_accepted;
    double reg_rho_attempted;
    double reg_rho_acceptance;
    double reg_rho_eta;


    double fm_eta_prior;     // noise scaling parameter
    double fm_initial_eta_prior_scale;
    double fm_min_eta_prior_scale;
    double fm_max_eta_prior_scale;
    double fm_prior_accepted;
    double fm_prior_attempted;
    double fm_prior_acceptance;
    vector<vector<double>> fm_parameters;
    vector<vector<string>> fm_prior_models;
    vector<vector<double>> fm_prior_sigmas;
    vector<vector<double>> fm_prior_mus;
    vector<vector<double>> fm_prior_devs;

    double pm_eta_prior;     // noise scaling parameter
    double pm_initial_eta_prior_scale;
    double pm_min_eta_prior_scale;
    double pm_max_eta_prior_scale;
    double pm_prior_accepted;
    double pm_prior_attempted;
    double pm_prior_acceptance;
    vector<double> pm_parameters;
    vector<string> pm_prior_models;
    vector<double> pm_prior_sigmas;
    vector<double> pm_prior_mus;
    vector<double> pm_prior_devs;

    double pm_extern_loss;
    vector<double> pm_extern_dloss;
    vector<double> pm_extern_d2loss;
    vector<double> pm_extern_dloss_sigma;

    double pm_eta_extern_dloss_sigma;
    double pm_min_eta_extern_dloss_sigma_scale;
    double pm_max_eta_extern_dloss_sigma_scale;
    double pm_extern_dloss_sigma_acceptance;
    double pm_extern_dloss_sigma_accepted;
    double pm_extern_dloss_sigma_attempted;

    vector<double> pm_detailed_balance_loss;
    vector<double> pm_detailed_balance_dloss;
    vector<double> pm_detailed_balance_loss_sigma;
    double pm_eta_DB_sigma;
    double pm_min_eta_DB_sigma_scale;
    double pm_max_eta_DB_sigma_scale;
    double pm_DB_sigma_acceptance;
    double pm_DB_sigma_accepted;
    double pm_DB_sigma_attempted;


    vector<double> pm_prob_conservation_loss;
    vector<double> pm_prob_conservation_dloss;
    vector<double> pm_prob_conservation_loss_sigma;
    double pm_eta_PC_sigma;
    double pm_min_eta_PC_sigma_scale;
    double pm_max_eta_PC_sigma_scale;
    double pm_PC_sigma_acceptance;
    double pm_PC_sigma_accepted;
    double pm_PC_sigma_attempted;



    PriorLosses prior_losses;

    double pm_eta_lambda;
    double pm_min_eta_lambda_scale;
    double pm_max_eta_lambda_scale;
    double pm_lambda_acceptance;
    double pm_lambda_accepted;
    double pm_lambda_attempted;

    double pm_eta_xi;
    double pm_min_eta_xi_scale;
    double pm_max_eta_xi_scale;
    double pm_xi_acceptance;
    double pm_xi_accepted;
    double pm_xi_attempted;

    string data_loss;

    vector<vector<double>> diff_state_energies;


};

struct SS
{
    // Sum of Squares
    vector<vector<double>> sqerr;
    vector<vector<double>> devs;
    vector<vector<double>> d;
    vector<vector<double>> dfX;
    vector<vector<double>> d2fX;
    vector<vector<double>> sqerrB;
    vector<vector<double>> sqerrSEM;
    vector<vector<double>> sem;
    vector<vector<double>> fX;
    vector<double> scale;
    MO mo;
};

// define a struct to hold the trajectory data
struct TrajectoryData {
    int l;
    int step;
    double energy;
    vector<double> expanded_values;
    bool accept;
    vector<int> states;
    vector<int> indices;
    vector<double> parameters;
    //vector<vector<double>> sep_parameters;
    double acceptance;
    SS ss;
    SEP sep;
};


struct PyTrajData {
    PyObject* traj;
    PyObject* trajectory;
//    PyObject* model_optimization;
    PyObject* traces;
    PyObject* sampled_parameters;
    PyObject* sem_trace;
    PyObject* sse_trace;
    int nreplicas;
    PyObject* indices_list;
    PyObject* parameters_list;
    PyObject* para;
    PyObject* sem_list;
    PyObject* sse_list;
    PyObject* _sem_list;
    PyObject* _sse_list;
    PyObject* sep_para_list;
    PyObject* sep_ind_list;
    PyObject* rest_ind_list;
    PyObject* rest_para_list;
    PyObject* state_list;
    PyObject* traj_list;

};


struct ModelOptConvergenceMetrics {
    double maxWeightChange;
    double averageAbsDerivative;
    double gradientNorm;
//    vector<double> gradientNormPerLayer;
    double movingAverageLoss;
    double externAverageLoss;
    double externGradientNorm;

    double fmpAcceptance;
    double pmpAcceptance;

    double fmpPriorAcceptance;
    double pmpPriorAcceptance;

    vector<double> weighted_frames;
    double extern_loss_sigma;
    double detailed_balance_sigma;
    double prob_conservation_sigma;
    vector<double> stationary_dist;

};


/*}}}*/

// Get seperated indices and parameters:{{{
SEP get_sep_indices_and_parameters(const vector<int>& indices,
        const vector<double>& parameters, const vector<int>& restraint_index) {
   /* Takes a vector<int> of parameter indices and seperates the indices into
     * their corresponding data restraints with sep_indices being
     * type: vector<vector<int>>. Does the same for parameters.
     *
     */


    if (indices.size() != parameters.size() || indices.size() != restraint_index.size()) {
        std::ostringstream msg;
        msg << "Index is out of range when separating indices and parameters based on restraint_index.\n";
        msg << "( indices.size(), parameters.size(), restraint_index.size() ) = ( ";
        msg << indices.size() << ", " << parameters.size() << ", " << restraint_index.size() << " )";
        throw std::out_of_range(msg.str());
    }

    SEP sep;
    sep.sep_indices.reserve(indices.size());
    sep.sep_parameters.reserve(parameters.size());

    vector<int> v_ind{indices[0]};
    vector<double> v_para{parameters[0]};

    sep.sep_indices.push_back(v_ind);
    sep.sep_parameters.push_back(v_para);

    for (size_t n = 1; n < indices.size(); ++n) {
        size_t m = n - 1;
        if (restraint_index[n] == restraint_index[m]) {
            sep.sep_indices[restraint_index[m]].push_back(indices[n]);
            sep.sep_parameters[restraint_index[m]].push_back(parameters[n]);
        }
        else {
            v_ind.clear();
            v_para.clear();
            v_ind.push_back(indices[n]);
            sep.sep_indices.push_back(v_ind);
            v_para.push_back(parameters[n]);
            sep.sep_parameters.push_back(v_para);
        }
    }
    return sep;
}
/*}}}*/

// Get ftilde:{{{
vector<vector<double>> get_ftilde(size_t nstates, size_t nsamples,
        const vector<vector<vector<double>>>& model) {
    /* ftilde is the forward model average over a very large number of replicas
     *
     */

    if(model.empty() || model[0].empty()) {
        throw std::out_of_range("Index is out of range");
    }
    vector<size_t> ftilde_states = rand_vec_of_ints(nstates, nsamples);
    size_t n_rest = model[0].size();
    vector<vector<double>> ftilde;
    for (size_t k = 0; k < n_rest; ++k) {
        if(model[0][k].empty()) {
            throw std::out_of_range("Index is out of range");
        }
        size_t Nd = model[0][k].size();
        vector<double> _ftilde;
        for (size_t i = 0; i < Nd; ++i) {
            double fX = 0.0;
            for (auto s: ftilde_states) {
                if(s >= model.size() || k >= model[s].size() || i >= model[s][k].size()) {
                    throw std::out_of_range("Index is out of range");
                }
                fX += model[s][k][i];
            }
            fX /= ftilde_states.size();
            _ftilde.push_back(fX);
        }
        ftilde.push_back(_ftilde);
    }
    return ftilde;
}
/*}}}*/

// get_state_energies:{{{
const vector<double> get_state_energies(PyObject* ensemble, string attr) {
    /* Extract the energies for each state using the ensemble PyObject*.
     *
     */

    vector<double> state_energies;
    size_t nstates = PyLong_AsLong(PyLong_FromSsize_t(PySequence_Length(ensemble)));
    for (size_t i = 0; i < nstates; ++i) {
        PyObject* s = PySequence_GetItem(ensemble, i);
        if (s == NULL) {
            PyErr_Format(PyExc_IndexError, "Index %zu is out of range for `ensemble`", i);
        }
        PyObject* energy_obj = PyObject_GetAttrString(PySequence_GetItem(s, 0), attr.c_str());
        if (energy_obj == NULL) {
            PyErr_Format(PyExc_AttributeError, "ensemble Object does not have the attribute %U", PyUnicode_FromString(attr.c_str()));
        }
        state_energies.push_back(PyFloat_AS_DOUBLE(energy_obj));
        Py_DECREF(energy_obj);
        Py_DECREF(s);
    }
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "error in get_state_energies()");
    }
    return state_energies;
}
/*}}}*/

// get_restraint_information:{{{
GRI get_restraint_information(PyObject* ensemble) {
    /* Use the ensemble PyObject* to extract important data restraint information
     * such as the number of degrees of freedom,
     * the data uncertainty ("single" sigma or "multiple" sigma), and
     * the statistical model/likelihood model.
     */

    GRI gri;
    PyObject* s = PySequence_GetItem(ensemble, 0);
    if (s == NULL) {
        PyErr_SetString(PyExc_IndexError, "Index is out of range for `ensemble`");
    }
    for (size_t i=0; i<static_cast<size_t>(Py_SIZE(s)); ++i) {  // iterate over restraints
        PyObject* R = PySequence_GetItem(s, i);
        if (R == NULL) {
            PyErr_Format(PyExc_IndexError, "Index %zu is out of range for `ensemble` state object", i);
        }
        gri.Ndofs.push_back(PyFloat_AS_DOUBLE(PyObject_GetAttrString(R, "Ndof")));
        gri.data_uncertainty.push_back(PyUnicode_AsUTF8(PyUnicode_FromObject(PyObject_GetAttrString(R, "data_uncertainty"))));
        gri.models.push_back(PyUnicode_AsUTF8(PyUnicode_FromObject(PyObject_GetAttrString(R, "stat_model"))));
    }
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "error in get_restraint_information()");
    }
    return gri;
}
/*}}}*/

// get_data_likelihood:{{{
const vector<string> get_data_likelihood(PyObject* ensemble) {
    /* Use the ensemble PyObject* to extract information about the data likelihood.
     * gaussian or lognormal
     * TODO: FIXME: this and `get_restraint_information` could be combined in some way
     */

    vector<string> data_likelihood;
    PyObject* s = PySequence_GetItem(ensemble, 0);
    if (s == NULL) {
        PyErr_SetString(PyExc_IndexError, "Index is out of range for `ensemble`");
    }
    for (size_t i=0; i<static_cast<size_t>(Py_SIZE(s)); ++i) {  // iterate over restraints
        PyObject* R = PySequence_GetItem(s, i);
        if (R == NULL) {
            PyErr_Format(PyExc_IndexError, "Index %zu is out of range for `ensemble` state object", i);
        }
        string likelihood_type = PyUnicode_AsUTF8(PyObject_GetAttrString(R, "data_likelihood"));
        data_likelihood.push_back(transformStringToLower(likelihood_type));
    }
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "error in get_data_likelihood()");
    }
    return data_likelihood;
}
/*}}}*/

// get_scaling_parameters:{{{
SCALE_OFFSET get_scaling_parameters(PyObject* ensemble) {
    /* TODO: Scaling parameters should really be a python dictionary inside the
     * child Restraint classes and converted to C++ data structures here...
     */

    SCALE_OFFSET so;
    PyObject* s = PySequence_GetItem(ensemble, 0);
    if (s == NULL) {
        PyErr_SetString(PyExc_IndexError, "Index is out of range for `ensemble`");
    }
    for (size_t i=0; i<static_cast<size_t>(Py_SIZE(s)); ++i) {  // iterate over restraints
        PyObject* R = PySequence_GetItem(s, i);
        if (R == NULL) {
            PyErr_Format(PyExc_IndexError, "Index %zu is out of range for `ensemble` state object", i);
        }
        double scale_f_exp = PyFloat_AsDouble(PyObject_GetAttrString(R, "scale_f_exp"));
        double scale_f = PyFloat_AsDouble(PyObject_GetAttrString(R, "scale_f"));
        double offset = PyFloat_AsDouble(PyObject_GetAttrString(R, "offset"));
        so.scales_f_exp.push_back(scale_f_exp);
        so.scales_f.push_back(scale_f);
        so.offsets.push_back(offset);
    }
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "error in get_scaling_parameters()");
    }
    return so;
}
/*}}}*/

// change_scaling_parameters:{{{
SCALE_OFFSET change_scaling_parameters(vector<double> &scales,
        vector<double> &offsets, vector<vector<double>> &sep_parameters) {
    /*
     *
     */

    SCALE_OFFSET so;
    for (size_t i=0; i<scales.size(); ++i) {
      if (sep_parameters[i].size() > 1) { so.scales_f_exp.push_back(sep_parameters[i][3]); }
      else { so.scales_f_exp.push_back(1.0); }
      if (sep_parameters[i].size() > 2) { so.scales_f.push_back(sep_parameters[i][4]); }
      else { so.scales_f.push_back(1.0); }
      if (sep_parameters[i].size() > 3) { so.offsets.push_back(sep_parameters[i][5]); }
      else { so.offsets.push_back(0.0); }
      //if (sep_parameters[i].size() > 4) { so.offsets.push_back(sep_parameters[i][4]); }
      //else { so.offsets.push_back(1.0); }
    }
    return so;
}

/*}}}*/

// sum_reference_potentials:{{{
vector<double> sum_reference_potentials(const vector<vector<double>> ref_potentials,
                                        const vector<int> states) {
    /*
     *
     */

    vector<double> reference_potentials(ref_potentials[0].size(), 0.0);
    for (auto state: states) {
      if ( isIndexValid(ref_potentials, static_cast<size_t>(state)) ) {
        for (size_t i=0; i<ref_potentials[state].size(); ++i) {
            reference_potentials[i] += ref_potentials[state][i];
        }
      }
      else {
        throw std::out_of_range("`state` Index is out of range for ref_potentials.");
      }
    }
    return reference_potentials;
}

/*}}}*/

// SEM and SSE:{{{
SS get_sse_and_sem(const vector<vector<vector<double>>> &model,
        const vector<vector<vector<double>>> &diff_fwd_model,
        const vector<vector<vector<double>>> &diff2_fwd_model,
        const vector<vector<vector<double>>> &experiments,
        const vector<vector<vector<double>>> &weights,
        const vector<int> &states, const vector<double>& scales_exp,
        const vector<double>& scales, const vector<double>& offsets,
        const vector<vector<double>>& parameters,
        const vector<string> &data_likelihood, string sem_method,
        const MO &mo = MO() ) {
        //const vector<int> &fwd_model_restraint_indices={}) {
    /* Calculated the sum of squared errors  (SSE) and the standard error of the mean
     * sigma (SEM).
     *
     */


    vector<double> pi;
    if ( mo.data_loss == "sse" ) {
        pi = mo.prior_losses.stationary_distribution;
    }


    SS ss;
    const size_t n_rest = model[0].size();
    const int nreplicas = states.size();
    //ss.scale.resize(n_rest, 1.00);

    // Pre-allocate and initialize SS vectors
    for (size_t k = 0; k < n_rest; ++k) {
        size_t Nd = model[0][k].size();
        ss.sqerr.emplace_back(Nd, 0.0);
        ss.devs.emplace_back(Nd, 0.0);
        ss.d.emplace_back(Nd, 0.0);
        ss.fX.emplace_back(Nd, 0.0);
        ss.dfX.emplace_back(Nd, 0.0);
        ss.d2fX.emplace_back(Nd, 0.0);
        ss.sem.emplace_back(Nd, 0.0);
        ss.sqerrSEM.emplace_back(Nd, 0.0);
    }

    for (size_t k = 0; k < n_rest; ++k) {
        const double scalar = parameters[k][4]; // alpha parameter
        size_t Nd = model[0][k].size();
        double offset = offsets[k];
        for (size_t j = 0; j < Nd; ++j) {
            double f_exp = experiments[0][k][j];
            double weight = weights[0][k][j];

            double devSEM = 0.0, meanDevSEM = 0.0;
            double sum_fX = 0.0, sum_dfX = 0.0, sum_d2fX = 0.0;

            if ( mo.data_loss == "sse" ) {
                for (size_t i = 0; i < pi.size(); ++i) {
                    ss.fX[k][j] += pi[i]*model[i][k][j];
                }


                  /* TODO: do/don't Account for the SEM sigma?
                   * Answer: just set nreplica = 1 and you'll be okay... */
            }
            else {
                // Compute sums for current data point across all states
                for (auto s : states) {
                    sum_fX += model[s][k][j];
                    sum_dfX += diff_fwd_model[s][k][j];
                    sum_d2fX += diff2_fwd_model[s][k][j];
                }
                // Compute averages
                ss.fX[k][j] = sum_fX / nreplicas;
                ss.dfX[k][j] = sum_dfX / nreplicas;
                ss.d2fX[k][j] = sum_d2fX / nreplicas;

            }

            if (nreplicas > 1) {
              if ( sem_method == "sem" ) {
                  /* NOTE: Standard deviation of finite sample mean */
                  //for (auto s: states) {
                  //    devSEM = (model[s][k][j] - ss.fX[k][j]);
                  //    ss.sqerrSEM[k][j] += devSEM * devSEM;
                  //}
                  //ss.sem[k][j] = parameters[k][6]*sqrt(ss.sqerrSEM[k][j] / nreplicas);

                  /* NOTE: Average replica deviation around finite sample mean.
                   * Checking bias from zero. Extra nuisance parameter to scale if needed (default = 1).
                   */
                  for (auto s: states) { devSEM += (model[s][k][j] - ss.fX[k][j]); }
                  meanDevSEM = devSEM / nreplicas;
                  //ss.sem[k][j] = weight * meanDevSEM;
                  ss.sem[k][j] = parameters[k][6] * weight * meanDevSEM;
              }
              else { throw std::invalid_argument("Unsupported method for calculating SEM."); }
            }

            if (data_likelihood[k] == "log normal") {
                ss.devs[k][j] = log(ss.fX[k][j] * scales[k] * scalar / (scales_exp[k] * f_exp));
            }
            else if (data_likelihood[k] == "gaussian") {
                ss.devs[k][j] = scales_exp[k] * f_exp + offset - scales[k] * ss.fX[k][j] * scalar;
            }
            else { throw std::invalid_argument("Unsupported data_likelihood. Use 'Gaussian' or 'Log Normal'."); }
            ss.d[k][j] = f_exp;
            ss.sqerr[k][j] = weight * ss.devs[k][j] * ss.devs[k][j];
        }
    }

    ss.mo = mo;

    if ( !ss.mo.fm_parameters.empty() ) {

        /* NOTE: TODO: Now, we should check to see if all of the init_fwd_model_parameters are all zero
         * if they are, then we should just pass this next code block
         */

        // Evaluate the deviations for the parameter prior and store inside data structure `ss`
        // parameter_prior_models has shape: ()
        for (size_t i = 0; i < ss.mo.fm_parameters.size(); ++i) {
            if ( !std::all_of(ss.mo.fm_prior_models[i].begin(), ss.mo.fm_prior_models[i].end(), [](const std::string& s) { return s == "uniform"; }) )
            {
                for (size_t j = 0; j < ss.mo.fm_parameters[i].size(); ++j) {
                    ss.mo.fm_prior_devs[i][j] = ss.mo.fm_parameters[i][j] - ss.mo.fm_prior_mus[i][j];
                }
            }
        }

    }

    if ( !ss.mo.pm_parameters.empty() ) {
        if ( !std::all_of(ss.mo.pm_prior_models.begin(), ss.mo.pm_prior_models.end(), [](const std::string& s) { return s == "uniform"; }) )
        {
            for (size_t i = 0; i < ss.mo.pm_parameters.size(); ++i) {
                ss.mo.pm_prior_devs[i] = ss.mo.pm_parameters[i] - ss.mo.pm_prior_mus[i];
            }
        }
    }

    return ss;
}
//:}}}

// Mixture SEM and SSE:{{{
SS get_mixture_sse_and_sem(shared_ptr<vector<vector<vector<vector<double>>>>> forward_model,
        const vector<vector<vector<double>>> &experiments,
        const vector<vector<vector<double>>> &weights,
        const vector<vector<int>> &states, const vector<double>& scales_exp, //TODO: scales needs to be a data structure to have multidimensional scaling parameters
        const vector<double>& scales, const vector<double>& offsets,
        const vector<vector<double>>& parameters,
        const vector<string> &data_likelihood, const vector<double> &fwd_model_weights)
{
    /* Calculated the sum of squared errors  (SSE) and the standard error of the mean
     * sigma (SEM).
     */

//    cout << "fwd_model_weights = " << fwd_model_weights << endl;
//    cout << endl;
//    for (size_t i=0; i<fwd_model_weights.size(); ++i) {
//        if (fwd_model_weights[i] == 0.0) { continue; }
//        cout << "states[i] = " << states[i] << endl;
//        cout << "forward_model->at(" << i << ")[0][0] = " << forward_model->at(i)[0][0] << endl;
//    }
//    cout << endl;

    SS ss;
    const size_t n_rest = forward_model->at(0).at(0).size();
    const int nreplicas = states[0].size();
    ss.scale = vector<double>(n_rest, 1.00);
    // Initialize the data structure
    for (size_t k=0; k<n_rest; ++k) {
      size_t Nd = forward_model->at(0).at(0).at(k).size();
      ss.sqerr.push_back(vector<double>(Nd, 0.0));
      ss.devs.push_back(vector<double>(Nd, 0.0));
      ss.d.push_back(vector<double>(Nd, 0.0));
      ss.fX.push_back(vector<double>(Nd, 0.0));
      ss.dfX.push_back(vector<double>(Nd, 0.0));
      ss.d2fX.push_back(vector<double>(Nd, 0.0));
      ss.sem.push_back(vector<double>(Nd, 0.0));
      ss.sqerrSEM.push_back(vector<double>(Nd, 0.0));
      ss.sqerrB.push_back(vector<double>(Nd, 0.0));
    }

    double devSEM, offset, f_exp, weight, _fX;
    const size_t n_fwd_models = static_cast<size_t>(forward_model->size());

//    cout << "n_fwd_models = " << n_fwd_models << endl;


    //#pragma omp parallel for shared(ss, parameters, forward_model, offsets, \
    //        fwd_model_weights) reduction(+:fX,_fX)

//    cout << n_fwd_models << ", " << n_rest << ", " << forward_model->at(0).at(0).at(0).size() << ", " << forward_model->at(0).at(0).at(1).size() << ", " << forward_model->at(0).at(0).at(2).size() << ", " <<endl;
//    exit(1);
    for (size_t k=0; k<n_rest; ++k) {
      const double scalar = parameters[k][4];  // alpha parameter
      size_t Nd = forward_model->at(0)[0][k].size();
      offset = offsets[k];
      //#pragma omp parallel for reduction(+:fX,_fX)
      for (size_t j=0; j<Nd; ++j) {
        devSEM = 0.0;
        for (size_t l=0; l<n_fwd_models; ++l) {
            if ( fwd_model_weights[l] == 0.0 ) { continue; }
            _fX = 0.0;
            // Average over forward model <f(X)> for a particular ensemble
            for (auto s: states[l]) { _fX += forward_model->at(l)[s][k][j]; }
            _fX /= nreplicas;
            // weighted average of forward model for each ensemble same for sigmaSEM
            ss.fX[k][j] += _fX*fwd_model_weights[l];
            // calcualte the sigmaSEM using the weigthed average fwd model
            for (auto s: states[l]) { devSEM += (forward_model->at(l)[s][k][j] - _fX)*fwd_model_weights[l]; }
        }

        f_exp = experiments[0][k][j];
        weight = weights[0][k][j];
        if (data_likelihood[k].compare("log normal") == 0)
        {
            ss.devs[k][j] = log(ss.fX[k][j]*scales[k]*scalar / (scales_exp[k]*f_exp));
        }
        else if (data_likelihood[k].compare("gaussian") == 0)
        {
            ss.devs[k][j] = scales_exp[k]*f_exp + offset - scales[k]*ss.fX[k][j]*scalar;
        }
        else
        {
            throw std::invalid_argument("Please choose a data_likelihood of the following:\n 'Gaussian', 'Log Normal'");
        }
        ss.d[k][j] = f_exp;
        ss.sqerr[k][j] = weight*ss.devs[k][j]*ss.devs[k][j];
        ss.sqerrSEM[k][j] = weight*devSEM*devSEM;
        if ( nreplicas > 1 ) { ss.sem[k][j] = sqrt(ss.sqerrSEM[k][j]/nreplicas); }
      }
    }
    return ss;
}
//:}}}

// Likelihood Models:{{{
/*
 * Note that Jefferys prior is automatically enforced since the domain for
 * sigma is log-spaced. However, we need a Jefferys prior for any additional
 * nuisnance parameters that are not log-spaced e.g., phi and beta.
 *
 *
 */

/* NOTE:
 * log(sqrt(x)) = log(x^{1/2}) = 0.5*log(x)
 * -log(1/sqrt(2.0*pi)) = 0.5*log(2*pi)
 * 0.5*log(0.5) === log(1/sqrt(2))
 */


double get_energy_Bayesian(vector<double> sqerr, vector<double> sigmaB, double Ndof) {
    /* The Bayesian model used with a single replica. */

    double result = 0.0;
    result += 0.5*Ndof*log(2.0*M_PI*sigmaB[0]*sigmaB[0]); // norm
    for (size_t i=0; i<sqerr.size(); ++i) {
        result += 0.5*sqerr[i] / (sigmaB[0]*sigmaB[0]);   // Regular sse (not just bayes)
    }
    return result;
}


double get_energy_Outliers(vector<double> sqerr, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Outlier's/Cauchy likelihood model that enforces a sigma parameter
     * for each data point.
     *
     * */

    double result = 0.0;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    for (size_t i=0; i<sqerr.size(); ++i) {
        const double a2 = 0.5*sqerr[i] + ss2;
        if ( sm2 > 0.0 ) { result += log(2.0*a2/(1.0 - exp(-a2/sm2))); }
        else { result += log(2.0*a2); }
        result += 0.5*log(0.5*M_PI*M_PI/ss2);              // Norm
    }
    return result;
}

double get_d_energy_Outliers(vector<double> devs, vector<double> dfX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Outlier's/Cauchy likelihood model that enforces a sigma parameter
     * for each data point.
     * */

    double result = 0.0;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    double ExpA;
    for (size_t i=0; i<devs.size(); ++i) {
        const double sqerr = devs[i]*devs[i];
        //cout << (2.0*ss2 + sqerr) << "   " << sm2 << "   " << dfX[i]*devs[i] << endl;
        ExpA = exp( (0.5*sqerr + ss2)/sm2 );
        if ( std::isinf(ExpA) ) { ExpA = 0.0; }
        const double A = (1.0 - ExpA);
        const double denom = ( sm2*A*(2.0*ss2 + sqerr) );
        result += dfX[i]*devs[i]*( -2.0*sm2*A - 2.0*ss2 - sqerr) / denom;
    }

    //cout << sm2 << endl;
    return result;
}
double get_d2_energy_Outliers(vector<double> devs, vector<double> dfX,
        vector<double> d2fX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Outlier's/Cauchy likelihood model that enforces a sigma parameter
     * for each data point.
     * */

    double result = 0.0;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    double ExpA;
    for (size_t i=0; i<devs.size(); ++i) {
        const double sqerr = devs[i]*devs[i];
        const double a2 = (2.0*ss2 + sqerr);
        ExpA = exp( (0.5*sqerr + ss2)/sm2 );
        if ( std::isinf(ExpA) ) { ExpA = 0.0; }
        const double A = (1.0 - ExpA);
        const double A2 = A*A;
        const double dfX2 = dfX[i]*dfX[i];
        const double denom = ( sm2*A*(2.0*ss2 + sqerr) );
        const double denom2 = denom*denom;
        const double B = ( 2.0*sm2*(-A) - 2.0*ss2 - sqerr);
        const double C = ( -2.0*sm2*(-A) + a2);
        const double num_1 = 2.0*sm2*A2*a2*sqerr*dfX2;
        const double num_2 = sm2*A*a2*devs[i]*B*d2fX[i];
        const double num_3 = sm2*A*a2*C*dfX2;
        const double num_4 = 2.0*sm2*A*sqerr*B*dfX2;
        const double num_5 = a2*sqerr*C*ExpA*dfX2;
        const double numerator = num_1 + num_2 + num_3 + num_4 + num_5;
        result += numerator / denom2;
    }
    return result;
}


double get_energy_Gaussian(vector<double> sqerr, vector<double> sigmaSEM,
        vector<double> sigmaB) {
    /* The Gaussian model.
     */

    double result = 0.0;
    for (size_t i=0; i<sqerr.size(); ++i) {
        const double sem2 = sigmaSEM[i]*sigmaSEM[i];
        const double s2 = sigmaB[0]*sigmaB[0] + sem2;
        result += 0.5*sqerr[i] / s2;
        result += 0.5*log(2.0*M_PI*s2);
    }
    return result;
}
double get_d_energy_Gaussian(vector<double> devs, vector<double> dfX, vector<double> sigmaSEM,
        vector<double> sigmaB) {
    /* The Gaussian model first derivative w/ respect to the forward model.
     */

    double result = 0.0;
    for (size_t i=0; i<devs.size(); ++i) {
        const double sem2 = sigmaSEM[i]*sigmaSEM[i];
        const double s2 = sigmaB[0]*sigmaB[0] + sem2;
        result += -dfX[i] * devs[i] / s2;
    }
    return result;
}
double get_d2_energy_Gaussian(vector<double> devs, vector<double> dfX,
        vector<double> d2fX, vector<double> sigmaSEM,
        vector<double> sigmaB) {
    /* The Gaussian model second derivative w/ respect to the forward model.
     */

    double result = 0.0;
    for (size_t i=0; i<devs.size(); ++i) {
        const double sem2 = sigmaSEM[i]*sigmaSEM[i];
        const double s2 = sigmaB[0]*sigmaB[0] + sem2;
        result += -d2fX[i]*devs[i] / s2;
        result += dfX[i]*dfX[i] / s2;
    }
    return result;
}

double get_energy_GB(vector<double> sqerr, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Good-Bad model negLogLikelihood.
     *
     */
    /* NOTE: this is integrated from sigmaSEM to infty */
    /* NOTE: this has beta integrated out.  */

    double result = 0.0;
    const double phi = sigmaB[2];
    const double phi2 = phi*phi;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    for (size_t i=0; i<sqerr.size(); ++i) {
        const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
        double Eb = exp(0.5*sqerr[i]/(phi2*ss2));
        double Ea = exp(0.5*sqerr[i]/ss2);
        double Ec = exp(-0.5*sqerr[i]*(phi2+1.0)/(phi2*ss2));
        // Handle the NaN case or return an appropriate value
        if (std::isnan(Eb)) { Eb = std::numeric_limits<double>::max(); }
        if (std::isnan(Ea)) { Ea = std::numeric_limits<double>::max(); }
        if (std::isnan(Ec)) { Ec = std::numeric_limits<double>::max(); }
        result += -log(sqrt(2.0)/(4.0*sqrt(M_PI*ss2)*phi) * ( -phi*(heaviside(sem - sqrt(ss2), 0.5) - 1.0)*Eb - (heaviside(-phi*sqrt(ss2)+sem, 0.5) - 1.0)*Ea)*Ec);
    }
    // Jeffrey's prior for phi parameter
    result += log(phi);
    if (std::isinf(result)) { return std::numeric_limits<double>::max(); }
    else{ return result; }
}

double get_d_energy_GB(vector<double> devs, vector<double> dfX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The GB model first derivative w/ respect to the forward model.
     */

    double result = 0.0;
    const double phi = sigmaB[2];
    const double phi2 = phi*phi;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    const double sigma = sqrt(ss2);
    for (size_t i=0; i<devs.size(); ++i) {
        const double sqerr = devs[i]*devs[i];
//// Handle the NaN case or return an appropriate value
//        if (std::isnan(Eb)) { Eb = std::numeric_limits<double>::max(); }
//        if (std::isnan(Ea)) { Ea = std::numeric_limits<double>::max(); }
        const double numerator_part1 = (1.0 - heaviside(sem - sigma, 0.5)) * std::exp(0.5*sqerr / (phi2*ss2));
        const double numerator_part2 = (1.0 - heaviside(-phi*sigma + sem, 0.5)) * std::exp(0.5*sqerr / (ss2));
        const double denominator_part1 = (heaviside(sem - sigma, 0.5) - 1.0) * std::exp(0.5*sqerr / (phi2*ss2));
        const double denominator_part2 = (heaviside(-phi*sigma + sem, 0.5) - 1.0) * std::exp(0.5*sqerr / (ss2));
        const double numerator = dfX[i] * devs[i] * (phi2*phi * numerator_part1 + numerator_part2);
        const double denominator = phi2 * ss2 * (phi * denominator_part1 + denominator_part2);
        result += numerator / denominator;
    }
    if (std::isinf(result)) { return std::numeric_limits<double>::max(); }
    else{ return result; }
}



double get_d2_energy_GB(vector<double> devs, vector<double> dfX,
        vector<double> d2fX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The GB model second derivative w/ respect to the forward model.
     */

    double result = 0.0;
    const double phi = sigmaB[2];
    const double phi2 = phi*phi;
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    const double sigma = sqrt(ss2);
    bool cond1 = (sem < sigma) && (phi * sigma - sem > 0);
    bool cond2 = (sem > sigma) && (phi * sigma - sem > 0);
    bool cond3 = (sem < sigma) && (phi * sigma - sem < 0);
    bool cond4 = (sem > sigma) && (phi * sigma - sem < 0);

    for (size_t i=0; i<devs.size(); ++i) {
        const double dfX2 = dfX[i]*dfX[i];
        const double sqerr = devs[i]*devs[i];
//        // Handle the NaN case or return an appropriate value
//        if (std::isnan(Eb)) { Eb = std::numeric_limits<double>::max(); }
//        if (std::isnan(Ea)) { Ea = std::numeric_limits<double>::max(); }

        if (cond1) {
            const double term1 = phi*phi2 * std::exp(sqerr / (2 * phi2 * ss2));
            const double term2 = std::exp(sqerr / (2 * ss2));
            const double denom = phi * ss2 * (phi * term1 + term2);
            result += -(devs[i]) * (term1 + term2) * d2fX[i] / phi2 * ss2 * (phi * term1 + term2)
                   - (devs[i]) * (-phi * (devs[i]) * term1 * dfX[i] / ss2 - (devs[i]) * term2 * dfX[i] / ss2) * dfX[i] / (phi2 * ss2 * (phi * term1 + term2))
                   - (devs[i]) * (term1 + term2) * ((devs[i]) * term2 * dfX[i] / ss2 + (devs[i]) * term1 * dfX[i] / (phi * sigma)) * dfX[i] / (denom*denom)
                   + (phi*phi2 * term1 + term2) * dfX2 / denom;
        } else if (cond2) {
            result += -(devs[i]) * d2fX[i] / (phi2 * ss2) + dfX2 / (phi2 * ss2);
        } else if (cond3) {
            result += -(devs[i]) * d2fX[i] / ss2 + dfX2 / ss2;
        } else if (cond4) {
            result += 0.0;
        } else {
            //std::cout << "Condition not met" << std::endl;
            result += 0.0;
        }
    }
    if (std::isinf(result)) { return std::numeric_limits<double>::max(); }
    else{ return result; }
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////





double get_energy_Students(vector<double> sqerr, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Student's model.  Beta version of the Cauchy
     * https://en.wikipedia.org/wiki/Student%27s_t-distribution
     * This version derives from integrating from sigmaSEM to infinity.
     *
     *
     * */

    double result = 0.0;
    const double beta = sigmaB[1];
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    const double sm2 = sem*sem;
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    for (size_t i=0; i<sqerr.size(); ++i) {
        const double ab2 = 0.5*sqerr[i] + beta*ss2;
        result += beta*log(1 + sqerr[i] / (2*beta*ss2));
        if ( sm2 > 0.0 ) {
            result += -log(tgamma_lower(beta, ab2/sm2));
        }
    }
    result += Ndof*log(tgamma(0.5*beta)/tgamma(0.5*(beta+1.)));
    result += 0.5*Ndof*log(2.*M_PI*beta*ss2);
    // Jeffrey's prior for beta parameter
    if ( int(sqerr.size()) < 4 ) { result += 2.*log(beta); }
    else { result += log(beta); }
    return result;
}

double get_d_energy_Students(vector<double> devs, vector<double> dfX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Students model first derivative w/ respect to the forward model.
     */

    double result = 0.0;
    const double beta = sigmaB[1];
    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    double sm2 = sem*sem;
    // It would be very unusual for SEM to be 0.0, but in the case it ever is,
    // let's make it very small.
    if ( sm2 == 0.0 ) { sm2 = 1e-15; }
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    for (size_t i=0; i<devs.size(); ++i) {
        const double sqerr = devs[i]*devs[i];
        const double a2 = (2.0*beta*ss2 + sqerr);
        const double a2b = pow(2.0*beta*ss2*a2, beta);
        const double a22b = pow(2.0*beta*ss2*pow(a2, 2), beta);
        const double x = (0.5*sqerr + beta*ss2)/sm2;
        const double lower_gamma = tgamma_lower(beta, x);
        const double denom = sm2*lower_gamma;
        const double B = (-2.0*beta*sm2*a2b*lower_gamma + pow(2.0*sm2, 1.0-beta)*a22b*exp(-x));
        const double numerator = dfX[i]*pow((1.0/(2.0*beta*ss2)), beta)*devs[i]*pow(a2, -beta-1.0)*B;
        result += numerator / denom;
    }
    //cout << result << endl;
    return result;
}

double get_d2_energy_Students(vector<double> devs, vector<double> dfX,
        vector<double> d2fX, vector<double> sigmaSEM,
        vector<double> sigmaB, double Ndof) {
    /* The Students model second derivative w/ respect to the forward model.
     * This becomes very difficult to compute with beta greater than 10, however
     * the second derivative should just be a constant value at beta > 10.
     */

    double result = 0.0;
    const double beta = sigmaB[1];
    const double beta2 = beta*beta;

    const double sem = *std::max_element(sigmaSEM.begin(), sigmaSEM.end());
    double sm2 = sem*sem;
    // It would be very unusual for SEM to be 0.0, but in the case it ever is,
    // let's make it very small.
    if ( sm2 == 0.0 ) { sm2 = 1e-15; }
    const double ss2 = sigmaB[0]*sigmaB[0] + sm2;
    for (size_t i=0; i<devs.size(); ++i) {
        const double dfX2 = dfX[i]*dfX[i];
        const double sqerr = devs[i]*devs[i];
        const double part1 = pow(1/(2*beta*ss2), beta);
        const double common_expr = 2*beta*ss2 + sqerr;
        const double common_term = pow(common_expr, -beta - 2);
        const double lgamma_val = tgamma_lower(beta, common_expr/(2*sm2));

        result += part1 * (-2*sm2*(-beta - 1)*sqerr*common_expr*common_term *
                    (-2*beta*sm2*pow(2*beta*ss2*common_expr,beta) *
                     exp(common_expr/(2*sm2))*lgamma_val +
                     pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr, 2)*beta) *
                    exp(common_expr/sm2)*lgamma_val*dfX2 +
                    sm2*(devs[i])*pow(common_expr, -beta - 1)*
                    (-2*beta*sm2*pow(2*beta*ss2*common_expr,beta) *
                     exp(common_expr/(2*sm2))*lgamma_val +
                     pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr,2)*beta) *
                    exp(common_expr/sm2)*lgamma_val*d2fX[i] +
                    sm2*common_expr*pow(common_expr, -beta - 1) *
                    (2*beta*sm2*pow(2*beta*ss2*common_expr,beta) *
                     exp(common_expr/(2*sm2))*lgamma_val -
                     pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr, 2)*beta) *
                    exp(common_expr/sm2)*lgamma_val*dfX2 +
                    pow(common_expr/(2*sm2), beta - 1)*sqerr*common_expr *
                    pow(common_expr, -beta - 1) *
                    (-2*beta*sm2*pow(2*beta*ss2*common_expr,beta) *
                     exp(common_expr/(2*sm2))*lgamma_val +
                     pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr, 2)*beta) *
                    exp(common_expr/(2*sm2))*dfX2 +
                    sqerr*pow(common_expr, -beta - 1) *
                    (4*beta2*sm2*sm2*pow(2*beta*ss2*common_expr,beta) *
                     exp(common_expr/(2*sm2))*lgamma_val -
                     4*beta*sm2*pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr,2)*beta +
                     2*beta*sm2*(pow(common_expr/(2*sm2), beta - 1) *
                                    pow(2*beta*ss2*common_expr,beta)*common_expr +
                                    pow(2*sm2, 1 - beta)*pow(2*beta*ss2*common_expr, 2)*beta*common_expr)) *
                    exp(common_expr/sm2)*lgamma_val*dfX2) *
                    exp(-3*common_expr/(2*sm2))/(sm2*sm2*common_expr*pow(lgamma_val,2));


    }
    //cout << result << endl;
    return result;
}




/*}}}*/

// Get data restraint energy:{{{
double get_data_restraint_energy(const int &nreplicas, const string &model, const vector<double> &sqerr,
        const vector<double> &sigmaSEM, const vector<double> &sigmaB,
        const double &Ndof, const vector<double> &sseB, const string &data_uncertainty) {
    /* Uses the model == stat_model to call on the correct energy function.
     *
     */

    //sigmaSEM.size() == N data points for single sigma
    //sigmaSEM.size() == 1 for multiple sigma
    double energy = 0.0;
    if ( model == "Bayesian" ) {
        energy = get_energy_Bayesian(sqerr, sigmaB, Ndof);
    }
    else if ( model == "Gaussian"  ) {
        energy = get_energy_Gaussian(sqerr, sigmaSEM, sigmaB);
    }
    else if ( model == "GB"  ) {
        energy = get_energy_GB(sqerr, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Outliers" ) {
        energy = get_energy_Outliers(sqerr, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Students" ) {
        energy = get_energy_Students(sqerr, sigmaSEM, sigmaB, Ndof);
    }
    else {
        /* NOTE: throw exception links:
         https://en.cppreference.com/w/cpp/language/throw
         https://en.cppreference.com/w/cpp/error/exception
         http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
         https://stackoverflow.com/a/13186007
        */
        throw std::invalid_argument("Please choose a statistical model (stat_model) of the following:\n 'Bayesian', 'Outliers', 'Students', 'Gaussian', 'GB'");
    }
    energy *= nreplicas;
    return energy;
}
/*}}}*/

// Get data restraint energy first derivative:{{{
double get_data_restraint_energy_first_derivative(const int &nreplicas, const string &model,
        const vector<double> &devs, const vector<double> &dfX,
        const vector<double> &sigmaSEM, const vector<double> &sigmaB,
        const double &Ndof, const vector<double> &sseB, const string &data_uncertainty) {
    /*
     *
     */

    double energy = 0.0;
    if ( model == "Bayesian" || model == "Gaussian" ) {
        // use Gaussian derivative (same as Bayes)
        energy = get_d_energy_Gaussian(devs, dfX, sigmaSEM, sigmaB);
    }
    else if ( model == "GB"  ) {
        energy = get_d_energy_GB(devs, dfX, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Outliers" ) {
        energy = get_d_energy_Outliers(devs, dfX, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Students" ) {
        energy = get_d_energy_Students(devs, dfX, sigmaSEM, sigmaB, Ndof);
    }
    else {
        /* NOTE: throw exception links:
         https://en.cppreference.com/w/cpp/language/throw
         https://en.cppreference.com/w/cpp/error/exception
         http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
         https://stackoverflow.com/a/13186007
        */
        throw std::invalid_argument("Please choose a statistical model (stat_model) of the following:\n 'Bayesian', 'Outliers', 'Students', 'Gaussian', 'GB'");
    }
    energy *= nreplicas;
    return energy;
}
/*}}}*/

// Get data restraint energy second derivative:{{{
double get_data_restraint_energy_second_derivative(const int &nreplicas, const string &model,
        const vector<double> &devs, const vector<double> &dfX,
        const vector<double> &d2fX, const vector<double> &sigmaSEM,
        const vector<double> &sigmaB,
        const double &Ndof, const vector<double> &sseB, const string &data_uncertainty) {
    /*
     *
     */

    double energy = 0.0;
    if ( model == "Bayesian" || model == "Gaussian" ) {
        // use Gaussian derivative (same as Bayes)
        energy = get_d2_energy_Gaussian(devs, dfX, d2fX, sigmaSEM, sigmaB);
    }
    else if ( model == "GB"  ) {
        energy = get_d2_energy_GB(devs, dfX, d2fX, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Outliers" ) {
        energy = get_d2_energy_Outliers(devs, dfX, d2fX, sigmaSEM, sigmaB, Ndof);
    }
    else if ( model == "Students" ) {
        energy = get_d2_energy_Students(devs, dfX, d2fX, sigmaSEM, sigmaB, Ndof);
    }
    else {
        /* NOTE: throw exception links:
         https://en.cppreference.com/w/cpp/language/throw
         https://en.cppreference.com/w/cpp/error/exception
         http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
         https://stackoverflow.com/a/13186007
        */
        throw std::invalid_argument("Please choose a statistical model (stat_model) of the following:\n 'Bayesian', 'Outliers', 'Students', 'Gaussian', 'GB'");
    }
    energy *= nreplicas;
    return energy;

}
/*}}}*/

// neglogP:{{{

double neglogLikelihood(const vector<string> &data_uncertainty, const vector<double> &Ndofs,
        const double &logZ, const vector<int> &states, const vector<string> &models,
        const vector<vector<double>> &sigmaB, struct SS &ss) {
    /* Evaluates the data restraint energy.
     * if single sigma, then ss.sqerr[k].size = (N data points)
     * if multiple sigma, then ss.sqerr[k].size = (1)
     *
     */

    double result = 0.0;
    for (size_t k=0; k<data_uncertainty.size(); ++k) {
        /* NOTE: -ln( P(X, sigma | D) ) */
        result += get_data_restraint_energy(states.size(), models[k], ss.sqerr[k], ss.sem[k],
                sigmaB[k], Ndofs[k], ss.sqerrB[k], data_uncertainty[k]);
    }
    return result;
}


double neglogParameterPrior(const vector<string> &parameterPriorModel,
        const vector<double> &devs, const vector<double> &sigma_prior) {
    /* Evaluates the prior on parameters energy.
     *
     */

    int Np = devs.size();
    double energy = 0.0;
    /* NOTE: -ln( P(\theta) ) */
    if ( parameterPriorModel[0] == "Gaussian" ) {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            const double sqerr = devs[p]*devs[p];
            const double s2 = sigma_prior[p]*sigma_prior[p];
            energy += 0.5*sqerr / s2;
            energy += 0.5*log(2.0*M_PI*s2);
            energy += log(sigma_prior[p]); // Jeffrey's prior
        }
    }
    else if ( parameterPriorModel[0] == "GaussianSP"  ) {
        double sse = 0.0;
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            sse += devs[p]*devs[p];
        }
        const double s2 = sigma_prior[0]*sigma_prior[0];
        energy += 0.5*sse / s2;
        energy += Np*0.5*log(2.0*M_PI*s2);
        energy += log(sigma_prior[0]); // Jeffrey's prior
    }
    else if ( parameterPriorModel[0] == "Laplace"  ) {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            energy += abs(devs[p]) / sigma_prior[p];
            energy += log(2.0*sigma_prior[p]);
            energy += log(sigma_prior[p]); // Jeffrey's prior
        }
    }
    else if (parameterPriorModel[0] == "SpikeAndSlab") {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            const double spike_penalty = 1e6; // Large penalty factor for non-zero values
            const double slab_variance = sigma_prior[p]*sigma_prior[p]; // Assume the slab is Gaussian
            const double sqerr = devs[p]*devs[p];

            // Spike component (applies a large penalty if the parameter is non-zero)
            if (devs[p] != 0) {
                energy += spike_penalty;
            }

            // Slab component (Gaussian part)
            energy += 0.5*sqerr / slab_variance;
            energy += Np*0.5*log(2.0*M_PI*slab_variance);
            energy += log(sigma_prior[p]); // Jeffrey's prior for the slab part
        }
    }
    else if ( parameterPriorModel[0] == "uniform"  ) {
        //energy += get_uniform_prior_energy(sqerr, sigma_prior);
    }
    else {
        throw std::invalid_argument("Please choose a prior (parameter_prior) of the following:\n 'Laplace', 'Gaussian', 'Uniform'");
    }
    return energy;

}


double cpp_neglogP(const vector<double> &state_energies, const vector<string> &data_uncertainty,
        const vector<double> &reference_potentials, const vector<double> &Ndofs,
        const double &logZ, const vector<int> &states, const vector<string> &models,
        const vector<vector<double>> &sigmaB, struct SS &ss, const vector<double> &expanded_values) {
    /* Evaluates the energy function -logP = prior energy + data restraint energy
     *
     * Computes :math:`-logP` for NOE during MCMC sampling.
     * Args:
     *   parameters(list): collection of parameters for a given step of MCMC
     *   parameter_indices(list): collection of indices for a given step of MCMC
     *
     * :rtype: float
     *
     * sigmaB.size()    == (N data points)
     * sigmaB[0].size() == (sigma, gamma, beta, epsilon)
     */
    const double lambda = expanded_values[0];
    const double xi = expanded_values[1];
    const int nreplicas = states.size();
    int fm_Np = 0;
    const int nstates = state_energies.size();

    double result = 0.0;
    for (auto s: states) {
        /* NOTE: -ln( P(X) ) = E(X)/kbT + logZ */
        result += state_energies[s] + logZ;
    }

    result += xi*neglogLikelihood(data_uncertainty, Ndofs, logZ, states, models, sigmaB, ss);

    if ( !ss.mo.fm_prior_devs.empty() ) {

        /* NOTE: -ln( P(\theta) ) */
        for (size_t k=0; k<ss.mo.fm_prior_models.size(); ++k) {
            fm_Np += ss.mo.fm_prior_models[k].size();
            result += xi*nreplicas*neglogParameterPrior(ss.mo.fm_prior_models[k], ss.mo.fm_prior_devs[k], ss.mo.fm_prior_sigmas[k]);
            //result += xi*neglogParameterPrior(ss.mo.fm_prior_models[k], ss.mo.fm_prior_devs[k], ss.mo.fm_prior_sigmas[k]);
        }
    }

        /* FIXME: Does this need to be scaled by lambda? */
    if ( !ss.mo.pm_prior_devs.empty() ) {
        /* NOTE: -ln( P(\theta) ) */
        result += lambda*nreplicas*neglogParameterPrior(ss.mo.pm_prior_models, ss.mo.pm_prior_devs, ss.mo.pm_prior_sigmas);
        //result += lambda*neglogParameterPrior(ss.mo.pm_prior_models, ss.mo.pm_prior_devs, ss.mo.pm_prior_sigmas);
    }




    for (size_t k=0; k<reference_potentials.size(); ++k) {  // iterate over restraints
        /* NOTE: -ln( Q_ref(D) ) */
        result -= reference_potentials[k];
    }
    return result;
}



//:}}}

// dneglogP:{{{
double dneglogParameterPrior(const vector<string> &parameterPriorModel,
        const vector<double> &devs, const vector<double> &sigma_prior) {
    /* Evaluates the prior on parameters energy.
     *
     */

    double energy = 0.0;
    /* NOTE: d -ln( P(\theta) ) */
    if ( parameterPriorModel[0] == "Gaussian" ) {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            const double s2 = sigma_prior[p]*sigma_prior[p];
            energy += devs[p] / s2;
        }
    }
    else if ( parameterPriorModel[0] == "GaussianSP"  ) {
        double sse = 0.0;
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            sse += devs[p];
        }
        const double s2 = sigma_prior[0]*sigma_prior[0];
        energy += sse / s2;
    }
    else if ( parameterPriorModel[0] == "Laplace"  ) {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            energy += (devs[p] >= 0 ? 1 : -1) / sigma_prior[p];
        }
    }
    else if (parameterPriorModel[0] == "SpikeAndSlab") {
        for (size_t p=0; p<parameterPriorModel.size(); ++p) {
            const double s2 = sigma_prior[p] * sigma_prior[p];
            if (devs[p] != 0) {
                // For non-zero deviations, use the slab component
                energy += devs[p] / s2;  // derivative of Gaussian part
            }
            // No contribution from the spike in derivative terms
        }
    }
    else if ( parameterPriorModel[0] == "uniform"  ) {
        //energy += get_uniform_prior_energy(sqerr, sigma_prior);
    }
    else {
        throw std::invalid_argument("Please choose a prior (parameter_prior) of the following:\n 'Laplace', 'Gaussian', 'Uniform'");
    }
    return energy;

}


double cpp_dneglogP(const vector<double> &state_energies,
        const vector<double> &diff_state_energies,
        const double &logZ, const vector<int> &states,
        const vector<string> &models, const vector<vector<double>> &dfX,
        const vector<string> &data_uncertainty, const vector<double> &Ndofs,
        const vector<vector<double>> &sigmaB, struct SS &ss, const vector<double> &expanded_values) {
    /* Evaluates the 1st derivative of the BICePs energy function
     *
     */

    const double lambda = expanded_values[0];
    const double xi = expanded_values[1];

    const int nreplicas = states.size();
    double result = 0.0;
    double num = 0.0;
    double dprior = 0.0;
    /* Derivative of prior term */
    if ( all_of(diff_state_energies.begin(), diff_state_energies.end(),
                [](double element){ return element == 0.0; }) == false )
    {
      for (auto s: states) { dprior += diff_state_energies[s]; }

      for (size_t i=0; i<state_energies.size(); ++i) {
        num += exp(-state_energies[i])*diff_state_energies[i];
      }
      dprior -= nreplicas*num/exp(logZ);
      result += dprior;
    }

    ////////////////////////////////////////////////////////////////////////////
    if ( !ss.mo.fm_prior_devs.empty() ) {
        /* NOTE: d -ln( P(\theta) ) */
        for (size_t k=0; k<ss.mo.fm_prior_models.size(); ++k) {
            result += xi*nreplicas*dneglogParameterPrior(ss.mo.fm_prior_models[k], ss.mo.fm_prior_devs[k], ss.mo.fm_prior_sigmas[k]);
            //result += xi*dneglogParameterPrior(ss.mo.fm_prior_models[k], ss.mo.fm_prior_devs[k], ss.mo.fm_prior_sigmas[k]);
        }
    }
        /* FIXME: Does this need to be scaled by lambda? */
    if ( !ss.mo.pm_prior_devs.empty() ) {
        /* NOTE: d -ln( P(\theta) ) */
        result += lambda*nreplicas*dneglogParameterPrior(ss.mo.pm_prior_models, ss.mo.pm_prior_devs, ss.mo.pm_prior_sigmas);
        //result += lambda*dneglogParameterPrior(ss.mo.pm_prior_models, ss.mo.pm_prior_devs, ss.mo.pm_prior_sigmas);
    }
    ////////////////////////////////////////////////////////////////////////////


//    /* NOTE: -ln( P( L_{PC}(\theta)) ) */
//    if ( !ss.mo.prior_losses.grad_L_PC.empty() ) {
//        // Gaussian prior on L_PC
//        double energy = 0.0;
//        const double L_PC = ss.mo.prior_losses.L_PC;
//        const double dloss_sigma = ss.mo.pm_prob_conservation_loss_sigma[0];
//        const double s2 = dloss_sigma*dloss_sigma;
//        energy += 0.5*L_PC / s2;
//        energy += 0.5*log(2.0*M_PI*s2);
//        energy += log(dloss_sigma); // Jeffrey's prior
//        result += energy;
//    }





    /* Derivative of data restraint term */
    if ( all_of(dfX.begin(), dfX.end(), [](const vector<double> vec1) {
      return all_of(vec1.begin(), vec1.end(), [](const double element) {
      return element == 0.0;});
      }) == false )
    {
      for (size_t k=0; k<data_uncertainty.size(); ++k) {
        /* NOTE: -ln( P(X, sigma | D) ) */
        result += xi*get_data_restraint_energy_first_derivative(
                nreplicas, models[k], ss.devs[k], dfX[k], ss.sem[k],
                sigmaB[k], Ndofs[k], ss.sqerrB[k], data_uncertainty[k]);
      }
    }

    return result;
}


//:}}}

// d2neglogP:{{{
double cpp_d2neglogP(const vector<double> &state_energies,
        const vector<double> &diff_state_energies,
        const vector<double> &diff2_state_energies,
        const double &logZ, const vector<int> &states, const vector<string> &models,
        const vector<vector<double>> &dfX, const vector<vector<double>> &d2fX,
        const vector<string> &data_uncertainty, const vector<double> &Ndofs,
        const vector<vector<double>> &sigmaB, struct SS &ss, const vector<double> &expanded_values) {
    /* Evaluates the 2nd derivative of the BICePs energy function
    */

    const double lambda = expanded_values[0];
    const double xi = expanded_values[1];

    double result = 0.0;
    const int nreplicas = states.size();

    if ( (all_of(diff_state_energies.begin(), diff_state_energies.end(),
                [](double element){ return element == 0.0; }) == false ) && \
         (all_of(diff2_state_energies.begin(), diff2_state_energies.end(),
                [](double element){ return element == 0.0; }) == false ) )
    {
      for (auto s: states) {
          result += diff2_state_energies[s];
      }
      double num1 = 0.0;
      double num2 = 0.0;
      double num3 = 0.0;
      for (size_t i=0; i<state_energies.size(); ++i) {
          num1 += exp(-state_energies[i])*diff2_state_energies[i];
          num2 += exp(-state_energies[i])*diff_state_energies[i]*diff_state_energies[i];
          num3 += exp(-state_energies[i])*diff_state_energies[i];
      }
      result -= nreplicas*num1/exp(logZ);
      result += nreplicas*num2/exp(logZ);
      result -= nreplicas*num3*num3/(exp(logZ)*exp(logZ));
    }


    /* Derivative of data restraint term */
    if ( ( all_of(dfX.begin(), dfX.end(), [](vector<double> vec1) {
      return all_of(vec1.begin(), vec1.end(), [](const double element) {
      return element == 0.0;});
      }) == false ) && \
      ( all_of(dfX.begin(), dfX.end(), [](vector<double> vec1) {
        return all_of(vec1.begin(), vec1.end(), [](const double element) {
        return element == 0.0;});
        }) == false ) )
    {

      for (size_t k=0; k<data_uncertainty.size(); ++k) {
        /* NOTE: -ln( P(X, sigma | D) ) */
        result += xi*get_data_restraint_energy_second_derivative(
                nreplicas, models[k], ss.devs[k], dfX[k], d2fX[k], ss.sem[k],
                sigmaB[k], Ndofs[k], ss.sqerrB[k],
                data_uncertainty[k]);
      }
    }
    return result;
}


//:}}}

// get_u_kln_and_states_kn:{{{
GFE get_u_kln_and_states_kn(PyObject* ensembles,
        vector<vector<vector<int>>> state_traces, vector<vector<double>> energy_traces,
        vector<vector<vector<double>>> parameter_traces, vector<vector<vector<double>>> expanded_traces,
        const vector<float> logZs, bool progress=true, bool capture_stdout=false,
        bool scale_energies=false, bool compute_derivative=false, bool multiprocess=true, PyObject* sampler=nullptr)
{

    /*  Returns the energy matrix u_kln to be passed to MBAR. The construction
     *  of this matrix is as follows: Suppose the energies sampled from each
     *  simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
     *  of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at
     *  reduced potential for state l. Initialize MBAR with reduced energies
     *  u_kln and number of uncorrelated configurations from each state N_k.
     *  u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where
     *  U_l(x) is the potential energy function for state l,
     *  beta is the inverse temperature, and and x_kn denotes uncorrelated
     *  configuration n from state k.
     *
     *  This fucntion also returns states_kn.
     *
     */

    std::signal(SIGINT, handleSignal);
    omp_set_dynamic(1);
    bool debug = false;
    //bool debug = true;
    SS ss;
    GFE u;
    SEP sep;
    GRI gri;
    SCALE_OFFSET so;

    // Number of thermodynamic ensembles
    const size_t K = state_traces.size();
    const size_t nsnaps = state_traces[0].size();
    // For nreplicas: Find the size of the largest vector<int>
    size_t nreplicas = 0;
    for (const auto& vec1 : state_traces) {
      for (const auto& vec2 : vec1) { if (vec2.size() > nreplicas) { nreplicas = vec2.size(); } }
    }
    string sem_method = "sem";
    vector<int> rest_index;
    rest_index = get_rest_index(PySequence_GetItem(ensembles,0), rest_index);
    gri = get_restraint_information(PySequence_GetItem(ensembles,0));
    so = get_scaling_parameters(PySequence_GetItem(ensembles,0));
    const vector<string> data_likelihood = get_data_likelihood(PySequence_GetItem(ensembles,0));
    vector<vector<double>> ref_potentials = build_reference_potentials(PySequence_GetItem(ensembles,0));
    shared_ptr<vector<vector<double>>> lam_state_energies = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<double>>> lam_diff_state_energies = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<double>>> lam_diff2_state_energies = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<vector<vector<double>>>>> forward_model = make_shared<vector<vector<vector<vector<double>>>>>();
    shared_ptr<vector<vector<vector<vector<double>>>>> lam_diff_fwd_model = make_shared<vector<vector<vector<vector<double>>>>>();
    shared_ptr<vector<vector<vector<vector<double>>>>> lam_diff2_fwd_model = make_shared<vector<vector<vector<vector<double>>>>>();
    for (size_t l=0; l<K; ++l) {
      PyObject* ensemble = PySequence_GetItem(ensembles,l);
      forward_model->push_back(get_restraint_attr(ensemble, "model"));
      lam_state_energies->push_back(get_state_energies(ensemble, "energy"));
      if (compute_derivative){
        lam_diff_fwd_model->push_back(get_restraint_attr(ensemble, "diff model"));
        lam_diff2_fwd_model->push_back(get_restraint_attr(ensemble, "diff2 model"));
      }
      else{
        size_t size0 = forward_model->at(l).size();
        size_t size1 = forward_model->at(l)[0].size();
        size_t size2 = forward_model->at(l)[0][0].size();
        lam_diff_fwd_model->push_back(vector<vector<vector<double>>>(size0, vector<vector<double>>(size1, vector<double>(size2, 0.0))));
        lam_diff2_fwd_model->push_back(vector<vector<vector<double>>>(size0, vector<vector<double>>(size1, vector<double>(size2, 0.0))));
      }
    }
    if (compute_derivative){
      for (size_t l=0; l<K; ++l) {
        PyObject* ensemble = PySequence_GetItem(ensembles,l);
        lam_diff_state_energies->push_back(get_state_energies(ensemble, "diff_energy"));
        lam_diff2_state_energies->push_back(get_state_energies(ensemble, "diff2_energy"));
      }
    }

//    vector<vector<double>> ftilde = get_ftilde(lam_state_energies->at(0).size(), 5000, forward_model->at(0));
    u.u_kln = vector<vector<vector<double>>>(K, vector<vector<double>>(K, vector<double>(nsnaps, 0.0)));                    // shape: (K, K, nsnaps)
    u.diff_u_kln = vector<vector<vector<double>>>(K, vector<vector<double>>(K, vector<double>(nsnaps, 0.0)));               // shape: (K, K, nsnaps)
    u.diff2_u_kln = vector<vector<vector<double>>>(K, vector<vector<double>>(K, vector<double>(nsnaps, 0.0)));              // shape: (K, K, nsnaps)
    u.states_kn = vector<vector<vector<double>>>(K, vector<vector<double>>(nsnaps, vector<double>(nreplicas, nan("NaN")))); // shape: (K, nsnaps, nreplica)
    u.Nr_array = vector<vector<int>>(K, vector<int>(nsnaps, 1));                                                            // shape: (K, nsnaps)
    vector<vector<vector<double>>> experiments = get_restraint_attr(PySequence_GetItem(ensembles,0), "exp");                          // shape: (nstates, nrestraints, Nd)
    vector<vector<vector<double>>> weights = get_restraint_attr(PySequence_GetItem(ensembles,0), "weight");                                  // shape: (nstates, nrestraints, Nd)
    //const int n_procs = omp_get_num_procs();
    //cout << n_procs << endl;

//    ////////////////////////////////////////////////////////////////////////////
//    /* NOTE: Prior on parameters */
//    vector<MO> ensembleMOs;
//    for (size_t l = 0; l < n_ensembles; ++l) {
//        MO mo; ensembleMOs.push_back(mo);
//        /* NOTE: setup the ensembleMOs */
//        if ( !fm_parameters.empty() ) {
//            ensembleMOs[l].fm_parameters = fm_parameters;
//            ensembleMOs[l].fm_prior_models = vector<string>(ensembleMOs[l].fm_parameters.size(), "Gaussian");
//            ensembleMOs[l].fm_prior_sigmas = ? // vector<double>(ensembleMOs[l].fm_parameters.size(), 2.0);
//            ensembleMOs[l].fm_prior_mus = ? //ensembleMOs[l].fm_parameters;
//            ensembleMOs[l].fm_prior_devs = vector<vector<double>>(ensembleMOs[l].fwd_model_parameters.size(), vector<double>(ensembleMOs[l].fwd_model_parameters[0].size(), 0.0));
//        }
//        if ( !pm_parameters.empty() ) {
//            ensembleMOs[l].pm_parameters = pm_parameters;
//            ensembleMOs[l].pm_prior_models = vector<string>(ensembleMOs[l].pm_parameters.size(), "Gaussian");
//            ensembleMOs[l].pm_prior_sigmas = ? //vector<double>(ensembleMOs[l].pm_parameters.size(), 2.0);
//            ensembleMOs[l].pm_prior_mus = ? //ensembleMOs[l].pm_parameters;
//            ensembleMOs[l].pm_prior_devs = vector<double>(ensembleMOs[l].pm_parameters.size(), 0.0);
//        }
//    }
//
//
//    ////////////////////////////////////////////////////////////////////////////

    vector<MO> ensembleMOs;
    for (size_t l = 0; l < K; ++l) { MO mo; ensembleMOs.push_back(mo); }

    bool use_fmo = false;
    bool use_pmo = false;
    if (sampler != Py_None) {
      PyObject* _use_fmo = PyObject_GetAttrString(sampler, "fmo");
      if ( _use_fmo == NULL ) {
          PyObject_Print(_use_fmo, stdout, Py_PRINT_RAW);printf("\n");
          PyRun_SimpleString("import sys");printf("\n");
          PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
          PyErr_PrintEx(1);
      }
      use_fmo = PyObject_IsTrue(_use_fmo);
    }
    if (use_fmo) {
      for (size_t l = 0; l < K; ++l) {
          //MO mo; ensembleMOs.push_back(mo);
          FMO fmo;
          fmo.phi = get_phi_angles(sampler);
          fmo.phi0 = get_phase_shifts(sampler);
          fmo.fwd_model_parameters = get_fwd_model_parameters(sampler, 0);
          //cout << fmo.fwd_model_parameters[0] << endl;
          ensembleMOs[l].fm_parameters = fmo.fwd_model_parameters;
          //cout << ensembleMOs[l].fm_parameters[0] << endl;
          ensembleMOs[l].fm_prior_models = get_fmp_prior_models(sampler);
          //cout << ensembleMOs[l].fm_prior_models[0] << endl;
          ensembleMOs[l].fm_prior_mus = get_fwd_model_parameter_attr(sampler, "fmp_prior_mus");
          //cout << ensembleMOs[l].fm_prior_mus[0] << endl;
          ensembleMOs[l].fm_prior_devs = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 0.0));
          //cout << ensembleMOs[l].fm_prior_devs[0] << endl;
          ensembleMOs[l].fm_prior_sigmas = get_fwd_model_parameter_attr(sampler, "fmp_prior_sigmas");
          //cout << ensembleMOs[l].fm_prior_sigmas[0] << endl;
      }
    }

    ////////////////////////////////////////////////////////////////////////////


    tqdm bar(K);
    //if (progress) { bar.set_label("u_kln"); }


    // compute cross-energies for all other ensembles for each iteration
    #pragma omp parallel for if(multiprocess) shared(debug, u, logZs) firstprivate(energy_traces, forward_model,\
            lam_diff_fwd_model, lam_diff2_fwd_model, \
            experiments, weights, so, sep, data_likelihood, \
            lam_state_energies, gri, ss, expanded_traces, lam_diff_state_energies, lam_diff2_state_energies)
    {
      for (size_t k=0; k<K; ++k) {
        for (size_t n=0; n<nsnaps; ++n) {
          if (debug == true) { cout << k << "," << n <<endl; }
          if ( n >= (static_cast<size_t>(state_traces[k].size()) - 1) ) { continue; }
          const vector<int> states = state_traces[k][n];
          u.Nr_array[k][n] = states.size();

          for (int r=0; r<int(states.size()); ++r) { u.states_kn[k][n][r] = states[r]; }
          vector<int> indices(parameter_traces[k][n].size(), 0);
          sep = get_sep_indices_and_parameters(indices, parameter_traces[k][n], rest_index);
          so = change_scaling_parameters(so.scales_f, so.offsets, sep.sep_parameters);
          gri.reference_potentials = sum_reference_potentials(ref_potentials, states);

          for (size_t l=0; l<K; ++l) {
            if ( n >= (static_cast<size_t>(energy_traces[l].size()) - 1) ) { continue; }
            float logZ = logZs[l];
            if (k==l) { u.u_kln[k][l][n] = energy_traces[k][n]; }
            else {
              ss = get_sse_and_sem(forward_model->at(l),
                    lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
                    experiments, weights, states,
                    so.scales_f_exp, so.scales_f, so.offsets,
                    sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);
                    //sep.sep_parameters, data_likelihood, sem_method);

//        const vector<vector<double>> &fwd_model_parameters={{}},
//        const vector<vector<double>> &fwd_model_parameter_sigmas={{}},
//fwd_model_parameter_mus
//        const vector<vector<string>> &parameter_prior_models={{}},



              u.u_kln[k][l][n] = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
                    gri.reference_potentials, gri.Ndofs,
                    logZ, states, gri.models, sep.sep_parameters, ss, expanded_traces[l][n]);
            }
            if ( scale_energies ) { u.u_kln[k][l][n] /= u.Nr_array[k][n]; }
            if (debug == true) { cout << "Energy (k,l):(" << k << "," << l << ") = " << u.u_kln[k][l][n] << endl; }
            if ( compute_derivative ) {
              if (k==l) {
                ss = get_sse_and_sem(forward_model->at(l),
                      lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
                      experiments, weights, states,
                      so.scales_f_exp, so.scales_f, so.offsets,
                      sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);
                      //sep.sep_parameters, data_likelihood, sem_method);
              }
//              u.diff_u_kln[k][l][n] = cpp_dneglogP(lam_state_energies->at(l),
//                      lam_diff_state_energies->at(l), logZ, states,
//                      gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
//                      sep.sep_parameters, ss, expanded_traces[l][n]);
//              u.diff2_u_kln[k][l][n] = cpp_d2neglogP(lam_state_energies->at(l),
//                      lam_diff_state_energies->at(l), lam_diff2_state_energies->at(l), logZ, states,
//                      gri.models, ss.dfX, ss.d2fX, gri.data_uncertainty, gri.Ndofs,
//                      sep.sep_parameters, ss, expanded_traces[l][n]);

              const double du = cpp_dneglogP(lam_state_energies->at(l),
                      lam_diff_state_energies->at(l), logZ, states,
                      gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
                      sep.sep_parameters, ss, expanded_traces[l][n]);
              const double d2u = cpp_d2neglogP(lam_state_energies->at(l),
                      lam_diff_state_energies->at(l), lam_diff2_state_energies->at(l), logZ, states,
                      gri.models, ss.dfX, ss.d2fX, gri.data_uncertainty, gri.Ndofs,
                      sep.sep_parameters, ss, expanded_traces[l][n]);

              //if ( k == 0 ) { cout << du << endl; }
              u.diff_u_kln[k][l][n] = du;
              //u.diff2_u_kln[k][l][n] = d2u - du*du - du/exp(-u.u_kln[k][l][n]);
              u.diff2_u_kln[k][l][n] = d2u;
              if ( scale_energies ) {
                  u.diff_u_kln[k][l][n] /= u.Nr_array[k][n];
                  u.diff2_u_kln[k][l][n] /= u.Nr_array[k][n];
              }
            }
          }
          if (progress)
          {
            bar.set_label("u_kln");
            bar.progress(k, n, nsnaps);
          }

        }
        bar.finish(k);
      }
    }
    if (progress) { bar.close(); }
    return u;
}
/*}}}*/

//##### cppHREPosteriorSampler::init ####:{{{
class cppHREPosteriorSampler {

public:
    cppHREPosteriorSampler(PyObject* ensembles, PyObject* sampler,
            int nreplicas=1, int change_Nr_every=0,
            int write_every=100, int move_ftilde_every=1,
            double dftilde=0.1, double ftilde_sigma=1.0,
            bool scale_and_offset=false, bool verbose=false):
        ensembles(ensembles), sampler(sampler), nreplicas(nreplicas),
        change_Nr_every(change_Nr_every), write_every(write_every),
        move_ftilde_every(move_ftilde_every), dftilde(dftilde),
        ftilde_sigma(ftilde_sigma), scale_and_offset(scale_and_offset)
    {}

    void write_to_trajectories();
    void update_xi_value(int l, double dXi);

    void perform_swap(int index1, int index2,
        bool swap_sigmas, bool swap_forward_model, bool swap_ftilde);

    void replica_exchange(int swap_every, int nsteps,
        bool swap_sigmas, bool swap_forward_model,
        bool swap_ftilde, bool same_lambda_or_xi_only, bool save);

    void move_states(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri);

    SS move_sigmas(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, const int sigma_batch_size, bool move_in_all_dim, bool rand_num_of_dim,
        vector<vector<int>> lam_rest_index);

    void move_fm_parameters_gaussian(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);


    void move_fm_parameters_sgd(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);

    double adamUpdate(int l, double gradient, int paramIndex);
    void move_fm_parameters_adam(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);


    void move_fm_parameters_uniform(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);

    void (cppHREPosteriorSampler::*move_fm_parameters)(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);



    void move_fm_prior_sigma(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);

    void move_regularization_rho(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);



    void cpp_sample(const int nsteps, int swap_every, const int burn,
            bool swap_sigmas, bool swap_forward_model, const int print_frequency,
            const int sigma_batch_size, bool walk_in_all_dim,
            bool find_optimal_nreplicas, bool verbose, bool progress,
            bool multiprocess, bool capture_stdout);

    SS _update_Nr(int l, const vector<string> &data_likelihood,
            struct SEP sep, struct SS ss,
            vector<vector<double>> ref_potentials, struct GRI gri);


    vector<vector<double>> get_populations();
    vector<double> get_lam_populations(int l);
    void dealloc();

    void get_Nr_learning_rate();
    //void get_Nr_learning_rate(int step);
    vector<double> get_restraint_intensity(vector<vector<double>> sigmaB, vector<vector<double>> sigmaSEM);
    PyObject* get_traj();
    PyObject* get_ti_info();
    PyObject* get_exchange_info();
    PyObject* get_N_replicas();
    PyObject* get_fmp_traj();
    PyObject* get_pmp_traj();
    PyObject* get_pm_prior_sigma_traj();
/* NOTE: FIXME: TODO: you might want to make this function return a list of dictionaries?
 * Yea, it should be the same as convergence_metrics, except for regularization sigmas
 * */



    PyObject* get_fm_prior_sigma_traj();
    PyObject* get_convergence_metrics();
    void build_ref_potentials();

    vector<double> get_entropy();
    vector<double> get_chi2();
    double get_chi_squared(int l);

    bool check_replica_convergence(int l, const double old_chi2, const double threshold, bool verbose);

    void update_lam_state_counts(int l);

    // Public variables
    int nreplicas = nreplicas;
    int change_Nr_every = change_Nr_every;
    const int write_every = write_every;
    const int move_ftilde_every = move_ftilde_every;
    const double dftilde = dftilde;
    const double ftilde_sigma = ftilde_sigma;
    bool scale_and_offset = scale_and_offset;
    PyObject* ensembles = ensembles;
    PyObject* sampler = sampler;
    int nstates;
    int n_rest;
    bool verbose;
    double energy;   // initial energy
    int nrestraints;
    PyObject* trajs; // = SamplingTrajectory(ensemble, nreplicas);
    PyObject* exchange_info;
//    PyObject* xi_value_list;
    PyObject* expanded_value_list;
    PyObject* data_rest_list;

    int step;
    double exchange_attempts, exchanges, exchange_percentage;
    HRE hre;
    size_t n_ensembles =  static_cast<size_t>(PyLong_AsLong(PyLong_FromSsize_t(PySequence_Length(ensembles))));
    shared_ptr<vector<double>> energies;
    shared_ptr<vector<vector<int>>> lam_states;
    shared_ptr<vector<vector<int>>> lam_indices;
    shared_ptr<vector<vector<double>>> lam_parameters;
    unique_ptr<vector<vector<TrajectoryData>>> hre_traj = make_unique<vector<vector<TrajectoryData>>>();

    unique_ptr<vector<vector<ModelOptConvergenceMetrics>>> ensembleConvergenceMetrics = make_unique<vector<vector<ModelOptConvergenceMetrics>>>();

    shared_ptr<vector<float>> accepted = make_shared<vector<float>>();
    shared_ptr<vector<float>> attempted = make_shared<vector<float>>();
    shared_ptr<vector<float>> acceptance = make_shared<vector<float>>();
    shared_ptr<vector<int>> accept = make_shared<vector<int>>();

    shared_ptr<vector<vector<float>>> lam_sep_accepted = make_shared<vector<vector<float>>>();// = new vector<vector<float>>;

    shared_ptr<vector<vector<float>>> nuisance_parameters = make_shared<vector<vector<float>>>();

    shared_ptr<vector<vector<int>>> lam_state_counts = make_shared<vector<vector<int>>>();

    shared_ptr<vector<vector<double>>> lam_state_energies = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<double>>> lam_scales_exp = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<double>>> lam_scales = make_shared<vector<vector<double>>>();
    shared_ptr<vector<vector<double>>> lam_offsets = make_shared<vector<vector<double>>>();

    shared_ptr<vector<vector<double>>> expanded_values = make_shared<vector<vector<double>>>();

    shared_ptr<vector<vector<vector<vector<double>>>>> forward_model = make_shared<vector<vector<vector<vector<double>>>>>();
    shared_ptr<vector<vector<vector<vector<double>>>>> lam_diff_fwd_model = make_shared<vector<vector<vector<vector<double>>>>>();
    shared_ptr<vector<vector<vector<vector<double>>>>> lam_diff2_fwd_model = make_shared<vector<vector<vector<vector<double>>>>>();


    shared_ptr<vector<double>> prior_populations = make_shared<vector<double>>();

    vector<double> lam_entropy;// = new vector<double>;
    vector<double> lam_chi2;// = new vector<double>;
    vector<double> lam_dchi2;// = new vector<double>;
    vector<vector<double>> lam_populations;// = new vector<vector<double>>;

    shared_ptr<vector<vector<vector<double>>>> ftilde = make_shared<vector<vector<vector<double>>>>();

    float move_sigma_std;
    vector<float> dsigma;
    shared_ptr<vector<vector<float>>> lam_dsigma = make_shared<vector<vector<float>>>();
    int continuous_space;
    double avg_restraint_intensity;
    double _restraint_intensity;
    double eff;
    double _eff;
    int total_steps;
    vector<vector<double>> fwd_model_weights;
    bool fwd_model_mixture;
    vector<vector<vector<double>>> experiments;
    vector<vector<vector<double>>> weights;
    vector<vector<vector<vector<double>>>> fmp_traj;
    vector<vector<vector<double>>> pmp_traj;
    vector<vector<vector<double>>> pm_prior_sigma_traj;
    vector<vector<vector<vector<double>>>> fm_prior_sigma_traj;

    vector<vector<double>> lam_diff_state_energies;
    vector<double> logZs;
    vector<FMO> ensembleFMOs;
//    vector<PMO> ensemblePMOs;
    vector<MO> ensembleMOs;
    int model_idx;
    bool use_fmo;
    bool use_pmo;
    string sem_method;


};


//:}}}

// write to trajectories:{{{
class PyGilState {
public:
    PyGilState() {
        state_ = PyGILState_Ensure();
    }

    ~PyGilState() {
        PyGILState_Release(state_);
    }

private:
    PyGILState_STATE state_;
};


void set_indices_parameters(PyTrajData& td, const TrajectoryData& d) {
    /* Append the parameters and indices to the Python list to evenutally
     * be stored in the trajectory.
     */

    const size_t indices_size = d.indices.size();
    td.indices_list = PyList_New(indices_size);
    td.parameters_list = PyList_New(indices_size);
    for (size_t i=0; i<indices_size; ++i) {
        PyList_SetItem(td.indices_list, i, PyLong_FromLong(d.indices[i]));
        PyList_SetItem(td.parameters_list, i, PyFloat_FromDouble(d.parameters[i]));
        td.para = PyList_GetItem(td.sampled_parameters, i);
        PySequence_SetItem(td.para, d.indices[i], PyLong_FromDouble(PyFloat_AsDouble(PySequence_GetItem(td.para, d.indices[i])) + 1.0));
    }
    PyList_Append(td.traces, td.parameters_list);
}

void set_sep_and_sigma_traces(PyTrajData& td, const TrajectoryData& d) {
    /* Bayesian sigma and SEM sigma traces are stored in Python lists.
     *
     */

    const size_t sem_size = d.ss.sem.size();
    td.sem_list = PyList_New(sem_size);
    td.sse_list = PyList_New(sem_size);

    td.sep_para_list = PyList_New(sem_size);
    td.sep_ind_list = PyList_New(sem_size);

    for (size_t j = 0; j < sem_size; ++j) {
        const size_t sem_j_size = d.ss.sem[j].size();
        td._sem_list = PyList_New(sem_j_size);
        td._sse_list = PyList_New(sem_j_size);
        for (size_t k=0; k<sem_j_size; ++k) {
            PyList_SetItem(td._sem_list, k, PyFloat_FromDouble(d.ss.sem[j][k]));
            PyList_SetItem(td._sse_list, k, PyFloat_FromDouble(d.ss.sqerr[j][k]));
        }
        PyList_SetItem(td.sem_list, j, td._sem_list);
        PyList_SetItem(td.sse_list, j, td._sse_list);

        td.rest_ind_list = PyList_New(0);
        const size_t sep_indices_n_size = d.sep.sep_indices[j].size();
        td.rest_para_list = PyList_New(sep_indices_n_size);
        for (size_t m = 0; m < sep_indices_n_size; ++m) {
            PyList_Append(td.rest_ind_list, PyLong_FromLong(d.sep.sep_indices[j][m]));
            PyList_SetItem(td.rest_para_list, m, PyFloat_FromDouble(d.sep.sep_parameters[j][m]));
        }
        PyList_SetItem(td.sep_para_list, j, td.rest_para_list);
        PyList_SetItem(td.sep_ind_list, j, td.rest_ind_list);

    }
    PyList_Append(td.sem_trace, td.sem_list);
    PyList_Append(td.sse_trace, td.sse_list);

}


void set_state_list(PyTrajData& td, const TrajectoryData& d) {
    /* Store the state trace inside a Python list.
     *
     */

    const size_t states_size = d.states.size();
    td.state_list = PyList_New(states_size);
    for (size_t s = 0; s < states_size; ++s) {
        PyList_SetItem(td.state_list, s, PyLong_FromLong(d.states[s]));
    }
}

void process_trajectory(int l, const std::vector<TrajectoryData>& data, PyTrajData& td) {
    /* Conversion and storage of data for the trajectory Python list
     *
     */

    PyGilState gil;
    try {
      // increment reference count of Python objects
      Py_INCREF(td.traj);
      Py_INCREF(td.trajectory);
//      Py_INCREF(td.model_optimization);
      Py_INCREF(td.traces);
      Py_INCREF(td.sampled_parameters);
      Py_INCREF(td.sem_trace);
      Py_INCREF(td.sse_trace);
      for (auto& d : data) {
          // Store sampled_parameters
          set_indices_parameters(td, d);

          // Store sem and sse traces
          set_sep_and_sigma_traces(td, d);

          // Set state list
          set_state_list(td, d);

          PyObject* expanded_values_list = PyList_New(0);
          PyList_Append(expanded_values_list, PyFloat_FromDouble(d.expanded_values[0]));
          PyList_Append(expanded_values_list, PyFloat_FromDouble(d.expanded_values[1]));

          // Store objects in trajectory object
          td.traj_list = PyList_New(8);
          PyList_SetItem(td.traj_list, 0, PyLong_FromLong(d.step));
          PyList_SetItem(td.traj_list, 1, PyFloat_FromDouble(d.energy));
          PyList_SetItem(td.traj_list, 2, PyLong_FromLong(d.accept));
          PyList_SetItem(td.traj_list, 3, td.state_list);
          PyList_SetItem(td.traj_list, 4, td.sep_ind_list);
          PyList_SetItem(td.traj_list, 5, td.sep_para_list);
          PyList_SetItem(td.traj_list, 6, PyFloat_FromDouble(d.acceptance));
          //PyList_SetItem(td.traj_list, 7, PyFloat_FromDouble(d.xi));
          PyList_SetItem(td.traj_list, 7, expanded_values_list);
          PyList_Append(td.trajectory, td.traj_list);



//          // Store objects in trajectory object
//          td.mo_dict = PyDict_New();
//          PyList_SetItem(td.mo_dict, "step", PyLong_FromLong(d.step));
//          PyList_SetItem(td.mo_dict, "step", PyFloat_FromDouble(d.step));
//          PyList_Append(td.model_optimization, td.mo_dict);
      }

      // decrement reference count of Python objects
      Py_DECREF(td.traj);
      Py_DECREF(td.trajectory);
//      Py_DECREF(td.model_optimization);
      Py_DECREF(td.traces);
      Py_DECREF(td.sampled_parameters);
      Py_DECREF(td.sem_trace);
      Py_DECREF(td.sse_trace);
    }
    catch(const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
}

PyObject* cppTrajectory(PyObject* sampler, int ensemble_index)
{
    // import the copy module in Python
    PyObject* copy_module = PyImport_ImportModule("copy");

    // get the copy() function from the copy module
    PyObject* copy_func = PyObject_GetAttrString(copy_module, "deepcopy");

    // call the copy() function with the sampler object to create a copy
    PyObject* sampler_copy = PyObject_CallFunction(copy_func, "O", sampler);

    // create a new instance of PosteriorTrajectory with the sampler copy and ensemble index
    PyObject* posterior_traj = Trajectory(sampler_copy, ensemble_index);

    // decrement the reference count of the copy module, copy function, and sampler copy
    Py_DECREF(copy_module);
    Py_DECREF(copy_func);
    Py_DECREF(sampler_copy);

    // return the new instance of PosteriorTrajectory
    return posterior_traj;
}



void cppHREPosteriorSampler::write_to_trajectories() {
    /* The main functio for conversion and storage of data for each of the trajectory objects.
     *
     * NOTE: This function only works only without multiprocess due to the Python objects.
     * It is possible that boost C++ (libboost) can convert these data structures into
     * Python objects faster than the current approach implemented here. However,
     * that would be an additional dependency.
     */

    if (!hre_traj) return;
    const size_t hre_traj_size = hre_traj->size();
    bool multiprocess = false;
    //bool multiprocess = true;
//    std::vector<PyTrajData> trajs_vec(hre_traj_size);
    PyTrajData td;

    #pragma omp parallel for if(multiprocess) private(td) //shared(hre_traj)
    for (size_t l = 0; l < hre_traj_size; ++l) {
        auto& data = (*hre_traj)[l];
        //auto& td = trajs_vec[l];

        #pragma omp critical
        {
          // initialize trajectory data
          td.traj = PyList_GetItem(trajs, l);
//          cout << "init traj" << endl;
          // Check to make sure that the trajectory has been initialized properly
          if ( td.traj == nullptr ) {
              PyObject_Print(td.traj, stdout, Py_PRINT_RAW);printf("\n");
              PyRun_SimpleString("import sys");printf("\n");
              PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
              PyErr_PrintEx(1);
          }

          td.trajectory = PyObject_GetAttrString(td.traj, "trajectory");
//          td.model_optimization = PyObject_GetAttrString(td.traj, "model_optimization");
          td.traces = PyObject_GetAttrString(td.traj, "traces");
          td.sampled_parameters = PyObject_GetAttrString(td.traj, "sampled_parameters");
          td.sem_trace = PyObject_GetAttrString(td.traj, "sem_trace");
          td.sse_trace = PyObject_GetAttrString(td.traj, "sse_trace");

/* add reference potential: {{{*/
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
        PyObject* trajectory_ref = PyObject_GetAttrString(td.traj, "ref");
        int condition1,condition2,condition3;
        nrestraints = PyLong_AsLong(PyLong_FromSsize_t(PySequence_Length(PySequence_GetItem(PySequence_GetItem(ensembles,l),0))));
        for (int i=0; i<nrestraints; ++i) {
            PyObject* traj_ref = PySequence_GetItem(trajectory_ref, i);
            PyObject* R = PySequence_GetItem(PySequence_GetItem(PySequence_GetItem(ensembles,l),0),i);
            PyObject* ref = PyObject_GetAttrString(R, "ref");
            condition1 = PyObject_RichCompareBool(ref, PyUnicode_FromString("uniform"), Py_EQ);
            condition2 = PyObject_RichCompareBool(ref, PyUnicode_FromString("exponential"), Py_EQ);
            condition3 = PyObject_RichCompareBool(ref, PyUnicode_FromString("gaussian"), Py_EQ);
            if ( condition1 ) {
                PyList_Append(traj_ref, PyUnicode_FromString("NaN"));
            }
            else if ( condition2 ) {
                if ( PyObject_HasAttrString(R, "precomputed") ) {
                    PyObject* precomputed = PyObject_GetAttrString(R, "precomputed");
                    if ( not precomputed ) {
                        build_exp_ref_pf(PySequence_GetItem(ensembles,l), i);
                    }
                }
                else {
                    build_exp_ref(PySequence_GetItem(ensembles,l), i);
                }
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "betas"));
            }
            else if ( condition3 ) {
                if ( PyObject_HasAttrString(R, "precomputed") ) {
                    PyObject* precomputed = PyObject_GetAttrString(R, "precomputed");
                    if ( not precomputed ) {
                        build_gaussian_ref_pf(PySequence_GetItem(ensembles,l), i);
                    }
                }
                else {
                    build_gaussian_ref(PySequence_GetItem(ensembles,l), i, PyObject_GetAttrString(R, "use_global_ref_sigma"));
                }
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "ref_mean"));
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "ref_sigma"));
            }
            else {
                throw std::invalid_argument("Please choose a reference potential of the following:\n 'uniform', 'exp', 'gaussian'");
            }
        }
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*}}}*/

        }

//        cout << "start" << endl;
        process_trajectory(l, data, td);

    }
//    cout << "Done... " << endl;

}


/*}}}*/

// Sample:{{{
void cppHREPosteriorSampler::cpp_sample(const int nsteps, int swap_every,
        const int burn, bool swap_sigmas, bool swap_forward_model, const int print_frequency,
        const int sigma_batch_size, bool walk_in_all_dim, bool find_optimal_nreplicas,
        bool verbose, bool progress, bool multiprocess, bool capture_stdout)
{
    /* Run Markov Chain Monte Carlo (MCMC) by taking random steps in
     * state-space and parameter space. Acceptance is determined by Metroplis-Hastings
     * criterion and evaluating the energy for each moveset.
     *
     *
     */

    std::signal(SIGINT, handleSignal);
    omp_set_dynamic(1);
    //omp_lock_t data_lock;
    //omp_init_lock(&data_lock);
    /* NOTE: First check to see if the current thread is holding the
     * Global Interpreter lock (GIL). If so, wait for the thread to complete
     * it's task...
     * https://docs.python.org/3/c-api/init.html?highlight=pygilstate_check#c.PyThreadState
     */
    bool pyGIL_checked = false;
    while ( pyGIL_checked == false )
    {
        if (PyGILState_Check()) { pyGIL_checked = true; }
    }

    PyObject* _xi_integration = PyObject_GetAttrString(sampler, "xi_integration");
    if ( _xi_integration == NULL ) {
        PyObject_Print(_xi_integration, stdout, Py_PRINT_RAW);printf("\n");
        PyRun_SimpleString("import sys");printf("\n");
        PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
        PyErr_PrintEx(1);
    }
    bool xi_integration = PyObject_IsTrue(_xi_integration);

    PyObject* _dXi = PyObject_GetAttrString(sampler, "dXi");
    if ( _dXi == NULL ) {
        PyObject_Print(_xi_integration, stdout, Py_PRINT_RAW);printf("\n");
        PyRun_SimpleString("import sys");printf("\n");
        PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
        PyErr_PrintEx(1);
    }
    double dXi = PyFloat_AsDouble(_dXi);

    int update_xi_every;
    if ( xi_integration ) {
      PyObject* change_xi_every = PyObject_GetAttrString(sampler, "change_xi_every");
      if ( change_xi_every == NULL ) {
          PyObject_Print(change_xi_every, stdout, Py_PRINT_RAW);printf("\n");
          PyRun_SimpleString("import sys");printf("\n");
          PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
          PyErr_PrintEx(1);
      }
      update_xi_every = PyLong_AsLong(change_xi_every);

      if ( verbose ) { cout << "update xi every: " << update_xi_every << endl; }
    }
    else { update_xi_every = 0; }


    int xi_schedule_idx = 0; // start at 1st index, since we always start at xi=1
    bool use_xi_schedule;
    vector<double> xi_schedule;
    if ( xi_integration ) {
      PyObject* _xi_schedule = PyObject_GetAttrString(sampler, "xi_schedule");
      if ( _xi_schedule == NULL ) {
          PyObject_Print(_xi_schedule, stdout, Py_PRINT_RAW);printf("\n");
          PyRun_SimpleString("import sys");printf("\n");
          PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
          PyErr_PrintEx(1);
      }
      else if (_xi_schedule == Py_None) {
        use_xi_schedule = false;
      }
      else if (PyList_Check(_xi_schedule)) {
        use_xi_schedule = true;
        Py_ssize_t len = PyList_Size(_xi_schedule);
        for (Py_ssize_t i = 0; i < len; ++i) {
            PyObject* item = PyList_GetItem(_xi_schedule, i);
            if ( (i == 0) && (PyFloat_AsDouble(item) == 1.0) ) {
                /* NOTE: */
                // pass. We start at Xi = 1.0. Therefore, it is automatically included in the xi_schedule
            }
            else {
                xi_schedule.push_back(PyFloat_AsDouble(item));
            }
        }
      }
      //Py_XDECREF(_xi_schedule);

      //cout << "xi_schedule = " << xi_schedule << endl;
    }




    vector<vector<double>> data_restraint_values;
    bool append_data_restraint_values = xi_integration;
    bool same_lambda_or_xi_only = true;
    //bool same_lambda_or_xi_only = false;

    bool move_in_all_dim = walk_in_all_dim;
    bool rand_num_of_dim;
    if ( sigma_batch_size == 0 ) { cout << "rand_num_of_dim " << endl;rand_num_of_dim = true; }
    else { rand_num_of_dim = false; }

    bool swap_ftilde = false;
    if (swap_forward_model) { swap_ftilde = true; }
    SS ss;
    SEP sep;
    GRI gri;
    HRE hre;
    SCALE_OFFSET so;
    //bool continuous_space = true;


    PyObject* _dsigma = PyObject_GetAttrString(sampler, "dsigma");

    const size_t n_ensembles = static_cast<size_t>(Py_SIZE(ensembles));
    shared_ptr<vector<vector<float>>> lam_dsigma = make_shared<vector<vector<float>>>();
    lam_dsigma->resize(n_ensembles); // Allocate memory

    // size of _dsigma
    auto dsigma_size = static_cast<size_t>(Py_SIZE(_dsigma));

    for (auto& dsigma : *lam_dsigma) {
        dsigma.reserve(dsigma_size); // Preallocate memory
        for (size_t s=0; s<dsigma_size; ++s) {
            dsigma.push_back(PyFloat_AsDouble(PySequence_GetItem(_dsigma, s)));
        }
    }

    move_sigma_std = PyFloat_AsDouble(PyObject_GetAttrString(sampler, "move_sigma_std"));
    continuous_space = PyLong_AsLong(PyObject_GetAttrString(sampler, "continuous_space"));

    fwd_model_mixture = PyObject_IsTrue(PyObject_GetAttrString(sampler, "fwd_model_mixture"));
    fwd_model_weights = get_fwd_model_weights(sampler);

    PyObject* prior = PyObject_GetAttrString(sampler, "prior_populations");

    auto prior_size = static_cast<size_t>(Py_SIZE(prior));
    shared_ptr<vector<double>> prior_populations = make_shared<vector<double>>(prior_size);

    for (size_t s = 0; s < prior_size; ++s) {
        (*prior_populations)[s] = PyFloat_AsDouble(PySequence_GetItem(prior, s));
    }

    const int attempt_move_state_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_state_every"));
    const int attempt_move_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_sigma_every"));
    const int attempt_move_fmp_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_fmp_every"));
    const int attempt_move_pmp_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pmp_every"));
    const int attempt_move_fm_prior_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_fm_prior_sigma_every"));
    const int attempt_move_pm_prior_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pm_prior_sigma_every"));
    const int attempt_move_pm_extern_loss_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_pm_extern_loss_sigma_every"));
    const int attempt_move_DB_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_DB_sigma_every"));
    const int attempt_move_PC_sigma_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_PC_sigma_every"));
    const int attempt_move_lambda_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_lambda_every"));
    const int attempt_move_xi_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_xi_every"));
    const int attempt_move_rho_every = PyLong_AsLong(PyObject_GetAttrString(sampler, "attempt_move_rho_every"));


    // iterate through each lambda value. Each lambda has a trajectory
    exchange_info = PyList_New(0);
    //vector<double> logZs;
    trajs = PyList_New(n_ensembles);
    expanded_values = make_shared<vector<vector<double>>>(n_ensembles, vector<double>(2, 0.0));
    PyObject* expand_vals = PyObject_GetAttrString(sampler, "expanded_values");
    PyObject* list_logZs = PyObject_GetAttrString(sampler, "logZs");
    for (size_t l=0; l<n_ensembles; ++l) {
        PyObject* _traj_ = cppTrajectory(sampler, l);
        // Check to make sure that the trajectory has been initialized properly
        if ( _traj_ == nullptr ) {
            PyObject_Print(_traj_, stdout, Py_PRINT_RAW);printf("\n");
            PyRun_SimpleString("import sys");printf("\n");
            PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
            PyErr_PrintEx(1);
        }
        PyList_SetItem(trajs, l, _traj_);
        logZs.push_back(PyFloat_AsDouble(PySequence_GetItem(list_logZs, l)));
        double _lam = PyFloat_AsDouble(PySequence_GetItem(PySequence_GetItem(expand_vals, l), 0));
        double _xi = PyFloat_AsDouble(PySequence_GetItem(PySequence_GetItem(expand_vals, l), 1));
        (*expanded_values)[l][0] = _lam;
        (*expanded_values)[l][1] = _xi;
    }

    PyObject* logZ_list = PyList_New(0);
    for (size_t l=0; l<logZs.size(); ++l) { PyList_Append(logZ_list, PyFloat_FromDouble(logZs[l])); }
    PyObject_SetAttrString(sampler, "logZ", logZ_list);

    hre_traj = make_unique<vector<vector<TrajectoryData>>>(expanded_values->size(), vector<TrajectoryData>());
    for (size_t l=0; l<logZs.size(); ++l) { hre_traj->at(l).clear(); }


    /* FIXME: Why do we have two functions that build reference potentials? */
//    build_ref_potentials(); // stores reference potentials in trajectories
    vector<vector<double>> ref_potentials = build_reference_potentials(PySequence_GetItem(ensembles,0));

    //forward_model = make_shared<vector<vector<vector<vector<double>>>>>();
    //lam_diff_fwd_model = make_shared<vector<vector<vector<vector<double>>>>>();
    //lam_diff2_fwd_model = new vector<vector<vector<vector<double>>>>;

    for (size_t l=0; l<n_ensembles; ++l) {
      PyObject* ensemble = PySequence_GetItem(ensembles,l);
      forward_model->push_back(get_restraint_attr(ensemble, "model"));
      lam_diff_fwd_model->push_back(get_restraint_attr(ensemble, "diff model"));
      lam_diff2_fwd_model->push_back(get_restraint_attr(ensemble, "diff2 model"));
    }

    ensembleConvergenceMetrics = make_unique<vector<vector<ModelOptConvergenceMetrics>>>(n_ensembles, vector<ModelOptConvergenceMetrics>());
    for (size_t l=0; l<n_ensembles; ++l) { ensembleConvergenceMetrics->at(l).clear(); }


    for (size_t l = 0; l < n_ensembles; ++l) { MO mo; ensembleMOs.push_back(mo); }

    /* NOTE: Forward model optimization protocol (if user activates the fmo argument
     * in PosteriorSampler).
     * Right now this code works for refining Karplus coefficients, but it
     * has been written to be general; other forward models can be implemented
     * with minimal edits/additions to the preexisting code structure.
     *
     * Currently, there's no functionality to switch between continuous and a
     * discrete forward model parameter space.  You must comment out the corresponding
     * function `move_fm_parameters`.
     */
    /* Forward model optimization initialization: {{{*/

    string fmo_method = PyUnicode_AsUTF8(PyObject_GetAttrString(sampler, "fmo_method"));
    fmo_method = transformStringToLower(fmo_method);
    if ( fmo_method == "sgd" ) { move_fm_parameters = &cppHREPosteriorSampler::move_fm_parameters_sgd; }
    else if ( fmo_method == "adam"  ) { move_fm_parameters = &cppHREPosteriorSampler::move_fm_parameters_adam; }
    else if ( fmo_method == "gaussian"  ) { move_fm_parameters = &cppHREPosteriorSampler::move_fm_parameters_gaussian; }
    else if ( fmo_method == "uniform"  ) { move_fm_parameters = &cppHREPosteriorSampler::move_fm_parameters_uniform; }
    else { throw std::invalid_argument("Please choose a fmo_method of the following:\n 'SGD', 'ADAM', 'Uniform'"); }

    PyObject* _use_fmo = PyObject_GetAttrString(sampler, "fmo");
    if ( _use_fmo == NULL ) {
        PyObject_Print(_use_fmo, stdout, Py_PRINT_RAW);printf("\n");
        PyRun_SimpleString("import sys");printf("\n");
        PyRun_SimpleString("print(sys.last_traceback)");printf("\n");
        PyErr_PrintEx(1);
    }
    use_fmo = PyObject_IsTrue(_use_fmo);

    model_idx = PyLong_AsLong(PyObject_GetAttrString(sampler, "fmo_model_idx"));


    if (use_fmo)
    {
      fmp_traj = vector<vector<vector<vector<double>>>>(n_ensembles);
      fm_prior_sigma_traj = vector<vector<vector<vector<double>>>>(n_ensembles);
      for (size_t l = 0; l < n_ensembles; ++l) {
          FMO fmo;
          fmo.phi = get_phi_angles(sampler);
          fmo.phi0 = get_phase_shifts(sampler);
          //d_fmp = get_d_fmp(sampler);
//          fmo.fwd_model_obj = PyObject_GetAttrString(sampler, "fwd_model_obj");


          fmo.batch_size = static_cast<size_t>(PyLong_AsLong(PyObject_GetAttrString(sampler, "fmp_batch_size")));

          fmo.fwd_model_parameters = get_fwd_model_parameters(sampler, l);
//          fmo.fwd_model_parameters = get_fwd_model_parameter_attr(sampler, l, "fwd_model_parameters");
          fmo.restraint_indices = get_fmo_restraint_indices(sampler);
          //fmo.min_max_parameters = vector<vector<double>>(fmo.fwd_model_parameters[0].size(), vector<double>(2, 0.0));

          //fmo.min_max_parameters = get_min_max_fwd_model_parameters(sampler);

          fmo.min_max_parameters = get_fwd_model_parameter_attr(sampler, "min_max_fwd_model_parameters");


          /* NOTE: setup the ensembleMOs */
          ensembleMOs[l].rho = 1.0;
          ensembleMOs[l].reg_rho_accepted = 0.0;
          ensembleMOs[l].reg_rho_attempted = 0.0;
          ensembleMOs[l].reg_rho_acceptance = 1.0;
          ensembleMOs[l].reg_rho_eta = 0.01;


          ensembleMOs[l].fm_prior_accepted = 0.0;
          ensembleMOs[l].fm_prior_attempted = 0.0;
          ensembleMOs[l].fm_prior_acceptance = 1.0;
          //ensembleMOs[l].fm_prior_acceptance = 0.0;

//          fmo.fmp_prior_models = get_fmp_prior_models(sampler);
//          fmo.fmp_prior_sigmas = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 1.0));
//          fmo.fmp_prior_mus = get_fmp_prior_mus(sampler);

          ensembleMOs[l].fm_parameters = fmo.fwd_model_parameters;

          //ensembleMOs[l].fm_prior_models = vector<vector<string>>(fmo.fwd_model_parameters.size() * vector<string>(fmo.fwd_model_parameters[0].size(), "Uniform"));
//          ensembleMOs[l].fm_prior_models = vector<vector<string>>(fmo.fwd_model_parameters.size(), vector<string>(fmo.fwd_model_parameters[0].size(), "Gaussian"));
          ensembleMOs[l].fm_prior_models = get_fmp_prior_models(sampler);


          //ensembleMOs[l].fm_prior_mus = get_fmp_prior_mus(sampler);

          //ensembleMOs[l].fm_prior_mus = fmo.fwd_model_parameters;

          //ensembleMOs[l].fm_prior_mus = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 0.0));
          ensembleMOs[l].fm_prior_mus = get_fwd_model_parameter_attr(sampler, "fmp_prior_mus");
          ensembleMOs[l].fm_prior_devs = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 0.0));
          ensembleMOs[l].fm_prior_sigmas = get_fwd_model_parameter_attr(sampler, "fmp_prior_sigmas");

          //cout << "Sigmas: " << ensembleMOs[l].fm_prior_sigmas[0] << endl;

          //cout << fmo.min_max_parameters[0] << endl;
          //cout << ensembleMOs[l].fm_prior_mus[0] << endl;

//          if (attempt_move_fm_prior_sigma_every != nsteps) {
//              //ensembleMOs[l].fm_prior_sigmas = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 10.0));
//              ensembleMOs[l].fm_prior_sigmas = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 5.0));
//          }
//          else {
//              ensembleMOs[l].fm_prior_sigmas = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 2.0));
//          }


          //cout << ensembleMOs[l].fm_parameters.size()  <<  ", " << ensembleMOs[l].fm_parameters[0].size() << endl;
          //cout << ensembleMOs[l].fm_prior_models.size()  <<  ", " << ensembleMOs[l].fm_prior_models[0].size() << endl;
          //cout << ensembleMOs[l].fm_prior_devs.size() <<  ", " << ensembleMOs[l].fm_prior_devs[0].size() << endl;
          //cout << ensembleMOs[l].fm_prior_sigmas.size() <<  ", " << ensembleMOs[l].fm_prior_sigmas[0].size() << endl;



          //fmo.fwd_model_parameters = get_fwd_model_parameter_attr(sampler, "min_max_fwd_model_parameters");

          //for (size_t i=0; i < fmo.fmp_prior_models.size(); ++i) {
          //    //fmo.min_max_parameters[i] = {-10.0, 10.0};
          //    cout << fmo.min_max_parameters[i] << endl;
          //}

          //for (size_t i=0; i < fmo.fmp_prior_models.size(); ++i) {
          //    cout << fmo.fmp_prior_models[i] << endl;
          //}

          //exit(1);
          ///////////////////////////////////////////////////////////////////////
          fmo.max_gradients = vector<double>(fmo.fwd_model_parameters[0].size(), 0.0);

          for (size_t i=0; i < fmo.min_max_parameters.size(); ++i) {
              fmo.max_gradients[i] = std::max(fabs(fmo.min_max_parameters[i][0]), fabs(fmo.min_max_parameters[i][1]))/3;
          }

//          fmo.max_gradients[0] = std::max(fabs(fmo.min_max_parameters[0][0]), fabs(fmo.min_max_parameters[0][1]))/3;
//          fmo.max_gradients[1] = std::max(fabs(fmo.min_max_parameters[1][0]), fabs(fmo.min_max_parameters[1][1]))/3;
//          fmo.max_gradients[2] = std::max(fabs(fmo.min_max_parameters[2][0]), fabs(fmo.min_max_parameters[2][1]))/5;

          fmo.decay_rate = 0.9;
          fmo.velocity = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 0.0));
          fmo.replica_scale_factor = 1.0 / sqrt(nreplicas);
          fmo.base_epsilon = 0.01; // Base epsilon without scaling
          fmo.target_acceptance = 0.234; // Common target for high-dimensional distributions
          fmo.acceptance_adjustment_factor = 0.1; // How quickly to adjust epsilon
          //fmo.initial_eta_scale = 0.01;//* sqrt(nreplicas);
          //fmo.min_eta_scale = 0.01;

          //fmo.max_eta_scale = 0.1*nreplicas*nreplicas;// * nreplicas;//*nreplicas;
          //fmo.max_eta_scale = 10.0; //*nreplicas*nreplicas;// * nreplicas;//*nreplicas;
          fmo.max_eta_scale = 1.0;
          fmo.min_eta_scale = 0.01;// * nreplicas;//*nreplicas;
          fmo.initial_eta_scale = fmo.min_eta_scale;// * nreplicas;//*nreplicas;
          fmo.eta = fmo.initial_eta_scale;

          ensembleMOs[l].fm_max_eta_prior_scale = 0.10;
          ensembleMOs[l].fm_min_eta_prior_scale = 0.001;// * nreplicas;//*nreplicas;
          ensembleMOs[l].fm_eta_prior = ensembleMOs[l].fm_min_eta_prior_scale;


          fmo.fmp_accepted = 0.0;
          fmo.fmp_attempted = 0.0;
          fmo.fmp_acceptance = 1.0;
          fmo.fmp_scale_factor = 1.0;

          //fmo.target_acceptance = 0.434; // Common target for high-dimensional distributions


          fmo.beta1 = 0.9;
          fmo.beta2 = 0.999;
          fmo.alpha = 0.001;
          //fmo.alpha = 0.01;
          fmo.alpha = 0.05;
          fmo.alpha = 0.01;


          fmo.epsilon_adam = 1e-8;
          fmo.t = 1;
          fmo.first_moment = vector<double>(fmo.fwd_model_parameters.size() * fmo.fwd_model_parameters[0].size(), 0.0);
          fmo.second_moment = vector<double>(fmo.fwd_model_parameters.size() * fmo.fwd_model_parameters[0].size(), 0.0);


          fmo.max_temperature = 100;
          fmo.temperature = fmo.max_temperature;
          fmo.cooling_rate = 0.99999;



          fmo.du_dtheta = vector<vector<double>>(fmo.fwd_model_parameters.size(), vector<double>(fmo.fwd_model_parameters[0].size(), 0.0));
          fmo.fwd_model_derivatives = vector<vector<vector<vector<double>>>>(fmo.fwd_model_parameters.size(), vector<vector<vector<double>>>(fmo.phi[0].size(), vector<vector<double>>(fmo.fwd_model_parameters[0].size())));

          for (size_t i = 0; i < fmo.fwd_model_parameters.size(); ++i) {
              for (size_t s = 0; s < fmo.phi[i].size(); ++s) {
                  fmo.fwd_model_derivatives[i][s] = vector<vector<double>>(fmo.fwd_model_parameters[0].size(), vector<double>(fmo.phi[i][s].size(), 0.0));
              }
          }
          tuple<vector<double>, vector<vector<double>>> couplings;
          fmo.new_forward_model = forward_model->at(0);
          if (fmo.fwd_model_parameters.size() != 0) {
            for (size_t i=0; i<fmo.fwd_model_parameters.size(); ++i) {
                size_t k = fmo.restraint_indices[i];
                for (size_t s = 0; s < fmo.phi[i].size(); ++s) {
                    couplings = cpp_get_scalar_couplings_with_derivatives(fmo.phi[i][s],
                                fmo.fwd_model_parameters[i], fmo.phi0[i], model_idx);
                                //fmo.fwd_model_parameters[i], fmo.phi0[i], fmo.fwd_model_obj);
                    fmo.new_forward_model[s][k] = std::get<0>( couplings );
                    fmo.fwd_model_derivatives[i][s] = std::get<1>(couplings);
                }
            }
          }
          ensembleFMOs.push_back(fmo);
          forward_model->at(l) = fmo.new_forward_model;
      }
    }
    /*}}}*/


    sem_method = PyUnicode_AsUTF8(PyObject_GetAttrString(sampler, "sem_method"));
    sem_method = transformStringToLower(sem_method);


    experiments = get_restraint_attr(PySequence_GetItem(ensembles,0), "exp"); // shape: (nstates, nrestraints, Nd)
    weights = get_restraint_attr(PySequence_GetItem(ensembles,0), "weight");         // shape: (nstates, nrestraints, Nd)

    nuisance_parameters = make_shared<vector<vector<float>>>(compile_nuisance_parameters(PySequence_GetItem(ensembles,0)));

    gri = get_restraint_information(PySequence_GetItem(ensembles,0));
    const vector<string> data_likelihood = get_data_likelihood(PySequence_GetItem(ensembles,0));
    /* NOTE: Get public variables from the sampler object */
    //int nstates = PyLong_AsLong(PyObject_GetAttrString(sampler, "nstates"));
    nstates = forward_model->at(0).size();
    /* NOTE: ///////////////////////////////////////////////////////// */
    //energy = PyFloat_AsDouble(PyObject_GetAttrString(sampler, "E"));
    PyObject* _energy = PyObject_GetAttrString(sampler, "E");
    PyObject* _states = PyObject_GetAttrString(sampler, "states");
    PyObject* _indices = PyObject_GetAttrString(sampler, "indices");
    PyObject* _parameters = PyObject_GetAttrString(sampler, "parameters");
    /* NOTE: Take the public variables and create multidimensioal vectors to
     * embed values for each lambda value */

    energies = make_shared<vector<double>>(expanded_values->size(), 0.0);
    lam_states = make_shared<vector<vector<int>>>(expanded_values->size(), vector<int>(nreplicas,0));
    lam_indices = make_shared<vector<vector<int>>>();
    lam_parameters = make_shared<vector<vector<double>>>();
    lam_state_counts = make_shared<vector<vector<int>>>(expanded_values->size(), vector<int>(nstates,1));

    vector<vector<int>> lam_rest_index;

    lam_state_energies = make_shared<vector<vector<double>>>();
    lam_scales_exp = make_shared<vector<vector<double>>>();
    lam_scales = make_shared<vector<vector<double>>>();
    lam_offsets = make_shared<vector<vector<double>>>();
    lam_sep_accepted = make_shared<vector<vector<float>>>();

    //lam_diff_state_energies = make_shared<vector<vector<double>>>();

    for (int l=0; l<int(expanded_values->size()); ++l) {
        (*energies).at(l) = PyFloat_AsDouble(PySequence_GetItem(_energy, l));
        vector<int> rest_index;
        rest_index = get_rest_index(PySequence_GetItem(ensembles,l), rest_index);
        lam_rest_index.push_back(rest_index);
        PyObject* l_states = PySequence_GetItem(_states, l);
        for (int i=0; i<nreplicas; ++i) {
            (*lam_states).at(l).at(i) = PyLong_AsLong(PySequence_GetItem(l_states, i));
        }
        vector<int> indices;
        vector<double> parameters;
        PyObject* l_indices = PySequence_GetItem(_indices, l);
        PyObject* l_parameters = PySequence_GetItem(_parameters, l);
        for (int i=0; i<int(rest_index.size()); ++i) {
            indices.push_back(PyLong_AsLong(PySequence_GetItem(l_indices, i)));
            parameters.push_back(PyFloat_AsDouble(PySequence_GetItem(l_parameters, i)));
        }
        lam_indices->push_back(indices);
        lam_parameters->push_back(parameters);

        vector<float> sep_accepted(int(indices.size()+1), 0.0);
        lam_sep_accepted->push_back(sep_accepted);
        vector<double> state_energies = get_state_energies(PySequence_GetItem(ensembles,l), "energy");

//
        lam_state_energies->push_back(state_energies);


        so = get_scaling_parameters(PySequence_GetItem(ensembles,l));
        lam_scales_exp->push_back(so.scales_f_exp);
        lam_scales->push_back(so.scales_f);
        lam_offsets->push_back(so.offsets);

        lam_diff_state_energies.push_back(get_state_energies(PySequence_GetItem(ensembles,l), "diff_energy"));

    }


    lam_entropy = vector<double>(lam_states->size(), 0.0);
    lam_chi2 = vector<double>(lam_states->size(), 0.0);
    lam_dchi2 = vector<double>(lam_states->size(), 0.0);

    vector<double> old_lam_chi2 = vector<double>(lam_states->size(), 0.0);

    avg_restraint_intensity = 0.0;
    _restraint_intensity = 0.0;
    eff = 0.0;
    _eff = 0.0;


    lam_populations = get_populations();

    //vector<double> reference_potentials = sum_reference_potentials(ref_potentials, lam_states[0]);
    // Set the random number generator
    //srand(time(nullptr));

    srand(getpid()); // Set the random number generator based on process ID (won't work on windows)
    //srand(0);

    int N = 5000;
    ftilde = make_shared<vector<vector<vector<double>>>>();
    for (size_t l=0; l<expanded_values->size(); ++l) {
        ftilde->push_back(get_ftilde(nstates, N, (*forward_model).at(0)));
    }

    total_steps = nsteps;

    accepted = make_shared<vector<float>>(int(expanded_values->size()), 0.0);
    attempted = make_shared<vector<float>>(int(expanded_values->size()), 0.0);
    acceptance = make_shared<vector<float>>(int(expanded_values->size()), 1.0);
    accept = make_shared<vector<int>>(int(expanded_values->size()), 0.0);

    step=0;
    exchange_attempts = 0.0;
    exchanges = 0.0;
    exchange_percentage = 0.0;

    int lam_step;
    const int _swap_every = swap_every;

    bool burning = false;

    int optimal_nreplicas = nreplicas; // store as a temporary value
    //if (find_optimal_nreplicas) { int burn_steps = burn; }

    int n_converged = 0; // count the number of times we've reached convergence
    int n_converged_steps = 5; // total number of consecutive iterations to convergence

    int stop_every = std::min({swap_every, change_Nr_every});

    if ( burn > 0 ) {
        burning = true;
        //swap_every = burn; // temporarily change
        total_steps = burn;
        stop_every = std::min({stop_every, burn});
    }

    //cout << "Number of processors: " << omp_get_num_procs() << endl;
    //cout << "Number of threads: " << omp_get_num_threads() << endl;

    /* NOTE: Progress bar */
    tqdm bar(int(expanded_values->size()));
    //bar.reset();
    //bar.set_theme_basic();
    //bar.set_theme_braille();
//    bar.set_theme_line();
//    bar.set_theme_vertical();



    pyGIL_checked = false;
    while ( pyGIL_checked == false ) {
        if (PyGILState_Check()) { pyGIL_checked = true; }
    }


    if ( verbose ) {
        std::string header = "Step  \tState \t\t\t\tIndices \t\t\tEnergy (kT) \t\tAcceptance (%)";
        std::cout << header << std::endl;
        std::cout.flush();
    }



//    pybind11::gil_scoped_release no_gil;
    PyThreadState* thread_state = PyEval_SaveThread();

    bool allowed_early_stopping = false;
//    bool allowed_early_stopping = true;
    double gradientNormThreshold = 0.0;

    int print_training_every = 100; // 1000;



//    cout << "Sampling..." << endl;
    while ( step < total_steps ) {
      #pragma omp parallel if(multiprocess) firstprivate(gri) private(lam_step, ss, sep) shared(nreplicas)
      {

        //srand(int(time(NULL)) ^ omp_get_thread_num());
        #pragma omp barrier
        {
          #pragma omp for
          for (int l=0; l<int(expanded_values->size()); ++l) {
            lam_step=0;

            bool earlyStop = false;
            while (lam_step < stop_every ) {
              if (earlyStop)
              {
                  break;
                  #pragma omp cancel for
              }

              #pragma omp atomic
              ++lam_step;
              sep = get_sep_indices_and_parameters((*lam_indices)[l], (*lam_parameters)[l], lam_rest_index[l]);


              // MCMC:{{{

              //if ( int(step+lam_step) == 1 ) {
              {
                  // for the first step, we need to get the ss data structure initialized
                  ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
                        experiments, weights, lam_states->at(l), so.scales_f_exp,
                        so.scales_f, so.offsets, sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

              }


              /* NOTE: Take a random step in parameter space */
              if ( int(step+lam_step)%attempt_move_sigma_every == 0 ) {
                  /*IMPORTANT: FIXME: do we need to return ss?? RR: 01-23-25 */
                 ss = move_sigmas(l, data_likelihood,
                    sep, ss, ref_potentials, gri, sigma_batch_size,
                    move_in_all_dim, rand_num_of_dim, lam_rest_index);
              }


              sep = get_sep_indices_and_parameters((*lam_indices)[l], (*lam_parameters)[l], lam_rest_index[l]);

              /* NOTE: Take a random step in state space */
              if ( int(step+lam_step)%attempt_move_state_every == 0 ) {
                move_states(l, data_likelihood, sep, ss, ref_potentials, gri);
              }


              /* NOTE: Take a random step in fwd model parameter space */
              if (use_fmo & ( int(step+lam_step) == 1 || int(step+lam_step)%attempt_move_fmp_every == 0 ))
              {
               (this->*move_fm_parameters)(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
              }

              /* NOTE: Take a random step in prior model prior sigma space */
              if (use_fmo & ( int(step+lam_step)%attempt_move_fm_prior_sigma_every == 0 ) & (attempt_move_fm_prior_sigma_every != total_steps) )
              {
                 move_fm_prior_sigma(l, data_likelihood, sep, ss, ref_potentials, gri, burning);

              }




///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


//              if (use_pmo & pmo_method == "adam") {
//                  // Save the current derivatives of u_biceps w.r.t theta_k
//                  ensemblePMOs[l].u_trace.push_back((*energies).at(l));
//
//                  if ( int(step+lam_step) == 1 )
//                  {
//                      double lambda = expanded_values->at(l)[0];
//                      std::tuple<std::vector<double>, std::vector<std::vector<double>>> prior_energy_model;
//                      prior_energy_model = compute_energies_and_derivatives(ensemblePMOs[l].module, ensemblePMOs[l].x, 1.0, lambda, true);
//                      //auto new_state_energies = std::get<0>(prior_energy_model);
//                      auto new_diff_state_energies = std::get<1>(prior_energy_model);
//                      for (auto& vec : new_diff_state_energies) {
//                          std::transform(vec.begin(), vec.end(), vec.begin(),
//                                         [lambda](double x) { return x * lambda; });
//                      }
//                      ensembleMOs[l].diff_state_energies = new_diff_state_energies;
//                  }
//
//                  size_t batch_size = ensemblePMOs[l].batch_size;
//                  //vector<size_t> indices = rand_vec_of_unique_ints(static_cast<size_t>(ensemblePMOs[l].parameters.size()), batch_size);
//                  vector<size_t> indices = generate_sequence(ensemblePMOs[l].parameters.size());
//                  vector<double> all_du_dtheta;
//                  for (auto p : indices)
//                  {
//                      double du_dtheta = cpp_dneglogP(lam_state_energies->at(l),
//                              ensembleMOs[l].diff_state_energies[p], logZs[l], lam_states->at(l),
//                              gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
//                              sep.sep_parameters, ss, expanded_values->at(l));
//                      all_du_dtheta.push_back(du_dtheta);
//                  }
//                  ensemblePMOs[l].du_dtheta_trace.push_back(all_du_dtheta);
//
//
//              }
//
//
//
//
//
//              if (use_pmo & pmo_method == "adam_vamp") {
//                  bool train_pi = ensemblePMOs[l].module.attr("train_pi").toBool();
//                  if (train_pi) {
//
//                  // Save the current derivatives of u_biceps w.r.t theta_k
//                  ensemblePMOs[l].u_trace.push_back((*energies).at(l));
//
//                  /* NOTE:  */
//                  load_parameters(ensemblePMOs[l].module, ensemblePMOs[l].parameters);
//
//                  if ( int(step+lam_step) == 1 )
//                  {
//                      PriorLosses prior_energy_model;
//                      prior_energy_model = compute_prior_losses_and_gradients(ensemblePMOs[l].module, forward_model->at(l), false);
//                      double lambda = expanded_values->at(l)[0];
//                      auto new_diff_state_energies = prior_energy_model.grad_prior_energies_vector;
//                      for (auto& vec : new_diff_state_energies) {
//                          std::transform(vec.begin(), vec.end(), vec.begin(),
//                                         [lambda](double x) { return x * lambda; });
//                      }
//                      ensembleMOs[l].diff_state_energies = new_diff_state_energies;
//                  }
//
//
//                  /*NOTE: Perturbation in pi parameters */
//                  string pi_training_parameter_set = ensemblePMOs[l].module.attr("pi_training_parameter_set").toStringRef();
//                  std::vector<int> pi_indices = get_parameter_indices(ensemblePMOs[l].module, pi_training_parameter_set);
//                  size_t pi_ind_size = static_cast<size_t>(pi_indices.size());
//                  vector<size_t> pi_ind = generate_sequence(pi_ind_size);
//
//                  vector<double> all_du_dtheta;
//                  for (auto p : pi_ind)
//                  {
//                      double du_dtheta = cpp_dneglogP(lam_state_energies->at(l),
//                              ensembleMOs[l].diff_state_energies[p], logZs[l], lam_states->at(l),
//                              gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
//                              sep.sep_parameters, ss, expanded_values->at(l));
//                      all_du_dtheta.push_back(du_dtheta);
//                  }
//                  ensemblePMOs[l].du_dtheta_trace.push_back(all_du_dtheta);
//                  }
//
//              }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////







//              /* NOTE: Take a random step in prior model parameter space */
//              if (use_pmo & ( int(step+lam_step) == 1 || int(step+lam_step)%attempt_move_pmp_every == 0 ))
//              {
//               //#pragma omp critical (MovePMP)
//               {
//
//                 (this->*move_pm_parameters)(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//                 //exit(1);
//
//                 if (pmo_method == "adam_vamp") {
//                     //tuple<vector<double>, vector<vector<double>>> prior_energy_model;
//                     //prior_energy_model = compute_prior_losses_and_gradients(ensemblePMOs[l].module, forward_model->at(l), false);
//                     ensembleMOs[l].prior_losses = compute_prior_losses_and_gradients(ensemblePMOs[l].module, forward_model->at(l), false);
//                     forward_model->at(l) = ensembleMOs[l].prior_losses.forward_model;
//                     //ss.mo.prior_losses = ensembleMOs[l].prior_losses;
//
//                     std::string score_method = ensemblePMOs[l].module.attr("score_method").toStringRef();
//                     auto vamp_model = compute_vamp_score_loss_and_gradients(ensemblePMOs[l].module, false, score_method.c_str());
//                     ensembleMOs[l].pm_extern_loss = std::get<0>(vamp_model);
//                     ensembleMOs[l].pm_extern_dloss = std::get<1>(vamp_model);
//
//                 }
//
//                 #pragma omp critical (PrintPMP)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                   //if (expanded_values->at(l)[0] != 0.0) {
//                   {
//
//                     //for (size_t p=0; p < ensemblePMOs[l].du_dtheta.size(); ++p) {
//                     //    cout << ensemblePMOs[l].parameter_labels[p] << ": " << ensemblePMOs[l].du_dtheta[p] << ", ";
//                     //}
//                     //cout << "\n" << endl;
//
//
//                     if (!ensembleConvergenceMetrics->at(l).empty()) {
//                       int end = ensembleConvergenceMetrics->at(l).size()-1;
//                       cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; Average Absolute Derivative: " <<
//                           ensembleConvergenceMetrics->at(l)[end].averageAbsDerivative << endl;
//                       cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; Gradient Norm: " <<
//                           ensembleConvergenceMetrics->at(l)[end].gradientNorm << endl;
//                       cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; Moving Average Loss: " <<
//                           ensembleConvergenceMetrics->at(l)[end].movingAverageLoss << endl;
//
//                       cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; pm_extern_loss = "
//                           << ensembleMOs[l].pm_extern_loss << endl;
//
//
//                       if (!ss.mo.prior_losses.grad_L_ll.empty()) {
//                           cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; L_ll = " << ss.mo.prior_losses.L_ll << endl;
//                       }
//
//
//                       cout << "(" << int(step+lam_step) << ")" << "l=" << l << "; " << "Prior Model Parameter acceptance: " <<
//                           ensemblePMOs[l].pmp_acceptance*100. << endl;
//                       cout << "p(X) = " << ss.mo.prior_losses.stationary_distribution << endl;
//                       vector<double> f_of_X;
//                       for (int j=0; j<forward_model->at(l).size(); ++j) {
//                           f_of_X.push_back(forward_model->at(l)[j][0][0]);
//                       }
//                       cout << "f(X) = " << f_of_X << endl;
//                     }
//                   }
//                 }
//                 }
//               }
//              }
//
//
//              /* NOTE: Take a random step in prior model prior sigma space */
//              if (use_pmo & ( int(step+lam_step)%attempt_move_pm_prior_sigma_every == 0 )  & (attempt_move_pm_prior_sigma_every != total_steps))
//              {
//                 move_pm_prior_sigma(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintPMPrior)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                     if (expanded_values->at(l)[0] != 0.0) {
//                       cout << "l = " << l << "; " << "pm_prior_sigma_acceptance = "
//                           << ensembleMOs[l].pm_prior_acceptance*100. << endl;
//                     }
//                 }
//                 }
//
//              }



              //int sample_batch_every = int(attempt_move_pmp_every*10);
//              int sample_batch_every = int(attempt_move_pmp_every*4);
//              if (use_pmo && (pmo_method == "adam_vamp") && ((int(step + lam_step) % sample_batch_every == 0) || (int(step+lam_step) == 1))   )
//              {
//                  bool ordered = false;
//                  ensemblePMOs[l].module.run_method("sample_batch", ordered);
//              }




//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%100 == 0 )  & (100 != total_steps))
//              {
//                  ensemblePMOs[l].module.run_method("sample_batch");
//              }


//              int switch_NN_every = 1000;
              int switch_NN_every = 100;

//              if (use_pmo && (pmo_method == "adam") && ((int(step + lam_step) % switch_NN_every == 0) || (int(step+lam_step) == 1))   )
//              {
//                  if (ensemblePMOs[l].module.find_method("check_early_stopping")) {
//                      ensemblePMOs[l].module.run_method("check_early_stopping", int(step + lam_step), int(total_steps));
//                  }
//              }
//
//
//              //if (use_pmo && (pmo_method == "adam_vamp") && (int(step + lam_step) % 1000 == 0))
//              if (use_pmo && (pmo_method == "adam_vamp") && ((int(step + lam_step) % switch_NN_every == 0) || (int(step+lam_step) == 1))   )
//              {
//                  cout << lam_populations.at(l) << endl;
//                  bool prev_train_pi = ensemblePMOs[l].module.attr("train_pi").toBool();
//                  ensemblePMOs[l].module.run_method("check_early_stopping", int(step + lam_step), int(total_steps));
//                  bool new_train_pi = ensemblePMOs[l].module.attr("train_pi").toBool();
//
//
//                  if (new_train_pi && !prev_train_pi) {
//                      PriorLosses prior_energy_model;
//                      prior_energy_model = compute_prior_losses_and_gradients(ensemblePMOs[l].module, forward_model->at(l), false);
//                      double lambda = expanded_values->at(l)[0];
//                      auto new_diff_state_energies = prior_energy_model.grad_prior_energies_vector;
//                      for (auto& vec : new_diff_state_energies) {
//                          std::transform(vec.begin(), vec.end(), vec.begin(),
//                                         [lambda](double x) { return x * lambda; });
//                      }
//                      ensembleMOs[l].diff_state_energies = new_diff_state_energies;
//                  }
//
//              }





//              /* NOTE: Take a random step in prior model prior sigma space */
//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%attempt_move_pm_extern_loss_sigma_every == 0 )  & (attempt_move_pm_extern_loss_sigma_every != total_steps))
//              {
//                 move_pm_extern_loss_sigma(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintPMExternLossPrior)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                       cout << "l = " << l << "; " << "pm_extern_loss_sigma = "
//                           << ensembleMOs[l].pm_extern_dloss_sigma << endl;
//                       cout << "l = " << l << "; " << "pm_extern_loss_sigma_acceptance = "
//                           << ensembleMOs[l].pm_extern_dloss_sigma_acceptance*100. << endl;
//
//                 }
//                 }
//              }
//
//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%attempt_move_PC_sigma_every == 0 )  & (attempt_move_PC_sigma_every != total_steps))
//              {
//                 move_pm_PC_sigma(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintPMPCPrior)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                       cout << "l = " << l << "; " << "pm_PC_sigma = "
//                           << ensembleMOs[l].pm_prob_conservation_loss_sigma << endl;
//                       cout << "l = " << l << "; " << "pm_PC_sigma_acceptance = "
//                           << ensembleMOs[l].pm_PC_sigma_acceptance*100. << endl;
//                       cout << "l = " << l << "; " << "L_PC = " << ss.mo.prior_losses.L_PC << endl;
//                 }
//                 }
//              }
//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%attempt_move_DB_sigma_every == 0 )  & (attempt_move_DB_sigma_every != total_steps))
//              {
//                 move_pm_DB_sigma(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintPMDBPrior)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                       cout << "l = " << l << "; " << "pm_DB_sigma = "
//                           << ensembleMOs[l].pm_detailed_balance_loss_sigma << endl;
//                       cout << "l = " << l << "; " << "pm_DB_sigma_acceptance = "
//                           << ensembleMOs[l].pm_DB_sigma_acceptance*100. << endl;
//                       cout << "l = " << l << "; " << "L_DB = " << ss.mo.prior_losses.L_DB << endl;
//                 }
//                 }
//              }
//
//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%attempt_move_lambda_every == 0 )  & (attempt_move_lambda_every != total_steps))
//              {
//                 move_lambda(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintLambda)
//                 {
//                 if (int(step+lam_step)%print_training_every == 0) {
//                       cout << "l = " << l << "; " << "lambda = "
//                           << expanded_values->at(l)[0] << endl;
//                       cout << "l = " << l << "; " << "pm_lambda_acceptance = "
//                           << ensembleMOs[l].pm_lambda_acceptance*100. << endl;
//                 }
//                 }
//              }
//              if (use_pmo & (pmo_method == "adam_vamp") & ( int(step+lam_step)%attempt_move_xi_every == 0 )  & (attempt_move_xi_every != total_steps))
//              {
//                 move_xi(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintLambda)
//                 {
//                 if (int(step+lam_step)%1000 == 0) {
//                       cout << "l = " << l << "; " << "xi = "
//                           << expanded_values->at(l)[1] << endl;
//                       cout << "l = " << l << "; " << "pm_xi_acceptance = "
//                           << ensembleMOs[l].pm_xi_acceptance*100. << endl;
//                 }
//                 }
//              }
//
//













              /* NOTE: Take a random step in rho space (Regularization for model complexity) */
//              if ((use_fmo || use_pmo) & ( int(step+lam_step)%attempt_move_rho_every == 0 ))
//              {
//                 move_regularization_rho(l, data_likelihood, sep, ss, ref_potentials, gri, burning);
//
//                 #pragma omp critical (PrintRho)
//                 {
//                 if (int(step+lam_step)%1000 == 0) {
//                       cout << "l = " << l << "; " << "rho = "
//                           << ensembleMOs[l].rho << "; " << "pm_prior_acceptance = "
//                           << ensembleMOs[l].reg_rho_acceptance*100. << endl;
//                 }
//                 }
//
//              }




              if ( verbose ) {
                if ( int(step+lam_step)%print_frequency == 0 ) {
                  #pragma omp critical (PrintTraj)
                  {
                    cout << stringFormat("%-i ", step+lam_step);
                    printf("\t[");
                    for (auto state : lam_states->at(l)) { printf("%d, ",state); };
                    cout << "]";
                    printf("\t\t[");
                    for (int k=0; k<int(sep.sep_indices.size()); k++) {
                        printf("[");
                        for (int j=0; j<int(sep.sep_indices[k].size()); j++) {
                            std::cout << sep.sep_indices[k][j] << ", ";
                        }
                        printf("]");
                    }
                    printf("]");
                    cout << stringFormat(" \t\t%.3f \t\t%.2g%%\t\n", double(energies->at(l)), float(100.0*acceptance->at(l)));
                    std::cout.flush();
                  }
                }
              }

              update_lam_state_counts(l);

//              if ( int(step+lam_step)%move_ftilde_every == 0 )
//              {
//                // Move tilde
//                #pragma omp critical (MoveTilde)
//                ftilde->at(l) = move_ftilde(ftilde->at(l), dftilde, model,
//                        lam_states->at(l), lam_scales->at(l), data_likelihood, energies->at(l),
//                        logZs[l], lam_state_energies->at(l), ftilde_sigma, sep, gri, ss);
//              }


              //:}}}

              if ( !burning ) {
                if ( int(step+lam_step)%write_every == 0 )
                {
                  TrajectoryData data;
                  data.l = l;
                  data.step = step+lam_step;
                  data.accept = accept->at(l);
                  data.energy = energies->at(l);
                  data.indices = lam_indices->at(l);
                  data.parameters = lam_parameters->at(l);
                  data.states = lam_states->at(l);
                  data.acceptance = acceptance->at(l);
                  data.ss = ss;
                  data.sep = sep;
                  data.expanded_values = expanded_values->at(l);
                  /* hre_traj is outside the omp critical region and is much faster */
                  hre_traj->at(l).push_back(data);

                  if (use_fmo) {
                      fmp_traj[l].push_back(ensembleFMOs[l].fwd_model_parameters);
                      fm_prior_sigma_traj[l].push_back(ensembleMOs[l].fm_prior_sigmas);
                  }




                  #pragma omp critical (TrajUpdate)
                  {
                    if (change_Nr_every==total_steps) { get_Nr_learning_rate(); }
                    // if lam_populations was a smart_ptr, this could be taken out of
                    // critical section
                    lam_populations[l] = get_lam_populations(l);
                    lam_chi2[l] = get_chi_squared(l);
                  }
                }
              }


              if ( (l == 0) && ((step+lam_step)%change_Nr_every == 0) && (step+lam_step != 0) && find_optimal_nreplicas )
                  //&& (int(step+lam_step) > write_every)
              {
                cout << "step: " << step << endl;

                #pragma omp critical (CheckOptimalReplicas)
                {
                  //update_lam_state_counts(l);
                  lam_populations[l] = get_lam_populations(l);
                  lam_chi2[l] = get_chi_squared(l);
                  if ( check_replica_convergence(l, old_lam_chi2[l], 0.05, true) )
                  {
                      n_converged += 1;
                      if ( n_converged >= n_converged_steps )
                      {
                          /* NOTE: THis works! I just need to correct the lam_chi2 and the old_lam_chi2! */
                        burning = false;
                        find_optimal_nreplicas = false;
                        optimal_nreplicas = nreplicas;
                        cout << "optimal_nreplicas = " << optimal_nreplicas << endl;
                        change_Nr_every = 0;
                      }
                  }
                  else
                  {
                    n_converged = 0;
                  }
                  old_lam_chi2[l] = lam_chi2[l];
                }
                if ( !find_optimal_nreplicas ) {
                    nreplicas = optimal_nreplicas;
                    for (int _=0; _<int(lam_states->size()); ++_)
                    {
                        if (int(lam_states->at(l).size()) != optimal_nreplicas)
                        {
                           lam_states->at(l).pop_back();
                        }
                        /* NOTE: The state counts have to be reset if this was
                         * part of a burn-in..*/
                        (*lam_state_counts)[l] = vector<int>(nstates, 0.0);
                        //if (burn > 0) {lam_state_counts->at(l) = vector<int>(nstates, 0.0);}
                    }
                    break;
                }

              }

              if ( ((step+lam_step)%change_Nr_every == 0) && (step+lam_step != 0) && (step+lam_step != total_steps) )
              {
                #pragma omp critical (getRestraintIntensity)
                {
                  vector<double> restraint_intensity = get_restraint_intensity(sep.sep_parameters, ss.sem);
                  avg_restraint_intensity = accumulate(restraint_intensity.begin(), restraint_intensity.end(), 0.0)/restraint_intensity.size();
                  //cout << "avg_restraint_intensity = " << avg_restraint_intensity << endl;

                  lam_populations[l] = get_lam_populations(l);
                  lam_entropy = get_entropy();
                  lam_chi2[l] = get_chi_squared(l);

                  ss = _update_Nr(l, data_likelihood, sep, ss, ref_potentials, gri);

                }

              }


              if ( !burning && (l == 0) && (append_data_restraint_values) )
              {
                if ( ((step+lam_step)%update_xi_every == 0) && (step+lam_step != 0) && (step+lam_step != total_steps) )
                {
                  //vector<int> indices(parameter_traces[k][n].size(), 0);
                  sep = get_sep_indices_and_parameters(lam_indices->at(l), lam_parameters->at(l), lam_rest_index[l]);
                  ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
                        experiments, weights, lam_states->at(l),
                        lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
                        sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

                  double val = neglogLikelihood(gri.data_uncertainty,
                          gri.Ndofs, logZs[l], lam_states->at(l), gri.models,
                          sep.sep_parameters, ss);

                  data_restraint_values.push_back({expanded_values->at(l)[0], expanded_values->at(l)[1], val});

                  if ( use_xi_schedule ) {
                      if ( static_cast<size_t>(xi_schedule_idx) < xi_schedule.size() ) {
                          (*expanded_values)[l][1] = xi_schedule[xi_schedule_idx];
                          xi_schedule_idx += 1;
                      }
                  }
                  else {
                      update_xi_value(l, dXi);
                  }

                  if ( data_restraint_values[int(data_restraint_values.size()-1)][1] == 0.0 ) {
                      append_data_restraint_values = false;
                  }
                }
              }

              /* NOTE: Progress bar */
              if ( verbose==false && progress==true)
              {
                if ( burning ) { bar.set_label("Burn"); }
                else { bar.set_label("MCMC"); }
                bar.progress(l, step+lam_step, total_steps);
                if ( step+lam_step == total_steps ) { bar.finish(l); }
              }

            }

          }
        }
        // barrier end
      }
      step += stop_every;

      nreplicas = int(lam_states->at(0).size());
      //if ( !burning )
      {
        replica_exchange(swap_every, nsteps, swap_sigmas, swap_forward_model,
                //energies, lam_states, lam_indices,
                swap_ftilde, same_lambda_or_xi_only, !burning);
      }
      if ( burning ) {
        if (step >= burn)
        {
            burning = false;
            step = 0;
            total_steps = nsteps;
            if (progress) { bar.close(); }
        }
        swap_every = _swap_every; // change the value back to it's original value
        stop_every = std::min({swap_every, change_Nr_every});
      }

    }
    if (progress) { bar.close(); }

    //pybind11::gil_scoped_acquire acquire_gil;
    PyEval_RestoreThread(thread_state); // Re-acquire Python GIL after loop

    //cout << "DONE prog bar" << endl;

    //auto start = std::chrono::high_resolution_clock::now();

    write_to_trajectories();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> elapsed = end - start;
    //std::cout << "Elapsed time for write_to_trajectories(): " << elapsed.count() << " ms" << std::endl;


    for (int l=0; l<int(expanded_values->size()); ++l) {
        const int data_size = hre_traj->at(l).size();
        PyObject_SetAttrString(PyList_GetItem(trajs,l), "nreplicas", PyLong_FromLong(hre_traj->at(l).at(data_size - 1).states.size()));
        PyObject* state_counts = PyObject_GetAttrString(PyList_GetItem(trajs,l), "state_counts");
        PyObject* sep_accept = PyObject_GetAttrString(PyList_GetItem(trajs,l), "sep_accept");
        /* NOTE: FIXME: TODO: */
        //cout << stringFormat("Acceptance Total: %.2g %%\n",100.0*acceptance->at(l));


        if ( verbose ) {
            printf("\n");
            cout << stringFormat("Acceptance Total: %.2g %%\n",100.0*acceptance->at(l));
            printf("Accepted [ ...Nuisance parameters..., state] %%\n");
            printf("Accepted [");
            for (auto sep_acc: lam_sep_accepted->at(l)) {
                cout << stringFormat("%.2g, ",100.0*sep_acc/attempted->at(l));
            }
            printf("] %%\n\n");
            std::cout.flush();
        }
        PyObject* total_accept_list = PyList_New(0);
        PyObject* accept_list = PyList_New(0);
        for (auto sep_acc: lam_sep_accepted->at(l)) {
            PyList_Append(accept_list, PyFloat_FromDouble(100.0*sep_acc/attempted->at(l)));
        }
        PyList_Append(total_accept_list, PyFloat_FromDouble(100.0*acceptance->at(l)));
        PyList_Append(sep_accept, accept_list); // separate acceptance
        PyList_Append(sep_accept, total_accept_list); // total acceptance

        for (int i=0; i<int(lam_state_counts->at(l).size()); ++i) {
            PyObject_SetItem(state_counts, PyLong_FromLong(i), PyLong_FromLong((*lam_state_counts)[l].at(i)));
        }

    }


    //lambda_value_list = PyList_New(0);
    //xi_value_list = PyList_New(0);
    expanded_value_list = PyList_New(0);
    data_rest_list = PyList_New(0);
    for (auto item: data_restraint_values) {
        //PyList_Append(lambda_value_list, PyFloat_FromDouble(item[0]));
        //PyList_Append(xi_value_list, PyFloat_FromDouble(item[1]));

        PyObject* temp_list = PyList_New(0);
        PyList_Append(temp_list, PyFloat_FromDouble(item[0]));
        PyList_Append(temp_list, PyFloat_FromDouble(item[1]));
        PyList_Append(expanded_value_list, temp_list);

        PyList_Append(data_rest_list, PyFloat_FromDouble(item[2]));
    }


}


/*}}}*/

// build_ref_potentials:{{{
void cppHREPosteriorSampler::build_ref_potentials() {
    /*
     *
     *
     *
     */


    // iterate through each lambda value. Each lambda has a trajectory
    for (int l=0; l<int(Py_SIZE(trajs)); ++l) {
        PyObject* trajectory_ref = PyObject_GetAttrString(PySequence_GetItem(trajs, l), "ref");
        Py_INCREF(trajectory_ref);
        int condition1,condition2,condition3;
        nrestraints = PyLong_AsLong(PyLong_FromSsize_t(PySequence_Length(PySequence_GetItem(PySequence_GetItem(ensembles,l),0))));
        for (int i=0; i<nrestraints; ++i) {
            PyObject* traj_ref = PySequence_GetItem(trajectory_ref, i);
            Py_INCREF(traj_ref);
            PyObject* R = PySequence_GetItem(PySequence_GetItem(PySequence_GetItem(ensembles,l),0),i);
            Py_INCREF(R);
            PyObject* ref = PyObject_GetAttrString(R, "ref");
            Py_INCREF(ref);
            condition1 = PyObject_RichCompareBool(ref, PyUnicode_FromString("uniform"), Py_EQ);
            condition2 = PyObject_RichCompareBool(ref, PyUnicode_FromString("exponential"), Py_EQ);
            condition3 = PyObject_RichCompareBool(ref, PyUnicode_FromString("gaussian"), Py_EQ);
            if ( condition1 ) {
                PyList_Append(traj_ref, PyUnicode_FromString("NaN"));
                //PyObject_Print(trajectory_ref, stdout, Py_PRINT_RAW);printf("\n");
            }
            else if ( condition2 ) {
                if ( PyObject_HasAttrString(R, "precomputed") ) {
                    PyObject* precomputed = PyObject_GetAttrString(R, "precomputed");
                    Py_INCREF(precomputed);
                    if ( not precomputed ) {
                        build_exp_ref_pf(PySequence_GetItem(ensembles,l), i);
                    }
                    Py_DECREF(precomputed);
                }
                else {
                    build_exp_ref(PySequence_GetItem(ensembles,l), i);
                }
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "betas"));
                //PyObject_Print(trajectory_ref, stdout, Py_PRINT_RAW);printf("\n");
            }
            else if ( condition3 ) {
                if ( PyObject_HasAttrString(R, "precomputed") ) {
                    PyObject* precomputed = PyObject_GetAttrString(R, "precomputed");
                    Py_INCREF(precomputed);
                    if ( not precomputed ) {
                        build_gaussian_ref_pf(PySequence_GetItem(ensembles,l), i);
                    }
                    Py_DECREF(precomputed);
                }
                else {
                    build_gaussian_ref(PySequence_GetItem(ensembles,l), i, PyObject_GetAttrString(R, "use_global_ref_sigma"));
                }
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "ref_mean"));
                PyList_Append(traj_ref, PyObject_GetAttrString(R, "ref_sigma"));
                //PyObject_Print(trajectory_ref, stdout, Py_PRINT_RAW);printf("\n");
            }
            else {
                throw std::invalid_argument("Please choose a reference potential of the following:\n 'uniform', 'exp', 'gaussian'");
            }
            Py_DECREF(R);Py_DECREF(ref);Py_DECREF(traj_ref);
        }
        //PyObject_Print(trajectory_ref, stdout, Py_PRINT_RAW);printf("\n");
        Py_DECREF(trajectory_ref);
    }
}

//:}}}

// Hamiltonian Replica Excahnge:{{{

void cppHREPosteriorSampler::perform_swap(int index1, int index2,
        bool swap_sigmas, bool swap_forward_model, bool swap_ftilde) {
    lam_states->at(index1).swap(lam_states->at(index2));
    iter_swap(energies->begin() + index1, energies->begin() + index2);
    if (swap_sigmas) {
        lam_indices->at(index1).swap(lam_indices->at(index2));
        lam_parameters->at(index1).swap(lam_parameters->at(index2));
    }
    if (swap_ftilde) {
        ftilde->at(index1).swap(ftilde->at(index2));
    }

    if (use_fmo)
    {
        ensembleFMOs[index1].fwd_model_parameters.swap(ensembleFMOs[index2].fwd_model_parameters);
        forward_model->at(index1).swap(forward_model->at(index2));
        lam_diff_fwd_model->at(index1).swap(lam_diff_fwd_model->at(index2));
    }
    if (!use_fmo && swap_forward_model) {
        forward_model->at(index1).swap(forward_model->at(index2));
        /* FIXME: */
        lam_state_energies->at(index1).swap(lam_state_energies->at(index2));
    }

}


void cppHREPosteriorSampler::replica_exchange(int swap_every, int nsteps,
        bool swap_sigmas, bool swap_forward_model,
        bool swap_ftilde, bool same_lambda_or_xi_only, bool save) {
    /* Separate swapping by xi values can be achieved by setting
     * same_lambda_or_xi_only==true, such that swaps can only be made
     * if pairs have xi values less than 1.0. However, if a xi value is 1.0, then it's
     * lambda value has to be 0.0.
     * On the other hand, swapping between pairs can also occur if and only if
     * pairs have xi values equal to 1.0.
     *
     *
     */

    vector<int> IND(2);
    Combinations<int> C;
    if (swap_every != nsteps) {

        if (same_lambda_or_xi_only) {
            vector<vector<int>> pairs;
            for (int i = 0; i < int(expanded_values->size() - 1); i++) {
                for (int j = i + 1; j < int(expanded_values->size()); j++) {
                    double xi1 = (*expanded_values)[i][1];
                    double xi2 = (*expanded_values)[j][1];
                    if (xi1 < 1.0 && xi2 < 1.0 && abs(xi1 - xi2) < 1.0) {
                        // if xi values are less than 1.0 and their difference is less than 1.0,
                        // then add the pair to the list of eligible pairs for swapping.
                        pairs.push_back({ i, j });
                    }
                    else if (xi1 == 1.0 && xi2 == 1.0) {
                        double lambda1 = (*expanded_values)[i][0];
                        double lambda2 = (*expanded_values)[j][0];
                        if (lambda1 == 0.0 || lambda2 == 0.0 || lambda1 == lambda2) {
                            // if xi values are equal to 1.0, then check their lambda values
                            // to see if they are eligible for swapping.
                            pairs.push_back({ i, j });
                        }
                    }
                }
            }

            while (pairs.size() != 0) {
                const int rand_pair_index = rand_vec_of_ints(static_cast<int>(pairs.size()), 1)[0];
                vector<int> pair = pairs[rand_pair_index];
                const int index1 = pair[0];
                const int index2 = pair[1];
                if ( (*expanded_values)[index1][0] > (*expanded_values)[index2][0] ) { IND = {index2, index1}; }
                else { IND = {index1, index2}; }

                // Sort the pair indices by increasing xi value
                if ( (*expanded_values)[index1][1] > (*expanded_values)[index2][1] ) { IND = {index2, index1}; }
                else { IND = {index1, index2}; }

                // See if the swap should be accepted
                double rnd = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                bool swap_accept = false;
                if (energies->at(index1) < energies->at(index2)) {
                    swap_accept = true;
                } else if (rnd < exp(energies->at(index2) - energies->at(index1))) {
                    swap_accept = true;
                }
                if (swap_accept) {
                    perform_swap(index1, index2, swap_sigmas, swap_forward_model, swap_ftilde);
                    //#pragma omp atomic
                    ++exchanges;
                    /* NOTE: if a index has been swapped, then remove that index from the pairs */
                    vector<int> locations = find_indices_of_2D_vec(pairs, pair);
                    for (int loc = locations.size() - 1; loc >= 0; loc--) {
                        pairs.erase(pairs.begin() + locations[loc]);
                    }
                }
                else {
                    pairs.erase(pairs.begin() + rand_pair_index);
                }
                //#pragma omp atomic
                ++exchange_attempts;
                exchange_percentage = 100.0 * (double)exchanges / (double)exchange_attempts;

                if (save)
                {
                /* NOTE: Store the exchange information as a list of dictionaries */
                PyObject* _energies = PyList_New(0);
                PyObject* _indices = PyList_New(0);

                for (auto Idx: IND) {
                    PyList_Append(_energies, PyFloat_FromDouble(energies->at(Idx)));
                    PyList_Append(_indices, PyLong_FromLong(Idx));
                }

                PyObject* exchange_results = PyDict_New();
                PyDict_SetItemString(exchange_results, "step", PyLong_FromLong(step));
                PyDict_SetItemString(exchange_results, "indices", _indices);
                PyDict_SetItemString(exchange_results, "energies", _energies);
                PyDict_SetItemString(exchange_results, "accepted", PyLong_FromLong(swap_accept));
                PyDict_SetItemString(exchange_results, "exchange \%", PyFloat_FromDouble(exchange_percentage));
                /* NOTE: Now store the dictionary to the list of exchanges */
                PyList_Append(exchange_info, exchange_results);
                }
            }
        }
        else {
            // original swapping code for treating all replicas the same
            vector<vector<int>> pairs = C.combinations(expanded_values->size()-1, 2);
            while ( pairs.size() != 0 ) {
                const int rand_pair_index = rand_vec_of_ints(static_cast<int>(pairs.size()), 1)[0];
                vector<int> pair = pairs[rand_pair_index];
                const int index1 = pair[0];
                const int index2 = pair[1];
                // Sort the pair indices by increasing lambda value
                if ( (*expanded_values)[index1][0] > (*expanded_values)[index2][0] ) { IND = {index2, index1}; }
                else { IND = {index1, index2}; }

                // Sort the pair indices by increasing xi value
                if ( (*expanded_values)[index1][1] > (*expanded_values)[index2][1] ) { IND = {index2, index1}; }
                else { IND = {index1, index2}; }

                // See if the swap should be accepted
                double rnd = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                bool swap_accept = false;
                if ( energies->at(IND[0]) < energies->at(IND[1]) ) { swap_accept = true; }
                else if ( rnd < exp( energies->at(IND[1]) - energies->at(IND[0]) ) ) { swap_accept = true; }
                if ( swap_accept ) {
                    perform_swap(index1, index2, swap_sigmas, swap_forward_model, swap_ftilde);
                    //#pragma omp atomic
                    ++exchanges;
                    /* NOTE: if a index has been swapped, then remove that index from the pairs */
                    vector<int> locations = find_indices_of_2D_vec(pairs, pair);
                    for (int loc=locations.size() - 1; loc >= 0; loc--)
                    {
                        pairs.erase(pairs.begin()+locations[loc]);
                    }
                }
                else {
                    pairs.erase(pairs.begin()+rand_pair_index);
                }
                //#pragma omp atomic
                ++exchange_attempts;
                exchange_percentage = 100.0*(double)exchanges / (double)exchange_attempts;
                if (save)
                {

                /* NOTE: Store the exchange information as a list of dictionaries */
                PyObject* _energies = PyList_New(0);

                for (auto Idx: IND) {
                    PyList_Append(_energies, PyFloat_FromDouble(energies->at(Idx)));
                }

                PyObject* exchange_results = PyDict_New();
                PyDict_SetItemString(exchange_results, "step", PyLong_FromLong(step));
                PyDict_SetItemString(exchange_results, "energies", _energies);
                PyDict_SetItemString(exchange_results, "accepted", PyLong_FromLong(swap_accept));
                PyDict_SetItemString(exchange_results, "exchange \%", PyFloat_FromDouble(exchange_percentage));
                /* NOTE: Now store the dictionary to the list of exchanges */
                PyList_Append(exchange_info, exchange_results);
                }

            }
        }
    }
}

/*}}}*/

// move_states:{{{
void cppHREPosteriorSampler::move_states(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri) {
    /* Takes a step in state-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     *
     */
    int ind = int(lam_indices->at(l).size()); // because state is the last element in the list

    vector<int> new_states = lam_states->at(l);
    // Copy states back into new_states
    //std::copy(lam_states->at(l).begin(), lam_states->at(l).end(), back_inserter(new_states));

    // Take a random step in state space
    new_states[rand() % lam_states->at(l).size()] = int(rand_vec_of_ints(nstates,1)[0]);

    if ( fwd_model_mixture ) {
      vector<vector<int>> _new_states(*lam_states);
      _new_states[l] = new_states;
      ss = get_mixture_sse_and_sem(forward_model, experiments, weights, _new_states,
            lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
            sep.sep_parameters, data_likelihood, fwd_model_weights[l]);
    }
    else {
      ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
            experiments, weights, new_states,
            lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
            sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    }

    gri.reference_potentials = sum_reference_potentials(ref_potentials, new_states);
    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], new_states, gri.models, sep.sep_parameters, ss, expanded_values->at(l));

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < energies->at(l) ) { accept->at(l) = 1; }
    else if ( rnd < exp( energies->at(l) - new_energy) ) { accept->at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        (*lam_states).at(l) = new_states;
        ++(*lam_sep_accepted).at(l).at(ind);
    }
}
//:}}}

// move_sigmas:{{{
SS cppHREPosteriorSampler::move_sigmas(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, const int sigma_batch_size, bool move_in_all_dim, bool rand_num_of_dim,
        vector<vector<int>> lam_rest_index)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * option for continous sigma, but gives no benefit.
     */

    SCALE_OFFSET so;
    int n_rest = forward_model->at(l).at(0).size(); //Py_SIZE(PySequence_GetItem(ensemble, 0));

    // Generate random number: -1, 0, 1
    int rand_int = get_random_int(-1, 1);

    vector<double> new_parameters = (*lam_parameters).at(l);
    vector<int> new_indices = (*lam_indices).at(l);

    unique_ptr<vector<int>> ind = make_unique<vector<int>>();

    // Make sure the index doesn't fall out of the boundry of the allowed values
    //vector<int> *vec = new vector<int>;
    unique_ptr<vector<int>> vec = make_unique<vector<int>>();
    if ( continuous_space == 0 ) {
        for (size_t k=0; k<lam_rest_index[l].size(); ++k) {
            if ( move_in_all_dim ) {
                if ( rand_num_of_dim ) {
                    vector<int> walk_ind = rand_vec_of_ints(n_rest,rand_vec_of_ints(n_rest,1)[0]);
                    // remove consecutive (adjacent) duplicates
                    auto last = std::unique(walk_ind.begin(), walk_ind.end());
                    // v now holds {1 2 1 3 4 5 4 x x x}, where 'x' is indeterminate
                    walk_ind.erase(last, walk_ind.end());
                    // sort followed by unique, to remove all duplicates
                    std::sort(walk_ind.begin(), walk_ind.end()); // {1 1 2 3 4 4 5}
                    last = std::unique(walk_ind.begin(), walk_ind.end());
                    // v now holds {1 2 3 4 5 x x}, where 'x' is indeterminate
                    walk_ind.erase(last, walk_ind.end());

                    for (int j=0; j<int(walk_ind.size()); ++j) {
                        if ( lam_rest_index[l][k] == walk_ind[j] ) { vec->push_back(k); }
                    }
                }
                else {
                    for (int k=0; k<int(lam_rest_index[l].size()); ++k) {
                        // What the fuck? Is this correct?
                        if ( lam_rest_index[l][k] == rand_vec_of_ints(n_rest,1)[0] ) { vec->push_back(k); }
                    }
                }
            }
            else if ( sigma_batch_size > 1 ) {
                vector<int> walk_ind = rand_vec_of_ints(n_rest,rand_vec_of_ints(sigma_batch_size,1)[0]);
                // remove consecutive (adjacent) duplicates
                auto last = std::unique(walk_ind.begin(), walk_ind.end());
                // v now holds {1 2 1 3 4 5 4 x x x}, where 'x' is indeterminate
                walk_ind.erase(last, walk_ind.end());
                // sort followed by unique, to remove all duplicates
                std::sort(walk_ind.begin(), walk_ind.end()); // {1 1 2 3 4 4 5}
                last = std::unique(walk_ind.begin(), walk_ind.end());
                // v now holds {1 2 3 4 5 x x}, where 'x' is indeterminate
                walk_ind.erase(last, walk_ind.end());

                for (int j=0; j<int(walk_ind.size()); ++j) {
                    if ( lam_rest_index[l][k] == walk_ind[j] ) { vec->push_back(k); }
                }
            }
            else {
                if ( lam_rest_index[l][k] == rand_vec_of_ints(n_rest,1)[0] ) { vec->push_back(k); }
            }
        }
    }

    if ( continuous_space == 0 ) {
         for (int k=0; k<int(vec->size()); ++k) {
             rand_int = get_random_int(-1, 1);
             int index = (*lam_indices).at(l)[vec->at(k)]+rand_int;
             int len = int(nuisance_parameters->at(vec->at(k)).size());
             new_indices[vec->at(k)] = py_mod(index,len);
             ind->push_back(vec->at(k));
         }
        // values e.g., [1.2122652, 0.832136160, ...]
    //    vector<double> parameters(new_indices.size());
//        vector<double> parameters(lam_indices->at(l).size());

        for (int i=0; i<int(new_indices.size()); ++i) {
            new_parameters[i] = (*nuisance_parameters).at(i).at(new_indices[i]);
        }
    }
    else {
        if (move_in_all_dim) {
            for (int i=0; i<int(lam_parameters->at(l).size()); ++i) {
                //new_parameters[i] = lam_parameters->at(l).at(i);
                double max = (*nuisance_parameters).at(i).at((*nuisance_parameters).at(i).size()-1);
                double min = (*nuisance_parameters).at(i).at(0);
                if (min != max) {
//                lam_dsigma->at(l).at(i) = 0.05*(sqrt(lam_dsigma->at(l).at(i)*nreplicas) - min);
                new_parameters[i] += lam_dsigma->at(l).at(i)*Gaussian(0.0, move_sigma_std);
                if(new_parameters[i] > max) {new_parameters[i] = 2.0 * max - new_parameters[i];}
                if(new_parameters[i] < min) {new_parameters[i] = 2.0 * min - new_parameters[i];}
                }
                ind->push_back(i);
            }
        }
        else {
            for (auto val: rand_vec_of_unique_ints(n_rest,sigma_batch_size)) {
                vec->push_back(val);
            }
            for (int k=0; k<int(vec->size()); ++k) {
                rand_int = get_random_int(-1, 1);
                int i = vec->at(k);
                ind->push_back(i);
                double max = (*nuisance_parameters).at(i).at((*nuisance_parameters).at(i).size()-1);
                double min = (*nuisance_parameters).at(i).at(0);
                if (min != max) {
                new_parameters[i] += lam_dsigma->at(l).at(i)*Gaussian(0.0, move_sigma_std);
                if(new_parameters[i] > max) {new_parameters[i] = 2.0 * max - new_parameters[i];}
                if(new_parameters[i] < min) {new_parameters[i] = 2.0 * min - new_parameters[i];}
                }
            }
        }
    }
    // Convert the list of indices and values into a list of list for each restraint
    sep = get_sep_indices_and_parameters(new_indices, new_parameters, lam_rest_index[l]);
    /* NOTE: This is where we change gamma (or a scaling parameter) */
    so = change_scaling_parameters(lam_scales->at(l), lam_offsets->at(l), sep.sep_parameters);
    if ( fwd_model_mixture ) {
      ss = get_mixture_sse_and_sem(forward_model, experiments, weights, (*lam_states),
            so.scales_f_exp, so.scales_f, so.offsets,
            sep.sep_parameters, data_likelihood, fwd_model_weights[l]);
    }
    else {
      ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
            experiments, weights, lam_states->at(l), so.scales_f_exp,
            so.scales_f, so.offsets, sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    }

    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));
    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < (*energies).at(l) ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( (*energies).at(l) - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        (*lam_indices).at(l) = new_indices;
        (*lam_parameters).at(l) = new_parameters;
        (*lam_scales_exp).at(l) = so.scales_f_exp;
        (*lam_scales).at(l) = so.scales_f;
        (*lam_offsets).at(l) = so.offsets;
        ++(*accepted).at(l);
        for (auto k: *ind) { ++(*lam_sep_accepted).at(l).at(k); }
    }
    ++(*attempted).at(l);
    if ((*attempted).at(l) == 0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (*acceptance).at(l) = float((double)(*accepted).at(l) / (double)(*attempted).at(l)); }
    return ss;

}
//:}}}

// move_fm_parameters_gaussian:{{{
/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_fm_parameters_gaussian(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * THis code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     * 1. Get derivatives from forward model parameters
     * 2. Get perturbed parameters from derivatives
     * 3. Get new forward model from perturbed parameters and compute energy
     * for the forward model with these new parameters
     * 4. See if this energy is less than previous. If so, accept this forward model and the new parameters.
     *
     */
    bool debug = 0;

    double acceptance_diff = ensembleFMOs[l].fmp_acceptance - ensembleFMOs[l].target_acceptance;
    double _scale = 1 /ensembleFMOs[l].replica_scale_factor;
    //if ( burning )
    {
    ensembleFMOs[l].epsilon = ensembleFMOs[l].base_epsilon; //* ensembleFMOs[l].replica_scale_factor; // Apply initial scaling based on replica count
    // After updating fmp_acceptance at the end of the function
    ensembleFMOs[l].epsilon *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff); // Adapt epsilon based on acceptance rate
    ensembleFMOs[l].epsilon = std::clamp(ensembleFMOs[l].epsilon, 1e-3*_scale, 1e-1*_scale);
    }
    ensembleFMOs[l].eta *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff);
    ensembleFMOs[l].eta = std::clamp(ensembleFMOs[l].eta, ensembleFMOs[l].min_eta_scale, 1e-1*nreplicas*nreplicas);
    //ensembleFMOs[l].eta = std::clamp(ensembleFMOs[l].eta, ensembleFMOs[l].min_eta_scale, 1.);



    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    // forward_model shape: (lam, nstates, nrestraints, Nd)
    // lam_diff_fwd_model shape: (nstates, nrestraints, Nd)
    //int n_rest = forward_model->at(l).at(0).size(); //Py_SIZE(PySequence_GetItem(ensemble, 0));

    // Ensure that 'l' is within the bounds of 'forward_model' and 'lam_diff_fwd_model'
    if (l < 0 || l >= static_cast<int>(forward_model->size()) || l >= static_cast<int>(lam_diff_fwd_model->size())) {
        std::cerr << "Error: Index 'l' is out of bounds." << std::endl;
        exit(1);
    }

    // Ensure that 'forward_model' and 'lam_diff_fwd_model' contain elements
    if (forward_model->at(l).empty() || lam_diff_fwd_model->at(l).empty()) {
        std::cerr << "Error: 'forward_model' or 'lam_diff_fwd_model' is empty for index 'l'." << std::endl;
        exit(1);
    }

    double old_energy = (*energies).at(l);

    ensembleFMOs[l].new_forward_model = forward_model->at(l);

    auto new_fwd_model_parameters = ensembleFMOs[l].fwd_model_parameters;
    auto new_diff_fwd_model = lam_diff_fwd_model->at(l);
    tuple<vector<double>, vector<vector<double>>> couplings;

    if (new_fwd_model_parameters.size() != ensembleFMOs[l].fwd_model_parameters.size()) {
        std::cerr << "Error: Failed to properly copy fwd_model_parameters." << std::endl;
        exit(1);
    }


    int i = get_random_int(0, ensembleFMOs[l].fwd_model_parameters.size()-1);

    /* NOTE: get perturbed parameters from derivatives */
    double updated_value;
    double perturbation;
    // Ensure proper derivative usage and adaptive step sizes in parameter updates
//    for (size_t i = 0; i < ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {

        if (debug) {cout << "dA, dB, dC = ";}
        for (size_t p=0; p < ensembleFMOs[l].fwd_model_parameters[0].size(); ++p) {
            perturbation = ensembleFMOs[l].epsilon * ensembleFMOs[l].eta * Gaussian(0.0, 1.0);
            updated_value = ensembleFMOs[l].fwd_model_parameters[i][p] + perturbation;
            new_fwd_model_parameters[i][p] = std::clamp(updated_value, ensembleFMOs[l].min_max_parameters[p][0], ensembleFMOs[l].min_max_parameters[p][1]);

            if (debug) {cout << perturbation << ", ";}
        }

//        if ( get_random_int(0, 1) == 0 ) {
//            new_fwd_model_parameters[i][1] = -new_fwd_model_parameters[i][1];
//        }

        if (debug) {
        cout << endl;
        cout << "A, B, C = " << new_fwd_model_parameters[i][0] << ", " << new_fwd_model_parameters[i][1] << ", " << new_fwd_model_parameters[i][2] << endl;
        }
    }


    /* NOTE: get new forward model from perturbed parameters and compute energy */
    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {
        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {

            couplings = cpp_get_scalar_couplings_with_derivatives(ensembleFMOs[l].phi[i][s],
                        new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], model_idx);
                        //new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], ensembleFMOs[l].fwd_model_obj);
            ensembleFMOs[l].new_forward_model[s][k] = std::get<0>( couplings );
            ensembleFMOs[l].fwd_model_derivatives[i][s] = std::get<1>(couplings);
        }
    }
    ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    //ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
    //      experiments, weights, lam_states->at(l),
    //      lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
    //      sep.sep_parameters, data_likelihood, sem_method,
    //      ensembleFMOs[l].fwd_model_parameters, ensembleFMOs[l].fmp_prior_sigmas,
    //      ensembleFMOs[l].fmp_prior_mus, ensembleFMOs[l].fmp_prior_models);



    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));
    /* NOTE: see if this energy is less than previous */

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        ensembleFMOs[l].fwd_model_parameters = std::move(new_fwd_model_parameters);
        forward_model->at(l) = ensembleFMOs[l].new_forward_model;
        lam_diff_fwd_model->at(l) = std::move(new_diff_fwd_model);

        ensembleMOs[l].fm_prior_devs = ss.mo.fm_prior_devs;
        ensembleMOs[l].fm_parameters = ensembleFMOs[l].fwd_model_parameters;

        ++ensembleFMOs[l].fmp_accepted;
    }
    ++ensembleFMOs[l].fmp_attempted;
    if (ensembleFMOs[l].fmp_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleFMOs[l].fmp_acceptance) = float((double)(ensembleFMOs[l].fmp_accepted) / (double)(ensembleFMOs[l].fmp_attempted)); }

    if (debug) { cout << "fmp_acceptance = " << ensembleFMOs[l].fmp_acceptance*100. << endl; }
    //return ensembleFMOs[l];



}
//:}}}

// move_fm_parameters_sgd:{{{
/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_fm_parameters_sgd(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * THis code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     * 1. Get derivatives from forward model parameters
     * 2. Get perturbed parameters from derivatives
     * 3. Get new forward model from perturbed parameters and compute energy
     * for the forward model with these new parameters
     * 4. See if this energy is less than previous. If so, accept this forward model and the new parameters.
     *
     */
    bool debug = 0;

    double acceptance_diff = ensembleFMOs[l].fmp_acceptance - ensembleFMOs[l].target_acceptance;
    double _scale = 1 /ensembleFMOs[l].replica_scale_factor;
    //if ( burning )
    {
    //ensembleFMOs[l].epsilon = ensembleFMOs[l].base_epsilon; //* ensembleFMOs[l].replica_scale_factor; // Apply initial scaling based on replica count
    // After updating fmp_acceptance at the end of the function
    ensembleFMOs[l].epsilon *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff); // Adapt epsilon based on acceptance rate
    ensembleFMOs[l].epsilon = std::clamp(ensembleFMOs[l].epsilon, 1e-3*_scale, 1e-1*_scale);
    }
    ensembleFMOs[l].eta *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff);

    double use_grad = burning ? 1.0 : 0.0;

    ensembleFMOs[l].eta = std::clamp(ensembleFMOs[l].eta, ensembleFMOs[l].min_eta_scale, ensembleFMOs[l].max_eta_scale);


    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    // forward_model shape: (lam, nstates, nrestraints, Nd)
    // lam_diff_fwd_model shape: (nstates, nrestraints, Nd)
    //int n_rest = forward_model->at(l).at(0).size(); //Py_SIZE(PySequence_GetItem(ensemble, 0));

    // Ensure that 'l' is within the bounds of 'forward_model' and 'lam_diff_fwd_model'
    if (l < 0 || l >= static_cast<int>(forward_model->size()) || l >= static_cast<int>(lam_diff_fwd_model->size())) {
        std::cerr << "Error: Index 'l' is out of bounds." << std::endl;
        exit(1);
    }

    // Ensure that 'forward_model' and 'lam_diff_fwd_model' contain elements
    if (forward_model->at(l).empty() || lam_diff_fwd_model->at(l).empty()) {
        std::cerr << "Error: 'forward_model' or 'lam_diff_fwd_model' is empty for index 'l'." << std::endl;
        exit(1);
    }

    double old_energy = (*energies).at(l);

    ensembleFMOs[l].new_forward_model = forward_model->at(l);

    auto new_fwd_model_parameters = ensembleFMOs[l].fwd_model_parameters;
    auto new_diff_fwd_model = lam_diff_fwd_model->at(l);
    tuple<vector<double>, vector<vector<double>>> couplings;

    if (new_fwd_model_parameters.size() != ensembleFMOs[l].fwd_model_parameters.size()) {
        std::cerr << "Error: Failed to properly copy fwd_model_parameters." << std::endl;
        exit(1);
    }


    int i = get_random_int(0, ensembleFMOs[l].fwd_model_parameters.size()-1);

    /* NOTE: get derivatives */
//    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {

        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t p=0; p<ensembleFMOs[l].fwd_model_derivatives[i][0].size(); ++p) {
            new_diff_fwd_model = lam_diff_fwd_model->at(l);
            for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {
                new_diff_fwd_model[s][k] = ensembleFMOs[l].fwd_model_derivatives[i][s][p];
            }

            ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, new_diff_fwd_model, lam_diff2_fwd_model->at(l),
                  experiments, weights, lam_states->at(l),
                  lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
                  sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

            //ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, new_diff_fwd_model, lam_diff2_fwd_model->at(l),
            //      experiments, weights, lam_states->at(l),
            //      lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
            //      sep.sep_parameters, data_likelihood, sem_method,
            //      ensembleFMOs[l].fwd_model_parameters, ensembleFMOs[l].fmp_prior_sigmas,
            //      ensembleFMOs[l].fmp_prior_mus, ensembleFMOs[l].fmp_prior_models);


            ensembleFMOs[l].du_dtheta[i][p] = cpp_dneglogP(lam_state_energies->at(l), lam_diff_state_energies[l],
                    logZs[l], lam_states->at(l), gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
                    sep.sep_parameters, ss, expanded_values->at(l));
        }
    }


    /* NOTE: get perturbed parameters from derivatives */
    // Ensure proper derivative usage and adaptive step sizes in parameter updates
//    for (size_t i = 0; i < ensembleFMOs[l].fwd_model_parameters.size(); ++i)

    {
        if (debug) {cout << "dA, dB, dC = ";}
        for (size_t p=0; p < ensembleFMOs[l].fwd_model_parameters[0].size(); ++p) {
            // Incorporate gradient information for targeted updates, including safeguards against NaN
            double gradient = clipGradient(ensembleFMOs[l].du_dtheta[i][p], ensembleFMOs[l].max_gradients[p]);

//            // Update velocity with momentum; learning_rate = epsilon; noise scaling factor = eta;
//            double new_velocity = ensembleFMOs[l].decay_rate * ensembleFMOs[l].velocity[i][p] - ensembleFMOs[l].epsilon * gradient;
//            if ( !std::isnan(new_velocity) ) { ensembleFMOs[l].velocity[i][p] = new_velocity; }
//
//            double perturbation = use_grad * ensembleFMOs[l].velocity[i][p] + ensembleFMOs[l].eta * Gaussian(0.0, 1.0);
//            double updated_value = ensembleFMOs[l].fwd_model_parameters[i][p] + perturbation;


            double perturbation = use_grad * (-ensembleFMOs[l].epsilon * gradient) + ensembleFMOs[l].eta * Gaussian(0.0, 1.0);
            new_fwd_model_parameters[i][p] += perturbation;
            new_fwd_model_parameters[i][p] = std::clamp(new_fwd_model_parameters[i][p], ensembleFMOs[l].min_max_parameters[p][0], ensembleFMOs[l].min_max_parameters[p][1]);
            if (debug) {cout << perturbation << ", ";}
        }
        if (debug) {
        cout << endl;
        cout << "A, B, C = " << new_fwd_model_parameters[i][0] << ", " << new_fwd_model_parameters[i][1] << ", " << new_fwd_model_parameters[i][2] << endl;
        }
    }


    /* NOTE: get new forward model from perturbed parameters and compute energy */
    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {
        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {

            couplings = cpp_get_scalar_couplings_with_derivatives(ensembleFMOs[l].phi[i][s],
                        new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], model_idx);
                        //new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], ensembleFMOs[l].fwd_model_obj);
            ensembleFMOs[l].new_forward_model[s][k] = std::get<0>( couplings );
            ensembleFMOs[l].fwd_model_derivatives[i][s] = std::get<1>(couplings);
        }
    }
    ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);


    //ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
    //      experiments, weights, lam_states->at(l),
    //      lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
    //      sep.sep_parameters, data_likelihood, sem_method,
    //      ensembleFMOs[l].fwd_model_parameters, ensembleFMOs[l].fmp_prior_sigmas,
    //      ensembleFMOs[l].fmp_prior_mus, ensembleFMOs[l].fmp_prior_models);


    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));
    /* NOTE: see if this energy is less than previous */

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        ensembleFMOs[l].fwd_model_parameters = std::move(new_fwd_model_parameters);
        forward_model->at(l) = ensembleFMOs[l].new_forward_model;
        lam_diff_fwd_model->at(l) = std::move(new_diff_fwd_model);

        ensembleMOs[l].fm_prior_devs = ss.mo.fm_prior_devs;
        ensembleMOs[l].fm_parameters = ensembleFMOs[l].fwd_model_parameters;



        ++ensembleFMOs[l].fmp_accepted;
    }
    ++ensembleFMOs[l].fmp_attempted;
    if (ensembleFMOs[l].fmp_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleFMOs[l].fmp_acceptance) = float((double)(ensembleFMOs[l].fmp_accepted) / (double)(ensembleFMOs[l].fmp_attempted)); }

    if (debug) { cout << "fmp_acceptance = " << ensembleFMOs[l].fmp_acceptance*100. << endl; }
    //return ensembleFMOs[l];



}
//:}}}

// move_fm_parameters_adam:{{{

double cppHREPosteriorSampler::adamUpdate(int l, double gradient, int paramIndex) {
    // Update biased first moment estimate
    ensembleFMOs[l].first_moment[paramIndex] = ensembleFMOs[l].beta1 * ensembleFMOs[l].first_moment[paramIndex] + (1 - ensembleFMOs[l].beta1) * gradient;
    // Update biased second raw moment estimate
    ensembleFMOs[l].second_moment[paramIndex] = ensembleFMOs[l].beta2 * ensembleFMOs[l].second_moment[paramIndex] + (1 - ensembleFMOs[l].beta2) * gradient * gradient;

    // Compute bias-corrected first moment estimate
    double m_hat = ensembleFMOs[l].first_moment[paramIndex] / (1 - pow(ensembleFMOs[l].beta1, ensembleFMOs[l].t));
    // Compute bias-corrected second raw moment estimate
    double v_hat = ensembleFMOs[l].second_moment[paramIndex] / (1 - pow(ensembleFMOs[l].beta2, ensembleFMOs[l].t));

    // Update parameter
    return ensembleFMOs[l].alpha * m_hat / (sqrt(v_hat) + ensembleFMOs[l].epsilon_adam);
}


/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_fm_parameters_adam(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * THis code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     * 1. Get derivatives from forward model parameters
     * 2. Get perturbed parameters from derivatives
     * 3. Get new forward model from perturbed parameters and compute energy
     * for the forward model with these new parameters
     * 4. See if this energy is less than previous. If so, accept this forward model and the new parameters.
     *
     */
    bool debug = 0;


    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    // forward_model shape: (lam, nstates, nrestraints, Nd)
    // lam_diff_fwd_model shape: (nstates, nrestraints, Nd)
    //int n_rest = forward_model->at(l).at(0).size(); //Py_SIZE(PySequence_GetItem(ensemble, 0));

    // Ensure that 'l' is within the bounds of 'forward_model' and 'lam_diff_fwd_model'
    if (l < 0 || l >= static_cast<int>(forward_model->size()) || l >= static_cast<int>(lam_diff_fwd_model->size())) {
        std::cerr << "Error: Index 'l' is out of bounds." << std::endl;
        exit(1);
    }

    // Ensure that 'forward_model' and 'lam_diff_fwd_model' contain elements
    if (forward_model->at(l).empty() || lam_diff_fwd_model->at(l).empty()) {
        std::cerr << "Error: 'forward_model' or 'lam_diff_fwd_model' is empty for index 'l'." << std::endl;
        exit(1);
    }

    double old_energy = (*energies).at(l);

    ensembleFMOs[l].new_forward_model = forward_model->at(l);

    auto new_du_dtheta = ensembleFMOs[l].du_dtheta;
    auto new_fwd_model_parameters = ensembleFMOs[l].fwd_model_parameters;
    auto new_diff_fwd_model = lam_diff_fwd_model->at(l);
    tuple<vector<double>, vector<vector<double>>> couplings;

    if (new_fwd_model_parameters.size() != ensembleFMOs[l].fwd_model_parameters.size()) {
        std::cerr << "Error: Failed to properly copy fwd_model_parameters." << std::endl;
        exit(1);
    }


    int i = get_random_int(0, ensembleFMOs[l].fwd_model_parameters.size()-1);

    /* NOTE: get derivatives */
//    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {

        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t p=0; p<ensembleFMOs[l].fwd_model_derivatives[i][0].size(); ++p) {
            new_diff_fwd_model = lam_diff_fwd_model->at(l);
            for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {
                new_diff_fwd_model[s][k] = ensembleFMOs[l].fwd_model_derivatives[i][s][p];
            }

            ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, new_diff_fwd_model, lam_diff2_fwd_model->at(l),
                  experiments, weights, lam_states->at(l),
                  lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
                  sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

            //ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, new_diff_fwd_model, lam_diff2_fwd_model->at(l),
            //      experiments, weights, lam_states->at(l),
            //      lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
            //      sep.sep_parameters, data_likelihood, sem_method,
            //      ensembleFMOs[l].fwd_model_parameters, ensembleFMOs[l].fmp_prior_sigmas,
            //      ensembleFMOs[l].fmp_prior_mus, ensembleFMOs[l].fmp_prior_models);


            new_du_dtheta[i][p] = cpp_dneglogP(lam_state_energies->at(l), lam_diff_state_energies[l],
                    logZs[l], lam_states->at(l), gri.models, ss.dfX, gri.data_uncertainty, gri.Ndofs,
                    sep.sep_parameters, ss, expanded_values->at(l));
        }
    }

    /* NOTE: get perturbed parameters from derivatives */
    double updated_value;

    // Ensure proper derivative usage and adaptive step sizes in parameter updates
//    for (size_t i = 0; i < ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {
        if (debug) {cout << "dA, dB, dC = ";}
        for (size_t p=0; p < ensembleFMOs[l].fwd_model_parameters[0].size(); ++p) {
            // Incorporate gradient information for targeted updates, including safeguards against NaN
//            double gradient = (ensembleFMOs[l].du_dtheta[i][p] != 0) ? 1.0 / ensembleFMOs[l].du_dtheta[i][p] : 0.0;
            double gradient = new_du_dtheta[i][p];
            gradient = clipGradient(gradient, ensembleFMOs[l].max_gradients[p]);

            int paramIndex = i * ensembleFMOs[l].fwd_model_parameters[0].size() + p;
            double perturbation = -adamUpdate(l, gradient, paramIndex);
            //updated_value = -adamUpdate(l, gradient, paramIndex);
//            perturbation = ensembleFMOs[l].epsilon * ensembleFMOs[l].eta * Gaussian(0.0, 1.0);

            updated_value = ensembleFMOs[l].fwd_model_parameters[i][p] + perturbation;
            new_fwd_model_parameters[i][p] = std::clamp(updated_value, ensembleFMOs[l].min_max_parameters[p][0], ensembleFMOs[l].min_max_parameters[p][1]);

            //if (debug) {cout << perturbation << ", ";}
        }

        //if ( get_random_int(0, 1) ) {
        //    new_fwd_model_parameters[i][1] = -new_fwd_model_parameters[i][1];
        //}

        if (debug) {
        cout << endl;
        cout << "A, B, C = " << new_fwd_model_parameters[i][0] << ", " << new_fwd_model_parameters[i][1] << ", " << new_fwd_model_parameters[i][2] << endl;
        }
    }
    ensembleFMOs[l].temperature = ensembleFMOs[l].max_temperature * pow(ensembleFMOs[l].cooling_rate, ensembleFMOs[l].t);
    ensembleFMOs[l].t += 1; // Increment time step
    ensembleFMOs[l].alpha *= (ensembleFMOs[l].temperature / ensembleFMOs[l].max_temperature);
//    cout << "alpha: " << ensembleFMOs[l].alpha << endl;




    /* NOTE: get new forward model from perturbed parameters and compute energy */
    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {
        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {

            couplings = cpp_get_scalar_couplings_with_derivatives(ensembleFMOs[l].phi[i][s],
                        new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], model_idx);
                        //new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], ensembleFMOs[l].fwd_model_obj);
            ensembleFMOs[l].new_forward_model[s][k] = std::get<0>( couplings );
            ensembleFMOs[l].fwd_model_derivatives[i][s] = std::get<1>(couplings);
        }
    }
    ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    //ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
    //      experiments, weights, lam_states->at(l),
    //      lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
    //      sep.sep_parameters, data_likelihood, sem_method,
    //      ensembleFMOs[l].fwd_model_parameters, ensembleFMOs[l].fmp_prior_sigmas,
    //      ensembleFMOs[l].fmp_prior_mus, ensembleFMOs[l].fmp_prior_models);



    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));
    /* NOTE: see if this energy is less than previous */

    ////////////////////////////////////////////////////////////////////////////


    // Compute gradients at old and new parameters
    double acceptance_ratio = old_energy - new_energy;
    // Loop over each parameter to compute the adjusted acceptance ratio
    for (size_t p = 0; p < ensembleFMOs[l].fwd_model_parameters.size(); ++p) {
        double delta_param = ensembleFMOs[l].fwd_model_parameters[i][p] - new_fwd_model_parameters[i][p];
        double drift_old = delta_param - 0.5 * ensembleFMOs[l].epsilon_adam * ensembleFMOs[l].du_dtheta[i][p];
        double drift_new = delta_param + 0.5 * ensembleFMOs[l].epsilon_adam * new_du_dtheta[i][p];
        acceptance_ratio += 0.5 * (drift_new * drift_new - drift_old * drift_old) /
                            (ensembleFMOs[l].epsilon_adam * ensembleFMOs[l].epsilon_adam);
    }
    acceptance_ratio = std::exp(acceptance_ratio);
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if (acceptance_ratio > rnd) { (*accept).at(l) = 1; }

    ////////////////////////////////////////////////////////////////////////////




//    ///////////////////////////////////////////////////////////////////////
//    // Accept or reject the MC move according to Metroplis criterion
//    // Generate random number
//    std::uniform_real_distribution<> dis(0, 1);
//    double rnd = dis(gen);
//    accept->at(l) = 0;
//    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
//    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        ensembleFMOs[l].fwd_model_parameters = std::move(new_fwd_model_parameters);
        forward_model->at(l) = ensembleFMOs[l].new_forward_model;
        lam_diff_fwd_model->at(l) = std::move(new_diff_fwd_model);
        ensembleFMOs[l].du_dtheta = new_du_dtheta;
        ++ensembleFMOs[l].fmp_accepted;
    }
    ++ensembleFMOs[l].fmp_attempted;
    if (ensembleFMOs[l].fmp_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleFMOs[l].fmp_acceptance) = float((double)(ensembleFMOs[l].fmp_accepted) / (double)(ensembleFMOs[l].fmp_attempted)); }

    if (debug) { cout << "fmp_acceptance = " << ensembleFMOs[l].fmp_acceptance*100. << endl; }
    //return ensembleFMOs[l];



}
//:}}}

// move_fm_parameters_uniform:{{{
/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_fm_parameters_uniform(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * THis code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     * 1. Get derivatives from forward model parameters
     * 2. Get perturbed parameters from derivatives
     * 3. Get new forward model from perturbed parameters and compute energy
     * for the forward model with these new parameters
     * 4. See if this energy is less than previous. If so, accept this forward model and the new parameters.
     *
     */
    bool debug = 0;

    double acceptance_diff = ensembleFMOs[l].fmp_acceptance - ensembleFMOs[l].target_acceptance;
    double _scale = 1 /ensembleFMOs[l].replica_scale_factor;
    //if ( burning )
    {
    ensembleFMOs[l].epsilon = ensembleFMOs[l].base_epsilon; //* ensembleFMOs[l].replica_scale_factor; // Apply initial scaling based on replica count
    // After updating fmp_acceptance at the end of the function
    ensembleFMOs[l].epsilon *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff); // Adapt epsilon based on acceptance rate
    ensembleFMOs[l].epsilon = std::clamp(ensembleFMOs[l].epsilon, 1e-3*_scale, 1e-1*_scale);
    }
    ensembleFMOs[l].eta *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff);
    ensembleFMOs[l].eta = std::clamp(ensembleFMOs[l].eta, ensembleFMOs[l].min_eta_scale, 1e-1*nreplicas*nreplicas);



    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    // forward_model shape: (lam, nstates, nrestraints, Nd)
    // lam_diff_fwd_model shape: (nstates, nrestraints, Nd)
    //int n_rest = forward_model->at(l).at(0).size(); //Py_SIZE(PySequence_GetItem(ensemble, 0));

    // Ensure that 'l' is within the bounds of 'forward_model' and 'lam_diff_fwd_model'
    if (l < 0 || l >= static_cast<int>(forward_model->size()) || l >= static_cast<int>(lam_diff_fwd_model->size())) {
        std::cerr << "Error: Index 'l' is out of bounds." << std::endl;
        exit(1);
    }

    // Ensure that 'forward_model' and 'lam_diff_fwd_model' contain elements
    if (forward_model->at(l).empty() || lam_diff_fwd_model->at(l).empty()) {
        std::cerr << "Error: 'forward_model' or 'lam_diff_fwd_model' is empty for index 'l'." << std::endl;
        exit(1);
    }

    double old_energy = (*energies).at(l);

    ensembleFMOs[l].new_forward_model = forward_model->at(l);

    auto new_fwd_model_parameters = ensembleFMOs[l].fwd_model_parameters;
    auto new_diff_fwd_model = lam_diff_fwd_model->at(l);
    tuple<vector<double>, vector<vector<double>>> couplings;

    if (new_fwd_model_parameters.size() != ensembleFMOs[l].fwd_model_parameters.size()) {
        std::cerr << "Error: Failed to properly copy fwd_model_parameters." << std::endl;
        exit(1);
    }


    int i = get_random_int(0, ensembleFMOs[l].fwd_model_parameters.size()-1);

    /* NOTE: get perturbed parameters from derivatives */
    double updated_value;
    double perturbation = 0.0;
    // Ensure proper derivative usage and adaptive step sizes in parameter updates
//    for (size_t i = 0; i < ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {

        if (debug) {cout << "dA, dB, dC = ";}
        for (size_t p=0; p < ensembleFMOs[l].fwd_model_parameters[0].size(); ++p) {
            updated_value = get_random_double(ensembleFMOs[l].min_max_parameters[p][0], ensembleFMOs[l].min_max_parameters[p][1]);
            //updated_value = ensembleFMOs[l].fwd_model_parameters[i][p] + perturbation;
            new_fwd_model_parameters[i][p] = std::clamp(updated_value, ensembleFMOs[l].min_max_parameters[p][0], ensembleFMOs[l].min_max_parameters[p][1]);

            if (debug) {cout << perturbation << ", ";}
        }

//        if ( get_random_int(0, 1) == 0 ) {
//            new_fwd_model_parameters[i][1] = -new_fwd_model_parameters[i][1];
//        }

        if (debug) {
        cout << endl;
        cout << "A, B, C = " << new_fwd_model_parameters[i][0] << ", " << new_fwd_model_parameters[i][1] << ", " << new_fwd_model_parameters[i][2] << endl;
        }
    }


    /* NOTE: get new forward model from perturbed parameters and compute energy */
    for (size_t i=0; i<ensembleFMOs[l].fwd_model_parameters.size(); ++i)
    {
        size_t k = ensembleFMOs[l].restraint_indices[i];
        for (size_t s = 0; s < ensembleFMOs[l].phi[i].size(); ++s) {

            couplings = cpp_get_scalar_couplings_with_derivatives(ensembleFMOs[l].phi[i][s],
                        new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], model_idx);
                        //new_fwd_model_parameters[i], ensembleFMOs[l].phi0[i], ensembleFMOs[l].fwd_model_obj);
            ensembleFMOs[l].new_forward_model[s][k] = std::get<0>( couplings );
            ensembleFMOs[l].fwd_model_derivatives[i][s] = std::get<1>(couplings);
        }
    }
    ss = get_sse_and_sem(ensembleFMOs[l].new_forward_model, lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);


    double new_energy = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs,
            logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));
    /* NOTE: see if this energy is less than previous */

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        (*energies).at(l) = new_energy;
        ensembleFMOs[l].fwd_model_parameters = std::move(new_fwd_model_parameters);
        forward_model->at(l) = ensembleFMOs[l].new_forward_model;
        lam_diff_fwd_model->at(l) = std::move(new_diff_fwd_model);
        ++ensembleFMOs[l].fmp_accepted;
    }
    ++ensembleFMOs[l].fmp_attempted;
    if (ensembleFMOs[l].fmp_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleFMOs[l].fmp_acceptance) = float((double)(ensembleFMOs[l].fmp_accepted) / (double)(ensembleFMOs[l].fmp_attempted)); }

    if (debug) { cout << "fmp_acceptance = " << ensembleFMOs[l].fmp_acceptance*100. << endl; }
    //return ensembleFMOs[l];



}
//:}}}

// move_fm_prior_sigma:{{{
/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_fm_prior_sigma(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * This code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     *
     */


    size_t batch_size = ensembleFMOs[l].batch_size;
    //size_t batch_size = 1;
    bool debug = 0;

    double lambda = expanded_values->at(l)[0];
    double acceptance_diff = ensembleMOs[l].fm_prior_acceptance - ensembleFMOs[l].target_acceptance;
    ensembleMOs[l].fm_eta_prior *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff);
    ensembleMOs[l].fm_eta_prior = std::clamp(ensembleMOs[l].fm_eta_prior, ensembleMOs[l].fm_min_eta_prior_scale, ensembleMOs[l].fm_max_eta_prior_scale);


    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    double old_energy = (*energies).at(l);


    ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    //vector<size_t> indices = rand_vec_of_unique_ints(static_cast<size_t>(ensembleFMOs[l].fwd_model_parameters.size()), batch_size);

    int i = get_random_int(0, ensembleFMOs[l].fwd_model_parameters.size()-1);

    vector<size_t> indices = rand_vec_of_unique_ints(static_cast<size_t>(ensembleFMOs[l].fwd_model_parameters[i].size()), batch_size);

    if ( std::all_of(ss.mo.fm_prior_models[i].begin(), ss.mo.fm_prior_models[i].end(), [](const std::string& s) { return s == "GaussianSP"; }) )
    {
        indices.clear();
        indices.push_back(0);
    }

    /* NOTE: set new parameters */
    {
        //for (size_t p=0; p < ensembleFMOs[l].fwd_model_parameters[0].size(); ++p) {
        for (auto p : indices) {
            ss.mo.fm_prior_sigmas[i][p] += ensembleMOs[l].fm_eta_prior * Gaussian(0.0, 1.0);
            ss.mo.fm_prior_sigmas[i][p] = std::clamp(ss.mo.fm_prior_sigmas[i][p], 0.5, 100.0);
            //ss.mo.fm_prior_sigmas[i][p] = std::clamp(ss.mo.fm_prior_sigmas[i][p], 0.01, 100.0);
        }
    }

    double new_energy = cpp_neglogP((*lam_state_energies).at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs, logZs[l], lam_states->at(l),
            gri.models, sep.sep_parameters, ss, expanded_values->at(l));

    /* NOTE: see if this energy is less than previous */

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        ensembleMOs[l].fm_prior_sigmas = ss.mo.fm_prior_sigmas;
        (*energies).at(l) = new_energy;
        ++ensembleMOs[l].fm_prior_accepted;
    }

    ++ensembleMOs[l].fm_prior_attempted;
    if (ensembleMOs[l].fm_prior_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleMOs[l].fm_prior_acceptance) = float((double)(ensembleMOs[l].fm_prior_accepted) / (double)(ensembleMOs[l].fm_prior_attempted)); }

    if (debug) { cout << "fm_prior_acceptance = " << ensembleMOs[l].fm_prior_acceptance*100. << endl; }


}
//:}}}

// move_regularization_rho:{{{
/* NOTE: continuous spacing / derivatives used for informative moves */
void cppHREPosteriorSampler::move_regularization_rho(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning)
{
    /* Takes a step in parameter-space and evaluate the energy.
     * Acceptance is determined by Metroplis-Hastings criterion.
     * This code uses the derivative to make informed moves
     * Here, we will perform the following steps:
     *
     */


    bool debug = 0;

    double lambda = expanded_values->at(l)[0];
    double acceptance_diff = ensembleMOs[l].reg_rho_acceptance - ensembleFMOs[l].target_acceptance;
    ensembleMOs[l].reg_rho_eta *= (1 + ensembleFMOs[l].acceptance_adjustment_factor * acceptance_diff);
    //ensembleMOs[l].reg_rho_eta = std::clamp(ensembleMOs[l].reg_rho_eta, ensembleMOs[l].fm_min_eta_prior_scale, ensembleMOs[l].fm_max_eta_prior_scale);
    ensembleMOs[l].reg_rho_eta = std::clamp(ensembleMOs[l].reg_rho_eta, 0.001, 1.0);


    /* NOTE: reference_potentials is not correctly implemented here. It will pass with no errors,
     * but reference_potentials will not work if used in fmo. */
    gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));

    double old_energy = (*energies).at(l);
    ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
          experiments, weights, lam_states->at(l),
          lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
          sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);

    {
      ss.mo.rho += ensembleMOs[l].reg_rho_eta * Gaussian(0.0, 1.0);
      ss.mo.rho = std::clamp(ss.mo.rho, 0.0, 100.0);
    }

    double new_energy = cpp_neglogP((*lam_state_energies).at(l), gri.data_uncertainty,
            gri.reference_potentials, gri.Ndofs, logZs[l], lam_states->at(l),
            gri.models, sep.sep_parameters, ss, expanded_values->at(l));

    /* NOTE: see if this energy is less than previous */

    ///////////////////////////////////////////////////////////////////////
    // Accept or reject the MC move according to Metroplis criterion
    // Generate random number
    std::uniform_real_distribution<> dis(0, 1);
    double rnd = dis(gen);
    accept->at(l) = 0;
    if ( new_energy < old_energy ) { (*accept).at(l) = 1; }
    else if ( rnd < exp( old_energy - new_energy) ) { (*accept).at(l) = 1; }
    // Update parameters based upon acceptance (Metroplis criterion)
    if ( accept->at(l) == 1 ) {
        // Save accepted moves and store information
        ensembleMOs[l].rho = ss.mo.rho;
        (*energies).at(l) = new_energy;
        ++ensembleMOs[l].reg_rho_accepted;
    }

    ++ensembleMOs[l].reg_rho_attempted;
    if (ensembleMOs[l].reg_rho_attempted == 0.0) { throw std::invalid_argument("Division by zero is not allowed."); }
    else { (ensembleMOs[l].reg_rho_acceptance) = float((double)(ensembleMOs[l].reg_rho_accepted) / (double)(ensembleMOs[l].reg_rho_attempted)); }

    if (debug) { cout << "reg_rho_acceptance = " << ensembleMOs[l].reg_rho_acceptance*100. << endl; }


}
//:}}}

/* get_restraint_intensity:{{{*/
vector<double> cppHREPosteriorSampler::get_restraint_intensity(
        vector<vector<double>> sigmaB, vector<vector<double>> sigmaSEM)
{
    /*
     *
     */

    vector<double> restraint_intensity;
    for (size_t k=0; k<sigmaSEM.size(); ++k) {  // iterate over restraints
        double sem = *std::max_element(sigmaSEM[k].begin(), sigmaSEM[k].end());
        restraint_intensity.push_back(1.0/(sigmaB[k][0]*sigmaB[k][0] + sem*sem));
    }
    return restraint_intensity;

}
/*}}}*/

/* update_Nr:{{{*/
SS cppHREPosteriorSampler::_update_Nr(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials, struct GRI gri)
{
    /* Updates the number of replicas and evaluates the energy.
     *
     */

    if ( int(lam_states->at(l).size()) < 5000 )
    {
      lam_states->at(l).push_back(rand_vec_of_ints(nstates,1)[0]);

      if ( fwd_model_mixture ) {
        ss = get_mixture_sse_and_sem(forward_model, experiments, weights, (*lam_states),
              lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
              sep.sep_parameters, data_likelihood, fwd_model_weights[l]);
      }
      else {
        ss = get_sse_and_sem(forward_model->at(l), lam_diff_fwd_model->at(l), lam_diff2_fwd_model->at(l),
              experiments, weights, lam_states->at(l),
              lam_scales_exp->at(l), lam_scales->at(l), lam_offsets->at(l),
              sep.sep_parameters, data_likelihood, sem_method, ensembleMOs[l]);


      }

      gri.reference_potentials = sum_reference_potentials(ref_potentials, lam_states->at(l));
      energies->at(l) = cpp_neglogP(lam_state_energies->at(l), gri.data_uncertainty,
              gri.reference_potentials, gri.Ndofs,
              logZs[l], lam_states->at(l), gri.models, sep.sep_parameters, ss, expanded_values->at(l));

      return ss;
    }
    else
    {
        return ss;
    }

}
/*}}}*/

// learning_Nr_rate:{{{
void cppHREPosteriorSampler::get_Nr_learning_rate()
{
    /* Determines the stopping point for the optimial number of replicas
     * by looking at the entropy and chi-squared.
     *
     */

    eff = avg_restraint_intensity;
    _eff = eff;
    _restraint_intensity = avg_restraint_intensity;
    lam_populations = get_populations();
    lam_entropy = get_entropy();

}
/*}}}*/

// get_entropy:{{{
vector<double> cppHREPosteriorSampler::get_entropy()
{
    /* Calculate Shannon's entropy
     *
     */

    vector<double> entropy;
    //vector<double> dentropy;
    for (size_t l=0; l<lam_populations.size(); ++l)
    {
      double S = 0.0;
      //double dS = 0.0;
      for (size_t i=0; i<prior_populations->size(); ++i)
      {
          double p = lam_populations.at(l).at(i);
          double p0 = prior_populations->at(i);
          S += p*log(p/p0);
          //dS += log(p/p0) + 1.0;
      }
      entropy.push_back(S);
      //dentropy.push_back(dS);
    }
    return entropy;
}
/*}}}*/

// get_chi2:{{{
vector<double> cppHREPosteriorSampler::get_chi2()
{
    return lam_chi2;
}
/*}}}*/

// get_chi_squared:{{{
double cppHREPosteriorSampler::get_chi_squared(int l)
{
    /*
     *
     *
     *
     */

    //cout << "trying to get_chi_squared..." << endl;
    double chi2 = 0.0;
//    const vector<vector<vector<double>>> model = forward_model->at(l);
    for (size_t j=0; j<forward_model->at(l)[0].size(); ++j)
    {
      for (size_t k=0; k<forward_model->at(l)[0][j].size(); ++k)
      {
        double val = 0.0;
        for (size_t i=0; i<forward_model->at(l).size(); ++i)
        {
          val += lam_populations[l][i]*forward_model->at(l)[i][j][k];
        }
        double dev = val - experiments[0][j][k];
        //chi2 += dev*dev;
        chi2 += dev*dev/forward_model->at(l)[0][j].size();
      }
    }
    //cout << "done..." << endl;
    return chi2;
}
/*}}}*/

// check_replica_convergence:{{{
bool cppHREPosteriorSampler::check_replica_convergence(int l, const double old_chi2,
        const double threshold, bool verbose)
{
    /* Determines the stopping point for the optimial number of replicas
     * by looking at the entropy and chi-squared.
     *
     */

    //lam_populations = get_populations();
    //double new_chi2 = get_chi_squared(l, model, experiments);
    //double relative_change = abs((lam_chi2[l]-new_chi2)/new_chi2);
    cout << "old_chi2 = " << old_chi2 << "; new_chi2[" << l << "] = " << lam_chi2[l] << endl;
    double relative_change = abs((old_chi2 - lam_chi2[l])/lam_chi2[l]);

    const vector<vector<vector<double>>> model = forward_model->at(l);
    if ( relative_change < threshold )
    {
        if (verbose)
        {
          cout << "convergence check: " << relative_change << " < " << threshold << endl;
        }
        return true;
    }
    else
    {
        if (verbose)
        {
            cout << "convergence check: " << relative_change << " > " << threshold << endl;
        }
        return false;
    }

}
/*}}}*/

// get_lam_populations:{{{

vector<double> cppHREPosteriorSampler::get_lam_populations(int l)
{
    const auto& counts = (*lam_state_counts)[l]; // Create a reference to avoid repeated access.
    double Z = accumulate(counts.begin(), counts.end(), 0.0);

    vector<double> populations(counts.size());
    transform(counts.begin(), counts.end(), populations.begin(), [Z](double count) {
        return count / Z;
    });

    return populations;
}

//:}}}

// get_populations:{{{
vector<vector<double>> cppHREPosteriorSampler::get_populations()
{
    const auto size = lam_state_counts->size();
    vector<vector<double>> lam_populations;
    lam_populations.reserve(size); // Reserve space upfront for efficiency.
    // Compute the populations.
    for (size_t l = 0; l < size; ++l) {
        lam_populations.push_back(get_lam_populations(l));
    }
    return lam_populations;
}

//:}}}

// update_state_counts:{{{
void cppHREPosteriorSampler::update_lam_state_counts(int l)
{
    for (auto s: lam_states->at(l)) {
        ++(*lam_state_counts)[l].at(s);
    }
}
//:}}}

// update_xi_values:{{{
void cppHREPosteriorSampler::update_xi_value(int l, double inc)
{
    /* Linearly scale (rachet) down the xi-value every N steps.
     * Used for thermodynamic integration.
     *
     */

    if ( (*expanded_values)[l][1] > 0.0) {
        if ( ( (*expanded_values)[l][1] - inc ) >= 0.0001 )
        {
          (*expanded_values)[l][1] -= inc;
        }
        else
        {
          (*expanded_values)[l][1] -= 0.99*(*expanded_values)[l][1];
        }
    }
    else { (*expanded_values)[l][1] = 0.0; }
}
//:}}}

// get_convergence_metrics:{{{
PyObject* cppHREPosteriorSampler::get_convergence_metrics() {
    PyObject* pyList = PyList_New(ensembleConvergenceMetrics->size());

    for (size_t i = 0; i < ensembleConvergenceMetrics->size(); ++i) {
        PyObject* sublist = PyList_New(ensembleConvergenceMetrics->at(i).size());

        for (size_t j = 0; j < ensembleConvergenceMetrics->at(i).size(); ++j) {
            PyObject* dict = PyDict_New();

            PyObject* avgAbsDerivative = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].averageAbsDerivative);
            PyObject* gradientNorm = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].gradientNorm);
            PyObject* movingAverageLoss = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].movingAverageLoss);

            PyObject* externAverageLoss = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].externAverageLoss);
            PyObject* externGradientNorm = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].externGradientNorm);


            //PyObject* fmpAcceptance = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].fmpAcceptance);
            //PyObject* fmpPriorAcceptance = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].fmpPriorAcceptance);


            PyObject* pmpAcceptance = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].pmpAcceptance);
            PyObject* pmpPriorAcceptance = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].pmpPriorAcceptance);


            PyObject* extern_loss_sigma = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].extern_loss_sigma);
            PyObject* detailed_balance_sigma = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].detailed_balance_sigma);
            PyObject* prob_conservation_sigma = PyFloat_FromDouble(ensembleConvergenceMetrics->at(i)[j].prob_conservation_sigma);


            vector<double> sd = ensembleConvergenceMetrics->at(i)[j].stationary_dist;
            PyObject* sdList = PyList_New(sd.size());
            for (size_t k = 0; k < sd.size(); ++k) {
                PyObject* val = PyFloat_FromDouble(sd[k]);
                PyList_SetItem(sdList, k, val); // PyList_SetItem steals a reference
            }


            PyDict_SetItemString(dict, "avg_abs_derivative", avgAbsDerivative);
            PyDict_SetItemString(dict, "gradient_norm", gradientNorm);
            PyDict_SetItemString(dict, "moving_average_loss", movingAverageLoss);
            PyDict_SetItemString(dict, "external_loss", externAverageLoss);
            PyDict_SetItemString(dict, "external_gradient_norm", externGradientNorm);
            //PyDict_SetItemString(dict, "fmp_acceptance", fmpAcceptance);
            //PyDict_SetItemString(dict, "fmp_prior_acceptance", fmpPriorAcceptance);

           PyDict_SetItemString(dict, "extern_loss_sigma", extern_loss_sigma);
           PyDict_SetItemString(dict, "detailed_balance_sigma", detailed_balance_sigma);
           PyDict_SetItemString(dict, "prob_conservation_sigma", prob_conservation_sigma);


            PyDict_SetItemString(dict, "pmp_acceptance", pmpAcceptance);
            PyDict_SetItemString(dict, "pmp_prior_acceptance", pmpPriorAcceptance);

            PyDict_SetItemString(dict, "stationary_dist", sdList);



            Py_DECREF(avgAbsDerivative);
            Py_DECREF(gradientNorm);
            Py_DECREF(movingAverageLoss);
            //Py_DECREF(fmpPriorAcceptance);

            // **Begin: Convert weighted_frames to Python list and add to dict**
            // Access the weighted_frames vector
            const std::vector<double>& weighted_frames = ensembleConvergenceMetrics->at(i)[j].weighted_frames;

            // Create a new Python list for weighted_frames
            PyObject* weightedFramesList = PyList_New(weighted_frames.size());
            if (!weightedFramesList) {
                Py_DECREF(dict);
                Py_DECREF(sublist);
                Py_DECREF(pyList);
                // Optionally, set an error message here
                return nullptr;
            }

            // Populate the Python list with the weighted_frames data
            for (size_t k = 0; k < weighted_frames.size(); ++k) {
                PyObject* num = PyFloat_FromDouble(weighted_frames[k]);
                if (!num) {
                    Py_DECREF(weightedFramesList);
                    Py_DECREF(dict);
                    Py_DECREF(sublist);
                    Py_DECREF(pyList);
                    // Optionally, set an error message here
                    return nullptr;
                }
                PyList_SetItem(weightedFramesList, k, num); // PyList_SetItem steals a reference
            }

            // Add the weighted_frames list to the dictionary
            PyDict_SetItemString(dict, "weighted_frames", weightedFramesList);

            // Decrement reference to the weightedFramesList as it's now owned by the dict
            Py_DECREF(weightedFramesList);
            // **End: Convert weighted_frames to Python list and add to dict**

            PyList_SetItem(sublist, j, dict);
        }
        PyList_SetItem(pyList, i, sublist);
    }
    return pyList;
}
// get_convergence_metrics:}}}

// get_fmp_traj:{{{
PyObject* cppHREPosteriorSampler::get_fmp_traj() {
    /* Convert the fmp_traj  = vector<vector<vector<vector<double>>>>(n_ensembles);
     * to the equivalent Python object (list of nested lists).
     */
    PyObject* pyList = PyList_New(fmp_traj.size());

    for (size_t i = 0; i < fmp_traj.size(); ++i) {
        PyObject* sublist1 = PyList_New(fmp_traj[i].size());

        for (size_t j = 0; j < fmp_traj[i].size(); ++j) {
            PyObject* sublist2 = PyList_New(fmp_traj[i][j].size());

            for (size_t k = 0; k < fmp_traj[i][j].size(); ++k) {
                PyObject* sublist3 = PyList_New(fmp_traj[i][j][k].size());
                for (size_t l = 0; l < fmp_traj[i][j][k].size(); ++l) {
                    PyObject* num = PyFloat_FromDouble(fmp_traj[i][j][k][l]);
                    PyList_SetItem(sublist3, l, num);
                }
                PyList_SetItem(sublist2, k, sublist3);
            }
            PyList_SetItem(sublist1, j, sublist2);
        }
        PyList_SetItem(pyList, i, sublist1);
    }
    return pyList;
}

//:}}}

// get_pmp_traj:{{{
PyObject* cppHREPosteriorSampler::get_pmp_traj() {
    /* Convert the pmp_traj  = vector<vector<vector<vector<double>>>>(n_ensembles);
     * to the equivalent Python object (list of nested lists).
     */
    PyObject* pyList = PyList_New(pmp_traj.size());

    for (size_t i = 0; i < pmp_traj.size(); ++i) {
        PyObject* sublist1 = PyList_New(pmp_traj[i].size());

        for (size_t j = 0; j < pmp_traj[i].size(); ++j) {
            PyObject* sublist2 = PyList_New(pmp_traj[i][j].size());

            for (size_t k = 0; k < pmp_traj[i][j].size(); ++k) {
                PyObject* num = PyFloat_FromDouble(pmp_traj[i][j][k]);
                PyList_SetItem(sublist2, k, num);
            }
            PyList_SetItem(sublist1, j, sublist2);
        }
        PyList_SetItem(pyList, i, sublist1);
    }
    return pyList;
}

//:}}}

// get_pm_prior_sigma_traj:{{{
PyObject* cppHREPosteriorSampler::get_pm_prior_sigma_traj() {
    /* Convert the fmp_traj  = vector<vector<vector<vector<double>>>>(n_ensembles);
     * to the equivalent Python object (list of nested lists).
     */
    PyObject* pyList = PyList_New(pm_prior_sigma_traj.size());

    for (size_t i = 0; i < pm_prior_sigma_traj.size(); ++i) {
        PyObject* sublist1 = PyList_New(pm_prior_sigma_traj[i].size());

        for (size_t j = 0; j < pm_prior_sigma_traj[i].size(); ++j) {
            PyObject* sublist2 = PyList_New(pm_prior_sigma_traj[i][j].size());

            for (size_t k = 0; k < pm_prior_sigma_traj[i][j].size(); ++k) {
                PyObject* num = PyFloat_FromDouble(pm_prior_sigma_traj[i][j][k]);
                PyList_SetItem(sublist2, k, num);
            }
            PyList_SetItem(sublist1, j, sublist2);
        }
        PyList_SetItem(pyList, i, sublist1);
    }
    return pyList;
}

//:}}}

// get_fm_prior_sigma_traj:{{{
PyObject* cppHREPosteriorSampler::get_fm_prior_sigma_traj() {
    /* Convert the fmp_traj  = vector<vector<vector<vector<double>>>>(n_ensembles);
     * to the equivalent Python object (list of nested lists).
     */
    PyObject* pyList = PyList_New(fm_prior_sigma_traj.size());

    for (size_t i = 0; i < fm_prior_sigma_traj.size(); ++i) {
        PyObject* sublist1 = PyList_New(fm_prior_sigma_traj[i].size());

        for (size_t j = 0; j < fm_prior_sigma_traj[i].size(); ++j) {
            PyObject* sublist2 = PyList_New(fm_prior_sigma_traj[i][j].size());

            for (size_t k = 0; k < fm_prior_sigma_traj[i][j].size(); ++k) {
                PyObject* sublist3 = PyList_New(fm_prior_sigma_traj[i][j][k].size());
                for (size_t l = 0; l < fm_prior_sigma_traj[i][j][k].size(); ++l) {
                    PyObject* num = PyFloat_FromDouble(fm_prior_sigma_traj[i][j][k][l]);
                    PyList_SetItem(sublist3, l, num);
                }
                PyList_SetItem(sublist2, k, sublist3);
            }
            PyList_SetItem(sublist1, j, sublist2);
        }
        PyList_SetItem(pyList, i, sublist1);
    }
    return pyList;
}

//:}}}

// get_traj:{{{
PyObject* cppHREPosteriorSampler::get_traj() {
    return trajs;
}
//:}}}

// get_ti_info:{{{
PyObject* cppHREPosteriorSampler::get_ti_info() {
    PyObject* info = PyList_New(0);
    //PyList_Append(info, lambda_value_list);
    //PyList_Append(info, xi_value_list);
    PyList_Append(info, expanded_value_list);
    PyList_Append(info, data_rest_list);
    return info;
}
//:}}}

// get_N_replicas:{{{
PyObject* cppHREPosteriorSampler::get_N_replicas() {
    return PyLong_FromLong(nreplicas);
}
//:}}}

// get exhcange info:{{{
PyObject* cppHREPosteriorSampler::get_exchange_info() {
    return exchange_info;

}
//:}}}

// dealloc:{{{
void cppHREPosteriorSampler::dealloc() {
    //Py_DECREF(trajs);
    if (Py_IsInitialized())
        Py_Finalize();
}
//:}}}


} // namespace PS




