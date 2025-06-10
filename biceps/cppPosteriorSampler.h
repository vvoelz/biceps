

/* Header file for sample.cpp */
#ifndef CPPPOSTERIORSAMPLER_H
#define CPPPOSTERIORSAMPLER_H

#include <iostream>
#include <Python.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <memory>
#include <thread>

#include "PosteriorSampler.h"
using namespace std;
using std::vector;


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

vector<vector<vector<double>>> get_forward_model(PyObject* ensembles, PyObject* traj,
        int nsnaps, int nstates, int nreplicas);

GFE get_u_kln_and_states_kn(PyObject* ensembles,
        vector<vector<vector<int>>> state_traces, vector<vector<double>> energy_traces,
        vector<vector<vector<double>>> parameter_traces, vector<vector<vector<double>>> expanded_traces,
        const vector<float> logZs,bool progress=true, bool capture_stdout=false,
        //bool scale_energies=false, bool compute_derivative=false, bool multiprocess=true);
        bool scale_energies=false, bool compute_derivative=false, bool multiprocess=true, PyObject* sampler=nullptr);


GFE get_u_kln_with_mapping_fwd_models(PyObject* ensembles,
        vector<vector<vector<int>>> state_traces, vector<vector<vector<int>>> state_mapping,
        vector<vector<double>> energy_traces,
        vector<vector<vector<double>>> parameter_traces, vector<vector<vector<double>>> expanded_traces,
        vector<vector<double>> fwd_model_weights,
        const vector<float> logZs, bool progress=true, bool capture_stdout=false,
        bool scale_energies=false, bool compute_derivative=false,
        bool fwd_model_mixture=false, bool multiprocess=true);


double get_data_restraint_energy(const int &nreplicas, const string &model, const vector<double> &sse,
        const vector<double> &sigmaSEM, const vector<double> &sigmaB,
        const double &Ndof, const vector<double> &sseB,
        const string &data_uncertainty);


class cppHREPosteriorSampler {

public:
    // class methods
    cppHREPosteriorSampler(PyObject* ensembles, PyObject* sampler, int nreplicas=1, int change_Nr_every=0,
            int write_every=100, int move_ftilde_every=1, double dftilde=0.1, double ftilde_sigma=1.0, bool scale_and_offset=false, bool verbose=false):
        ensembles(ensembles), sampler(sampler), nreplicas(nreplicas), change_Nr_every(change_Nr_every), write_every(write_every),
        move_ftilde_every(move_ftilde_every), dftilde(dftilde), ftilde_sigma(ftilde_sigma), scale_and_offset(scale_and_offset)
    {}

    void write_to_trajectory(int l, vector<int> indices,
       vector<int> states, vector<int> state_counts, double energy, int step,
       struct SEP sep, bool accept, struct SS ss
       );

    PyObject* convert_trajectory_to_python();

    void perform_swap(int index1, int index2,
        bool swap_sigmas, bool swap_forward_model, bool swap_ftilde);


    void replica_exchange(int swap_every, int nsteps,
        bool swap_sigmas, bool swap_forward_model,
        bool swap_ftilde, bool same_lambda_or_xi_only, bool save);


    void move_states(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri
        );

    SS move_sigmas(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri,
        const int sigma_batch_size,
        bool move_in_all_dim, bool rand_num_of_dim,
        vector<vector<int>> lam_rest_index//, const vector<vector<float>> nuisance_parameters
        );

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





    void move_regularization_rho(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);






    void move_fm_prior_sigma(int l, const vector<string> &data_likelihood,
        struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
        struct GRI gri, bool burning);




    void cpp_sample(const int nsteps, int swap_every, const int burn,
            bool swap_sigmas, bool swap_forward_model, const int print_frequency, const int sigma_batch_size,
            bool walk_in_all_dim,
            bool find_optimal_nreplicas, bool verbose, bool progress,
            bool multiprocess, bool capture_stdout);

    void _update_Nr(int l, const vector<string> &data_likelihood,
            struct SEP sep, struct SS ss, vector<vector<double>> ref_potentials,
            struct GRI gri);

    vector<vector<double>> get_populations();
    vector<double> get_lam_populations(int l);
    void get_Nr_learning_rate();
    //void get_Nr_learning_rate(int step);
    vector<double> get_restraint_intensity(vector<vector<double>> sigmaB, vector<vector<double>> sigmaSEM);
    void dealloc();
    //float cpp_compute_logZ();
    PyObject* get_traj();
    PyObject* get_ti_info();
    PyObject* get_exchange_info();
    PyObject* get_N_replicas();
    PyObject* get_fmp_traj();
    PyObject* get_pmp_traj();
    PyObject* get_pm_prior_sigma_traj();
    PyObject* get_fm_prior_sigma_traj();
    PyObject* get_convergence_metrics();
    void build_ref_potentials();

    vector<double> get_entropy();
    vector<double> get_chi2();

    double get_chi_squared(int l);

    void update_xi_value(int l, double dXi);

    bool check_replica_convergence(int l, const double old_chi2,
        const double threshold, bool verbose);

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
    PyObject* xi_value_list;
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
    //vector<vector<float>> *lam_dsigma = new vector<vector<float>>;
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
//    vector<vector<double>> fwd_model_parameters;
//    vector<vector<double>> phi;
    vector<vector<vector<vector<double>>>> fmp_traj;
    vector<vector<vector<double>>> pmp_traj;
    vector<vector<vector<double>>> pm_prior_sigma_traj;
    vector<vector<vector<vector<double>>>> fm_prior_sigma_traj;
//    vector<double> phi0;
//    vector<vector<double>> d_fmp;

    vector<vector<double>> lam_diff_state_energies;
    vector<double> logZs;
//    vector<vector<double>> velocity;


    vector<FMO> ensembleFMOs;
//    vector<PMO> ensemblePMOs;
    vector<MO> ensembleMOs;
    bool use_fmo;
    bool use_pmo;
    string sem_method;

    int model_idx;

};


}

#endif
