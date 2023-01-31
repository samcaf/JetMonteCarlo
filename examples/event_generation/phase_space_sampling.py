# Local utilities for numerics
from jetmontecarlo.jets.jet_numerics import *

# Local utilities for files
from examples.data_management import save_new_data
from examples.data_management import load_data
# from examples.data_management import load_and_interpolate

# Parameters
from examples.params import Z_CUTS
from examples.params import ALL_MC_PARAMS
from examples.params import USE_PRECRIT, USE_CRIT, USE_CRIT_SUB

# Flags for saving or loading
from examples.params import LOAD_MC_EVENTS, SAVE_MC_EVENTS
from examples.params import LOAD_MC_RADS, SAVE_MC_RADS
from examples.params import LOAD_SPLITTING_FNS
from examples.params import SAVE_SPLITTING_FNS


# =====================================
# Parameters
# =====================================
params = ALL_MC_PARAMS

# DEBUG: Fixing beta to 2 for now
beta = 2

jet_type = params['jet type']
fixed_coupling = params['fixed coupling']
obs_acc = params['observable accuracy']
splitfn_acc = params['splitting function accuracy']

num_mc_events = params['number of MC events']
epsilon = params['epsilon']
bin_space = params['bin space']

num_rad_bins = params['number of radiator bins']
num_splitfn_bins = params['number of splitting function bins']


# =====================================
# Phase Space Sampling
# =====================================
# Telling something to the audience
print("Loading Monte Carlo events and integrals." if LOAD_MC_EVENTS
      else "Generating Monte Carlo events and integrals.")

# ---------------------------------
# - - - - - - - - - - - - - - - - -
# NOTE on the structure of the code below
# - - - - - - - - - - - - - - - - -
# ---------------------------------
# First looping over all zcut values
#   Loading or saving splitting functions
#   Then considering each type of emission, and
#       Loading or saving samples and radiators

# ---------------------------------
# Looping over zcut values
# ---------------------------------
for iz, z_cut in enumerate(Z_CUTS):
    print(f"    Considering {z_cut = }.", flush=True)

    params['z_cut'] = z_cut

    # - - - - - - - - - - - - - - - - -
    # Splitting Functions
    # - - - - - - - - - - - - - - - - -
    # Loading data
    if LOAD_SPLITTING_FNS:
        splitting_function = load_data("serialized function",
                                       "splitting function",
                                       params)

    # Generating data
    else:
        split_fn = gen_normalized_splitting(num_mc_events, z_cut,
                         jet_type=jet_type, accuracy=splitfn_acc,
                         fixed_coupling=fixed_coupling,
                         num_bins=num_splitfn_bins)
        # Saving data
        if SAVE_SPLITTING_FNS:
            save_new_data(split_fn, "serialized function",
                          "splitting function", params,
                          '.pkl')


    # - - - - - - - - - - - - - - - - -
    # Critical Emissions
    # - - - - - - - - - - - - - - - - -
    if USE_CRIT:
        # - - - - - - - - - - - - - - - - -
        # Critical phase space
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_EVENTS:
            crit_sampler = load_data("montecarlo samples",
                                     "critical phase space",
                                     params)
        # Generating data
        else:
            crit_sampler = criticalSampler(bin_space, zc=z_cut,
                                           epsilon=epsilon)
            crit_sampler.generateSamples(num_mc_events)

            # Saving data
            if SAVE_MC_EVENTS:
                save_new_data(crit_sampler, "montecarlo samples",
                              "critical phase space", params,
                              ".pkl")

        # - - - - - - - - - - - - - - - - -
        # Critical radiators
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_RADS:
            critical_radiator_data = load_data("numerical integral",
                                          "critical radiator",
                                          params)
        # Generating data
        else:
            _, critical_radiator_data = gen_numerical_radiator(
                                crit_sampler, 'crit',
                                jet_type,
                                obs_accuracy=obs_acc,
                                splitfn_accuracy=splitfn_acc,
                                beta=None,
                                bin_space=bin_space,
                                fixed_coupling=fixed_coupling,
                                num_bins=num_rad_bins)

            # Saving data
            if SAVE_MC_RADS:
                save_new_data(critical_radiator_data,
                              "numerical integral", "critical radiator",
                              params, ".npz")

    # - - - - - - - - - - - - - - - - -
    # Pre-critical Emissions
    # - - - - - - - - - - - - - - - - -
    if USE_PRECRIT:
        # - - - - - - - - - - - - - - - - -
        # Pre-critical phase space
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_EVENTS:
            pre_sampler = load_data("montecarlo samples",
                                    "pre-critical phase space",
                                    params)
        # Generating data
        else:
            pre_sampler = precriticalSampler(bin_space, zc=z_cut,
                                             epsilon=epsilon)
            pre_sampler.generateSamples(num_mc_events)

            # Saving data
            if SAVE_MC_EVENTS:
                save_new_data(pre_sampler, "montecarlo samples",
                              "pre-critical phase space", params,
                              ".pkl")

        # - - - - - - - - - - - - - - - - -
        # Pre-critical radiators
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_RADS:
            precrit_radiator_data = load_data("numerical integral",
                                          "pre-critical radiator",
                                          params)
        # Generating data
        else:
            _, precrit_radiator_data = gen_numerical_radiator(
                                crit_sampler, 'crit',
                                jet_type,
                                obs_accuracy=obs_acc,
                                splitfn_accuracy=splitfn_acc,
                                beta=None,
                                bin_space=bin_space,
                                fixed_coupling=fixed_coupling,
                                num_bins=num_rad_bins)

            # Saving data
            if SAVE_MC_RADS:
                save_new_data(precrit_radiator_data,
                              "numerical integral", "pre-critical radiator",
                              params, ".npz")

    # - - - - - - - - - - - - - - - - -
    # Subsequent Emissions
    # - - - - - - - - - - - - - - - - -
    if USE_CRIT_SUB:
        # Noting that subsequent phase space is independent of z_cut
        sub_sampler_params = {key: params[key]
            for key in params.keys() - {'z_cut'}}

        # - - - - - - - - - - - - - - - - -
        # Subsequent phase space
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_EVENTS and iz == 0:
            sub_sampler = load_data("montecarlo samples",
                                    "subsequent phase space",
                                    params)
        # Generating data
        elif not LOAD_MC_EVENTS and iz == 0:
            sub_sampler = ungroomedSampler(bin_space, epsilon=epsilon)
            sub_sampler.generateSamples(num_mc_events)

            # Saving data
            if SAVE_MC_EVENTS:
                save_new_data(sub_sampler, "montecarlo samples",
                              "subsequent phase space", params,
                              ".pkl")

        # - - - - - - - - - - - - - - - - -
        # Subsequent radiators
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_RADS:
            sub_radiator_data = load_data("numerical integral",
                                          "subsequent radiator",
                                          params)
        # Generating data
        else:
            _, sub_radiator_data = gen_crit_sub_num_rad(sub_sampler,
                                           jet_type,
                                           obs_accuracy=obs_acc,
                                           splitfn_accuracy=splitfn_acc,
                                           beta=beta,
                                           epsilon=epsilon,
                                           bin_space=bin_space,
                                           fixed_coupling=fixed_coupling,
                                           num_bins=num_rad_bins)

            # Saving data
            if SAVE_MC_RADS:
                save_new_data(sub_radiator_data,
                              "numerical integral", "subsequent radiator",
                              params, ".npz")

print("Complete!")
