# =====================================
# Imports
# =====================================
# ---------------------------------
# Local utilities
# ---------------------------------
# Local utilities for phase space sampling
from jetmontecarlo.numerics.radiators.samplers import criticalSampler,\
            precriticalSampler, ungroomedSampler

# Local utilities for function generation
from jetmontecarlo.numerics.radiators.generation import *
from jetmontecarlo.numerics.splitting import\
    gen_normalized_splitting

# Local utilities for files
from file_management.data_management import save_new_data
from file_management.data_management import load_data

# Parameters
from examples.params import tab
from examples.params import ALL_MONTECARLO_PARAMS
from examples.params import PHASESPACE_PARAMS
from examples.params import RADIATOR_PARAMS
from examples.params import SPLITTINGFN_PARAMS

from examples.params import BETAS, Z_CUTS
from examples.params import USE_PRECRIT, USE_CRIT, USE_CRIT_SUB

# ---------------------------------
# Flags for saving or loading
# ---------------------------------
from examples.params import LOAD_MC_EVENTS, SAVE_MC_EVENTS
from examples.params import LOAD_MC_RADS, SAVE_MC_RADS
from examples.params import LOAD_SPLITTING_FNS
from examples.params import SAVE_SPLITTING_FNS


# =====================================
# Parameters
# =====================================
params             = ALL_MONTECARLO_PARAMS

splittingfn_params = SPLITTINGFN_PARAMS

phasespace_params  = PHASESPACE_PARAMS

radiator_params    = RADIATOR_PARAMS
del radiator_params['beta']

# Unpacking parameters
jet_type         = params['jet type']
fixed_coupling   = params['fixed coupling']
obs_acc          = params['observable accuracy']
splitfn_acc      = params['splitting function accuracy']

num_mc_events    = params['number of MC events']
epsilon          = params['epsilon']
bin_space        = params['bin space']

num_rad_bins     = params['number of radiator bins']
num_splitfn_bins = params['number of splitting function bins']


# =====================================
# Phase Space Sampling
# =====================================
# Telling something to the audience
print("Loading Monte Carlo events and integrals" if LOAD_MC_EVENTS
      else "Generating Monte Carlo events and integrals")
print()

# ---------------------------------
# - - - - - - - - - - - - - - - - -
# NOTE on the structure of the code below
# - - - - - - - - - - - - - - - - -
# ---------------------------------
# First looping over all zcut values
#   Loading or saving splitting functions
#   Then considering each type of emission, and
#       Loading or saving samples and radiators
# Then loading or saving subsequent radiators,
#  which are independent of z_cut

# ---------------------------------
# Looping over zcut values
# ---------------------------------
for iz, z_cut in enumerate(Z_CUTS):
    print(tab+f"Considering {z_cut = }", flush=True)

    splittingfn_params['z_cut'] = z_cut
    phasespace_params['z_cut']  = z_cut
    radiator_params['z_cut']    = z_cut

    # - - - - - - - - - - - - - - - - -
    # Splitting Functions
    # - - - - - - - - - - - - - - - - -
    # Loading data
    if LOAD_SPLITTING_FNS:
        print(tab+tab+"Loading splitting functions",
              flush=True)
        splitting_function = load_data("serialized function",
                                       "splitting function",
                                       splittingfn_params)

    # Generating data
    else:
        print(tab+tab+"Generating splitting functions",
              flush=True)
        split_fn = gen_normalized_splitting(num_mc_events, z_cut,
                         jet_type=jet_type, accuracy=splitfn_acc,
                         fixed_coupling=fixed_coupling,
                         num_bins=num_splitfn_bins)
        # Saving data
        if SAVE_SPLITTING_FNS:
            save_new_data(split_fn, "serialized function",
                          "splitting function", splittingfn_params,
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
            print(tab+tab+"Loading critical phase space",
                  flush=True)
            crit_sampler = load_data("montecarlo samples",
                                     "critical phase space",
                                     phasespace_params)
        # Generating data
        else:
            print(tab+tab+"Generating critical phase space",
                  flush=True)
            crit_sampler = criticalSampler(bin_space, zc=z_cut,
                                           epsilon=epsilon)
            crit_sampler.generateSamples(num_mc_events)

            # Saving data
            if SAVE_MC_EVENTS:
                save_new_data(crit_sampler, "montecarlo samples",
                              "critical phase space",
                              phasespace_params,
                              ".pkl")

        # - - - - - - - - - - - - - - - - -
        # Critical radiators
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_RADS:
            print(tab+tab+"Loading critical radiators",
                  flush=True)
            critical_radiator_data = load_data("numerical integral",
                                          "critical radiator",
                                          radiator_params)
        # Generating data
        else:
            print(tab+tab+"Generating critical radiators",
                  flush=True)
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
                              radiator_params, ".npz")

    # - - - - - - - - - - - - - - - - -
    # Pre-critical Emissions
    # - - - - - - - - - - - - - - - - -
    if USE_PRECRIT:
        # - - - - - - - - - - - - - - - - -
        # Pre-critical phase space
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_EVENTS:
            print(tab+tab+"Loading pre-critical phase space",
                  flush=True)
            pre_sampler = load_data("montecarlo samples",
                                    "pre-critical phase space",
                                    phasespace_params)
        # Generating data
        else:
            print(tab+tab+"Generating pre-critical phase space",
                  flush=True)
            pre_sampler = precriticalSampler(bin_space, zc=z_cut,
                                             epsilon=epsilon)
            pre_sampler.generateSamples(num_mc_events)

            # Saving data
            if SAVE_MC_EVENTS:
                save_new_data(pre_sampler, "montecarlo samples",
                              "pre-critical phase space",
                              phasespace_params,
                              ".pkl")

        # - - - - - - - - - - - - - - - - -
        # Pre-critical radiators
        # - - - - - - - - - - - - - - - - -
        # Loading data
        if LOAD_MC_RADS:
            print(tab+tab+"Loading pre-critical radiators",
                  flush=True)
            precrit_radiator_data = load_data("numerical integral",
                                          "pre-critical radiator",
                                          radiator_params)
        # Generating data
        else:
            print(tab+tab+"Generating pre-critical radiators",
                  flush=True)
            _, precrit_radiator_data = gen_pre_num_rad(
                                pre_sampler, crit_sampler,
                                jet_type,
                                splitfn_accuracy=splitfn_acc,
                                bin_space=bin_space,
                                fixed_coupling=fixed_coupling,
                                num_bins=num_rad_bins)

            # Saving data
            if SAVE_MC_RADS:
                save_new_data(precrit_radiator_data,
                              "numerical integral", "pre-critical radiator",
                              radiator_params, ".npz")

# ---------------------------------
# Subsequent Emissions
# ---------------------------------
# Noting that subsequent phase space and radiator is
# independent of z_cut:
del phasespace_params['z_cut']
del radiator_params['z_cut']

if USE_CRIT_SUB:
    print("\n"+tab+"Considering subsequent emissions",
          flush=True)
    # - - - - - - - - - - - - - - - - -
    # Subsequent phase space
    # - - - - - - - - - - - - - - - - -
    # Loading data
    if LOAD_MC_EVENTS:
        print(tab+tab+"Loading subsequent phase space",
              flush=True)
        sub_sampler = load_data("montecarlo samples",
                                "subsequent phase space",
                                phasespace_params)
    # Generating data
    elif not LOAD_MC_EVENTS:
        print(tab+tab+"Generating subsequent phase space",
              flush=True)
        sub_sampler = ungroomedSampler(bin_space, epsilon=epsilon)
        sub_sampler.generateSamples(num_mc_events)

        # Saving data
        if SAVE_MC_EVENTS:
            save_new_data(sub_sampler, "montecarlo samples",
                          "subsequent phase space",
                          phasespace_params,
                          ".pkl")

    # - - - - - - - - - - - - - - - - -
    # Subsequent radiators
    # - - - - - - - - - - - - - - - - -
    # Loading data
    if LOAD_MC_RADS:
        print(tab+tab+f"Loading subsequent radiators",
              flush=True)
        # Looping over beta values
        for beta in BETAS:
            print(tab+tab+tab+f"Considering {beta = }", flush=True)
            radiator_params['beta'] = beta

            sub_radiator_data = load_data("numerical integral",
                                          "subsequent radiator",
                                          radiator_params)
    # Generating data
    else:
        print(tab+tab+f"Generating subsequent radiators",
              flush=True)

        # Looping over beta values
        for beta in BETAS:
            print(tab+tab+tab+f"Considering {beta = }", flush=True)
            radiator_params['beta'] = beta

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
                              "numerical integral",
                              "subsequent radiator",
                              radiator_params, ".npz")

print("\nComplete!")
