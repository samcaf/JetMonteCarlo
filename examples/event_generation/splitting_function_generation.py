# =====================================
# Imports
# =====================================
# ---------------------------------
# Local utilities
# ---------------------------------
# Local utilities for function generation
from jetmontecarlo.numerics.splitting import\
    gen_normalized_splitting

# Local utilities for files
from file_management.data_management import save_new_data
from file_management.data_management import load_data

# Parameters
from examples.params import tab
from examples.params import SPLITTINGFN_PARAMS

# ---------------------------------
# Flags for saving or loading
# ---------------------------------
from examples.params import LOAD_SPLITTING_FNS
from examples.params import SAVE_SPLITTING_FNS


# =====================================
# Parameters
# =====================================
splittingfn_params = SPLITTINGFN_PARAMS

# Unpacking parameters
jet_type         = splittingfn_params['jet type']
fixed_coupling   = splittingfn_params['fixed coupling']
splitfn_acc      = splittingfn_params['splitting function accuracy']
num_mc_events    = splittingfn_params['number of MC events']
num_splitfn_bins = splittingfn_params['number of splitting function bins']


# =====================================
# Phase Space Sampling
# =====================================
# Telling something to the audience
print("Loading Monte Carlo splitting functions" if LOAD_SPLITTING_FNS
      else "Generating Monte Carlo splitting functions")
print()

# ---------------------------------
# Looping over zcut values
# ---------------------------------
for _, z_cut in enumerate(Z_CUTS):
    print(tab+f"Considering {z_cut = }", flush=True)

    splittingfn_params['z_cut'] = z_cut

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

print("\nComplete!")
