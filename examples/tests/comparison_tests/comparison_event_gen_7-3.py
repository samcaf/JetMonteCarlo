import dill as pickle
from pathlib import Path

# Local utilities for numerics
from jetmontecarlo.numerics.radiators.generation import *

# Local comparison utilities
from examples.utils.plot_comparisons import *

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
# Shower events
SHOWER_CUTOFF = MU_NP
if FIXED_COUPLING:
    SHOWER_CUTOFF = 1e-10
SHOWER_BETA = 2.

NUM_SHOWER_EVENTS = int(1e5)
LOAD_SHOWER_EVENTS = True
SAVE_SHOWER_EVENTS = True

# MC events
NUM_MC_EVENTS = int(1e7)
LOAD_MC_EVENTS = True
SAVE_MC_EVENTS = True

NUM_RAD_BINS = 1000

# Choosing which emissions to plot
COMPARE_CRIT = True
COMPARE_CRIT_AND_SUB = True
COMPARE_PRE_AND_CRIT = False
COMPARE_ALL = False

COMPARE_SUB = True

if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
            COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
    COMPARE_SUB = False

###########################################
# Generating or Loading Events:
###########################################
# ------------------------------------
# File paths
# ------------------------------------
# Shower file
if FIXED_COUPLING:
    jetfile = 'jet_list_shower_{:.0e}_fc_{:.0e}cutoff.pkl'.format(
                                                           NUM_SHOWER_EVENTS,
                                                           SHOWER_CUTOFF)
else:
    jetfile = 'jet_list_shower_{:.0e}_rc_{:.0e}cutoff.pkl'.format(
                                                           NUM_SHOWER_EVENTS,
                                                           SHOWER_CUTOFF)
if SHOWER_BETA != 1.:
    jetfile = jetfile + '_beta'+str(SHOWER_BETA)

# MC files
critfile = 'crit_samplers_{:.0e}.pkl'.format(NUM_MC_EVENTS)
prefile = 'pre_samplers_{:.0e}.pkl'.format(NUM_MC_EVENTS)
subfile = 'sub_samplers_{:.0e}.pkl'.format(NUM_MC_EVENTS)

# Radiator files
critradfile = 'crit_rads_{:.0e}events_{:.0e}bins.pkl'.format(NUM_MC_EVENTS,
                                                             NUM_RAD_BINS)
preradfile = 'pre_rads_{:.0e}events_{:.0e}bins.pkl'.format(NUM_MC_EVENTS,
                                                           NUM_RAD_BINS)
subradfile = 'sub_rads_{:.0e}events_{:.0e}bins.pkl'.format(NUM_MC_EVENTS,
                                                           NUM_RAD_BINS)

# Relative paths
# NOTE: Haven't checked this since changing output file structure
sample_folder = Path("output/montecarlo_samples/")
jetfile_path = sample_folder / "parton_showers" / jetfile
critfile_path = sample_folder / "radiators" / critfile
prefile_path = sample_folder / "radiators" / prefile
subfile_path = sample_folder / "radiators" / subfile

fn_folder = Path("output/serialized_functions/radiators/")
critrad_path = fn_folder / critradfile
prerad_path = fn_folder / preradfile
subrad_path = fn_folder / subradfile

# ------------------------------------
# List of jets (parton shower)
# ------------------------------------
# Preparing a list of ungroomed jets for general use
if LOAD_SHOWER_EVENTS:
    print("Loading parton shower events...")
    with open(jetfile_path, 'rb') as f:
        JET_LIST = pickle.load(f)
    print("Parton shower events loaded!")
else:
    if SAVE_SHOWER_EVENTS:
        print("Preparing to save new shower events...")
    JET_LIST = gen_jets(NUM_SHOWER_EVENTS, beta=SHOWER_BETA, radius=1.,
                        jet_type=JET_TYPE, acc=ACC,
                        cutoff=SHOWER_CUTOFF)
    if SAVE_SHOWER_EVENTS:
        with open(jetfile_path, 'wb') as f:
            pickle.dump(JET_LIST, f)

# ------------------------------------
# Integrator and Samplers (MC integration)
# ------------------------------------
# Setting up samplers
CRIT_SAMPLERS = []
PRE_SAMPLERS = []
SUB_SAMPLERS = []

# Choosing which samples to load or to generate
use_crit = True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                    COMPARE_PRE_AND_CRIT, COMPARE_ALL]
use_precrit = COMPARE_PRE_AND_CRIT or COMPARE_ALL
use_sub = COMPARE_SUB or COMPARE_CRIT_AND_SUB or COMPARE_ALL

# Preparing a list of samplers
if not LOAD_MC_EVENTS:
    # print("Loading events for Monte Carlo integration...")
    # # Loading critical samplers:
    # with open(critfile_path, 'rb') as f:
    #     CRIT_SAMPLERS = pickle.load(f)
    # # Loading pre-critical samplers:
    # with open(prefile_path, 'rb') as f:
    #     PRE_SAMPLERS = pickle.load(f)
    # # Loading subsequent samplers:
    # with open(subfile_path, 'rb') as f:
    #     SUB_SAMPLERS = pickle.load(f)
    # else:
    print("Generating events for Monte Carlo integration...")
    if SAVE_MC_EVENTS:
        print("    Preparing to save new events for MC integration...")

    if use_crit:
        for _, z_cut in enumerate(Z_CUTS):
            print("    Generating events with cutoff z_cut="+str(z_cut))

            # Critical samplers
            crit_sampler_i = criticalSampler(BIN_SPACE, zc=z_cut,
                                             epsilon=EPSILON)
            crit_sampler_i.generateSamples(NUM_MC_EVENTS)
            CRIT_SAMPLERS.append(crit_sampler_i)

            # Pre-critical sampler
            pre_sampler_i = precriticalSampler(BIN_SPACE, zc=z_cut,
                                               epsilon=EPSILON)

            # If we should generate pre-critical samples:
            if use_precrit:
                pre_sampler_i.generateSamples(NUM_MC_EVENTS)
            PRE_SAMPLERS.append(pre_sampler_i)

    # Subsequent sampler:
    sub_sampler = ungroomedSampler(BIN_SPACE, epsilon=EPSILON)
    # If we should generate subsequent samples:
    if use_sub:
        sub_sampler.generateSamples(NUM_MC_EVENTS)
    SUB_SAMPLERS.append(sub_sampler)

    if SAVE_MC_EVENTS:
        # Saving critical samplers:
        if use_crit:
            with open(critfile_path, 'wb') as f:
                pickle.dump(CRIT_SAMPLERS, f)
        # Saving pre-critical samplers:
        if use_precrit:
            with open(prefile_path, 'wb') as f:
                pickle.dump(PRE_SAMPLERS, f)
        # Saving subsequent sampler:
        if use_sub:
            with open(subfile_path, 'wb') as f:
                pickle.dump(SUB_SAMPLERS, f)

# ------------------------------------
# Radiators
# ------------------------------------
# Setting up radiators
CRIT_RADIATORS = []
PRE_RADIATORS = []
SUB_RADIATORS = []

if LOAD_MC_EVENTS:
    print("Loading radiators...")
    print("    Loading critical radiators:")
    with open(critrad_path, 'rb') as f:
        CRIT_RADIATORS = pickle.load(f)
    print("    Loading pre-critical radiators:")
    with open(prerad_path, 'rb') as f:
        PRE_RADIATORS = pickle.load(f)
    print("    Loading subsequent radiators:")
    with open(subrad_path, 'rb') as f:
        SUB_RADIATORS = pickle.load(f)
    print("Radiators loaded!")
else:
    print("Generating radiators for Monte Carlo integration:")
    if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        for i, z_cut in enumerate(Z_CUTS):
            # print("    Generating radiators with cutoff epsilon={:.0e}"
            #       .format(epsilon))
            print("    Generating critical radiator with cutoff z_cut="
                  +str(z_cut))

            # Critical sampler
            crit_rad_i = gen_numerical_radiator(CRIT_SAMPLERS[i], 'crit',
                                                JET_TYPE, ACC,
                                                beta=None,
                                                bin_space=BIN_SPACE,
                                                epsilon=EPSILON,
                                                fixed_coupling=FIXED_COUPLING,
                                                num_bins=NUM_RAD_BINS)
            CRIT_RADIATORS.append(crit_rad_i)

            # Pre-critical sampler
            pre_rad_i = None
            if COMPARE_PRE_AND_CRIT or COMPARE_ALL:
                pass
            PRE_RADIATORS.append(pre_rad_i)

    # Subsequent sampler
    sub_rad = None
    if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
        # Critical sampler
        sub_rad = gen_numerical_radiator(SUB_SAMPLERS[0], 'sub',
                                         JET_TYPE, ACC,
                                         beta=BETA,
                                         bin_space=BIN_SPACE,
                                         epsilon=EPSILON,
                                         fixed_coupling=FIXED_COUPLING,
                                         num_bins=NUM_RAD_BINS)
        SUB_RADIATORS.append(sub_rad)

    elif COMPARE_SUB:
        for _, beta in enumerate(BETAS):
            print("    Generating subsequent radiator with beta="
                  +str(beta))
            sub_rad = gen_numerical_radiator(SUB_SAMPLERS[0], 'sub',
                                             JET_TYPE, ACC,
                                             beta=beta,
                                             bin_space=BIN_SPACE,
                                             epsilon=EPSILON,
                                             fixed_coupling=FIXED_COUPLING,
                                             num_bins=NUM_RAD_BINS)
            SUB_RADIATORS.append(sub_rad)

    if SAVE_MC_EVENTS:
        # Saving critical samplers:
        if use_crit:
            with open(critrad_path, 'wb') as f:
                pickle.dump(CRIT_RADIATORS, f)
        # Saving pre-critical samplers:
        if use_precrit:
            with open(prerad_path, 'wb') as f:
                pickle.dump(PRE_RADIATORS, f)
        # Saving subsequent sampler:
        if use_sub:
            with open(subrad_path, 'wb') as f:
                pickle.dump(SUB_RADIATORS, f)
