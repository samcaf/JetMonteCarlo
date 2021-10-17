import dill as pickle
from pathlib import Path

# Local utilities for numerics
from jetmontecarlo.jets.jet_numerics import *

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Physics inputs
# ------------------------------------
FIXED_COUPLING = True
ACC = 'MLL'
if FIXED_COUPLING:
    ACC = 'LL'

# Jet and grooming parameters
Z_CUTS = [.025, .05, .1, .2]
Z_CUT = .1
BETA = 2.
BETAS = [1., 2., 3., 4.]
F_SOFT = 1./2.
JET_TYPE = 'quark'

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
EPSILON = 1e-15

BIN_SPACE = 'log'


###########################################
# Monte Carlo parameters
###########################################
USE_SHOWERS = False
USE_MC_INT = True

# ------------------------------------
# Shower events
# ------------------------------------
SHOWER_CUTOFF = MU_NP
if FIXED_COUPLING:
    SHOWER_CUTOFF = 1e-10
SHOWER_BETA = 2.

NUM_SHOWER_EVENTS = int(1e5)
LOAD_SHOWER_EVENTS = False
SAVE_SHOWER_EVENTS = True

# ------------------------------------
# MC events
# ------------------------------------
NUM_MC_EVENTS = int(1e6)

LOAD_MC_EVENTS = True
# Default False
SAVE_MC_EVENTS = True

LOAD_MC_RADS = True
# Default True
SAVE_MC_RADS = True

LOAD_SPLITTING_FNS = False
SAVE_SPLITTING_FNS = True

NUM_RAD_BINS = int(5e3)

# ------------------------------------
# Choosing which emissions to plot
# ------------------------------------
COMPARE_CRIT = True
MAKE_CRIT_RAD = True
COMPARE_CRIT_AND_SUB = True
COMPARE_PRE_AND_CRIT = True
COMPARE_ALL = True
COMPARE_UNGROOMED = True

if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
            COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
    COMPARE_UNGROOMED = False

###########################################
# Generating or Loading Events:
###########################################


# ------------------------------------
# Shower file
# ------------------------------------
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

# ------------------------------------
# MC files
# ------------------------------------
# Sampler files
critfile = 'crit_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
prefile = 'pre_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
subfile = 'sub_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)

# Radiator files
critradfile = 'crit_{}_rads_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                             NUM_MC_EVENTS,
                                                             NUM_RAD_BINS)
preradfile = 'pre_{}_rads_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                           NUM_MC_EVENTS,
                                                           NUM_RAD_BINS)
subradfile = 'sub_{}_rads_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                           NUM_MC_EVENTS,
                                                           NUM_RAD_BINS)

splitfn_file = 'split_fns_{:.0e}events_{:.0e}bins_full.pkl'.format(NUM_MC_EVENTS,
                                                              NUM_RAD_BINS)

if not FIXED_COUPLING:
    # Radiator files
    critradfile = 'crit_{}_rads_rc_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                                 NUM_MC_EVENTS,
                                                                 NUM_RAD_BINS)
    preradfile = 'pre_{}_rads_rc_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                               NUM_MC_EVENTS,
                                                               NUM_RAD_BINS)
    subradfile = 'sub_{}_rads_rc_{:.0e}events_{:.0e}bins_full.pkl'.format(BIN_SPACE,
                                                               NUM_MC_EVENTS,
                                                               NUM_RAD_BINS)

    splitfn_file = 'split_fns_rc_{:.0e}events_{:.0e}bins_full.pkl'.format(
                                                                NUM_MC_EVENTS,
                                                                NUM_RAD_BINS)

if not COMPARE_UNGROOMED:
    # If we are not comparing ungroomed emissions, we remember that
    # the subsequent emissions are actually linked to the critical
    # emissions by angular ordering
    subfile = 'crit_' + subfile
    subradfile = 'crit_' + subradfile

# ------------------------------------
# Relative paths
# ------------------------------------
# Folders
ps_folder = Path("jetmontecarlo/utils/samples/parton_showers/")
sampler_folder = Path("jetmontecarlo/utils/samples/phase_space_samplers/")
rad_folder = Path("jetmontecarlo/utils/functions/radiators/")
splitfn_folder = Path("jetmontecarlo/utils/functions/splitting_fns/")

# Parton showers
jetfile_path = ps_folder / jetfile

# Samplers
critfile_path = sampler_folder / critfile
prefile_path = sampler_folder / prefile
subfile_path = sampler_folder / subfile

# Radiators
critrad_path = rad_folder / critradfile
prerad_path = rad_folder / preradfile
subrad_path = rad_folder / subradfile

# Splitting functions
splitfn_path = splitfn_folder / splitfn_file

# ------------------------------------
# List of jets (parton shower)
# ------------------------------------
if USE_SHOWERS:
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


if USE_MC_INT:
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
    use_sub = COMPARE_UNGROOMED or COMPARE_CRIT_AND_SUB or COMPARE_ALL

    # Preparing a list of samplers
    if LOAD_MC_EVENTS:
        print("Loading Monte Carlo events...")
        if use_crit:
            print("    Loading critical events...")
            with open(critfile_path, 'rb') as f:
                CRIT_SAMPLERS = pickle.load(f)
        if use_precrit:
            print("    Loading pre-critical events...")
            with open(prefile_path, 'rb') as f:
                PRE_SAMPLERS = pickle.load(f)
        if use_sub:
            print("    Loading subsequent events...")
            with open(subfile_path, 'rb') as f:
                SUB_SAMPLERS = pickle.load(f)
        print("Monte Carlo events loaded!")
    else:
        print("Generating events for Monte Carlo integration...")
        if SAVE_MC_EVENTS:
            print("    Preparing to save new events for MC integration...")

        if use_crit:
            for _, z_cut in enumerate(Z_CUTS):
                print("    Generating critical emissions with cutoff z_cut="
                      +str(z_cut)+"...")
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
                    print("    Generating pre-critical emissions with cutoff z_cut="
                          +str(z_cut)+"...")
                    pre_sampler_i.generateSamples(NUM_MC_EVENTS)
                PRE_SAMPLERS.append(pre_sampler_i)

        # Subsequent sampler:
        sub_sampler = ungroomedSampler(BIN_SPACE, epsilon=EPSILON)
        # If we should generate subsequent samples:
        if use_sub:
            print("    Generating subsequent emissions...")
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

    if LOAD_MC_RADS:
        print("Loading radiators...")
        print("    Loading critical radiators...")
        with open(critrad_path, 'rb') as f:
            CRIT_RADIATORS = pickle.load(f)
        print("    Loading pre-critical radiators...")
        with open(prerad_path, 'rb') as f:
            PRE_RADIATORS = pickle.load(f)
        print("    Loading subsequent radiators...")
        with open(subrad_path, 'rb') as f:
            SUB_RADIATORS = pickle.load(f)
        print("Radiators loaded!")
    elif not MAKE_CRIT_RAD:
        print("Loading critical radiators...")
        with open(critrad_path, 'rb') as f:
            CRIT_RADIATORS = pickle.load(f)
        print("Radiators loaded!")

    if not LOAD_MC_RADS and SAVE_MC_RADS:
        print("Generating radiators for Monte Carlo integration:")
        if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                    COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
            for i, z_cut in enumerate(Z_CUTS):
                if MAKE_CRIT_RAD:
                    print("    Generating critical radiator with cutoff z_cut="
                          +str(z_cut)+"...")
                    # Critical sampler
                    crit_rad_i = gen_numerical_radiator(CRIT_SAMPLERS[i], 'crit',
                                                        JET_TYPE, ACC,
                                                        beta=None,
                                                        bin_space=BIN_SPACE,
                                                        epsilon=EPSILON,
                                                        fixed_coupling=FIXED_COUPLING,
                                                        save=False,
                                                        num_bins=NUM_RAD_BINS)
                    CRIT_RADIATORS.append(crit_rad_i)

                # Pre-critical sampler
                pre_rad_i = None
                if COMPARE_PRE_AND_CRIT or COMPARE_ALL:
                    print("    Generating pre-critical radiator with cutoff z_cut="
                          +str(z_cut)+"...")
                    pre_rad_i = gen_pre_num_rad(PRE_SAMPLERS[i],
                                                CRIT_SAMPLERS[i],
                                                JET_TYPE, ACC,
                                                bin_space=BIN_SPACE,
                                                epsilon=EPSILON,
                                                fixed_coupling=FIXED_COUPLING,
                                                num_bins=100)
                PRE_RADIATORS.append(pre_rad_i)

            # Subsequent sampler
            sub_rad = None
            if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
                print("    Generating critical/subsequent radiator "
                      "with beta="+str(BETA)+"...")
                sub_rad = gen_crit_sub_num_rad(SUB_SAMPLERS[0],
                                               JET_TYPE, ACC,
                                               beta=BETA,
                                               epsilon=EPSILON,
                                               bin_space=BIN_SPACE,
                                               fixed_coupling=FIXED_COUPLING,
                                               num_bins=NUM_RAD_BINS)
            SUB_RADIATORS.append(sub_rad)

        elif COMPARE_UNGROOMED:
            for _, beta in enumerate(BETAS):
                print("    Generating subsequent radiator with beta="
                      +str(beta)+"...")
                sub_rad = gen_numerical_radiator(SUB_SAMPLERS[0], 'sub',
                                                 JET_TYPE, ACC,
                                                 beta=beta,
                                                 bin_space=BIN_SPACE,
                                                 epsilon=EPSILON,
                                                 fixed_coupling=FIXED_COUPLING,
                                                 save=False,
                                                 num_bins=NUM_RAD_BINS)
                SUB_RADIATORS.append(sub_rad)
        if SAVE_MC_RADS:
            # Saving critical radiators:
            if use_crit and MAKE_CRIT_RAD:
                print("Saving critical radiator to "+str(critrad_path))
                with open(critrad_path, 'wb') as f:
                    pickle.dump(CRIT_RADIATORS, f)
                print("Saving complete!")
            # Saving pre-critical radiators:
            if use_precrit:
                print("Saving pre-crit radiator to "+str(prerad_path))
                with open(prerad_path, 'wb') as f:
                    pickle.dump(PRE_RADIATORS, f)
                print("Saving complete!")
            # Saving subsequent radiators:
            if use_sub:
                if COMPARE_UNGROOMED:
                    desc = 'sub'
                else:
                    desc = 'crit-sub'
                print("Saving "+desc+" radiator to "+str(subrad_path))
                with open(subrad_path, 'wb') as f:
                    pickle.dump(SUB_RADIATORS, f)
                print("Saving complete!")

    # ------------------------------------
    # Splitting Functions
    # ------------------------------------
    # Setting up
    SPLITTING_FNS = []

    # Preparing a list of samplers
    if SAVE_SPLITTING_FNS and not LOAD_SPLITTING_FNS:
        print("Saving normalized splitting functions...")
        for _, z_cut in enumerate(Z_CUTS):
            print("    Generating splitting function with cutoff z_cut="
                  +str(z_cut)+"...")
            # Critical samplers
            split_fn = gen_normalized_splitting(NUM_MC_EVENTS, z_cut,
                                         jet_type=JET_TYPE, accuracy=ACC,
                                         fixed_coupling=FIXED_COUPLING,
                                         num_bins=NUM_RAD_BINS)

            SPLITTING_FNS.append(split_fn)

        with open(splitfn_path, 'wb') as f:
            pickle.dump(SPLITTING_FNS, f)
        print("Saved splitting functions to "+str(splitfn_path)+".")

    elif LOAD_SPLITTING_FNS:
        print("Loading normalized splitting functions...")
        with open(splitfn_path, 'rb') as f:
            SPLITTING_FNS = pickle.load(f)
