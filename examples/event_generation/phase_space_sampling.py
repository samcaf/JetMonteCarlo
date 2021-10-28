from __future__ import absolute_import
import dill as pickle

# Local utilities for numerics
from jetmontecarlo.jets.jet_numerics import *
from examples.params import *

###########################################
# MC Integration
############################################
# ----------------------------------
# Labels:
# ----------------------------------
# z_cut labels for pre-critical and critical quantities
zc_labels = Z_CUTS.copy()
zc_labels.insert(0, 'z_cuts')

# Additional beta labels for subsequent quantities
beta_labels = BETAS.copy()
beta_labels.insert(0, 'betas' if COMPARE_UNGROOMED else 'betas (2nd index)')

# =====================================
# Phase Space Sampling
# =====================================
# ----------------------------------
# Setting Up Samplers
# ----------------------------------
CRIT_SAMPLERS = []
PRE_SAMPLERS = []
SUB_SAMPLERS = []

# Choosing which samples to load or to generate
use_crit = True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                    COMPARE_PRE_AND_CRIT, COMPARE_ALL]
use_precrit = COMPARE_PRE_AND_CRIT or COMPARE_ALL
use_sub = COMPARE_UNGROOMED or COMPARE_CRIT_AND_SUB or COMPARE_ALL

# ----------------------------------
# Loading Samplers
# ----------------------------------
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

# ----------------------------------
# Generating Samplers
# ----------------------------------
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

    # ----------------------------------
    # Labeling
    # ----------------------------------
    # Additional information for z_cut dependent samplers:
    zcsampdict = {'z_cuts' : Z_CUTS,
                  'epsilon' : EPSILON,
                  'sample space' : BIN_SPACE,
                  'num events' : NUM_MC_EVENTS}
    CRIT_SAMPLERS.append(zcsampdict)
    PRE_SAMPLERS.append(zcsampdict)

    # No z_cut label for subsequent samplers:
    subsampdict = {'epsilon' : EPSILON,
               'sample space' : BIN_SPACE,
               'num events' : NUM_MC_EVENTS}
    SUB_SAMPLERS.append(subsampdict)

    # ----------------------------------
    # Saving Samplers
    # ----------------------------------
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

# =====================================
# Radiators by integration
# =====================================
# Setting up radiators
CRIT_RADIATORS = []
PRE_RADIATORS = []
SUB_RADIATORS = []

# ----------------------------------
# Loading Radiators
# ----------------------------------
print()
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

# ----------------------------------
# Generating Radiators
# ----------------------------------
if not LOAD_MC_RADS and SAVE_MC_RADS:
    print("Generating radiators for Monte Carlo integration:")
    if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        for i, z_cut in enumerate(Z_CUTS):
            if MAKE_CRIT_RAD:
                print("    Generating critical radiator with cutoff z_cut="
                      +str(z_cut)+"...")
                # Critical radiators
                crit_rad_i = gen_numerical_radiator(CRIT_SAMPLERS[i], 'crit',
                                                    JET_TYPE,
                                                    obs_accuracy=OBS_ACC,
                                                    splitfn_accuracy=SPLITFN_ACC,
                                                    beta=None,
                                                    bin_space=BIN_SPACE,
                                                    epsilon=EPSILON,
                                                    fixed_coupling=FIXED_COUPLING,
                                                    save=False,
                                                    num_bins=NUM_RAD_BINS)
                CRIT_RADIATORS.append(crit_rad_i)

            # Pre-critical radiators
            pre_rad_i = None
            if COMPARE_PRE_AND_CRIT or COMPARE_ALL:
                print("    Generating pre-critical radiator with cutoff z_cut="
                      +str(z_cut)+"...")
                pre_rad_i = gen_pre_num_rad(PRE_SAMPLERS[i],
                                            CRIT_SAMPLERS[i],
                                            JET_TYPE,
                                            obs_accuracy=OBS_ACC,
                                            splitfn_accuracy=SPLITFN_ACC,
                                            bin_space=BIN_SPACE,
                                            epsilon=EPSILON,
                                            fixed_coupling=FIXED_COUPLING,
                                            num_bins=100)
            PRE_RADIATORS.append(pre_rad_i)

        sub_rad = None

        # Subsequent radiators
        if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
            for _, beta in enumerate(BETAS):
                print("    Generating critical/subsequent radiator "
                      "with beta="+str(beta)+"...")
                sub_rad = gen_crit_sub_num_rad(SUB_SAMPLERS[0],
                                               JET_TYPE,
                                               obs_accuracy=OBS_ACC,
                                               splitfn_accuracy=SPLITFN_ACC,
                                               beta=beta,
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
                                             JET_TYPE,
                                             obs_accuracy=OBS_ACC,
                                             splitfn_accuracy=SPLITFN_ACC,
                                             beta=beta,
                                             bin_space=BIN_SPACE,
                                             epsilon=EPSILON,
                                             fixed_coupling=FIXED_COUPLING,
                                             save=False,
                                             num_bins=NUM_RAD_BINS)
            SUB_RADIATORS.append(sub_rad)

    # ----------------------------------
    # Labeling
    # ----------------------------------
    # Additional information for z_cut dependent radiators:
    zcraddict = {'z_cuts' : Z_CUTS,
                 'jet type' : JET_TYPE,
                 'fixed coupling' : FIXED_COUPLING,
                 'observable accuracy' : OBS_ACC,
                 'split fn accuracy' : SPLITFN_ACC,
                 'epsilon' : EPSILON,
                 'sample space' : BIN_SPACE,
                 'num events' : NUM_MC_EVENTS,
                 'num bins' : NUM_RAD_BINS}
    CRIT_SAMPLERS.append(zcraddict)
    PRE_SAMPLERS.append(zcraddict)

    # Additional information for subsequent radiators:
    subraddict = {'betas' : BETAS,
                  'jet type' : JET_TYPE,
                  'fixed coupling' : FIXED_COUPLING,
                  'observable accuracy' : OBS_ACC,
                  'split fn accuracy' : SPLITFN_ACC,
                  'epsilon' : EPSILON,
                  'sample space' : BIN_SPACE,
                  'num events' : NUM_MC_EVENTS,
                  'num bins' : NUM_RAD_BINS}
    SUB_SAMPLERS.append(subraddict)

    # ----------------------------------
    # Saving Radiators:
    # ----------------------------------
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

# =====================================
# Splitting Functions by integration
# =====================================
# Setting up
SPLITTING_FNS = []

# ----------------------------------
# Generating Splitting Functions
# ----------------------------------
print()
if SAVE_SPLITTING_FNS and not LOAD_SPLITTING_FNS:
    print("Saving normalized splitting functions...")
    for _, z_cut in enumerate(Z_CUTS):
        print("    Generating splitting function with cutoff z_cut="
              +str(z_cut)+"...")
        # Splitting function generation
        split_fn = gen_normalized_splitting(NUM_MC_EVENTS, z_cut,
                                     jet_type=JET_TYPE, accuracy=SPLITFN_ACC,
                                     fixed_coupling=FIXED_COUPLING,
                                     num_bins=NUM_SPLITFN_BINS)

        SPLITTING_FNS.append(split_fn)

    # ----------------------------------
    # Labeling
    # ----------------------------------
    # Additional information for z_cut dependent splitting functions:
    splitdict = {'z_cuts' : Z_CUTS,
                 'jet type' : JET_TYPE,
                 'fixed coupling' : FIXED_COUPLING,
                 'accuracy' : SPLITFN_ACC,
                 'epsilon' : EPSILON,
                 'sample space' : BIN_SPACE,
                 'num events' : NUM_MC_EVENTS,
                 'num bins' : NUM_SPLITFN_BINS}
    SPLITTING_FNS.append(splitdict)

    # ----------------------------------
    # Saving Splitting Functions
    # ----------------------------------
    with open(splitfn_path, 'wb') as f:
        pickle.dump(SPLITTING_FNS, f)
    print("Saved splitting functions to "+str(splitfn_path)+".")

# ----------------------------------
# Loading Splitting Functions
# ----------------------------------
elif LOAD_SPLITTING_FNS:
    print("Loading normalized splitting functions...")
    with open(splitfn_path, 'rb') as f:
        SPLITTING_FNS = pickle.load(f)
