from __future__ import absolute_import
import dill as pickle

# Local utilities for numerics
from jetmontecarlo.jets.jet_numerics import *
from examples.params import *
from examples.filenames import *

###########################################
# MC Integration
############################################
# =====================================
# Phase Space Sampling
# =====================================
# Choosing which samples to load or to generate
use_crit = True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                    COMPARE_PRE_AND_CRIT, COMPARE_ALL]
use_precrit = COMPARE_PRE_AND_CRIT or COMPARE_ALL
use_sub = COMPARE_RAW or COMPARE_CRIT_AND_SUB or COMPARE_ALL

# ----------------------------------
# Loading Samplers
# ----------------------------------
if LOAD_MC_EVENTS:
    print("Loading Monte Carlo events...")
    if use_crit:
        print("    Loading critical events...", flush=True)
        with open(critfile_path, 'rb') as file:
            CRIT_SAMPLERS = pickle.load(file)
    if use_precrit:
        print("    Loading pre-critical events...", flush=True)
        with open(prefile_path, 'rb') as file:
            PRE_SAMPLERS = pickle.load(file)
    if use_sub:
        print("    Loading subsequent events...", flush=True)
        with open(subfile_path, 'rb') as file:
            SUB_SAMPLERS = pickle.load(file)
    print("Monte Carlo events loaded!", flush=True)

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
                  +str(z_cut)+"...", flush=True)
            # Critical samplers
            crit_sampler = criticalSampler(BIN_SPACE, zc=z_cut,
                                           epsilon=EPSILON)
            crit_sampler.generateSamples(NUM_MC_EVENTS)
            CRIT_SAMPLERS.append(crit_sampler)

            # Pre-critical sampler
            pre_sampler = precriticalSampler(BIN_SPACE, zc=z_cut,
                                             epsilon=EPSILON)

            # If we should generate pre-critical samples:
            if use_precrit:
                print("    Generating pre-critical emissions with cutoff z_cut="
                      +str(z_cut)+"...", flush=True)
                pre_sampler.generateSamples(NUM_MC_EVENTS)
            PRE_SAMPLERS.append(pre_sampler)

    # Subsequent sampler:
    sub_sampler = ungroomedSampler(BIN_SPACE, epsilon=EPSILON)
    # If we should generate subsequent samples:
    if use_sub:
        print("    Generating subsequent emissions...", flush=True)
        sub_sampler.generateSamples(NUM_MC_EVENTS)
    SUB_SAMPLERS.append(sub_sampler)

    # ----------------------------------
    # Labeling
    # ----------------------------------
    # Additional information for z_cut dependent samplers:
    # DEBUG: Removing dicts because they play poorly with dill
    # zcsampdict = {'z_cuts' : Z_CUTS,
    #               'epsilon' : EPSILON,
    #               'sample space' : BIN_SPACE,
    #               'num events' : NUM_MC_EVENTS}
    # CRIT_SAMPLERS.append(zcsampdict)
    # PRE_SAMPLERS.append(zcsampdict)

    # No z_cut label for subsequent samplers:
    # DEBUG: Removing dicts because they play poorly with dill
    # subsampdict = {'epsilon' : EPSILON,
    #                'sample space' : BIN_SPACE,
    #                'num events' : NUM_MC_EVENTS}
    # SUB_SAMPLERS.append(subsampdict)

    # ----------------------------------
    # Saving Samplers
    # ----------------------------------
    if SAVE_MC_EVENTS:
        # Saving critical samplers:
        if use_crit:
            with open(critfile_path, 'wb') as file:
                pickle.dump(CRIT_SAMPLERS, file)
        # Saving pre-critical samplers:
        if use_precrit:
            with open(prefile_path, 'wb') as file:
                pickle.dump(PRE_SAMPLERS, file)
        # Saving subsequent sampler:
        if use_sub:
            with open(subfile_path, 'wb') as file:
                pickle.dump(SUB_SAMPLERS, file)

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
    print("    Loading critical radiators...", flush=True)
    with open(critrad_path, 'rb') as file:
        CRIT_RADIATORS = pickle.load(file)
    print("    Loading pre-critical radiators...", flush=True)
    with open(prerad_path, 'rb') as file:
        PRE_RADIATORS = pickle.load(file)
    print("    Loading subsequent radiators...", flush=True)
    with open(subrad_path, 'rb') as file:
        SUB_RADIATORS = pickle.load(file)
    print("Radiators loaded!")
elif not MAKE_CRIT_RAD:
    print("Loading critical radiators...", flush=True)
    with open(critrad_path, 'rb') as file:
        CRIT_RADIATORS = pickle.load(file)
    print("Radiators loaded!")

# ----------------------------------
# Generating Radiators
# ----------------------------------
if not LOAD_MC_RADS and SAVE_MC_RADS:
    print("Generating radiators for Monte Carlo integration:")
    if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        # Setting up radiator discrete integrals
        CRIT_INTEGRAL_DATA = {z_cut : {} for z_cut in Z_CUTS}
        if use_precrit:
            PRE_INTEGRAL_DATA = {z_cut : {} for z_cut in Z_CUTS}
        if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
            SUB_INTEGRAL_DATA = {z_cut : {} for z_cut in Z_CUTS}
        elif COMPARE_RAW:
            SUB_INTEGRAL_DATA = {beta : {} for beta in BETAS}

        for i, z_cut in enumerate(Z_CUTS):
            if MAKE_CRIT_RAD:
                print("    Generating critical radiator with cutoff z_cut="
                      +str(z_cut)+"...", flush=True)
                # Critical radiators
                crit_rad, crit_rad_data = gen_numerical_radiator(
                                                    CRIT_SAMPLERS[i], 'crit',
                                                    JET_TYPE,
                                                    obs_accuracy=OBS_ACC,
                                                    splitfn_accuracy=SPLITFN_ACC,
                                                    beta=None,
                                                    bin_space=BIN_SPACE,
                                                    fixed_coupling=FIXED_COUPLING,
                                                    num_bins=NUM_RAD_BINS)
                CRIT_RADIATORS.append(crit_rad)
                CRIT_INTEGRAL_DATA[z_cut] = crit_rad_data

            # Pre-critical radiators
            pre_rad = None
            if COMPARE_PRE_AND_CRIT or COMPARE_ALL:
                print("    Generating pre-critical radiator with cutoff z_cut="
                      +str(z_cut)+"...", flush=True)
                pre_rad, pre_rad_data = gen_pre_num_rad(PRE_SAMPLERS[i],
                                            CRIT_SAMPLERS[i],
                                            JET_TYPE,
                                            obs_accuracy=OBS_ACC,
                                            splitfn_accuracy=SPLITFN_ACC,
                                            bin_space=BIN_SPACE,
                                            fixed_coupling=FIXED_COUPLING,
                                            num_bins=NUM_RAD_BINS)
            PRE_RADIATORS.append(pre_rad)
            PRE_INTEGRAL_DATA[z_cut] = pre_rad_data

        sub_rad = None

        # Subsequent radiators
        if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
            for _, beta in enumerate(BETAS):
                print("    Generating critical/subsequent radiator "
                      "with beta="+str(beta)+"...", flush=True)
                sub_rad, sub_rad_data = gen_crit_sub_num_rad(SUB_SAMPLERS[0],
                                               JET_TYPE,
                                               obs_accuracy=OBS_ACC,
                                               splitfn_accuracy=SPLITFN_ACC,
                                               beta=beta,
                                               epsilon=EPSILON,
                                               bin_space=BIN_SPACE,
                                               fixed_coupling=FIXED_COUPLING,
                                               num_bins=NUM_RAD_BINS)
                SUB_RADIATORS.append(sub_rad)
                SUB_INTEGRAL_DATA[z_cut][beta] = sub_rad_data

    elif COMPARE_RAW:
        for _, beta in enumerate(BETAS):
            print("    Generating subsequent radiator with beta="
                  +str(beta)+"...", flush=True)
            sub_rad, sub_rad_data = gen_numerical_radiator(SUB_SAMPLERS[0], 'sub',
                                             JET_TYPE,
                                             obs_accuracy=OBS_ACC,
                                             splitfn_accuracy=SPLITFN_ACC,
                                             beta=beta,
                                             bin_space=BIN_SPACE,
                                             fixed_coupling=FIXED_COUPLING,
                                             num_bins=NUM_RAD_BINS)
            SUB_RADIATORS.append(sub_rad)
            SUB_INTEGRAL_DATA[beta] = sub_rad_data

    # ----------------------------------
    # Labeling
    # ----------------------------------
    # Additional information for z_cut dependent radiators:
    # DEBUG: Removing dicts because they play poorly with dill
    # zcraddict = {'z_cuts' : Z_CUTS,
    #              'jet type' : JET_TYPE,
    #              'fixed coupling' : FIXED_COUPLING,
    #              'observable accuracy' : OBS_ACC,
    #              'split fn accuracy' : SPLITFN_ACC,
    #              'epsilon' : EPSILON,
    #              'sample space' : BIN_SPACE,
    #              'num events' : NUM_MC_EVENTS,
    #              'num bins' : NUM_RAD_BINS}
    # CRIT_SAMPLERS.append(zcraddict)
    # PRE_SAMPLERS.append(zcraddict)

    # Additional information for subsequent radiators:
    # DEBUG: Removing dicts because they play poorly with dill
    # subraddict = {'betas' : BETAS,
    #               'jet type' : JET_TYPE,
    #               'fixed coupling' : FIXED_COUPLING,
    #               'observable accuracy' : OBS_ACC,
    #               'split fn accuracy' : SPLITFN_ACC,
    #               'epsilon' : EPSILON,
    #               'sample space' : BIN_SPACE,
    #               'num events' : NUM_MC_EVENTS,
    #               'num bins' : NUM_RAD_BINS}
    # SUB_SAMPLERS.append(subraddict)

    # ----------------------------------
    # Saving Radiators:
    # ----------------------------------
    if SAVE_MC_RADS:
        # Saving critical radiators:
        if use_crit and MAKE_CRIT_RAD:
            print("Saving critical radiator to "+str(critrad_path), flush=True)
            # Saving interpolating functions
            with open(critrad_path, 'wb') as file:
                print(np.array(CRIT_RADIATORS))
                print(CRIT_RADIATORS[0])
                np.save(file, np.array(CRIT_RADIATORS))
            # Saving discretely generated numerical data
            with open(critrad_int_path, 'wb') as file:
                np.savez(file, **CRIT_INTEGRAL_DATA)
            print("Saving complete!", flush=True)
        # Saving pre-critical radiators:
        if use_precrit:
            print("Saving pre-crit radiator to "+str(prerad_path), flush=True)
            # Saving interpolating functions
            with open(prerad_path, 'wb') as file:
                pickle.dump(PRE_RADIATORS, file)
            # Saving discretely generated numerical data
            with open(prerad_int_path, 'wb') as file:
                np.savez(file, **PRE_INTEGRAL_DATA)
            print("Saving complete!", flush=True)
        # Saving subsequent radiators:
        if use_sub:
            if COMPARE_RAW:
                desc = 'sub'
            else:
                desc = 'crit-sub'
            print("Saving "+desc+" radiator to "+str(subrad_path), flush=True)
            # Saving interpolating functions
            with open(subrad_path, 'wb') as file:
                pickle.dump(SUB_RADIATORS, file)
            # Saving discretely generated numerical data
            with open(subrad_int_path, 'wb') as file:
                np.savez(file, **SUB_INTEGRAL_DATA)
            print("Saving complete!", flush=True)

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
              +str(z_cut)+"...", flush=True)
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
    # DEBUG: Removing dicts because they play poorly with dill
    # splitdict = {'z_cuts' : Z_CUTS,
    #              'jet type' : JET_TYPE,
    #              'fixed coupling' : FIXED_COUPLING,
    #              'accuracy' : SPLITFN_ACC,
    #              'epsilon' : EPSILON,
    #              'sample space' : BIN_SPACE,
    #              'num events' : NUM_MC_EVENTS,
    #              'num bins' : NUM_SPLITFN_BINS}
    # SPLITTING_FNS.append(splitdict)

    # ----------------------------------
    # Saving Splitting Functions
    # ----------------------------------
    with open(splitfn_path, 'wb') as file:
        pickle.dump(SPLITTING_FNS, file)
    print("Saved splitting functions to "+str(splitfn_path)+".", flush=True)

# ----------------------------------
# Loading Splitting Functions
# ----------------------------------
elif LOAD_SPLITTING_FNS:
    print("Loading normalized splitting functions...", flush=True)
    with open(splitfn_path, 'rb') as file:
        SPLITTING_FNS = pickle.load(file)
