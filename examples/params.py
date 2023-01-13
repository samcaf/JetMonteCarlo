from __future__ import absolute_import
from pathlib import Path
import dill as pickle

# Local imports
from jetmontecarlo.analytics.QCD_utils import MU_NP, LAMBDA_QCD

VERBOSE = 3

###########################################
# Notes:
###########################################
# Done:

# Run with running coupling, LL, with 5e6
# Run with running coupling, MLL, with 5e6
# Run with fixed coupling, LL, with 5e6
    # Something funky with crit results
# Run with running coupling, LL, with 1e6
    # crit and pre-crit rads suspiciously fast, probably true of fc too

###########################################
# Definitions and Parameters
###########################################
# =====================================
# Physics Inputs
# =====================================
FIXED_COUPLING = True
MULTIPLE_EMISSIONS = False

# Observable accuracy
OBS_ACC = 'LL'

# Splitting Function generation accuracy
SPLITFN_ACC = 'LL'

# Angular ordering takes more time, and
# does not change the agreement between our
# analytic, semi-analytic, and parton shower
# results!
ANGULAR_ORDERING = False

# ------------------------------------
# Jet and grooming parameters
# ------------------------------------
JET_TYPE = 'quark'

Z_CUTS = [.1, .2]
F_SOFTS = [.5, .75, 1]
# Getting the allowed values of f*z_cut for our calculations modulo duplicates
Z_CUTS = sorted(set([f*zc for zc in Z_CUTS for f in F_SOFTS]))

# List of betas for the C_1^{(betas)} in which we are interested:
# (use ints when possible for consistency across files)
BETAS = [1./2., 1, 2, 3, 4]
# BETAS = [2, 1, 3, 4, 1./2.]

# Setting up dictionaries to facilitate calling functions of z_cut and beta.
# In particular, the radiators are organized as lists, ordered by z_cut but without
# an exact reference to z_cut. Dictionaries might be better in the future.
INDEX_ZC = {zc : i for i, zc in enumerate(Z_CUTS)}
INDEX_BETA = {beta : i for i, beta in enumerate(BETAS)}

# Confusingly, the parton shower zcuts are enumerated differently
PS_INDEX_ZC = {zc: i for i, zc in enumerate([.05, .1, .2])}

# =====================================
# Monte Carlo parameters
# =====================================
# ------------------------------------
# MC Event Parameters
# ------------------------------------
# Number of generated events
NUM_MC_EVENTS = int(1e4)
NUM_SHOWER_EVENTS = int(5e5)

# MC Sampling Switches:
LOAD_MC_EVENTS = False
# Default True: phase space doesn't change, only weights do
SAVE_MC_EVENTS = True

# MC Radiator Switches:
LOAD_MC_RADS = False
# Default False, to generate radiators with the correct parameters
SAVE_MC_RADS = True

# MC Splitting Function Switches:
LOAD_SPLITTING_FNS = False
# Default False, to generate splitting functions with the correct parameters
SAVE_SPLITTING_FNS = True

# Number of bins for integration of radiators and splitting functions:
NUM_RAD_BINS = int(5e2)
NUM_SPLITFN_BINS = int(5e2)

# Determining whether to load inverse transform samples for full computation
LOAD_INV_SAMPLES = LOAD_MC_EVENTS

# ------------------------------------
# Sampling Parameters
# ------------------------------------
# -----------------
# MC Integration (DEBUG NOTE: EPSILON WAS 1e-10 before)
# -----------------
EPSILON = 1e-15 if FIXED_COUPLING else 1e-10
BIN_SPACE = 'log'

# -----------------
# Parton showering
# -----------------
# Shower cutoff:
SHOWER_CUTOFF = 1e-15

# Shower ordering variable (ordered by angularity e^{(beta)})
SHOWER_BETA = None
if FIXED_COUPLING:
    SHOWER_BETA = 2

# Additional info to label parton shower files
SHOWER_INFO = 'cutoff'+str(SHOWER_CUTOFF)\
           if (not FIXED_COUPLING and SHOWER_CUTOFF not in [1e-10, MU_NP])\
           or (FIXED_COUPLING and SHOWER_CUTOFF not in [1e-10, 1e-15])\
       else ''

# ------------------------------------
# Emissions to Generate
# ------------------------------------
COMPARE_CRIT = True
MAKE_CRIT_RAD = True
COMPARE_CRIT_AND_SUB = True
COMPARE_PRE_AND_CRIT = True
COMPARE_ALL = True
COMPARE_UNGROOMED = True and not\
                    True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                             COMPARE_PRE_AND_CRIT, COMPARE_ALL]

###########################################
# Setting Up Event Generation:
###########################################
# =====================================
# MC Filenames
# =====================================
# Sampler files
critfile = 'crit_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
prefile = 'pre_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
subfile = 'sub_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)

# Radiator files
critradfile = 'crit_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
preradfile = 'pre_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
subradfile = 'sub_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

splitfn_file = 'split_fns_{}_{}_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 SPLITFN_ACC, BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

if not FIXED_COUPLING:
    # Radiator files
    critradfile = 'crit_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
    preradfile = 'pre_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
    subradfile = 'sub_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

    splitfn_file = 'split_fns_{}_{}_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                     SPLITFN_ACC, BIN_SPACE,
                                                     NUM_MC_EVENTS, NUM_RAD_BINS)

if not COMPARE_UNGROOMED:
    # If we are not comparing ungroomed emissions, we remember that
    # the subsequent emissions are actually linked to the critical
    # emissions by angular ordering
    subfile = 'crit_' + subfile
    subradfile = 'crit_' + subradfile

# =====================================
# Relative paths
# =====================================
# Folders
ps_folder = Path("jetmontecarlo/utils/samples/parton_showers/")
sampler_folder = Path("jetmontecarlo/utils/samples/phase_space_samplers/")
rad_folder = Path("jetmontecarlo/utils/functions/radiators/")
splitfn_folder = Path("jetmontecarlo/utils/functions/splitting_fns/")

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


# ====================================
# Printing Information
# ====================================
if __name__ == '__main__':
    if VERBOSE > 0:
        print("\n# =====================================\n# Parameters:\n# =====================================")

        print("    # -----------------------------\n    # Physics:\n    # -----------------------------")
        print("    # Jet type: "+str(JET_TYPE))
        print("    # Fixed coupling: "+str(FIXED_COUPLING))
        print("    # Multiple Emissions:", MULTIPLE_EMISSIONS)
        print("    # Observable accuracy: "+str(OBS_ACC))
        print("    # Splitting function accuracy: "+str(OBS_ACC))

        print("\n    # -----------------------------\n    # Grooming:\n    # -----------------------------")
        print("    # f * z_cut values: " + str(Z_CUTS))
        if VERBOSE > 2:
            print("        # z_cut dictionary: " + str(INDEX_ZC))


        print("\n    # -----------------------------\n    # Monte Carlo:\n    # -----------------------------")
        print("    # Number of events for MC integration: {:.1e}".format(NUM_MC_EVENTS))
        if VERBOSE > 1:
            # Basic MC Integraion Information:
            print("        # Integration space: " + str(BIN_SPACE) + " (if log, integration cutoff of " + str(EPSILON) + ")")
            if VERBOSE > 2:
                print("            # Load MC events: " + str(LOAD_MC_EVENTS))
                print("            # Save MC events: " + str(SAVE_MC_EVENTS))
            # Radiator Information:
            print("        # Number of radiator bins: {:.1e}".format(NUM_RAD_BINS))
            if VERBOSE > 2:
                print("            # Load MC radiators: " + str(LOAD_MC_RADS))
                print("            # Save MC radiators: " + str(SAVE_MC_RADS))
            # Spliting Function Information:
            print("        # Number of splitting function bins:  {:.1e}".format(NUM_RAD_BINS))
            if VERBOSE > 2:
                print("            # Load MC splitting functions: " + str(LOAD_MC_EVENTS))
                print("            # Save MC splitting functions: " + str(SAVE_MC_EVENTS))

        # Parton Shower Information:
        print("    # Number of parton shower events: {:.1e}".format(NUM_SHOWER_EVENTS))
        if VERBOSE > 1:
            print("        # Shower cutoff:  {:.1e}".format(SHOWER_CUTOFF))
            print("        # Angularity beta for shower ordering: " + str(SHOWER_BETA))
            print("        # Shower information:", SHOWER_INFO)
        print("\n")
