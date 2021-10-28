from __future__ import absolute_import
from pathlib import Path

# Local imports
from jetmontecarlo.analytics.QCD_utils import MU_NP


###########################################
# Notes:
###########################################
# To Do
# Run with running coupling, LL, with 5e6
# Run with running coupling, MLL, with 5e6

# Done:
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
FIXED_COUPLING = False

# Observable accuracy
OBS_ACC = 'MLL'
if FIXED_COUPLING:
    OBS_ACC = 'LL'

# Parton shower generation accuracy
SPLITFN_ACC = 'MLL'
if FIXED_COUPLING:
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

# List of betas for the C_1^{(betas)} we will calculate:
# (use ints when possible for consistency across files)
BETAS = [1./2., 1, 2, 3, 4]

# =====================================
# Monte Carlo parameters
# =====================================
# ------------------------------------
# MC Event Parameters
# ------------------------------------
# Number of generated events
NUM_MC_EVENTS = int(5e3)
NUM_SHOWER_EVENTS = int(5e2)

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
NUM_RAD_BINS = int(1e2)
NUM_SPLITFN_BINS = int(1e2)

# ------------------------------------
# Sampling Parameters
# ------------------------------------
# MC Integration
EPSILON = 1e-15
BIN_SPACE = 'log'

# Parton showering
SHOWER_CUTOFF = MU_NP
SHOWER_BETA = 2

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
