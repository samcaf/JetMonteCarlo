# Local imports
from jetmontecarlo.analytics.qcd_utils import MU_NP, LAMBDA_QCD,\
    TEN_MeV, ONE_MeV

VERBOSE = 3

###########################################
# Notes:
###########################################


###########################################
# Definitions and Parameters
###########################################
tab = '    '

# =====================================
# Physics Inputs
# =====================================
FIXED_COUPLING = False
MULTIPLE_EMISSIONS = False

# Observable accuracy
OBS_ACC = 'MLL'

# Splitting Function generation accuracy
SPLITFN_ACC = 'MLL'

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
#F_SOFTS = [.5, .75, 1]
F_SOFTS = [1]
# Getting the allowed values of f*z_cut for our calculations modulo duplicates
Z_CUTS = sorted(set([f*zc for zc in Z_CUTS for f in F_SOFTS]))

# List of betas for the C_1^{(betas)} in which we are interested:
# (use ints when possible for consistency across files)
BETAS = [2]
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
NUM_MC_EVENTS = int(5e6)
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
NUM_RAD_BINS = int(5e3)
NUM_SPLITFN_BINS = int(5e3)

# Determining whether to load inverse transform samples for full computation
LOAD_INV_SAMPLES = LOAD_MC_EVENTS

# ------------------------------------
# Sampling Parameters
# ------------------------------------
# -----------------
# MC Integration
# -----------------
EPSILON = 1e-15 if FIXED_COUPLING else 1e-10
BIN_SPACE = 'log'

# -----------------
# Parton showering
# -----------------
# Shower cutoff:
SHOWER_CUTOFF = TEN_MeV

# Shower ordering variable (ordered by angularity e^{(beta)})
SHOWER_BETA = None
if FIXED_COUPLING:
    SHOWER_BETA = 2

# Additional info to label parton shower files
SHOWER_INFO = 'cutoff'+str(SHOWER_CUTOFF)\
           if (not FIXED_COUPLING and SHOWER_CUTOFF not in [1e-10, MU_NP])\
           or (FIXED_COUPLING and SHOWER_CUTOFF not in [1e-10, 1e-15])\
       else ''

# =====================================
# All Monte Carlo related parameters
# =====================================
PHASESPACE_PARAMS = {
    'number of MC events' : NUM_MC_EVENTS,
    'epsilon' : EPSILON,
    'bin space' : BIN_SPACE,
    # z_cut will need to be defined for each instance
    'z_cut' : None
}

RADIATOR_PARAMS = {
    'jet type' : JET_TYPE,
    'fixed coupling' : FIXED_COUPLING,
    'number of MC events' : NUM_MC_EVENTS,
    'number of bins' : NUM_RAD_BINS,
    'epsilon' : EPSILON,
    'bin space' : BIN_SPACE,
    'observable accuracy' : OBS_ACC,
    'splitting function accuracy' : SPLITFN_ACC,
    # z_cut and beta will need to be defined for each instance
    'z_cut' : None,
    'beta' : None
}

SPLITTINGFN_PARAMS = {
    'jet type' : JET_TYPE,
    'fixed coupling' : FIXED_COUPLING,
    'number of MC events' : NUM_MC_EVENTS,
    'number of bins' : NUM_SPLITFN_BINS,
    'accuracy' : SPLITFN_ACC,
    # z_cut will need to be defined for each instance
    'z_cut' : None
}

SHOWER_PARAMS = {
    'fixed coupling' : FIXED_COUPLING,
    'multiple emissions' : MULTIPLE_EMISSIONS,
    'observable accuracy' : OBS_ACC,
    'jet type' : JET_TYPE,
    'number of shower events' : NUM_SHOWER_EVENTS,
    'shower cutoff' : SHOWER_CUTOFF,
    # The angularity which orders the shower will need
    # to be defined for each instance
    'shower beta' : None
}


ALL_MONTECARLO_PARAMS = {
    'fixed coupling' : FIXED_COUPLING,
    'multiple emissions' : MULTIPLE_EMISSIONS,
    'observable accuracy' : OBS_ACC,
    'splitting function accuracy' : SPLITFN_ACC,
    'jet type' : JET_TYPE,
    'number of MC events' : NUM_MC_EVENTS,
    'number of radiator bins' : NUM_RAD_BINS,
    'number of splitting function bins' : NUM_SPLITFN_BINS,
    'epsilon' : EPSILON,
    'bin space' : BIN_SPACE,
    'number of shower events' : NUM_SHOWER_EVENTS,
    'shower cutoff' : SHOWER_CUTOFF,
}



# ------------------------------------
# Emissions to compare
# ------------------------------------
COMPARE_CRIT = True
MAKE_CRIT_RAD = True
COMPARE_CRIT_AND_SUB = True
COMPARE_PRE_AND_CRIT = True
COMPARE_ALL = True
COMPARE_RAW = True not in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                             COMPARE_PRE_AND_CRIT, COMPARE_ALL]

# - - - - - - - - - - - - - - - - -
# Given emissions up for comparison, emissions to generate
# - - - - - - - - - - - - - - - - -
USE_CRIT = True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB,
                    COMPARE_PRE_AND_CRIT, COMPARE_ALL]
USE_PRECRIT = COMPARE_PRE_AND_CRIT or COMPARE_ALL
USE_CRIT_SUB = COMPARE_RAW or COMPARE_CRIT_AND_SUB or COMPARE_ALL



###########################################
# Printing Information
###########################################
if __name__ == '__main__':
    if VERBOSE > 0:
        print("\n# =====================================\n"+
              "# Parameters:\n# =====================================")

        print(tab+"# -----------------------------\n"+
              tab+"# Physics:\n"+tab+"# -----------------------------")
        print(tab+tab+"# Jet type: "+str(JET_TYPE))
        print(tab+tab+"# Fixed coupling: "+str(FIXED_COUPLING))
        print(tab+tab+"# Multiple Emissions:", MULTIPLE_EMISSIONS)
        print(tab+tab+"# Observable accuracy: "+str(OBS_ACC))
        print(tab+tab+"# Splitting function accuracy: "+str(OBS_ACC))

        print("\n"+tab+"# -----------------------------"+
              "\n"+tab+"# Grooming:\n"+tab+"# -----------------------------")
        print(tab+tab+"# f * z_cut values: " + str(Z_CUTS))
        if VERBOSE > 2:
            print(tab+tab+"# z_cut dictionary: " + str(INDEX_ZC))


        print("\n"+tab+"# -----------------------------"
              +"\n"+tab+"# Monte Carlo:\n"+tab+"# -----------------------------")
        print(tab+tab+"# Number of events for MC integration: {:.1e}".format(NUM_MC_EVENTS))
        if VERBOSE > 1:
            print("\n"+tab+tab+"# Basic MC Integration Information:")
            print(tab+tab+tab+"# Integration space: " + str(BIN_SPACE) +
                  " (if log, integration cutoff of " + str(EPSILON) + ")")
            if VERBOSE > 2:
                print(tab+tab+tab+"# Load MC events: " + str(LOAD_MC_EVENTS))
                print(tab+tab+tab+"# Save MC events: " + str(SAVE_MC_EVENTS))
            print("\n"+tab+tab+"# Radiator Information:")
            print(tab+tab+tab+"# Number of radiator bins: {:.1e}".format(NUM_RAD_BINS))
            if VERBOSE > 2:
                print(tab+tab+tab+"# Load MC radiators: " + str(LOAD_MC_RADS))
                print(tab+tab+tab+"# Save MC radiators: " + str(SAVE_MC_RADS))
            print("\n"+tab+tab+"# Spliting Function Information:")
            print(tab+tab+tab+"# Number of splitting function bins:  {:.1e}".format(NUM_RAD_BINS))
            if VERBOSE > 2:
                print(tab+tab+tab+"# Load MC splitting functions: " + str(LOAD_MC_EVENTS))
                print(tab+tab+tab+"# Save MC splitting functions: " + str(SAVE_MC_EVENTS))

        print("\n"+tab+tab+"# Parton Shower Information:")
        print(tab+tab+tab+"# Number of parton shower events: {:.1e}".format(NUM_SHOWER_EVENTS))
        if VERBOSE > 1:
            print(tab+tab+tab+"# Shower cutoff:  {:.1e}".format(SHOWER_CUTOFF))
            print(tab+tab+tab+"# Angularity beta for shower ordering: " + str(SHOWER_BETA))
            print(tab+tab+tab+"# Shower information:", SHOWER_INFO)
        print("\n")
