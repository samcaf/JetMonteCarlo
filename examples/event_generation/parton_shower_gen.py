from __future__ import absolute_import

# Loading parton shower class and parameters:
from jetmontecarlo.montecarlo.partonshower import *
from examples.params import *
from examples.filenames import *

SAVE_SHOWER_EVENTS = False
LOAD_SHOWER_EVENTS = False
SAVE_SHOWER_CORRELATIONS = True

"""
####################################
# Example parton shower usage:
####################################
# =====================================
# Initializing the Shower:
# =====================================
# Showers are ordered by an angularity e_beta
# Arguments are:
#       * the accuracy of the coupling;
#       * the cutoff angularity, at which the shower stops;
#       * the value of beta for the angularity e_beta which orders the shower;
#       * the type of parton initiating the parton shower.
shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                       shower_cutoff=SHOWER_CUTOFF,
                       shower_beta=SHOWER_BETA,
                       jet_type=JET_TYPE)

# =====================================
# Generating or Loading Events:
# =====================================
shower.gen_events(NUM_SHOWER_EVENTS)
shower.save_events()
#shower.load_events(NUM_SHOWER_EVENTS)

for beta in BETAS:
    # =====================================
    # Saving Jet Observables:
    # =====================================
    shower.save_correlations(beta, OBS_ACC, f_soft=1)
    shower.save_correlations(beta, OBS_ACC, f_soft=.75)
    shower.save_correlations(beta, OBS_ACC, f_soft=.5)
    print()
"""

for b in BETAS:
    shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                           shower_cutoff=SHOWER_CUTOFF,
                           shower_beta=b,
                           jet_type=JET_TYPE)

    if LOAD_SHOWER_EVENTS:
        shower.load_events(NUM_SHOWER_EVENTS)
    else:
        shower.gen_events(NUM_SHOWER_EVENTS)
        if SAVE_SHOWER_EVENTS:
            shower.save_events()

    if SAVE_SHOWER_CORRELATIONS:
        shower.save_correlations(b, OBS_ACC, f_soft=1, info=SHOWER_INFO)
        shower.save_correlations(b, OBS_ACC, f_soft=.75, info=SHOWER_INFO)
        shower.save_correlations(b, OBS_ACC, f_soft=.5, info=SHOWER_INFO)
