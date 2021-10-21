# Loading parton_shower class:
from jetmontecarlo.montecarlo.partonshower import *

####################################
# Example parton shower usage:
####################################
# =====================================
# Initializing the Shower:
# =====================================
# Showers are ordered by an angularity e_beta
# Arguments are:
#       * the accuracy of the coupling,
#       * the cutoff angularity, at which the shower stops, and
#       * the value of beta for the angularity e_beta which orders the shower
shower = parton_shower(fixed_coupling=False, shower_cutoff=1e-10, shower_beta=2)

# =====================================
# Generating or Loading Events:
# =====================================
shower.gen_events(5e5)
shower.save_events()
#shower.load_events(1e5)

# =====================================
# Saving Jet Observables:
# =====================================
betas = [2]
shower.save_correlations(betas, 'LL', f_soft=1)
shower.save_correlations(betas, 'LL', f_soft=.75)
shower.save_correlations(betas, 'LL', f_soft=.5)

# Run with 1) false fixed coupling, LL observable, then 2) false fixed coupling, MU_NP, MLL
# then 3) false fixed coupling, MU_NP, MLL for each beta
