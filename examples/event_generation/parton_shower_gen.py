# Local utilities for numerics
from jetmontecarlo.montecarlo.partonshower import *

####################################
# Example parton shower usage:
####################################
shower = parton_shower(fixed_coupling=True, shower_cutoff=1e-10, shower_beta=2)
# shower.gen_events(5e5)
# shower.save_events()
shower.load_events(1e4)
shower.save_correlations([2], 'LL', f_soft=.55)
