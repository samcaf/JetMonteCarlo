from dataclasses import dataclass
from abc import ABC, abstractmethod

from jetmontecarlo.numerics.ewoc.montecarlo_ewoc import NPoint_MonteCarloEWOC,\
    NPoint_MonteCarloSubJetEWOC


class EnergyEnergyCorrelator(NPoint_MonteCarloEWOC, ABC):
    pass

class SubJetEnergyEnergyCorrelator(NPoint_MonteCarloSubJetEWOC, ABC):
    pass
