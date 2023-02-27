from dataclasses import dataclass
from abc import ABC, abstractmethod

from jetmontecarlo.numerics.ewoc.montecarlo_ewoc import NPoint_MonteCarloEWOC,\
    NPoint_MonteCarloSubJetEWOC


class MassSquaredEWOC(NPoint_MonteCarloEWOC, ABC):
    pass

class MassSquaredSubJetEWOC(NPoint_MonteCarloSubJetEWOC, ABC):
    pass
