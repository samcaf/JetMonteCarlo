from dataclasses import dataclass
import matplotlib.pyplot as plt

from jetmontecarlo.montecarlo.integrator import integrator
from jetmontecarlo.montecarlo.process import MonteCarloProcess
from jetmontecarlo.montecarlo.process import MonteCarloObservable

@dataclass
class Npoint_MonteCarloEWOC(integrator):
    """N-point Monte Carlo EWOC integrator"""
    montecarlo_process: MonteCarloProcess
    montecarlo_observable: MonteCarloObservable
    N: int = 2
    weights = (1, 1)


    def generate_ewoc(self, numbins, binspacing):
        """Generates the EWOC for the given observable"""
        self.setBins(numbins=numbins,
                     observables=self.montecarlo_observable.observables,
                     binspacing=binspacing)
        self.setDensity(observables=self.montecarlo_observable.observables,
                        weights=self.montecarlo_observable.weights,
                        area=self.montecarlo_process.area)

    def plot_ewoc(self, ax=None):
        """Plots the EWOC"""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.bin_midpoints, self.density)


    def __post_init__(self):
        assert self.N == 2, "Only N=2 is implemented"
        assert len(self.weights) == self.N, "Weights must be of length N"


class Npoint_MonteCarloSubjetEWOC(Npoint_MonteCarloEWOC):
    """N-point Monte Carlo EWOC integrator for (sub)jet observables"""
    # jet_info
    # subjet_info

    # Need to translate from (sub)jet info to phase space constraints
    # and place these on the MonteCarloProcess
    # Then, can simply use these to make a mask for the observables
    # and make the associated EWOC easily

    pass
