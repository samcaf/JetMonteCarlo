import numpy as np
import matplotlib.pyplot as plt

from jetmontecarlo.montecarlo.integrator import integrator
from jetmontecarlo.montecarlo.process import MonteCarloProcess
from jetmontecarlo.montecarlo.process import MonteCarloObservable


class Npoint_MonteCarloEWOC(integrator):
    """N-point Monte Carlo EWOC integrator"""

    def set_weights(self, z1Observable: MonteCarloObservable,
                    z2Observable: MonteCarloObservable,
                    constraintObservable: MonteCarloObservable = None):
        """Sets the weights for the EWOC.
        z_i_function is the functional form of the energy of the i-th particle
        in terms of the kinematic variables of the process.
        """
        (w1, w2) = self.energy_weights
        z1 = np.array(z1Observable.observables)
        z2 = np.array(z2Observable.observables)
        self.weights = z1**w1 * z2**w2 *\
            np.array(self.montecarlo_observable.weights)
        self.has_weights = True


    def generate_ewoc(self, numbins, binspacing):
        """Generates the EWOC for the given observable"""
        if not self.has_weights:
            raise ValueError("Weights have not been set. Please use\n"
                             "    `ewocInstance.set_weights(z1_function, "
                             "z2_function)`,\n"
                             "where zi_function is the functional form "
                             "of the ith energy fraction in terms of "
                             "the kinematic variables of the process.")

        self.setBins(numbins=numbins,
                     observables=self.montecarlo_observable.observables,
                     binspacing=binspacing)

        # Not taking sum into account...
        self.setDensity(observables=self.montecarlo_observable.observables,
                        weights=self.weights,
                        area=self.montecarlo_process.area)

    def plot_ewoc(self, ax=None):
        """Plots the EWOC"""
        show_plot = False
        if ax is None:
            _, ax = plt.subplots()
            show_plot = True

        ax.plot(self.bin_midpoints, self.density)

        if show_plot:
            ax.set_xlabel(self.montecarlo_observable.name)
            ax.set_ylabel(self.montecarlo_observable.name + " EWOC")
            ax.set_title("EWOC for " + self.montecarlo_process.name)
            plt.show()


    def __init__(self, montecarlo_process: MonteCarloProcess,
                 montecarlo_observable: MonteCarloObservable,
                 # or derived class of MonteCarloObservable
                 N=2, energy_weights=(1, 1)):
        self.montecarlo_process = montecarlo_process
        self.montecarlo_observable = montecarlo_observable
        self.N = N
        self.energy_weights = energy_weights
        self.has_weights = False

        assert self.N == 2, "Only N=2 is implemented"
        assert len(self.energy_weights) == self.N,\
            "Weights must be of length N"

        super().__init__()


# DEPRECATED
# new approach: implement phase space constraaints directly above
# and then build boolean observables for (sub)jet combination and apply
# them to produce subjet EWOCs
class Npoint_MonteCarloSubjetEWOC(Npoint_MonteCarloEWOC):
    """N-point Monte Carlo EWOC integrator for (sub)jet observables"""
    # Need to translate from (sub)jet info to phase space constraints
    # and place these on the MonteCarloProcess
    # Then, can simply use these to make a mask for the observables
    # and make the associated EWOC easily

    def set_weights(self, z1Observable: MonteCarloObservable,
                    z2Observable: MonteCarloObservable):
        """Sets the weights for the EWOC.
        z_i_function is the functional form of the energy of the i-th particle
        in terms of the kinematic variables of the process.
        """
        (w1, w2) = self.energy_weights
        z1 = np.array(z1Observable.observables)
        z2 = np.array(z2Observable.observables)

        # Make new boolean observables for jet/subjet constraints,
        # then use these to only give non-zero weight to
        # configurations that satisfy the constraints
        self.weights = z1**w1 * z2**w2 *\
            np.array(self.montecarlo_observable.weights)
        self.has_weights = True


    def __init__(self, jetInfo, subjetInfo, **kwargs):
        self.jet_info = jetInfo
        self.subjet_info = subjetInfo

        super().__init__(**kwargs)
