import numpy as np
from dataclasses import dataclass

from jetmontecarlo.montecarlo.process import MonteCarloProcess
from jetmontecarlo.montecarlo.process import MonteCarloObservable
from jetmontecarlo.analytics.qcd_utils import alpha1loop  #, alpha_em


class ee2hadrons_LO(MonteCarloProcess):
    """A Monte Carlo process for $e^+ e^- \to$ hadrons (at LO)"""
    def _phasespace_constraints(self, **kwargs):
        """The LO phase space constraints for e+e- -> hadrons."""
        x_1, x_2 = 1 - kwargs['1 - x_1'], 1 - kwargs['1 - x_2']
        x_3 = 2 - x_1 - x_2
        return 0 < x_1 < 1 and 0 < x_2 < 1 and 0 < x_3 < 1


    def _differential_xsec(self, **kwargs):
        """The LO differential cross section for e+e- -> hadrons."""
        Q = self.energy

        # DEBUG: Include alpha_s and alpha_em
        # alpha_s = alpha1loop(Q)
        prefactor = Q**2. / (2.*np.pi)**3.
        # * alpha_s * alpha_em(Q)**2. * (...)

        # DEBUG: Prefactor is 1 for now
        prefactor = 1.

        x_1, x_2 = 1 - kwargs['1 - x_1'], kwargs['1 - x_2']

        return prefactor * (x_1**2. + x_2**2.)/((1.-x_1) * (1.-x_2))


    def named_samples(self):
        """Returns a dict whose keys are the names of the kinematic
        variables and whose values are the corresponding samples.
        """
        x_1 = [1 - sample[0] for sample in self.samples]
        x_2 = [1 - sample[1] for sample in self.samples]
        x_3 = [2 - x_1[i] - x_2[i] for i in range(len(x_1))]
        return {
            'x_1': x_1,
            'x_2': x_2,
            'x_3': x_3,
        }


    def __init__(self, energy, num_samples, sampleMethod='log',
                 **kwargs):
        kwargs.update({
            # Basic Info
            'name': r'$e^+ e^- \to$ hadrons (LO)',
            'accuracy': 'LO',
            # Sampling Info
            'num_samples': num_samples,
            'sampleMethod': sampleMethod,
            'epsilon': 1e-8,
            # Kinematic Info
            'energy': energy,
            'kinematic_vars': ['1 - x_1', '1 - x_2'],
            'kinematic_bounds': [(0, 1), (0, 1)],
            # DEBUG: Total cross section is 1 for now
            'total_cross_section': 1.,
        })
        super().__init__(**kwargs)
        self.generateSamples(num_samples)
        self.setArea()


# =====================================
# Process Dependent Observables
# =====================================
@dataclass
class ee2hLO_PairwiseObservable(MonteCarloObservable):
    """The MonteCarloObservable for costhetaij in e+e- -> hadrons
    at LO."""

    def update(self):
        """Update the observable."""
        x_1 = np.array(self.process.named_samples()['x_1'])
        x_2 = np.array(self.process.named_samples()['x_2'])
        x_3 = np.array([2 - x_1val - x_2val
                        for x_1val, x_2val in zip(x_1, x_2)])

        self.observables = []
        self.weights = []
        for (x_i, x_j) in [(x_1, x_2), (x_2, x_3), (x_3, x_1)]:
            self.observables = np.append(self.observables,
                                 self.kinematic_function(x_i, x_j))
            self.weights += self.process.jacobians
        self.weights = np.array(self.weights)


# ---------------------------------
# Angle
# ---------------------------------
def costheta_ij(x_i, x_j):
    """The functional form of the costheta_ij observable for
    e+e- -> hadrons at LO.
    """
    return 2./(x_i*x_j) - 2./x_i - 2./x_j + 1


# ---------------------------------
# Mass-Squared
# ---------------------------------
def m2_ij(x_i, x_j):
    """The functional form of the mass-squared observable
    between particles i and j, for e+e- -> hadrons at LO.
    """
    return 1 - x_i - x_j


# ---------------------------------
# Generalized Jet Energy Correlation
# ---------------------------------
def gecf_ij(x_i, x_j, kappa):
    """The functional form of the generalized jet energy
    correlation function for e+e- -> hadrons at LO.
    """
    return 2**(kappa/2-2) * x_i*x_j *\
                (1 - costheta_ij(x_i, x_j))**(kappa/2)


# ---------------------------------
# Energy Weights (for EWOCs):
# ---------------------------------
def z_i(x_i, x_j):
    """The functional form of the energy weight for particle i,
    for e+e- -> hadrons at LO.
    """
    return x_i/2

def z_j(x_i, x_j):
    """The functional form of the energy weight for particle i,
    for e+e- -> hadrons at LO.
    """
    return x_j/2


# ---------------------------------
# Rough example
# ---------------------------------
if __name__ == '__main__':
    process = ee2hadrons_LO(energy=1000., num_samples=100)
    print('x_1:', process.named_samples()['x_1'])
    print('x_2:', process.named_samples()['x_2'])
    print('x_3:', process.named_samples()['x_3'])
    print([x_1 + x_2 + x_3 for x_1, x_2, x_3 in
           zip(*process.named_samples().values())])
