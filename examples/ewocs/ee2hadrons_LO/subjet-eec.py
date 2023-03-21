import matplotlib.pyplot as plt
import numpy as np

from jetmontecarlo.utils.interpolation import lin_log_mixed_list

from jetmontecarlo.analytics.ewocs.eec import eec

from jetmontecarlo.montecarlo.ewoc import Npoint_MonteCarloSubJetEWOC

from jetmontecarlo.utils.jet_utils import AlgorithmAgnosticApproximation

from examples.processes.ee2hadrons_LO import ee2hadrons_LO
from examples.processes.ee2hadrons_LO import ee2hLO_PairwiseObservable
from examples.processes.ee2hadrons_LO import theta_ij, costheta_ij, z_i, z_j

# Redefining name for brevity
Observable = ee2hLO_PairwiseObservable

# Parameters
subjet_radius = .01
jet_radius = .7
# Redefining name for brevity
Observable = ee2hLO_PairwiseObservable

# ---------------------------------
# Monte Carlo and Process
# ---------------------------------
num_samples = int(1e4)
sample_method = 'log'

energy = 1000.

# For analytic plot:
singular_only = True

# =====================================
# Monte Carlo
# =====================================
if __name__ == "__main__":
    # Define the process
    process = ee2hadrons_LO(energy=energy, num_samples=num_samples,
                            sampleMethod=sample_method)

    # Get observables
    costhetas = Observable(process, costheta_ij, r'$\cos\theta_{ij}$')
    thetas = Observable(process, theta_ij, r'$\theta_{ij}$')

    # Set up jet and subjet algorithms
    subjet_algorithm = AlgorithmAgnosticApproximation(subjet_radius)
    jet_algorithm = AlgorithmAgnosticApproximation(jet_radius)

    # Find the SubJet EWOC
    ee2hadrons_subjet_ewoc = Npoint_MonteCarloSubJetEWOC(
                                            jet_algorithm, subjet_algorithm,
                                            montecarlo_process=process,
                                            montecarlo_observable=costhetas)

    z_i_Observable = Observable(process, z_i, r'$z_i$')
    z_j_Observable = Observable(process, z_j, r'$z_j$')
    ee2hadrons_subjet_ewoc.set_weights(z_i_Observable, z_j_Observable,
                                       thetas)
    ee2hadrons_subjet_ewoc.generate_ewoc(numbins=100, binspacing='lin')

    # Plot the EWOC
    fig, ax = plt.subplots()

    # Numerical
    ee2hadrons_subjet_ewoc.plot_ewoc(ax, label='Numeric', color='black', lw=2)

    # Analytic
    cos_vals = lin_log_mixed_list(np.min(ee2hadrons_subjet_ewoc.observables),
                                  np.max(ee2hadrons_subjet_ewoc.observables),
                                  1000)
    cos_vals = np.array([-.95, -.9, *cos_vals, .9, .95])
    zs = (1-cos_vals)/2
    eecs = np.array([eec(z, process.energy, 'lo',
                         Rjet=jet_radius, rsub=subjet_radius,
                         singular_only=singular_only) for z in zs])
    ax.plot(cos_vals, eecs, label='Analytic', ls='dashed', color='grey', lw=2)

    ax.set_ylim(0, 1e3)
    # DEBUG: Strange numerical behavior
    ax.set_ylim(0, .5)

    plt.legend()
    plt.tight_layout()
    plt.show()
