from jetmontecarlo.montecarlo.process import MonteCarloObservable
from jetmontecarlo.montecarlo.ewoc import Npoint_MonteCarloEWOC

from examples.processes.ee2hadrons_LO import ee2hadrons_LO
from examples.processes.ee2hadrons_LO import ee2hLO_PairwiseObservable
from examples.processes.ee2hadrons_LO import costheta_ij, z_i, z_j

# Redefining name for brevity
Observable = ee2hLO_PairwiseObservable


if __name__ == "__main__":
    # Define the process
    process = ee2hadrons_LO(energy=1000., num_samples=100000,
                            sampleMethod='lin')

    # Get observables
    costhetas = Observable(process, costheta_ij, r'$\cos\theta_{ij}$')

    # Find the EWOC
    ee2hadrons_ewoc = Npoint_MonteCarloEWOC(process, costhetas)
    z_i_Observable = Observable(process, z_i, r'$z_i$')
    z_j_Observable = Observable(process, z_j, r'$z_j$')
    ee2hadrons_ewoc.set_weights(z_i_Observable, z_j_Observable)
    ee2hadrons_ewoc.generate_ewoc(numbins=100, binspacing='lin')

    # Plot the EWOC
    ee2hadrons_ewoc.plot_ewoc()
