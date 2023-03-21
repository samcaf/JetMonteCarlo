"""Plotting mass-squared subjet EWOCs for e+ e- -> hadrons at LO, using the
algorithm agnostic approximation for jet finding.
"""
# Basic plotting, etc.
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Custom plotting
from jetmontecarlo.utils.color_utils import adjust_lightness
from jetmontecarlo.utils.plot_utils import get_colors_colorbar

# Extra data utils
from jetmontecarlo.utils.interpolation import lin_log_mixed_list

# Jet utils
from jetmontecarlo.utils.jet_utils import AlgorithmAgnosticApproximation

# EWOC analytic utils
from jetmontecarlo.analytics.ewocs.mass_squared_ewoc import\
    mass_squared_ewoc_ee2hadronsLO

# EWOC numerical utils
from jetmontecarlo.montecarlo.ewoc import Npoint_MonteCarloSubJetEWOC

from examples.processes.ee2hadrons_LO import ee2hadrons_LO
from examples.processes.ee2hadrons_LO import ee2hLO_PairwiseObservable
from examples.processes.ee2hadrons_LO import theta_ij, m2_ij, z_i, z_j


# Redefining name for brevity
Observable = ee2hLO_PairwiseObservable


# =====================================
# Parameters
# =====================================
# ---------------------------------
# Monte Carlo and Process
# ---------------------------------
num_samples = int(1e7)
sample_method = 'log'

energy = 1000.

# ---------------------------------
# Jets
# ---------------------------------
jet_radius = .8
jet_algorithm = AlgorithmAgnosticApproximation(jet_radius)

subjet_radii = [.1, .2, .3, .4, .5, .6]
# subjet_radii = [.1]
colors, cbar_data = get_colors_colorbar(subjet_radii)

# ---------------------------------
# Plotting
# ---------------------------------
xscale = 'log'
yscale = 'log'


# =====================================
# Monte Carlo
# =====================================
if __name__ == "__main__":
    # Define the process
    process = ee2hadrons_LO(energy=energy, num_samples=num_samples,
                            sampleMethod=sample_method)

    # Get observables
    mass_squared = Observable(process, m2_ij, r'$m^2_{ij}/Q^2$')
    thetas = Observable(process, theta_ij, r'$\theta_{ij}$')

    # Setting up EWOC plots
    fig, ax = plt.subplots()
    ax.set_prop_cycle((cycler(color=colors)))

    # Looping over subjet radii and plotting EWOCs
    for i, (subjet_radius, color) in enumerate(zip(subjet_radii, colors)):
        # Setting up subjet algorithm
        subjet_algorithm = AlgorithmAgnosticApproximation(subjet_radius)

        # Find the subjet EWOC
        ee2hadrons_subjet_ewoc = Npoint_MonteCarloSubJetEWOC(
                                                jet_algorithm, subjet_algorithm,
                                                montecarlo_process=process,
                                                montecarlo_observable=mass_squared)

        z_i_Observable = Observable(process, z_i, r'$z_i$')
        z_j_Observable = Observable(process, z_j, r'$z_j$')
        ee2hadrons_subjet_ewoc.set_weights(z_i_Observable, z_j_Observable,
                                           thetas)
        ee2hadrons_subjet_ewoc.generate_ewoc(numbins=100, binspacing=sample_method)

        # - - - - - - - - - - - - - - - - -
        # Plot the EWOC
        # - - - - - - - - - - - - - - - - -
        # Numerical
        ee2hadrons_subjet_ewoc.plot_ewoc(ax=ax, lw=2, label='Numeric' if i == 0 else None,
                                         color=color)

        # Analytic
        xis = lin_log_mixed_list(np.min(ee2hadrons_subjet_ewoc.observables),
                                 np.max(ee2hadrons_subjet_ewoc.observables),
                                 1000)
        ewocs = mass_squared_ewoc_ee2hadronsLO(xis, subjet_radius, jet_radius,
                                               energy)
        ax.plot(xis, ewocs, label='Analytic' if i == 0 else None,
                ls='dashed', color=adjust_lightness(color, 1.3), lw=2)

    # Adding subjet radius color bar
    cbar = fig.colorbar(cbar_data, ax=ax)
    cbar.ax.set_ylabel(r'$R_{\rm sub}$')

    # Formatting
    ax.set_xlabel(mass_squared.name)
    ax.set_ylabel(mass_squared.name + " EWOC")
    ax.set_title("EWOC for " + process.name)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    plt.tight_layout()
    plt.show()
