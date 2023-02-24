import numpy as np

from jetmontecarlo.analytics.qcd_utils import alpha1loop


# ---------------------------------
# EEC Utilities
# ---------------------------------

# - - - - - - - - - - - - - - - - -
# Analytic Expressions
# - - - - - - - - - - - - - - - - -
def eec(z, mu,
        acc: str = 'nnlo'):
    """The energy energy correlator as a function of
      z = (1-cos chi)/2
    .
    """
    if acc == 'lo':
        return eec_lo(z, mu)

    if acc == 'nnlo':
        return eec_nnlo(z, mu)

    raise AssertionError(f"Invalid accuracy {acc}")


def eec_lo(z, mu):
    """The energy energy correlator as a function of
      z = (1-cos chi)/2
    .
    """
    raise AssertionError("Not implemented yet.")


def eec_nnlo(z, mu):
    """NNLO EEC."""
    a_s = alpha1loop(mu) / (4 * np.pi)
    # Not quite right yet, need to run at higher acccuracy

    z = z.astype(float)

    nlo_piece  = -11.5333*np.log(z) + 81.4809
    nnlo_piece = 45.1489*np.log(z)**2. - 1037.73*np.log(z) + 2871.36

    return (2.*a_s + a_s**2.*nlo_piece + a_s**3.*nnlo_piece)/(z*(1-z))


# - - - - - - - - - - - - - - - - -
# Plotting
# - - - - - - - - - - - - - - - - -
def plot_eec_analytic(ax, observable : str,
                      energy: float = 2000,
                      acc: str = 'nnlo'):
    """Plots an analytic EEC to the given accuracy on the given axes."""

    # Plot the full analytic EEC
    if observable in ['z', 'zs']:
        xs = np.concatenate((np.linspace(1e-8, 1, 250),
                             np.logspace(-8, 0, 250)))
        xs = np.sort(xs)
        ax.plot(xs, eec(xs, energy, acc), 'k--',
                linewidth=2, zorder=4)
    if observable in ['cos', 'costhetas']:
        xs = np.linspace(-1, 1, 500)
        ax.plot(xs, eec((1-xs)/2, energy, acc), 'k--',
                linewidth=2, zorder=4)
    return
