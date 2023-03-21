import numpy as np

from jetmontecarlo.analytics.qcd_utils import alpha1loop, alpha_em_1loop, sum_square_quark_charges


# ---------------------------------
# Prefactor for LO EEC/EWOC
# ---------------------------------
def ee2hadrons_LO_prefactor(energy):
    """A prefactor for the LO EEC/EWOC,
    which emerges in the calculation of the
    (differential) cross section for
    e+ e- -> q qbar g
    """
    alpha_s = alpha1loop(energy)
    prefactor = 8 * energy**2. * alpha_s * alpha_em_1loop(energy)**2.\
               * sum_square_quark_charges(energy)
    return prefactor


# ---------------------------------
# EEC Utilities
# ---------------------------------

# - - - - - - - - - - - - - - - - -
# Analytic Expressions
# - - - - - - - - - - - - - - - - -
def eec(z, mu, acc: str = 'nnlo', **kwargs):
    """The energy energy correlator as a function of
      z = (1-cos chi)/2
    .
    """
    if acc == 'lo':
        return eec_lo(z, mu, **kwargs)

    if acc == 'nnlo':
        return eec_nnlo(z, mu)

    raise AssertionError(f"Invalid accuracy {acc}")


def eec_lo(z, mu, Rjet=None, rsub=None,
           singular_only=False):
    """The energy energy correlator as a function of
      z = (1-cos chi)/2
    Note that the normalization is different here
    """
    # Prefactor
    prefactor = ee2hadrons_LO_prefactor(mu)

    if singular_only:
        eec_val = (prefactor/2) * (\
            (3/2)/z + (-3 +(-2)*np.log(1-z))/(1-z)\
            + (-10)*np.log(1-z)
        )

    else:
        # Pieces
        J12_log = np.log(1-z)/(1-z) * (-8/z**5 + 6/z**4)
        J12_pole = 1/(1-z) * (-8/z**4 + 2/z**3 + (1/3)/z**2)
        J12 = J12_log + J12_pole

        J13_log = np.log(1-z) * (13/z**5 + (-14)/z**4 + 4/z**3)
        J13_pole = 1/(1-z) * (13/z**4 + (-41/2)/z**3 + (53/6)/z**3)
        J13 = J13_log + J13_pole

        J23 = J13

        # Final
        eec_val = (prefactor/2) * (J12 + J13 + J23)

    if Rjet is not None:
        eec_val *= (2 - 2*z > np.cos(Rjet))
    if rsub is not None:
        eec_val *= (2 - 2*z < np.cos(rsub))

    return eec_val


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
