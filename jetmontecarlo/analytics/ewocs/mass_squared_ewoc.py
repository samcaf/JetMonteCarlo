import numpy as np

from jetmontecarlo.analytics.qcd_utils import alpha1loop, alpha_em_1loop, sum_square_quark_charges

# Diff. xsec prefactor
from jetmontecarlo.analytics.ewocs.eec import\
    ee2hadrons_LO_prefactor
prefactor = ee2hadrons_LO_prefactor

def mass_squared_ewoc(xi, rsub, Rjet,
                      energy,
                      process='e+ e- -> hadrons, LO'):
    """Calculate the mass squared of an EWOC.

    Parameters
    ----------
    xi : float
        Value at xi = m^2/Q^2 at which to calculate the EWOC.
    rsub : float
        Subjet radius.
    Rjet : float
        Jet radius.
    energy : float
        Energy of the incoming particles.
    process : str
        Process for which to calculate the mass-squared EWOC.

    Returns
    -------
    float
        Value of the mass-squared EWOC.
    """
    if process == 'e+ e- -> hadrons, LO':
        return mass_squared_ewoc_ee2hadronsLO(xi, rsub, Rjet, energy)

    raise ValueError(f'Unknown process: {process}')


def mass_squared_ewoc_ee2hadronsLO(xi, rsub, Rjet, energy):
    """Calculate the mass-squared EWOC for e+ e- -> hadrons at LO.

    Parameters
    ----------
    xi : float
        Value at xi = m^2/Q^2 at which to calculate the EWOC.
    rsub : float
        Subjet radius.
    Rjet : float
        Jet radius.
    energy : float
        Energy of the incoming particles.

    Returns
    -------
    float
        Value of the mass-squared EWOC.
    """
    # ---------------------------------
    # Useful pieces
    # ---------------------------------
    def s_factor(r):
        """A factor involving a square root that appears in
        several places.
        """
        return np.sqrt(1. + xi*(2 - 4/r**2. + xi))


    # ---------------------------------
    # Combining pieces
    # ---------------------------------
    subjet_singular = (3 + xi) * s_factor(rsub) / (6.*rsub**2.)
    subjet_singular = np.nan_to_num(subjet_singular)

    jet_singular = -(3 + xi) * s_factor(Rjet) / (6.*Rjet**2.)
    jet_singular = np.nan_to_num(jet_singular)

    mass_singular = (s_factor(Rjet) - s_factor(rsub)) *\
        (-9 + xi * (31 + xi*(-13 + 7*xi))) / (12 * xi)
    mass_singular = np.nan_to_num(mass_singular)

    subjet_log = -(2-(xi-2)*xi) * np.log(
            (1-xi+s_factor(rsub)) / (1-xi-s_factor(rsub))
        )
    subjet_log = np.nan_to_num(subjet_log)

    jet_log = (2-(xi-2)*xi) * np.log(
            (1-xi+s_factor(Rjet)) / (1-xi-s_factor(Rjet))
        )
    jet_log = np.nan_to_num(jet_log)

    # DEBUG
    # print('subjet_singular', subjet_singular)
    # print('jet_singular', jet_singular)
    # print('mass_singular', mass_singular)
    # print('subjet_log', subjet_log)
    # print('jet_log', jet_log)

    ewoc =  prefactor(energy) *\
            (subjet_singular + jet_singular +\
             mass_singular + subjet_log + jet_log)

    # DEBUG
    # print('ewoc', ewoc)

    return ewoc
