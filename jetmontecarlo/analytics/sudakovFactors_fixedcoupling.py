import warnings
import numpy as np
from scipy.special import *

# Local imports
from jetmontecarlo.analytics.QCD_utils import *

# ---------------------------------------------------
# Leading Log (critical emission only) Sudakov Factor:
# ---------------------------------------------------
def critSudakov_fc_LL(C, z_c, beta, jet_type, f=1., alpha=alpha_fc):
    """Sudakov factor for a single critical emission
    at fixed coupling.
    """

    eta = (2.*CR(jet_type)*alpha)/(beta*np.pi) * np.log(1./(2.*f*z_c))

    def oneEmissionSCFixed(C, f, z_c, beta, jet_type):
        """Soft-Collinear contribution to the fixed coupling,
        single emission Sudakov exponent/cumulative distribution"""
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Changing C and z_c for the case of f =/= 1
        C = C/(1.-(1.-f)*z_c)
        z_c = f*z_c

        # Main piece of the radiator
        simple = (C / (1./2. - z_c))**eta

        # Corrections to the radiator
        prefac = ((2. * CR(jet_type) * alpha / (beta * np.pi))
                  * (C/z_c)**eta / (eta*(eta-1.)))

        c1 = (z_c/C)**(eta-1.) * hyp2f1_vec(1., 1.-eta,
                                            2.-eta, -C/z_c)
        c2 = (z_c/C)**eta * (eta-1.) * np.log(1. + C/z_c)
        c3 = (
            -(z_c/(1./2.-z_c))**(eta-1.)
            * hyp2f1_vec(1., 1.-eta, 2.-eta, 1.-1./(2.*z_c))
            )
        c4 = -(z_c/(1./2.-z_c))**eta * (eta-1.) * np.log(1./(2.*z_c))

        corr = prefac * (c1 + c2 + c3 + c4)

        return simple + corr

    fullSudakov = oneEmissionSCFixed(C, f, z_c, beta, jet_type)

    Cmax = (1./2.-f*z_c)
    warnings.filterwarnings("default", category=RuntimeWarning)

    return fullSudakov * (C < Cmax) + 1. * (C > Cmax)
