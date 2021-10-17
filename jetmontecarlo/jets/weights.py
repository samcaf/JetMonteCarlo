import numpy as np

# Local imports
from jetmontecarlo.analytics.QCD_utils import *
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *

######################################################
# Various weights on phase space for MC Integration:
######################################################

# ---------------------------------------------------
# Radiators:
# ---------------------------------------------------
def radiatorWeight(z, theta, jet_type, fixedcoupling=True, acc='LL'):
    """Provides a weight function to integrate on the sampled phase space"""
    # Setting coupling
    if fixedcoupling:
        alpha = alpha_fixed
    else:
        alpha = alpha_s(z, theta)

    linweight = (splittingFn(z, jet_type, acc)
                 * alpha / (theta * np.pi))

    return linweight

# ---------------------------------------------------
# Single Emissions:
# ---------------------------------------------------
def criticalEmissionWeight(z, theta, z_c, jet_type='quark',
                           fixedcoupling=True):
    """Provides a weight function to integrate on the sampled phase space"""
    # TODO: Include f dependence, accuracy dependence
    if fixedcoupling:
        linweight = critPDFAnalytic_fc_LL(z, theta, z_c=z_c,
                                          jet_type=jet_type)
    else:
        linweight = critPDFAnalytic(z, theta, z_c=z_c,
                                    jet_type=jet_type)

    return linweight

def precriticalEmissionWeight(z_pre, theta_crit, z_c, jet_type='quark',
                              fixedcoupling=True):
    """Provides a weight function to integrate on the sampled phase space"""
    if fixedcoupling:
        linweight = (2. * CR(jet_type) * alpha_fixed / np.pi
                     * np.log(R0/theta_crit) / z_pre
                     *
                     np.exp(-2. * CR(jet_type) * alpha_fixed/np.pi
                            * np.log(R0/theta_crit)
                            * np.log(z_c/(z_pre))
                           )
                    ) * (z_pre > 0.) * (z_pre < z_c)
    else:
        raise TypeError("Can only accept fixed coupling for now!")

    return linweight

# ---------------------------------------------------
# Double Emission:
# ---------------------------------------------------
def twoEmissionWeight(critsample, subsample, z_c, beta,
                      jet_type='quark', fixedcoupling=True):
    """Provides a weight function to integrate on the sampled phase space"""
    assert len(critsample) == len(subsample), \
        "Must have the same amount of critical and subsequent samples!"

    z_crit = critsample[:, 0]
    theta_crit = critsample[:, 1]
    z_sub = subsample[:, 0]
    theta_sub = subsample[:, 1]

    csub = z_sub * theta_sub**beta * theta_sub**beta

    if fixedcoupling:
        crit_pdf = (critPDFAnalytic_fc_LL(z_crit, theta_crit,
                                          z_c=z_c, jet_type=jet_type)
                    .astype('float'))
        sub_pdf = subPDFAnalytic_fc_LL(csub,
                                       beta, jet_type,
                                       maxRadius=theta_crit)

    else:
        crit_pdf = (critPDFAnalytic(z_crit, theta_crit,
                                    z_c=z_c, jet_type=jet_type)
                    .astype('float'))
        sub_pdf = subPDFAnalytic(csub,
                                 beta, jet_type,
                                 maxRadius=theta_crit)

    linweight = crit_pdf * sub_pdf

    # Jacobian associated with the the given pdf being a pdf for
    # c_sub, rather than theta_sub and z_sub:
    subjacobian = theta_sub**beta * theta_crit**beta
    # note that this is jacobian is valid at leading log only

    return linweight * subjacobian
