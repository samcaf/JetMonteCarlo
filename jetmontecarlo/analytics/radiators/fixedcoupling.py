import numpy as np
from jetmontecarlo.analytics.qcd_utils import *

############################################################
# Fixed Coupling Critical Radiators and pdf:
############################################################
def critRadAnalytic_fc_LL(theta, z_c, jet_type='quark'):
    """Fixed coupling, leading logarithmic critical radiator
    for a jet_type jet groomed by a grooming parameter z_c
    at angle theta.
    """
    cutoff = 1e-100
    theta = theta + cutoff
    z_c = z_c + cutoff
    rads = (-2.* CR(jet_type) * np.log(R0/theta)
            * np.log(2.*z_c) * alpha_fc/np.pi)

    return rads

def critRadPrimeAnalytic_fc_LL(theta, z_c, jet_type='quark'):
    """Derivative of the fixed coupling, leading logarithmic
    critical radiator for a jet_type jet groomed by a grooming
    parameter z_c at the angle theta.
    """
    cutoff = 1e-100
    theta = theta + cutoff
    z_c = z_c + cutoff
    return (-2.* CR(jet_type) * np.log(2.*z_c)
            * alpha_fc/(np.pi*theta))

def critRadAnalytic_fc(theta, z_c, jet_type='quark'):
    """Fixed coupling critical radiator for a jet_type jet
    groomed by a grooming parameter z_c at the angle theta.
    """
    cutoff = 1e-100
    theta = theta + cutoff
    z_c = z_c + cutoff
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return b_q_bar(z_c) * np.log(R0/theta) * alpha_fc/np.pi
    # elif jet_type == 'gluon':
    return b_g_bar(z_c) * np.log(R0/theta) * alpha_fc/np.pi

def critRadPrimeAnalytic_fc(theta, z_c, jet_type='quark'):
    """Derivative of the fixed coupling critical radiator
    for a jet_type jet groomed by a grooming parameter z_c
    at the angle theta.
    """
    cutoff = 1e-100
    theta = theta + cutoff
    z_c = z_c + cutoff
    # Minus sign relative to the actual derivative!
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return b_q_bar(z_c) * alpha_fc/(np.pi * theta)
    # elif jet_type == 'gluon':
    return b_g_bar(z_c) * alpha_fc/(np.pi * theta)

def critPDFAnalytic_fc_LL(z, theta, z_c, jet_type='quark'):
    """Fixed coupling, leading logarithmic critical probability
    distribution function encoding the probability of a splitting
    of energy fraction z and angle theta for a jet_type jet
    groomed by a grooming parameter z_c.
    """
    cutoff = 1e-100
    theta = theta + cutoff
    z_c = z_c + cutoff
    return (2.* CR(jet_type) * alpha_fc / (np.pi*theta*z)
            * np.exp(np.log(1./(2.*z_c)) * np.log(theta/R0)
                     * 2.* CR(jet_type) * alpha_fc/np.pi)
            * (z_c < z) * (z < 1./2.)
           )

############################################################
# Fixed Coupling Subsequent Radiators:
############################################################
def subRadAnalytic_fc_LL(C, beta, jet_type='quark', maxRadius=1.):
    """Fixed coupling, leading logarithmic subsequent radiator
    for a jet_type jet as a function of the observable C,
    indicating a generalized jet energy correlation function.

    Assumes that angles of the subsequent emissions are less than
    maxRadius
    """
    cutoff = 1e-100
    C = C + cutoff
    rad = (CR(jet_type)*alpha_fc/(beta*np.pi)
           * np.log(2.*C / maxRadius**beta)**2.
          )
    return rad * (C < maxRadius**beta/2.)

def subRadPrimeAnalytic_fc_LL(C, beta, jet_type='quark', maxRadius=1.):
    """Derivative w.r.t. C of the fixed coupling, leading
    logarithmic subsequent radiator for a jet_type jet
    as a function of the observable
        C = C_beta,
    indicating a generalized jet energy correlation function.

    Assumes that angles of the subsequent emissions are less than
    maxRadius
    """
    cutoff = 1e-100
    C = C + cutoff
    dv_rad_c = (2.*CR(jet_type)*alpha_fc/(beta*np.pi)
                * np.log(2.*C / maxRadius**beta)
                / C
               )
    return dv_rad_c * (C < maxRadius**beta/2.)

def subRadAnalytic_fc(C, beta=2., jet_type='quark'):
    """Fixed coupling subsequent radiator
    for a jet_type jet as a function of the observable
    C = C_beta,
    indicating a generalized jet energy correlation function.

    Assumes that angles of the subsequent emissions are
    less than maxRadius = 1.
    """
    cutoff = 1e-100
    C = C + cutoff
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return CF*(
            9. - 18.*C+np.pi**2.
            -12.*np.log(1.-C)*np.log(C)
            +6.*np.log(C)**2.
            +9.*np.log(2.*C)
            -12.*polylog_vec(2., 1.-C)
            ) * alpha_fc/(6.*beta*np.pi)

    # elif jet_type == 'gluon':
    return (
        137.*CA - 288.*C*CA + 36.*C**2.*CA
        -16.*C**3.*CA - 12.*CA*np.pi**2. - 88.*N_F*TF
        +144.*C*N_F*TF - 72.*C**2.*N_F*TF + 32.*C**3.*TF
        +72.*CA*np.log(C)**2. + 132.*CA*np.log(2.*C)
        -48.*N_F*TF*np.log(2.*C) + 144.*CA*polylog_vec(2., C)
        ) * alpha_fc/(72*beta*np.pi)

def subRadPrimeAnalytic_fc(C, beta=2., jet_type='quark'):
    """Derivative w.r.t. -C of the fixed coupling
    subsequent radiator for a jet_type jet
    as a function of the observable C_beta, indicating a
    generalized jet energy correlation function.

    Assumes that angles of the subsequent emissions are
    less than maxRadius = 1.
    """
    cutoff = 1e-100
    C = C + cutoff
    # Minus sign relative to the actual derivative!
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return (-18.+9./C - 12.*np.log(1.-C)/C
                +12.*np.log(C)/C) * CF * alpha_fc/(6*beta*np.pi)
    #elif jet_type == 'gluon':
    return (-288.*CA+132.*CA/C + 72.*C*CA - 48.*C**2.*CA
            +144.*N_F*TF - 48*N_F*TF/C - 144.*C*N_F*TF
            +96*C**2.*N_F*TF - 144.*CA*np.log(1.-C)/C
            +144.*CA*np.log(C)/C) * alpha_fc/(72*beta*np.pi)

def subPDFAnalytic_fc_LL(C, beta, jet_type='quark', maxRadius=1.):
    """Probability distribution function for the
    generalized jet energy correlation function C_beta
    of a subsequent emission of a jet_type jet.

    Assumes that angles of the subsequent emissions are
    less than maxRadius.
    """
    cutoff = 1e-100
    C = C + cutoff
    sudakov_factor = np.exp(-subRadAnalytic_fc_LL(C, beta,
                                                  jet_type, maxRadius))
    derivative = -subRadPrimeAnalytic_fc_LL(C, beta,
                                            jet_type, maxRadius)
    return sudakov_factor * derivative

############################################################
# Pre-critical Radiators:
############################################################
def preRadAnalytic_fc_LL(z_pre, theta_crit, z_cut,
                         jet_type='quark', maxRadius=1.):
    """Fixed coupling, leading logarithmic pre-critical radiator
    for a jet_type jet as a function of energy fraction z_pre
    of a pre-critical emission and a critical angle theta_crit.
    """
    cutoff = 1e-100
    rad = (2*CR(jet_type)*alpha_fc/np.pi
           * np.log(z_pre / z_cut) * np.log(theta_crit/maxRadius)
          )
    return rad * (0. < z_pre) * (z_pre < z_cut)\
            * (0. < theta_crit) * (theta_crit < maxRadius)
