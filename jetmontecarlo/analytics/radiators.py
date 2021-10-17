import numpy as np

from jetmontecarlo.analytics.QCD_utils import *

############################################################
# Subsequent Emissions, Running Coupling:
############################################################
# ---------------------------------------------------
# Soft Collinear Pieces
# ---------------------------------------------------
def subSC1(C, beta=2, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the subsequent, running
    coupling radiator, as a function of C.

    Kicks in at high C, C > MU_NP.
    """
    sc = beta/(2.*alpha*beta_0*(beta - 1.)) * (
        W(1. + Lambda(1./2., alpha))*(1.-1./beta)
        - W(1. + Lambda(C/2.**(beta-1.), alpha)/beta)
        + W(1. + Lambda(C, alpha))/beta
        )
    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
        return sc
    # elif jet_type == 'gluon':
    sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)
    return sc

def subSC2(C, beta=2, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the subsequent, running
    coupling radiator, as a function of C.

    Kicks in at medium C, MU_NP^beta 2^(beta-1) < C < MU_NP.
    """
    sc_p = beta/(2.*alpha*beta_0*(beta - 1.)) * (
        W(1. + Lambda(1./2., alpha)*(1.-1./beta)
          + Lambda(MU_NP, alpha)/beta
          )
        - W(1. + Lambda(1./2., alpha)*(1.-1./beta)
            + Lambda(C, alpha)/beta
           )
        + Lambda(C/MU_NP, alpha)/beta
        * (1 + np.log(1 + Lambda(MU_NP, alpha)))
        )
    sc_np = 2.*alpha*beta_0 * (np.log(MU_NP/C))**2. / (2.*(beta-1.))
    sc = sc_p + sc_np

    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)

    return sc

def subSC3(C, beta=2, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the subsequent, running
    coupling radiator, as a function of C.

    Kicks in at small values of C, C < MU_NP^beta 2^(beta-1).
    """
    sc = (
        np.log(2.*C)**2./beta - beta * np.log(2.*MU_NP)**2.
        ) * 2.*alpha*beta_0 / 2.
    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)

    return sc

# ---------------------------------------------------
# Hard Collinear Pieces
# ---------------------------------------------------
def subHC1(C, beta=2, jet_type='quark', alpha=alpha_fixed):
    """A hard-collinear piece of the subsequent, running
    coupling radiator which comes from non-singular
    pieces of the splitting function. It is a function of C.

    Kicks in at high C, C >  MU_NP^beta 2^(beta-1)
    """
    hc = (
        np.log(1 + Lambda(1./2., alpha))
        - np.log(1 + Lambda(C/2.**(beta-1.), alpha)/beta)
        )

    check_jet_type(jet_type)
    if jet_type == 'quark':
        hc = (b_q_bar(C)*hc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        hc = (b_g_bar(C)*hc)*alpha/(np.pi*2.*alpha*beta_0)

    return hc

def subHC2(C, beta=2, jet_type='quark', alpha=alpha_fixed):
    """A hard-collinear piece of the subsequent, running
    coupling radiator which comes from non-singular
    pieces of the splitting function. It is a function of C.

    Kicks in at low C, C <  MU_NP^beta 2^(beta-1)
    """
    hc = (
        2.*alpha*beta_0
        * np.log(MU_NP**beta*2.**(beta-1)/C)
        /
        ((1.+Lambda(MU_NP, alpha)) * beta)
        )

    check_jet_type(jet_type)
    if jet_type == 'quark':
        hc = (b_q_bar(C)*hc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        hc = (b_g_bar(C)*hc)*alpha/(np.pi*2.*alpha*beta_0)

    return hc

def subRadAnalytic(C, beta=2, jet_type='quark', maxRadius=1.):
    """Full subsequent, running coupling radiator as a function of C."""
    C = C / maxRadius**beta
    alpha = alpha1loop(np.maximum(P_T * maxRadius, 1.))

    soft_collinear = (
        (C > MU_NP)*subSC1(C, beta, jet_type, alpha)
        +
        (MU_NP**beta * 2**(beta-1.) < C) * (C < MU_NP)
        * (subSC1(MU_NP, beta, jet_type, alpha)
           + subSC2(C, beta, jet_type, alpha))
        +
        (C < MU_NP**beta * 2**(beta-1.))
        * (subSC1(MU_NP, beta, jet_type, alpha)
           + subSC2(MU_NP**beta * 2**(beta-1.),
                    beta, jet_type, alpha)
           + subSC3(C, beta, jet_type, alpha))
        )
    hard_collinear = (
        (C > MU_NP**beta * 2**(beta-1.))
        * subHC1(C, beta, jet_type, alpha)
        +
        (C < MU_NP**beta * 2**(beta-1.))
        * (subHC1(MU_NP**beta * 2**(beta-1.),
                  beta, jet_type, alpha)
           + subHC2(C, beta, jet_type, alpha))
        )

    radiator = soft_collinear + hard_collinear
    return radiator * (C < 1./2.)


############################################################
# Critical Emission, Running Coupling:
############################################################
# ---------------------------------------------------
# Soft Collinear Pieces
# ---------------------------------------------------
def critSC1(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the critical, running
    coupling radiator, as a function of theta.

    Kicks in at high theta, theta > MU_NP/z_c
    """
    sc = (W(1. + Lambda(1./2., alpha))
          - W(1. + Lambda(z_c, alpha))
          - W(1. + Lambda(theta/2., alpha))
          + W(1. + Lambda(z_c*theta, alpha))
         ) / (2.*alpha*beta_0)

    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)

    return sc

def critSC2(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the critical, running
    coupling radiator, as a function of theta.

    Kicks in at medium theta, 2 MU_NP < theta < MU_NP/z_c
    """
    sc_p = (
        W(1. + Lambda(MU_NP/(2.*z_c), alpha))
        - W(1 + Lambda(theta/2., alpha))
        + Lambda(theta*z_c/MU_NP, alpha)
        * (1. + np.log(1 + Lambda(MU_NP, alpha)))
        ) / (2.*alpha*beta_0)
    sc_np = ((2.*alpha*beta_0) * np.log(theta*z_c/MU_NP)**2.
             / (2.*(1. + Lambda(MU_NP, alpha)))
            )
    sc = sc_p + sc_np

    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)

    return sc

def critSC3(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """A soft-collinear piece of the critical, running
    coupling radiator, as a function of theta.

    Kicks in at low theta, theta < 2 MU_NP
    """
    sc = (-(2.*alpha*beta_0)*np.log(2.*z_c)
          * np.log(2.*MU_NP/theta)
          /
          (1. + Lambda(MU_NP, alpha)))

    check_jet_type(jet_type)
    if jet_type == 'quark':
        sc = (2*CF*sc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        sc = (2*CA*sc)*alpha/(np.pi*2.*alpha*beta_0)

    return sc

# ---------------------------------------------------
# Hard Collinear Pieces
# ---------------------------------------------------
def critHC1(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """A hard-collinear piece of the critical, running
    coupling radiator which comes from non-singular
    pieces of the splitting function.
    It is a function of theta.

    Kicks in at high theta, theta > 2 MU_NP
    """
    hc = -np.log(1. + Lambda(theta, alpha))

    check_jet_type(jet_type)
    if jet_type == 'quark':
        hc = (b_q_bar(z_c)*hc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        hc = (b_g_bar(z_c)*hc)*alpha/(np.pi*2.*alpha*beta_0)

    return hc

def critHC2(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """A hard-collinear piece of the critical, running
    coupling radiator which comes from non-singular
    pieces of the splitting function.
    It is a function of theta.

    Kicks in at low theta, theta < 2 MU_NP
    """
    hc = Lambda(MU_NP*2./theta, alpha) / (1.+Lambda(MU_NP, alpha))

    check_jet_type(jet_type)
    if jet_type == 'quark':
        hc = (b_q_bar(z_c)*hc)*alpha/(np.pi*2.*alpha*beta_0)
    elif jet_type == 'gluon':
        hc = (b_g_bar(z_c)*hc)*alpha/(np.pi*2.*alpha*beta_0)

    return hc

# ---------------------------------------------------
# Full Radiator
# ---------------------------------------------------
def critRadAnalytic(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """Full critical, running coupling radiator
    as a function of theta.
    """
    soft_collinear = (
        (theta > MU_NP/z_c) * critSC1(theta, z_c, jet_type, alpha)
        +
        (2.*MU_NP < theta) * (theta < MU_NP/z_c)
        * (critSC1(MU_NP/z_c, z_c, jet_type, alpha)
           + critSC2(theta, z_c, jet_type, alpha))
        +
        (theta < 2.*MU_NP)
        * (critSC1(MU_NP/z_c, z_c, jet_type, alpha)
           + critSC2(2.*MU_NP, z_c, jet_type)
           + critSC3(theta, z_c, jet_type))
        )

    hard_collinear = (
        (theta > 2.*MU_NP) * critHC1(theta, z_c, jet_type, alpha)
        +
        (theta < 2.*MU_NP)
        * (critHC1(2.*MU_NP, z_c, jet_type, alpha)
           + critHC2(theta, z_c, jet_type, alpha))
        )

    radiator = soft_collinear + hard_collinear
    return radiator



############################################################
# Pre-Critical Emissions, Running Coupling with no freezing:
############################################################
# ---------------------------------------------------
# Soft Collinear Pieces
# ---------------------------------------------------
def preSC_nofreeze(z_pre, theta_crit, z_cut,
                   alpha=alpha_fixed, jet_type='quark'):
    check_jet_type(jet_type)
    sc = W(1.+2.*alpha*beta_0*np.log(z_cut))\
         - W(1.+2.*alpha*beta_0*np.log(z_pre))\
         - W(1.+2.*alpha*beta_0*np.log(z_cut/theta_crit))\
         + W(1.+2.*alpha*beta_0*np.log(z_pre/theta_crit))
    prefactor = CR(jet_type) / (2. * alpha * beta_0**2. * np.pi)
    return -prefactor * sc

def preRadAnalytic_nofreeze(z_pre, theta_crit, z_cut,
                            alpha=alpha_fixed, jet_type='quark'):
    return preSC_nofreeze(z_pre, theta_crit, z_cut,
                          alpha=alpha_fixed, jet_type='quark')
# ---------------------------------------------------
# Hard Collinear Pieces
# ---------------------------------------------------



############################################################
# Additonal Info:
############################################################

# ---------------------------------------------------
# Radiator Derivatives:
# ---------------------------------------------------
## Derivatives of Radiators (_not_ logarithmic derivatives):
# critRadPrimeAnalytic = dRcrit/dtheta
# subRadPrimeAnalytic  = dRsub/dc

def critRadPrimeAnalytic(theta, z_c, jet_type='quark', alpha=alpha_fixed):
    """Derivative of the running coupling critical radiator
    for a jet_type jet groomed by a grooming parameter z_c
    at the angle theta.
    """
    m = np.maximum(z_c, np.minimum(MU_NP/theta, 1./2.))
    sc = (
        np.log((1+2.*alpha*beta_0*np.log(theta/2.))
               /(1+2.*alpha*beta_0*np.log(m*theta)))
        /(2.*alpha*beta_0)
        +
        np.log(m/z_c) / (1 + 2.*alpha*beta_0*np.log(MU_NP))
        )

    hc = 1. / (1. + 2.*alpha*beta_0
               *np.log(np.maximum(theta/2., MU_NP)))

    check_jet_type(jet_type)
    if jet_type == 'quark':
        return alpha*(2.*CF*sc + b_q_bar(z_c)*hc) / (np.pi * theta)
    # elif jet_type == 'gluon':
    return alpha*(2.*CA*sc + b_g_bar(z_c)*hc) / (np.pi * theta)


def subRadPrimeAnalytic(c, beta=2., jet_type='quark', maxRadius=1.):
    """Derivative w.r.t. C of the running coupling subsequent
    radiator for a jet_type jet as a function of the observable
        C = C_beta,
    indicating a generalized jet energy correlation function.

    Assumes that angles of the subsequent emissions are less than
    maxRadius.
    """
    c = c / maxRadius**beta
    jac = 1./maxRadius**beta
    alpha = alpha1loop(np.maximum(P_T * maxRadius, 1.))

    m = np.maximum(c,
                   np.minimum((MU_NP**beta / c)**(1./(beta-1.)), 1./2.))
    sc = (
        np.log((1+2.*alpha*beta_0*np.log(c/2.**(beta-1.))/beta)
               /(1+2.*alpha*beta_0*np.log(c*m**(beta-1.))/beta))
        * beta/((beta - 1.)*2.*alpha*beta_0)
        +
        np.log(m/c) / (1 + 2.*alpha*beta_0*np.log(MU_NP))
        )
    hc = 1. / (1. + 2.*alpha*beta_0 * np.log(
                np.maximum(2**(-(beta-1.)/beta)*c**(1/beta), MU_NP))
              )

    check_jet_type(jet_type)
    if jet_type == 'quark':
        return jac * alpha*(2.*CF*sc + b_q_bar(c)*hc) / (np.pi * beta * c)
    # elif jet_type == 'gluon':
    return jac * alpha*(2.*CA*sc + b_g_bar(c)*hc) / (np.pi * beta * c)


# ---------------------------------------------------
# Critical emission, joint (z, theta) distribution:
# ---------------------------------------------------
def critPDFAnalytic(z, theta, z_c, jet_type='quark'):
    """Running coupling critical probability distribution
    function encoding the probability of a splitting
    of energy fraction z and angle theta for a jet_type jet
    groomed by a grooming parameter z_c.
    """
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return (
            q_splitting_bar(z) * alpha_s(z, theta)
            * np.exp(-1.*critRadAnalytic(theta, z_c, jet_type))
            / (np.pi*theta)
            )
    # elif jet_type == 'gluon':
    return (
        g_splitting_bar(z) * alpha_s(z, theta)
        * np.exp(-1.*critRadAnalytic(theta, z_c, jet_type))
        / (np.pi*theta)
        )

# ---------------------------------------------------
# Subsequent emission, joint (z, theta) distribution:
# ---------------------------------------------------
def subPDFAnalytic(c, beta=2., jet_type='quark', maxRadius=1.):
    """Probability distribution function for the
    generalized jet energy correlation function C_beta
    of a subsequent emission of a jet_type jet.

    Assumes that angles of the subsequent emissions are
    less than maxRadius.
    """
    return (
        subRadPrimeAnalytic(c, beta, jet_type, maxRadius=maxRadius)
        * np.exp(
            -subRadAnalytic(c, beta, jet_type, maxRadius=maxRadius)
            )
        )
