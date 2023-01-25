import numpy as np

# Local imports
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

######################################################
# Weight Functions of test cases:
######################################################
def critUniform_c_weight(sample, zc, beta, f):
    """Provides a weight function to integrate on the sampled phase space"""
    z = sample[:,0]; theta = sample[:,1]

    Acrit = (1./2.-f*zc)/(beta+1)
    linweight = 1. / Acrit

    # Jacobian associated with the variable being uniform in
    # c_crit, rather than theta_crit:
    jacobian = theta**beta

    return linweight * jacobian


def critUniform_z_weight(sample, zc, beta, f):
    """Provides a weight function to integrate on the sampled phase space"""
    z = sample[:,0]; theta = sample[:,1]

    Acrit = (1./2.-f*zc)
    linweight = np.ones(len(sample)) / Acrit

    return linweight


def twoEmUniform_c_weight(critsample, subsample, zc,beta,f,):
    """Provides a weight function to integrate on the sampled phase space"""
    assert len(critsample)==len(subsample), \
        "Must have the same amount of critical and subsequent samples!"
    zcrit = critsample[:,0]; thetacrit = critsample[:,1];

    Acrit = (1./2.-f*zc)/(beta+1); Asub = 1./2.
    linweight = 1. / (Acrit * Asub)

    # Jacobian associated with the variable being uniform in
    # c_crit, rather than theta_crit:
    jacobian = thetacrit**beta

    return linweight * jacobian


def twoEmUniform_z_weight(critsample, subsample, zc,beta,f,):
    """Provides a weight function to integrate on the sampled phase space"""
    assert len(critsample)==len(subsample), \
        "Must have the same amount of critical and subsequent samples!"
    Acrit = (1./2.-f*zc); Asub = 1./2.
    linweight = np.ones(len(critsample)) / (Acrit * Asub)

    return linweight


def twoEmLinear_weight(critsample, subsample, zc,beta,f,):
    """Provides a weight function to integrate on the sampled phase space"""
    assert len(critsample)==len(subsample), \
        "Must have the same amount of critical and subsequent samples!"

    zcrit = critsample[:,0]; thetacrit = critsample[:,1];
    zsub  = subsample[:,0];  thetasub  = subsample[:,1]

    csub = zsub * thetasub**beta * thetasub**beta

    N = 64. * (beta+1.)*(beta+2.) / (1.-4.*zc**2.)
    linweight = N * zcrit*thetacrit * csub

    return linweight


def subtest2_weight(sample, beta):
    """Provides a weight function to integrate on the sampled phase space"""
    z = sample[:,0]; theta = sample[:,1]

    # for subtest1, linweight = 1. * np.ones(len(sample))
    c = z * theta**beta
    linweight = (c/z)**(1./beta) / (beta * c)

    return linweight

######################################################
# Analytic cumulative distributions for test cases:
######################################################
# CDFs for functions of uniform variables
# in the critical and subsequent emissions
def uniformcdf_c(C,zc,beta,f):
    """Cumulative distribution function of the sum of two random variables,
    one uniformly distributed from c_sub = 0 to 1/2 and the other uniformly
    distributed over the critical (c_crit,theta_crit) phase space"""

    Ctilde = (C/(1./2.-f*zc))**(1./beta)

    def reg1(C):
        return (
                (beta+1.)*C**2.*(1.+Ctilde*(2./(beta+1.)-1./(2.*beta+1.)-1))
                /(1./2.-zc)
                ) * (C < 1./2. - f*zc)

    def reg2(C):
        return (2.*C - (1./2.-f*zc)*(beta+1.)/(2.*beta+1.)) * (1./2. - f*zc < C)

    def reg3(C):
        Cbar = ((C-1./2.)/(1./2.-zc))**(1./beta)
        I = (C-1./2.)**2. * (-2.*Cbar/(beta+1.) + Cbar/(2.*beta+1.) - (1.-Cbar))
        fn = (beta+1.) * I / (1./2.-zc) * (1./2. < C)
        return np.nan_to_num(fn, nan=0.)


    return reg1(C) + reg2(C) + reg3(C)


def uniformcdf_z(C,zc,beta,f):
    """Cumulative distribution function of the sum of two random variables,
    one uniformly distributed from c_sub = 0 to 1/2 and the other uniformly
    distributed over the critical (z_crit,theta_crit) phase space"""

    Ctilde = (C/(1./2.-f*zc))**(1./beta)

    def reg1(C):
        return ( 2.*C *
            ((C * (1.+beta)
            -
            2.**(1./beta) * (C/(1.-2.*f*zc))**(1./beta) * beta**2.
            +
            2.**(1.+1./beta) * (C/(1.-2.*f*zc))**(1./beta)* f*zc * beta**2.))
            /((-1.+2.*f*zc)*(-1.+beta)*(1.+beta))
            ) * (0 < C) * (C < 1./2.)

    def reg2(C):
        fn = (
                (1./((-1.+2.*f*zc)*(-1.+ beta)
                * (1.+ beta)))*(1.-2*f*zc)**(-1./beta)
                *
                (
                    -(1.- 2*f*zc)**((1./beta)) +
                    4*C * (1.- 2*f*zc)**(1./beta) -
                    2*C**2 * (1.- 2*f*zc)**(1./beta) +
                    2*(1.- 2*f*zc)**(1./beta) * f*zc -
                    4*C * (1.- 2*f*zc)**(1./beta) * f*zc -
                    2*(1.- 2*f*zc)**(1./beta) * f*zc**2 +
                    2*C * (1.- 2*f*zc)**(1./beta)*beta -
                    2*C**2 * (1.- 2*f*zc)**(1./beta)*beta -
                    2*(1.- 2*f*zc)**(1./beta) * f*zc * beta +
                    2*(1.- 2*f*zc)**(1./beta) * f*zc**2 * beta -
                    (-1.+ 2*C)**(1./beta)*beta**2 +
                    2*C * (-1.+ 2*C)**(1./beta)*beta**2 -
                    2*C * (1.- 2*f*zc)**(1./beta)*beta**2 +
                    2*(-1.+ 2*C)**(1./beta) * f*zc * beta**2 -
                    4*C * (-1.+ 2*C)**(1./beta) * f*zc * beta**2 +
                    4*C * (1.- 2*f*zc)**(1./beta) * f*zc * beta**2
                )
            ) * (1./2. < C) * (C < 1.-f*zc)
        return np.nan_to_num(fn, nan=0.)

    def reg3(C): return 1. * (1.-zc < C)

    return reg1(C) + reg2(C) + reg3(C)

# CDFs for functions of uniform variables in the critical emission only
def uniformcdfcrit_c(C,zc):
    """Cumulative distribution function of a (critical emission) random
     =variable uniformly distributed from c_crit = 0 to 1/2"""
    return C/(1./2.-zc) * (0<C)*(C<1./2.-zc) + (1./2.-zc<C)

def uniformcdfcrit_z(C,zc):
    """Cumulative distribution function of the function (z-zc)theta^beta
    of (critical emission) random variables, with z uniformly distributed
    from zc to 1/2 and theta uniformly distributed from 0 to 1"""
    return (
        (C + C*np.log( (1./2.-zc)/C) )/(1./2.-zc) * (0<C)*(C<1./2.-zc)
        +
        (1./2.-zc<C)
        )

def get_testcaseCDF(vals, algorithm):
    if algorithm=='uniform_c':
        cdfAnalytic = uniformcdf_c(vals)
    elif algorithm=='uniform_z':
        cdfAnalytic = uniformcdf_z(vals)
    elif algorithm=='uniformcrit_c':
        cdfAnalytic = uniformcdfcrit_c(vals)
    elif algorithm=='uniformcrit_z':
        cdfAnalytic = uniformcdfcrit_z(vals)
    else: raise AssertionError("Unsupported test case.")


######################################################
# Functions for plotting test case analytics:
######################################################
def plotTestCase_CDF(axes, vals, algorithm, num=0):
    """Returns a plot for the cumulative distribution
    of C for the above test cases.
    """
    cdfAnalytic = get_testcaseCDF(vals, algorithm)

    axes.plot(vals, cdfAnalytic, color=compcolors[num],
                **style_dashed, label='Analytic CDF')


def plotTestCase_PDF(axes, vals, bins, binInput,
                    algorithm, num=0):
    """Returns a plot for the probability distribution
    of C for the above test cases.
    """
    cdfAnalytic = get_testcaseCDF(vals, algorithm)

    _ , pdfAnalytic = histDerivative(cdfAnalytic, bins,
                              giveHist=True, binInput=binInput)

    axes.plot(vals, pdfAnalytic, color=compcolors[num],
                **style_dashed, label='Analytic PDF')
