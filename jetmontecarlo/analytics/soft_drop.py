import numpy as np
from scipy.special import gamma

# Local imports for analytics
from jetmontecarlo.analytics.QCD_utils import *

# Local imports for plotting
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

############################################################
# Fixed Coupling Soft Drop Radiator:
############################################################
def softdrop_rad_fc(c, z_cut, beta, beta_sd=0,
                    jet_type='quark'):
    return CR(jet_type)*alpha_fixed / (beta*np.pi) * (
           np.log(2.*c)**2. * (c > z_cut)
           +
           (np.log(2.*z_cut)**2 - np.log(2.*z_cut)*np.log(z_cut/c)*2.)
           * (c < z_cut)
           )

def softdrop_radprime_fc(c, z_cut, beta, beta_sd=0,
                         jet_type='quark'):
    return 2. * CR(jet_type)*alpha_fixed / (c * beta*np.pi) * (
           np.log(2.*c) * (c > z_cut)
           +
           np.log(2.*z_cut) * (c < z_cut)
           )

def softdrop_sudakov_fc(c, z_cut, beta, beta_sd=0,
                        jet_type='quark', acc='LL'):
    sudakov_LL = np.exp(-softdrop_rad_fc(c, z_cut, beta, beta_sd,
                                         jet_type))
    if acc=='LL':
        return sudakov_LL
    else:
        rad_logprime = -c * softdrop_radprime_fc(c, z_cut, beta, beta_sd,
                                                 jet_type)
        me_factor = np.exp(-euler_constant*rad_logprime)/gamma(1.+rad_logprime)
        return sudakov_LL * me_factor

############################################################
# Running Coupling Soft Drop Radiator:
############################################################

# Full radiator from Soft Drop paper
def softdrop_rad_rc(c, z_cut, beta, beta_sd=0,
                    alpha=alpha_fixed, jet_type='quark'):
    # Check that the jet is a quark or gluon jet
    check_jet_type(jet_type)
    Bi = b_i_sing(jet_type)

    # ----------------------------------------------------
    # Auxiliary definitions, Soft Drop Equations A.16-19
    # ----------------------------------------------------
    prefactor = CR(jet_type) / (2.*np.pi*alpha*beta_0**2.)

    L = np.log(1./c)
    lambda_sd = 2.*alpha*beta_0*L

    Lc = np.log(1./z_cut)
    lambda_c = 2.*alpha*beta_0*Lc

    Lmu = np.log(1./MU_NP)
    lambda_mu = 2.*alpha*beta_0*Lmu

    def F1(x):
        return ((1.+beta_sd)*x-(beta+beta_sd)*Lmu+(beta-1.)*Lc)**2.\
                /((beta-1.)*(1.+beta_sd)*(beta+beta_sd))

    def F2(x):
        return ((1.+beta_sd)*x-(beta+beta_sd)*Lmu+(beta-1.)*Lc)\
         *(beta_sd*(beta+beta_sd)*Lmu + 2*Bi*(1.+beta_sd)(beta+beta_sd)\
           + Lc*(2*beta+beta*beta_sd+beta_sd) + x*beta_sd*(1.+beta_sd))\
         /(beta * (1.+beta_sd)**2. * (beta+beta_sd))

    # ----------------------------------------------------
    # Cutting out phase space
    # ----------------------------------------------------
    if beta > 1:
        bound1 = z_cut**((1.-beta)/(1.+beta_sd))\
                *MU_NP**((beta+beta_sd)/(1.+beta_sd))
        bound2 = MU_NP**beta
    elif beta < 1:
        bound1 = MU_NP**beta
        bound2 = z_cut**((1.-beta)/(1.+beta_sd))\
                *MU_NP**((beta+beta_sd)/(1.+beta_sd))
    else:
        return 0

    # ----------------------------------------------------
    # Radiator in diffferent phase space regions
    # ----------------------------------------------------
    # z_cut <= c
    def reg1():
        return prefactor * (
                W(1.-lambda_sd)/(beta-1.)
                - beta*W(1.-lambda_sd/beta)/(beta-1.)
                - 2.*alpha*beta_0*Bi*np.log(1.-lambda_sd/beta)
               )

    # bound1 <= c < z_cut
    def reg2():
        return prefactor * (
                -W(1.-lambda_c)/(1.+beta_sd)
                - beta*W(1.-lambda_sd/beta)/(beta-1.)
                - 2.*alpha*beta_0*Bi*np.log(1.-lambda_sd/beta)
                +
                (beta+beta_sd)*W(1.-(1.+beta_sd)/(beta+beta_sd)*lambda_sd
                                 -(beta-1.)/(beta+beta_sd)*lambda_c)
                              /((beta-1.)*(1.+beta_sd))
               )

    # bound2 <= c <bound1
    def reg3():
        if beta > 1:
            return prefactor * (
                    -W(1.-lambda_c)/(1.+beta_sd)
                    - beta*W(1.-lambda_sd/beta)/(beta-1.)
                    - 2.*alpha*beta_0*Bi*np.log(1.-lambda_sd/beta)
                    +
                    (1.+np.log(1.-lambda_mu)) * (
                        (beta-1.)*lambda_c
                        +(1.+beta_sd)*lambda_sd
                        -(beta+beta_sd)*lambda_mu
                    )
                    /((beta-1.)*(1.+beta_sd))
                    +
                    (beta+beta_sd)*W(1.-lambda_mu)/((beta-1.)*(1.+beta_sd))
                   )\
                   +\
                   CR(jet_type)*alpha1loop(MU_NP)*F1(L)/np.pi

        if beta < 1:
            return prefactor * (
                    -W(1.-lambda_c)/(1.+beta_sd)
                    - beta*W(1.-lambda_mu)/(beta-1.)
                    + (lambda_sd-beta*lambda_mu)*(1.+np.log(1.-lambda_mu))
                      /(beta-1.)
                    - 2.*alpha*beta_0*Bi*np.log(1.-lambda_mu)
                    +
                    (beta+beta_sd)*W(1.-(1.+beta_sd)*lambda_sd/(beta+beta_sd)\
                                     -(beta-1.)*lambda_c/(beta+beta_sd))
                      /((beta-1.)*(1.+beta_sd))
                   )\
                   +\
                   CR(jet_type)*alpha1loop(MU_NP)/np.pi\
                   *(L/beta-Lmu)*((L-beta*L_mu)/(1.-beta)+2.*Bi)

    # c < bound2
    def reg4():
        if beta > 1:
            return prefactor * (
                    -W(1.-lambda_c)/(1.+beta_sd)
                    - beta_sd*W(1.-lambda_mu)/(1.+beta_sd)
                    - 2.*alpha*beta_0*Bi*np.log(1.-lambda_mu)
                    - (1.+np.log(1.-lambda_mu))*(lambda_c+beta_sd*lambda_mu)
                      /(1.+beta_sd)
                   )\
                   + CR(jet_type)*alpha1loop(MU_NP)/np.pi * (
                    F1(beta*Lmu)
                    +
                    (L/beta-Lmu)*(2.*beta*Lc/(beta+beta_sd) + 2.*Bi
                                  + beta_sd*(L+beta*Lmu)/(beta+beta_sd))
                   )

        if beta < 1:
            return prefactor * (
                    -W(1.-lambda_c)/(1.+beta_sd)
                    - beta_sd*W(1.-lambda_mu)/(1.+beta_sd)
                    - (lambda_c + beta_sd*lambda_mu)/(1.+beta_sd)
                      *(1.+np.log(1.-lambda_mu))

                    - 2.*alpha*beta_0*Bi*np.log(1.-lambda_mu)
                   )\
                   + CR(jet_type)*alpha1loop(MU_NP)/np.pi * (
                     F2(L)
                     +
                     (1.-beta)*(beta_sd*Lmu+Lc)
                     *(2.*(1.+beta_sd)*Bi+beta_sd*Lmu+Lc)
                     /(beta * (1.+beta_sd)**2.)
                   )

    # ----------------------------------------------------
    # Full radiator
    # ----------------------------------------------------
    return reg1()*(z_cut<=c) + reg2()*(bound1<=c)*(c<z_cut)\
           + reg3()*(bound2<=c)*(c<bound1) + reg4()*(c<bound2)

############################################################
# Plotting:
############################################################

def plot_softdrop_analytic(axespdf, axescdf, bin_space,
                           z_cut, beta, beta_sd=0,
                           jet_type='quark', acc='LL',
                           fixed_coupling=True,
                           icol=-1, label='Analytic'):
    """Plot the critical emission analytic result."""
    # Preparing the bins
    if bin_space == 'lin':
        bins = np.linspace(0, .8, 500)
        xs = (bins[:-1] + bins[1:])/2.

    if bin_space == 'log':
        bins = np.logspace(-13, np.log10(.8), 500)
        xs = np.sqrt(bins[:-1]*bins[1:])

    assert 0 < z_cut < 1./2., "Invalid z_cut value " + str(z_cut)

    # Preparing the appropriate cdf
    if fixed_coupling:
        cdf = softdrop_sudakov_fc(xs, z_cut, beta, beta_sd,
                                  jet_type, acc)

        # Getting pdf from cdf by taking the numerical derivative
        _, pdf = histDerivative(cdf, bins, giveHist=True,
                                binInput=bin_space)
        if bin_space == 'log':
            pdf = xs * pdf * np.log(10) # d sigma / d log10 C

    else:
        sudakov_exp = np.exp(-softdrop_rad_rc(xs, z_cut, beta, beta_sd,
                                              jet_type=jet_type))
        if acc=='LL':
            cdf = sudakov_exp
            # Getting pdf from cdf by taking the numerical derivative
            _, pdf = histDerivative(cdf, bins, giveHist=True,
                                    binInput=bin_space)
            if bin_space == 'log':
                pdf = xs * pdf * np.log(10) # d sigma / d log10 C

        else:
            # Getting radiator and derivative
            rad = softdrop_rad_rc(xs, z_cut, beta, beta_sd, jet_type=jet_type)
            _, radprime = histDerivative(rad, bins, giveHist=True,
                                         binInput=bin_space)
            rad_logprime = -xs * radprime

            # Getting the approximate multiple emission CDF
            me_factor = np.exp(-euler_constant*rad_logprime)\
                        /gamma(1.+rad_logprime)
            cdf = sudakov_exp * me_factor

            # Getting pdf from cdf by taking the numerical derivative
            # Note that this pdf is now given at the x2 values
            _, pdf = histDerivative(cdf, bins, giveHist=True,
                                    binInput=bin_space)
            if bin_space == 'log':
                pdf = xs * pdf * np.log(10) # d sigma / d log10 C

    # Plotting
    col = compcolors[(icol, 'light')]
    axespdf[0].plot(xs, pdf, **style_dashed,
                    color=col, label=label)
    axescdf[0].plot(xs, cdf, **style_dashed,
                    color=col, label=label)

    if len(axespdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_dashed, color=col,
                        zorder=.5)
    if len(axescdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_dashed, color=col,
                        zorder=.5)
