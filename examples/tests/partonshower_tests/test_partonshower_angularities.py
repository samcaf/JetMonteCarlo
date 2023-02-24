import numpy as np
from scipy.special import gamma

# Local imports:
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.hist_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.utils.partonshower_utils import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.numerics.observables import *
from jetmontecarlo.analytics.qcd_utils import *
from jetmontecarlo.analytics.radiators.running_coupling import *

###########################################
# Parameters and setup:
###########################################
# Monte Carlo
NUM_EVENTS = 1e2
NUM_BINS = min(int(NUM_EVENTS/20), 50)

# Physics
BIN_SPACE = 'log'
JET_TYPE = 'quark'
ACCURACY = 'MLL'

# BETA_LIST = [.5, 1.5, 2., 4.]
BETA_LIST = [2., 4., 6., 8.]

FIX_BETA = 0.

# Plotting
if BIN_SPACE == 'lin':
    ylim_1 = (0, 1.)
    ylim_2 = (0, 1.)
    xlim = (0, .5)
if BIN_SPACE == 'log':
    ylim_1 = (0, .15)
    ylim_2 = (0, .25)
    xlim = (1e-8, .5)
    if ACCURACY in ['MLL', 'ME']:
        ylim_1 = (0, .25)
        ylim_2 = (0, .25)
        xlim = (4e-5, .5)

INFO = '{:.0e} {} jets'.format(NUM_EVENTS, JET_TYPE)
SHOW_FIGTEXT = False
if SHOW_FIGTEXT:
    LABEL_PS = 'Parton shower'
else:
    LABEL_PS = 'Parton shower,\n'+INFO

SHOW_PLOTS = False
SAVE_PLOTS = True

FREEZE_COUPLING = False

comparisonplot = False

###########################################
# Simple Tools:
###########################################
def verify_bin_space(bin_space):
    """Verifying valid bin_space."""
    assert(bin_space in ['lin', 'log']), \
    "'bin_space' must be 'lin' or 'log', but is" + bin_space

def verify_accuracy(acc):
    """Verifying valid accuracy."""
    assert(acc in ['LL', 'MLL', 'ME']), \
    "'acc' must be 'LL', 'MLL', or 'ME', but is " + acc

gamma_vec = np.frompyfunc(gamma, 1, 1)


###########################################
# Analytics:
###########################################
# -----------------------------------------
# Analytic Pieces, LL
# -----------------------------------------
# Analytic angularity distributions at LL
def ang_factor_LL(beta=2., jet_type='quark'):
    """Useful factor for angularity with parameter beta."""
    return CR(jet_type) * alpha_fixed / (beta * np.pi)

def pdf_ang_LL(ang, beta=2., jet_type='quark', R=1.):
    """LL pdf for angularity with parameter beta."""
    return (-2.*ang_factor_LL(beta, jet_type)
            *np.log(2.*ang/R**beta) / ang
            * np.exp(-ang_factor_LL(beta, jet_type)
                     * np.log(ang*2./R**beta)**2.)
            )

def cdf_ang_LL(ang, beta=2., jet_type='quark', R=1.):
    """LL cdf for angularity with parameter beta."""
    return np.exp(-ang_factor_LL(beta, jet_type)
                  * np.log(ang*2./R**beta)**2.)

# -----------------------------------------
# Analytic Pieces, MLL
# -----------------------------------------
# ----------
# Radiator:
# ----------
# Stolen from jetmontecarlo.analytics.radiators, with tiny changes
def ang_rad_MLL_NP(ang, beta=2, jet_type='quark', maxRadius=1.):
    """The full radiator for ungroomed angularity, including
    non-singular pieces of parton splitting functions and
    non-perturbative freezing of the QCD coupling below the scale
    MU_NP.
    """
    ang = ang / maxRadius**beta
    alphaS = alpha1loop(np.maximum(P_T * maxRadius, 1.))

    soft_collinear = (
        (ang > MU_NP)*subSC1(ang, beta, jet_type, alphaS)
        +
        (MU_NP**beta * 2**(beta-1.) < ang) * (ang < MU_NP)
        * (subSC1(MU_NP, beta, jet_type, alphaS)
           + subSC2(ang, beta, jet_type, alphaS))
        +
        (ang < MU_NP**beta * 2**(beta-1.))
        * (subSC1(MU_NP, beta, jet_type, alphaS)
           + subSC2(MU_NP**beta * 2**(beta-1.),
                    beta, jet_type, alphaS)
           + subSC3(ang, beta, jet_type, alphaS))
        )
    hard_collinear = (
        (ang > MU_NP**beta*2**(beta-1.))
        * subHC1(ang, beta, jet_type, alphaS)
        +
        (ang < MU_NP**beta*2**(beta-1.))
        * (subHC1(MU_NP**beta * 2**(beta-1.),
                  beta, jet_type, alphaS)
           + subHC2(ang, beta, jet_type, alphaS))
        )

    radiator = soft_collinear + hard_collinear
    return radiator

def ang_radprime_MLL_NP(ang, beta=2., jet_type='quark', maxRadius=1.):
    """Derivative w.r.t. angularity of the running coupling subsequent
    radiator for a jet_type jet as a function of the observable
        ang = e_beta,
    indicating an ungroomed angularity.

    Assumes that angles of the subsequent emissions are less than
    maxRadius.
    """
    ang = ang / maxRadius**beta
    jac = 1./maxRadius**beta
    alpha = alpha1loop(np.maximum(P_T * maxRadius, 1.))

    m = np.maximum(ang,
                   np.minimum((MU_NP**beta / ang)**(1./(beta-1.)), 1./2.))
    sc = (
        np.log((1+2.*alpha*beta_0*np.log(ang/2.**(beta-1.))/beta)
               /(1+2.*alpha*beta_0*np.log(ang*m**(beta-1.))/beta))
        * beta/((beta - 1.)*2.*alpha*beta_0)
        +
        np.log(m/ang) / (1 + 2.*alpha*beta_0*np.log(MU_NP))
        )
    hc = 1. / (1. + 2.*alpha*beta_0 * np.log(
                np.maximum(2**(-(beta-1.)/beta)*ang**(1/beta), MU_NP))
              )

    check_jet_type(jet_type)
    if jet_type == 'quark':
        return jac * alpha*(2.*CF*sc + b_q_bar(ang)*hc) / (np.pi * beta * ang)
    # elif jet_type == 'gluon':
    return jac * alpha*(2.*CA*sc + b_g_bar(ang)*hc) / (np.pi * beta * ang)

# -----------------------------------------
# Analytic Pieces, MLL, with singular splitting
# -----------------------------------------
# ----------
# Radiator:
# ----------
def ang_rad_radprime_MLL(ang, beta=2, jet_type='quark',
                         maxRadius=1.):
    """MLL radiator for angularity with parameter beta."""
    alphaS = alpha1loop(P_T*maxRadius/2.)
    # Normalizing angularity so that analytic
    # expressions are nicer
    ang_n = 2.*ang / maxRadius**beta

    nga = 2.*beta_0*alphaS
    prefactor = CR(jet_type)/(nga*beta_0*np.pi*(beta-1.))

    factor_1 = 1. + nga*np.log(ang_n)
    factor_2 = 1. + nga*np.log(ang_n)/beta

    cdf_factor_1 = W(factor_1)
    cdf_factor_2 = - beta * W(factor_2)

    rad = prefactor * (cdf_factor_1+cdf_factor_2)

    # We have pdf_factor_i = d(cdf_factor_i)/(d ang)
    # -> pdf_factor_i = (2/radius^beta) d(cdf_factor_i)/(d ang_n)
    pdf_factor_1 = ((2./maxRadius**beta)
                    *nga*(1. + np.log(factor_1)) / ang_n)
    pdf_factor_2 = (-(2./maxRadius**beta)
                    *nga*(1. + np.log(factor_2)) / ang_n)

    # radprime = d rad / d ang
    radprime = prefactor * (pdf_factor_1+pdf_factor_2)

    return rad, radprime

def quickplot_rad_radprime():
    """Makes a simple plot of the radiator and
    its derivative, as defined above.
    Was useful as a quick double check of some bugs
    and minus signs.
    """
    angs = np.logspace(-8, np.log10(1/2), 1000)
    rs, rps = ang_rad_radprime_MLL(angs,
                                   beta=2.,
                                   jet_type='quark')
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(angs, rs, color='blue')
    plt.plot(angs, -angs*rps, color='red')
    ax1.set_xscale('log')
    plt.show()

# ----------
# Get CDF:
# ----------
# Defining Euler gamma
EULER_GAMMA = 0.57721566490153286061

def distrib_ang_MLL(angs, beta=2., jet_type='quark', R=1.,
                    binInput='log', acc='MLL'):
    """MLL or ME pdf and cdf for angularity with parameter beta."""
    # NOTE: This uses numerical derivatives. Providing more
    #       angularities will therefore give better accuracy

    # Finding angularities between the given angularities,
    # to use the histDerivative method.
    assert acc in ['MLL', 'ME'], \
        "Accuracy must be 'MLL' or 'ME', but is " + acc

    if binInput == 'lin':
        angs_central = (angs[:-1] + angs[1:])/2.
    elif binInput == 'log':
        angs_central = np.sqrt(angs[:-1] * angs[1:])

    if FREEZE_COUPLING:
        rad = ang_rad_MLL_NP(angs_central, beta=beta,
                             jet_type='quark',
                             maxRadius=1.)
        radprime = -ang_radprime_MLL_NP(angs_central, beta=beta,
                                        jet_type='quark',
                                        maxRadius=1.)
    else:
        # Getting radiator without freezing coupling,
        # useful in testing
        rad, radprime = ang_rad_radprime_MLL(angs_central, beta, jet_type, R)

    # Pieces which contribute to the CDF
    cdf = np.exp(-rad)
    pdf = -radprime * cdf

    if acc == 'ME':
        # Rprime is defined as -1 times the logarithmic derivative of R
        Rprime = -radprime * angs_central
        f_MLL = np.exp(-EULER_GAMMA*Rprime)/gamma_vec(1+Rprime)

        cdf = cdf * f_MLL

        _, pdf = histDerivative(cdf, angs, giveHist=True,
                                binInput=binInput)
    return cdf, pdf, angs_central

# -----------------------------------------
# Plotting Analytic Results:
# -----------------------------------------
# ----------
# PDF plot:
# ----------
def angpdf_analytic(axespdf, beta=2.,
                    radius=1., jet_type='quark',
                    bin_space='lin', plotnum=0,
                    label='Analytic',
                    acc='LL'):
    """Plots the analytic angularity pdf on axespdf."""
    verify_bin_space(bin_space)
    verify_accuracy(acc)

    if bin_space == 'lin':
        angs = np.linspace(0, .5, 1000)
    if bin_space == 'log':
        angs = np.logspace(-8, np.log10(.6), 1000)

    if acc == 'LL':
        pdf = pdf_ang_LL(angs, beta=beta, jet_type=jet_type,
                         R=radius)
    if acc in ['MLL', 'ME']:
        _, pdf, angs = distrib_ang_MLL(angs, beta=beta,
                                       jet_type=jet_type,
                                       R=radius,
                                       binInput=bin_space,
                                       acc=acc)

    if bin_space == 'log':
        pdf = angs*pdf

    col = compcolors[(plotnum, 'light')]

    # Analytic plot:
    axespdf[0].plot(angs, pdf, **style_dashed,
                    color=col, zorder=.5, label=label)
    if len(axespdf) > 1:
        axespdf[1].plot(angs, np.ones(len(angs)),
                        **style_dashed, color=col,
                        zorder=.5)

# ----------
# CDF plot:
# ----------
def angcdf_analytic(axescdf, beta=2.,
                    radius=1., jet_type='quark',
                    bin_space='lin', plotnum=0,
                    label='Analytic',
                    acc='LL'):
    """Plots the analytic angularity cdf on axescdf."""
    verify_bin_space(bin_space)
    verify_accuracy(acc)

    if bin_space == 'lin':
        angs = np.linspace(0, .5, 1000)
    if bin_space == 'log':
        angs = np.logspace(-8, np.log10(.5), 1000)

    if acc == 'LL':
        cdf = cdf_ang_LL(angs, beta=beta, jet_type=jet_type, R=radius)
    if acc in ['MLL', 'ME']:
        cdf, _, angs = distrib_ang_MLL(angs, beta,
                                       jet_type=jet_type,
                                       R=radius,
                                       binInput=bin_space,
                                       acc=acc)

    col = compcolors[(plotnum, 'light')]

    # Analytic plot:
    axescdf[0].plot(angs, cdf, **style_dashed,
                    color=col, zorder=.5, label=label)
    if len(axescdf) > 1:
        axescdf[1].plot(angs, np.ones(len(angs)),
                        **style_dashed, color=col,
                        zorder=.5)

###########################################
# Plotting Utilities:
###########################################
# -----------------------------------------
# Plotting angularities:
# -----------------------------------------
def angplot_shower(angs, axespdf, axescdf,
                   beta=2., radius=1., jet_type='quark',
                   bin_space='lin', plotnum=0, acc='LL'):
    """Plots the pdf and cdf associated with the
    set of angularities angs on axespdf and axescdf.
    """
    verify_bin_space(bin_space)

    # Finding pdf:
    showerInt = integrator()
    showerInt.setLastBinBndCondition([1., 'minus'])

    if bin_space == 'lin':
        bins = np.linspace(0, radius**beta, NUM_BINS)
    else:
        bins = np.logspace(-8, np.log10(radius**beta), NUM_BINS)
        bins = np.append([0], bins)
    showerInt.bins = bins

    num_in_bin, _ = np.histogram(angs, bins)
    pdf, _ = np.histogram(angs, bins, density=True)

    showerInt.densityErr = pdf / np.sqrt(num_in_bin)
    showerInt.density = pdf
    showerInt.hasMCDensity = True

    # Finding cdf by integrating pdf:
    showerInt.integrate()
    integral = showerInt.integral
    interr = showerInt.integralErr

    col = compcolors[(plotnum, 'dark')]

    if comparisonplot:
        if plotnum == 1:
            label = "Larkoski's algorithm"
        if plotnum == 2:
            label = "Reweighted algorithm"
        if plotnum == 3:
            label = "New algorithm"
    else:
        label = LABEL_PS

    if axespdf is not None:
        # ------------------
        # PDF plots:
        # ------------------
        xs = (bins[1:] + bins[:-1])/2.

        if acc == 'LL':
            pdf_an = pdf_ang_LL(xs, beta=beta, jet_type=jet_type,
                                R=radius)
        if acc in ['MLL', 'ME']:
            _, pdf_an, angs = distrib_ang_MLL(bins, beta=beta,
                                              jet_type=jet_type,
                                              R=radius,
                                              binInput=bin_space,
                                              acc=acc)

        if bin_space == 'log':
            pdf = xs*pdf
            pdf_an = xs*pdf_an

        axespdf[0].errorbar(xs, pdf,
                            yerr=pdf/np.sqrt(num_in_bin),
                            xerr=(bins[1:] - bins[:-1])/2.,
                            **modstyle, color=col,
                            label=label)

        if len(axespdf) > 1:
            axespdf[1].errorbar(xs, pdf/pdf_an,
                                yerr=pdf/
                                (pdf_an * np.sqrt(num_in_bin)),
                                xerr=(bins[1:] - bins[:-1])/2.,
                                **modstyle, color=col)

    if axescdf is not None:
        # ------------------
        # CDF plots:
        # ------------------

        if acc == 'LL':
            cdf_an = cdf_ang_LL(xs, beta=beta, jet_type=jet_type,
                                R=radius)
        if acc in ['MLL', 'ME']:
            cdf_an, _, _ = distrib_ang_MLL(bins, beta=beta,
                                           jet_type=jet_type,
                                           R=radius,
                                           binInput=bin_space,
                                           acc=acc)

        xs = bins[:-1]

        _, _, bars = axescdf[0].errorbar(xs, integral,
                                         yerr=interr,
                                         **style_yerr,
                                         color=col, ecolor=col,
                                         label=label)
        bars = [b.set_alpha(.5) for b in bars]

        if len(axescdf) > 1:
            _, _, bars_r = axescdf[1].errorbar(xs,
                                               integral/cdf_an,
                                               yerr=interr/cdf_an,
                                               **style_yerr,
                                               color=col, ecolor=col)
            bars_r = [b.set_alpha(.5) for b in bars_r]

# -----------------------------------------
# Plotting many sets of angularities:
# -----------------------------------------
def angularity_multiplot(num_events, axespdf, axescdf,
                         radius=1., jet_type='quark',
                         bin_space='lin', acc='LL'):
    """Plots a set of angularities with different betas,
    next to the corresponding analytic expressions,
    at a given accuracy.

    Designed to produce nice validation plots for our
    parton showers.
    """
    # Preparing to store lists of angularities
    all_angs = []

    # ------------------
    # Setting up plots:
    # ------------------
    for ibeta, beta in enumerate(BETA_LIST):
        beta = BETA_LIST[ibeta]

        angpdf_analytic(axespdf, beta=beta,
                        radius=radius, jet_type=jet_type,
                        bin_space=bin_space, plotnum=ibeta,
                        label=r'$\beta=$'+str(beta),
                        acc=acc)
        angcdf_analytic(axescdf, beta=beta,
                        radius=radius, jet_type=jet_type,
                        bin_space=bin_space, plotnum=ibeta,
                        label=r'$\beta=$'+str(beta),
                        acc=acc)

    # Labelling
    if bin_space == 'lin':
        labelLines(axespdf[0].get_lines())
        labelLines(axescdf[0].get_lines())
    elif bin_space == 'log':
        xvals1 = [9e-4, 7e-5, 3e-6, 6e-8]
        xvals2 = [1e-2, 1e-3, 5e-5, 1e-6]
        if acc != 'LL':
            #xvals1 = [4e-2, 1e-2, 1e-3, 1e-4]
            xvals1 = [1e-1, 7e-2, 1.75e-2, 4e-3]
            xvals2 = [1e-1, 1e-2, 1e-3, 1e-4]
        labelLines(axespdf[0].get_lines(), xvals=xvals1)
        labelLines(axescdf[0].get_lines(), xvals=xvals2)


    for ibeta, beta in enumerate(BETA_LIST):
        beta = BETA_LIST[ibeta]

        # Parton showering
        jet_list = gen_jets(num_events, beta=beta,
                            radius=radius,
                            jet_type=jet_type,
                            acc=acc)

        # Getting angularities
        angs = getangs(jet_list, beta=beta, acc=acc)
        all_angs.append(angs)

        # Plotting
        angplot_shower(angs, axespdf, axescdf,
                       beta=beta, radius=radius, jet_type=jet_type,
                       bin_space=bin_space, plotnum=ibeta,
                       acc=acc)

    # Legend
    legend_darklight(axespdf[0], darklabel=LABEL_PS, errtype='modstyle')
    legend_darklight(axescdf[0], darklabel=LABEL_PS, errtype='yerr')

#########################################################
# Main Methods/Actual Plotting:
#########################################################
def showTestPlot(jet_list, bin_space, acc):
    """Shows tests plots for beta=2 angularity distributions."""
    # ---------------------
    # Setting up figures
    # ---------------------
    if bin_space == 'lin':
        ylabel = r"$\frac{{\rm d}\sigma}{{\rm d}e_2}$"
    if bin_space == 'log':
        ylabel = (r"$\frac{1}{\sigma}$"
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\ln e_2}$")

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$e_2$",
                                   ylabel=ylabel,
                                   ylim=ylim_1,
                                   xlim=xlim,
                                   title="Angularity PDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=True)
    axespdf[0].set_ylabel(ylabel, labelpad=25, rotation=0,
                          fontsize=18)
    axespdf[1].set_ylabel('Ratio', labelpad=0, rotation=0)

    if SHOW_FIGTEXT:
        set_figtext(figpdf, INFO, loc=(.878, .7),
                    rightjustify=True)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$e_2$",
                                   ylabel=r"$\Sigma(e_2)$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Angularity CDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=True)
    axescdf[0].set_ylabel(r"$\Sigma(e_2)$", labelpad=23, rotation=0,
                          fontsize=15)
    axescdf[1].set_ylabel('Ratio', labelpad=-5, rotation=0)

    if SHOW_FIGTEXT:
        set_figtext(figcdf, INFO, loc=(.1475, .7))

    if bin_space == 'log':
        axes = [axespdf[0], axespdf[1], axescdf[0], axescdf[1]]
        for ax in axes:
            ax.set_xscale('log')

    print("Creating an example "+acc+" angularity distribution...")
    # Angularities
    angs = getangs(jet_list, beta=2., acc=acc)

    # Plotting
    angpdf_analytic(axespdf, beta=2.,
                    radius=1., jet_type=JET_TYPE,
                    bin_space=bin_space, plotnum=1,
                    acc=acc)
    angcdf_analytic(axescdf, beta=2.,
                    radius=1., jet_type=JET_TYPE,
                    bin_space=bin_space, plotnum=1,
                    acc=acc)

    angplot_shower(angs, axespdf, axescdf,
                   beta=2., radius=1., jet_type=JET_TYPE,
                   bin_space=bin_space, plotnum=1,
                   acc=acc)

    axespdf[0].legend()
    legend_yerr(axescdf[0])

    figpdf.subplots_adjust(left=.5)

    figpdf.tight_layout()
    figcdf.tight_layout()

    if SAVE_PLOTS:
        figpdf.savefig("angularity_2_"+acc+"_pdf_"+bin_space
                       +"_"+JET_TYPE+"_test.pdf")
        figcdf.savefig("angularity_2_"+acc+"_cdf_"+bin_space
                       +"_"+JET_TYPE+"_test.pdf")
    if SHOW_PLOTS:
        plt.show()

def saveMultiplot(bin_space, acc='LL'):
    """Saves a set of angularity plots for different beta."""
    # ---------------------
    # Setting up figures
    # ---------------------
    if bin_space == 'lin':
        ylabel = r"$\frac{{\rm d}\sigma}{{\rm d}e_\beta}$"
    if bin_space == 'log':
        ylabel = (r"$\frac{1}{\sigma}$"
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\ln e_\beta}$")

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$e_\beta$",
                                   ylabel=ylabel,
                                   ylim=ylim_2,
                                   xlim=xlim,
                                   title="Angularity PDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=False)
    axespdf[0].set_ylabel(ylabel, labelpad=23, rotation=0,
                          fontsize=18)
    if SHOW_FIGTEXT:
        set_figtext(figpdf, INFO, loc=(.878, .7),
                    rightjustify=True)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$e_\beta$",
                                   ylabel=r"$\Sigma(e_\beta)$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Angularity CDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=False)
    axescdf[0].set_ylabel(r"$\Sigma(e_\beta)$", labelpad=23,
                          rotation=0, fontsize=15)
    if SHOW_FIGTEXT:
        set_figtext(figcdf, INFO, loc=(.1475, .7))

    if bin_space == 'log':
        axes = [axespdf[0], axescdf[0]]
        for ax in axes:
            ax.set_xscale('log')

    angularity_multiplot(NUM_EVENTS, axespdf, axescdf,
                         radius=1., jet_type=JET_TYPE,
                         bin_space=bin_space, acc=acc)

    figpdf.subplots_adjust(left=.5)

    figpdf.tight_layout()
    figcdf.tight_layout()

    figpdf.savefig("angularity_multi_"+acc+"_pdf_"+bin_space
                   +"_"+JET_TYPE+"_test.pdf")
    figcdf.savefig("angularity_multi_"+acc+"_cdf_"+bin_space
                   +"_"+JET_TYPE+"_test.pdf")

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    jets = gen_jets(NUM_EVENTS, beta=2.,
                    radius=1.,
                    acc=ACCURACY)
    showTestPlot(jets, BIN_SPACE, ACCURACY)
    if SAVE_PLOTS:
        saveMultiplot(BIN_SPACE, ACCURACY)
