import numpy as np

# Local imports:
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.hist_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.partonshower_utils import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

# Test utility imports
from jetmontecarlo.tests.partonshower_tests.one_emission.one_em_groomed_test_utils import *
from jetmontecarlo.tests.partonshower_tests.pre_and_crit.precrit_groomed_test_utils import *

###########################################
# Parameters and setup:
###########################################
# Monte Carlo
NUM_EVENTS = 1e5
NUM_BINS = min(int(NUM_EVENTS/20), 50)

# Physics
BIN_SPACE = 'log'
JET_TYPE = 'quark'
ACCURACY = 'LL'
SHOWER_TYPE = 'pre_and_crit'

# Valid shower types:
# 'crit': Generate a single, critical emission via veto
#
# 'pre_and_crit': Generate a single, critical emission via veto, after
#                 throwing out emisisons with z < z_cut and reducing
#                 z_cut at every step. This should reproduce the
#                 effects of both critical and pre-critical emissions.
#
# 'crit_and_sub': Not yet coded, and an interesting challenge
#
# None: Keeps pre-critical, critical, and subsequent emissions.
#       This can be done fairly simply via veto when we have
#       MLL accuracy and a non-perturbative cutoff. For fixed
#       coupling, there is no cutoff for the subsequent emissions,
#       and this needs to be developed.
assert (SHOWER_TYPE in
        ['crit', 'pre_and_crit', 'crit_and_sub', None]),\
        "SHOWER_TYPE has the invalid value: "+str(SHOWER_TYPE)


BETA_LIST = [.5, 1., 2., 4.]
Z_CUT = .1
BETA = 2.
F_SOFT = 1.

FIX_BETA = 0.

# Plotting
if BIN_SPACE == 'lin':
    ylim_1 = (0, 40.)
    ylim_2 = (0, 40.)
    xlim = (0, .5)
if BIN_SPACE == 'log':
    ylim_1 = (0, .12)
    ylim_2 = (0, .25)
    xlim = (1e-7, .5)
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

# -----------------------------------------
# Plotting Analytic Results:
# -----------------------------------------
def plot_analytic(axespdf, axescdf,
                  z_cut, beta=2., f=1.,
                  jet_type='quark',
                  bin_space='lin', plotnum=0,
                  label='Analytic, LL'):
    """Plots the analytic groomed GECF cdf on axescdf,
    and the groomed GECF pdf on axespdf.
    """
    verify_bin_space(bin_space)

    if bin_space == 'lin':
        angs = np.linspace(0, .5, 1000)
        angs_central = (angs[:-1] + angs[1:])/2.
    if bin_space == 'log':
        angs = np.logspace(-8, np.log10(.5), 1000)
        angs_central = np.sqrt(angs[:-1] * angs[1:])

    cdf = critSudakov_fc_LL(angs_central, z_cut, beta, f=f,
                            jet_type=jet_type)

    col = compcolors[(plotnum, 'light')]

    # Analytic plot:
    axescdf[0].plot(angs_central, cdf, **style_dashed,
                    color=col, zorder=.5, label=label)
    if len(axescdf) > 1:
        axescdf[1].plot(angs, np.ones(len(angs)),
                        **style_dashed, color=col,
                        zorder=.5)
    if axespdf is not None:
        _, pdf = histDerivative(cdf, angs, giveHist=True,
                                binInput=bin_space)
        if bin_space == 'log':
            pdf = angs_central*pdf
        # Analytic plot:
        axespdf[0].plot(angs_central, pdf, **style_dashed,
                        color=col, zorder=.5, label=label)
        if len(axespdf) > 1:
            axespdf[1].plot(angs, np.ones(len(angs)),
                            **style_dashed, color=col,
                            zorder=.5)

###########################################
# Plotting Utilities:
###########################################
# -----------------------------------------
# Plotting angularities:
# -----------------------------------------
def angplot_shower(angs, axespdf, axescdf,
                   z_cut, beta, f, radius=1., jet_type='quark',
                   bin_space='lin', plotnum=0,
                   label=LABEL_PS):
    """Plots the pdf and cdf associated with the
    set of angularities angs on axespdf and axescdf.
    """
    verify_bin_space(bin_space)

    # Finding numerical pdf:
    showerInt = integrator()
    showerInt.setLastBinBndCondition([1., 'minus'])

    if bin_space == 'lin':
        bins = np.linspace(0, radius**beta, NUM_BINS)
    else:
        bins = np.logspace(-8, np.log10(radius**beta), NUM_BINS)
        bins = np.append(1e-50, bins)
    showerInt.bins = bins

    # angs = np.array(angs)*(1 - 1/(beta*(beta-1)))
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

    # Analytic cdf and pdf:
    xs = (bins[:-1] + bins[1:])/2.
    if axespdf is not None:
        # Numerically obtaining pdf:
        # cdf_an = critSudakov_fc_LL(xs, z_cut, beta, f=f,
        #                            jet_type=jet_type)
        cdf_an = critSudakov_fc_LL(xs, z_cut, beta, f=f,
                                   jet_type=jet_type)
        _, pdf_an = histDerivative(cdf_an, bins, giveHist=True,
                                   binInput=bin_space)
        pdf_an = np.array(pdf_an.tolist(), dtype=float)
        # ------------------
        # PDF plots:
        # ------------------
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
        xs = bins[:-1]
        cdf_an = critSudakov_fc_LL(xs, z_cut, beta, f=f,
                                   jet_type=jet_type)
        cdf_an = np.array(cdf_an.tolist(), dtype=float)

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
def gecf_groomed_multiplot(num_events, axespdf, axescdf,
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

        plot_analytic(axespdf, axescdf,
                      z_cut=Z_CUT, beta=beta, f=F_SOFT,
                      jet_type=jet_type, bin_space=bin_space,
                      plotnum=ibeta, label=r"$\beta=$"+str(beta))
    # Labelling
    if bin_space == 'lin':
        labelLines(axespdf[0].get_lines())
        labelLines(axescdf[0].get_lines())
    elif bin_space == 'log':
        # BETA_LIST = [.5, 1., 2., 4.]
        xvals1 = [1e-1, 2e-3, 2e-4, 3e-5]
        xvals2 = [1e-1, 1e-2, 2e-4, 1e-6]
        # BETA_LIST = [2., 4., 6., 8.]
        # xvals1 = [1e-3, 2e-4, 6e-5, 3e-5]
        # xvals2 = [1e-2, 1e-3, 5e-5, 1e-6]
        if acc != 'LL':
            #xvals1 = [4e-2, 1e-2, 1e-3, 1e-4]
            xvals1 = [1e-1, 7e-2, 1.75e-2, 4e-3]
            xvals2 = [1e-1, 1e-2, 1e-3, 1e-4]
        labelLines(axespdf[0].get_lines(), xvals=xvals1)
        labelLines(axescdf[0].get_lines(), xvals=xvals2)


    for ibeta, beta in enumerate(BETA_LIST):
        beta = BETA_LIST[ibeta]

        # Parton showering
        if SHOWER_TYPE == 'crit':
            jet_list = gen_jets_groomed_crit(num_events,
                                             z_cut=Z_CUT, beta=beta, f=F_SOFT,
                                             radius=radius,
                                             jet_type=jet_type,
                                             acc=acc)

        elif SHOWER_TYPE == 'pre_and_crit':
            jet_list = gen_jets_groomed_precrit(num_events,
                                                z_cut=Z_CUT, beta=beta, f=F_SOFT,
                                                radius=radius,
                                                jet_type=jet_type,
                                                acc=acc)
        else:
            jet_list = gen_jets_groomed(num_events,
                                        z_cut=Z_CUT, beta=beta, f=F_SOFT,
                                        radius=radius,
                                        jet_type=jet_type,
                                        acc=acc, cutoff=SHOWER_TYPE)
        # Getting angularities
        angs = getangs(jet_list, beta=beta, acc=acc,
                       emission_type=None)

        all_angs.append(angs)

        # Plotting
        angplot_shower(angs, axespdf, axescdf,
                       z_cut=Z_CUT, beta=beta, f=F_SOFT,
                       radius=radius, jet_type=jet_type,
                       bin_space=bin_space, plotnum=ibeta)

    # Legend
    legend_darklight(axespdf[0], darklabel=LABEL_PS, errtype='modstyle')
    legend_darklight(axescdf[0], darklabel=LABEL_PS, errtype='yerr')

#########################################################
# Main Methods/Actual Plotting:
#########################################################
def showTestPlot(jet_list, bin_space, acc):
    """Shows tests plots for beta=2 GECF distributions."""
    # ---------------------
    # Setting up figures
    # ---------------------
    if bin_space == 'lin':
        ylabel = r"$\frac{{\rm d}\sigma}{{\rm d}\lambda_{(2)}}$"
    if bin_space == 'log':
        ylabel = (r"$\frac{1}{\sigma}$"
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\ln \lambda_{(2)}}$")

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$\lambda_{(2)}$",
                                   ylabel=ylabel,
                                   ylim=ylim_1,
                                   xlim=xlim,
                                   title="Groomed ECF PDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=True)
    axespdf[0].set_ylabel(ylabel, labelpad=25, rotation=0,
                          fontsize=18)
    axespdf[1].set_ylabel('Ratio', labelpad=0, rotation=0)

    if SHOW_FIGTEXT:
        set_figtext(figpdf, INFO, loc=(.878, .7),
                    rightjustify=True)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$\lambda_{(2)}$",
                                   ylabel=r"$\Sigma(\lambda_{(2)})$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Groomed ECF CDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=True)
    axescdf[0].set_ylabel(r"$\Sigma(\lambda_{(2)})$", labelpad=23, rotation=0,
                          fontsize=15)
    axescdf[1].set_ylabel('Ratio', labelpad=-5, rotation=0)

    if SHOW_FIGTEXT:
        set_figtext(figcdf, INFO, loc=(.1475, .7))

    if bin_space == 'log':
        axes = [axespdf[0], axespdf[1], axescdf[0], axescdf[1]]
        for ax in axes:
            ax.set_xscale('log')

    print("Creating an example "+acc+" ECF distribution...")
    # Angularities
    angs = getangs(jet_list, beta=2., acc=acc,
                   emission_type=None)

    # Plotting
    plot_analytic(axespdf, axescdf,
                  z_cut=Z_CUT, beta=2., f=F_SOFT,
                  jet_type=JET_TYPE,
                  bin_space=bin_space, plotnum=1)

    angplot_shower(angs, axespdf, axescdf,
                   z_cut=Z_CUT, beta=2., f=F_SOFT,
                   radius=1., jet_type=JET_TYPE,
                   bin_space=bin_space, plotnum=1)

    axespdf[0].legend()
    legend_yerr(axescdf[0])

    figpdf.subplots_adjust(left=.5)

    figpdf.tight_layout()
    figcdf.tight_layout()

    if SAVE_PLOTS:
        figpdf.savefig("gecf_groomed_2_"+acc+"_pdf_"+bin_space
                       +"_"+JET_TYPE+"_test.pdf")
        figcdf.savefig("gecf_groomed_2_"+acc+"_cdf_"+bin_space
                       +"_"+JET_TYPE+"_test.pdf")
    if SHOW_PLOTS:
        plt.show()

def saveMultiplot(bin_space, acc='LL'):
    """Saves a set of groomed GECF plots for different beta."""
    # ---------------------
    # Setting up figures
    # ---------------------
    if bin_space == 'lin':
        ylabel = r"$\frac{{\rm d}\sigma}{{\rm d}\lambda_{(\beta)}}$"
    if bin_space == 'log':
        ylabel = (r"$\frac{1}{\sigma}$"
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\ln \lambda_{(\beta)}}$")

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$\lambda_{(\beta)}$",
                                   ylabel=ylabel,
                                   ylim=ylim_2,
                                   xlim=xlim,
                                   title="Groomed GECF PDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=False)
    axespdf[0].set_ylabel(ylabel, labelpad=23, rotation=0,
                          fontsize=18)
    if SHOW_FIGTEXT:
        set_figtext(figpdf, INFO, loc=(.878, .7),
                    rightjustify=True)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$\lambda_{(\beta)}$",
                                   ylabel=r"$\Sigma(\lambda_{(\beta)})$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Groomed GECF CDF ("+acc+")",
                                   showdate=True,
                                   ratio_plot=False)
    axescdf[0].set_ylabel(r"$\Sigma(\lambda_{(\beta)})$", labelpad=23,
                          rotation=0, fontsize=15)
    if SHOW_FIGTEXT:
        set_figtext(figcdf, INFO, loc=(.1475, .7))

    if bin_space == 'log':
        axes = [axespdf[0], axescdf[0]]
        for ax in axes:
            ax.set_xscale('log')

    gecf_groomed_multiplot(NUM_EVENTS, axespdf, axescdf,
                           radius=1., jet_type=JET_TYPE,
                           bin_space=bin_space, acc=acc)

    figpdf.subplots_adjust(left=.5)

    figpdf.tight_layout()
    figcdf.tight_layout()

    figpdf.savefig("gecf_groomed_multi_"+acc+"_pdf_"+bin_space
                   +"_"+JET_TYPE+"_test.pdf")
    figcdf.savefig("gecf_groomed_multi_"+acc+"_cdf_"+bin_space
                   +"_"+JET_TYPE+"_test.pdf")

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    if SHOWER_TYPE == 'crit':
        jets = gen_jets_groomed_crit(NUM_EVENTS,
                                     z_cut=Z_CUT, beta=2., f=F_SOFT,
                                     radius=1., acc=ACCURACY)
    elif SHOWER_TYPE == 'pre_and_crit':
        jets = gen_jets_groomed_precrit(NUM_EVENTS,
                                        z_cut=Z_CUT, beta=2., f=F_SOFT,
                                        radius=1., acc=ACCURACY)
    else:
        jets = gen_jets_groomed(NUM_EVENTS,
                                z_cut=Z_CUT, beta=2., f=F_SOFT,
                                radius=1., acc=ACCURACY,
                                cutoff=SHOWER_TYPE)
    showTestPlot(jets, BIN_SPACE, ACCURACY)
    if SAVE_PLOTS:
        saveMultiplot(BIN_SPACE, ACCURACY)
