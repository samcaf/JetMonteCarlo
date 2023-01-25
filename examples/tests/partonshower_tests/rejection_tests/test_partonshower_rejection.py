import numpy as np

# Local imports:
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.utils.partonshower_utils import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.jets.observables import *

# Test imports
from examples.tests.partonshower_tests.rejection_tests.rejection_test_utils import *


###########################################
# Parameters and setup:
###########################################
NUM_EVENTS = 1e4
NUM_BINS = min(int(NUM_EVENTS/20), 50)
BIN_SPACE = 'log'
JET_TYPE = 'quark'
# All LL tests:
# betalist = [.5, 1., 2., 4.]
# Test >= 4 (MLL tests):
betalist = [2., 3., 4.]

FIX_BETA = 0.

if BIN_SPACE == 'lin':
    YLIM_PDF = (0, 1.)
    XLIM = (0, .5)
else:
    YLIM_PDF = (0, .4)
    XLIM = (1e-8, .5)

SHOW_PLOTS = False
SAVE_PLOTS = True

# Enumerating the tests of rejection sampling for LL,
# with various levels of complexity
TEST_NUMS = [-1, 8, 6]

def test_acc(test_num):
    """Accuracy that we wish to probe with
    a particular test.
    """
    if 0 <= test_num <= 3:
        return 'LL'
    return 'MLL'

###########################################
# LL Rejection-based Angularity Tests:
###########################################

# -----------------------------------------
# Parton Showers:
# -----------------------------------------
def generate_jets_LL_rejection(num_events, beta=2.,
                               radius=1., jet_type='quark',
                               test_num=0):
    """Generates a list of num_events jet_type jets
    through angularity based parton showering, using
    a test algorithm designed to test acceptance-rejection
    sampling in the parton shower.
    """
    # Setting up for angularities
    jet_list = []

    if FIX_BETA:
        beta = FIX_BETA

    for _ in range(int(num_events)):
        # Initializing a parton
        ang_init = radius**beta / 2.
        momentum = Vector([0, P_T, 0])

        # Performing parton shower to produce jet
        jet = Jet(momentum, radius, partons=None)
        mother = jet.partons[0]
        angularity_shower_LL_rej(parton=mother, ang_init=ang_init,
                                 beta=beta, jet_type=jet_type,
                                 jet=jet,
                                 split_soft=False,
                                 test_num=test_num)
        jet.has_showered = True

        jet_list.append(jet)

    return jet_list

#########################################################
# Main Methods:
#########################################################
def showTestPlot_rej(jet_list_list, bin_space, test_num):
    """Shows tests plots for beta=2 angularity distributions,
    using a test algorithm designed to test acceptance-rejection
    sampling in the parton shower.
    """
    # ---------------------
    # Setting up figures
    # ---------------------
    # Fig and axes for plotting pdf
    if len(TEST_NUMS) == 1:
        title_pdf = "Angularity PDF (rejection test {})".format(test_num)
        title_cdf = "Angularity CDF (rejection test {})".format(test_num)
    if (len(TEST_NUMS)) > 1:
        title_pdf = "Angularity PDF"
        title_cdf = "Angularity CDF"
    figpdf, axespdf = aestheticfig(xlabel=r"$e^{(2)}$",
                                   ylabel=r"$\rho(e^{(2)})$",
                                   ylim=YLIM_PDF,
                                   xlim=XLIM,
                                   title=title_pdf,
                                   showdate=True,
                                   ratio_plot=True)
    axespdf[0].set_ylabel(r"$\rho(e^{(2)})$")

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$e^{(2)}$",
                                   ylabel=r"$\Sigma(e^{(2)})$",
                                   ylim=(0, 1.1),
                                   xlim=XLIM,
                                   title=title_cdf,
                                   showdate=True,
                                   ratio_plot=True)

    if bin_space == 'log':
        axes = [axespdf[0], axespdf[1], axescdf[0], axescdf[1]]
        for ax in axes:
            ax.set_xscale('log')

    print("Creating an example "+test_acc(test_num)
          +" angularity distribution...")

    # ---------------------
    # Plotting Analytic
    # ---------------------
    if len(jet_list_list) == 1:
        plotnuman = 1
    else:
        plotnuman = -1
    angpdf_analytic(axespdf, beta=2., radius=1.,
                    jet_type=JET_TYPE, bin_space=bin_space,
                    plotnum=plotnuman, acc=test_acc(test_num))
    angcdf_analytic(axescdf, beta=2., radius=1.,
                    jet_type=JET_TYPE, bin_space=bin_space,
                    plotnum=plotnuman, acc=test_acc(test_num))

    # ---------------------
    # Plotting Monte Carlo
    # ---------------------
    for j, jet_list in enumerate(jet_list_list):
        angs = getangs(jet_list, beta=2., acc=test_acc(test_num))
        angplot_shower(angs, axespdf, axescdf,
                       beta=2., radius=1., jet_type=JET_TYPE,
                       bin_space=bin_space, plotnum=j+1,
                       acc=test_acc(test_num))

    axespdf[0].legend()
    legend_yerr(axescdf[0])
    if SAVE_PLOTS:
        figpdf.savefig("angularity_2_LL_pdf_"+bin_space
                       +"_"+JET_TYPE+"_rej_"+str(test_num)+
                       "_test.pdf")
        figcdf.savefig("angularity_2_LL_cdf_"+bin_space
                       +"_"+JET_TYPE+"_rej_"+str(test_num)+
                       "_test.pdf")
    if SHOW_PLOTS:
        plt.show()


def angularity_LL_multiplot_rej(num_events, axespdf, axescdf,
                                radius=1., jet_type='quark',
                                bin_space='lin', test_num=0):
    """Plots a set of angularities with different betas,
    next to the corresponding analytic expressions,
    at a given accuracy.

    Uses a test algorithm designed to test acceptance-rejection
    sampling in the parton shower.
    """
    # ------------------
    # Plotting:
    # ------------------
    for ibeta, beta in enumerate(betalist):
        beta = betalist[ibeta]
        angpdf_analytic(axespdf, beta=beta,
                        radius=radius, jet_type=jet_type,
                        bin_space=bin_space, plotnum=ibeta,
                        label=r'$\beta=$'+str(beta),
                        acc=test_acc(test_num))
        angcdf_analytic(axescdf, beta=beta,
                        radius=radius, jet_type=jet_type,
                        bin_space=bin_space, plotnum=ibeta,
                        label=r'$\beta=$'+str(beta),
                        acc=test_acc(test_num))

    # Labelling
    if BIN_SPACE == 'lin':
        labelLines(axespdf[0].get_lines())
        labelLines(axescdf[0].get_lines())
    elif BIN_SPACE == 'log':
        labelLines(axespdf[0].get_lines(),
                   xvals=np.logspace(-6, -1, 4)[::-1])
        labelLines(axescdf[0].get_lines(),
                   xvals=np.logspace(-6, -1, 4)[::-1])

    for ibeta, beta in enumerate(betalist):
        # Parton showering
        jet_list = generate_jets_LL_rejection(num_events, beta=beta,
                                              radius=radius,
                                              jet_type=jet_type,
                                              test_num=test_num)

        angs = getangs(jet_list, beta=beta, acc=test_acc(test_num))

        # Plotting
        angplot_shower(angs, axespdf, axescdf,
                       beta=beta, radius=radius,
                       jet_type=jet_type,
                       bin_space=bin_space, plotnum=ibeta,
                       acc=test_acc(test_num))

    # Legend
    legend_darklight(axespdf[0], errtype='modstyle')
    legend_darklight(axescdf[0], errtype='yerr')

def saveMultiplot_rej(bin_space, test_num=0):
    """Saves a set of angularity plots for different beta,
    using a test algorithm designed to test acceptance-rejection
    sampling in the parton shower.
    """
    # ---------------------
    # Setting up figures
    # ---------------------
    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$e^{(\beta)}$",
                                   ylabel=r"$\rho(e^{(\beta)})$",
                                   ylim=YLIM_PDF,
                                   xlim=XLIM,
                                   title="Angularity PDF"
                                   +" (rejection test {})"
                                   .format(test_num),
                                   showdate=True,
                                   ratio_plot=False)
    axespdf[0].set_ylabel(r"$\rho(e^{(2)})$", labelpad=-7)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$e^{(\beta)}$",
                                   ylabel=r"$\Sigma(e^{(\beta)})$",
                                   ylim=(0, 1.1),
                                   xlim=XLIM,
                                   title="Angularity CDF"
                                   +" (rejection test {})"
                                   .format(test_num),
                                   showdate=True,
                                   ratio_plot=False)
    if bin_space == 'log':
        axes = [axespdf[0], axescdf[0]]
        for ax in axes:
            ax.set_xscale('log')

    angularity_LL_multiplot_rej(NUM_EVENTS, axespdf, axescdf,
                                radius=1., jet_type=JET_TYPE,
                                bin_space=bin_space,
                                test_num=test_num)
    figpdf.savefig("angularity_multi_LL_pdf_"+BIN_SPACE
                   +"_"+JET_TYPE+"_rej_"+str(test_num)+
                   "_test.pdf")
    figcdf.savefig("angularity_multi_LL_cdf_"+BIN_SPACE
                   +"_"+JET_TYPE+"_rej_"+str(test_num)+
                   "_test.pdf")

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    jetlist_by_alg = []
    for testnum in TEST_NUMS:
        jets = generate_jets_LL_rejection(NUM_EVENTS,
                                          beta=2., radius=1.,
                                          jet_type=JET_TYPE,
                                          test_num=testnum)
        jetlist_by_alg.append(jets)

    showTestPlot_rej(jetlist_by_alg, BIN_SPACE, testnum)
    if not comparisonplot:
        assert len(TEST_NUMS) > 1, "If not making a comparison plot, "\
                                   + "you must choose a single test."
    for testnum in TEST_NUMS:
        if SAVE_PLOTS:
            saveMultiplot_rej(BIN_SPACE, testnum)
