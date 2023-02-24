import dill as pickle
from pathlib import Path

# Local utilities for comparison
from examples.utils.plot_comparisons import *
from jetmontecarlo.utils.hist_utils import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

# ------------------------------------
# Parameters
# ------------------------------------
FIXED_COUPLING = False

# MC events
NUM_MC_EVENTS = int(1e5)
NUM_RAD_BINS = int(1e3)

THETAS = [.01, .03, .1, 0.3, 1.]

Z_CUT = .1
Z_CUTS = [.003, .01, .03, .1, .2]

PLOT_SUB = False
PLOT_PRE = True
# ------------------------------------
# File paths
# ------------------------------------
# File folders
rad_folder = Path("output/serialized_functions/radiators/")

# Radiator paths:
rad_extension = ("_{:.0e}events".format(NUM_MC_EVENTS)
                 +"_{:.0e}bins".format(NUM_RAD_BINS)
                 +".pkl")

crit_rad_file_path = rad_folder / ("crit_{}_rads_rc".format(BIN_SPACE)
                                   + rad_extension)
sub_rad_file_path = rad_folder / ("sub_{}_rads_rc".format(BIN_SPACE)
                                  + rad_extension)
crit_sub_rad_file_path = rad_folder / ("crit_sub_{}_rads_rc".format(BIN_SPACE)
                                       + rad_extension)
pre_rad_file_path = rad_folder / ("pre_{}_rads_rc".format(BIN_SPACE)
                                  + rad_extension)

if FIXED_COUPLING:
    crit_rad_file_path = rad_folder / ("crit_{}_rads".format(BIN_SPACE)
                                       + rad_extension)
    sub_rad_file_path = rad_folder / ("sub_{}_rads".format(BIN_SPACE)
                                      + rad_extension)
    crit_sub_rad_file_path = rad_folder / ("crit_sub_{}_rads".format(BIN_SPACE)
                                           + rad_extension)
    pre_rad_file_path = rad_folder / ("pre_{}_rads".format(BIN_SPACE)
                                      + rad_extension)

index_zc = {.05: 0, .1: 1, .2: 2}

print("Loading radiators...")
if PLOT_SUB:
    print("    Loading subsequent radiator...")
    with open(crit_sub_rad_file_path, 'rb') as file:
        rad_crit_sub = pickle.load(file)[0]
    def cdf_crit_sub_conditional(c_sub, theta, z_cut):
        return np.exp(-rad_crit_sub(c_sub, theta))

if PLOT_PRE:
    print("    Loading pre-critical radiator...")
    with open(pre_rad_file_path, 'rb') as file:
        rad_pre = pickle.load(file)
    def cdf_pre_conditional(z_pre, theta, z_cut):
        return np.exp(-rad_pre[index_zc[z_cut]](z_pre, theta))

##########################################
# Tests/Plot comparison
##########################################
# ------------------------------------
# Critical and Subsequent
# ------------------------------------
def test_crit_sub_radiator():
    print("Preparing to plot crit-sub radiator...")
    # Preparing figures
    fig_rad, axes_rad, _, _ = \
            get_axes('crit and sub', ratio_plot=False)
    axes_rad[0].set_ylim(1e-20, 12)
    axes_rad[0].set_xlim(1e-10, 1)
    axes_rad[0].set_yscale('log')
    # Analytic plot
    for icol, radius in enumerate(THETAS):
        # Setup for cdf
        bins = np.logspace(-10, 0, 10000)
        xs = np.sqrt(bins[:-1]*bins[1:])
        rad_an = subRadAnalytic_fc_LL(xs, beta=2., jet_type=JET_TYPE,
                                      maxRadius=radius)
        rad_num = rad_crit_sub(xs, radius)

        # Plotting
        col_an = compcolors[(icol%4, 'light')]
        col_num = compcolors[(icol%4, 'dark')]

        axes_rad[0].plot(xs, rad_num, **style_solid,
                         color=col_num)
        axes_rad[0].plot(xs, rad_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius))

    labelLines(axes_rad[0].get_lines(), xvals=np.logspace(-7,-2,len(THETAS)))

    legend_darklight(axes_rad[0], darklabel='Numerical',
                     lightlabel='Analytic')

    fig_rad.savefig(JET_TYPE+'_crit_sub_num_rad_test.pdf',
                    format='pdf')
    print("Plotting complete!")

# ------------------------------------
# Critical and Subsequent, Running Coupling
# ------------------------------------
def test_crit_sub_radiator_rc():
    print("Preparing to plot crit-sub radiator...")
    # Preparing figures
    fig_rad, axes_rad, _, _ = \
            get_axes('crit and sub', ratio_plot=False)
    axes_rad[0].set_ylim(0, 12)
    axes_rad[0].set_xlim(1e-10, 1)
    # Analytic plot
    for icol, radius in enumerate(THETAS):
        # Setup for cdf
        bins = np.logspace(-10, 0, 10000)
        xs = np.sqrt(bins[:-1]*bins[1:])

        rad_an = subRadAnalytic(xs, beta=2., jet_type=JET_TYPE,
                                maxRadius=radius)
        rad_num = rad_crit_sub(xs, radius)

        # Plotting
        col_an = compcolors[(icol%4, 'light')]
        col_num = compcolors[(icol%4, 'dark')]

        axes_rad[0].plot(xs, rad_num, **style_solid,
                         color=col_num)
        axes_rad[0].plot(xs, rad_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius))

    labelLines(axes_rad[0].get_lines(), xvals=np.logspace(-7,-2,len(THETAS)))

    legend_darklight(axes_rad[0], darklabel='Numerical',
                     lightlabel='Analytic')

    fig_rad.savefig(JET_TYPE+'_crit_sub_num_rc_rad_test.pdf',
                    format='pdf')
    print("Plotting complete!")

def test_crit_sub_sudakov():
    print("Preparing to plot crit-sub sudakov factor...")
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit and sub', ratio_plot=False)
    axes_cdf[0].set_xlim(1e-10, 1)
    axes_pdf[0].set_xlim(1e-10, 1)

    # Analytic plot
    for icol, radius in enumerate(THETAS):
        # Setup for cdf
        bins = np.logspace(-20, .1, 1000)
        xs = np.sqrt(bins[:-1]*bins[1:])
        cdf_an = np.exp(-subRadAnalytic_fc_LL(xs, beta=2., jet_type=JET_TYPE,
                                              maxRadius=radius))
        cdf_num = cdf_crit_sub_conditional(xs, radius, z_cut=.05)

        # Getting pdf from cdf by taking the numerical derivative
        _, pdf_an = histDerivative(cdf_an, bins, giveHist=True,
                                   binInput=BIN_SPACE)
        _, pdf_num = histDerivative(cdf_num, bins, giveHist=True,
                                    binInput=BIN_SPACE)

        # Preparing to plot logarithmically
        pdf_an = xs * pdf_an
        pdf_num = xs * pdf_num

        # Plotting
        col_an = compcolors[(icol%4, 'light')]
        col_num = compcolors[(icol%4, 'dark')]

        axes_pdf[0].plot(xs, pdf_num, **style_solid,
                         color=col_num)
        axes_cdf[0].plot(xs, cdf_num, **style_solid,
                         color=col_num)

        axes_pdf[0].plot(xs, pdf_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius))
        axes_cdf[0].plot(xs, cdf_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius))

    labelLines(axes_pdf[0].get_lines(), xvals=np.logspace(-7,-2,len(THETAS)))
    labelLines(axes_cdf[0].get_lines(), xvals=np.logspace(-7,-2,len(THETAS)))

    legend_darklight(axes_pdf[0], darklabel='Numerical',
                     lightlabel='Analytic', errtype=None,
                     twosigma=False)
    legend_darklight(axes_cdf[0], darklabel='Numerical',
                     lightlabel='Analytic', errtype=None,
                     twosigma=False)

    fig_pdf.savefig(JET_TYPE+'_crit_sub_num_pdf_test.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_crit_sub_num_cdf_test.pdf',
                    format='pdf')
    print("Preparing to plot crit-sub sudakov factor...")

# ------------------------------------
# Pre-critical and Critical
# ------------------------------------
def test_pre_crit_radiator():
    print("Preparing to plot pre-crit radiator...")
    # Preparing figures
    fig_rad, axes_rad, _, _ = \
            get_axes('pre and crit', ratio_plot=False)
    axes_rad[0].set_ylim(1e-15, 35)
    axes_rad[0].set_xlim(1e-20, .1)

    # Analytic plot
    for icol, radius in enumerate(THETAS[:-1]):
        # Setup for cdf
        if BIN_SPACE == 'log':
            bins = np.logspace(-100, np.log10(Z_CUT), 10000)
            xs = np.sqrt(bins[:-1]*bins[1:])
        elif BIN_SPACE == 'lin':
            bins = np.linspace(0, Z_CUT, 1000)
            xs = (bins[:-1]+bins[1:])/2.
        if FIXED_COUPLING:
            rad_an = preRadAnalytic_fc_LL(xs, radius, z_cut=Z_CUT,
                                          jet_type=JET_TYPE)
        else:
            rad_an = preRadAnalytic_nofreeze(xs, radius, z_cut=Z_CUT,
                                             jet_type=JET_TYPE)
        rad_num = (rad_pre[index_zc[Z_CUT]](xs, radius))

        monotone = rad_num[:-1] >= rad_num[1:]
        print(monotone)
        print(monotone.all())
        # Plotting
        col_an = compcolors[(icol%4, 'light')]
        col_num = compcolors[(icol%4, 'dark')]

        axes_rad[0].plot(xs, rad_num, **style_solid,
                         color=col_num)
        axes_rad[0].plot(xs, rad_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius),
                         zorder=5)

    labelLines(axes_rad[0].get_lines(), xvals=np.logspace(-7,-2,len(THETAS)))

    legend_darklight(axes_rad[0], darklabel='Numerical',
                     lightlabel='Analytic')
    if FIXED_COUPLING:
        fig_rad.savefig(JET_TYPE+'_pre_crit_num_rad_test.pdf',
                        format='pdf')
    else:
        fig_rad.savefig(JET_TYPE+'_pre_crit_num_rc_rad_test.pdf',
                        format='pdf')
    print("Plotting complete!")

# ------------------------------------
# Main
# ------------------------------------
if __name__ == '__main__':
    if not FIXED_COUPLING:
        if PLOT_SUB:
            test_crit_sub_radiator_rc()
        if PLOT_PRE:
            test_pre_crit_radiator()
    else:
        if PLOT_SUB:
            test_crit_sub_radiator()
            test_crit_sub_sudakov()
        if PLOT_PRE:
            test_pre_crit_radiator()
