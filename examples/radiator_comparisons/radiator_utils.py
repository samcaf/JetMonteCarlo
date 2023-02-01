# Plotting utilities
import matplotlib.backends.backend_pdf

# Local utilities
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

# Local analytics
from jetmontecarlo.analytics.QCD_utils import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.radiators import *

# Parameters and Utilities
from examples.params import tab, BETAS, RADIATOR_PARAMS
from examples.data_management import load_and_interpolate
from examples.file_management import fig_folder


# =====================================
# Definitions and Parameters
# =====================================
test_monotonicity = False

# ---------------------------------
# Unpacking parameters
# ---------------------------------
params = RADIATOR_PARAMS
del params['z_cut']
del params['beta']

jet_type = params['jet type']
fixed_coupling = params['fixed coupling']

num_mc_events = params['number of MC events']

num_rad_bins = params['number of bins']


# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]

ylims = {'quark': [5.5,3.5,2.5,1.25],
         'gluon': [10,7.5,5,3]}

def fig_file_name(rad_type):
    """Returns a file name associated with the given radiator
    type and the information imported from `params.py`.
    """
    return str(fig_folder) + "/" + rad_type+"rads_"+jet_type\
           + ('_fc_num' if fixed_coupling else '_rc_num')\
           + "_{:.0e}samples".format(num_mc_events)\
           + "_{:.0e}bins.pdf".format(num_rad_bins)


# =====================================
# Getting radiators
# =====================================
critical_radiator = {}
precritical_radiator = {}
subsequent_radiator = {}

print("Loading critical radiators\n")
for z_cut in Z_CUT_PLOT:
    critical_radiator[z_cut] = load_and_interpolate(
                    'critical radiator',
                    params=dict(**params, **{'z_cut': z_cut}),
                    monotonic=True, bounds=(1e-10, 1),
    )

print("Loading pre-critical radiators\n")
for z_cut in Z_CUT_PLOT:
    precritical_radiator[z_cut] = load_and_interpolate(
                    'pre-critical radiator',
                    params=dict(**params, **{'z_cut': z_cut}),
                    interpolation_method="RectangularGrid")

print("Loading subsequent radiators\n")
for b in BETAS:
    subsequent_radiator[b] = load_and_interpolate(
                    'subsequent radiator',
                    params=dict(**params, **{'beta': b}),
                    interpolation_method='Nearest')


###########################################
# Plotting Radiators
###########################################
# ==========================================
# Critical Radiator:
# ==========================================
def compare_crit_rad():
    print("Comparing numerical and analytic critical radiators")
    # Setting up plot
    _, axes = aestheticfig(xlabel=r'$\theta$',
                        ylabel=r'$R_{{\rm crit}}(\theta)$',
                        xlim=(1e-8,1),
                        ylim=(0, ylims[jet_type][2]),
                        title = 'Critical '+jet_type+' radiator, '
                                + (' fixed' if fixed_coupling else ' running')
                                + r' $\alpha_s$',
                        showdate=False,
                        ratio_plot=False)
    axes[0].set_xscale('log')
    pnts = np.logspace(-8.5, 0, 1000)

    for izc, zcut in enumerate(Z_CUT_PLOT):
        # Plotting numerical result
        num_result = critical_radiator[zcut](pnts)

        # Monotonicity tests
        if test_monotonicity:
            where_monotonic = num_result[1:] <= num_result[:-1]
            print(tab+f"is_monotonic : {where_monotonic.all()}")
            if not where_monotonic.all():
                print(tab+f"{pnts[:-1][~where_monotonic] = }")
                print(tab+"Non-monotonic values: ")
                print(tab+
                      np.transpose([num_result[:-1][~where_monotonic],
                                num_result[1:][~where_monotonic]]))
                print(tab+"percentage non-monotonic: " +
                  f"{len(pnts[:-1][~where_monotonic])/(len(pnts)-1)}")

        axes[0].plot(pnts, num_result,
                    **style_solid, color=compcolors[(izc, 'dark')],
                    label=r'Numeric, $z_{{\rm cut}}$={}'.format(zcut))

        # Plotting analytic result
        if fixed_coupling:
            an_result = critRadAnalytic_fc_LL(pnts, zcut,
                                              jet_type=jet_type)
        else:
            an_result = critRadAnalytic(pnts, zcut,
                                        jet_type=jet_type)
        axes[0].plot(pnts, an_result,
                    **style_dashed, color=compcolors[(izc, 'light')],
                    label=r'Analytic, $z_{{\rm cut}}$={}'.format(zcut))

    axes[0].legend()
    plt.savefig(fig_file_name('crit'), format='pdf')

    print(tab+"Plotting complete!")
    print(tab+f"Figure saved to {fig_file_name('crit')}\n")

# ==========================================
# Pre-Critical Radiator:
# ==========================================
def compare_pre_rad():
    print("Comparing numerical and analytic pre-critical radiators")
    # Setting up file to save several plots
    pdffile = matplotlib.backends.backend_pdf.PdfPages(fig_file_name('precrit'))

    for izc, zcut in enumerate(Z_CUT_PLOT):
        # Setting up plot
        _, axes = aestheticfig(xlabel=r'$z_{{\rm pre}}$',
                            ylabel=r'$R_{{\rm pre}}(z_{{\rm pre}})$',
                            xlim=(1e-8, zcut),
                            ylim=(-.01,ylims[jet_type][izc]),
                            title = 'Pre-Critical '+jet_type+', '
                                    + (' fixed' if fixed_coupling else ' running')
                                    + r' $\alpha_s$, $z_c$='+str(zcut),
                            showdate=False,
                            ratio_plot=True)
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')

        # Getting numerical radiator, choosing angles to plot
        theta_list = [.05, .1, .5, .9]
        pnts = np.logspace(-8.5, 0, 1000)

        for i, theta in enumerate(theta_list):
            # Numerical
            num_result = precritical_radiator[zcut]([[pnt, theta]
                                                     for pnt in pnts])

            # Monotonicity tests
            if test_monotonicity:
                where_monotonic = num_result[1:] <= num_result[:-1]
                print(tab+f"is_monotonic : {where_monotonic.all()}")
                if not where_monotonic.all():
                    print(tab+f"{pnts[:-1][~where_monotonic] = }")
                    print(tab+"Non-monotonic values: ")
                    print(tab+np.transpose([num_result[:-1][~where_monotonic],
                                   num_result[1:][~where_monotonic]]))
                    print(tab+"percentage non-monotonic: " +
                      f"{len(pnts[:-1][~where_monotonic])/(len(pnts)-1)}")

            # Analytic
            if fixed_coupling:
                an_result = preRadAnalytic_fc_LL(pnts, theta, zcut, jet_type=jet_type)
            else:
                an_result = preRadAnalytic_nofreeze(pnts, theta, zcut, jet_type=jet_type)

            # Ratio
            num_ratio = num_result/an_result

            # Plotting numerical result
            axes[0].plot(pnts, num_result,
                        **style_solid, color=compcolors[(i, 'dark')],
                        label=r'Numeric, $\theta$={}'.format(theta))

            axes[1].plot(pnts, num_result / an_result,
                         **style_solid, color=compcolors[(i, 'dark')])

            # Plotting analytic result
            axes[0].plot(pnts, an_result,
                         **style_dashed, color=compcolors[(i, 'light')],
                         label=r'Analytic, $\theta$={}'.format(theta))

            axes[1].plot(pnts, np.ones(len(pnts)),
                         **style_dashed, color='black')

        # Legend, saving
        axes[0].legend()
        plt.savefig(pdffile, format='pdf')

    pdffile.close()

    print(tab+"Plotting complete!")
    print(tab+f"Figure saved to {fig_file_name('precrit')}\n")

# ==========================================
# Subsequent Radiator:
# ==========================================
def compare_sub_rad():
    print("Comparing numerical and analytic subsequent radiators")
    pdffile = matplotlib.backends.backend_pdf.PdfPages(
        fig_file_name('sub'))

    # Getting numerical radiator, choosing angles to plot
    theta_list = [.05, .1, .5, .9]
    pnts = np.logspace(-8.5, 0, 1000)

    for b in BETAS:
        # Setting up plot
        _, axes = aestheticfig(xlabel=r'$C$',
                                 ylabel=r'$R_{{\rm sub}}(C)$',
                                 xlim=(1e-8,1),
                                 ylim=(0,ylims[jet_type][0]),
                                 title = 'Subsequent '+jet_type
                                 + (' fixed' if fixed_coupling else ' running')
                                 + r' $\alpha_s$, $\beta$={}'.format(b),
                                 showdate=False,
                                 ratio_plot=False)
        axes[0].set_xscale('log')
        pnts = np.logspace(-8.5, 0, 1000)

        for i, theta in enumerate(theta_list):
            # Plotting numerical result
            num_result = subsequent_radiator[b](pnts, theta)

            # Monotonicity tests
            if test_monotonicity:
                where_monotonic = num_result[1:] <= num_result[:-1]
                print(tab+f"is_monotonic : {where_monotonic.all()}")
                if not where_monotonic.all():
                    print(tab+f"{pnts[:-1][~where_monotonic] = }")
                    print(tab+"Non-monotonic values: ")
                    print(tab+np.transpose([num_result[:-1][~where_monotonic],
                                       num_result[1:][~where_monotonic]]))
                    print(tab+"percentage non-monotonic: " +
                      f"{len(pnts[:-1][~where_monotonic])/(len(pnts)-1)}")

            axes[0].plot(pnts, num_result,
                         **style_solid, color=compcolors[(i, 'dark')],
                         label=r'Numeric, $\theta$={}'.format(theta))

            # Plotting analytic result
            if fixed_coupling:
                an_result = subRadAnalytic_fc_LL(pnts/theta**b, b,
                                                 jet_type=jet_type)
            else:
                an_result = subRadAnalytic(pnts, b,
                                           jet_type=jet_type,
                                           maxRadius=theta)
            axes[0].plot(pnts, an_result,
                         **style_dashed, color=compcolors[(i, 'light')],
                         label=r'Analytic, $\theta$={}'.format(theta))

        # Legend, saving
        axes[0].legend()
        plt.savefig(pdffile, format='pdf')

    pdffile.close()

    print(tab+"Plotting complete!")
    print(tab+f"Figure saved to {fig_file_name('sub')}\n")
