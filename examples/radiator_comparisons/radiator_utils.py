from __future__ import absolute_import

# Plotting utilities
import matplotlib.backends.backend_pdf

# Local utilities
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

# Local analytics
from jetmontecarlo.analytics.QCD_utils import *
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *

# Parameters
from examples.params import *

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]

if FIXED_COUPLING:
    extra_label = '_fc_num'
else:
    extra_label = '_rc_num'

ylims = {'quark': [5.5,3.5,2.5,1.25],
         'gluon': [10,7.5,5,3]}


###########################################
# Plotting Radiators
###########################################
# ==========================================
# Critical Radiator:
# ==========================================
def compare_crit_rad():
    # Files and filenames
    filename = JET_TYPE+extra_label+"_critical_radiators.pdf"

    with open(critrad_path, 'rb') as file:
        rad_crit_list = pickle.load(file)
    global rad_crit
    def rad_crit(theta, z_cut):
        return rad_crit_list[INDEX_ZC[z_cut]](theta)

    # Setting up plot
    fig, axes = aestheticfig(xlabel=r'$\theta$',
                        ylabel=r'$R_{{\rm crit}}(\theta)$',
                        xlim=(1e-8,1),
                        ylim=(0, ylims[JET_TYPE][2]),
                        title = 'Critical '+JET_TYPE+' radiator, '
                                + ('fixed' if FIXED_COUPLING else 'running')
                                + r' $\alpha_s$',
                        showdate=False,
                        ratio_plot=False)
    axes[0].set_xscale('log')
    pnts = np.logspace(-8.5, 0, 1000)

    for izc, zcut in enumerate(Z_CUT_PLOT):
        # Plotting numerical result
        axes[0].plot(pnts, rad_crit(pnts, zcut),
                    **style_solid, color=compcolors[(izc, 'dark')],
                    label=r'Numeric, $z_{{\rm cut}}$={}'.format(zcut))

        # Plotting analytic result
        axes[0].plot(pnts,
                    critRadAnalytic_fc_LL(pnts, zcut, jet_type=JET_TYPE),
                    **style_dashed, color=compcolors[(izc, 'light')],
                    label=r'Analytic, $z_{{\rm cut}}$={}'.format(zcut))

    axes[0].legend()
    plt.savefig(filename, format='pdf')

    print("Plotting complete!")

# ==========================================
# Pre-Critical Radiator:
# ==========================================
def compare_pre_rad():
    # Files and filenames
    filename = JET_TYPE+extra_label+"_precrit_radiators.pdf"
    pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

    with open(prerad_path, 'rb') as file:
        rad_pre_list = pickle.load(file)
    global rad_pre
    def rad_pre(z_pre, theta, z_cut):
        return rad_pre_list[INDEX_ZC[z_cut]](z_pre, theta)

    for izc, zcut in enumerate(Z_CUT_PLOT):
        # Setting up plot
        fig, axes = aestheticfig(xlabel=r'$z_{{\rm pre}}$',
                            ylabel=r'$R_{{\rm pre}}(z_{{\rm pre}})$',
                            xlim=(1e-8,1),
                            ylim=(0,ylims[JET_TYPE][izc]),
                            title = 'Pre-Critical '+JET_TYPE+', '
                                    + ('fixed' if FIXED_COUPLING else 'running')
                                    + r' $\alpha_s$, $z_c$='+str(zcut),
                            showdate=False,
                            ratio_plot=False)
        axes[0].set_xscale('log')

        # Getting numerical radiator, choosing angles to plot
        theta_list = [.05, .1, .5, .9]
        pnts = np.logspace(-8.5, 0, 1000)

        for i, theta in enumerate(theta_list):
            # Plotting numerical result
            axes[0].plot(pnts, rad_pre(pnts, theta, zcut),
                        **style_solid, color=compcolors[(i, 'dark')],
                        label=r'Numeric, $\theta$={}'.format(theta))

            # Plotting analytic result
            axes[0].plot(pnts,
                        preRadAnalytic_fc_LL(pnts, theta, zcut, jet_type=JET_TYPE),
                        **style_dashed, color=compcolors[(i, 'light')],
                        label=r'Analytic, $\theta$={}'.format(theta))

        # Legend, saving
        axes[0].legend()
        plt.savefig(pdffile, format='pdf')

    pdffile.close()

    print("Plotting complete!")

# ==========================================
# Subsequent Radiator:
# ==========================================
def compare_sub_rad():
    # Files and filenames
    filename = JET_TYPE+extra_label+"_sub_radiators.pdf"
    pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

    # Getting numerical radiator, choosing angles to plot
    with open(subrad_path, 'rb') as file:
        rad_sub_list = pickle.load(file)
    global rad_sub
    def rad_sub(c_sub, theta, beta):
        return rad_sub_list[INDEX_BETA[beta]](c_sub, theta)

    theta_list = [.05, .1, .5, .9]
    pnts = np.logspace(-8.5, 0, 1000)

    for ib, beta in enumerate(BETAS):
        print(beta)
        # Setting up plot
        fig, axes = aestheticfig(xlabel=r'$C$',
                                 ylabel=r'$R_{{\rm sub}}(C)$',
                                 xlim=(1e-8,1),
                                 ylim=(0,ylims[JET_TYPE][0]),
                                 title = 'Subsequent '+JET_TYPE
                                 + ('fixed' if FIXED_COUPLING else 'running')
                                 + r' $\alpha_s$, $\beta$={}'.format(beta),
                                 showdate=False,
                                 ratio_plot=False)
        axes[0].set_xscale('log')
        pnts = np.logspace(-8.5, 0, 1000)

        for i, theta in enumerate(theta_list):
            print(rad_sub(pnts, theta, beta))
            # Plotting numerical result
            axes[0].plot(pnts, rad_sub(pnts, theta, beta),
                        **style_solid, color=compcolors[(i, 'dark')],
                        label=r'Numeric, $\theta$={}'.format(theta))

            # Plotting analytic result
            axes[0].plot(pnts,
                         subRadAnalytic_fc_LL(pnts/theta**beta, beta, jet_type=JET_TYPE),
                        **style_dashed, color=compcolors[(i, 'light')],
                        label=r'Analytic, $\theta$={}'.format(theta))

        # Legend, saving
        axes[0].legend()
        plt.savefig(pdffile, format='pdf')

    pdffile.close()

    print("Plotting complete!")
