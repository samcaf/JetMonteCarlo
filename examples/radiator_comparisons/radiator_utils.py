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
    extra_label = '_fc_num_'
else:
    extra_label = '_rc_num_'

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
        return rad_pre_list[INDEX_ZC[z_cut]](theta)

    # Setting up plot
    fig, axes = aestheticfig(xlabel=r'$\theta$',
                        ylabel=r'R_{\rm crit}($\theta$)',
                        xlim=(1e-8,1),
                        ylim=(0, ylims[JET_TYPE][izc]),
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
                    **style_solid, color=compcolors[(i, 'dark')],
                    label=r'Numeric, $z_{\rm cut}$={}'.format(zcut))

        # Plotting analytic result
        axes[0].plot(pnts,
                    critRadAnalytic_fc_LL(pnts, zcut, JET_TYPE=JET_TYPE),
                    **style_dashed, color=compcolors[(i, 'light')],
                    label=r'Analytic, $z_{\rm cut}$={}'.format(zcut))

    plt.legend(axes[0])
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
        fig, axes = aestheticfig(xlabel=r'$z_{\rm pre}$',
                            ylabel=r'R_{\rm pre}(z_{\rm pre}~|~$\theta$)',
                            xlim=(1e-8,1),
                            ylim=(0,ylims[JET_TYPE][izc]),
                            title = 'Pre-Critical '+JET_TYPE+' radiator, '
                                    + ('fixed' if FIXED_COUPLING else 'running')
                                    + r' $\alpha_s$, $z_c$='+str(zcut),
                            showdate=False,
                            ratio_plot=False)
        axes[0].set_xscale('log')

        # Getting numerical radiator, choosing angles to plot
        theta_list = [.05, .1, .5, .9]

        for i, theta in enumerate(theta_list):
            pnts = np.logspace(-8.5+np.log10(theta), np.log10(theta), 1000)

            # Plotting numerical result
            axes[0].plot(pnts, rad_pre(pnts, theta, zcut),
                        **style_solid, color=compcolors[(i, 'dark')],
                        label=r'Numeric, $\theta$={}'.format(theta))

            # Plotting analytic result
            axes[0].plot(pnts,
                        preRadAnalytic_fc_LL(pnts, theta, zcut, JET_TYPE=JET_TYPE),
                        **style_dashed, color=compcolors[(i, 'light')],
                        label=r'Analytic, $\theta$={}'.format(theta))

        # Legend, saving
        plt.legend(axes[0])
        plt.savefig(pdffile, format='pdf')

    pdffile.close()

    print("Plotting complete!")

# ==========================================
# Subsequent Radiator:
# ==========================================
def compare_sub_rad():
    # Files and filenames
    filename = JET_TYPE+extra_label+"_sub_radiators.pdf"

    with open(subrad_path, 'rb') as file:
        rad_sub_list = pickle.load(file)
    global rad_sub
    def rad_sub(c_sub, beta):
        return rad_pre_list[INDEX_BETA[beta]](c_sub)

    # Setting up plot
    fig, axes = aestheticfig(xlabel=r'$C$',
                        ylabel=r'R_{\rm sub}(C)',
                        xlim=(1e-8,1),
                        ylim=(0,ylims[JET_TYPE][izc]),
                        title = 'Subsequent '+JET_TYPE+' radiator, '
                                + ('fixed' if FIXED_COUPLING else 'running')
                                + r' $\alpha_s$, $z_c$='+str(zcut),
                        showdate=False,
                        ratio_plot=False)
    axes[0].set_xscale('log')
    pnts = np.logspace(-8.5, 0, 1000)

    for ib, beta in enumerate(BETAS):
        # Plotting numerical result
        axes[0].plot(pnts, rad_pre(pnts, beta),
                    **style_solid, color=compcolors[(i, 'dark')],
                    label=r'Numeric, $\beta$={}'.format(beta))

        # Plotting analytic result
        axes[0].plot(pnts,
                    subRadAnalytic_fc_LL(pnts, beta, JET_TYPE=JET_TYPE),
                    **style_dashed, color=compcolors[(i, 'light')],
                    label=r'Analytic, $\beta$={}'.format(beta))

    # Legend, saving
    plt.legend(axes[0])
    plt.savefig(filename, format='pdf')

    print("Plotting complete!")
