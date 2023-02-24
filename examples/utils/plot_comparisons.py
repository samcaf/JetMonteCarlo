from __future__ import absolute_import
from scipy.special import spence

# Local utilities
from jetmontecarlo.utils.partonshower_utils import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

# Local analytics
from jetmontecarlo.analytics.QCD_utils import *
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

from examples.params import *

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Physics inputs
# ------------------------------------
# Jet and grooming parameters
Z_CUT = .1
F_SOFT = 1
BETA = 2

# ------------------------------------
# Plotting parameters
# ------------------------------------
NUM_BINS = 100
BIN_SPACE = 'log'

if BIN_SPACE == 'lin':
    EPSILON = None
    ylim_1 = (0, 40.)
    ylim_2 = (0, 40.)
    ylim_3 = (0, 40.)
    xlim = (0, .5)
if BIN_SPACE == 'log':
    ylim_1 = (0, .12)
    ylim_2 = (0, .40)
    ylim_3 = (0, .65)
    xlim = (1e-8, .5) if SHOWER_CUTOFF == MU_NP else (1e-20, .5)

"""Verifying valid bin_space."""
assert(BIN_SPACE in ['lin', 'log']), \
"'bin_space' must be 'lin' or 'log', but is" + BIN_SPACE

def dilog(x): return spence(1.-x)

###########################################
# Plotting utilities
###########################################
# ------------------------------------
# Setting up figures
# ------------------------------------
def get_axes(title_info, ratio_plot=False):
    """Shows tests plots for beta=2 GECF distributions."""
    # ---------------------
    # Setting up figures
    # ---------------------
    if BIN_SPACE == 'lin':
        ylabel = r"$\frac{{\rm d}\sigma}{{\rm d}\lambda_{(2)}}$"
    if BIN_SPACE == 'log':
        ylabel = (r"$\frac{1}{\sigma}$"
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\log_{10} C_1^{(2)}}$")
    xlabel=r"$C_1^{(2)} \approx m^2 / p_T^2$"

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=xlabel,
                                   ylabel=ylabel,
                                   ylim=ylim_3,
                                   xlim=xlim,
                                   title=None,#"Groomed ECF PDF ("+title_info+")",
                                   showdate=False,
                                   ratio_plot=ratio_plot,
                                   labeltext=None)

    axespdf[0].set_ylabel(ylabel, rotation=90, fontsize=21)
    axespdf[0].set_xlabel(xlabel, fontsize=20)
    axespdf[0].set_title(None)
    axespdf[0].tick_params(axis='both', which='major', labelsize=12)
    if len(axespdf) > 1:
        axespdf[1].set_ylabel('Ratio', labelpad=0, rotation=0)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$C_1^{(2)}$",
                                   ylabel=r"$\Sigma(C_1^{(2)})$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Groomed ECF CDF ("+title_info+")",
                                   showdate=True,
                                   ratio_plot=ratio_plot)
    axescdf[0].set_ylabel(r"$\Sigma(C_1^{(2)})$", labelpad=23, rotation=0,
                          fontsize=15)
    if len(axescdf) > 1:
        axescdf[1].set_ylabel('Ratio', labelpad=-5, rotation=0)

    if BIN_SPACE == 'log':
        if len(axespdf) > 1:
            axes = [axespdf[0], axespdf[1], axescdf[0], axescdf[1]]
        else:
            axes = [axespdf[0], axescdf[0]]
        for ax in axes:
            ax.set_xscale('log')

    plt.tight_layout()

    return figpdf, axespdf, figcdf, axescdf

# ------------------------------------
# Plot f.c. critical analytic result
# ------------------------------------
def plot_crit_approx(axespdf, axescdf, z_cut, beta=BETA,
                     icol=-1, label='Analytic',
                     multiple_em=False):
    """Plot an approximation to the critical emission
    analytic result.
    """
    # Preparing the bins
    if BIN_SPACE == 'lin':
        bins = np.linspace(0, .5, 100)
        xs = (bins[:-1] + bins[1:])/2.
    if BIN_SPACE == 'log':
        bins = np.logspace(-13, np.log10(.5), 100)
        bins = np.insert(bins, 0, 1e-100)
        xs = np.sqrt(bins[:-1]*bins[1:])

    # Preparing the appropriate cdf
    if 0 < z_cut < 1./2.:
        def rad(c):
            dilog_piece = -dilog(-c/z_cut)
            log_piece = np.log(c)*np.log(2.*z_cut)
            return (dilog_piece + log_piece)\
                   * (2. * alpha_fixed * CA / (beta * np.pi))
        rad_tot = rad(xs) - rad(1./2.-z_cut)
        cdf = np.exp(-rad_tot)

        if multiple_em:
            rad_at_bins = rad(bins) - rad(1./2.-z_cut)
            _, drad = histDerivative(cdf, bins, giveHist=True,
                                     binInput='log') # Only log for now
            rad_logprime = -xs * drad
            me_factor = np.exp(-euler_constant*rad_logprime)\
                        /gamma(1.+rad_logprime)

            cdf = cdf*me_factor

    elif z_cut == 0.:
        # Finding cdf and pdf
        cdf = np.exp(-subRadAnalytic_fc_LL(xs, beta, jet_type=JET_TYPE))
    else:
        raise ValueError(str(z_cut)+' is not a valid z_cut value.')

    # Getting pdf from cdf by taking the numerical derivative
    _, pdf = histDerivative(cdf, bins, giveHist=True,
                            binInput=BIN_SPACE)
    if BIN_SPACE == 'log':
        pdf = xs * pdf * np.log(10) # d sigma / d log10 C

    # Plotting
    col = compcolors[(icol, 'light')]
    axespdf[0].plot(xs, pdf, **style_dashed,
                    color=col, label=label)
    axescdf[0].plot(xs, cdf, **style_dashed,
                    color=col, label=label)
    if len(axespdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_solid, color=col,
                        zorder=.5)
    if len(axescdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_solid, color=col,
                        zorder=.5)

def plot_crit_analytic(axespdf, axescdf, z_cut, beta=BETA, f_soft=1.,
                       jet_type='quark', icol=-1, label='Analytic',
                       fixed_coupling=True, col=None,
                       extra_emission=False):
    """Plot the critical emission analytic result."""
    # Preparing the bins
    if BIN_SPACE == 'lin':
        bins = np.linspace(0, .5, 1000)
        xs = (bins[:-1] + bins[1:])/2.
    if BIN_SPACE == 'log':
        bins = np.logspace(-13, np.log10(.5), 10000)
        bins = np.insert(bins, 0, 1e-100)
        xs = np.sqrt(bins[:-1]*bins[1:])

    # Preparing the appropriate cdf
    if 0 < z_cut < 1./2.:
        cdf = critSudakov_fc_LL(xs, z_cut, beta,
                                jet_type=jet_type,
                                f=f_soft)
        if extra_emission:
            # a is a factor that emerges at LL, e.g. through
            # rho(theta_crit) ~ theta_crit^(a-1).
            # In general, this holds only for f_soft = 1, but we
            # jump around that here by upgrading zcut -> f zcut
            a = 2.*CR(JET_TYPE)*alpha_fixed / np.pi * np.log(f_soft * z_cut)
            mean_theta_crit = a / (1. + a)
            cdf *= np.exp(-subRadAnalytic_fc_LL(xs, beta, jet_type=JET_TYPE,
                                                maxRadius=mean_theta_crit))
    elif z_cut == 0.:
        # Finding cdf and pdf
        if fixed_coupling:
            cdf = np.exp(-subRadAnalytic_fc_LL(xs, beta, jet_type=JET_TYPE))
        else:
            print("Plotting analytic running coupling...")
            cdf = np.exp(-subRadAnalytic(xs, beta, jet_type=JET_TYPE))
    else:
        raise ValueError(str(z_cut)+' is not a valid z_cut value.')

    # Getting pdf from cdf by taking the numerical derivative
    _, pdf = histDerivative(cdf, bins, giveHist=True,
                            binInput=BIN_SPACE)
    if BIN_SPACE == 'log':
        pdf = xs * pdf * np.log(10) # d sigma / d log10 C

    # Plotting
    if col is None:
        col = compcolors[(icol, 'light')]
    axespdf[0].plot(xs, pdf, **style_dashed,
                    color=col, label=label)
    axescdf[0].plot(xs, cdf, **style_dashed,
                    color=col, label=label)
    if len(axespdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_solid, color=col,
                        zorder=.5)
    if len(axescdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_solid, color=col,
                        zorder=.5)

# ------------------------------------
# Plotting Monte Carlo Integrals
# ------------------------------------
def plot_mc_pdf(axespdf, pdf, pdferr, bins, icol=0,
                label='M.C. Integration', col=None):
    """Plots a set of pdf values."""
    if col is None:
        col = compcolors[(icol, 'dark')]

    if BIN_SPACE == 'lin':
        xs = (bins[:-1] + bins[1:])/2.
        xerr = (bins[1:] - bins[:-1])
    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        xerr = (xs - bins[:-1], bins[1:]-xs)
        pdf = xs * pdf * np.log(10) # d sigma / d log10 C
        pdferr = xs * pdferr * np.log(10) # d sigma / d log10 C
    axespdf[0].errorbar(xs, pdf, yerr=pdferr,
                        xerr=xerr, **modstyle,
                        color=col, ecolor=col,
                        label=label)

def plot_mc_cdf(axescdf, cdf, cdferr, bins, icol=0,
                label='M.C. Integration', col=None):
    """Plots a set of cdf values."""
    if col is None:
        col = compcolors[(icol, 'dark')]

    _, _, bars = axescdf[0].errorbar(bins[:-1], cdf, yerr=cdferr,
                                     **style_yerr,
                                     color=col, ecolor=col,
                                     label=label)
    bars = [b.set_alpha(.5) for b in bars]

# ------------------------------------
# Plotting Parton Shower Distributions
# ------------------------------------
def plot_shower_pdf_cdf(vals, axespdf, axescdf,
                        label='Parton Shower',
                        z_cut=Z_CUT,
                        beta=BETA,
                        colnum=1, col=None,
                        style=modstyle_ps,
                        verbose=2):
    """Plots the pdf and cdf associated with the
    set of correlators (vals) on axespdf and axescdf.
    """
    if BIN_SPACE == 'lin':
        # Bins
        bins = np.linspace(0, 1., NUM_BINS)

        # Bin midpoints
        xs = (bins[:-1] + bins[1:])/2.
    else:
        # Bins
        bins = np.logspace(-10, 0., NUM_BINS) if SHOWER_CUTOFF == MU_NP\
               else np.logspace(-18, 0., NUM_BINS)
        bins = np.insert(bins, 0, 1e-100) # zero bin

        # Bin midpoints
        xs = np.sqrt(bins[1:-1] * bins[2:])
        xs = np.insert(xs, 0, 1e-50) # zero bin
        # DEBUG: remains to be seen whether zero bin will throw off normalization

    ps_integrator = integrator()
    ps_integrator.setLastBinBndCondition([1., 'minus'])
    ps_integrator.bins = bins

    if verbose > 3:
        print("bins:", bins)
        print("vals:", vals)
    try:
        num_in_bin, _ = np.histogram(np.array(vals), bins)
    except TypeError:
        vals = vals[0]
        num_in_bin, _ = np.histogram(np.array(vals), bins)
    pdf, _ = np.histogram(vals, bins, density=True)
    pdf_error = pdf / (1e-100 + np.sqrt(num_in_bin))

    ps_integrator.densityErr = pdf_error
    ps_integrator.density = pdf
    ps_integrator.hasMCDensity = True

    ps_integrator.integrate()
    integral = ps_integrator.integral
    interr = ps_integrator.integralErr

    if col is None:
        col = compcolors[(colnum, 'dark')]
        # col = compcolors[(colnum, 'medium')]

    # Numerically obtaining pdf for comparison plots:
    cdf_an = critSudakov_fc_LL(xs, z_cut, beta, f=F_SOFT,
                               jet_type=JET_TYPE)
    _, pdf_an = histDerivative(cdf_an, bins, giveHist=True,
                               binInput=BIN_SPACE)
    pdf_an = np.array(pdf_an.tolist(), dtype=float)

    # ------------------
    # PDF plots:
    # ------------------
    if BIN_SPACE == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_error = xs*pdf_error * np.log(10) # d sigma / d log10 C
        pdf_an = xs*pdf_an * np.log(10) # d sigma / d log10 C
    axespdf[0].errorbar(xs, pdf,
                        yerr=pdf_error,
                        xerr=(bins[1:] - bins[:-1])/2.,
                        **style, color=col,
                        label=label)

    if verbose > 2:
        print("Shower normalization:",
              np.sum(pdf * (np.log10(xs) - np.log10(xs))))


    if len(axespdf) > 1:
        axespdf[1].errorbar(xs, pdf/pdf_an,
                            yerr=pdf_error/pdf_an,
                            xerr=(bins[1:] - bins[:-1])/2.,
                            **style, color=col)

    # ------------------
    # CDF plots:
    # ------------------
    xs = bins[:-1]
    cdf_an = critSudakov_fc_LL(xs, z_cut, beta, f=F_SOFT,
                               jet_type=JET_TYPE)
    cdf_an = np.array(cdf_an.tolist(), dtype=float)

    _, _, bars = axescdf[0].errorbar(xs, integral,
                                     yerr=interr,
                                     **style_yerr_ps,
                                     color=col, ecolor=col,
                                     label=label)
    bars = [b.set_alpha(.5) for b in bars]

    if len(axescdf) > 1:
        _, _, bars_r = axescdf[1].errorbar(xs,
                                           integral/cdf_an,
                                           yerr=interr/cdf_an,
                                           **style_yerr_ps,
                                           color=col, ecolor=col)
        bars_r = [b.set_alpha(.5) for b in bars_r]

def plot_pythia_pdf_cdf(vals, axespdf, axescdf,
                        label='Pythia',
                        z_cut=Z_CUT, beta=BETA,
                        colnum=1, col=None,
                        verbose=0):
    """Plots the pdf and cdf associated with the
    set of correlators (vals) on axespdf and axescdf.
    """
    if BIN_SPACE == 'lin':
        # Bins
        bins = np.linspace(0, 1., NUM_BINS)

        # Bin midpoints
        xs = (bins[:-1] + bins[1:])/2.
    else:
        # Bins
        bins = np.logspace(-10, 0., NUM_BINS) if SHOWER_CUTOFF == MU_NP\
               else np.logspace(-18, 0., NUM_BINS)
        bins = np.insert(bins, 0, 1e-100) # zero bin

        # Bin midpoints
        xs = np.sqrt(bins[1:-1] * bins[2:])
        xs = np.insert(xs, 0, 1e-50) # zero bin
        # DEBUG: remains to be seen whether zero bin will throw off normalization

    bins = np.append(1e-100, bins)
    ps_integrator = integrator()
    ps_integrator.setLastBinBndCondition([1., 'minus'])
    ps_integrator.bins = bins

    if verbose > 3:
        print("bins:", bins)
        print("vals:", vals)
    num_in_bin, _ = np.histogram(np.array(vals), bins)
    pdf, _ = np.histogram(vals, bins, density=True)
    pdf_error = pdf / (1e-100 + np.sqrt(num_in_bin))

    ps_integrator.densityErr = pdf_error
    ps_integrator.density = pdf
    ps_integrator.hasMCDensity = True

    ps_integrator.integrate()
    integral = ps_integrator.integral
    interr = ps_integrator.integralErr

    if col is None:
        col = compcolors[(colnum, 'light')]

    # Analytic cdf and pdf:
    xs = (bins[:-1] + bins[1:])/2.

    # ------------------
    # PDF plots:
    # ------------------
    if BIN_SPACE == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_error = xs*pdf_error * np.log(10) # d sigma / d log10 C
    axespdf[0].errorbar(xs, pdf,
                        yerr=pdf_error,
                        xerr=(bins[1:] - bins[:-1])/2.,
                        **modstyle_ps, color=col,
                        label=label)
    if verbose > 2:
        print("Pythia normalization:",
              np.sum(pdf * (np.log10(xs) - np.log10(xs))))

    # ------------------
    # CDF plots:
    # ------------------
    xs = bins[:-1]
    _, _, bars = axescdf[0].errorbar(xs, integral,
                                     yerr=interr,
                                     **style_yerr_ps,
                                     color=col, ecolor=col,
                                     label=label)
    bars = [b.set_alpha(.5) for b in bars]
