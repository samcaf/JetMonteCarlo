# Local analytics
from jetmontecarlo.utils.montecarlo_utils import getLinSample
from jetmontecarlo.jets.observables import *
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *
from jetmontecarlo.montecarlo.partonshower import *

# Plotting utilities
from examples.comparison_plot_utils import *

# Parameters and Filenames
# DEBUG: Remove * imports
from examples.params import *

# Loading Data and Functions
from examples.load_data import load_radiators
from examples.load_data import load_splittingfns
from examples.load_data import load_sudakov_samples


# =====================================
# Definitions and Parameters
# =====================================

# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]
Z_CUT_PLOT = [F_SOFT * zc for zc in Z_CUT_PLOT]

F_SOFT_PLOT = [.5, 1]
F_SOFT_PLOT_IVS = [.5, 1, 'ivs']
F_SOFT_STR = ['1/2', '1']

f_colors = {.5: 'goldenrod', 1: 'forestgreen', 'ivs': 'darkmagenta'}

plot_colors = {k: {
                'fc': adjust_lightness(f_colors[k], 1),
                'num': adjust_lightness(f_colors[k], .75),
                'shower': adjust_lightness(f_colors[k], .75),
                'pythia': adjust_lightness(f_colors[k], .5)
                } for k in f_colors.keys()}

def f_ivs(theta):
    return 1./2. - theta/(np.pi*R0)


if FIXED_COUPLING:
    plot_label = '_fc_num_'+str(OBS_ACC)
else:
    plot_label = '_rc_num_'+str(OBS_ACC)

if MULTIPLE_EMISSIONS:
    plot_label += 'ME_'

plot_label += '_showerbeta'+str(SHOWER_BETA)
if F_SOFT:
    plot_label += '_f{}'.format(F_SOFT)


# ---------------------------------
# Loading Samples
# ---------------------------------
# DEBUG: Need to implement
sudakov_inverse_transforms = load_sudakov_samples()

# ------------------------------------
# Loading Functions:
# ------------------------------------
splitting_functions = load_splittingfns()

radiators = {}
if not(LOAD_MC_EVENTS):
    radiators = load_radiators()


###########################################
# Additional Plot Utils
###########################################
# DEBUG: Too buggy, leaving out
def plot_mc_banded(ax, ys, err, bins, label, col, drawband=False):
    if BIN_SPACE == 'lin':
        xs = (bins[:-1] + bins[1:])/2.
        xerr = (bins[1:] - bins[:-1])
    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        xerr = (xs - bins[:-1], bins[1:]-xs)
        ys = xs * ys * np.log(10) # dY / d log10 C
        err = xs * err * np.log(10) # delta( dY / d log10 C)

    line = ax.plot(xs, ys, ls='-', lw=2., color=col, label=label)

    # DEBUG: Leaving out error bands because they act weird
    # if drawband:
    #     band = draw_error_band(ax, xs, ys, err,
    #                            color=col, alpha=.4)
    #     return line, band

    return line, None

def full_legend(ax, labels, loc='upper left', drawband=False):
    ax.plot(-100, -100, **style_dashed, color=compcolors[(-1, 'medium')],
            label=labels[0])
    line, _ = plot_mc_banded(ax, [-100,-100], [1,1], np.array([-100,-99,-98]),
                             label=labels[1], col=compcolors[(-1, 'dark')],
                             drawband=drawband)
    if len(labels)>=3:
        ax.errorbar(-100., -100, yerr=1., xerr=1., **modstyle,
                    color=compcolors[(-1, 'dark')],
                    label=labels[2])
    if len(labels)>=4:
        ax.hist(np.arange(-100,-90), 5,
                histtype='step', lw=2, edgecolor=compcolors[(-1, 'dark')],
                label=labels[3])

    handles, _ = ax.get_legend_handles_labels()
    if len(handles)>=4:
        new_handles = [handles[0], handles[1], handles[3], handles[2]]
    elif len(handles)>=3:
        new_handles = [handles[0], handles[1], handles[2]]
    elif len(handles)>=2:
        new_handles = [handles[0], handles[1]]

    ax.legend(new_handles, labels, loc=loc)

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                 load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    # DEBUG: Need to use both samples and weights when saving
    theta_crits        = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['samples']
    theta_crit_weights = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['weights']
    # theta_crits, theta_crit_weights, load = get_theta_crits(
    #                       z_cut, beta, load=load, save=True,
    #                       rad_crit=radiators.get('critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = splitting_functions[z_cut](z_crits, theta_crits)

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=f_soft, acc=OBS_ACC)

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1,
                                          np.log10(.75),
                                          NUM_BINS)
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdf_err = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr
    bins = sud_integrator.bins

    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        print("Critical MC normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))
        print("    Method 2: ", integral[-1])

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf,
                                      2.*pdf_err, bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral,
                                      integralerr, bins,
                                      label=None, col=col)

    if BIN_SPACE == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_err = xs*pdf_err * np.log(10) # d sigma / d log10 C
        print("Critical MC adjusted normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))


    return pdfline, pdfband, cdfline, cdfband


###########################################
# All Emissions
###########################################
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    # DEBUG: Need to use both samples and weights when saving
    theta_crits        = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['samples']
    theta_crit_weights = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['weights']
    # theta_crits, theta_crit_weights, load = get_theta_crits(
    # theta_crits, theta_crit_weights, load = get_theta_crits(
    #                       z_cut, beta, load=load, save=True,
    #                       rad_crit=radiators.get('critical', None))

    # DEBUG: Need to use both samples and weights when saving
    c_subs        = sudakov_inverse_transforms['subsequent']\
                                    [z_cut][beta]['samples']
    c_sub_weights = sudakov_inverse_transforms['subsequent']\
                                    [z_cut][beta]['weights']
    # c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
    #                      load=load, save=True, theta_crits=theta_crits,
    #                      rad_crit_sub=radiators.get('subsequent', None))

    # DEBUG: Need to use both samples and weights when saving
    z_pres        = sudakov_inverse_transforms['pre-critical']\
                                    [z_cut]['samples']
    z_pre_weights = sudakov_inverse_transforms['pre-critical']\
                                    [z_cut]['weights']
    # z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=True,
    #                     theta_crits=theta_crits,
    #                     rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = splitting_functions[z_cut](z_crits, theta_crits)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=f_soft, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdf_err = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr
    bins = sud_integrator.bins

    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        print("All MC normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf,
                                      2.*pdf_err, bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral,
                                      integralerr, bins,
                                      label=None, col=col)

    if BIN_SPACE == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_err = xs*pdf_err * np.log(10) # d sigma / d log10 C

        print("All MC adjusted normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))

    return pdfline, pdfband, cdfline, cdfband
