# Local MC utilities
from jetmontecarlo.utils.montecarlo_utils import getLinSample
from jetmontecarlo.utils.montecarlo_utils import samples_from_cdf
from jetmontecarlo.montecarlo.integrator import integrator
from jetmontecarlo.montecarlo.partonshower import *

# Local functions and analytics
from jetmontecarlo.jets.observables import *
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

# Plotting utilities
from jetmontecarlo.utils.color_utils import compcolors,\
    adjust_lightness
from jetmontecarlo.utils.plot_utils import modstyle, style_dashed

# Parameters
from examples.comparison_plot_utils import F_SOFT
from examples.params import SHOWER_BETA, MULTIPLE_EMISSIONS, \
    Z_CUTS, BETAS, LOAD_MC_EVENTS
from examples.params import RADIATOR_PARAMS

# Loading Data and Functions
from examples.data_management import save_new_data
from examples.load_data import load_radiators
from examples.load_data import load_splittingfns
from examples.load_data import load_sudakov_samples


# =====================================
# Definitions and Parameters
# =====================================
sudakov_params = RADIATOR_PARAMS.copy()
del sudakov_params['z_cut']
del sudakov_params['beta']

# Unpacking parameters
fixed_coupling   = sudakov_params['fixed coupling']
obs_acc          = sudakov_params['observable accuracy']

num_mc_events    = sudakov_params['number of MC events']
epsilon          = sudakov_params['epsilon']
bin_space        = sudakov_params['bin space']

num_bins         = sudakov_params['number of bins']

load_mc_events   = LOAD_MC_EVENTS

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
    """The (theta dependent) value of f_soft which I
    expect may be used for IVS.
    """
    return 1./2. - theta/(np.pi*R0)


if fixed_coupling:
    plot_label = f'_fc_num_{obs_acc}'
else:
    plot_label = f'_rc_num_{obs_acc}'

if MULTIPLE_EMISSIONS:
    plot_label += 'ME_'

plot_label += f'_showerbeta{SHOWER_BETA}'
if F_SOFT:
    plot_label += '_f{F_SOFT}'


# =====================================
# Loading or Generating Samples
# =====================================

# ---------------------------------
# Functions which generate samples
# ---------------------------------
def generate_critical_samples(radiators, params,
                              z_cut,
                              verbose=3):
    """Generates samples via the inverse transform method
    for critical angles from Sudakov factors.
    """
    print(f"    Making critical samples with z_c={z_cut}...")

    def cdf_crit(theta):
        return np.exp(-1.*radiators['critical'][z_cut](theta))

    theta_crits, theta_crit_weights = samples_from_cdf(cdf_crit,
                                            num_mc_events,
                                            domain=[0.,1.],
                                            backup_cdf=None,
                                            verbose=verbose)

    # Formatting samples
    theta_crit_weights = np.where(np.isinf(theta_crits), 0,
                                  theta_crit_weights)
    theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

    # Saving
    save_new_data({'samples': theta_crits,
                   'weights': theta_crit_weights},
                  'montecarlo samples',
                  'critical sudakov inverse transform',
                  params=dict(**params,
                              **{'z_cut': z_cut}),
                  extension='.npz')

    return theta_crits, theta_crit_weights


def generate_precritical_samples(radiators, params,
                                theta_crits, z_cut,
                                verbose=10):
    """Generates samples via the inverse transform method
    for pre-critical energy fractions from Sudakov factors.
    """
    print("    Making pre-critical samples from critical" \
          f" samples with z_c={z_cut}...")

    z_pres = []
    z_pre_weights = []

    for i, theta in enumerate(theta_crits):
        def cdf_pre_conditional(z_pre):
            if hasattr(z_pre, "__len__"):
                return np.exp(-1.*radiators['pre-critical'][z_cut](
                                   [[z_p, theta] for z_p in z_pre]))
            else:
                return np.exp(-1.*radiators['pre-critical'][z_cut](
                                   [z_pre, theta]))

        z_pre, z_pre_weight = samples_from_cdf(cdf_pre_conditional, 1,
                                            domain=[0, z_cut],
                                            backup_cdf=None,
                                            force_monotone=True,
                                            verbose=verbose)
        # Processing and saving one sample at a time
        z_pre = z_pre[0]
        z_pre_weight = z_pre_weight[0]

        z_pres.append(z_pre)
        z_pre_weights.append(z_pre_weight)
        if (i+1)%(len(theta_crits)/10) == 0:
            print("        Generated "+str(i+1)+" events...",
                  flush=True)

    # Formatting samples
    z_pres = np.array(z_pres)
    z_pre_weights = np.array(z_pre_weights)
    z_pre_weights = np.where(np.isinf(z_pres), 0, z_pre_weights)
    z_pres = np.where(np.isinf(z_pres), 0, z_pres)

    # Saving
    save_new_data({'samples': z_pres,
                   'weights': z_pre_weights},
                  'montecarlo samples',
                  'pre-critical sudakov inverse transform',
                  params=dict(**params,
                              **{'z_cut': z_cut}),
                  extension='.npz')

    return z_pres, z_pre_weights


def generate_subsequent_samples(radiators, params,
                                theta_crits, z_cut, beta,
                                verbose=3):
    """Generates samples via the inverse transform method
    for subsequent ECFs from Sudakov factors.
    """
    print(f"    Making subsequent samples with {beta=}" \
          f" from critical samples with z_c={z_cut}...")

    c_subs = []
    c_sub_weights = []

    for i, theta in enumerate(theta_crits):
        def cdf_sub_conditional(c_sub):
            return np.exp(-1.*radiators['subsequent'][beta]\
                                        (c_sub, theta))

        if theta**beta/2. < 1e-10:
            # Assigning to an underflow bin for small observable values
            c_sub = 1e-100
            c_sub_weight = 1.
        else:
            c_sub, c_sub_weight = samples_from_cdf(cdf_sub_conditional, 1,
                                            domain=[0.,theta**beta/2.],
                                            backup_cdf=None,
                                            force_monotone=True,
                                            verbose=verbose)
            c_sub, c_sub_weight = c_sub[0], c_sub_weight[0]

        # Processing and saving one sample at a time
        c_subs.append(c_sub)
        c_sub_weights.append(c_sub_weight)
        if (i+1)%(len(theta_crits)/10) == 0:
            print("        Generated "+str(i+1)+" events...", flush=True)

    # Formatting samples
    c_subs = np.array(c_subs)
    c_sub_weights = np.array(c_sub_weights)
    c_sub_weights = np.where(np.isinf(c_subs), 0, c_sub_weights)
    c_subs = np.where(np.isinf(c_subs), 0, c_subs)

    # Saving
    save_new_data({'samples': c_subs,
                   'weights': c_sub_weights},
                  'montecarlo samples',
                  'subsequent sudakov inverse transform',
                  params=dict(**params,
                              **{'z_cut': z_cut,
                                 'beta': beta}),
                  extension='.npz')

    return c_subs, c_sub_weights


def generate_sudakov_samples(radiators, params,
                             z_cuts=Z_CUTS, betas=BETAS,
                             emissions=['pre-critical',
                                        'critical', 'subsequent']):
    """Generates samples via the inverse transform method for
    Sudakov factors for different emissions and parameters.
    """
    all_theta_crits = {zc: None for zc in z_cuts}
    if 'critical' in emissions:
        for z_cut in z_cuts:
            theta_crits, _ = generate_critical_samples(
                                             radiators, params,
                                             z_cut)
            all_theta_crits[z_cut] = theta_crits

    if 'pre-critical' in emissions:
        assert 'critical' in emissions, \
            "Critical Sudakov samples must be generated before " \
            "pre-critical ones."
        for z_cut in z_cuts:
            generate_precritical_samples(radiators, params,
                                         all_theta_crits[z_cut],
                                         z_cut)

    if 'subsequent' in emissions:
        assert 'critical' in emissions, \
            "Critical Sudakov samples must be generated before " \
            "subsequent ones."
        for z_cut in z_cuts:
            for beta in betas:
                generate_subsequent_samples(radiators, params,
                                            all_theta_crits[z_cut],
                                            z_cut, beta)

    del all_theta_crits, theta_crits

    sudakov_inverse_transforms = load_sudakov_samples()
    return sudakov_inverse_transforms


# ---------------------------------
# Loading
# ---------------------------------
sudakov_inverse_transforms = None

if load_mc_events:
    try:
        sudakov_inverse_transforms = load_sudakov_samples()
    except FileNotFoundError:
        print("Files for Sudakov inverse transform samples not found.")
    load_mc_events = False

# ------------------------------------
# Loading Functions, Generating Samples:
# ------------------------------------
splitting_functions = load_splittingfns()

if not load_mc_events:
    radiator_fns = load_radiators()
    sudakov_inverse_transforms = generate_sudakov_samples(radiator_fns,
                                                          sudakov_params)


###########################################
# Additional Plot Utils
###########################################
# DEBUG: Too buggy, leaving out
def plot_mc_banded(ax, ys, err, bins, label, col, drawband=False):
    if bin_space == 'lin':
        xs = (bins[:-1] + bins[1:])/2.
    if bin_space == 'log':
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
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, f_soft, col):
    """Plots the distribution obtained through Monte Carlo for
    jet ECFs using only critical emissions.
    """
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits        = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['samples']
    theta_crit_weights = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['weights']

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])
    weights = splitting_functions[z_cut](z_crits, theta_crits)
    weights = weights * theta_crit_weights

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=f_soft, acc=obs_acc)

    # Weights, binned observables, and area
    if bin_space == 'lin':
        sud_integrator.bins = np.linspace(0, .5, num_bins)
    if bin_space == 'log':
        sud_integrator.bins = np.logspace(np.log10(epsilon)-1,
                                          np.log10(.75),
                                          num_bins)
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdf_err = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr
    bins = sud_integrator.bins

    if bin_space == 'log':
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

    if bin_space == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_err = xs*pdf_err * np.log(10) # d sigma / d log10 C
        print("Critical MC adjusted normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))


    return pdfline, pdfband, cdfline, cdfband


###########################################
# All Emissions
###########################################
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, f_soft, col):
    """Plots the distribution obtained through Monte Carlo for
    jet ECFs using all emissions.
    """
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits        = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['samples']
    theta_crit_weights = sudakov_inverse_transforms['critical']\
                                    [z_cut][beta]['weights']

    c_subs        = sudakov_inverse_transforms['subsequent']\
                                    [z_cut][beta]['samples']
    c_sub_weights = sudakov_inverse_transforms['subsequent']\
                                    [z_cut][beta]['weights']

    z_pres        = sudakov_inverse_transforms['pre-critical']\
                                    [z_cut]['samples']
    z_pre_weights = sudakov_inverse_transforms['pre-critical']\
                                    [z_cut]['weights']

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])
    weights = splitting_functions[z_cut](z_crits, theta_crits)
    weights = weights * theta_crit_weights * c_sub_weights * z_pre_weights

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=f_soft, acc=obs_acc)
    obs = np.maximum(c_crits, c_subs)

    # Weights, binned observables, and area
    if bin_space == 'lin':
        sud_integrator.bins = np.linspace(0, .5, num_bins)
    if bin_space == 'log':
        sud_integrator.bins = np.logspace(np.log10(epsilon)-1, np.log10(.5),
                                          num_bins)
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdf_err = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr
    bins = sud_integrator.bins

    if bin_space == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        print("All MC normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf,
                                      2.*pdf_err, bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral,
                                      integralerr, bins,
                                      label=None, col=col)

    if bin_space == 'log':
        pdf = xs*pdf * np.log(10) # d sigma / d log10 C
        pdf_err = xs*pdf_err * np.log(10) # d sigma / d log10 C

        print("All MC adjusted normalization:",
              np.sum(pdf * (np.log10(bins[1:])-np.log10(bins[:-1]))))

    return pdfline, pdfband, cdfline, cdfband
