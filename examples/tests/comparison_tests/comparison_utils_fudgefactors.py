# Local MC integration imports
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.numerics.observables import *
from jetmontecarlo.numerics.weights import *
from jetmontecarlo.numerics.radiators.samplers import *

# Local parton shower imports
from jetmontecarlo.utils.partonshower_utils import *
from examples.tests.partonshower_tests.one_emission.one_em_groomed_test_utils import *
from examples.tests.partonshower_tests.pre_and_crit.precrit_groomed_test_utils import *
from examples.tests.partonshower_tests.pre_and_crit.precrit_se_groomed_test_utils import *

# Local generic utility imports
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *

# Local analytics
from jetmontecarlo.analytics.sudakov_factors.fixedcoupling import *
from jetmontecarlo.analytics.radiators.running_coupling import *

# Older version of comparison code.
# Includes veto-based showers and fudge factors for numerical integration.
# This was saved on 6-24, just in case.

# Definitions and Parameters
# Physics inputs
FIXED_COUPLING = True

# Running coupling parameters
ACC = 'MLL'
SHOWER_CUTOFF = MU_NP

# Fixed coupling parameters
if FIXED_COUPLING:
    ACC = 'LL'
    SHOWER_CUTOFF = 1e-10

# Jet and grooming parameters
Z_CUT = .1
BETA = 2
F_SOFT = 1.
JET_TYPE = 'quark'

# Monte Carlo parameters
NUM_SHOWER_EVENTS = int(1e3)
SAVE_SHOWER_EVENTS = False
LOAD_SHOWER_EVENTS = False

NUM_MC_EVENTS = int(1e5)
SAVE_MC_EVENTS = False
LOAD_MC_EVENTS = False
EPSILONS = [1e-10, 1e-12, 1e-15]

# Plotting parameters
NUM_BINS = 75
BIN_SPACE = 'log'

if BIN_SPACE == 'lin':
    ylim_1 = (0, 40.)
    ylim_2 = (0, 40.)
    xlim = (0, .5)
if BIN_SPACE == 'log':
    ylim_1 = (0, .12)
    ylim_2 = (0, .25)
    xlim = (1e-8, .5)

# Choosing which emissions to plot
COMPARE_CRIT = False
COMPARE_PRE_AND_CRIT = False
COMPARE_CRIT_AND_SUB = True
COMPARE_ALL = False

"""Verifying valid bin_space."""
assert(BIN_SPACE in ['lin', 'log']), \
"'bin_space' must be 'lin' or 'log', but is" + BIN_SPACE

###########################################
# Generating Events:
###########################################
# ------------------------------------
# List of jets (parton shower)
# ------------------------------------
# Preparing a list of ungroomed jets for general use
JET_LIST = gen_jets(NUM_SHOWER_EVENTS, beta=BETA, radius=1.,
                    jet_type=JET_TYPE, acc=ACC,
                    cutoff=SHOWER_CUTOFF)
if LOAD_SHOWER_EVENTS:
    pass
    if SAVE_SHOWER_EVENTS:
        pass
else:
    pass
    # JET_LIST = gen_jets(NUM_SHOWER_EVENTS, beta=BETA, radius=1.,
    #                     jet_type=JET_TYPE, acc=ACC,
    #                     cutoff=SHOWER_CUTOFF)

# Angular ordering
# JET_LIST = [angular_ordered(jet) for jet in JET_LIST]

# ------------------------------------
# Integrator and Samplers (MC integration)
# ------------------------------------
# Setting up integrator
INTEGRATOR = integrator()
# Integrating to find a CDF
INTEGRATOR.setLastBinBndCondition([1., 'minus'])

# Setting up samplers
CRIT_SAMPLERS = []
PRE_SAMPLERS = []
SUB_SAMPLERS = []

print("Generating events for Monte Carlo integration:")
for _, epsilon in enumerate(tqdm(EPSILONS)):
    print("  Generating events with cutoff epsilon={:.0e}".format(epsilon))

    # Critical sampler
    crit_sampler_i = criticalSampler(BIN_SPACE, zc=Z_CUT, epsilon=epsilon)
    crit_sampler_i.generateSamples(NUM_MC_EVENTS)
    CRIT_SAMPLERS.append(crit_sampler_i)

    # Pre-critical sampler
    pre_sampler_i = precriticalSampler(BIN_SPACE, zc=Z_CUT, epsilon=epsilon)
    if COMPARE_PRE_AND_CRIT or COMPARE_ALL:
        pre_sampler_i.generateSamples(NUM_MC_EVENTS)
    PRE_SAMPLERS.append(pre_sampler_i)

    # Subsequent sampler
    sub_sampler_i = ungroomedSampler(BIN_SPACE, epsilon=epsilon)
    if COMPARE_CRIT_AND_SUB or COMPARE_ALL:
        sub_sampler_i.generateSamples(NUM_MC_EVENTS)
    SUB_SAMPLERS.append(sub_sampler_i)

# ------------------------------------
# Parton Shower
# ------------------------------------
def get_ps_ECFs(jet_list, emission_type, few_emissions=False):
    """Generates a set of critical emissions
    via a veto method parton shower.
    Returns a set of angularities.
    """
    ecfs = getECFs_groomed(jet_list, z_cut=Z_CUT, beta=BETA, f=F_SOFT,
                           acc=ACC, emission_type=emission_type,
                           few_emissions=few_emissions)
    return ecfs

###########################################
# Critical Emission Only
###########################################
# ------------------------------------
# Analytic
# ------------------------------------
def plot_crit_analytic(axespdf, axescdf, label='Analytic'):
    """Plot the critical emission analytic result."""
    if BIN_SPACE == 'lin':
        bins = np.linspace(0, .5, 100)
        xs = (bins[:-1] + bins[1:])/2.
    if BIN_SPACE == 'log':
        bins = np.logspace(-13, np.log10(.5), 100)
        xs = np.sqrt(bins[:-1]*bins[1:])

    # Finding cdf and pdf
    cdf = critSudakov_fc_LL(xs, Z_CUT, BETA,
                            jet_type=JET_TYPE)
    _, pdf = histDerivative(cdf, bins, giveHist=True,
                            binInput=BIN_SPACE)
    if BIN_SPACE == 'log':
        pdf = xs * pdf

    # Plotting
    col = compcolors[(-1, 'light')]
    axespdf[0].plot(xs, pdf, **style_dashed,
                    color=col, label=label)
    axescdf[0].plot(xs, cdf, **style_dashed,
                    color=col, label=label)
    if len(axespdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_dashed, color=col,
                        zorder=.5)
    if len(axescdf) > 1:
        axespdf[1].plot(xs, np.ones(len(xs)),
                        **style_dashed, color=col,
                        zorder=.5)

# ------------------------------------
# Monte Carlo Integration
# ------------------------------------
def get_mc_crit(crit_sampler):
    """Generates a set of critical emissions.
    Returns a pdf, error, cdf, error, and bins to
    find the associated groomed ECF values.
    """
    # Sampling
    samples = crit_sampler.getSamples()
    z = samples[:, 0]
    theta = samples[:, 1]
    obs = C_groomed(z, theta, Z_CUT, BETA)

    # Weights, binned observables, and area
    INTEGRATOR.setBins(NUM_BINS, obs, BIN_SPACE)
    weights = criticalEmissionWeight(z, theta, Z_CUT, JET_TYPE,
                                     fixedcoupling=FIXED_COUPLING)
    jacs = crit_sampler.jacobians
    area = crit_sampler.area

    INTEGRATOR.setDensity(obs, weights * jacs, area)
    INTEGRATOR.integrate()

    pdf = INTEGRATOR.density
    pdferr = INTEGRATOR.densityErr
    integral = INTEGRATOR.integral
    integralerr = INTEGRATOR.integralErr

    return pdf, pdferr, integral, integralerr, INTEGRATOR.bins

# ------------------------------------
# Parton Shower
# ------------------------------------
def get_ps_crit_veto():
    """Generates a set of critical emissions
    via a veto method parton shower.
    Returns a set of angularities.
    """
    jet_list = gen_jets_groomed_crit(NUM_SHOWER_EVENTS,
                                     z_cut=Z_CUT, beta=BETA, f=F_SOFT,
                                     radius=1.,
                                     jet_type=JET_TYPE,
                                     acc=ACC, veto=True)
    angs = getangs(jet_list, beta=BETA, acc=ACC,
                   emission_type='crit')
    return angs

###########################################
# Pre-critical and Critical Emissions
###########################################
# ------------------------------------
# Monte Carlo Integration
# ------------------------------------
def get_mc_pre_and_crit(crit_sampler, pre_sampler,
                        integrate_out_pre, use_pre_fudge_factor=False):
    """Generates a set of critical and pre-critical emissions.
    Returns a pdf, error, cdf, error, and bins to
    find the associated groomed ECF values.
    """
    # Critical Sampling
    samples = crit_sampler.getSamples()
    z_crit = samples[:, 0]
    theta_crit = samples[:, 1]

    # Pre-critical Sampling
    z_pre = pre_sampler.getSamples()

    obs = C_groomed(z_crit, theta_crit, Z_CUT, BETA,
                    z_pre=z_pre, f=F_SOFT, acc='LL')

    # Adding in a fudge factor for precritical emissions
    pre_fudge_factor = 1.
    if integrate_out_pre:
        # Integrating out pre-critical emissions by using
        # the critical observable only:
        obs = C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                        z_pre=0., f=F_SOFT, acc='LL')
    if BIN_SPACE == 'log' and use_pre_fudge_factor:
        pre_fudge_factor = 1/(1.
                          -
                          np.exp(-2. * CR(JET_TYPE) * alpha_fixed/np.pi
                                 * np.log(R0/theta_crit)
                                 * np.log(Z_CUT/pre_sampler.epsilon)
                                 ))

        print(); print(); print();
        print("epsilon: " + str(pre_sampler.epsilon))
        print("Pre-crit fudge factors: " + str(pre_fudge_factor))
        print(); print(); print();

    # print("Observables: " + str(obs))
    # print("Maximum observable: " + str(max(obs)))
    # print("Minimum observable: " + str(min(obs)))
    # if min(obs) == 0:
    #     print("Number of zeros: " + str(len([o for o in obs if o == 0])))

    # Weights, binned observables, and area
    INTEGRATOR.setBins(NUM_BINS, obs, BIN_SPACE)

    weights = (
        criticalEmissionWeight(z_crit, theta_crit,
                               Z_CUT, JET_TYPE,
                               fixedcoupling=FIXED_COUPLING)
        *
        precriticalEmissionWeight(z_pre, theta_crit,
                                  Z_CUT, JET_TYPE,
                                  fixedcoupling=FIXED_COUPLING)
        ) * pre_fudge_factor

    jacs = (np.array(crit_sampler.jacobians)
            * np.array(pre_sampler.jacobians))
    area = (np.array(crit_sampler.area)
            * np.array(pre_sampler.area))

    INTEGRATOR.setDensity(obs, weights * jacs, area)
    INTEGRATOR.integrate()

    pdf = INTEGRATOR.density
    pdferr = INTEGRATOR.densityErr
    integral = INTEGRATOR.integral
    integralerr = INTEGRATOR.integralErr

    return pdf, pdferr, integral, integralerr, INTEGRATOR.bins

# ------------------------------------
# Parton Shower
# ------------------------------------
def get_ps_pre_and_crit_veto_se():
    """Generates a set of critical emissions
    with a single pre-critical each emission
    through a parton shower.
    Returns a set of angularities.
    """
    jet_list = gen_jets_groomed_precrit_se(NUM_SHOWER_EVENTS,
                                           z_cut=Z_CUT, beta=BETA, f=F_SOFT,
                                           radius=1.,
                                           jet_type=JET_TYPE,
                                           acc=ACC)
    angs = getangs(jet_list, beta=BETA, acc=ACC,
                   emission_type=None)
    return angs

def get_ps_pre_and_crit_veto():
    """Generates a set of critical emissions
    via a veto method parton shower.
    Takes multiple precritical emissions into
    account.
    Returns a set of angularities.
    """
    jet_list = gen_jets_groomed_precrit(NUM_SHOWER_EVENTS,
                                        z_cut=Z_CUT, beta=BETA, f=F_SOFT,
                                        radius=1.,
                                        jet_type=JET_TYPE,
                                        acc=ACC)
    angs = getangs(jet_list, beta=BETA, acc=ACC,
                   emission_type=None)
    return angs

###########################################
# Critical and Subsequent Emissions
###########################################
# ------------------------------------
# Monte Carlo Integration
# ------------------------------------
def get_mc_crit_and_sub(crit_sampler, sub_sampler,
                        integrate_out_sub, use_sub_fudge_factor=False):
    """Generates a set of critical and pre-critical emissions.
    Returns a pdf, error, cdf, error, and bins to
    find the associated groomed ECF values.
    """
    # Critical Sampling
    samples = crit_sampler.getSamples()
    z_crit = samples[:, 0]
    theta_crit = samples[:, 1]

    # Subsequent Sampling
    samples = sub_sampler.getSamples()
    #print("samples: " + str(samples))
    c_sub = samples[:, 0] * theta_crit**BETA
    # Since z_sub and c_sub have the same range of
    # integration,we can pretend that we are instead
    # sampling over c_sub here

    obs_crit = C_groomed(z_crit, theta_crit, Z_CUT, BETA,
                         z_pre=0., f=F_SOFT, acc='LL')
    obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,
                               z_pre=0., f=F_SOFT, acc='LL'), c_sub)

    smaller = [index for index, o_c in enumerate(obs_crit)
                                    if o_c < obs[index]]

    print('Percentage disagreeing: ' + str(len(smaller)/NUM_MC_EVENTS))

    # Adding in a fudge factor for subsequent emissions
    sub_fudge_factor = 1.
    if integrate_out_sub:
        # Integrating out subsequent emissions by using
        # the critical observable only:
        obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,
                                   z_pre=0., f=F_SOFT, acc='LL'), 0.)
    if BIN_SPACE == 'log' and use_sub_fudge_factor:
        sub_fudge_factor = 1./(1.-
            np.exp(-subRadAnalytic_fc_LL(sub_sampler.epsilon,
                                 BETA, jet_type=JET_TYPE))
            )

        print(); print(); print();
        print("epsilon: " + str(sub_sampler.epsilon))
        print("Sub fudge factors: " + str(sub_fudge_factor))
        print(); print(); print();

    # print("Observables: " + str(obs))
    # print("Maximum observable: " + str(max(obs)))
    # print("Minimum observable: " + str(min(obs)))
    # if min(obs) == 0:
    #     print("Number of zeros: " + str(len([o for o in obs if o == 0])))

    # Weights, binned observables, and area
    INTEGRATOR.setBins(NUM_BINS, obs, BIN_SPACE)

    # only fc for now
    print("Only generating fixed coupling crit + sub.")
    weights = (
        criticalEmissionWeight(z_crit, theta_crit,
                               Z_CUT, JET_TYPE,
                               fixedcoupling=FIXED_COUPLING)
        *
        subPDFAnalytic_fc_LL(c_sub/theta_crit**BETA, BETA,
                             jet_type=JET_TYPE)
        *
        sub_fudge_factor
        )

    weightscrit = (
        criticalEmissionWeight(z_crit, theta_crit,
                               Z_CUT, JET_TYPE,
                               fixedcoupling=FIXED_COUPLING)
        )

    print("weight ratio: " + str(weights/weightscrit))

    jacs = (np.array(crit_sampler.jacobians)
            * np.array(sub_sampler.jacobians))
    area = (np.array(crit_sampler.area)
            * np.array(sub_sampler.area))

    INTEGRATOR.setDensity(obs, weights * jacs, area)
    INTEGRATOR.integrate()

    pdf = INTEGRATOR.density
    pdferr = INTEGRATOR.densityErr
    integral = INTEGRATOR.integral
    integralerr = INTEGRATOR.integralErr

    return pdf, pdferr, integral, integralerr, INTEGRATOR.bins

# ------------------------------------
# Parton Shower
# ------------------------------------
"""No veto-based parton shower here -- I got lazy!
More precisely, since the veto parton shower agrees
so well with my usual parton shower method, I decided
not to code this until it became an important check.
If you're reading this, it never did!
"""

###########################################
# All Emissions
###########################################
# ------------------------------------
# Monte Carlo Integration
# ------------------------------------
def get_mc_all_emissions(crit_sampler, pre_sampler, sub_sampler,
                         integrate_out_pre, integrate_out_sub,
                         use_pre_fudge_factor=False,
                         use_sub_fudge_factor=False):
    """Generates a set of critical and pre-critical emissions.
    Returns a pdf, error, cdf, error, and bins to
    find the associated groomed ECF values.
    """
    # Critical Sampling
    samples = crit_sampler.getSamples()
    z_crit = samples[:, 0]
    theta_crit = samples[:, 1]

    # Subsequent Sampling
    samples = sub_sampler.getSamples()
    c_sub = samples[:, 0] * theta_crit**BETA
    # Since z_sub and c_sub have the same range of
    # integration,we can pretend that we are instead
    # sampling over c_sub here

    # Pre-critical Sampling
    z_pre = pre_sampler.getSamples()

    obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                               z_pre=z_pre, f=F_SOFT, acc='LL'), c_sub)

    print("Observables: " + str(obs))
    print("Maximum observable: " + str(max(obs)))
    print("Minimum observable: " + str(min(obs)))
    if min(obs) == 0:
        print("Number of zeros: " + str(len([o for o in obs if o == 0])))

    # Adding in a fudge factor for precritical emissions
    pre_fudge_factor = 1.
    if integrate_out_pre:
        # Integrating out pre-critical emissions by using
        # the critical observable only:
        obs = C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                        z_pre=0., f=F_SOFT, acc='LL')
    if BIN_SPACE == 'log' and use_pre_fudge_factor:
        pre_fudge_factor = 1/(1.
                          -
                          np.exp(-2. * CR(JET_TYPE) * alpha_fixed/np.pi
                                 * np.log(R0/theta_crit)
                                 * np.log(Z_CUT/pre_sampler.epsilon)
                                 ))

        print(); print(); print();
        print("epsilon: " + str(pre_sampler.epsilon))
        print("Pre-crit fudge factors: " + str(pre_fudge_factor))
        print(); print(); print();

    # Adding in a fudge factor for subsequent emissions
    sub_fudge_factor = 1.
    if integrate_out_sub:
        # Integrating out subsequent emissions by using
        # the critical observable only:
        obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,
                                   z_pre=0., f=F_SOFT, acc='LL'), 0.)
    if BIN_SPACE == 'log' and use_sub_fudge_factor:
        sub_fudge_factor = 1./(1.-
            np.exp(-subRadAnalytic_fc_LL(sub_sampler.epsilon,
                                 BETA, jet_type=JET_TYPE))
            )

        print(); print(); print();
        print("epsilon: " + str(sub_sampler.epsilon))
        print("Sub fudge factors: " + str(sub_fudge_factor))
        print(); print(); print();

    # Integrating out different combinations of emissions:
    if integrate_out_pre and integrate_out_sub:
        # Integrating out both pre-critical and subsequent emissions
        obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                                   z_pre=0., f=F_SOFT, acc='LL'), 0.)
    elif integrate_out_pre:
        # Integrating out only pre-critical emissions
        obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                                   z_pre=0., f=F_SOFT, acc='LL'), c_sub)
    elif integrate_out_sub:
        # Integrating out only subsequent emissions
        obs = np.maximum(C_groomed(z_crit, theta_crit, Z_CUT, BETA,\
                                   z_pre=z_pre, f=F_SOFT, acc='LL'), 0.)

    print("Observables: " + str(obs))
    print("Maximum observable: " + str(max(obs)))
    print("Minimum observable: " + str(min(obs)))
    if min(obs) == 0:
        print("Number of zeros: " + str(len([o for o in obs if o == 0])))

    # Weights, binned observables, and area
    INTEGRATOR.setBins(NUM_BINS, obs, BIN_SPACE)

    # only fc for now
    print("Only generating fixed coupling pre + crit + sub.")
    weights = (
        criticalEmissionWeight(z_crit, theta_crit,
                               Z_CUT, JET_TYPE,
                               fixedcoupling=FIXED_COUPLING)
        *
        subPDFAnalytic_fc_LL(c_sub/theta_crit**BETA, BETA,
                             jet_type=JET_TYPE)
        *
        precriticalEmissionWeight(z_pre, theta_crit,
                                  Z_CUT, JET_TYPE,
                                  fixedcoupling=FIXED_COUPLING)
        * pre_fudge_factor * sub_fudge_factor
        )

    jacs = (np.array(crit_sampler.jacobians)
            * np.array(pre_sampler.jacobians)
            * np.array(sub_sampler.jacobians))
    area = (np.array(crit_sampler.area)
            * np.array(pre_sampler.area)
            * np.array(sub_sampler.area))

    INTEGRATOR.setDensity(obs, weights * jacs, area)
    INTEGRATOR.integrate()

    pdf = INTEGRATOR.density
    pdferr = INTEGRATOR.densityErr
    integral = INTEGRATOR.integral
    integralerr = INTEGRATOR.integralErr

    return pdf, pdferr, integral, integralerr, INTEGRATOR.bins

# ------------------------------------
# Parton Shower
# ------------------------------------
"""No veto-based parton shower here -- I got lazy!
More precisely, since the veto parton shower agrees
so well with my usual parton shower method, I decided
not to code this until it became an important check.
If you're reading this, it never did!
"""


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
                  +r"$\frac{{\rm d}~\sigma}{{\rm d}~\ln \lambda_{(2)}}$")

    # Fig and axes for plotting pdf
    figpdf, axespdf = aestheticfig(xlabel=r"$\lambda_{(2)}$",
                                   ylabel=ylabel,
                                   ylim=ylim_1,
                                   xlim=xlim,
                                   title="Groomed ECF PDF ("+title_info+")",
                                   showdate=True,
                                   ratio_plot=ratio_plot)
    axespdf[0].set_ylabel(ylabel, labelpad=25, rotation=0,
                          fontsize=18)
    if len(axespdf) > 1:
        axespdf[1].set_ylabel('Ratio', labelpad=0, rotation=0)

    # Fig and axes for plotting cdf
    figcdf, axescdf = aestheticfig(xlabel=r"$\lambda_{(2)}$",
                                   ylabel=r"$\Sigma(\lambda_{(2)})$",
                                   ylim=(0, 1.1),
                                   xlim=xlim,
                                   title="Groomed ECF CDF ("+title_info+")",
                                   showdate=True,
                                   ratio_plot=ratio_plot)
    axescdf[0].set_ylabel(r"$\Sigma(\lambda_{(2)})$", labelpad=23, rotation=0,
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

    return figpdf, axespdf, figcdf, axescdf

# ------------------------------------
# Plotting Monte Carlo Integrals
# ------------------------------------
def plot_mc_pdf(axespdf, pdf, pdferr, bins,
                label='M.C. Integration'):
    """Plots a set of pdf values."""
    col = compcolors[(0, 'dark')]

    if BIN_SPACE == 'lin':
        xs = (bins[:-1] + bins[1:])/2.
        xerr = (bins[1:] - bins[:-1])
    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        xerr = (xs - bins[:-1], bins[1:]-xs)
        pdf = xs * pdf
        pdferr = xs * pdferr

    axespdf[0].errorbar(xs, pdf, yerr=pdferr,
                        xerr=xerr, **modstyle,
                        color=col, ecolor=col,
                        label=label)

def plot_mc_cdf(axescdf, cdf, cdferr, bins,
                label='M.C. Integration'):
    """Plots a set of cdf values."""
    col = compcolors[(0, 'dark')]

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
                        colnum=1):
    """Plots the pdf and cdf associated with the
    set of correlators (vals) on axespdf and axescdf.
    """
    if BIN_SPACE == 'lin':
        bins = np.linspace(0, 1., NUM_BINS)
    else:
        bins = np.logspace(-8, 0., NUM_BINS)
        bins = np.append(1e-100, bins)

    int_bins_temp = INTEGRATOR.bins
    INTEGRATOR.bins = bins

    num_in_bin, _ = np.histogram(vals, bins)
    pdf, _ = np.histogram(vals, bins, density=True)
    pdf_error = pdf / np.sqrt(num_in_bin)

    INTEGRATOR.densityErr = pdf_error
    INTEGRATOR.density = pdf
    INTEGRATOR.hasMCDensity = True

    INTEGRATOR.integrate()
    integral = INTEGRATOR.integral
    interr = INTEGRATOR.integralErr

    col = compcolors[(colnum, 'dark')]

    # Analytic cdf and pdf:
    xs = (bins[:-1] + bins[1:])/2.

    # Numerically obtaining pdf:
    cdf_an = critSudakov_fc_LL(xs, Z_CUT, BETA, f=F_SOFT,
                               jet_type=JET_TYPE)
    _, pdf_an = histDerivative(cdf_an, bins, giveHist=True,
                               binInput=BIN_SPACE)
    pdf_an = np.array(pdf_an.tolist(), dtype=float)

    # ------------------
    # PDF plots:
    # ------------------
    if BIN_SPACE == 'log':
        pdf = xs*pdf
        pdf_error = xs*pdf_error
        pdf_an = xs*pdf_an
    axespdf[0].errorbar(xs, pdf,
                        yerr=pdf_error,
                        xerr=(bins[1:] - bins[:-1])/2.,
                        **modstyle, color=col,
                        label=label)

    if len(axespdf) > 1:
        axespdf[1].errorbar(xs, pdf/pdf_an,
                            yerr=pdf_error/pdf_an,
                            xerr=(bins[1:] - bins[:-1])/2.,
                            **modstyle, color=col)


    # ------------------
    # CDF plots:
    # ------------------
    xs = bins[:-1]
    cdf_an = critSudakov_fc_LL(xs, Z_CUT, BETA, f=F_SOFT,
                               jet_type=JET_TYPE)
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

    INTEGRATOR.bins = int_bins_temp


###########################################
# Comparisons:
###########################################
# ------------------------------------
# Critical Emission Only:
# ------------------------------------
def compare_crit(crit_sampler):
    """Compares the Monte Carlo integration, parton shower,
    and analytic results for critical emissions only.
    """
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit', ratio_plot=False)

    # Generating data
    pdf, pdferr, cdf, cdferr, bins = get_mc_crit(crit_sampler)
    shower_correlations = get_ps_ECFs(JET_LIST, 'crit')
    # shower_correlations_veto = get_ps_crit_veto()

    # Plotting
    plot_crit_analytic(axes_pdf, axes_cdf)
    plot_mc_pdf(axes_pdf, pdf, pdferr, bins)
    plot_mc_cdf(axes_cdf, cdf, cdferr, bins)
    plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                        label='Parton Shower', colnum=1)
    # plot_shower_pdf_cdf(shower_correlations_veto, axes_pdf, axes_cdf,
    #                     label='Parton Shower (veto)', colnum=2)

    # Saving plots
    axes_pdf[0].legend()
    legend_yerr(axes_cdf[0])

    extra_label = ''
    if BIN_SPACE == 'log':
        extra_label = '_{:.0e}cutoff'.format(crit_sampler.epsilon)

    fig_pdf.savefig(JET_TYPE+'_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_crit_'+BIN_SPACE+'_cdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord.pdf',
                    format='pdf')

# ------------------------------------
# Pre + Critical Emission Only:
# ------------------------------------
def compare_pre_and_crit(crit_sampler, pre_sampler,
                         integrate_out_pre,
                         use_pre_fudge_factor=False):
    """Compares the Monte Carlo integration, parton shower,
    and analytic results for precritical + critical emissions.
    """
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('precrit + crit', ratio_plot=False)

    # Generating data
    pdf, pdferr, cdf, cdferr, bins = get_mc_pre_and_crit(crit_sampler,
                                                         pre_sampler,
                                                         integrate_out_pre,
                                                         use_pre_fudge_factor)
    shower_correlations_se = get_ps_ECFs(JET_LIST, 'precrit',
                                         few_emissions=True)
    shower_correlations = get_ps_ECFs(JET_LIST, 'precrit')
    # shower_correlations_veto_se = get_ps_pre_and_crit_veto_se()
    # shower_correlations_veto = get_ps_pre_and_crit_veto()

    # Plotting
    plot_crit_analytic(axes_pdf, axes_cdf, label='Analytic, crit only')
    plot_mc_pdf(axes_pdf, pdf, pdferr, bins,
                label='Monte Carlo, 1 precrit')
    plot_mc_cdf(axes_cdf, cdf, cdferr, bins,
                label='Monte Carlo, 1 precrit')
    plot_shower_pdf_cdf(shower_correlations_se, axes_pdf, axes_cdf,
                        label='Parton Shower, 1 precrit',
                        colnum=1)
    plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                        label='Parton Shower, multiple precrit',
                        colnum=2)
    # plot_shower_pdf_cdf(shower_correlations_veto_se, axes_pdf, axes_cdf,
    #                     label='Parton Shower, 1 precrit (veto)',
    #                     colnum=3)
    # plot_shower_pdf_cdf(shower_correlations_veto, axes_pdf, axes_cdf,
    #                     label='Parton Shower, multiple precrit (veto)',
    #                     colnum=4)

    # Saving plots
    axes_pdf[0].legend()
    axes_pdf[0].set_ylim((0, .2))
    legend_yerr(axes_cdf[0])

    extra_label = ''
    if BIN_SPACE == 'log':
        assert(crit_sampler.epsilon == pre_sampler.epsilon),\
            "Samplers must have the same cutoff"
        extra_label = '_{:.0e}cutoff'.format(crit_sampler.epsilon)
    if integrate_out_pre:
        extra_label = extra_label + '_intout_pre'
    if use_pre_fudge_factor:
        extra_label = extra_label + '_prefudged'

    fig_pdf.savefig(JET_TYPE+'_pre_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_pre_crit_'+BIN_SPACE+'_cdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord.pdf',
                    format='pdf')


# ------------------------------------
# Critical + Subsequent Emission Only:
# ------------------------------------
def compare_crit_and_sub(crit_sampler, sub_sampler,
                         integrate_out_sub,
                         use_sub_fudge_factor=False):
    """Compares the Monte Carlo integration, parton shower,
    and analytic results for critical + subsequent emissions.
    """
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit + sub', ratio_plot=False)

    # Generating data
    pdf, pdferr, cdf, cdferr, bins = get_mc_crit_and_sub(crit_sampler,
                                                         sub_sampler,
                                                         integrate_out_sub,
                                                         use_sub_fudge_factor)
    shower_correlations_se = get_ps_ECFs(JET_LIST, 'critsub',
                                         few_emissions=True)
    # shower_correlations = get_ps_ECFs(JET_LIST, 'critsub')
    # No veto generated correlations

    # Plotting
    plot_crit_analytic(axes_pdf, axes_cdf, label='Analytic, crit only')
    plot_mc_pdf(axes_pdf, pdf, pdferr, bins,
                label='Monte Carlo, 1 sub')
    plot_mc_cdf(axes_cdf, cdf, cdferr, bins,
                label='Monte Carlo, 1 sub')
    plot_shower_pdf_cdf(shower_correlations_se, axes_pdf, axes_cdf,
                        label='Parton Shower, 1 sub',
                        colnum=1)
    # plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
    #                     label='Parton Shower, multiple sub',
    #                     colnum=2)

    # Saving plots
    axes_pdf[0].legend()
    axes_pdf[0].set_ylim((0, .2))
    legend_yerr(axes_cdf[0])

    extra_label = ''
    if BIN_SPACE == 'log':
        assert(crit_sampler.epsilon == sub_sampler.epsilon),\
            "Samplers must have the same cutoff"
        extra_label = '_{:.0e}cutoff'.format(crit_sampler.epsilon)
    if integrate_out_sub:
        extra_label = extra_label + '_intout_sub'
    if use_sub_fudge_factor:
        extra_label = extra_label + '_subfudged'

    fig_pdf.savefig(JET_TYPE+'_crit_sub_'+BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord_fixsub.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_crit_sub_'+BIN_SPACE+'_cdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, len(crit_sampler.samples))
                    +str(extra_label)
                    +'_noangord_fixsub.pdf',
                    format='pdf')

# ------------------------------------
# All Emissions:
# ------------------------------------
def compare_all(crit_sampler, pre_sampler, sub_sampler,
                integrate_out_pre, integrate_out_sub,
                use_pre_fudge_factor=False, use_sub_fudge_factor=False):
    """Compares the Monte Carlo integration, parton shower,
    and analytic results for precritical + critical + subsequent
    emissions.
    """
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('pre + crit + sub', ratio_plot=False)

    # Generating data
    pdf, pdferr, cdf, cdferr, bins = get_mc_all_emissions(crit_sampler,
                                                          pre_sampler,
                                                          sub_sampler,
                                                          integrate_out_pre,
                                                          integrate_out_sub,
                                                          use_pre_fudge_factor,
                                                          use_sub_fudge_factor)
    shower_correlations_se = get_ps_ECFs(JET_LIST, 'precritsub',
                                         few_emissions=True)
    shower_correlations = get_ps_ECFs(JET_LIST, 'precritsub')
    # No veto showers

    # Plotting
    plot_crit_analytic(axes_pdf, axes_cdf, label='Analytic, crit only')
    plot_mc_pdf(axes_pdf, pdf, pdferr, bins,
                label='Monte Carlo, 1 pre, 1 sub')
    plot_mc_cdf(axes_cdf, cdf, cdferr, bins,
                label='Monte Carlo, 1 pre, 1 sub')
    plot_shower_pdf_cdf(shower_correlations_se, axes_pdf, axes_cdf,
                        label='Parton Shower, 1 pre, 1 sub',
                        colnum=1)
    plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                        label='Parton Shower, multiple pre/sub',
                        colnum=2)

    # Saving plots
    axes_pdf[0].legend()
    axes_pdf[0].set_ylim((0, .2))
    legend_yerr(axes_cdf[0])

    extra_label = None
    if BIN_SPACE == 'log':
        assert(crit_sampler.epsilon == sub_sampler.epsilon
               == pre_sampler.epsilon),\
            "Samplers must have the same cutoff"
        extra_label = '_{:.0e}cutoff'.format(crit_sampler.epsilon)
    if integrate_out_pre and integrate_out_sub:
        extra_label = extra_label + '_intout_presub'
    elif integrate_out_pre:
        extra_label = extra_label + '_intout_pre'
    elif integrate_out_sub:
        extra_label = extra_label + '_intout_sub'
    if use_pre_fudge_factor and use_sub_fudge_factor:
        extra_label = extra_label + '_presubfudged'
    elif use_pre_fudge_factor:
        extra_label = extra_label + '_prefudged'
    elif use_sub_fudge_factor:
        extra_label = extra_label + '_subfudged'

    fig_pdf.savefig(JET_TYPE+'_all_em_'+BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'_noangord_fixsub.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_all_em_'+BIN_SPACE+'_cdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'_noangord_fixsub.pdf',
                    format='pdf')


if __name__ == '__main__':
    for ieps in range(len(EPSILONS)):
        # For each value of epsilon we want to use as an integration cutoff:
        if COMPARE_CRIT:
            compare_crit(CRIT_SAMPLERS[ieps])
        if COMPARE_PRE_AND_CRIT:
            # compare_pre_and_crit(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps],
            #                      integrate_out_pre=True,
            #                      use_pre_fudge_factor=True)
            # compare_pre_and_crit(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps],
            #                      integrate_out_pre=True,
            #                      use_pre_fudge_factor=False)
            compare_pre_and_crit(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps],
                                 integrate_out_pre=False,
                                 use_pre_fudge_factor=False)
            compare_pre_and_crit(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps],
                                 integrate_out_pre=False,
                                 use_pre_fudge_factor=True)
        if COMPARE_CRIT_AND_SUB:
            compare_crit_and_sub(CRIT_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
                                 integrate_out_sub=True,
                                 use_sub_fudge_factor=True)
            compare_crit_and_sub(CRIT_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
                                 integrate_out_sub=True,
                                 use_sub_fudge_factor=False)
            compare_crit_and_sub(CRIT_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
                                 integrate_out_sub=False)
        if COMPARE_ALL:
            # compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
            #             integrate_out_pre=True,
            #             use_pre_fudge_factor=True,
            #             integrate_out_sub=True,
            #             use_sub_fudge_factor=True)
            # compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
            #             integrate_out_pre=True,
            #             use_pre_fudge_factor=False,
            #             integrate_out_sub=True,
            #             use_sub_fudge_factor=True)
            # compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
            #             integrate_out_pre=True,
            #             use_pre_fudge_factor=True,
            #             integrate_out_sub=True,
            #             use_sub_fudge_factor=False)
            # compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
            #             integrate_out_pre=True,
            #             use_pre_fudge_factor=True,
            #             integrate_out_sub=False)
            # compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
            #             integrate_out_pre=False,
            #             integrate_out_sub=True,
            #             use_sub_fudge_factor=True)
            compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
                        integrate_out_pre=False,
                        use_pre_fudge_factor=True,
                        integrate_out_sub=False,
                        use_sub_fudge_factor=True)
            compare_all(CRIT_SAMPLERS[ieps], PRE_SAMPLERS[ieps], SUB_SAMPLERS[ieps],
                        integrate_out_pre=False,
                        use_pre_fudge_factor=False,
                        integrate_out_sub=False,
                        use_sub_fudge_factor=False)
