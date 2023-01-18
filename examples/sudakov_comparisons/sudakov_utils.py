from __future__ import absolute_import
import dill as pickle
from pathlib import Path

from scipy.misc import derivative

# Local utilities for comparison
from jetmontecarlo.utils.montecarlo_utils import getLinSample
from jetmontecarlo.jets.observables import *
from examples.comparison_plot_utils import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *
from jetmontecarlo.montecarlo.partonshower import *

# Parameters and Filenames
from examples.params import *
from examples.filenames import *


###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]
Z_CUT_PLOT = [F_SOFT * zc for zc in Z_CUT_PLOT]

F_SOFT_PLOT = [.5, 1]
F_SOFT_PLOT_IVS = [.5, 1, 'ivs']
F_SOFT_STR = ['1/2', '1']

save_cdf = False

#f_colors = {.5: OrangeShade3, 1: GreenShade2, 'ivs': PurpleShade2}
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


# ------------------------------------
# Loading Functions:
# ------------------------------------
split_fn_num = get_splitting_function()

radiators = {}
if not(LOAD_MC_EVENTS):
    radiators = get_radiator_functions()


###########################################
# Additional Plot Utils
###########################################
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
    if drawband:
        band = draw_error_band(ax, xs, ys, err, color=col, alpha=.4)
        return line, band
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

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

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

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
                         load=load, save=True, theta_crits=theta_crits,
                         rad_crit_sub=radiators.get('subsequent', None))

    z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=save,
                        theta_crits=theta_crits,
                        rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

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

###########################################
# IVS
###########################################
def plot_mc_ivs(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                emissions=['crit'],
                load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if 'crit' in emissions:
        theta_crits, theta_crit_weights, load = get_theta_crits(
                              z_cut, beta, load=load, save=True,
                              rad_crit=radiators.get('critical', None))

    if 'sub' in emissions:
        c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
                             load=load, save=True, theta_crits=theta_crits,
                             rad_crit_sub=radiators.get('subsequent', None))

    if 'pre' in emissions:
        z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=save,
                            theta_crits=theta_crits,
                            rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = np.zeros_like(z_crits)
    if 'pre' in emissions:
        assert 'crit' in emissions, "Must consider critical emissions to " \
            + "consider pre-critical emissions."
        c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                            z_pre=z_pres, f=f_ivs(theta_crits),
                            acc=OBS_ACC)
    elif 'crit' in emissions:
        c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                            z_pre=np.zeros_like(z_crits),
                            f=f_ivs(theta_crits),
                            acc=OBS_ACC)
    if 'sub' in emissions:
        obs = np.maximum(c_crits, c_subs)
    else:
        obs = c_crits

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

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf, 2.*pdf_err,
                                      sud_integrator.bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral, integralerr,
                                      sud_integrator.bins,
                                      label=None, col=col)

    return pdfline, pdfband, cdfline, cdfband


# DEBUG
# Deprecated code
"""
###########################################
# Subsequent Emissions
###########################################
def plot_mc_sub(axes_pdf, axes_cdf, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])
    if LOAD_INV_SAMPLES:
        if (beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+"...",
                  flush=True)
            c_subs = np.load((beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        def this_cdf_sub(c_sub):
            return rad_sub(c_sub, beta)

        c_subs = samples_from_cdf(this_cdf_sub, NUM_MC_EVENTS, domain=[0.,.5],
                                  verbose=3)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save((beta), c_subs)

    obs = c_subs

    # Weights, binned observables, and area
    sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                      NUM_BINS)
    sud_integrator.hasBins = True
    sud_integrator.setDensity(obs, np.ones(len(obs)), 1.)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdf_err = sud_integrator.densityErr

    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdf_err, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)


###########################################
# Critical and Subsequent Emissions
###########################################
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if LOAD_INV_SAMPLES:
        if sudakov_crit_sample_file(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...", flush=True)
            theta_crits = np.load(sudakov_crit_sample_file(z_cut, beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making critical samples with z_c="+str(z_cut)+"...", flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS,
                                       domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(sudakov_crit_sample_file(z_cut, beta), theta_crits)

    if LOAD_INV_SAMPLES:
        if sudakov_crit_sub_sample_file(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            c_subs = np.load(sudakov_crit_sub_sample_file(z_cut, beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()


    if not LOAD_INV_SAMPLES:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_sub(c_sub):
                return cdf_sub_conditional(c, theta)

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-50
            else:
                c_sub = samples_from_cdf(this_cdf_sub, 1,
                                     domain=[0.,theta**beta/2.],
                                     verbose=3)[0]
            c_subs.append(c_sub)
            if (i+1)%(len(theta_crits)/10)==0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(sudakov_crit_sub_sample_file(z_cut, beta), c_subs)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=0., f=F_SOFT, acc=OBS_ACC)
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

    plot_mc_pdf(axes_pdf, pdf, pdf_err, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

###########################################
# Pre + Critical Emissions
###########################################
def plot_mc_pre_and_crit(axes_pdf, axes_cdf, z_cut, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=save,
                        theta_crits=theta_crits,
                        rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = c_crits

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

    plot_mc_pdf(axes_pdf, pdf, pdf_err, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

"""
