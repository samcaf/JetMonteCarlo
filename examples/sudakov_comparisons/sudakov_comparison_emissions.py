# Parameters and plotting utilities
from examples.params import ALL_MONTECARLO_PARAMS,\
    RADIATOR_PARAMS, SHOWER_PARAMS
from examples.utils.plot_comparisons import *

from examples.sudakov_comparsions.sudakov_comparison_numeric import \
    plot_mc_crit, plot_mc_all
# Import plot pythia


# =====================================
# Definitions and Parameters
# =====================================
params          = ALL_MONTECARLO_PARAMS

radiator_params = RADIATOR_PARAMS
del radiator_params['z_cut']
del radiator_params['beta']

shower_params = SHOWER_PARAMS

# ---------------------------------
# Unpacking parameters
# ---------------------------------
jet_type = params['jet type']
fixed_coupling = params['fixed coupling']

num_mc_events = params['number of MC events']

num_rad_bins = params['number of bins']

# =====================================
# Calculations
# =====================================
def get_mc_crit(z_cut, beta,
                load=True,
                verbose=5):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=F_SOFT, acc=OBS_ACC)

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights

    if verbose > 1:
        arg = np.argmax(obs)
        print("zc: " + str(z_cut))
        print("obs_acc: " + OBS_ACC)
        print("maximum observable: " + str(obs[arg]))
        print("associated with\n    z = "+str(z_crits[arg])
              +"\n    theta = "+str(theta_crits[arg]))
        print('', flush=True)

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density

    return sud_integrator.bin_midpoints, pdf

def get_mc_all(z_cut, beta,
               load=True):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
                         load=load, save=True, theta_crits=theta_crits,
                         rad_crit_sub=radiators.get('subsequent', None))

    z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=True,
                        theta_crits=theta_crits,
                        rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights * c_sub_weights * z_pre_weights

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density

    return sud_integrator.bin_midpoints, pdf

# =====================================
# Plotting
# =====================================
def compare_sudakov(z_cut, beta):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('all', ratio_plot=False)

    # Analytic plot
    if fixed_coupling:
        # DEBUG: High zorder so easy to see
        # Solid, black
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                               beta, icol=0,
                           # DEBUG: Make color black
                               jet_type='quark',
                               f_soft=F_SOFT,
                               label=r'Analytic, 1 Em.')
    else:
        # DEBUG: Get and plot pythia information
        #plot_pythia
        pass

    # Get MC info
    one_em_mc_bins, one_em_mc_pdf = get_mc_crit(z_cut, beta)
    mul_em_mc_bins, mul_em_mc_pdf = get_mc_all(z_cut, beta)

    # Get PS info
    # DEBUG: Reverse engineer plot_shower_pdf_cdf
    plot_shower_pdf_cdf(
        ps_correlations(beta)['rss_c1s_two'][i],
        # ps_correlations(beta)['rss_c1s_one'][i],
        axes_pdf, axes_cdf,
        label='Parton Shower', colnum=i)


    # Plot ME MC
    plt.plot(mul_em_mc_bins, mul_em_mc_pdf,
             linewidth=2, linestyle='solid',
             color='indianred')
    

    # Plot ME PS
    # DEBUG: Get info and plot manually
    plt.plot(mul_em_ps_bins, mul_em_ps_pdf,
             linewidth=2, linestyle='solid',
             color='royalblue')
    


    # Make legends
    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})

    # DEBUG: Coloring
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    # Plot 1E MC
    plt.plot(one_em_mc_bins, one_em_mc_pdf,
             linewidth=2, linestyle='dashed',
             color='coralred')
    

    # Plot 1E PS
    # DEBUG: Get info and plot manually
    plt.plot(one_em_ps_bins, one_em_ps_pdf,
             linewidth=2, linestyle='dashed',
             color='cornflowerblue')
    

    # Saving and closing figure
    fig_pdf.savefig('sudakov-comparison_'+
                    f'zcut-{z_cut}_'+
                    f'beta-{beta}_'+
                    'fixed' if fixed_coupling else 'running'+
                    '-coupling',
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +'.pdf',
                    format='pdf')

    plt.close(fig_pdf)

    print("Plotting complete!", flush=True)

# =====================================
# Main:
# =====================================
if __name__ == '__main__':
    compare_sudakov(.1, 2)
