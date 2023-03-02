# pdf utilities
from jetmontecarlo.utils.hist_utils import vals_to_pdf

# Parameters and plotting utilities
from examples.params import ALL_MONTECARLO_PARAMS,\
    RADIATOR_PARAMS, SHOWER_PARAMS
from examples.utils.plot_comparisons import F_SOFT
from examples.utils.plot_comparisons import *

from examples.sudakov_comparisons.sudakov_utils import\
    get_mc_crit, get_mc_all, get_pythia_data


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
# Plotting
# =====================================
def compare_sudakov_emissions(z_cut, beta):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('all', ratio_plot=False)

    # Analytic plot
    if fixed_coupling:
        plot_crit_analytic(axes_pdf, axes_cdf,
                           z_cut, beta,
                           jet_type='quark',
                           f_soft=F_SOFT,
                           col='black',
                           label=r'Analytic, 1 Em.')
        legend_info = 'LL '
    else:
        pythia_data = get_pythia_data(include=['raw', 'rss'],
                                      levels=['hadrons'])
        # Narrowing in on jets with P_T > 3 TeV 
        cond_floor = (3000 < np.array(pythia_data['raw']['hadrons']['pt'][beta]))
        inds = np.where(cond_floor)[0]
        # Narrowing in on jets with P_T between 3 and 3.5 TeV
        # cond_ceil = (np.array(pythia_data['raw']['hadrons']['pt'][beta]) < 3500)
        # inds = np.where(cond_floor * cond_ceil)[0]

        # Getting substructure
        # DEBUG: Testing syntax
        try:
            pythia_vals = pythia_data['rss']['hadrons']\
                [z_cut][F_SOFT]['C1'][beta]
        except:
            print(f"{pythia_data['rss']['hadrons'].keys()=}")
        pythia_vals = np.array(pythia_c2s)[inds]

        pythia_xs, pythia_pdf = vals_to_pdf(pythia_vals,
            num_rad_bins, bin_space='log',
            log_cutoff=-10)
        legend_info = 'MLL '

    # Get MC info
    one_em_mc_bins, one_em_mc_pdf = get_mc_crit(z_cut, beta)
    mul_em_mc_bins, mul_em_mc_pdf = get_mc_all(z_cut, beta)

    # Get PS info
    one_em_ps_vals = ps_correlations(beta)['rss_c1s_two']
    mul_em_ps_vals = ps_correlations(beta)['rss_c1s_one']

    one_em_ps_bins, one_em_ps_pdf = vals_to_pdf(
        one_em_ps_vals, num_rad_bins, bin_space='log',
        log_cutoff=-20 if fixed_coupling else -10)
    mul_em_ps_bins, mul_em_ps_pdf = vals_to_pdf(
        mul_em_ps_vals, num_rad_bins, bin_space='log',
        log_cutoff=-20 if fixed_coupling else -10)


    # Plot ME MC
    plt.plot(mul_em_mc_bins, mul_em_mc_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Monte Carlo',
             color='indianred')
    

    # Plot ME PS
    plt.plot(mul_em_ps_bins, mul_em_ps_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Parton Shower',
             color='royalblue')


    # Make legends
    legend = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})

    # Plot 1E MC
    plt.plot(one_em_mc_bins, one_em_mc_pdf,
             linewidth=2, linestyle='dashed',
             color='coralred')

    # Plot 1E PS
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
    compare_sudakov_emissions(.1, 2)
