# Loading data
from file_management.catalog_utils import fig_folder
from file_management.load_data import load_partonshower_samples

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
num_shower_events = shower_params['number of shower events']
del shower_params['shower beta']

# ---------------------------------
# Unpacking parameters
# ---------------------------------
jet_type = params['jet type']
fixed_coupling = params['fixed coupling']

num_mc_events = params['number of MC events']

num_rad_bins = params['number of radiator bins']


# =====================================
# Plotting
# =====================================
def compare_sudakov_emissions(z_cut, beta):
    # Preparing figures
    fig_pdf, axes_pdf, _, axes_cdf = \
            get_axes('all', ratio_plot=False)

    # Analytic plot
    if fixed_coupling:
        legend_info = 'LL '
        plot_crit_analytic(axes_pdf, axes_cdf,
                           z_cut, beta,
                           jet_type='quark',
                           f_soft=F_SOFT,
                           col='black',
                           label=r'Analytic, 1 Em.')
    else:
        legend_info = 'MLL '
    if False:
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
        except KeyError as error:
            print(f"{pythia_data['rss']['hadrons'].keys()=}")
            raise error
        pythia_vals = np.array(pythia_vals)[inds]

        pythia_xs, pythia_pdf = vals_to_pdf(pythia_vals,
            num_rad_bins, bin_space='log',
            log_cutoff=-10)

        axes_pdf[0].plot(pythia_xs, pythia_xs * pythia_pdf,
                 linewidth=2, linestyle='solid',
                 label=f'{legend_info}'
                      r'\texttt{Pythia 8.244}',
                 color='black')

    # Get MC info
    one_em_mc_bins, one_em_mc_pdf = get_mc_crit(z_cut, beta)
    mul_em_mc_bins, mul_em_mc_pdf = get_mc_all(z_cut, beta)

    # Get PS info
    one_em_ps_vals = load_partonshower_samples('rss',
                               n_emissions='1',
                               emission_type='crit',
                               z_cuts=[z_cut], betas=[beta],
                               f_soft=F_SOFT)[z_cut][beta]
    mul_em_ps_vals = load_partonshower_samples('rss',
                               n_emissions='2',
                               emission_type='precritsub',
                               z_cuts=[z_cut], betas=[beta],
                               f_soft=F_SOFT)[z_cut][beta]

    one_em_ps_bins, one_em_ps_pdf = vals_to_pdf(
        one_em_ps_vals, num_rad_bins, bin_space='log',
        log_cutoff=1e-20 if fixed_coupling else 1e-10)
    mul_em_ps_bins, mul_em_ps_pdf = vals_to_pdf(
        mul_em_ps_vals, num_rad_bins, bin_space='log',
        log_cutoff=1e-20 if fixed_coupling else 1e-10)

    # Plot ME MC
    axes_pdf[0].plot(mul_em_mc_bins, mul_em_mc_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Monte Carlo',
             color='indianred')

    # Plot PS MC
    axes_pdf[0].plot(mul_em_ps_bins, mul_em_ps_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Parton Shower',
             color='mediumvioletred')

    # Make legends for color
    axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15}, frameon=False)

    # Plot 1E MC
    axes_pdf[0].plot(one_em_mc_bins, one_em_mc_pdf,
             linewidth=2, linestyle='dashed',
             color='lightcoral')

    # Plot 1E PS
    axes_pdf[0].plot(one_em_ps_bins, one_em_ps_pdf,
             linewidth=2, linestyle='dashed',
             color='orchid')

    # Make legends for linestyle
    # DEBUG: Make legend that indicates solid = ME, dashed = 1E

    # Stamp
    obsname = r'$C_1^{(2)}$' if beta == 2 else\
        (r'$C_1^{(1)}$' if beta == 1 else r'$C_1^{(XXX)}$')
    line1 = r'$\bf{P-RSF_{1/2}~Groomed~}$'+obsname
    coupling = 'Fixed Coupling' if fixed_coupling else 'Running Coupling'
    line2 = coupling + r', $z_{\rm cut}=$'+f'{z_cut:.1f}'

    fig_pdf.text(0.5, 0.925, line1,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=15)
    fig_pdf.text(0.5, 0.87, line2,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=15)

    fig_pdf.text(0.2, 0.4,  'Solid: Mult. Em.',
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontsize=15)
    fig_pdf.text(0.2, 0.35, 'Dashed: One Em.',
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontsize=15)

    # Saving and closing figure
    fig_pdf.tight_layout()
    fig_loc = str(fig_folder) + '/sudakov-comparison_'+\
        f'zcut-{z_cut}_'+f'beta-{beta}_'+\
        ('fixed' if fixed_coupling else 'running')+\
        '-coupling'+\
        f'_{num_shower_events:.0e}showers'+\
        f'_{num_mc_events:.0e}mc'+\
        '.pdf'

    print(f"Saving figure to {fig_loc}")
    fig_pdf.savefig(fig_loc,  bbox_inches='tight')

    plt.close(fig_pdf)

    print("Plotting complete!", flush=True)


# =====================================
# Main:
# =====================================
if __name__ == '__main__':
    compare_sudakov_emissions(.1, 2)
