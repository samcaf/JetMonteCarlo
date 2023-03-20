# Loading data
from file_management.catalog_utils import fig_folder
from file_management.load_data import load_partonshower_samples

# pdf utilities
from jetmontecarlo.utils.hist_utils import vals_to_pdf
from jetmontecarlo.utils.plot_utils import stamp

# Soft Drop plotting
from jetmontecarlo.analytics.radiators.soft_drop import plot_softdrop_analytic

# Parameters and plotting utilities
from examples.params import tab
from examples.params import ALL_MONTECARLO_PARAMS,\
    RADIATOR_PARAMS, SHOWER_PARAMS
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

num_bins = 100


# =====================================
# Plot Setup
# =====================================
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

stampsize=14
legsize=14

# ---------------------------------
# Plotting flags
# ---------------------------------
# Groomers
plot_rsf1 = True
plot_mmdt = True

# Data types
plot_mc = True
plot_ps = True

# Pythia
plot_pythia_partons = True
plot_pythia_hadrons = True

# ---------------------------------
# Plotting colors 
# ---------------------------------
pythiacolor_parton='grey'
pythiacolor_hadron='black'

analyticcolor='black'

mccolor = {'dashed': 'darkorange',
           'solid': 'orange'}

pscolor = {'dashed': 'firebrick',
           'solid': 'indianred'}

# ---------------------------------
# Legend info
# ---------------------------------
stamptext_loc = (0.94, .94)

legend_loc = (.03, .75) if fixed_coupling else (.48, 0.0)
soliddashed_loc = (.03, .5) if fixed_coupling else (.03, .94)


# =====================================
# Plotting
# =====================================
def compare_sudakov_emissions(z_cut, beta,
                              groomer='rsf1'):
    # Preparing figures
    fig_pdf, axes_pdf, _, axes_cdf = \
            get_axes('all', ratio_plot=False)

    # - - - - - - - - - - - - - - - - -
    # Setting up for different groomers
    # - - - - - - - - - - - - - - - - -
    if groomer == 'rsf1':
        gname = 'rss'
        kwargs = {'f_soft': 1}
        params = (z_cut, 1)
        groomer_tex = r'$\bf{P}$-$\bf{RSF_{f=1}}$'
    elif groomer == 'mmdt':
        gname = 'softdrop'
        kwargs = {'beta_sd': 0}
        params = (0, z_cut, 1)
        groomer_tex = r'$\bf{SD_{\beta=0}}$'

    # - - - - - - - - - - - - - - - - -
    # Analytic plot
    # - - - - - - - - - - - - - - - - -
    if fixed_coupling:
        legend_info = 'LL '
        if groomer == 'rsf1':
            plot_crit_analytic(axes_pdf, axes_cdf,
                           z_cut, beta,
                           jet_type='quark',
                           col=analyticcolor,
                           label=r'LL Analytic',
                           **kwargs)

        if groomer == 'mmdt':
            plot_softdrop_analytic(axes_pdf, axes_cdf,
                                   bin_space='log',
                                   z_cut=z_cut, beta=beta,
                                   jet_type='quark',
                                   col=analyticcolor,
                                   label='LL Analytic',
                                   **kwargs)
            pass
    # - - - - - - - - - - - - - - - - -
    # Pythia Plot
    # - - - - - - - - - - - - - - - - -
    else:
        legend_info = 'MLL '
        try:
            print("\nLoading Pythia data...\n", flush=True)
            pythia_data = get_pythia_data(include=['raw', gname],
                                          levels=['partons', 'hadrons'])

            # - - - - - - - - - - - - 
            # Jet cuts
            # - - - - - - - - - - - - 
            # Narrowing in on jets with P_T > 3 TeV
            p_cond_floor = (3000 < np.array(pythia_data['raw']['partons']['pt'][beta]))
            # Narrowing in on jets with P_T between 3 and 3.5 TeV
            p_cond_ceil = (np.array(pythia_data['raw']['partons']['pt'][beta]) < 3500)

            # Narrowing in on jets with P_T > 3 TeV
            h_cond_floor = (3000 < np.array(pythia_data['raw']['hadrons']['pt'][beta]))
            # Narrowing in on jets with P_T between 3 and 3.5 TeV
            h_cond_ceil = (np.array(pythia_data['raw']['hadrons']['pt'][beta]) < 3500)

            parton_inds = np.where(p_cond_floor * p_cond_ceil)[0]
            hadron_inds = np.where(h_cond_floor * h_cond_ceil)[0]

            # - - - - - - - - - - - - 
            # Getting substructure
            # - - - - - - - - - - - - 
            pythia_parton_vals = pythia_data[gname]['partons']\
                                    [params]['C1'][beta]
            pythia_parton_vals = np.array(pythia_parton_vals)[parton_inds]

            pythia_hadron_vals = pythia_data[gname]['hadrons']\
                                    [params]['C1'][beta]
            pythia_hadron_vals = np.array(pythia_hadron_vals)[hadron_inds]

            # - - - - - - - - - - - - 
            # Plotting
            # - - - - - - - - - - - - 
            print("\nGetting Pythia pdfs...\n", flush=True)
            for add_plot, pythia_vals, pythiacolor, plot_type in zip(
                    [plot_pythia_partons, plot_pythia_hadrons],
                    [pythia_parton_vals, pythia_hadron_vals],
                    [pythiacolor_parton, pythiacolor_hadron],
                    ['Partons', 'Hadrons']):
                pythia_xs, pythia_pdf = vals_to_pdf(pythia_vals,
                    num_bins, bin_space='log',
                    log_cutoff=1e-10)

                axes_pdf[0].plot(pythia_xs, pythia_pdf,
                         linewidth=2, linestyle='solid',
                         label=r'$\texttt{Pythia}$, '+plot_type,
                         color=pythiacolor)

        except FileNotFoundError as error:
            print('Pythia data not found;'
                  'got FileNotFoundError\n'+tab+f'{error}.\n'
                  'Skipping Pythia plot.')

    # - - - - - - - - - - - - - - - - -
    # Get MC info
    # - - - - - - - - - - - - - - - - -
    print("\nGetting one emission MC pdf...\n", flush=True)
    one_em_mc_bins, one_em_mc_pdf = get_mc_crit(z_cut, beta,
                                                groomer=groomer,
                                                nbins=num_bins)
    print("\nGetting all emissions MC pdf...\n", flush=True)
    mul_em_mc_bins, mul_em_mc_pdf = get_mc_all(z_cut, beta,
                                               groomer=groomer,
                                               nbins=num_bins)

    # - - - - - - - - - - - - - - - - -
    # Get PS info
    # - - - - - - - - - - - - - - - - -
    # Single Emission
    if groomer == 'rsf1':
        # Emission type is only for rsf
        kwargs['emission type'] = 'crit'

    one_em_ps_vals = load_partonshower_samples(gname,
                               n_emissions='1',
                               z_cuts=[z_cut], betas=[beta],
                               **kwargs)[z_cut][beta]

    # Multiple Emissions
    if groomer == 'rsf1':
        # Emission type is only for rsf
        kwargs['emission type'] = 'precritsub'

    mul_em_ps_vals = load_partonshower_samples(gname,
                               n_emissions='2',
                               z_cuts=[z_cut], betas=[beta],
                               **kwargs)[z_cut][beta]

    # PDFs
    print("\nGetting one emission parton shower pdf...\n", flush=True)
    one_em_ps_bins, one_em_ps_pdf = vals_to_pdf(
        one_em_ps_vals, num_bins, bin_space='log',
        log_cutoff=1e-20 if fixed_coupling else 1e-10)
    print("\nGetting all emissions parton shower pdf...\n", flush=True)
    mul_em_ps_bins, mul_em_ps_pdf = vals_to_pdf(
        mul_em_ps_vals, num_bins, bin_space='log',
        log_cutoff=1e-20 if fixed_coupling else 1e-10)

    # - - - - - - - - - - - - - - - - -
    # Multiple Emission Plots
    # - - - - - - - - - - - - - - - - -
    # Plot ME MC
    if plot_mc:
        axes_pdf[0].plot(mul_em_mc_bins, mul_em_mc_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Convolution',
             color=mccolor['solid'])

    # Plot ME PS
    if plot_ps:
        axes_pdf[0].plot(mul_em_ps_bins, mul_em_ps_pdf,
             linewidth=2, linestyle='solid',
             label=f'{legend_info}Shower',
             color=pscolor['solid'])

    # - - - - - - - - - - - - - - - - -
    # Make legends for color
    # - - - - - - - - - - - - - - - - -
    axes_pdf[0].legend(loc=legend_loc,
                       prop={'size': legsize}, frameon=False)

    # - - - - - - - - - - - - - - - - -
    # One Emission Plots
    # - - - - - - - - - - - - - - - - -
    # Plot 1E MC
    if plot_mc:
        axes_pdf[0].plot(one_em_mc_bins, one_em_mc_pdf,
             linewidth=2, linestyle='dashed',
             color=mccolor['dashed'])

    # Plot 1E PS
    if plot_ps:
        axes_pdf[0].plot(one_em_ps_bins, one_em_ps_pdf,
             linewidth=2, linestyle='dashed',
             color=pscolor['dashed'])
    # - - - - - - - - - - - - - - - - -
    # Stamps
    # - - - - - - - - - - - - - - - - -
    # Informational stamp
    coupling = r'Fixed Coupling' if fixed_coupling else\
        r'Running Coupling'
    obsname = r'$\mathbf{C_1^{(2)}}$ $\textbf{Distribution}$' if beta == 2 else\
        (r'$\mathbf{C_1^{(1)}}$ $\textbf{Distribution$}' if beta == 1 else\
         r'$C_1^{(XXX)}$')
    line_0 = groomer_tex+' '+obsname
    line_1 = coupling
    line_2 = r'$p_T$ = 3 TeV, $R$ = 1'
    line_3 = r'$z_{\rm cut}=$'+f' {z_cut:.1f}'

    stamp(*stamptext_loc, axes_pdf[0],
          line_0=line_0,
          textops_update={'fontsize': stampsize,
                          'horizontalalignment': 'right'})
    stamp(*stamptext_loc, axes_pdf[0],
          line_0='', line_1=line_1,
          line_2=line_2, line_3=line_3,
          textops_update={'fontsize': stampsize-2,
                          'horizontalalignment': 'right'})

    # Extra solid-dashed legend stamp
    stamp(*soliddashed_loc, axes_pdf[0],
          line_0='Solid: Mult. Em.',
          line_1='Dashed: One Em.',
          textops_update={'fontsize': stampsize})

    # - - - - - - - - - - - - - - - - -
    # Saving and closing figure
    # - - - - - - - - - - - - - - - - -
    fig_pdf.tight_layout()
    fig_loc = str(fig_folder) + '/sudakov-comparison_'+\
        groomer+'_'+f'zcut-{z_cut}_'+f'beta-{beta}_'+\
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
    # RSF-1
    if plot_rsf1:
        compare_sudakov_emissions(z_cut=.1, beta=2, groomer='rsf1')
        compare_sudakov_emissions(z_cut=.1, beta=4, groomer='rsf1')
        # compare_sudakov_emissions(z_cut=.2, beta=2, groomer='rsf1')
        # compare_sudakov_emissions(z_cut=.1, beta=1, groomer='rsf1')
        # compare_sudakov_emissions(z_cut=.2, beta=1, groomer='rsf1')
    # mMDT
    if plot_mmdt:
        compare_sudakov_emissions(z_cut=.1, beta=2, groomer='mmdt')
        compare_sudakov_emissions(z_cut=.1, beta=4, groomer='mmdt')
        # compare_sudakov_emissions(z_cut=.2, beta=2, groomer='mmdt')
        # compare_sudakov_emissions(z_cut=.1, beta=1, groomer='mmdt')
        # compare_sudakov_emissions(z_cut=.2, beta=1, groomer='mmdt')
