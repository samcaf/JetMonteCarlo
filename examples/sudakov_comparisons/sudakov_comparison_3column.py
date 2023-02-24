from __future__ import absolute_import
import itertools

# Parameters
from examples.params import *
from file_management.catalog_utils import fig_folder
from examples.sudakov_comparisons.sudakov_utils import *

index_zc = {.05: 0, .1: 1, .2: 2}

xlabels = {.5: r'$C_1^{(1/2)}$',
           1: r'$C_1^{(1)}$',
           2: r'$C_1^{(2)}$',
           3: r'$C_1^{(3)}$',
           4: r'$C_1^{(4)}$'}
ylabels = {.5: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(1/2)}}$',
           1: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(1)}}$',
           2: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(2)}}$',
           3: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(3)}}$',
           4: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(4)}}$'}

ylim_dict = {.5: 1.3,
             1: .65,
             2: .39,
             3: .4,
             4: .4}

xlim_dict = {.5: 1e-5,
             1: 1e-6,
             2: 1e-7,
             3: 1e-8,
             4: 1e-9}

level_label = {'partons': 'Partons',
               'hadrons': 'Hadrons',
               'charged': 'Charged'}

level_style = {'partons': 'solid',
               'hadrons': 'dashed',
               'charged': 'dotted'}

level_col = {'partons': 1.7,
             'hadrons': 1.5,
             'charged': 1.1}

def plot_rss_pythia(z_cut, beta,
                    plot_levels=['partons', 'hadrons', 'charged']):
    # Preparing figures
    fig_pdf, axes_pdf = aestheticfig(xlabel=xlabels[beta],
                            ylabel=ylabels[beta],
                            title=None, showdate=False,
                            xlim=(1e-8, 1), ylim=(0, 1),
                            ratio_plot=False, labeltext=None)
    fig_cdf, axes_cdf = aestheticfig(xlabel=xlabels[beta],
                            ylabel=r'$\Sigma(C)$',
                            title=None, showdate=False,
                            xlim=(1e-8, 1), ylim=(0, 1),
                            ratio_plot=False, labeltext=None)
    axes_pdf[0].set_ylabel(ylabels[beta], size=20)
    axes_pdf[0].set_xscale('log')
    axes_cdf[0].set_xscale('log')

    axes_pdf[0].set_xlim(xlim_dict[beta], 1)
    axes_pdf[0].set_ylim(0, ylim_dict[beta])


    # Pythia
    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        bins = np.linspace(0, .5, 100)
    if BIN_SPACE == 'log':
        bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5), 100)

    """
    # Narrowing in on jets with hadronic P_T between 3 and 3.5 TeV
    cut_level = 'hadrons'
    cond_floor = (3000 < np.array(pythia_data['raw'][cut_level]['pt'][beta]))
    cond_ceil = (np.array(pythia_data['raw'][cut_level]['pt'][beta]) < 3500)
    inds = np.where(cond_floor * cond_ceil)[0]
    """

    for f_soft in [.5, 1]:
        for level in plot_levels:
            # Narrowing in on jets with hadronic P_T between 3 and 3.5 TeV
            cond_floor = (3000 < np.array(pythia_data['raw'][level]['pt'][beta]))
            cond_ceil = (np.array(pythia_data['raw'][level]['pt'][beta]) < 3500)
            inds = np.where(cond_floor * cond_ceil)[0]

            # Gettings substructure observables
            params = (z_cut, f_soft)
            pythia_c2s = np.array(pythia_data['rss'][level][params]['C1'][beta])
            if len(pythia_c2s) == 1:
                pythia_c2s = np.array(pythia_c2s[0])
                print("Problematic elements:", pythia_c2s[1021683::])

            pythia_c2s = pythia_c2s[inds]

            # Getting correct normalization for logspace:
            heights, _ = np.histogram(pythia_c2s, bins, weights=pythia_c2s)
            norm = np.sum(heights * (np.log10(bins[1:]) - np.log10(bins[:-1])))
            norm_heights, _, _ = axes_pdf[0].hist(pythia_c2s, bins=bins,
                                    weights=pythia_c2s/norm,
                                    histtype='step', lw=1.5,
                                    label=level_label[level],
                                    linestyle=level_style[level],
                                    edgecolor=adjust_lightness(
                                    plot_colors[f_soft]['pythia'],
                                    level_col[level]
                                    )
                                )

    axes_pdf[0].text(0.006, 1.08, r'RSF Substructure ('+xlabels[beta]+')\n'
                     + 'Quark jets, ' + r'\texttt{Pythia 8.244}'
                     + '\n' + r'$p_T$ = 3 TeV, AKT1',
                     ha='center')

    fig_pdf.savefig(str(fig_folder) + '/' + 'pythia_rss'+'_beta'+str(beta)+'_zc'+str(z_cut)
                    +'.pdf',
                    format='pdf')

    plt.close(fig_pdf)
    print("Plotting complete!", flush=True)


def plot_rss_pQCD(z_cut, beta,
                  plot_analytic=False,
                  plot_numerical='all',
                  plot_shower=True):
    # Preparing figures
    fig_pdf, axes_pdf = aestheticfig(xlabel=xlabels[beta],
                            ylabel=ylabels[beta],
                            title=None, showdate=False,
                            xlim=(1e-8, 1), ylim=(0, 1),
                            ratio_plot=False, labeltext=None)
    fig_cdf, axes_cdf = aestheticfig(xlabel=xlabels[beta],
                            ylabel=r'$\Sigma(C)$',
                            title=None, showdate=False,
                            xlim=(1e-8, 1), ylim=(0, 1),
                            ratio_plot=False, labeltext=None)
    axes_pdf[0].set_ylabel(ylabels[beta], size=20)
    axes_pdf[0].set_xscale('log')
    axes_cdf[0].set_xscale('log')

    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .7)
    else:
        axes_pdf[0].set_xlim(xlim_dict[beta], 1)
        axes_pdf[0].set_ylim(0, ylim_dict[beta])

    # Analytic plot
    if plot_analytic:
        for i_f, f_soft in enumerate(F_SOFT_PLOT):
            plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                               beta, jet_type='quark', f_soft=f_soft,
                               col=plot_colors[f_soft]['fc'],
                               label=None)

    # Legend for Color
    f_lines = [Line2D([0], [0], color=f_colors[f_soft], lw=2)
               for f_soft in F_SOFT_PLOT]
    f_labels = [r'$P-RSF_{1/2}$', r'$P-RSF_1$']
    leg1 = axes_pdf[0].legend(f_lines, f_labels, loc=(.0155, .545))
    leg2 = axes_cdf[0].legend(f_lines, f_labels, loc=(.0155, .545))
    for i_f, text in enumerate(leg1.get_texts()):
        text.set_color(plot_colors[F_SOFT_PLOT[i_f]]['num'])

    # Numerical plot
    if plot_numerical == 'crit':
        for i_f, f_soft in enumerate(F_SOFT_PLOT):
            plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, f_soft,
                         col=plot_colors[f_soft]['num'])
    elif plot_numerical == 'all':
        for i_f, f_soft in enumerate(F_SOFT_PLOT):
            plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, f_soft,
                        col=plot_colors[f_soft]['num'])

    # Shower plot
    if plot_shower:
        for i_f, f_soft in enumerate(F_SOFT_PLOT):
            shower_c2s = ps_correlations(beta, f_soft)['rss_c1s_'+plot_numerical][index_zc[z_cut]]
            plot_shower_pdf_cdf(shower_c2s, axes_pdf, axes_cdf,
                                style=modstyle, label=None,
                                col=plot_colors[f_soft]['shower'],
                                verbose=3)

    # Saving plots
    labels = []
    if plot_analytic:
        labels.append('LL pQCD')
    if plot_numerical is not None:
        labels.append('MLL pQCD')
    if plot_shower:
        labels.append('MLL Shower')
    for ax, leg in zip([axes_pdf[0], axes_cdf[0]], [leg1, leg2]):
        full_legend(ax, labels, loc='upper left')
        ax.add_artist(leg)

        ax.text(0.006, 1.08, r'$\bf{Quark~jets}, pQCD$'
                + '\n' + r'$p_T$ = 3 TeV'
                +'\n'+r'$R$ = 1.0, $z_{\rm cut}$ = '+str(z_cut),
                ha='center')

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig('paper/pqcd_rss'
                    +('_num'+plot_numerical if plot_numerical is not None else '')
                    +'_beta'+str(beta)+'_zc'+str(z_cut)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')

    fig_cdf.savefig('cdf_rss'
                    +('_num'+plot_numerical if plot_numerical is not None else '')
                    +'_beta'+str(beta)+'_zc'+str(z_cut)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')

    plt.close(fig_pdf)
    print("Plotting complete!", flush=True)



###########################################
# Main:
###########################################
if __name__ == '__main__':
    # Arguments:
    # (zcut, beta)
    args_list = itertools.product([.1, .2], [.5, 1, 2])
    for args in args_list:
        if not FIXED_COUPLING and OBS_ACC == 'MLL':
            # Pythia
            plot_rss_pythia(*args)
        # pQCD
        for plot_num in ['crit', 'all']:
            plot_rss_pQCD(*args, plot_numerical=plot_num)
