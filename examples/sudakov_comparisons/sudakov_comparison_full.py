from __future__ import absolute_import
import itertools

# Parameters
from examples.params import *
from examples.file_manager import fig_folder
from examples.sudakov_comparisons.sudakov_utils import *

save_cdf = False
plot_level = 'partons'

index_zc = {.05: 0, .1: 1, .2: 2}

xlabels = {.5: r'$C_1^{(1/2)}$',
           1: r'$C_1^{(1)}$',
           2: r'$C_1^{(2)}$',
           3: r'$C_1^{(3)}$',
           4: r'$C_1^{(4)}$'}
ylabels = {.5: r'$\frac{{\rm d} \sigma}{{\rm d}{\rm log}_{10} C_1^{(.5)}}$',
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

def compare_ecf_pdf(z_cut, beta, emission='crit', plot_ivs=True):
    assert emission in ['crit', 'all']

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
    for i_f, f_soft in enumerate(F_SOFT_PLOT):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                           beta, jet_type='quark', f_soft=f_soft,
                           col=plot_colors[f_soft]['fc'],
                           label=None)

    # Legend for Colors
    if not plot_ivs:
        f_lines = [Line2D([0], [0], color=f_colors[f_soft], lw=2)
                   for f_soft in F_SOFT_PLOT]
        f_labels = [r'$f_{\rm RSS}=1/2$', r'$f_{\rm RSS}=1$']
        leg1 = axes_pdf[0].legend(f_lines, f_labels, loc=(.0155, .545))
        leg2 = axes_cdf[0].legend(f_lines, f_labels, loc=(.0155, .545))
        for i_f, text in enumerate(leg1.get_texts()):
            text.set_color(plot_colors[F_SOFT_PLOT[i_f]]['num'])
    elif plot_ivs:
        lines = [Line2D([0], [0], color=f_colors[1], lw=2),
                 Line2D([0], [0], color=f_colors[1/2], lw=2),
                 Line2D([0], [0], color=f_colors['ivs'], lw=2)]
        labels = [r'$f_{\rm RSS}=1/2$', r'$f_{\rm RSS}=1$', r'IVS']
        leg1 = axes_pdf[0].legend(lines, labels, loc=(.0155, .545))
        leg2 = axes_cdf[0].legend(lines, labels, loc=(.0155, .545))
        for i_f, text in enumerate(leg1.get_texts()):
            text.set_color(plot_colors[F_SOFT_PLOT_IVS[i_f]]['num'])

    print("Getting samples...", flush=True)
    for i_f, f_soft in enumerate(F_SOFT_PLOT):
        # Monte Carlo Integration
        if emission == 'crit':
            plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, f_soft,
                         col=plot_colors[f_soft]['num'])
        elif emission == 'all':
            plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, f_soft,
                        col=plot_colors[f_soft]['num'])

        # MLL Parton Shower
        shower_c2s = ps_correlations(beta, f_soft)['rss_c1s_'+emission][index_zc[z_cut]]
        plot_shower_pdf_cdf(shower_c2s, axes_pdf, axes_cdf,
                            style=modstyle, label=None,
                            col=plot_colors[f_soft]['shower'])

        # Pythia
        # Weights, binned observables, and area
        if BIN_SPACE == 'lin':
            bins = np.linspace(0, .5, 100)
        if BIN_SPACE == 'log':
            bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5), 100)
        params = (z_cut, f_soft)

        # Narrowing in on jets with P_T between 3 and 3.5 TeV
        cond_floor = (3000 < np.array(pythia_data['raw'][plot_level]['pt'][beta]))
        cond_ceil = (np.array(pythia_data['raw'][plot_level]['pt'][beta]) < 3500)
        inds = np.where(cond_floor * cond_ceil)[0]

        # Gettings substructure observables
        pythia_c2s = pythia_data['rss'][plot_level][params]['C1'][beta]
        pythia_c2s = np.array(pythia_c2s)[inds]
        height, _ = np.histogram(pythia_c2s, bins)
        norm = np.sum(height * (np.log10(bins[1:]) - np.log10(bins[:-1])))
        axes_pdf[0].hist(pythia_c2s, bins=bins,
                         weights=np.ones(len(pythia_c2s))/norm,
                         histtype='step', lw=1.5,
                         edgecolor=plot_colors[f_soft]['pythia'])

        figtemp = plt.figure()
        axtemp = [figtemp.add_subplot(1, 1, 1)]

        plot_pythia_pdf_cdf(pythia_c2s, axtemp, axes_cdf,
                            label=None,
                            col=plot_colors[f_soft]['pythia'])

    if plot_ivs:
        print("Plotting IVS...", flush=True)
        plot_mc_ivs(axes_pdf, axes_cdf, z_cut, beta, f_soft,
                    col=plot_colors['ivs']['num'])
        params = z_cut
        pythia_c2s = pythia_data['ivs'][plot_level][params]['C1'][beta][inds]
        height, _ = np.histogram(pythia_c2s, bins)

        height, _ = np.histogram(pythia_c2s, bins)
        norm = np.sum(height * (np.log10(bins[1:]) - np.log10(bins[:-1])))
        axes_pdf[0].hist(pythia_c2s, bins=bins,
                         weights=np.ones(len(pythia_c2s))/norm,
                         histtype='step', lw=1.5,
                         edgecolor=plot_colors['ivs']['pythia'])

        figtemp = plt.figure()
        axtemp = [figtemp.add_subplot(1, 1, 1)]

        plot_pythia_pdf_cdf(pythia_c2s, axtemp, axes_cdf,
                            label=None,
                            col=plot_colors['ivs']['pythia'])

    # Saving plots
    labels=['LL pQCD', 'MLL pQCD', 'MLL Shower', 'Pythia 8.244']
    full_legend(axes_pdf[0], labels, loc='upper left')
    full_legend(axes_cdf[0], labels, loc='upper left')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    axes_pdf[0].text(0.006, 1.08, r'$\bf{Quark~jets}$'
                     + '\n' + r'$p_T$ = 3 TeV'
                     +'\n'+r'$R$ = 1.0, $z_{\rm cut}$ = '+str(z_cut),
                     ha='center')
    #axes_pdf[0].text(0.006, .93, r'$\bf{Preliminary}$',
    #                 ha='center', color='red')

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_'+emission+'_full_pdf_comp'
                    +'_beta'+str(beta)+'_zc'+str(z_cut)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_'+emission+'_full_pdf_comp'
                        +'_beta'+str(beta)+'_zc'+str(z_cut)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')
    plt.close(fig_pdf)
    plt.close(fig_cdf)
    print("Plotting complete!", flush=True)


###########################################
# Main:
###########################################
if __name__ == '__main__':
    # Arguments:
    # (zcut, beta, emission, plot_ivs)
    args_list = itertools.product([.1, .2], [.5, 1, 2],
                                  ['crit', 'all'], [False])  #, True])
    for args in args_list:
        compare_ecf_pdf(*args)

