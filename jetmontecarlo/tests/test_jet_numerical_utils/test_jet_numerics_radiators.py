# Local utilities
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.jets.jet_numerics import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

###########################################
# Parameters
###########################################

JET_TYPE = 'quark'
ACC = 'MLL'
FIXED_COUPLING = False

Z_CUTS = [.05, .1, .2, .35]
BETAS = [1., 2., 3., 4.]
EPSILON = 1e-10

NUM_MC_EVENTS = int(1e5)
NUM_BINS = 500

BIN_SPACE = 'log'
ylim_sud = (0, 1.1)

if BIN_SPACE == 'lin':
    EPSILON = None
    ylim_1 = (0, 1.)
    ylim_2 = (0, 1.)
    xlim_crit = (0, 1.)
    xlim_sub = (0, .5)
    x_vals = np.linspace(.1, .4, 4)
if BIN_SPACE == 'log':
    ylim_1 = (0, 10)
    ylim_2 = (0, 20)
    xlim_crit = (EPSILON, 1.)
    xlim_sub = (1e-8, .5)
    x_vals = np.logspace(-7, -2, 4)

FUN_FIG = False

###########################################
# Fixed Coupling
###########################################
# ------------------------------------
# Critical Test:
# ------------------------------------
def test_crit_rad_fc():
    fig_rad, axes_rad = aestheticfig(xlabel=r'$\theta$', ylabel=r'R($\theta$)',
                                     title='Critical '+JET_TYPE+' radiator, '
                                     + r'fixed $\alpha_s$',
                                     ylim=ylim_1,
                                     xlim=(EPSILON, .5),
                                     ratio_plot=False, showdate=False)
    fig_sud, axes_sud = aestheticfig(xlabel=r'$\theta$',
                                     ylabel=r'$\Sigma(\theta$)',
                                     title = 'Critical '+JET_TYPE
                                     +' Sudakov Factor, '
                                     + r'fixed $\alpha_s$',
                                     ylim=ylim_sud,
                                     xlim=(EPSILON, .5),
                                     ratio_plot=False, showdate=False)
    if FUN_FIG:
        fig_fun, axes_fun = aestheticfig(xlabel=r'$\theta$',
                                         ylabel=r'$\Sigma(\theta$)',
                                         title = 'Critical '+JET_TYPE
                                         +' Sudakov Factor, '
                                         + r'fixed $\alpha_s$',
                                         ylim=ylim_sud,
                                         xlim=(EPSILON, 1.),
                                         ratio_plot=False, showdate=False)
    pnts = np.linspace(-.5, 1.5, 10000)

    if BIN_SPACE == 'log':
        pnts = np.logspace(np.log10(EPSILON), 0, 10000)
        axes_rad[0].set_xscale('log')
        axes_sud[0].set_xscale('log')
        if FUN_FIG:
            axes_fun[0].set_xscale('log')

    for icol, z_cut in enumerate(Z_CUTS):
        axes_rad[0].plot(pnts,
                         critRadAnalytic_fc_LL(pnts, z_cut, jet_type=JET_TYPE),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$z_c$='+str(z_cut))
        axes_sud[0].plot(pnts,
                         np.exp(-critRadAnalytic_fc_LL(pnts, z_cut,
                                                       jet_type=JET_TYPE)),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$z_c$='+str(z_cut))
        if FUN_FIG:
            axes_fun[0].plot(pnts,
                             np.exp(-critRadAnalytic_fc_LL(pnts, z_cut,
                                                           jet_type=JET_TYPE)),
                             **style_solid,
                             color=compcolors[(icol,'light')],
                             label=r'$z_c$='+str(z_cut))
    labelLines(axes_rad[0].get_lines(), xvals=x_vals)
    labelLines(axes_sud[0].get_lines(), xvals=x_vals)

    for icol, z_cut in enumerate(Z_CUTS):
        crit_sampler = criticalSampler(BIN_SPACE, zc=z_cut, epsilon=EPSILON)
        crit_sampler.generateSamples(NUM_MC_EVENTS)
        rad = gen_numerical_radiator(crit_sampler, 'crit', JET_TYPE, ACC,
                                     beta=None,
                                     bin_space=BIN_SPACE, epsilon=EPSILON,
                                     fixed_coupling=True,
                                     save=False)
        axes_rad[0].plot(pnts, rad(pnts),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
        axes_sud[0].plot(pnts, np.exp(-rad(pnts)),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
    fig_rad.savefig('critrad_1.pdf', format='pdf')
    fig_sud.savefig('critsud_1.pdf', format='pdf')
    if FUN_FIG:
        axes_fun[0].plot([0, 1], [.5, .5], ls='-', lw=3,
                         color=MyGreen)
        set_figtext(fig_fun, "uniform random variable", (.15,.5),
                    color=MyGreen)
        set_figtext(fig_fun, r"$\star~\to~\theta$ from inverse transform",
                    (0.55, 0.03), color=MyGreen)

        # Blue
        axes_fun[0].plot([2.7*1e-6, 2.7*1e-6], [0, .5], ls='--', lw=2.5,
                         color=compcolors[(3,'medium')])
        axes_fun[0].scatter([2.7*1e-6], [0], marker='*', s=200,
                            color=compcolors[(3,'dark')],
                            clip_on=False, zorder=100)

        # Red
        axes_fun[0].plot([7e-3, 7e-3], [0, .5], ls='--', lw=2.5,
                         color=compcolors[(2,'medium')])
        axes_fun[0].scatter([7e-3], [0], marker='*', s=200,
                            color=compcolors[(2,'dark')],
                            clip_on=False, zorder=100)

        # Purple
        axes_fun[0].plot([5.5*1e-2, 5.5*1e-2], [0, .5], ls='--', lw=2.5,
                         color=compcolors[(1,'medium')])
        axes_fun[0].scatter([5.5*1e-2], [0], marker='*', s=200,
                            color=compcolors[(1,'dark')],
                            clip_on=False, zorder=100)

        # Orange
        axes_fun[0].plot([1.5*1e-1, 1.5*1e-1], [0, .5], ls='--', lw=2.5,
                         color=compcolors[(0,'medium')])
        axes_fun[0].scatter([1.5*1e-1], [0], marker='*', s=200,
                            color=compcolors[(0,'dark')],
                            clip_on=False, zorder=100)

        fig_fun.savefig('critsud_2.pdf', format='pdf')

# ------------------------------------
# Critical Test:
# ------------------------------------
def test_sub_rad_fc():
    fig_rad, axes_rad = aestheticfig(xlabel=r'$C$', ylabel=r'R($C$)',
                                     title='Subsequent '+JET_TYPE+' radiator, '
                                     +r'fixed $\alpha_s$',
                                     ylim=(-.1, .1),
                                     xlim=(EPSILON, .5),
                                     ratio_plot=False, showdate=False)
    fig_sud, axes_sud = aestheticfig(xlabel=r'$C$',
                                     ylabel=r'$\Sigma(C$)',
                                     title='Subsequent '+JET_TYPE
                                     +' Sudakov Factor, '
                                     + r'fixed $\alpha_s$',
                                     ylim=(.9, 1.1),
                                     xlim=(EPSILON, .5),
                                     ratio_plot=False, showdate=False)

    pnts = np.linspace(-.5, 1., 10000)

    if BIN_SPACE == 'log':
        pnts = np.logspace(np.log10(EPSILON), np.log10(.5), 10000)
        axes_rad[0].set_xscale('log')
        axes_sud[0].set_xscale('log')

    for icol, beta in enumerate(BETAS):
        axes_rad[0].plot(pnts,
                         subRadAnalytic_fc_LL(pnts, beta=beta,
                                              jet_type=JET_TYPE),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$\beta=$'+str(beta))
        axes_sud[0].plot(pnts,
                         np.exp(-subRadAnalytic_fc_LL(pnts, beta=beta,
                                                      jet_type=JET_TYPE)),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$\beta=$'+str(beta))
    labelLines(axes_rad[0].get_lines(), xvals=x_vals)
    labelLines(axes_sud[0].get_lines(), xvals=x_vals)

    sub_sampler = ungroomedSampler(BIN_SPACE, epsilon=EPSILON)
    sub_sampler.generateSamples(NUM_MC_EVENTS)
    for icol, beta in enumerate(BETAS):
        rad = gen_numerical_radiator(sub_sampler, 'sub', JET_TYPE, ACC,
                                     beta=beta,
                                     bin_space=BIN_SPACE, epsilon=EPSILON,
                                     fixed_coupling=True,
                                     save=False,
                                     num_bins=NUM_BINS)
        axes_rad[0].plot(pnts, rad(pnts),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
        axes_sud[0].plot(pnts, np.exp(-rad(pnts)),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
    fig_rad.savefig('subrad_1.pdf', format='pdf')
    fig_sud.savefig('subsud_1.pdf', format='pdf')


###########################################
# Running Coupling
###########################################
# ------------------------------------
# Critical Test:
# ------------------------------------
def test_crit_rad_rc():
    fig_rad, axes_rad = aestheticfig(xlabel=r'$\theta$', ylabel=r'R($\theta$)',
                                     title='Critical '+JET_TYPE+' radiator, '
                                     + r'running $\alpha_s$',
                                     ylim=ylim_1,
                                     xlim=(EPSILON, 1.),
                                     ratio_plot=False, showdate=False)
    fig_sud, axes_sud = aestheticfig(xlabel=r'$\theta$',
                                     ylabel=r'$\Sigma(\theta$)',
                                     title = 'Critical '+JET_TYPE
                                     +' Sudakov Factor, '
                                     + r'running $\alpha_s$',
                                     ylim=ylim_sud,
                                     xlim=(EPSILON, 1.),
                                     ratio_plot=False, showdate=False)
    pnts = np.linspace(-.5, 1.5, 10000)

    if BIN_SPACE == 'log':
        pnts = np.logspace(np.log10(EPSILON), 0, 10000)
        axes_rad[0].set_xscale('log')
        axes_sud[0].set_xscale('log')
        if FUN_FIG:
            axes_fun[0].set_xscale('log')

    for icol, z_cut in enumerate(Z_CUTS):
        axes_rad[0].plot(pnts,
                         critRadAnalytic(pnts, z_cut, jet_type=JET_TYPE),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$z_c$='+str(z_cut))
        axes_sud[0].plot(pnts,
                         np.exp(-critRadAnalytic(pnts, z_cut,
                                                 jet_type=JET_TYPE)),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$z_c$='+str(z_cut))
    labelLines(axes_rad[0].get_lines(), xvals=x_vals)
    labelLines(axes_sud[0].get_lines(), xvals=x_vals)

    for icol, z_cut in enumerate(Z_CUTS):
        crit_sampler = criticalSampler(BIN_SPACE, zc=z_cut, epsilon=EPSILON)
        crit_sampler.generateSamples(NUM_MC_EVENTS)
        rad = gen_numerical_radiator(crit_sampler, 'crit', JET_TYPE, ACC,
                                     beta=None,
                                     bin_space=BIN_SPACE, epsilon=EPSILON,
                                     fixed_coupling=False,
                                     save=False)
        axes_rad[0].plot(pnts, rad(pnts),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
        axes_sud[0].plot(pnts, np.exp(-rad(pnts)),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
    fig_rad.savefig('critrad_1_rc.pdf', format='pdf')
    fig_sud.savefig('critsud_1_rc.pdf', format='pdf')

# ------------------------------------
# Critical Test:
# ------------------------------------
def test_sub_rad_rc():
    fig_rad, axes_rad = aestheticfig(xlabel=r'$C$', ylabel=r'R($C$)',
                                     title='Subsequent '+JET_TYPE+' radiator, '
                                     +r'running $\alpha_s$',
                                     ylim=(0, 1),
                                     xlim=(1e-5, 1.),
                                     ratio_plot=False, showdate=False)
    fig_sud, axes_sud = aestheticfig(xlabel=r'$C$',
                                     ylabel=r'$\Sigma(C$)',
                                     title='Subsequent '+JET_TYPE
                                     +' Sudakov Factor, '
                                     + r'running $\alpha_s$',
                                     ylim=(.9, 1.1),
                                     xlim=(1e-5, 1.),
                                     ratio_plot=False, showdate=False)

    pnts = np.linspace(-.5, 1., 10000)

    if BIN_SPACE == 'log':
        pnts = np.logspace(np.log10(EPSILON), np.log10(.5), 10000)
        axes_rad[0].set_xscale('log')
        axes_sud[0].set_xscale('log')

    for icol, beta in enumerate(BETAS):
        axes_rad[0].plot(pnts,
                         subRadAnalytic(pnts, beta=beta,
                                        jet_type=JET_TYPE),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$\beta=$'+str(beta))
        axes_sud[0].plot(pnts,
                         np.exp(-subRadAnalytic(pnts, beta=beta,
                                                jet_type=JET_TYPE)),
                         **style_solid,
                         color=compcolors[(icol,'light')],
                         label=r'$\beta=$'+str(beta))
    labelLines(axes_rad[0].get_lines(), xvals=x_vals)
    labelLines(axes_sud[0].get_lines(), xvals=x_vals)

    sub_sampler = ungroomedSampler(BIN_SPACE, epsilon=EPSILON)
    sub_sampler.generateSamples(NUM_MC_EVENTS)
    for icol, beta in enumerate(BETAS):
        rad = gen_numerical_radiator(sub_sampler, 'sub', JET_TYPE, ACC,
                                     beta=beta,
                                     bin_space=BIN_SPACE, epsilon=EPSILON,
                                     fixed_coupling=False,
                                     save=False,
                                     num_bins=NUM_BINS)
        axes_rad[0].plot(pnts, rad(pnts),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
        axes_sud[0].plot(pnts, np.exp(-rad(pnts)),
                         **style_dashed,
                         color=compcolors[(icol,'dark')])
    fig_rad.savefig('subrad_1_rc.pdf', format='pdf')
    fig_sud.savefig('subsud_1_rc.pdf', format='pdf')

# ------------------------------------
# Main:
# ------------------------------------
if __name__ == '__main__':
    if FUN_FIG:
        with plt.xkcd():
            test_crit_rad_fc()
    elif FIXED_COUPLING:
        test_crit_rad_fc()
        test_sub_rad_fc()
    else:
        test_crit_rad_rc()
        test_sub_rad_rc()
