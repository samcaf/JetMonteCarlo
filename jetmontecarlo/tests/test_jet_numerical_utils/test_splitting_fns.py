import dill as pickle
from pathlib import Path

# Local utilities for comparison
from examples.comparison_plots.comparison_plot_utils import *

# ------------------------------------
# Parameters
# ------------------------------------
# MC events
NUM_MC_EVENTS = int(1e3)
NUM_SPLITFN_BINS = int(1e2)

THETAS = [.03, .1, 0.3, 1.]

Z_CUT = .1
Z_CUTS = [.003, .01, .03, .1, .2]

splitfn_folder = Path("jetmontecarlo/utils/functions/splitting_fns/")

splitfn_file = 'split_fns_{:.0e}events_{:.0e}bins.pkl'.format(NUM_MC_EVENTS,
                                                              NUM_SPLITFN_BINS)
if not FIXED_COUPLING:
    splitfn_file = 'split_fns_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                                NUM_MC_EVENTS,
                                                                NUM_SPLITFN_BINS)
splitfn_path = splitfn_folder / splitfn_file
with open(splitfn_path, 'rb') as f:
    SPLITTING_FNS = pickle.load(f)

# Index of z_cut values in the splitting function file
index_zc = {.05: 0, .1: 1, .2: 2}

def split_fn_an(z, theta, z_cut):
    if FIXED_COUPLING:
        return -1 / (z * np.log(2.*z_cut)) * (z_cut < z) * (z < 1./2.)

    nga = 2*alpha_fixed*beta_0
    tfactor = 1+nga*np.log(theta)
    logfactor = np.log((tfactor+nga*np.log(1./2.))/(tfactor+nga*np.log(z_cut)))
    return alpha_s(z, theta) * nga / (z * alpha_fixed * logfactor)\
           * (z_cut < z) * (z < 1./2.)

def split_fn_num(z, theta, z_cut):
    return SPLITTING_FNS[index_zc[z_cut]](z, theta)

def test_splitfn_norm():
    for z_cut in [.05, .1, .2]:
        for theta in THETAS:
            def splitfn(z): return split_fn_num(z, theta, z_cut)
            integral, _, _ = integrate_1d(splitfn, [0,1./2.])
            if np.abs(integral-1.) > 3e-1:
                print("WARNING: Splitting function has the poor normalization "
                      + str(integral))

def test_splitfn_plot():
    print("Preparing to plot fixed coupling splitting function...")
    # Preparing figures
    fig_rad, axes_rad, _, _ = \
            get_axes('pre and crit', ratio_plot=False)
    axes_rad[0].set_ylim(-.5, 10)
    axes_rad[0].set_xlim(1e-20, 1)

    # Plotting
    for icol, radius in enumerate(THETAS[:-1]):
        if BIN_SPACE == 'log':
            bins = np.logspace(-20, 0, 1000)
            xs = np.sqrt(bins[:-1]*bins[1:])
            label_xvals = np.logspace(-7,-2,len(THETAS))
        elif BIN_SPACE == 'lin':
            bins = np.linspace(0, .6, 1000)
            xs = (bins[:-1]+bins[1:])/2.
            label_xvals = np.linspace(0, .5, len(THETAS))
        fn_an = split_fn_an(xs, radius, Z_CUT)
        fn_num = split_fn_num(xs, radius, Z_CUT)

        # Plotting
        col_an = compcolors[(icol%4, 'light')]
        col_num = compcolors[(icol%4, 'dark')]

        axes_rad[0].plot(xs, fn_num, **style_solid,
                         color=col_num)
        axes_rad[0].plot(xs, fn_an, **style_dashed,
                         color=col_an, label=r'$\theta$='+str(radius),
                         zorder=5)

    labelLines(axes_rad[0].get_lines(), xvals=label_xvals)

    legend_darklight(axes_rad[0], darklabel='Numerical',
                     lightlabel='Analytic')

    fig_rad.savefig(JET_TYPE+'_split_fn_fc_test.pdf',
                    format='pdf')
    print("Plotting complete!")

if __name__ == '__main__':
    test_splitfn_norm()
    test_splitfn_plot()
