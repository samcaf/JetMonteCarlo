import numpy as np
import matplotlib as plt

# Local imports
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.hist_utils import *

showPlots = True
savePlots = False
# ------------------------------------
# Histogram Derivative Test:
# ------------------------------------
def test_histDerivLin():
    bins = np.linspace(0,1,51)
    pnts = (bins[1:]+bins[:-1])/2.

    fig, axes = aestheticfig(
                            xlabel='x', ylabel='f(x)',
                            title='Hist Derivative Lin Test',
                            xlim=(0,1),ylim=(0,4),
                            ratio_plot=False
                            )

    axes[0].set_ylabel('f(x)', labelpad=10)

    for i in range(1,5):
        interp, hist = histDerivative(pnts**i, bins,
                                giveHist=True, binInput='lin')

        axes[0].scatter(pnts,hist,s=15, facecolors='none',
                    edgecolors=compcolors[(i,'dark')])

        label = r'd$x^{num}/$d$x$'.format(num=i)
        axes[0].plot(pnts,interp(pnts),**style_solid,
                    color=compcolors[(i,'dark')],
                    label=label)

        axes[0].plot(pnts,i * pnts**(i-1),**style_dashed,
                    color=compcolors[(i,'light')])


    # Legend/Labelling
    labelLines(axes[0].get_lines())

    custom_handles = [
                Line2D([0], [0], marker='o', markerfacecolor='none',
                      markeredgecolor='black', markersize=5, ls='none'),
                Line2D([0], [0], **style_solid,
                        color='black'),
                Line2D([0], [0], **style_dashed,
                        color='darkgray')
                    ]

    axes[0].legend(custom_handles,
                ['Finite Difference','Interpolation','Analytic'])

    if showPlots: plt.show()
    elif savePlots: plt.savefig('histDerivative_lin_test.pdf')


def test_histDerivLog():
    bins = np.logspace(-5,0,101)
    pnts = np.exp(
                (np.log(bins[1:])+np.log(bins[:-1]))/2.
                )

    fig, axes = aestheticfig(
                            xlabel='x', ylabel='f(x)',
                            title='Hist Derivative Log Test',
                            xlim=(1e-5,1),ylim=(0,4),
                            ratio_plot=False
                            )

    #fig.suptitle(r'Logarithmic Hist Derivative Test for $x^n$')
    axes[0].set_ylabel('f(x)', labelpad=10)

    axes[0].set_xscale('log')

    for i in range(1,5):
        interp, hist = histDerivative(pnts**i, bins,
                                giveHist=True, binInput='log')

        axes[0].scatter(pnts,hist,s=15, facecolors='none',
                    edgecolors=compcolors[(i,'dark')])

        label = r'd$x^{num}/$d$x$'.format(num=i)
        axes[0].plot(pnts,interp(pnts),**style_solid,
                    color=compcolors[(i,'dark')],
                    label=label)

        axes[0].plot(pnts,i * pnts**(i-1),**style_dashed,
                    color=compcolors[(i,'light')])


    # Legend/Labelling
    labelLines(axes[0].get_lines(),xvals=[1e-4,.06,.25,.5])

    custom_handles = [
                Line2D([0], [0], marker='o', markerfacecolor='none',
                      markeredgecolor='black', markersize=5, ls='none'),
                Line2D([0], [0], **style_solid,
                        color='black'),
                Line2D([0], [0], **style_dashed,
                        color='darkgray')
                    ]

    axes[0].legend(custom_handles,
                ['Finite Difference','Interpolation','Analytic'])

    if showPlots: plt.show()
    elif savePlots: plt.savefig('histDerivative_log_test.pdf')


#########################################################
# Tests:
#########################################################
if __name__ == '__main__':
    test_histDerivLin()
    test_histDerivLog()
