import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Local imports
from jetmontecarlo.tests.simple_tests.test_simpleSampler import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.utils.color_utils import *

# Parameters
NUM_SAMPLES = int(1e4)
NUM_BINS = 10

EPSILON = 1e-5

showPlots = True
savePlots = False


def test_weight(x, y, n, m):
    weight = (n+1.)*x**n * (m+1.)*y**m
    return weight

# ------------------------------------
# Linear Integrators:
# ------------------------------------


def test_Simple2DLinIntegrator_firstbin(plot_2d=False):
    # Sampling
    testSampler_1 = simpleSampler('lin')
    testSampler_1.generateSamples(NUM_SAMPLES)
    samples_1 = testSampler_1.getSamples()

    testSampler_2 = simpleSampler('lin')
    testSampler_2.generateSamples(NUM_SAMPLES)
    samples_2 = testSampler_2.getSamples()

    # Setting up integrator
    testInt = integrator_2d()
    testInt.setFirstBinBndCondition(0.)
    testInt.setBins(NUM_BINS, [samples_1, samples_2], 'lin')

    for n in range(4):
        m = 1
        # Weights, binned observables, and area
        weights = test_weight(samples_1, samples_2, n, m)
        obs = [samples_1, samples_2]

        testInt.setDensity(obs, weights)
        testInt.integrate()

        integral = testInt.integral
        int_err = testInt.integralErr
        xs = testInt.bins[0][1:]
        ys = testInt.bins[1][1:]

        testInt.makeInterpolatingFn()
        interp_mc = testInt.interpFn

        integral_interp = interp_mc(xs, ys)
        xs, ys = np.meshgrid(xs, ys)
        zs = [integral, integral_interp, xs**(n+1) * ys**(m+1)]
        zs.append(abs(zs[0] - zs[1]))

        zlims = [(0, 1), (0, 1), (0, 1), (0, .1)]
        titles = ['Monte Carlo', 'Interpolation',
                  'Analytic', '|Difference|']

        projection = '3d'
        figsize = plt.figaspect(0.5)
        if plot_2d:
            projection = None
            figsize = (15, 4)

        fig = plt.figure(figsize=figsize)
        fig.suptitle('MC Integration to determine '
                     + 'x^{} y^{}'.format(n+1, m+1))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, projection=projection)
            ax.set_title(titles[i])
            if plot_2d:
                axes.append(ax)
                im = ax.pcolormesh(xs, ys, zs[i], vmin=0, vmax=1)
            else:
                my_col = cm.coolwarm(zs[i])
                ax.plot_surface(xs, ys, zs[i],
                                rstride=1, cstride=1,
                                facecolors=my_col,
                                linewidth=0, antialiased=False)
                ax.set_zlim(zlims[i])
                if i == 0 or i == 3:
                    # Plotting errorbars
                    fx = xs.flatten()
                    fy = ys.flatten()
                    fz = zs[i].flatten()
                    fzerr = int_err.flatten()
                    fcols = my_col.reshape(fx.shape[0], 4)
                    for j in np.arange(0, len(fx)):
                        ax.plot([fx[j], fx[j]], [fy[j], fy[j]],
                                [fz[j]+fzerr[j], fz[j]-fzerr[j]],
                                marker="|", color=fcols[j], zorder=5)

        if plot_2d:
            axes = np.array(axes)
            fig.colorbar(im, ax=axes.ravel().tolist())

        fig.savefig('simple_2d_lin_firstbin_test_'
                    + str(n+1) + '_' + str(m+1) + '.pdf',
                    format='pdf')


def test_Simple2DLinIntegrator_lastbin(plot_2d=False):
    # Sampling
    testSampler_1 = simpleSampler('lin')
    testSampler_1.generateSamples(NUM_SAMPLES)
    samples_1 = testSampler_1.getSamples()

    testSampler_2 = simpleSampler('lin')
    testSampler_2.generateSamples(NUM_SAMPLES)
    samples_2 = testSampler_2.getSamples()

    # Setting up integrator
    testInt = integrator_2d()
    testInt.setLastBinBndCondition([0., 'plus'])
    testInt.setBins(NUM_BINS, [samples_1, samples_2], 'lin')

    for n in range(4):
        m = 1
        # Weights, binned observables, and area
        weights = test_weight(samples_1, samples_2, n, m)
        obs = [samples_1, samples_2]

        testInt.setDensity(obs, weights)
        testInt.integrate()

        integral = testInt.integral
        int_err = testInt.integralErr
        xs = testInt.bins[0][:-1]
        ys = testInt.bins[1][:-1]

        testInt.makeInterpolatingFn()
        interp_mc = testInt.interpFn

        integral_interp = interp_mc(xs, ys)
        xs, ys = np.meshgrid(xs, ys)
        zs = [integral, integral_interp,
              (1-xs**(n+1)) * (1-ys**(n+1))]
        zs.append(abs(zs[0] - zs[1]))

        zlims = [(0, 1), (0, 1), (0, 1), (0, .1)]
        titles = ['Monte Carlo', 'Interpolation',
                  'Analytic', '|Difference|']

        projection = '3d'
        figsize = plt.figaspect(0.5)
        if plot_2d:
            projection = None
            figsize = (15, 4)

        fig = plt.figure(figsize=figsize)
        fig.suptitle('MC Integration to determine '
                     + '(1-x^{})(1-y^{})'.format(n+1, m+1))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, projection=projection)
            ax.set_title(titles[i])
            if plot_2d:
                axes.append(ax)
                im = ax.pcolormesh(xs, ys, zs[i], vmin=0, vmax=1)
            else:
                my_col = cm.coolwarm(zs[i])
                ax.plot_surface(xs, ys, zs[i],
                                rstride=1, cstride=1,
                                facecolors=my_col,
                                linewidth=0, antialiased=False)
                ax.set_zlim(zlims[i])
                if i == 0 or i == 3:
                    # Plotting errorbars
                    fx = xs.flatten()
                    fy = ys.flatten()
                    fz = zs[i].flatten()
                    fzerr = int_err.flatten()
                    fcols = my_col.reshape(fx.shape[0], 4)
                    for j in np.arange(0, len(fx)):
                        ax.plot([fx[j], fx[j]], [fy[j], fy[j]],
                                [fz[j]+fzerr[j], fz[j]-fzerr[j]],
                                marker="|", color=fcols[j], zorder=5)

        if plot_2d:
            axes = np.array(axes)
            fig.colorbar(im, ax=axes.ravel().tolist())

        fig.savefig('simple_2d_lin_lastbin_test_'
                    + str(n+1) + '_' + str(m+1) + '.pdf',
                    format='pdf')

# ------------------------------------
# Logarithmic Integrators:
# ------------------------------------


def test_Simple2DLogIntegrator_firstbin(plot_2d=False):
    # Sampling
    testSampler_1 = simpleSampler('log', epsilon=EPSILON)
    testSampler_1.generateSamples(NUM_SAMPLES)
    samples_1 = testSampler_1.getSamples()

    testSampler_2 = simpleSampler('log', epsilon=EPSILON)
    testSampler_2.generateSamples(NUM_SAMPLES)
    samples_2 = testSampler_2.getSamples()

    # Setting up integrator
    testInt = integrator_2d()
    testInt.setFirstBinBndCondition(0.)
    testInt.setBins(NUM_BINS, [samples_1, samples_2], 'log')

    for n in range(4):
        m = 1
        # Weights, binned observables, and area
        weights = test_weight(samples_1, samples_2, n, m)
        obs = [samples_1, samples_2]

        testInt.setDensity(obs, weights)
        testInt.integrate()

        integral = testInt.integral
        int_err = testInt.integralErr
        xs = testInt.bins[0][1:]
        ys = testInt.bins[1][1:]

        testInt.makeInterpolatingFn()
        interp_mc = testInt.interpFn

        integral_interp = interp_mc(xs, ys)
        xs, ys = np.meshgrid(xs, ys)
        zs = [integral, integral_interp, xs**(n+1) * ys**(m+1)]
        zs.append(abs(zs[0] - zs[1]))

        xs = np.log10(xs)
        ys = np.log10(ys)

        zlims = [(0, 1), (0, 1), (0, 1), (0, .1)]
        titles = ['Monte Carlo', 'Interpolation',
                  'Analytic', '|Difference|']

        projection = '3d'
        figsize = plt.figaspect(0.5)
        if plot_2d:
            projection = None
            figsize = (15, 4)

        fig = plt.figure(figsize=figsize)
        fig.suptitle('MC Integration to determine '
                     + 'x^{} y^{}'.format(n+1, m+1))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, projection=projection)
            ax.set_title(titles[i])

            if plot_2d:
                axes.append(ax)
                im = ax.pcolormesh(xs, ys, zs[i], vmin=0, vmax=1)
            else:
                my_col = cm.coolwarm(zs[i])
                ax.plot_surface(xs, ys, zs[i],
                                rstride=1, cstride=1,
                                facecolors=my_col,
                                linewidth=0, antialiased=False)
                ax.set_zlim(zlims[i])
                if i == 0 or i == 3:
                    # Plotting errorbars
                    fx = xs.flatten()
                    fy = ys.flatten()
                    fz = zs[i].flatten()
                    fzerr = int_err.flatten()
                    fcols = my_col.reshape(fx.shape[0], 4)
                    for j in np.arange(0, len(fx)):
                        ax.plot([fx[j], fx[j]], [fy[j], fy[j]],
                                [fz[j]+fzerr[j], fz[j]-fzerr[j]],
                                marker="|", color=fcols[j], zorder=5)

        if plot_2d:
            axes = np.array(axes)
            fig.colorbar(im, ax=axes.ravel().tolist())

        fig.savefig('simple_2d_log_firstbin_test_'
                    + str(n+1) + '_' + str(m+1) + '.pdf',
                    format='pdf')


def test_Simple2DLogIntegrator_lastbin(plot_2d=False):
    # Sampling
    testSampler_1 = simpleSampler('log', epsilon=EPSILON)
    testSampler_1.generateSamples(NUM_SAMPLES)
    samples_1 = testSampler_1.getSamples()

    testSampler_2 = simpleSampler('log', epsilon=EPSILON)
    testSampler_2.generateSamples(NUM_SAMPLES)
    samples_2 = testSampler_2.getSamples()

    # Setting up integrator
    testInt = integrator_2d()
    testInt.setLastBinBndCondition([0., 'plus'])
    testInt.setBins(NUM_BINS, [samples_1, samples_2], 'log')

    for n in range(4):
        m = 1
        # Weights, binned observables, and area
        weights = test_weight(samples_1, samples_2, n, m)
        obs = [samples_1, samples_2]

        testInt.setDensity(obs, weights)
        testInt.integrate()

        integral = testInt.integral
        int_err = testInt.integralErr
        xs = testInt.bins[0][:-1]
        ys = testInt.bins[1][:-1]

        testInt.makeInterpolatingFn()
        interp_mc = testInt.interpFn

        integral_interp = interp_mc(xs, ys)
        xs, ys = np.meshgrid(xs, ys)
        zs = [integral, integral_interp,
              (1-xs**(n+1)) * (1-ys**(m+1))]
        zs.append(abs(zs[0] - zs[1]))

        xs = np.log10(xs)
        ys = np.log10(ys)

        zlims = [(0, 1), (0, 1), (0, 1), (0, .1)]
        titles = ['Monte Carlo', 'Interpolation',
                  'Analytic', '|Difference|']

        projection = '3d'
        figsize = plt.figaspect(0.5)
        if plot_2d:
            projection = None
            figsize = (15, 4)

        fig = plt.figure(figsize=figsize)
        fig.suptitle('MC Integration to determine '
                     + '(1-x^{})(1-y^{})'.format(n+1, m+1))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, projection=projection)
            ax.set_title(titles[i])

            if plot_2d:
                axes.append(ax)
                im = ax.pcolormesh(xs, ys, zs[i], vmin=0, vmax=1)
            else:
                my_col = cm.coolwarm(zs[i])
                ax.plot_surface(xs, ys, zs[i],
                                rstride=1, cstride=1,
                                facecolors=my_col,
                                linewidth=0, antialiased=False)
                ax.set_zlim(zlims[i])
                if i == 0 or i == 3:
                    # Plotting errorbars
                    fx = xs.flatten()
                    fy = ys.flatten()
                    fz = zs[i].flatten()
                    fzerr = int_err.flatten()
                    fcols = my_col.reshape(fx.shape[0], 4)
                    for j in np.arange(0, len(fx)):
                        ax.plot([fx[j], fx[j]], [fy[j], fy[j]],
                                [fz[j]+fzerr[j], fz[j]-fzerr[j]],
                                marker="|", color=fcols[j], zorder=5)

        if plot_2d:
            axes = np.array(axes)
            fig.colorbar(im, ax=axes.ravel().tolist())

        fig.savefig('simple_2d_log_lastbin_test_'
                    + str(n+1) + '_' + str(m+1) + '.pdf',
                    format='pdf')


# Implementing tests
if __name__ == '__main__':
    test_Simple2DLinIntegrator_firstbin()
    test_Simple2DLinIntegrator_lastbin()

    test_Simple2DLogIntegrator_firstbin()
    test_Simple2DLogIntegrator_lastbin()
