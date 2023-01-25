import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Local imports
from jetmontecarlo.jets.jetSamplers import *
from jetmontecarlo.utils.plot_utils import *

# Parameters
showPlots = True
savePlots = False

# ---------------------------------------------------
# Plotting for Samplers:
# ---------------------------------------------------
def plotSamples(sampleMethod, samples, samplertype):
    if sampleMethod=='lin':   color='darkorchid'
    elif sampleMethod=='log': color='mediumseagreen'

    if samplertype=='crit':
        zs     = samples[:,0]
        thetas = samples[:,1]
        plt.scatter(zs,thetas,c=color,**style_scatter)

    if samplertype=='sub':
        if len(samples[0])==2:
            zs = samples[:,0]
            thetas = samples[:,1]
            plt.scatter(zs,thetas,c=color,**style_scatter)
        else:
            hist, bins  = np.histogram(samples, bins=100)
            xs   = (bins[1:]+bins[:-1])/2.
            xerr = (bins[1:]-bins[:-1])/2.
            plt.errorbar(xs,hist,xerr=xerr,c=color,**modstyle)
            plt.ylim((0.,np.max(hist)*1.1))

# ------------------------------------
# Jet Sampler Tests:
# ------------------------------------
def test_critLinSampling():
    zcuts    = [.01,.05,.1]
    testNums = [1000,5000,10000]

    for zcut in zcuts:
        # Instantiate the sampler
        testSampler = criticalSampler('lin',zc=zcut)

        # Check that the area is the expected area
        assert(testSampler.area == (1./2. - zcut))

        # Check that the correct number of samples are being generated
        # and that they reproduce the correct area
        for testNum in testNums:
            # Testing that we generate the correct number of samples
            testSampler.clearSamples()
            testSampler.generateSamples(testNum)
            assert(len(testSampler.getSamples()) == testNum)

            # Calculating the area we expect from these samples
            zmin     = np.min(testSampler.getSamples()[:,0])
            zmax     = np.max(testSampler.getSamples()[:,0])
            thetamin = np.min(testSampler.getSamples()[:,1])
            thetamax = np.max(testSampler.getSamples()[:,1])
            areaCalc = (zmax - zmin)*(thetamax - thetamin)

            # Verifying that these ranges and area are close to correct
            assert(abs(1. - (zmax-zmin)/(1./2.-zcut)) < .02)
            assert(abs(1. - (thetamax-thetamin)) < .02)
            assert(abs(1. - areaCalc/testSampler.area) < .02)

def test_critLogSampling():
    zcuts    = [.01,.05,.1]
    testNums = [1000,5000,10000]
    epsilons = [1e-3,1e-5,1e-10]

    for zcut in zcuts:
        for eps in epsilons:
            # Instantiate the sampler
            testSampler = criticalSampler('log',zc=zcut,epsilon=eps)

            # Check that the area is the expected area
            assert(testSampler.area == np.log(1./eps)**2.)

            # Check that the correct number of samples are being generated
            # and that they reproduce the correct area
            for testNum in testNums:
                # Testing that we generate the correct number of samples
                testSampler.clearSamples()
                testSampler.generateSamples(testNum)
                assert(len(testSampler.getSamples()) == testNum)

                # Calculating the area we expect from these samples
                zmin     = np.min(testSampler.getSamples()[:,0])
                zmax     = np.max(testSampler.getSamples()[:,0])
                logzprimemin = np.log(zmin - zcut)
                logzprimemax = np.log(zmax - zcut)
                thetamin = np.min(testSampler.getSamples()[:,1])
                thetamax = np.max(testSampler.getSamples()[:,1])
                logthetamin = np.log(thetamin); logthetamax = np.log(thetamax)
                areaCalc = (
                    (logzprimemax - logzprimemin)*(logthetamax - logthetamin)
                    )

                # Verifying that these ranges and area are close to correct
                assert(
                    abs(1. - (logzprimemax - logzprimemin) / (np.log(1./eps)))
                    < .01)
                assert(abs(1. - (logthetamax - logthetamin) / (np.log(1./eps)))
                    < .01)
                assert(abs(1. - areaCalc/testSampler.area) < .02)

def test_subLinSampling():
    testNums = [1000,5000,10000]

    # Instantiate the sampler, and check the sampling type
    testSampler = ungroomedSampler('lin')

    # Check that the area is the expected area
    assert(testSampler.area == 1./2.)

    # Check that the correct number of samples are being generated
    # and that they reproduce the correct area
    for testNum in testNums:
        # Testing that we generate the correct number of samples
        testSampler.clearSamples()
        testSampler.generateSamples(testNum)
        assert(len(testSampler.getSamples()) == testNum)

        # Calculating the area we expect from these samples
        zmin = np.min(testSampler.getSamples()[:,0])
        zmax = np.max(testSampler.getSamples()[:,0])
        thetamin = np.min(testSampler.getSamples()[:,1])
        thetamax = np.max(testSampler.getSamples()[:,1])
        areaCalc = (zmax-zmin)*(thetamax-thetamin)

        # Verifying that the range of cs
        assert(abs(1. - (zmax-zmin)/(1./2.)) < .02)
        assert(abs(1. - (thetamax-thetamin)) < .02)
        assert(abs(1. - areaCalc/testSampler.area) < .02)

def test_subLogSampling():
    testNums = [1000,5000,10000]
    epsilons = [1e-3,1e-5,1e-10]

    for eps in epsilons:
        # Instantiate the sampler, and check the sampling type
        testSampler = ungroomedSampler('log',epsilon=eps)

        # Check that the area is the expected area
        assert(testSampler.area == np.log(1./eps)**2.)

        # Check that the correct number of samples are being generated
        # and that they reproduce the correct area
        for testNum in testNums:
            # Testing that we generate the correct number of samples
            testSampler.clearSamples()
            testSampler.generateSamples(testNum)
            assert(len(testSampler.getSamples()) == testNum)

            # Calculating the area we expect from these samples
            logzmin = np.log(np.min(testSampler.getSamples()[:,0]))
            logzmax = np.log(np.max(testSampler.getSamples()[:,0]))
            logthetamin = np.log(np.min(testSampler.getSamples()[:,1]))
            logthetamax = np.log(np.max(testSampler.getSamples()[:,1]))
            areaCalc = (logzmax-logzmin)*(logthetamax-logthetamin)

            # Verifying that the range of cs
            assert(abs(1. - (logzmax-logzmin)/np.log(1./eps)) < .02)
            assert(abs(1. - (logthetamax-logthetamin)/np.log(1./eps)) < .02)
            assert(abs(1. - areaCalc/testSampler.area) < .02)

def test_fileSave():
    # Make and save samples
    zc = .1; eps = 1e-3
    testSampler = criticalSampler('log',zc=zc,epsilon=eps)
    testSampler.generateSamples(10000)
    testSampler.saveSamples('test')

    # Check that samples are saved
    outfile = "test.npz"
    assert os.path.exists(outfile)

    # Clear and load samples, check that they are correct
    testSampler.clearSamples()
    testSampler.loadSamples(outfile)
    assert(len(testSampler.getSamples()) == 10000)
    assert(testSampler.area == np.log(1./eps)**2.)

    # Remove the file
    os.remove(outfile)

def test_samplePlots():
    # Visualizing the sampled phase spaces:
    zc=.1; eps=1e-5

    # Initializing samplers of all types
    testSamplerCritLin = criticalSampler('lin',zc=zc)
    testSamplerCritLog = criticalSampler('log',zc=zc,epsilon=eps)
    testSamplerSubLin  = ungroomedSampler('lin')
    testSamplerSubLog  = ungroomedSampler('log',epsilon=eps)

    testSamplers = [testSamplerCritLin, testSamplerCritLog,
                    testSamplerSubLin, testSamplerSubLog]
    samplertypes = ['crit', 'crit', 'sub', 'sub']


    for i in range(len(testSamplers)):
        testSampler  = testSamplers[i]
        sampleMethod = testSampler.getSampleMethod()
        samplertype  = samplertypes[i]

        # Plotting the sampled points
        testSampler.generateSamples(10000)
        samples = testSampler.getSamples()

        if showPlots or savePlots:
            types = ['Critical', 'Ungroomed']
            zmin  = [zc, 0]
            title = (types[math.floor(i/2)] + " "
                    + sampleMethod + " samples")
            fig, axes = aestheticfig(
                            xlabel='z', ylabel=r'$\theta$',
                            title=title,
                            xlim=(zmin[math.floor(i/2)],1/2),
                            ylim=(0,1),
                            ratio_plot=False
                        )

            plotSamples(sampleMethod, samples, samplertype);
            if showPlots: plt.show()
            elif savePlots:
                filename = (samplertype+'Sampler_'+sampleMethod
                            +'_test.pdf')
                plt.savefig(filename)


# Testing whether invalid samplers break as expected
def test_invalidCritLogSampling():
    try:
        testSampler = criticalSampler('log',zc=.1)
        raise ValueError("Instantiated sampler without"
                            + "valid parameters.")
    except AssertionError:
        return

def test_invalidSubLogSampling():
    try:
        testSampler = ungroomedSampler('log')
        raise ValueError("Instantiated sampler without"
                            + "valid parameters.")
    except AssertionError:
        return

#########################################################
# Tests:
#########################################################
if __name__ == '__main__':
    # Main tests
    test_critLinSampling()
    test_critLogSampling()
    test_subLinSampling()
    test_subLogSampling()

    # Utils
    test_fileSave()
    test_samplePlots()

    # Testing invalid samplers
    test_invalidCritLogSampling()
    test_invalidSubLogSampling()
