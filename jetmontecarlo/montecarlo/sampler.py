from abc import ABC, abstractmethod
import numpy as np
from jetmontecarlo.utils.montecarlo_utils import *

########################################
# Base Sampler Class:
########################################
class sampler(ABC):
    """An abstract class designed to sample over phase space
    for Monte Carlo integration
    """
    # ------------------
    # Sampling Methods:
    # ------------------
    @abstractmethod
    def setArea(self):
        """Sets the area of the phase space associated
        with the sampler."""
        pass

    @abstractmethod
    def addSample(self):
        """Adds a single sample to the list self.samples,
        and adds the associated jacobian factor of integration
        to the list self.jacobians.

        Not implemented in the abstract base class."""
        pass

    def generateSamples(self, num):
        """Generates some number of samples for the sampler."""
        for _ in range(num):
            self.addSample()
        self.samples = np.array(self.samples)

    def getSamples(self):
        """Returns the list of samples."""
        return self.samples

    def getSampleMethod(self):
        """Returns the sample method."""
        return self.sampleMethod

    def clearSamples(self):
        """Clears the list of samples."""
        self.samples = []

    # ------------------
    # Saving/Loading:
    # ------------------
    def saveSamples(self, filename):
        """Saves the samples and area to filename.npz"""
        np.savez(filename, samples=self.samples, area=self.area)

    def loadSamples(self, filename):
        """Loads samples and area from filename.npz"""
        npzfile = np.load(filename, allow_pickle=True,
                          mmap_mode='c')
        self.samples = npzfile['samples']
        self.area = npzfile['area']

    # ------------------
    # Validity Checks:
    # ------------------
    def checkSampleMethod(self):
        """Checking that the monte carlo sample method is
        one of our supported sample methods."""
        validSampleMethods = ['lin', 'log']
        assert self.sampleMethod in validSampleMethods, \
            str(self.sampleMethod) + "is not a supported sample method."

    def checkValidEpsilon(self):
        """Checking that the keyword argument 'epsilon' is
        valid, if the sampling method is logarithmic."""
        eps = self.epsilon
        eps = 0 if eps is None else eps
        valid_eps = 0 < eps < 1.
        if self.sampleMethod == 'log':
            assert valid_eps, \
                ("epsilon={epsilon} is invalid for {type} sampling."
                 .format(epsilon=eps, type=self.sampleMethod))

    @abstractmethod
    # ------------------
    # Init:
    # ------------------
    def __init__(self, sampleMethod, epsilon=0.):
        """Prepares a MC sampler with
        * The sample method ('lin'/'log') for the sample phase space, and
        * any additional valid keyword arguments:
            epsilon -- the lower cutoff of the logarithmic sampling.
        """
        # Setting up:
        self.sampleMethod = sampleMethod
        self.epsilon = epsilon

        # Initializing samples:
        self.samples = []
        self.jacobians = []

        # Checking validity
        self.checkSampleMethod()
        self.checkValidEpsilon()

        self.setArea()


# ------------------------------------
# Simple sampler class:
# ------------------------------------
class simpleSampler(sampler):
    """A simple sampler which samples by default from 0 to 1."""
    # ------------------
    # Sampling:
    # ------------------
    def setArea(self):
        if self.sampleMethod=='lin':
            self.area = self.bounds[1] - self.bounds[0]
        elif self.sampleMethod=='log':
            self.area = np.log(1./self.epsilon)

    def addSample(self):
        if self.sampleMethod=='lin':
            sample = getLinSample(self.bounds[0], self.bounds[1])
            jac = 1.
        if self.sampleMethod=='log':
            sample = getLogSample(self.bounds[0], self.bounds[1], self.epsilon)
            jac = sample
        self.samples.append(sample)
        self.jacobians.append(jac)

    # ------------------
    # Init:
    # ------------------
    def __init__(self, sampleMethod, bounds=[0,1], **kwargs):
        self.bounds = bounds
        super().__init__(sampleMethod,**kwargs)
