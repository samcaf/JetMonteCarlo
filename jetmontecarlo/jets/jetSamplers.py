import numpy as np

# Local imports
from jetmontecarlo.montecarlo.sampler import sampler
from jetmontecarlo.utils.montecarlo_utils import *

########################################
# Groomed Jet Sampler Classes:
########################################
# Samplers for the phase space of groomed jet emissions.

def checkValidzcut(zc):
    """Checking that the grooming parameter zc is valid."""
    zc = 0 if zc is None else zc
    valid_zc = 0 < zc < 1./2.
    assert valid_zc, \
        ("zc={zcut} is an invalid grooming parameter."
         .format(zcut=zc))

# --------------------------------------
# Sampler for the critical phase space:
# --------------------------------------
class criticalSampler(sampler):
    """A sampler which samples over the phase space of critical
    jet emissions. See the documentation of addSample for details.
    """
    # ------------------
    # Sampling Methods:
    # ------------------
    def setArea(self):
        """Sets the total area of the sampled phase space"""
        if self.sampleMethod == 'lin':
            self.area = (1./2. - self.zc) * self.radius
        elif self.sampleMethod == 'log':
            self.area = np.log(1./self.epsilon)**2.
        else: raise ValueError('Invalid sample method.')

    def addSample(self):
        """Adds a single sample to the list self.samples,
        and adds the associated jacobian factor of integration
        to the list self.jacobians.

        Samples from the critical phase space:
            - z_crit in (zc, 1/2),
            - theta_crit in (0,1).
        """
        if self.sampleMethod == 'lin':
            # Linear critical sampling:
            z, theta = (getLinSample(self.zc, 1./2.),
                        getLinSample(0, self.radius))
            jac = 1.

        elif self.sampleMethod == 'log':
            # Logarithmic critical sampling
            eps = self.epsilon
            z, theta = (getLogSample(self.zc, 1./2., eps),
                        getLogSample(0, self.radius, eps))
            jac = (z-self.zc) * theta
        else: raise ValueError('Invalid sample method.')

        # Adding the sample and jacobian
        self.samples.append([z, theta])
        self.jacobians.append(jac)

    # ------------------
    # Post-Processing:
    # ------------------
    def rescaleEnergies(self, zs):
        """
        Rescales the energy fractions associated with the emissions.
        Also rescales the jacobians associated with each emission,
        EXPLANATION

        We rescale the energy fractions of the emissions by
        the list (1-zs).
        This is important when taking multiple emissions into account;
        in this case the dominantly contributing, most energetic,
        critical emissions are emitted by the harder branch of a
        previous, pre-critical emission.
        Their energy fraction should thus be rescaled, by a factor
        (1-z_{soft, previous}).

        For example, zs = 0 does not modify the emissions at all;
        this corresponds to a calculation at leading log accuracy.

        Parameters
        ----------
        zs : type
            A set of energy fractions.
        """
        self.samples[:, 0] = (1. - zs) * self.samples[:, 0]

        assert False, "Incomplete rescaling!"
        self.jacobians = self.jacobians * (1. - zs)

    # ------------------
    # Init:
    # ------------------
    def __init__(self, sampleMethod, zc, epsilon=0., radius=1.):
        """Prepares a MC sampler which samples over the phase space
        of a ``critical'' emission.

        Parameters
        ----------
        sampleMethod : str
            The sample method ('lin'/'log') for the sample phase space.
        zc : float
            The minimum value of the energy fraction z of the critical
            emission.
        epsilon : float
            The minimum cutoff of the logarithmic sampling.
            Only used if sampleMethod is 'log'
        """
        self.zc = zc
        checkValidzcut(zc)

        assert radius > 0, "Jet radius must be greater than zero!"
        self.radius = radius

        super().__init__(sampleMethod, epsilon)


# --------------------------------------
# Sampler for the pre-critical phase space:
# --------------------------------------
class precriticalSampler(sampler):
    """A sampler which samples over the phase space of pre-critical
    jet emissions. See the documentation of addSample for details.
    """
    # ------------------
    # Sampling Methods:
    # ------------------
    def setArea(self):
        """Sets the total area of the sampled phase space"""
        if self.sampleMethod == 'lin':
            self.area = self.zc
        elif self.sampleMethod == 'log':
            self.area = np.log(1./self.epsilon)
        else: raise ValueError('Invalid sample method.')

    def addSample(self):
        """Adds a single sample to the list self.samples,
        sampling from the pre-critical phase space:
            - z_pre in (0, zc)
        """
        if self.sampleMethod == 'lin':
            # Linear pre-critical sampling:
            zpre = getLinSample(0., self.zc)
            jac = 1.

        elif self.sampleMethod == 'log':
            # Logarithmic pre-critical sampling
            eps = self.epsilon
            zpre = getLogSample(0., self.zc, eps)
            jac = zpre
        else: raise ValueError('Invalid sample method.')

        # Adding the sample and jacobian
        self.samples.append(zpre)
        self.jacobians.append(jac)

    # ------------------
    # Init:
    # ------------------
    def __init__(self, sampleMethod, zc, epsilon=0.):
        """Prepares a MC sampler which samples over the phase space
        of a ``critical'' emission.

        Parameters
        ----------
        sampleMethod : str
            The sample method ('lin'/'log') for the sample phase space.
        zc : float
            The minimum value of the energy fraction z of the critical
            emission.
        epsilon : float
            The minimum cutoff of the logarithmic sampling.
            Only used if sampleMethod is 'log'
        """
        self.zc = zc
        checkValidzcut(zc)
        super().__init__(sampleMethod, epsilon)

########################################
# Ungroomed Jet Sampler Class:
########################################
# Associated with ungroomed/subsequent jet emissions,
# relevant for either entirely ungroomed jets, or for
# emissions that occur after the grooming process.

class ungroomedSampler(sampler):
    """A sampler which samples over the phase space of subsequent
    jet emissions. See the documentation of addSample for details.
    """
    # ------------------
    # Sampling Methods:
    # ------------------
    def setArea(self):
        """Sets the total area of the sampled phase space"""
        if self.sampleMethod == 'lin':
            self.area = self.radius/2.
        elif self.sampleMethod == 'log':
            self.area = np.log(1./self.epsilon)**2.
        else: raise ValueError('Invalid sample method.')

    def addSample(self):
        """Adds a single sample to the list self.samples,
        sampling in the subsequent phase space
            - z_sub in (0, 1/2),
            - theta_sub in (0,1).
        """
        if self.sampleMethod == 'lin':
            # Linear critical sampling:
            z, theta = (getLinSample(0., 1./2.),
                        getLinSample(0, self.radius))
            jac = 1.

        elif self.sampleMethod == 'log':
            # Logarithmic ungroomed sampling
            eps = self.epsilon
            z, theta = (getLogSample(0., 1./2., eps),
                        getLogSample(0, self.radius, eps))
            jac = z * theta
        else: raise ValueError('Invalid sample method.')

        # Adding the sample and jacobian
        self.samples.append([z, theta])
        self.jacobians.append(jac)

    # ------------------
    # Post-Processing:
    # ------------------
    def rescaleEnergies(self, zs):
        """
        Rescales the energy fractions associated with the emissions.
        Also rescales the jacobians associated with each emission,
        EXPLANATION

        We rescale the energy fractions of the emissions by
        the list (1-zs).
        This is important when taking multiple emissions into account;
        in this case the dominantly contributing, most energetic,
        ungroomed emissions are emitted by the harder branch of a
        previous emission.
        Their energy fraction should thus be rescaled, by a factor
        (1-z_{soft, previous}).

        For example, zs = 0 does not modify the emissions at all;
        this corresponds to a calculation at leading log accuracy.

        Parameters
        ----------
        zs : type
            A set of energy fractions.
        """
        self.samples[:, 0] = (1. - zs) * self.samples[:, 0]

        assert False, "Incomplete rescaling!"
        self.jacobians = self.jacobians * (1. - zs)

    def rescaleAngles(self, theta_maxes):
        """
        Rescales the angles associated with the emissions.
        Also rescales the jacobians associated with each emission,
        EXPLANATION

        We rescale the angles of the emissions directly by the
        list thetas.
        This list could correspond to, for example, the angles
        of previous emissions. This procedure can then be used to
        enforce angular ordering. In particular, the emissions which
        are unaffected by a grooming procedure will have to be
        narrower than those that are groomed).

        Parameters
        ----------
        thetas : type
            A list of angles by which we rescale the angles of
            the ungroomed emissions.
        """
        self.samples[:, 1] = theta_maxes * self.samples[:, 1]

        if self.sampleMethod == 'log':
            self.jacobians = self.jacobians * theta_maxes
        assert False, "Incomplete rescaling!"

    # ------------------
    # Init:
    # ------------------
    def __init__(self, sampleMethod, epsilon=0., radius=1.):
        """Prepares a MC sampler which samples over the phase space
        of a ``subsequent'' emission.

        Parameters
        ----------
        sampleMethod : str
            The sample method ('lin'/'log') for the sampled phase space.
        epsilon : float
            The minimum cutoff of the logarithmic sampling.
            Only used if sampleMethod is 'log'
        """
        self.radius = radius
        assert radius > 0, "Jet radius must be greater than zero!"
        super().__init__(sampleMethod, epsilon)
