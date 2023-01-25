import numpy as np
from scipy import interpolate
import dill

# Local utils
from jetmontecarlo.utils.hist_utils import histDerivative
from jetmontecarlo.montecarlo.sampler import simpleSampler
from jetmontecarlo.utils.interpolation_function_utils import get_1d_interpolation
from jetmontecarlo.utils.interpolation_function_utils import get_2d_interpolation

MIN_LOG_BIN = 1e-15


#######################################
# Integrator Class:
#######################################
class integrator():
    """A Monte Carlo integrator designed to produce probability and
    cumulative distributions for observables on phase space.

    In order to perform the integration, it works together with:
        - A sampler class which gives us
            - A set of samples on the phase space.
            - A set of observables associated with those samples.
            - A jacobian associated with the sampling/integration
        - The functional form of a weight/probability distribution
        on the phase space, in terms of the sampled variables.
    """

    # ------------------
    # Integration Tools:
    # ------------------
    def setBins(self, numbins, observables, binspacing,
                min_log_bin=MIN_LOG_BIN):
        """Sets the bins to be used in weight histograms, pdfs, etc."""
        if binspacing == 'lin':
            self.bins = np.linspace(0, np.max(observables), numbins)
        elif binspacing == 'log':
            min_bin = np.log10(max(np.min(observables), min_log_bin))
            max_bin = np.log10(np.max(observables))
            self.bins = np.logspace(min_bin, max_bin, numbins)
        self.binspacing = binspacing
        self.hasBins = True

    def setLastBinBndCondition(self, bndCond):
        """Sets a boundary condition on the value of the integral at the
        rightmost bin.
        """
        self.lastBinBndCond = bndCond
        self.useLastBinBndCond = True

    def setFirstBinBndCondition(self, bndCond):
        """Sets a boundary condition on the value of the integral at the
        rightmost bin.
        """
        self.firstBinBndCond = bndCond
        self.useFirstBinBndCond = True

    def setMonotone(self, monotone=True):
        """If monotone is true, the integrator will enforce monotonicity
        of the integral and any interpolating functions.
        """
        self.monotone = monotone


    # ------------------
    # Actual Integration:
    # ------------------
    def setDensity(self, observables, weights, area):
        """Sets a distribution/density on the space defined by the observable
        which can later be integrated. Also finds the error associated with this
        distribution, and sets the bins
        as a list of bin edges, if self.bins is only the number of requested
        bins.
        """
        # One needs bins to produce the density
        assert self.hasBins, "Need bins to produce density histogram."

        # Producing histograms of weights, binned in the given observable
        weightHist, _ = np.histogram(observables, self.bins,
                                     weights=weights)
        square_weightHist, _ = np.histogram(observables, self.bins,
                                            weights=np.square(weights))
        num_in_bin, _ = np.histogram(observables, self.bins)

        # Here we calculate the weighted distribution of the observable:
        # density(bin) = \int dx weightFn(A) delta(bin - observable(A))
        #           ~ (average weight in bin) * dx_bin / binWidth

        # We proceed by finding
        # (average weight in bin) = (total weight in bin) / num_in_bin
        # dA_bin = (total area) * num_in_bin / num_total

        # Therefore,
        # density ~ (total weight in bin / binWidth)
        #           * (total area)/num_total

        binWidths = (self.bins[1:]-self.bins[:-1])

        self.density = ((weightHist/binWidths)
                        * (area/len(observables)))
        self.densityErr = ((np.sqrt(square_weightHist)/binWidths)
                           * (area/len(observables)))

        # Setting x values for the density as well
        if self.binspacing == 'lin':
            self.density_xs = (self.bins[1:] + self.bins[:-1])/2.
        elif self.binspacing == 'log':
            self.density_xs = np.exp((np.log(self.bins[1:])
                         +np.log(self.bins[:-1]))/2.)

        # No points in a given bin -> no information about that bin:
        self.density = np.where(num_in_bin == 0, 0, self.density)
        self.densityErr = np.where(num_in_bin == 0, 0, self.densityErr)

        # Letting the integrator know that it has a weight density:
        self.hasMCDensity = True

    def integrate(self):
        """Integrates by assuming the final value of the integral and
        working backwards. Roughly,
        integral at point p =
            boundary condition for the integral
            - integral from p up to the boundary condition
        """
        # One needs bins to produce the density
        assert self.hasMCDensity, \
            "Need density function to perform integration"

        # Finding bin widths for integration
        binWidths = (self.bins[1:]-self.bins[:-1])

        self.useLastBinBndCond = (
            not(self.lastBinBndCond is None)
            and self.firstBinBndCond is None
            )
        self.useFirstBinBndCond = (
            not(self.firstBinBndCond is None)
            and self.lastBinBndCond is None
            )

        ambiguousBC = (
            not(self.useLastBinBndCond or self.useFirstBinBndCond)
            or
            (self.useLastBinBndCond and self.useFirstBinBndCond)
            )
        assert not(ambiguousBC), \
            "Integration requires unambiguous boundary conditions"

        if self.useFirstBinBndCond:
            # Finding the integral from the boundary condition up to a bin
            cumInt = np.cumsum(self.density*binWidths)

            # Finding the value of the integral up to each bin
            self.integral = self.firstBinBndCond + cumInt

            # Storing the x values of the integral
            self.integral_xs = self.bins[1:]

            # Finding the error associated with this integration procedure
            self.integralErr = np.cumsum(self.densityErr*binWidths)


        elif self.useLastBinBndCond:
            # Finding the integral from a bin up to the boundary condition
            reverse_cumInt = np.cumsum((self.density*binWidths)[::-1])[::-1]

            # Finding the value of the integral up to each bin
            assert self.lastBinBndCond[1] in ['plus', 'minus'], \
                "Integration requires lastBinBndCond (plus or minus)"
            if self.lastBinBndCond[1] == 'plus':
                self.integral = self.lastBinBndCond[0] + reverse_cumInt
            elif self.lastBinBndCond[1] == 'minus':
                self.integral = self.lastBinBndCond[0] - reverse_cumInt

            # Storing the x values of the integral
            self.integral_xs = self.bins[:-1]

            # Finding the error associated with this integration procedure
            self.integralErr = np.cumsum(
                (self.densityErr*binWidths)[::-1]
                )[::-1]

        # Telling the integrator that it has an integral evaulated with MC:
        self.hasMCIntegral = True

        # Checking for monotonicity, if specified by user:
        if self.monotone:
            is_monotone = ((self.integral[1:] <= self.integral[:-1]).all() or
                            (self.integral[1:] >= self.integral[:-1]).all())
            assert is_monotone, "User asked for a monotone integral,"\
                " but the integral is not monotone!"

    def montecarlo_data_dict(self, info=None):
        return {'bins': self.bins,
                'density': self.density,
                'density xs': self.density_xs,
                'density error': self.densityErr,
                'integral': self.integral,
                'integral xs:': self.integral_xs,
                'integral error': self.integralErr,
                'info': info}

    def save_montecarlo_data(self, filename, info=None):
        """Saves the data used to produce the integral to a file"""
        assert self.hasMCDensity and self.hasMCIntegral, \
            "Need density function and integral to save MC data"

        # Saving the data to a file
        np.savez(filename,
                 **self.monte_carlo_data_dict(info=info))

    # ------------------
    # Integral Interpolation:
    # ------------------
    def makeInterpolatingFn(self, **kwargs):
        """Makes an interpolating function for the integral. Assumes linear
        binning."""
        assert self.hasMCIntegral, \
            "Need MC integral to produce interpolation"

        self.interpFn = get_1d_interpolation(self.integral_xs,
                                             self.integral,
                                             monotone=self.monotone,
                                             **kwargs)

        self.hasInterpIntegral = True

    def saveInterpolatingFn(self, fileName, save_integral=True):
        """Saves the interpolating function for the integral
        to the file fileName"""
        assert self.hasInterpIntegral, \
            "No interpolating function to save"
        file = open(fileName, 'wb')
        dill.dump(self.interpFn, file)

    def loadInterpolatingFn(self, fileName):
        """Loads an interpolating function for an integral
        from a file fileName"""
        file = open(fileName, 'rb')
        self.interpFn = dill.load(file)
        self.hasInterpIntegral = True

    # ------------------
    # Density Interpolation:
    # ------------------
    def makeInterpolatingDensity(self, binspacing, monotone=False):
        """Makes an interpolating function for the density."""
        assert self.hasMCDensity, \
            "Need MC density to produce interpolation"
        bins = self.bins

        self.interpDensity = get_1d_interpolation(xs, self.density,
                                             monotone=False,
                                             bounds=(bins[0], bins[-1]),
                                             bound_values=(0., 0.))
        self.hasInterpDensity = True

    def saveInterpolatingDensity(self, fileName):
        """Saves an interpolating function for an
        density/integrand to a file fileName"""
        assert self.hasInterpDensity, \
            "No interpolating function to save"
        file = open(fileName, 'wb')
        dill.dump(self.interpDensity, file)

    def loadInterpolatingDensity(self, fileName):
        """Loads an interpolating function for an
        density/integrand from a file fileName"""
        file = open(fileName, 'rb')
        self.interpDensity = dill.load(file)
        self.hasInterpDensity = True

    # ------------------
    # Analytic Results
    # ------------------
    def findAnalyticIntegral(self):
        """Searches for an analytic form of the integral"""
        # No default analytic integral:
        self.hasAnalyticIntegral = False

    def analyticIntegral(self, obs):
        """Provides an analytic integral at a given value of the relevant
        observable."""
        if not self.hasAnalyticIntegral:
            return
        pass

    def setAnalyticDensity(self, binspacing):
        """Finds and sets the analytic integrand/density function,
        self.analyticDensity(observable), by finding a finite
        difference derivative of the analytic integral function.
        """
        if not self.hasAnalyticIntegral:
            return

        xs = self.density_xs

        self.analyticDensity = histDerivative(self.analyticIntegral(xs),
                                              bins, giveHist=False,
                                              binInput=binspacing)
        self.hasAnalyticDensity = True

    # ------------------
    # Saving/Loading Data:
    # ------------------
    def saveIntegral(self, intFile, errFile, binFile):
        """Saves the integral, error, and bins to
        intFile, errFile, and binFile"""
        assert self.hasMCIntegral, \
            "No integral histogram to save"
        np.savetxt(intFile, self.integral, delimiter=',')
        np.savetxt(errFile, self.integralErr, delimiter=',')
        np.savetxt(binFile, self.bins, delimiter=',')

    def loadIntegral(self, intFile, errFile, binFile):
        """Saves an integral, error, and bins from
        intFile, errFile, and binFile"""
        self.integral = np.loadtxt(intFile, delimiter=',')
        self.integralErr = np.loadtxt(errFile, delimiter=',')
        self.bins = np.loadtxt(binFile, delimiter=',')

        self.hasMCIntegral = True

    def saveDensity(self, denFile, errFile, binFile):
        """Saves the density, error, and bins to
        denFile, errFile, and binFile"""
        assert self.hasMCDensity, \
            "No density histogram to save"
        np.savetxt(denFile, self.density, delimiter=',')
        np.savetxt(errFile, self.densityErr, delimiter=',')
        np.savetxt(binFile, self.bins, delimiter=',')

    def loadDensity(self, denFile, errFile, binFile):
        """Loads density, error, and bins from
        denFile, errFile, and binFile"""
        self.density = np.loadtxt(denFile, delimiter=',')
        self.densityErr = np.loadtxt(errFile, delimiter=',')
        self.bins = np.loadtxt(binFile, delimiter=',')

        self.hasMCDensity = True

    # ------------------
    # Validity Checks:
    # ------------------
    def checkValidAttributes(self):
        """Checks that the parameters given to the sampler are valid
        """
        pass

    # ------------------
    # Init:
    # ------------------
    def __init__(self):
        """Initializes the integrator class by letting it know that it has
        no information yet. This is fixed when information is added to the
        integrator. Detailed documentation is included in the README_utils
        file.
        """
        self.hasBins = False
        self.hasMCDensity = False
        self.hasMCIntegral = False
        self.hasInterpIntegral = False
        self.hasInterpDensity = False
        self.firstBinBndCond = None
        self.useFirstBinBndCond = False
        self.lastBinBndCond = None
        self.useLastBinBndCond = False
        self.monotone = False
        self.hasAnalyticIntegral = False
        self.hasAnalyticDensity = False

        self.checkValidAttributes()



def integrate_1d(function, bounds,
                 bin_space='lin', epsilon=1e-10,
                 num_samples=1e5, num_bins=2,
                 bnd_cond=0, bnd_cond_bin='first'):
    """Performs a 1d integral over the given bounds.

    Parameters
    ----------
    function : function
        The function to be integrated.
    bounds : list
        The bounds of integration.
    bin_space : string
        The space over which we sample. Must be `lin` or `log`
    epsilon : float
        The cutoff for logarithmic sampling.
    num_samples : int
        Number of Monte Carlo samples for the integration
    num_bins : int
        Number of bins into which we divide the integration region.
        Default is 2, which leads to a single number for the integral.
    bnd_cond : float
        The boundary condition for the integral. For example, when integrating
        to find a cumulative distribution function, we want a result that
        yields 1 at the final bin.
    bnd_cond_bin :
        The bin at which the boundary condition is applied.
        Must be `first` or `last`.

    Returns
    -------
    integral, integral_error, xs
        The integral values and integral errors corresponding to given
        values of the argument of the integral.
    """
    if bin_space == 'lin':
        epsilon = None
    # Sampling
    this_sampler = simpleSampler(bin_space, epsilon=epsilon, bounds=bounds)
    this_sampler.generateSamples(int(num_samples))
    samples = this_sampler.getSamples()

    # Setting up integrator
    this_integrator = integrator()
    if bnd_cond_bin == 'first':
        this_integrator.setFirstBinBndCondition(bnd_cond)
    elif bnd_cond_bin == 'last':
        this_integrator.setLastBinBndCondition(bnd_cond)
    else:
        raise AssertionError("Bin at which we place a boundary condition"
                             +"must be 'first' or 'last'.")

    this_integrator.setBins(num_bins, samples, bin_space)

    weights = function(samples)
    jacs = this_sampler.jacobians
    area = this_sampler.area

    this_integrator.setDensity(samples, weights * jacs, area)
    this_integrator.integrate()

    integral = this_integrator.integral
    error = this_integrator.integralErr
    xs = this_integrator.integral_xs

    if num_bins == 2:
        return integral[0], error[0], xs[0]
    return integral, error, xs


#######################################
# 2 dimensional Integrator Class:
#######################################
def multidim_cumsum(a):
    out = a.cumsum(-1)
    for i in range(2,a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

class integrator_2d():
    """A Monte Carlo integrator designed to produce probability and
    cumulative distributions for observables on phase space.

    In order to perform the integration, it works together with:
        - A sampler class which gives us
            - A set of samples on the phase space.
            - A set of observables associated with those samples.
            - A jacobian associated with the sampling/integration
        - The functional form of a weight/probability distribution
        on the phase space, in terms of the sampled variables.
    """
    # ------------------
    # Integration Tools:
    # ------------------
    def setBins(self, numbins, observables, binspacing,
                min_log_bin=MIN_LOG_BIN):
        """Sets the bins to be used in weight histograms, pdfs, etc."""
        bins = []
        for i in range(2):
            if binspacing == 'lin':
                bins.append(np.linspace(0, np.max(observables[i]), numbins))
            elif binspacing == 'log':
                min_bin = np.log10(max(np.min(observables[i]), min_log_bin))
                max_bin = np.log10(np.max(observables[i]))
                bins.append(np.logspace(min_bin, max_bin, numbins))
        self.bins = bins
        self.hasBins = True

    def setLastBinBndCondition(self, bndCond):
        """Sets a boundary condition on the value of the integral at the
        'rightmost' and 'upper' bins."""
        self.lastBinBndCond = bndCond
        self.useLastBinBndCond = True

    def setFirstBinBndCondition(self, bndCond):
        """Sets a boundary condition on the value of the integral at the
        'leftmost' and 'bottom' bins."""
        self.firstBinBndCond = bndCond
        self.useFirstBinBndCond = True

    # ------------------
    # Actual Integration:
    # ------------------
    def setDensity(self, observables, weights, area):
        """Sets a distribution/density on the space defined by the observable
        which can later be integrated. Also finds the error associated with this
        distribution, and sets the bins
        as a list of bin edges, if self.bins is only the number of requested
        bins.
        """
        # One needs bins to produce the density
        assert self.hasBins, "Need bins to produce density histogram."

        # Producing histograms of weights, binned in the given observable
        weightHist, _, _ = np.histogram2d(observables[0], observables[1],
                                          self.bins, weights=weights)
        square_weightHist, _, _ = np.histogram2d(observables[0], observables[1],
                                                 self.bins,
                                                 weights=np.square(weights))
        num_in_bin, _, _ = np.histogram2d(observables[0], observables[1],
                                          self.bins)

        # density(bin) = \int dA weightFn(A) delta(bin - observable(A))
        #           ~ (average weight in bin) * dA_bin / binArea

        # We proceed by finding
        # (average weight in bin) = (total weight in bin) / num_in_bin
        # dA_bin = (total area) * num_in_bin / num_total

        # Therefore,
        # density ~ (total weight in bin / binWidth)
        #           * (total area)/num_total

        binWidths_1 = self.bins[0][1:]-self.bins[0][:-1]
        binWidths_2 = self.bins[1][1:]-self.bins[1][:-1]
        binWidths_1, binWidths_2 = np.meshgrid(binWidths_1, binWidths_2)
        binAreas = binWidths_1 * binWidths_2
        # np.histogram2d confusingly gives the transpose of what
        # you might naively expect; taking this into account:
        binAreas = binAreas.T

        areafactor = area / (len(observables[0]) * binAreas)

        self.density = weightHist * areafactor
        self.densityErr = np.sqrt(square_weightHist) * areafactor


        # No points in a given bin -> no information about that bin:
        self.density = np.where(num_in_bin == 0, 0, self.density)
        self.densityErr = np.where(num_in_bin == 0, 0, self.densityErr)

        # According to the histogram2d documentation, need to transpose
        self.density = self.density.T
        self.densityErr = self.densityErr.T

        # Getting the x and y values associated with the density
        self.density_xs, self.density_ys = self.bins[0], self.bins[1]

        # Letting the integrator know that it has a weight density:
        self.hasMCDensity = True

    def integrate(self):
        """Integrates by assuming the final value of the integral and
        working backwards. Roughly,
        integral at point p =
            boundary condition for the integral
            - integral from p up to the boundary condition
        """
        # One needs bins to produce the density
        assert self.hasMCDensity, \
            "Need density function to perform integration"

        # Finding bin areas for integration
        binWidths_1 = self.bins[0][1:]-self.bins[0][:-1]
        binWidths_2 = self.bins[1][1:]-self.bins[1][:-1]
        binWidths_1, binWidths_2 = np.meshgrid(binWidths_1, binWidths_2)
        binAreas = binWidths_1 * binWidths_2

        self.useLastBinBndCond = (
            not(self.lastBinBndCond is None)
            and self.firstBinBndCond is None
            )
        self.useFirstBinBndCond = (
            not(self.firstBinBndCond is None)
            and self.lastBinBndCond is None
            )

        ambiguousBC = (
            not(self.useLastBinBndCond or self.useFirstBinBndCond)
            or
            (self.useLastBinBndCond and self.useFirstBinBndCond)
            )
        assert not(ambiguousBC), \
            "Integration requires unambiguous boundary conditions"

        if self.useFirstBinBndCond:
            # Finding the integral from the boundary condition up to a bin
            cumInt = multidim_cumsum(self.density*binAreas)

            # Finding the value of the integral up to each bin
            self.integral = self.firstBinBndCond + cumInt

            # Finding the error associated with this integration procedure
            self.integralErr = multidim_cumsum(self.densityErr*binAreas)

        elif self.useLastBinBndCond:
            # Finding the integral from a bin up to the boundary condition
            reverse_cumInt = multidim_cumsum(np.flip(self.density*binAreas))
            reverse_cumInt = np.flip(reverse_cumInt)

            # Finding the value of the integral up to each bin
            assert self.lastBinBndCond[1] in ['plus', 'minus'], \
                "Integration requires lastBinBndCond (plus or minus)"
            if self.lastBinBndCond[1] == 'plus':
                self.integral = self.lastBinBndCond[0] + reverse_cumInt
            elif self.lastBinBndCond[1] == 'minus':
                self.integral = self.lastBinBndCond[0] - reverse_cumInt

            # Finding the error associated with this integration procedure
            cumError = multidim_cumsum(np.flip(self.densityErr*binAreas))
            cumError = np.flip(cumError)
            self.integralErr = cumError

        # Getting the x and y values associated with the integral
        self.integral_xs, self.integral_ys = self.bins[0], self.bins[1]

        # Telling the integrator that it has an integral evaulated with MC:
        self.hasMCIntegral = True


    def montecarlo_data_dict(self, info=None):
        return {'bins': self.bins,
                'density': self.density,
                'density xs': self.density_xs,
                'density ys': self.density_ys,
                'density error': self.densityErr,
                'integral': self.integral,
                'integral xs:': self.integral_xs,
                'integral ys:': self.integral_ys,
                'integral error': self.integralErr,
                'info': info}


    def save_montecarlo_data(self, filename, info=None):
        """Saves the data used to produce the integral to a file"""
        assert self.hasMCDensity and self.hasMCIntegral, \
            "Need density function and integral to save MC data"

        # Saving the data to a file
        np.savez(filename,
                 **self.montecarlo_data_dict(info=info))


    # ------------------
    # Integral Interpolation:
    # ------------------
    def makeInterpolatingFn(self, interpolate_error=False,
                            # Extra arguments for the interpolation:
                            **kwargs):
        """Makes an interpolating function for the integral."""
        assert self.hasMCIntegral, \
            "Need MC integral to produce interpolation"
        x, y = self.integral_xs, self.integral_ys

        if self.useFirstBinBndCond:
            z = []
            z.append(np.ones(len(self.bins[0])) * self.firstBinBndCond)
            for z_i in self.integral:
                z.append(np.append(self.firstBinBndCond, z_i))
            z = np.array(z).T

            if interpolate_error:
                zerr = []
                zerr.append(np.zeros(len(self.bins[0])))
                for z_i in self.integralErr:
                    zerr.append(np.append(0, z_i))
                zerr = np.array(zerr).T

        elif self.useLastBinBndCond:
            z = []
            for z_i in self.integral:
                z.append(np.append(z_i, self.lastBinBndCond[0]))
            z.append(np.ones(len(self.bins[0])) * self.lastBinBndCond[0])

            z = np.array(z).T

            if interpolate_error:
                zerr = []
                for z_i in self.integralErr:
                    zerr.append(np.append(0, z_i))
                zerr.append(np.zeros(len(self.bins[0])))
                zerr = np.array(zerr).T

        z = z.flatten()
        self.interpFn = get_2d_interpolation(x, y, z, **kwargs)

        if interpolate_error:
            zerr = zerr.flatten()
            self.interpErr = get_2d_interpolation(x, y, zerr, **kwargs)
        else:
            self.interpErr = None

        self.hasInterpIntegral = True

    # ------------------
    # Density Interpolation:
    # ------------------
    def makeInterpolatingDensity(self, binspacing, monotone=False):
        """Makes an interpolating function for the density."""
        assert self.hasMCDensity, \
            "Need MC density to produce interpolation"
        bins = self.bins

        assert False, "Unsupported function."

        xs, ys = self.denisty_xs, self.density_ys

        if binspacing == 'lin':
            xs = (xs[1:] + xs[:-1])/2
            ys = (ys[1:] + ys[:-1])/2
        elif binspacing == 'log':
            xs = np.exp((np.log(xs[1:])
                         +np.log(xs[:-1]))/2.)
            ys = np.exp((np.log(ys[1:])
                         +np.log(ys[:-1]))/2.)

        self.interpDensity = interpolate.RegularGridInterpolator(
                                    (x, y), self.density,
                                    method='linear', bounds_error=False,
                                    fill_value=None)
        self.hasInterpDensity = True

    # ------------------
    # Validity Checks:
    # ------------------
    def checkValidAttributes(self):
        """Checks that the parameters given to the sampler are valid
        """
        pass

    # ------------------
    # Init:
    # ------------------
    def __init__(self):
        """Initializes the integrator class by letting it know that it has
        no information yet. This is fixed when information is added to the
        integrator. Detailed documentation is included in the README_utils
        file.
        """
        self.hasBins = False
        self.hasMCDensity = False
        self.hasMCIntegral = False
        self.hasInterpIntegral = False
        self.hasInterpDensity = False
        self.firstBinBndCond = None
        self.useFirstBinBndCond = False
        self.lastBinBndCond = None
        self.useLastBinBndCond = False
        self.hasAnalyticIntegral = False
        self.hasAnalyticDensity = False

        self.checkValidAttributes()

def integrate_2d(function, bounds,
                 bin_space='lin', epsilon=1e-10,
                 num_samples=1e5, num_bins=2,
                 bnd_cond=0, bnd_cond_bin='first'):
    """Performs a 1d integral over the given bounds.

    Parameters
    ----------
    function : function
        The function to be integrated.
    bounds : list
        The bounds of integration, in the format
        [[xmin, xmax], [ymin, ymax]]
    bin_space : string
        The space over which we sample. Must be `lin` or `log`.
    epsilon : float
        The cutoff for logarithmic sampling.
    num_samples : int
        Number of Monte Carlo samples for the integration
    num_bins : int
        Number of bins into which we divide the integration region.
        Default is 2, which leads to a single number for the integral.
    bnd_cond : float
        The boundary condition for the integral. For example, when integrating
        to find a cumulative distribution function, we want a result that
        yields 1 at the final bin.
    bnd_cond_bin :
        The bin at which the boundary condition is applied.
        Must be `first` or `last`.

    Returns
    -------
    integral, integral_error, xs
        The integral values and integral errors corresponding to given
        values of the argument of the integral.
    """
    if bin_space == 'lin':
        epsilon = None
    # Sampling
    testSampler_1 = simpleSampler(bin_space, bounds=bounds[0], epsilon=epsilon)
    testSampler_1.generateSamples(int(num_samples))
    samples_1 = testSampler_1.getSamples()

    testSampler_2 = simpleSampler(bin_space, bounds=bounds[1], epsilon=epsilon)
    testSampler_2.generateSamples(int(num_samples))
    samples_2 = testSampler_2.getSamples()

    # Setting up integrator
    this_integrator = integrator_2d()
    if bnd_cond_bin == 'first':
        this_integrator.setFirstBinBndCondition(bnd_cond)
    elif bnd_cond_bin == 'last':
        this_integrator.setLastBinBndCondition(bnd_cond)
    else:
        raise AssertionError("Bin at which we place a boundary condition"
                             +"must be 'first' or 'last'.")

    this_integrator.setBins(num_bins, [samples_1, samples_2], bin_space)

    weights = function(samples_1, samples_2)
    obs = [samples_1, samples_2]

    this_integrator.setDensity(obs, weights)
    this_integrator.integrate()

    integral = this_integrator.integral
    error = this_integrator.integralErr

    if bnd_cond_bin == 'first':
        xs = this_integrator.bins[0][1:]
        ys = this_integrator.bins[1][1:]
    else:
        xs = this_integrator.bins[0][:-1]
        ys = this_integrator.bins[1][:-1]

    if num_bins == 2:
        return integral[0][0], error[0][0], xs[0], ys[0]
    return integral, error, xs, ys
