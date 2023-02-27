from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from jetmontecarlo.utils.montecarlo_utils import getLinSample,\
                                                 getLogSample
from jetmontecarlo.montecarlo.sampler import sampler

# =====================================
# Basic Process
# =====================================

@dataclass
class BasicProcess():
    """A class designed to contain basic information associated
    with physical processes.
    """
    name: str
    energy: float

    def __str__(self):
        return f"The process [{self.name}] "\
            f"at an energy of {self.energy} GeV."\


# =====================================
# Kinematic Process
# =====================================
def assert_kinematic_kwargs(func):
    """Decorator for functions in a KinematicProcess
    which asserts that the keyword arguments passed to a
    function are the same as the Process' kinematic variables.
    """
    def wrapper(*args, **kwargs):
        kinematic_vars = args[0].kinematic_vars
        assert set(kwargs.keys()) == set(kinematic_vars), \
            "The parameters given to the function are not"\
            "the same as the kinematic variables."
        res = func(*args, **kwargs)
        return res
    return wrapper


@dataclass
class KinematicProcess(ABC, BasicProcess):
    """A class designed to contain basic information, descriptions of
    kinematic variables, the phase space for the kinematics and the
    associated differential and total cross sections for a physical
    process.
    """
    # - - - - - - - - - - - - - - - - -
    # Kinematic Variables
    # - - - - - - - - - - - - - - - - -
    # A dict whose keys are the names of kinematic variables and whose
    # values are their descriptions
    kinematic_vars: list
    total_cross_section: float

    # - - - - - - - - - - - - - - - - -
    # Phase space constraints
    # - - - - - - - - - - - - - - - - -
    @assert_kinematic_kwargs
    def lies_in_phasespace(self, **kwargs):
        """Returns True if the given kinematic variables lie in the
        phase space for the process, otherwise False.

        Ideally, not changed by subclasses.
        """
        return self.__phasespace_constraints(**kwargs)

    @abstractmethod
    def __phasespace_constraints(self, **kwargs):
        """The set of phase space constraints for the given process.
        NEEDS TO BE IMPLEMENTED BY SUBCLASS.
        """
        pass


    # - - - - - - - - - - - - - - - - -
    # Differential cross section
    # - - - - - - - - - - - - - - - - -
    @assert_kinematic_kwargs
    def differential_xsec(self, **kwargs):
        """Returns the differential cross section for the given
        kinematic variables.

        Ideally, not changed by subclasses.
        """
        return self.__differential_xsec(**kwargs)

    @abstractmethod
    def __differential_xsec(self, **kwargs):
        """The differential cross section for the given process.
        NEEDS TO BE IMPLEMENTED BY SUBCLASS.
        """
        pass

    # - - - - - - - - - - - - - - - - -
    # Additional Initialization
    # - - - - - - - - - - - - - - - - -
    def __post_init__(self):
        self.num_variables = len(self.kinematic_vars)


# =====================================
# Monte Carlo Process
# =====================================

@dataclass
class MonteCarloProcess(sampler, KinematicProcess, ABC):
    """A class designed to contain and kinematic information associated
    with a physical process, and to sample from the associated phase space.
    """
    kinematic_bounds: list

    def addSample(self):
        """Adds a sample to the process' sample list."""
        while True:
            # Generating a sample
            sample = []
            jac = 1.
            for bound in self.kinematic_bounds:
                if self.sampleMethod == 'lin':
                    sample.append(getLinSample(bound[0], bound[1]))
                    jac *= 1.
                elif self.sampleMethod == 'log':
                    sample.append(getLogSample(bound[0], bound[1],
                                               self.epsilon))
                    jac *= sample[-1]

            # Checking if the sample lies in the phase space
            phase_space_point = {varname: sample[i] for i, varname in
                                 enumerate(self.kinematic_vars)}
            if self.lies_in_phasespace(**phase_space_point):
                self.samples.append(sample)

                jac *= self.differential_xsec(**phase_space_point)
                self.jacobians.append(jac)
                break

            # If not, incrementing the number of rejected samples
            self.num_rejected_samples += 1


    def named_samples(self):
        """Returns a list of dicts, each of which contains a sample
        and the corresponding kinematic variables.
        """
        return [{varname: sample[i] for i, varname in
                 enumerate(self.kinematic_vars)}
                for sample in self.samples]


    def __max_area(self):
        """The maximum area of the phase space for the given bounds
        on the kinematic variables.
        """
        max_area = 1.
        for bound in self.kinematic_bounds:
            max_area *= bound[1] - bound[0]
        return max_area


    def setArea(self):
        """Calculates the area of the phase space. Requires that
        phase space samples have already been generated.
        """
        if len(self.samples) == 0:
            self.area = None

        efficiency = len(self.samples) \
            / (len(self.samples) + self.num_rejected_samples)
        return self.__max_area() * efficiency


    # - - - - - - - - - - - - - - - - -
    # Additional Initialization
    # - - - - - - - - - - - - - - - - -
    # DEBUG: Do this -- montecarloprocess initialization
    # def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # super().__post_init__()
        # super().__init__(sampleMethod, **kwargs)

    def __post_init__(self):
        assert len(self.kinematic_bounds) == self.num_variables, \
            "The number of bounds for the kinematic variables doesn't"\
            " match the number of kinematic variables."

        # Preparing for area calculations for the phase space
        self.num_rejected_samples = 0



if __name__ == "__main__":
    a = BasicProcess("test", 1.0)
    print(a)


# =====================================
# Observable for a Process
# =====================================
# Takes in the set of kinematic variables associated with a process
# and returns an observable (function of the kinematic variables)
@dataclass
class MonteCarloObservable:
    """A class designed to contain information associated with
    a particular observable for a MonteCarloProcess.
    Translates from the kinematic observables of the MonteCarloProcess
    to the observable of interest.
    """
    process: MonteCarloProcess
    function: Callable
    name: str = ''

    @assert_kinematic_kwargs
    def kinematic_function(self, **kwargs):
        """The function of the kinematic variables associated with
        the process that is to be integrated over the phase space
        to obtain the observable.
        """
        return self.function(**kwargs)


    def update(self):
        """Updates the observable's values based on the current
        samples in the process.
        """
        # Setting up observables
        self.observables = [self.kinematic_function(**sample)
                            for sample in self.process.named_samples()]
        # Weights (includes weight from differential cross section)
        self.weights = self.process.jacobians


    def __post_init__(self):
        self.num_variables = self.process.num_variables
        self.kinematic_vars = self.process.kinematic_vars

        self.observables = None
        self.weights = None
