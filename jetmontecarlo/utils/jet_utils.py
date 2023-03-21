from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class JetAlgorithm(ABC):
    """Class for keeping track of jet algorithms."""
    radius: float
    description: str


@dataclass
class TwoParticleJetAlgorithm():
    """Class for keeping track of very simple jet algorithms
    which can simply take in two particles and decide whether
    they are in a jet (i.e. without any global information)"""
    radius: float
    description: str

    # Adding a method which takes in an angle, and perhaps some
    # other information, and returns whether the two particles
    # are in the jet
    is_in_jet: Callable[[float, ...], bool]


class AlgorithmAgnosticApproximation(TwoParticleJetAlgorithm):
    def __init__(self, radius):
        description = "Algorithm Agnostic Approximation (AAA):"\
            " A simple algorithm which approximates two particles"\
            " as being in the same jet if they are within an angular"\
            " distance R (the jet radius) of one another."

        def is_in_jet(angle):
            return angle < radius

        super().__init__(radius, description, is_in_jet)


def alg_to_string(jet_alg, latex=True):
    """
    Returns a string naming the jet algorithm with the
    given index.

    Parameters
    ----------
        jet_alg : int
            Integer indicating the FastJet index associated with the jet
            algorithm.

    Returns
    -------
        str
            A (LaTeX-compatible, if requested) string naming the algorithm.
    """
    # - - - - - - - - - - - - - - - - -
    # pp algorithms
    # - - - - - - - - - - - - - - - - -
    if jet_alg in [0, "0", "kt"]:
        if latex:
            return r'$k_T$'
        else:
            return 'kt'
    elif jet_alg in [1, "1", "ca", "cambridge-aachen"]:
        if latex:
            return r'C/A'
        else:
            return 'ca'
    elif jet_alg in [2, "2", "akt", "anti-kt", "antikt"]:
        if latex:
            return r'anti-$k_T$'
        else:
            return 'akt'
    # - - - - - - - - - - - - - - - - -
    # ee algorithms
    # - - - - - - - - - - - - - - - - -
    if jet_alg in ["ee_0", "ee_kt"]:
        if latex:
            return r'$k_T^{(e^+ e^-)}$'
        else:
            return 'ee_kt'
    elif jet_alg in ["ee_1", "ee_ca", "ee_cambridge-aachen"]:
        if latex:
            return r'C/A$\?^{(e^+ e^-)}$'
        else:
            return 'ee_ca'
    elif jet_alg in ["ee_2", "ee_akt", "ee_anti-kt", "ee_antikt"]:
        if latex:
            return r'anti-$k_T^{(e^+ e^-)}$'
        else:
            return 'ee_akt'
    else:
        raise AssertionError(f"Invalid jet algorithm {jet_alg}")


