import sys
import math
import random
import numpy as np

# Utils stolen and then adjusted from the lorentz package,
# as well as the energyflow package:
# https://github.com/lukasheinrich/lorentz
# https://github.com/pkomiske/EnergyFlow/

mink_metric = [-1, 1, 1, 1]

def contract(lhs, rhs, metric=None):
    """Performs a contraction between two vectors,
    given a metric. In the absence of a metric,
    returns a Euclidean dot product.
    """
    return sum(m*l*r for m, l, r in
               zip(metric if metric else [1]*len(lhs),
                   lhs, rhs))

class Vector:
    """A class designed to contain the information
    of a vector in an arbitrary number of dimensions,
    and quickly return important information and
    perform important manipulations.
    """
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    # Basic properties
    def dim(self):
        """Returns the dimension of the vector."""
        return len(self.vector)

    def mag2(self):
        """Returns the magnitude squared of the vector."""
        return contract(self.vector, self.vector)

    def mag(self):
        """Returns the magnitude of the vector."""
        return math.sqrt(self.mag2())

    #
    def perp2(self):
        """Returns the magnitude squared of the transverse
        components of the vector, where here 'transverse'
        means transverse to the "z" or -1 axis.
        """
        transvers_comps = self.vector[:-1]
        return contract(transvers_comps, transvers_comps)

    def perp(self):
        """Returns the magnitude of the transverse
        components of the vector, where here 'transverse'
        means transverse to the "z" or -1 axis.
        """
        return math.sqrt(self.perp2())

    @property
    def theta(self):
        """Returns the angle between the vector and
        the "z" or -1 axis.
        """
        return math.atan2(self.perp(), self.vector[-1])

    # Properties which single out the "x" and "y" axes
    @property
    def phi(self):
        """Returns the angle of the vector in
        the "x-y" or 0-1 plane.
        """
        return math.atan2(self.vector[1], self.vector[0])

    # Operations
    def unit(self):
        """Returns a unit vector in the direction
        of this vector.
        """
        return Vector(self.vector/self.mag())

    def rotate_around(self, axis, rot_angle):
        """Rotates this vector around a given axis by a
        given angle using Rodrigues' rotation formula:
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        Only designed to work in three dimensions.
        """
        # Rotation of a
        assert (axis.dim() == 3 and self.dim() == 3), \
            "Rodrigues' rotation formula is only valid in 3D."

        vec = self.vector
        axvec = axis.vector / axis.mag()


        vector_rot = (
            vec * np.cos(rot_angle)
            +
            np.cross(axvec, vec) * np.sin(rot_angle)
            +
            axvec*contract(vec, axvec) * (1-np.cos(rot_angle))
            )

        return Vector(vector_rot)

def rand_perp_vector(vector):
    """Returns a vector perpendicular to a given vector,
    with a random angle in the subspace perpendicular to
    that vector.

    Only designed to work in three dimensions.
    """
    assert vector.mag() > 0, \
        "All vectors are perpendicular to the zero vector"
    assert vector.dim() == 3, \
        "Can only do 3D vectors right now!"

    perp_vector = Vector(np.zeros(len(vector.vector)))

    index = 0
    while perp_vector.mag() == 0:
        ref_vec = np.zeros(vector.dim())
        ref_vec[index] = 1.
        perp_vector.vector = np.cross(vector.vector, ref_vec)

    rot_angle = random.random() * 2. * math.pi
    perp_vector = perp_vector.rotate_around(vector, rot_angle)

    return perp_vector

def angle(vec1, vec2):
    """Returns the angle between two vectors."""
    vec1 = vec1.unit()
    vec2 = vec2.unit()

    return np.arccos(contract(vec1.vector, vec2.vector))

class FourVector(Vector):
    """A subclass of class vector specialized to
    four dimensional Minkowski space.
    """
    def __init__(self, vector):
        assert(len(vector) == 4), \
            "Invalid number of components."
        super().__init__(vector)

    # Rapidity and pseudorapidity
    @property
    def eta(self):
        """Pseudorapidity of this four vector."""
        if abs(self.perp()) < sys.float_info.epsilon:
            return (float('inf') if self.vector[3] >= 0
                    else float('-inf'))
        return -math.log(math.tan(self.theta()/2.))

    @property
    def y(self):
        """Rapidity of this four vector."""
        return -math.atanh(self.vector[3]
                           /self.vector[0])

    # Magnitude of vectors
    def m2(self):
        """Mass squared of the four vector, or the
        magnitude squared using the Minkowski metric
        with mostly minus signature.
        """
        return -contract(self.vector, self.vector,
                         metric=mink_metric)

    def m(self):
        """Mass of the four vector, or the magnitude using
        the Minkowski metric with mostly minus signature.
        """
        return math.sqrt(self.m2())

    def mag2(self):
        """Magnitude squared of the spatial components
        of the four vector.
        """
        spatial_comps = self.vector[1:]
        return contract(spatial_comps, spatial_comps)

    def mag(self):
        """Magnitude of the spatial components
        of the four vector.
        """
        return math.sqrt(self.mag2())

    # Redefinitionss for the new timelike coordinate
    @property
    def phi(self):
        """Magnitude squared of the spatial components
        of the four vector.
        """
        return math.atan2(self.vector[2],
                          self.vector[1])

    def perp2(self):
        transverse_comps = self.vector[1:-1]
        return contract(transverse_comps, transverse_comps)
