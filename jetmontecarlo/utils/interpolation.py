import numpy as np

from scipy import interpolate
# For monotonicity checks:
from scipy.misc import derivative

# For saving data and functions


# =====================================
# Misc. Utilities
# =====================================
# Includes functions to make bins with both linear and logarithmic
# spacing, and to check if an array or function is monotonic in a
# given domain; these will eventually be useful for
# interpolation functions and beyond.

def lin_log_mixed_list(lower_bound, upper_bound, num_vals):
    """Creates a array containing both linearly and
    logarithmically spaced data.

    Parameters
    ----------
        lower_bound : lower bound (>0) for the array
        upper_bound : upper bound (>0) for the array
        num_vals : number of values in the array

    Returns
    -------
    np.array :  array with linearly and logarithmically spaced data
    """
    # Mixing linear and logarithmic bins for the list of thetas
    mixed_list = np.logspace(np.log10(lower_bound), np.log10(upper_bound),
                             int(num_vals/2)+2)
    mixed_list = np.append(mixed_list,
                           np.linspace(lower_bound, upper_bound,
                                       int(num_vals/2)))
    # Sorting and removing the duplicate values of upper and lower bound
    mixed_list = np.sort(mixed_list[1:-1])
    mixed_list = mixed_list[~np.isnan(mixed_list)]
    return mixed_list


def where_monotonic_arr(arr, check_only):
    """
    Returns the indices where an array is monotonic.

    Parameters
    ----------
        arr :           array to be checked
        check_only :    "increasing" or "decreasing"

    Returns
    -------
    bool :      where the array is monotonic
    """
    assert check_only in ['increasing', 'decreasing']
    if check_only in ['increasing']:
        return np.where(arr[1:] >= arr[:-1])[0]
    # otherwise, if check_only is 'decreasing':
    return np.where(arr[:-1] >= arr[1:])[0]


def is_monotonic_arr(arr, check_only=None):
    """
    Checks if an array is monotonic.

    Parameters
    ----------
        arr :           array to be checked
        check_only :    "increasing", "decreasing", or None

    Returns
    -------
    bool :      whether or not the array is monotonic
    """
    assert check_only in ['increasing', 'decreasing', None]
    if check_only in ['increasing']:
        return (arr[1:] >= arr[:-1]).all()
    if check_only in ['decreasing']:
        return (arr[:-1] >= arr[1:]).all()
    # Otherwise, if check_only is None:
    return (arr[1:] >= arr[:-1]).all() or (arr[:-1] >= arr[1:]).all()


def is_monotonic_func(func, domain, check_only=None):
    """
    Checks if a function is monotonic in the given domain.

    Parameters
    ----------
        func :          function to check
        domain :        domain in which we want to stay
        check_only :    "increasing", "decreasing", or None

    Returns
    -------
    bool :      whether or not the function is monotonic
    """
    arr = func(lin_log_mixed_list(domain[0], domain[1], 500))
    return is_monotonic_arr(arr, check_only)


def monotonic_domain(func, domain, include_point=None, check_only=None):
    """

    Parameters
    ----------
        func :          function to check
        domain :        domain in which we want to stay
        include_point : point to include in the domain
        check_only :    "increasing", "decreasing", or None

    Returns
    -------
    domain :    domain (including `include_point`) in which the function
                is monotonic
    """
    if include_point is None:
        include_point = (domain[0]+domain[1])/2
    assert domain[0] < include_point < domain[1]
    # x = Symbol("x", real=True)
    # deriv = lambdify(x, diff(func(x), x))
    def dfunc_dx(x):
        return derivative(func, x, 1e-8)

    # Easy check: the derivative must be
    # in the right direction at the point
    # we require to be included (`include_point`)
    if (check_only in ['increasing'] and
        dfunc_dx(include_point) < 0)\
        or\
       (check_only in ['decreasing'] and
        dfunc_dx(include_point) > 0):
        return None

    # If it is, we continue: there is a monotonic domain containing the point

    # First, finding where the derivative changes sign
    # (i.e. the derivative passes through zero)
    if domain[0] > 0:
        xs = lin_log_mixed_list(domain[0]+1e-20, domain[1], 10000)
    else:
        xs = np.linspace(domain[0], domain[1], 10000)
    derivatives = dfunc_dx(xs)

    sign_change_inds = []
    for i in range(len(derivatives)-1):
        if derivatives[i] * derivatives[i+1] < 0:
            sign_change_inds.append(i)

    # If the derivative never changes sign, the function is monotonic
    # (modulo the possibility that our sampling of the domain is not fine enough)
    if len(sign_change_inds) == 0:
        return domain

    # Otherwise, we find the interval containing `include_point`
    lower_bound = domain[0]
    upper_bound = domain[1]
    for ind in sign_change_inds:
        if include_point >= xs[ind+1]:
            lower_bound = xs[ind+1]
        if include_point <= xs[ind]:
            upper_bound = xs[ind]
            return (lower_bound, upper_bound)
        if xs[ind] <= include_point <= xs[ind+1]:
            recursed_bounds = monotonic_domain(
                func, (xs[ind], xs[ind+1]), include_point, check_only)

            assert recursed_bounds is not None
            assert recursed_bounds != domain

            recursed_lower, recursed_upper = recursed_bounds
            if include_point > recursed_lower:
                lower_bound = recursed_lower
            if include_point < recursed_upper:
                upper_bound = recursed_upper
                return (lower_bound, upper_bound)
    return (lower_bound, upper_bound)


# =====================================
# Interpolation Function Utilities
# =====================================

def get_1d_interpolation(xs, fs, monotonic=True,
                         bounds=None, bound_values=[None, None],
                         verbose=0):
    """Returns a function that interpolates the data `(x, y)`
    using the interpolation method `kind`.

    """
    if monotonic:
        # If we want monotonic interpolation, we can use np.interp
        def interpolating_function(x):
            return np.interp(x, xs, fs)

        # Testing monotonicity:
        interp_vals = interpolating_function(xs)
        is_monotone = ((interp_vals[1:] <= interp_vals[:-1]).all() or
                        (interp_vals[1:] >= interp_vals[:-1]).all())
        assert is_monotone, "User asked for a monotone integral,"\
            " but the integral's interpolating function is not monotone!"
    else:
        # If we do not need to enforce monotonicity
        interpolating_function = interpolate.interp1d(x=xs, y=fs,
                                         fill_value="extrapolate")

    # If we require monotonicity, the we restrict the interpolating
    # function to the monotonic domain
    if monotonic and bounds is None:
        if verbose >= 1:
            print("Setting the bounds of the interpolating function to its"
                  + " monotonic domain.")
        bounds = monotonic_domain(interpolating_function, [xs[0], xs[1]])
        if verbose >= 2:
            print(f"monotonic domain: {bounds}")

    # Setting the interpolating function outside the bounds, if given
    if bounds is not None:
        # Setting up the values of the interpolating function on the
        # boundaries
        if bound_values is None:
            bound_values = [None, None]
        assert len(bound_values) == 2, "bound_values must be a tuple of length 2"

        # NOTE: Enforcing continuity by hand.
        #       This may not always be desirable.
        if bound_values[0] is None:
            bound_values[0] = interpolating_function(xs[0])
        if bound_values[1] is None:
            bound_values[1] = interpolating_function(xs[-1])

        # Setting the interpolating function below the lower bound,
        # between the upper and lower bound,
        # and above the upper bound, respectively
        def bounded_interpolating_function(x):
            return bound_values[0] * (x <= bounds[0])\
            + (bounds[0] <= x) * (x <= bounds[1]) * interpolating_function(x)\
            + bound_values[1] * (x >= bounds[1])
    else:
        assert bound_values is None, "Cannot assign boundary values if"+\
            " no bounds for the interpolation function are given."

    return bounded_interpolating_function


def get_2d_interpolation(xs, ys, zs,
        interpolation_method="RectangularGrid",
        **kwargs):
    """Returns a function that interpolates the data `(x, y, z)`
    using the interpolation method `interpolation_method`.
    """
    if interpolation_method == "RectangularGrid":
        default_kwargs = {'method': 'linear', 'bounds_error': False,
                          'fill_value': None}
        kwargs = {**default_kwargs, **kwargs}
        interpolating_function = interpolate.RegularGridInterpolator(
                                    (xs, ys), zs, **kwargs)
    elif interpolation_method in ["Linear", "linear"]:
        interpolating_function = interpolate.interp2d(xs, ys, zs,
                                                     kind="linear")
    elif interpolation_method in ["Cubic", "cubic"]:
        interpolating_function = interpolate.interp2d(xs, ys, zs,
                                                     kind="cubic")
    elif interpolation_method in ["Nearest", "nearest"]:
        points = list(zip(xs, ys))
        interpolating_function = interpolate.NearestNDInterpolator(
                                    points, zs)
    else:
        raise ValueError(f"Unknown interpolation method {interpolation_method}")

    return interpolating_function



# =====================================
# Testing
# =====================================
if __name__ == "__main__":
    # Monotonicity tests
    print(monotonic_domain(lambda x: x**2, [-10, 10], 1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [0, 10], 1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], 1, check_only='decreasing'))
    print(monotonic_domain(lambda x: x**2, [0, 10], 1, check_only='decreasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], -1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], -1, check_only='decreasing'))
