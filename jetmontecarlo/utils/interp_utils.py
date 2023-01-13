import numpy as np

from scipy.misc import derivative
from scipy.optimize import fsolve

# DEBUG
# from sympy.solvers import solve
# from sympy import Symbol
# from sympy import diff, lambdify
import matplotlib.pyplot as plt

def lin_log_mixed_list(lower_bound, upper_bound, num_bins):
    # Mixing linear and logarithmic bins for the list of thetas
    mixed_list = np.logspace(np.log10(lower_bound), np.log10(upper_bound),
                             int(num_bins/2)+2)
    mixed_list = np.append(mixed_list,
                           np.linspace(lower_bound, upper_bound,
                                       int(num_bins/2)))
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
    elif check_only in ['decreasing']:
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
    # DEBUG
    # print("xs: ", [(xs[i], xs[i+1]) for i in sign_change_inds])
    # print([(derivatives[i], derivatives[i+1]) for i in sign_change_inds])

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

    roots = fsolve(dfunc_dx, include_point)
    roots = roots[dfunc_dx(roots) <= 1e-3]
    print(f"{roots = }")
    print(f"{deriv(roots) = }")

    if len(roots) != 0:
        # Checking monotonicity around each flat point
        mid_derivs = [deriv(midpoint) for midpoint in
                      np.array(roots[1:]+roots[:-1])/2]
        print(f"{mid_derivs = }")
        mid_derivs = [deriv(roots[0]-.25), *mid_derivs,
                      deriv(roots[-1]+.25)]
        mid_derivs = np.array(mid_derivs)

        assert len(mid_derivs) == len(roots)+1
        print(f"{mid_derivs = }")

        # Removing roots where the derivative on either side
        # has the same sign (i.e. the function is still monotonic)
        good_roots = []
        for i in range(len(roots)):
            if mid_derivs[i]*mid_derivs[i+1] <= 0:
                good_roots.append(True)
            else:
                good_roots.append(False)
        roots = roots[good_roots]

    print(roots, flush=True)

    plt.plot(np.linspace(0, 1, 100), func(np.linspace(0, 1, 100)),
             label="function")
    plt.plot(np.linspace(0, 1, 100), deriv(np.linspace(0, 1, 100)),
             label="derivative")
    plt.ylim(-.01, 1)
    plt.legend()
    plt.show()

    if len(roots) == 0:
        # If there are no flat points, the function is monotonic
        return [-np.inf, np.inf]

    # Checking edges of the function
    if include_point < roots[0]:
        return [-np.inf, roots[0]]
    if include_point > roots[-1]:
        return [roots[-1], np.inf]

    assert len(roots) > 1
    # Should never be activated, after checking edges

    # Checking in the middle of the function
    for i in range(len(roots) - 1):
        if include_point < roots[i+1]:
            return [roots[i], roots[i+1]]

    # We should have a monotonic domain if df/dx(include_point)
    # has the right sign
    raise AssertionError("Unexpectedly, no monotonic domain found")


# Monotonicity tests
if __name__ == "__main__":
    print(monotonic_domain(lambda x: x**2, [-10, 10], 1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [0, 10], 1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], 1, check_only='decreasing'))
    print(monotonic_domain(lambda x: x**2, [0, 10], 1, check_only='decreasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], -1, check_only='increasing'))
    print(monotonic_domain(lambda x: x**2, [-10, 10], -1, check_only='decreasing'))
