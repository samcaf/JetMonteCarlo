# Utils for interpolation
from scipy import interpolate
from jetmontecarlo.montecarlo.integrator import integrate_1d

# Utils for finding reasonable theta values
from jetmontecarlo.utils.interpolation import lin_log_mixed_list

# QCD utils
from jetmontecarlo.analytics.qcd_utils import alpha_s,\
    alpha_fixed, splittingFn

# =====================================
# Splitting Functions:
# =====================================
# Generation of Normalizing Factors for Splitting Functions:
def gen_normalized_splitting(num_samples, z_cut,
                             jet_type='quark', accuracy='LL',
                             fixed_coupling=True,
                             bin_space='lin', epsilon=1e-15,
                             num_bins=100):
    # Preparing a list of thetas, and normalizations which will depend on theta
    theta_calc_list, norms = lin_log_mixed_list(epsilon, 1., num_bins), []

    progress_bar_size = 20

    for itheta, theta in enumerate(theta_calc_list):
        # Progress bar:
        itheta_max = len(theta_calc_list) - 1
        if itheta % int(itheta_max/progress_bar_size) == 0:
            done = int(progress_bar_size * itheta/itheta_max)
            left = progress_bar_size - done
            print('['+'#'*done+' '*left+']', flush=True)

        # Preparing the weight we want to normalize
        def weight(z):
            if fixed_coupling:
                alpha = alpha_fixed
            else:
                alpha = alpha_s(z, theta)
            return alpha * splittingFn(z, jet_type, accuracy)
        # Finding the normalization factor
        n, _, _ = integrate_1d(weight, [z_cut, 1./2.],
                               bin_space=bin_space, epsilon=epsilon,
                               num_samples=num_samples)
        norms.append(n)

    # Making an interpolating function for the splitting fn normalization
    normalization = interpolate.interp1d(x=theta_calc_list,
                                         y=norms,
                                         fill_value="extrapolate")

    def normed_splitting_fn(z, theta):
        if fixed_coupling:
            alpha = alpha_fixed
        else:
            alpha = alpha_s(z, theta)
        splitfn =  alpha*splittingFn(z, jet_type, accuracy)/normalization(theta)
        return splitfn * (z_cut < z) * (z < 1./2.)

    return normed_splitting_fn
