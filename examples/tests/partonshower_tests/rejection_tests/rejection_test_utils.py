import random
import numpy as np

# Local utils:
# Utilities for parton showering
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.utils.partonshower_utils import *
from jetmontecarlo.analytics.QCD_utils import *

# Utilities for analytic expressions
from jetmontecarlo.analytics.radiators import *
from examples.tests.partonshower_tests.test_partonshower_angularities import *

########################################
# Accept-Reject based angular showers:
########################################
def angularity_split_rej(ang_init, beta, jet_type, test_num):
    """A set of splitting methods designed to test
    acceptance-rejection sampling in the context of
    parton showers.
    """
    alpha = alpha_fixed

    accept_emission = False
    while not accept_emission:
        if test_num == -1:
            # Control: Larkoski's MLL algorithm.
            # ---------------------------------------------
            alpha = 1.
            r1, r2 = random.random(), random.random()

            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(r1))) / 2.

            z = (2.*ang_final)**r2 / 2.
            theta = (2.*ang_final)**((1-r2) / beta)

            # In the MLL case, we use the acceptance-rejection
            # method to generate events associated with running coupling.
            # Notice that alpha is 1. for the MLL case. This is always
            # larger than alpha(freezing scale = 1 GeV) ~ 0.35
            cut = alpha_s(z, theta) / alpha

            # Note: there is a typo in Eqn 5.4 of the resource I
            # cite above the angularity_split method which is
            # important here (a stray factor of 2 on the RHS),
            # but the conclusion in Equation 5.5, which we use
            # here, is correct

            if random.random() < cut:
                accept_emission = True
            # These next two lines of code are the soul of the veto
            # algorithm. Rather than generating from scratch, as you
            # would for the von Neumann acceptance-rejection algorithm,
            # you use this scale as the scale for the next emission.
            # This correctly takes into account the exponentiation of
            # multiple emissions, as described in the Pythia manual:
            # https://arxiv.org/pdf/hep-ph/0603175.pdf#page=66&zoom=150,0,240
            else:
                ang_init = ang_final

        if test_num == 0:
            # Control: This is precisely the LL algorithm.
            # ---------------------------------------------
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.
            r = random.random()
            z = (2.*ang_final)**r / 2.
            theta = (2.*ang_final)**((1-r) / beta)
            accept_emission = True

        if test_num == 1:
            # Accept-reject algorithm on the z and theta
            # of the splitting.
            # Successful in reproducing LL results.
            # ---------------------------------------------
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.
            thmin = (2.*ang_final)**(1./beta)
            theta = random.random()*(1 - thmin) + thmin
            z = ang_final / theta**beta

            r = random.random()
            if r < (1.-thmin)/theta**(beta+1.):
                accept_emission = True

        if test_num == 2:
            # Accept-reject algorithm on final angularity.
            # sampling final angularity in lin space.
            # Attempts to reproduce LL results.
            # This fails by small corrections --
            # I believe this is because the pdf_max defined below,
            # which is essential in acceptance-rejection sampling,
            # is way too small. I think the correct pdf_max
            # is actually closer to 10^10 or so.
            # ---------------------------------------------
            # 1) Generating a final angularity less than the
            # given initial value:
            ang_final = getLinSample(0, ang_init)

            # 2) Defining analytic expressions:
            # A useful factor in shortening the LL pdf expression:
            f_LL = CR(jet_type)*alpha/(np.pi*beta)

            # Analytic expression for the LL cdf and pdf
            cdf = np.exp(-f_LL*(np.log(2.*ang_final)**2.
                                - np.log(2.*ang_init)**2.))
            pdf = -2.*f_LL*np.log(2.*ang_final) * cdf / ang_final

            pdf_max = 100.

            # 3) Implementing the accept-reject algorithm
            if random.random() < pdf / pdf_max:
                r = random.random()
                z = (2.*ang_final)**r / 2.
                theta = (2.*ang_final)**((1-r) / beta)
                accept_emission = True

        if test_num == 3:
            # Accept-reject algorithm on final angularity,
            # sampling final angularity in log space.
            # Successful in reproducing LL results, and
            # overccomes the weaknesses of the previous
            # algorithm (test_num=2) by going into log space.
            # ---------------------------------------------
            # 1) Generating a final angularity less than the
            # given initial value:
            ang_final = getLogSample(0, ang_init)

            # 2) Defining analytic expressions:
            # A useful factor in shortening the LL pdf expression:
            f_LL = CR(jet_type)*alpha/(np.pi*beta)

            # Analytic expression for the LL cdf and pdf
            cdf = np.exp(-f_LL*(np.log(2.*ang_final)**2.
                                - np.log(2.*ang_init)**2.))
            pdf = -2.*f_LL*np.log(2.*ang_final) * cdf

            pdf_max = 10.

            # 3) Implementing the accept-reject algorithm
            if random.random() < pdf / pdf_max:
                r = random.random()
                z = (2.*ang_final)**r / 2.
                theta = (2.*ang_final)**((1-r) / beta)
                accept_emission = True

        if test_num == 4:
            # Accept-reject algorithm on final angularity,
            # sampling final angularity in log space.
            # Attempts to reproduce results with runnning
            # coupling, but only the singular pieces of
            # splitting functions.
            # ---------------------------------------------
            # 1) Generating a final angularity less than the
            # given initial value:
            ang_final = getLogSample(0, ang_init,
                                     epsilon=5e-5)

            # 2) Analytic expressions for the MLL pdf,
            # normalizing by the CDF at ang_init
            rad_f, radprime_f = ang_rad_radprime_MLL(ang_final,
                                                     beta=beta,
                                                     jet_type=jet_type)
            rad_i, _ = ang_rad_radprime_MLL(ang_init,
                                            beta=beta,
                                            jet_type=jet_type)
            pdf = (-radprime_f * ang_final
                   * np.exp(-rad_f)/np.exp(-rad_i)
                  )
            # print("pdf: " + str(pdf))
            pdf_max = 3.

            # 3) Implementing the accept-reject algorithm
            if random.random() < pdf / pdf_max:
                r = random.random()
                z = (2.*ang_final)**r / 2.
                theta = (2.*ang_final)**((1-r) / beta)
                accept_emission = True

        if test_num == 5:
            # Accept-reject algorithm on final angularity,
            # sampling final angularity by inverse transform.
            # Attempts to reproduce results with runnning
            # coupling, but only the singular pieces of
            # splitting functions.
            # ---------------------------------------------
            # 1) Generating a final angularity using the inverse
            # transfom method at LL
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.

            # 2) Analytic expressions for the MLL pdf,
            # normalizing by the CDF at ang_init
            rad_f, radprime_f = ang_rad_radprime_MLL(ang_final,
                                                     beta=beta,
                                                     jet_type=jet_type)
            rad_i, _ = ang_rad_radprime_MLL(ang_init,
                                            beta=beta,
                                            jet_type=jet_type)
            pdf = (-radprime_f * np.exp(-rad_f)/np.exp(-rad_i))

            # Analytic expression for the LL cdf and pdf
            f_LL = CR(jet_type)*alpha/(np.pi*beta)
            cdf_LL = np.exp(-f_LL*(np.log(2.*ang_final)**2.
                                   - np.log(2.*ang_init)**2.))
            pdf_LL = -2.*f_LL*np.log(2.*ang_final) * cdf_LL / ang_final
            pdf_ratio_max = 3.

            # 3) Implementing the accept-reject algorithm
            pdf_ratio = pdf / pdf_LL
            if random.random() < pdf_ratio / (pdf_ratio_max):
                r = random.random()
                z = (2.*ang_final)**r / 2.
                theta = (2.*ang_final)**((1-r) / beta)
                accept_emission = True

        if test_num == 6:
            # Accept-reject algorithm on final angularity,
            # sampling final angularity by inverse transform.
            # Attempts to reproduce results with runnning
            # coupling, but only the singular pieces of
            # splitting functions.
            # Afterwards, samples z and theta using accept-reject
            # sampling as well, to take running coupling effects
            # into account.
            # ---------------------------------------------
            # 1) Generating a final angularity using the inverse
            # transfom method at LL
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.

            # 2) Analytic expressions for the MLL pdf,
            # normalizing by the CDF at ang_init
            rad_f, radprime_f = ang_rad_radprime_MLL(ang_final,
                                                     beta=beta,
                                                     jet_type=jet_type)
            rad_i, _ = ang_rad_radprime_MLL(ang_init,
                                            beta=beta,
                                            jet_type=jet_type)
            pdf = (-radprime_f * np.exp(-rad_f)/np.exp(-rad_i))

            # Analytic expression for the LL cdf and pdf
            f_LL = CR(jet_type)*alpha/(np.pi*beta)
            cdf_LL = np.exp(-f_LL*(np.log(2.*ang_final)**2.
                                   - np.log(2.*ang_init)**2.))
            pdf_LL = -2.*f_LL*np.log(2.*ang_final) * cdf_LL / ang_final
            pdf_ratio_max = 3.

            # 3) Implementing the accept-reject algorithm
            pdf_ratio = pdf / pdf_LL
            if random.random() < pdf_ratio / (pdf_ratio_max):
                # Once we have a suitable final angularity,
                # accept-reject sampling z and theta:
                accept_ztheta = False
                while not accept_ztheta:
                    r = random.random()
                    # Logarithmically sampling z and theta
                    z = (2.*ang_final)**r / 2.
                    theta = (2.*ang_final)**((1-r) / beta)

                    if random.random() < alpha_s(z, theta)/alpha_s(0, 0):
                        accept_ztheta = True
                accept_emission = True

        if test_num == 7:
            # Accept-reject algorithm on final angularity,
            # sampling final angularity by inverse transform.
            # Attempts to reproduce results with runnning
            # coupling, but only the singular pieces of
            # splitting functions.
            # Afterwards, samples z and theta using accept-reject
            # sampling as well, to take running coupling effects
            # into account.
            # ---------------------------------------------
            # 1) Generating a final angularity using the inverse
            # transfom method at LL
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.

            # 2) Analytic expressions for the MLL pdf,
            # normalizing by the CDF at ang_init
            rad_f, radprime_f = ang_rad_radprime_MLL(ang_final,
                                                     beta=beta,
                                                     jet_type=jet_type)
            rad_i, _ = ang_rad_radprime_MLL(ang_init,
                                            beta=beta,
                                            jet_type=jet_type)
            pdf = (-radprime_f * np.exp(-rad_f)/np.exp(-rad_i))

            # Analytic expression for the LL cdf and pdf
            f_LL = CR(jet_type)*alpha/(np.pi*beta)
            cdf_LL = np.exp(-f_LL*(np.log(2.*ang_final)**2.
                                   - np.log(2.*ang_init)**2.))
            pdf_LL = -2.*f_LL*np.log(2.*ang_final) * cdf_LL / ang_final
            pdf_ratio_max = 3.

            # 3) Implementing the accept-reject algorithm
            pdf_ratio = pdf / pdf_LL
            if random.random() < pdf_ratio / (pdf_ratio_max):
                # Once we have a suitable final angularity,
                # accept-reject sampling z and theta:
                accept_ztheta = False
                while not accept_ztheta:
                    r = random.random()
                    # Logarithmically sampling z and theta
                    z = (2.*ang_final)**r / 2.
                    theta = (2.*ang_final)**((1-r) / beta)

                    if random.random() < alpha_s(z, theta)/alpha_s(0, 0):
                        accept_ztheta = True
                accept_emission = True

        if test_num == 8:
            # Sampling final angularity by inverse transform.
            # Attempts to reproduce results with runnning
            # coupling, but only the singular pieces of
            # splitting functions.
            # Afterwards, samples z and theta using accept-reject
            # sampling, to take running coupling effects
            # into account.
            # ---------------------------------------------
            # Generating a final angularity using the inverse
            # transfom method at LL
            alpha = alpha_s(0, 0)
            ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                        - np.pi*beta/(CR(jet_type)*alpha)
                                        * np.log(random.random())
                                        )) / 2.
            # Logarithmically sampling z and theta
            r = random.random()
            z = (2.*ang_final)**r / 2.
            theta = (2.*ang_final)**((1-r) / beta)

            # The Larkoski-Thaler algorithm attempts to accept-reject
            # based on the following procedure, with no enhancement factor.
            # When I perform some pen and paper calulations, I find that
            # I might be able to get _my_ results, if I enhance the
            # acceptance probability of low angularities by an additional
            # factor:

            enhancement_factor = (
                np.exp(CR(jet_type)*(alpha-alpha_fixed)
                       * np.log(2.*ang_final)**2.
                       / (beta*np.pi))
                ) / 50.
            cut = alpha_s(z, theta) / alpha * enhancement_factor

            if random.random() < cut:
                accept_emission = True
                # print("ACCEPTED")
                # print(ang_final)
    return ang_final, z, theta

# ----------------------------------
# Recursive Shower:
# ----------------------------------
def angularity_shower_LL_rej(parton, ang_init, beta, jet_type, jet,
                             split_soft=True, test_num=0):
    """Performs an angularity shower with the algorithms
    we designed above to test acceptance-rejection sampling
    in the context of parton showers.
    """
    if ang_init > MU_NP:
        ang_final, z, theta = angularity_split_rej(ang_init, beta,
                                                   jet_type, test_num)
        if ang_final is None:
            return

        parton.split(z, theta)
        d1, d2 = parton.daughter1, parton.daughter2

        jet.partons.append(d1)
        if split_soft:
            angularity_shower_LL_rej(parton=d1,
                                     ang_init=ang_final,
                                     beta=beta,
                                     jet_type=jet_type,
                                     jet=jet,
                                     split_soft=split_soft,
                                     test_num=test_num)

        jet.partons.append(d2)
        angularity_shower_LL_rej(parton=d2,
                                 ang_init=ang_final,
                                 beta=beta,
                                 jet_type=jet_type,
                                 jet=jet,
                                 split_soft=split_soft,
                                 test_num=test_num)
