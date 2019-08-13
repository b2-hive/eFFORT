import abc

import numpy as np
import scipy.integrate
import uncertainties

from eFFORT.BRhoLepNuRateExp import getDiffRatedq2
from eFFORT.utility import PDG


class BToVLNu:
    """
    A class containing functions specific to the differential decay rate of the B to Dstar transitions with the BCL/BGL
    parametrization. If not states otherwise, the numerical values and variable/function definitions are taken from:
    https://arxiv.org/abs/1702.01521v2 and ...

    Zero mass approximation for the lepton is implicit.
    The Blaschke factors do not explicitly appear, because they are 1.
    """

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066) -> None:
        self.m_B = m_B
        self.m_V = m_V
        self.m_L = m_L
        self._V_ub = V_ub
        self.eta_EW = eta_EW

        self.N0 = PDG.G_F ** 2 / (192. * np.pi ** 3 * self.m_B ** 3)  # Save some computing time.

        self.tplus = (self.m_B + self.m_V) ** 2
        self.tminus = (self.m_B - self.m_V) ** 2
        self.tzero = self.tplus * (1 - (1 - self.tminus / self.tplus) ** 0.5)

        self.q2min = self.m_L ** 2
        self.q2min += 1e-3  # numerical stability
        self.q2max = self.m_B ** 2 + self.m_V ** 2 - 2 * self.m_B * self.m_V
        self.q2max -= 1e-3  # numerical stability

        self.gamma = None

    @property
    def V_ub(self):
        return self._V_ub

    @V_ub.setter
    def V_ub(self, V_ub):
        self._V_ub = V_ub
        self.gamma = None  # Clear cache of normalization integral when updating Vub

    def kaellen(self, q2):
        return ((self.m_B + self.m_V) ** 2 - q2) * ((self.m_B - self.m_V) ** 2 - q2)

    @abc.abstractmethod
    def A0(self, q2):
        return 0

    @abc.abstractmethod
    def A1(self, q2):
        return 0

    @abc.abstractmethod
    def A12(self, q2):
        return 0

    @abc.abstractmethod
    def V(self, q2):
        return 0

    @staticmethod
    def blaschke_pole(q2, m_pole):
        return (1 - q2 / m_pole ** 2) ** -1

    def z(self, q2):
        return ((self.tplus - q2) ** 0.5 - (self.tplus - self.tzero) ** 0.5) / (
                (self.tplus - q2) ** 0.5 + (self.tplus - self.tzero) ** 0.5)

    def Hplus(self, q2):
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_V) + (self.m_B + self.m_V) * self.A1(q2)

    def Hminus(self, q2):
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_V) - (self.m_B + self.m_V) * self.A1(q2)

    def Hzero(self, q2):
        return 8 * self.m_B * self.m_V / q2 ** 0.5 * self.A12(q2)

    def Hscalar(self, q2):
        return self.kaellen(q2) ** 0.5 / q2 ** 0.5 * self.A0(q2)

    def dGamma_dq2(self, q2):
        try:
            return self.N0 * self.V_ub ** 2 * self.kaellen(q2) ** 0.5 * q2 * (1 - self.m_L ** 2 / q2) ** 2 * (
                    (1 + self.m_L ** 2 / (2 * q2)) * (self.Hplus(q2) ** 2 + self.Hminus(q2) ** 2 + self.Hzero(q2) ** 2)
                    + (3 * self.m_L ** 2 / (2 * q2) * self.Hscalar(q2) ** 2)
            )
        except TypeError:
            return 0
        except ZeroDivisionError:
            return 0

    def deltaGamma_deltaq2(self, lower, upper):
        if lower < self.q2min:
            lower = self.q2min
        if upper > self.q2max:
            upper = self.q2max
        return scipy.integrate.quad(lambda x: uncertainties.nominal_value(self.dGamma_dq2(x)),
                                    lower, upper, epsabs=1e-20)[0] / (upper - lower)

    def Gamma(self):
        if self.gamma is None:
            self.gamma = self.deltaGamma_deltaq2(self.q2min, self.q2max)
            self.gamma *= (self.q2max - self.q2min)  # Total rate should not be divided by bin width
        return self.gamma


class BToVLNuBCL(BToVLNu):

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToVLNuBCL, self).__init__(m_B, m_V, m_L, V_ub, eta_EW)
        self._coefficients = None

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients
        self.gamma = None  # Clear cache of normalization integral when updating coefficients

    def form_factor(self, q2, m_pole, coefficients):
        return BToVLNu.blaschke_pole(q2, m_pole) * sum(
            [par * (self.z(q2) - self.z(0)) ** k for k, par in enumerate(coefficients)])

    def AP(self, q2):
        """Form factor under equations of motion."""
        # AP = -2 Mr / (m_b + m_u)  A0
        lambdaBar = 0.5
        m_b = self.m_B - lambdaBar
        m_u = 0
        return -2 * self.m_V / (m_b + m_u) * self.A0(q2)

    def A0(self, q2):
        m_pole = 5.279
        coefficients = [8 * self.m_B * self.m_V / (self.m_B ** 2 - self.m_V ** 2) * self.coefficients[5],
                        self.coefficients[0], self.coefficients[1]]

        return self.form_factor(q2, m_pole, coefficients)

    def A1(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[2], self.coefficients[3], self.coefficients[4]]

        return self.form_factor(q2, m_pole, coefficients)

    def A12(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[5], self.coefficients[6], self.coefficients[7]]

        return self.form_factor(q2, m_pole, coefficients)

    def V(self, q2):
        m_pole = 5.325
        coefficients = [self.coefficients[8], self.coefficients[9], self.coefficients[10]]

        return self.form_factor(q2, m_pole, coefficients)

    def T1(self, q2):
        m_pole = 5.325
        coefficients = [self.coefficients[11], self.coefficients[12], self.coefficients[13]]

        return self.form_factor(q2, m_pole, coefficients)

    def T2(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[11], self.coefficients[14], self.coefficients[15]]

        return self.form_factor(q2, m_pole, coefficients)

    def T23(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[16], self.coefficients[17], self.coefficients[18]]

        return self.form_factor(q2, m_pole, coefficients)

    def dGamma_dq2_NP(self, q2, WCs=None):
        if WCs is None:
            WCs = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        FFs = [self.AP(q2), self.V(q2), self.A0(q2), self.A1(q2), self.A12(q2), self.T1(q2), self.T2(q2), self.T23(q2)]
        glebsch_gordan_fix = 0.5
        return self._V_ub ** 2 * glebsch_gordan_fix * getDiffRatedq2(self.m_B, self.m_V, self.m_L, q2, WCs, FFs)


class BToVLNuEvtGen(BToVLNu):

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToVLNuEvtGen, self).__init__(m_B, m_V, m_L, V_ub, eta_EW)
        self.sse_parameters = [
            0.261, -0.29, -0.415, 1,  # A1
            0.223, -0.93, -0.092, 1,  # A2
            0.338, -1.37, 0.315, 1,  # V
            0.372, -1.40, 0.437, 1,  # A0
        ]

    def F(self, q2, pars):
        return pars[0] / (1 + pars[1] * (q2 / self.m_B ** 2) + pars[2] * (q2 / self.m_B ** 2) ** 2) ** pars[3]

    def A1(self, q2):
        return self.F(q2, self.sse_parameters[0:4])

    def A2(self, q2):
        return self.F(q2, self.sse_parameters[4:8])

    def V(self, q2):
        return self.F(q2, self.sse_parameters[8:12])

    def A0(self, q2):
        return self.F(q2, self.sse_parameters[12:16])

    def A12(self, q2):
        return ((self.m_B + self.m_V) ** 2 * (self.m_B ** 2 - self.m_V ** 2 - q2) * self.A1(q2) - self.kaellen(
            q2) * self.A2(q2)) / (16 * self.m_B * self.m_V ** 2 * (self.m_B + self.m_V))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from uncertainties import correlated_values
    from tabulate import tabulate
    from eFFORT.plotting import init_thesis_plot_style, plot_with_errorband

    init_thesis_plot_style()

    coefficient_labels = [
        r'$\alpha_1^{A_0}$', r'$\alpha_2^{A_0}$',
        r'$\alpha_0^{A_1}$', r'$\alpha_1^{A_1}$', r'$\alpha_2^{A_1}$',
        r'$\alpha_0^{A_{12}}$', r'$\alpha_1^{A_{12}}$', r'$\alpha_2^{A_{12}}$',
        r'$\alpha_0^{V}$', r'$\alpha_1^{V}$', r'$\alpha_2^{V}$',
        r'$\alpha_1^{T_1}$', r'$\alpha_2^{T_1}$',
        r'$\alpha_0^{T_2}$', r'$\alpha_1^{T_2}$', r'$\alpha_2^{T_2}$',
        r'$\alpha_0^{T_{23}}$', r'$\alpha_1^{T_{23}}$', r'$\alpha_2^{T_{23}}$',
    ]

    lcsr_Brho = np.load('inputs/Brho_LCSR_central.npy')
    lcsr_Brho_cov = np.load('inputs/Brho_LCSR_covariance.npy')
    lcsr_Bomega = np.load('inputs/Bomega_LCSR_central.npy')
    lcsr_Bomega_cov = np.load('inputs/Bomega_LCSR_covariance.npy')

    plt.imshow(lcsr_Brho_cov / np.outer(lcsr_Brho_cov.diagonal()**0.5, lcsr_Brho_cov.diagonal()**0.5),
               vmin=-1, vmax=1, cmap='seismic')
    plt.xticks(range(len(coefficient_labels)), coefficient_labels, rotation='vertical', fontsize=10)
    plt.yticks(range(len(coefficient_labels)), coefficient_labels, rotation='horizontal', fontsize=10)
    plt.colorbar()
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.savefig('lcsr_Brho_correlation.pdf')
    plt.show()
    plt.close()

    with open('lcsr_Brho_correlation.tex', 'w') as f:
        f.write(tabulate(lcsr_Brho_cov / np.outer(lcsr_Brho_cov.diagonal()**0.5, lcsr_Brho_cov.diagonal()**0.5),
                tablefmt='latex_raw', floatfmt='.2f'))

    plt.imshow(lcsr_Bomega_cov / np.outer(lcsr_Bomega_cov.diagonal()**0.5, lcsr_Bomega_cov.diagonal()**0.5),
               vmin=-1, vmax=1, cmap='seismic')
    plt.xticks(range(len(coefficient_labels)), coefficient_labels, rotation='vertical', fontsize=10)
    plt.yticks(range(len(coefficient_labels)), coefficient_labels, rotation='horizontal', fontsize=10)
    plt.colorbar()
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.savefig('lcsr_Bomega_correlation.pdf')
    plt.show()
    plt.close()

    with open('lcsr_Bomega_correlation.tex', 'w') as f:
        f.write(tabulate(lcsr_Bomega_cov / np.outer(lcsr_Bomega_cov.diagonal()**0.5, lcsr_Bomega_cov.diagonal()**0.5),
                tablefmt='latex_raw', floatfmt='.2f'))

    lcsr_Brho_correlated = correlated_values(lcsr_Brho, lcsr_Brho_cov)
    lcsr_Bomega_correlated = correlated_values(lcsr_Bomega, lcsr_Bomega_cov)

    bcl_rho = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.775, m_L=0, V_ub=3.72e-3)
    bcl_rho.coefficients = lcsr_Brho_correlated

    bcl_omega = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.782, m_L=0, V_ub=3.72e-3)
    bcl_omega.coefficients = lcsr_Bomega_correlated

    q2range = np.linspace(bcl_rho.q2min, bcl_rho.q2max)

    plot_with_errorband(q2range, bcl_rho.A1(q2range), label=r'$A_{1}$', ls='-')
    plot_with_errorband(q2range, bcl_rho.A12(q2range), label=r'$A_{12}$', ls='--')
    plot_with_errorband(q2range, bcl_rho.V(q2range), label=r'$V$', ls=':')
    plt.ylim(0, 2.3)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$F_i(q^2)$')
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Brho_FF.pdf')
    plt.show()
    plt.close()

    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.A1(q2range)], label=r'$A_{1}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.A12(q2range)], label=r'$A_{12}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.V(q2range)], label=r'$V$', ls='-')
    plt.ylim(0, 0.25)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$\sigma F_i(q^2)$ / $F_i(q^2)$')
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Brho_FF_relative_error.pdf')
    plt.show()
    plt.close()

    plot_with_errorband(q2range, bcl_omega.A1(q2range), label=r'$A_{1}$', ls='-')
    plot_with_errorband(q2range, bcl_omega.A12(q2range), label=r'$A_{12}$', ls='--')
    plot_with_errorband(q2range, bcl_omega.V(q2range), label=r'$V$', ls=':')
    plt.ylim(0, 2.3)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$F_i(q^2)$')
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Bomega_FF.pdf')
    plt.show()
    plt.close()

    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.A1(q2range)], label=r'$A_{1}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.A12(q2range)], label=r'$A_{12}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.V(q2range)], label=r'$V$', ls='-')
    plt.ylim(0, 0.25)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$\sigma F_i(q^2)$ / $F_i(q^2)$')
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Bomega_FF_relative_error.pdf')
    plt.show()
    plt.close()

    bcl_rho = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.775, m_L=0, V_ub=3.72e-3)
    bcl_rho.coefficients = lcsr_Brho

    bcl_omega = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.782, m_L=0, V_ub=3.72e-3)
    bcl_omega.coefficients = lcsr_Bomega

    evtgen_rho = BToVLNuEvtGen(m_B=PDG.m_Bzero, m_V=0.775, m_L=0, V_ub=3.72e-3)
    evtgen_omega = BToVLNuEvtGen(m_B=PDG.m_Bzero, m_V=0.782, m_L=0, V_ub=3.72e-3)

    plt.plot(q2range, bcl_rho.dGamma_dq2(q2range) / bcl_rho.Gamma(),
             label=r'$B \rightarrow \rho l \nu$ BCL', color='red')
    plt.plot(q2range, evtgen_rho.dGamma_dq2(q2range) / evtgen_rho.Gamma(),
             label=r'$B \rightarrow \rho l \nu$ EvtGen', color='red', ls='--')

    plt.plot(q2range, bcl_omega.dGamma_dq2(q2range) / bcl_omega.Gamma(),
             label=r'$B \rightarrow \omega(\rightarrow 3\pi) l \nu$ BCL', color='blue')
    plt.plot(q2range, evtgen_omega.dGamma_dq2(q2range) / evtgen_omega.Gamma(),
             label=r'$B \rightarrow \omega(\rightarrow 3\pi) l \nu$ EvtGen', color='blue', ls='--')

    plt.ylim(0, None)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$1/ \Gamma$ $\mathrm{d}\Gamma / \mathrm{d}q^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_normalized_rates.pdf')
    plt.show()
    plt.close()

    with open('lcsr_central_values.tex', 'w') as f:
        f.write(tabulate(zip(coefficient_labels,
                             lcsr_Brho, lcsr_Brho_cov.diagonal() ** 0.5,
                             lcsr_Bomega, lcsr_Bomega_cov.diagonal() ** 0.5),
                         [r'',
                          r'$B\rightarrow\rho$', r'$\sigma(B\rightarrow\rho)$',
                          r'$B\rightarrow\omega$', r'$\sigma(B\rightarrow\omega)$'],
                         tablefmt='latex_raw', floatfmt='.2f'))
