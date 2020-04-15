import numpy as np
from eFFORT.utility import BGL_form_factor, z_var, PDG, w
import abc
import scipy.integrate
import functools


class BToDstar2SLNu:

    def __init__(self, m_B: float, m_Dstar2S: float, V_cb: float, eta_EW: float = 1.0066) -> None:
        # Some of the following can be inherited from a parent class / initialized from a super constructor in the
        # future.
        self.m_B = m_B
        self.m_Dstar2S = m_Dstar2S
        self.V_cb = V_cb
        self.eta_EW = eta_EW
        self.G_F = PDG.G_F

        self.w_min = 1
        self.w_max = (m_B ** 2 + m_Dstar2S ** 2) / (2 * m_B * m_Dstar2S)

        # Variables which are often used and can be computed once
        self.r = self.m_Dstar2S / self.m_B
        #self.rprime = 2 * np.sqrt(self.m_B * self.m_Dstar2S) / (self.m_B + self.m_Dstar2S)

        self._gamma_int = self._Gamma()

    @abc.abstractmethod
    def G(self, w: float) -> float:
        pass

    def dGamma_dw(self, w):
        # For easier variable handling in the equations
        m_B = self.m_B
        m_D = self.m_Dstar2S
        r = m_D/m_B

        return self.G_F**2 * self.eta_EW**2 * self.V_cb**2 * m_D**3 / (48*np.pi**3) * (m_B-m_D)**2 * np.sqrt(w**2-1) * (w+1)**2 * (1 + 4*w/(w+1) * (1-2*r*w+r**2)/(1-r)**2) * self.G(w)**2

    def _Gamma(self):
        w_min = 1
        w_max = (self.m_B ** 2 + self.m_Dstar2S ** 2) / (2 * self.m_B * self.m_Dstar2S)
        return scipy.integrate.quad(self.dGamma_dw, w_min, w_max)[0]


class BToDstar2SLNuBLT(BToDstar2SLNu):

    def __init__(self, m_B: float, m_Dstar2S: float, V_cb: float, eta_EW: float = 1.0066, beta_coeff=(0.0, 0.9812, 0.0)):

        self.beta_0 = beta_coeff[0]
        self.beta_1 = beta_coeff[1] 
        self.beta_2 = beta_coeff[2] 

        super(BToDstar2SLNuBLT, self).__init__(m_B, m_Dstar2S, V_cb, eta_EW)
        #super().__init__(m_B, m_Dstar2S, V_cb, eta_EW)

        


    def G(self, w: float) -> float:
        #self.fp = self.beta_0 + self.beta_1*(w-1.0) + self.beta_2*(w-1.0)*(w-1.0)
        # Debugging to find discrepancy between generated and modeled BLT data
        self.fp = 1
        
        return self.fp


    
