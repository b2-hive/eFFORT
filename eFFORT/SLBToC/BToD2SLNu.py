import numpy as np
from eFFORT.utility import BGL_form_factor, z_var, PDG, w
import abc
import scipy.integrate
import functools


class BToD2SLNu:

    def __init__(self, m_B: float, m_D2S: float, V_cb: float, eta_EW: float = 1.0066) -> None:
        # Some of the following can be inherited from a parent class / initialized from a super constructor in the
        # future.
        self.m_B = m_B
        self.m_D2S = m_D2S
        self.V_cb = V_cb
        self.eta_EW = eta_EW
        self.G_F = PDG.G_F

        self.w_min = 1
        self.w_max = (m_B ** 2 + m_D2S ** 2) / (2 * m_B * m_D2S)

        # Variables which are often used and can be computed once
        self.r = self.m_D2S / self.m_B
        #self.rprime = 2 * np.sqrt(self.m_B * self.m_D2S) / (self.m_B + self.m_D2S)

        self._gamma_int = self._Gamma()

    @abc.abstractmethod
    def G(self, w: float) -> float:
        pass

    def dGamma_dw(self, w):
        # For easier variable handling in the equations
        m_B = self.m_B
        m_D = self.m_D2S

        return self.G_F**2 * m_D**3 / 48 / np.pi**3 * (m_B + m_D) *2 * (w**2 - 1)**(3/2) * self.eta_EW ** 2 * self.V_cb ** 2 * self.G(w)**2

    def _Gamma(self):
        w_min = 1
        w_max = (self.m_B ** 2 + self.m_D2S ** 2) / (2 * self.m_B * self.m_D2S)
        return scipy.integrate.quad(self.dGamma_dw, w_min, w_max)[0]


class BToD2SLNuBLT(BToD2SLNu):

    def __init__(self, m_B: float, m_D2S: float, V_cb: float, eta_EW: float = 1.0066, beta_coeff=(0.13, 1.95, -6.75)):

        self.beta_0 = beta_coeff[0]
        self.beta_1 = beta_coeff[1] 
        self.beta_2 = beta_coeff[2] 

        super(BToD2SLNuBLT, self).__init__(m_B, m_D2S, V_cb, eta_EW)
        #super().__init__(m_B, m_D2S, V_cb, eta_EW)

        


    def G(self, w: float) -> float:
        self.fp = self.beta_0 + self.beta_1*(w-1.0) + self.beta_2*(w-1.0)*(w-1.0)
        # Debugging to find discrepancy between generated and modeled BLT data
        # self.fp = 1
        
        return self.fp


        

class BToD2SLNuISGW2(BToD2SLNu):

    def __init__(self, m_B: float, m_D2S: float, V_cb: float, eta_EW: float = 1.0066):

        super(BToD2SLNuISGW2, self).__init__(m_B, m_D2S, V_cb, eta_EW)

        # ISGW2 specifics
        

    def G(self, w: float) -> float:

        #Probably quark masses b and d quark in GeV: See ISGW1 below equation (14), but b quark mass would be slightly off
        #Not actual masses but masses if only valence quarks contributed to hadron mass?
        msb=5.2
        msd=0.33
        
        # Beta_B^2
        bb2=0.431*0.431

        # Hyperfine-averaged physical mass of B-meson
        mbb=5.31

        # Number of flavours below b 
        nf = 4.0

        # Mass of decay meson b->qlv (from now on called daughter meson), in this case should be a charm quark
        msq=1.82
        # 
        bx2=0.45*0.45
        # Probably Hyperfine-averaged physical mass of daughter meson
        mbx=0.75*2.01+0.25*1.87
        # N_f^' (N f prime): Number of flavours below daughter meson (probably)
        nfp = 3.0

        # Probably m_B(meson) = m_b(quark) + m_d(quark)    mtb for m tilde B: see ISGW1 page 804, second column, first line
        mtb = msb + msd
        # Probalby same here: m_X(daughter meson) = m_(daughter quark q: c or u quark) + m_d(quark)
        mtx = msq + msd

        # B-Meson mass in GeV, already defined in class variable m_B #############################################################################################
        mb=self.m_B
        # Mass of X in B->Xlv
        mx=self.m_D2S

        # ISGW1 Equation (B4): mu_+ and mu_-
        mup=1.0/(1.0/msq+1.0/msb)
        mum=1.0/(1.0/msq-1.0/msb)

        # ISGW1 Equation (B2): Beta_BX^2 meaning ???
        bbx2=0.5*(bb2+bx2)
        # ISGW1 Equation (B3): Maximum momentum transfer 
        tm=(mb-mx)*(mb-mx)
        t = mb ** 2 + mx ** 2 - 2 * w * mb * mx
        # t = self.q2(w)

        # Debugging
        a = [1.2, 2.3, 3.4, 4.5]
        self.typetest3 = a
        self.typetest = t
        self.typetest2 = tm

        # If t=q2 above maximum, reduce it accordingly
        try:
            for i in t:
                if i > tm:
                    i=0.99*tm
        except TypeError:
            if t > tm:
                t = 0.99*tm
        # Equation (20): w~
        wt=1.0+(tm-t)/(2.0*mbb*mbx)

        # Quark model scale where running coupling has been assumed to saturate (see APPENDIX A)  from page 21, first paragraph
        mqm = 0.1

        # Strong coupling constant at scale mqm
        As0 = self.Getas(mqm, mqm)

        # Strong coupling constant at scale mqs
        As2 = self.Getas(msq, msq)

        self.ratio = As0/As2

        # Equation (24) including (25): Convential charge radius r^2
        # As0 = alpha_s(mu_qm)     As2 = alpha_s(m_q)
        r2 = 3.0/(4.0*msb*msq) + 3*msd*msd/(2*mbb*mbx*bbx2) + (16.0/(mbb*mbx*(33.0-2.0*nfp)))*np.log(As0/As2)

        # ISGW1 Equation (B1) but not exactly. Some approximation?
        # See first sentence second paragraph of APPENDIX C: Leads to equation (27) which replaces exp(..) in (B1) with term in (27) where N=2 it seems.
        # N = 2 + n + n'       n and n' are the harmonicoscillator quantum numbers of the initial and final wavefunctions 
        # (i.e., N=2 for S-wave to S-wave, N=3 for S-wave to P-wave, N=4 for S-wave to S′-wave, etc.)
        f3 = np.sqrt(mtx/mtb) * (np.sqrt(bx2*bb2)/bbx2)**1.5 / (1.0+r2*(tm-t)/24.0)**4.0

        # Equation (?): F_3^(f_+ + f_-)      See first few sentences in second paragraph of APPENDIX C
        f3fppfm = f3 * (mbb/mtb)**(-0.5) * (mbx/mtx)**0.5
        # Equation (?): F_3^(f_+ - f_-)
        f3fpmfm = f3 * (mbb/mtb)**0.5 * (mbx/mtx)**(-0.5)

        # Equation (126) 
        tau = msd*msd*bx2*(wt-1)/(bb2*bbx2)
        # Equation (124)
        udef = (bb2-bx2)/(2.0*bbx2) + bb2*tau/(3.0*bbx2)
        # Equation (125)
        vdef = bb2 * (1.0 + msq/msb) * (7.0 - bb2*(5+tau)/bbx2) / (6.0*bbx2)

        # Debugging
        self.db_tau = tau
        self.db_u = udef
        self.db_v = vdef

        # Equation (122): f_+ + f_- 
        fppfm = f3fppfm*np.sqrt(1.5) * ((1.0-(msd/msq)) * udef-(msd*vdef/msq))
        # Equation (123): f_+ - f_- 
        fpmfm = f3fpmfm*np.sqrt(1.5) * (mtb/msq) * (udef+(msd*vdef/mtx))

        fppf = (fppfm + fpmfm) / 2.0
        fpmf = (fppfm - fpmfm) / 2.0

        return fppf



    def q2(self, w):
        q2 = (self.m_B ** 2 + self.m_D2S ** 2 - 2 * w * self.m_B * self.m_D2S)
        return q2

    def Getas(self, massq, massx):
        lqcd2 = 0.04
        nflav = 4
        temp = 0.6

        if massx > 0.6:
            if massq < 1.85:
                nflav = 3
            
            temp = 12.0*np.pi / (33.0-2.0*nflav*np.log(massx*massx/lqcd2))
        
        return temp
