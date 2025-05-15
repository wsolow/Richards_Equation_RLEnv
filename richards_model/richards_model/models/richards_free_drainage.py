"""
Implementation of the Richard's Irrigation Model using NumPy NdArrays

Written by Will Solow, 2025
"""
import numpy as np
from scipy.integrate import ode

from richards_model.models.states_rates import NDArray
from richards_model.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from richards_model.models.base_model import Model
from traitlets_pcse import Float
       
EPS = 1e-12
class RE_FreeDrain(Model):
    """Implements Richard's Equation Irrigation model
    """

    _INIT_COND = {"HS": 0, "FP":1}

    class Parameters(ParamTemplate):
        PSI_MIN    = Float(-99.)  # Water balance minimum
        PSI_MAX    = Float(-99.)  # Water balance maximum
        THETA_R    = Float(-99.)  # theta r parameter
        THETA_S    = Float(-99.)  # theta s parameter
        ALPHA      = Float(-99.)  # alpha parameter
        N          = Float(-99.)  # n parameter
        M          = Float(-99.)  # m parameter
        K_S        = Float(-99.)  # k s parameter
        SS         = Float(-99.)  # SS parameter
        NETA       = Float(-99.)  # Neta parameter
        MATRIC_POT = Float(-99.)  # Initial PSI

        INITCOND   = Float(-99.)  # initial bound condition (0,1)

        SPACE_S    = Float(-99.)  # Space step
        SD         = Float(-99.)  # Soil depth 
        TS         = Float(-99.)  # Time step

    class RateVariables(RatesTemplate):
        BCT = Float(-99.) # Irrigation pulses
        BCB = Float(-99.) # Bottom Drainage

    class StateVariables(StatesTemplate):
        PSI_0   = NDArray(-99.) # Initial Water balance
        PSI     = NDArray(-99.) # Current water balance
        THETA   = NDArray(-99.) # Theta values
        DV      = NDArray(-99.) # Soil water levels
        WBS     = Float(-99.)   # Total water balance in soil
        QT      = Float(-99.)   # Top soil water content
        QB      = Float(-99.)   # Bottom soil water content
      
    def __init__(self, parvalues:dict):

        super().__init__()

        self.params = self.Parameters(parvalues)

        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = np.arange((p.SPACE_S / 2), p.SD, 
                         p.SPACE_S)
        
        self._N = len(z)

        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT

        # Initial value for DV
        DV = np.zeros((self._N+2))
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = np.reshape(self.theta_func(INIT_PSI.reshape(-1)), INIT_PSI.shape)
        WBS = np.sum(THETA * p.SPACE_S)

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV, QT=DV[0], QB=DV[-1])
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        self.solver.set_integrator('vode', method='BDF', uband=1,lband=1)

    def calc_rates(self, IRRIG):
        """Calculates the rates for irrigation
        """
        r = self.rates
        p = self.params

        r.BCB = p.MATRIC_POT
        r.BCT = IRRIG

    def integrate(self):
        """
        Updates the state variable and checks for phenologic stages
        """
        p = self.params
        s = self.states
        r = self.rates

        self.solver.set_initial_value(s.DV.copy(), 0)

        ode_init_vals = (r.BCT[0], r.BCB[0], s.DV)
        self.solver.set_f_params(*ode_init_vals)
            
        self.solver.integrate(p.TS)
        s.DV = self.solver.y
        
        s.PSI = s.DV[1:-1]
        s.THETA = np.reshape(self.theta_func(s.PSI.reshape(-1)), s.PSI.shape)
        s.WBS = np.sum(s.THETA * p.SPACE_S)

    def get_output(self):
        """
        Return the Waterbalance as output
        """
        return self.states.WBS

    def reset(self):
        """
        Reset the model
        """
        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        z = np.arange((p.SPACE_S / 2), p.SD, 
                         p.SPACE_S)
        self._N = len(z)
        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT

        # Initial value for DV
        DV = np.zeros((self._N+2))
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = np.reshape(self.theta_func(INIT_PSI.reshape(-1)), INIT_PSI.shape)
        WBS = np.sum(THETA * p.SPACE_S)

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV)
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        self.solver.set_integrator('vode',method='BDF',uband=1,lband=1)

    def theta_func(self, psi):
        """
        Compute the theta function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)

        Se[ psi>0. ] = 1.0

        return p.THETA_R + (p.THETA_S - p.THETA_R) * Se

    def c_func(self, psi):
        """
        Compute the C Function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)

        Se[ psi>0. ] = 1.0

        dSedh = p.ALPHA * p.M / ( 1 - p.M ) * Se ** ( 1 / p.M ) * ( 1 - Se ** (1/p.M) ) ** p.M

        return Se * p.SS+( p.THETA_S - p.THETA_R) * dSedh

    def k_func(self, psi):
        """
        Compute the K function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)
        Se[ psi>0. ] = 1.0

        return p.K_S * Se ** p.NETA * ( 1 - ( 1 - Se ** (1 / p.M ) ) ** p.M)**2

    def c_inv_func(self, psi, psi_n):
        """
        Inverse of C Function
        """
        Cinv = 1 / self.c_func(psi)
        return Cinv

    def boundary_fluxes(self, BC_T, BC_B, psiTn, psiBn):
        """
        Compute boundary fluxes
        """

        # Upper BC: Type 2 specified infiltration flux
        qT = BC_T

        # Lower BC: Type 1 specified pressure head
        qB = self.k_func(np.array([psiBn]))

        return qT, qB
    
    def ode_func_blockcentered(self, t, DV, BC_T, BC_B, psi_n):
        """
        Function to be called by ODE solver. 
        Wrapper for compatibility with SciPy interface
        """
        return self.ode_func_call(t, DV, BC_T, BC_B, psi_n)

    def ode_func_call(self, t, DV, BC_T, BC_B, psi_n):
        """Ode function call for the derivative of 
        Richards Equation"""

        p = self.params
        # Unpack the dependent variable
        psi = DV[1:-1]
        psi_n = psi_n[1:-1]

        q = np.zeros(self._N+1)
        K = np.zeros(self._N+1)
        K = self.k_func(psi)
        Kmid = (K[1:]+K[:-1])/2.
        
        # Boundary fluxes
        qT,qB = self.boundary_fluxes(BC_T, BC_B, psi[0], psi[-1])
        q[0] = qT
        q[-1] = qB

        # Internal nodes
        q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S - 1)

        # Continuity
        Cinv = self.c_inv_func(psi, psi_n)
        dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S

        # Pack up dependent variable:
        dDVdt = np.hstack((np.array([qT]),dpsidt,np.array([qB])))

        return dDVdt
    
