"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np

class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####

        # Dryden gust model parameters (Based on Table 4.1-page 56 UAV book) 
        # Check the altitude and Turbulence
        
        Altitude = 200
        Turbulence = "light"

        if abs(Altitude)<600: # low altitude
            Lu = 200
            Lv = 200
            Lw = 50
            
            if Turbulence == "light":      # light turbulence
                delta_u = 1.06
                delta_v = 1.06
                delta_w = 0.7
            
            else:                          # moderate turbulence
                delta_u = 1.06
                delta_v = 1.06
                delta_w = 0.7  
        else:                              # medium altitude
            Lu = 533
            Lv = 533
            Lw = 533
            
            if Turbulence == "light":     # light turbulence
                delta_u = 1.5
                delta_v = 1.5
                delta_w = 1.5
            else:                         # moderate turbulence
                delta_u = 3.0
                delta_v = 3.0
                delta_w = 3.0
                
        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
                        
        # PLEASE REPLACE the CORRECT Value of Va
        Va = 25
        
        Hu = delta_u*np.sqrt((2*Va)/(Lu))
        Hv = delta_v*np.sqrt((3*Va)/Lv)
        Hw = delta_w*np.sqrt((3*Va)/Lw)
        
        # example of encoding system into num and den
        # Considering the system = (s + 2) / (s^3 + 4s^2 + 5s + 6)
        # num = np.array([[1, 2]]) and den = np.array([[1, 4, 5, 6]])

        self.u_w = TransferFunction(num=np.array([[Hu]]), den=np.array([[1, (Va/Lu)]]),Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[Hv, Va/(np.sqrt(3)*Lv)]]), den=np.array([[1, (2*Va/Lv), (Va/Lv)**2]]), Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[Hw, Va/(np.sqrt(3)*Lw)]]), den=np.array([[1, (2*Va/Lw), (Va/Lw)**2]]), Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

