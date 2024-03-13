
from models.mav_dynamics_control import MavDynamics
from message_types.msg_delta import MsgDelta
from scipy.optimize import minimize
import numpy as np

def compute_trim(mav: MavDynamics, delta: MsgDelta):
    
    # parameters to input for the trim longitudinally is (alpha, elevator, throttle)
    
    # guess input
    x0 = [mav._alpha, delta.elevator, delta.throttle]
    
    # the boundry of input parameters
    bounds = [(0, np.deg2rad(12)), (-1, 1), (0, 1)]
    
    # res is the results of optimization calculate_trim_output function defined in mav_dynamic_control.py
    # to find the optimize values for alpha = x[0]; delta.elevator = x[1]; delta.elevator = x[2]
    # such that the obejective function is minimized
    res = minimize(mav.calculate_trim_output, x0, bounds=bounds, method='SLSQP')
    
    # print(res)
    return(res.x[0], res.x[1], res.x[2])