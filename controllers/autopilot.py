"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np
import parameters.control_parameters as AP

# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control import PDControl
from controllers.pid_control import PIDControl
from controllers.tf_control import TFControl
from controllers.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from models.mav_dynamics_control import MavDynamics

################## saturation limits ######################
alpha_min = np.radians(-2)
alpha_max = np.radians(12)

gamma_min = np.radians(-15)
gamma_max = np.radians(15) 

roll_min = np.radians(-45)
roll_max = np.radians(+45)

course_min = np.radians(-180)
course_max = np.radians(+180)

################## Longitudinal gains ######################
# gains to regulate Va by throttle
airspeed_throttle_kp = 0.03
airspeed_throttle_ki = 0.02

# gains to regulate alpha by elevator
alpha_elevator_kp = -20
alpha_elevator_ki = -20
alpha_elevator_kd = -3

# gains to regulate gamma by alpha/theta (Alpha/Theta Controller)
gamma_alpha_kp = 1 
gamma_alpha_ki = 0.5
gamma_alpha_kd = 0.001 

## gains to regulate altitude by gamma (Gamma Controller)
altitude_gamma_kp = 0.15
altitude_gamma_ki = 0.0
altitude_gamma_kd = 0.05

###################### Lateral gains #######################
## gains to regulate beta by elevator
yaw_damper_kp = 10.0
yaw_damper_kd = 1.00

# gains to regulate roll angle by ailerons (Roll Controller)
roll_aileron_kp = 0.20
roll_aileron_ki = 0.1
roll_aileron_kd = 0.01

## gains to regulate course angle by roll angle (Chi Controller)
course_roll_kp = 1.0
course_roll_ki = 0.1
course_roll_kd = 0.01

######################### Autopilot ########################

class Autopilot:
    def __init__(self, ts_control, mav: MavDynamics, delta):
        
        ################################## Longitudinal Autopilots ################################
        # regulate airspeed by throttle    
        self.throttle_from_airspeed = PIControl(kp = airspeed_throttle_kp,
                                                ki = airspeed_throttle_ki,
                                                Ts = ts_control,
                                                min = 0.0,
                                                max = 1.0,
                                                init_integrator = delta.throttle / airspeed_throttle_ki)
               
        # regulate altitude by gamma (Altitude Controller)
        self.gamma_from_altitude = PIDControl(kp = altitude_gamma_kp,
                                              ki = altitude_gamma_ki,
                                              kd = altitude_gamma_kd,
                                              Ts = ts_control,
                                              min = gamma_min, 
                                              max= gamma_max,
                                              init_integrator = 0.0)
        
        # regulate gamma by alpha (Gamma Controller)
        self.alpha_from_gamma = PIDControl(kp = gamma_alpha_kp ,
                                           ki = gamma_alpha_ki,
                                           kd = gamma_alpha_kd,
                                           Ts = ts_control,
                                           min = alpha_min,
                                           max = alpha_max,
                                           init_integrator = mav.true_state.alpha / gamma_alpha_ki)
        
        # regulate alpha by elevator (Alpha Controller)
        self.elevator_from_alpha = PIDControl(kp = alpha_elevator_kp ,
                                              ki = alpha_elevator_ki,
                                              kd = alpha_elevator_kd,
                                              Ts = ts_control,
                                              min = -1.0,
                                              max = +1.0,
                                              init_integrator = delta.elevator / alpha_elevator_ki)
               
        ################################## Lateral Autopilots ################################
        # regulate beta by rudder
        self.yaw_damper = PDControl(kp = yaw_damper_kp ,
                                    kd = yaw_damper_kd,
                                    Ts = ts_control,
                                    limit = 1.0)

        # regulate roll angle (phi) by ailerons (Roll Controller)
        self.aileron_from_roll = PIDControl(kp = roll_aileron_kp,
                                            ki = roll_aileron_ki,
                                            kd = roll_aileron_kd,
                                            Ts = ts_control,
                                            min = -1.0,
                                            max = +1.0,
                                            init_integrator = delta.aileron / roll_aileron_ki)
                
        # regulate course angle (chi) by roll angle (phi) (Chi Controller)
        self.roll_from_course = PIDControl(kp = course_roll_kp,
                                           ki = course_roll_ki,
                                           kd = course_roll_kd,
                                           Ts = ts_control,
                                           min = roll_min,
                                           max = roll_max,
                                           init_integrator = 0)
        
    def update(self, commands, estimated_state): # cmd=command,state are the desired/command, current states
        
        ################################### TODO ###################################
        delta = MsgDelta(elevator=0, aileron=0, rudder=0, throttle=0)
        
        ################################ Longitudinal ##############################
        delta.throttle = self.throttle_from_airspeed.update(commands.airspeed_command, estimated_state.Va)

        cmd_gamma = self.gamma_from_altitude.update(commands.altitude_command, estimated_state.altitude)        
        cmd_alpha = self.alpha_from_gamma.update(cmd_gamma, estimated_state.gamma)
        delta.elevator = self.elevator_from_alpha.update(cmd_alpha, estimated_state.alpha)

        ################################## Lateral ##################################
        delta.rudder = self.yaw_damper.update(0, estimated_state.beta)
        cmd_roll = self.roll_from_course.update(commands.course_command, estimated_state.chi)
        delta.aileron = self.aileron_from_roll.update(cmd_roll, estimated_state.phi)
 
        ############# construct control outputs and commanded states ################
        self.commanded_state = MsgState()
        
        self.commanded_state.altitude = commands.altitude_command
        self.commanded_state.Va = commands.airspeed_command
        self.commanded_state.chi = commands.course_command
        
        
        self.commanded_state.alpha = cmd_alpha
        self.commanded_state.gamma = cmd_gamma
        self.commanded_state.theta = cmd_alpha + cmd_gamma
        
        self.commanded_state.beta = 0.0
        self.commanded_state.phi = cmd_roll

        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output