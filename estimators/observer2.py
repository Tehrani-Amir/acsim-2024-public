"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors

# state = [north, east, altitude, u, v, w, phi, theta, psi, p, q, r]
# state = [alpha, beta, Va, Vg, gamma, chi, wn, we, wd]

# measurement = mav.sensors
# measurement = [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, mag_x, mag_y, mag_z]
# measurement = [abs_pressure, diff_pressure, gps_eta_n, gps_eta_e, gps_eta_d, gps_Vg, gps_course]

class Observer:
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        
        # initialized estimated state message
        self.estimated_state = MsgState()
        
        # use alpha filters to low pass filter gyros and accels
        # alpha = self.Ts/(self.Ts + tau) where tau is the LPF time constant

        ################################### TODO #############################
        ############################## Low-Pass Filter #######################
        # LPF for p,q, r estimation
        self.lpf_gyro_x = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_z)
        
        # LPF for theta, phi estimation
        self.lpf_accel_x = AlphaFilter(alpha=0.5, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.5, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.5, y0=initial_measurements.accel_z)
        
        # LPF for altitude and airspeed estimation
        self.lpf_abs = AlphaFilter(alpha=0.5, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.5, y0=initial_measurements.diff_pressure)
        
        ################################### TODO #############################
        ############################## Kalman Filter #########################
        # ekf for phi and theta (Extended Kalman Filter)
        self.attitude_ekf = EkfAttitude()
        
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

        ################################# UpDATE ##############################
    def update(self, measurement):
        ##### TODO #####
        ################################ LPF Update ###########################
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)
     
        # invert sensor model to get altitude and airspeed
        # self.estimated_state.altitude = (1/(CTRL.rho * CTRL.gravity))*self.lpf_abs.update(measurement.abs_pressure)
        P0 = 101325
        T0 = 288.15
        L0 = -0.0065
        R = 8.31432
        M = 0.0289644
        gravity = CTRL.gravity
        P = self.lpf_abs.update(measurement.abs_pressure)
        self.estimated_state.altitude = (T0/L0) * (1- (P/P0)**((R*L0)/(gravity*M)))
        
        self.estimated_state.Va = np.sqrt((2.0/CTRL.rho)*self.lpf_diff.update(measurement.diff_pressure))
        
        # self.estimated_state.phi = np.arctan2(self.lpf_accel_y.update(measurement.accel_y), self.lpf_accel_z.update(measurement.accel_z))
        # self.estimated_state.theta = np.arcsin(self.lpf_accel_y.update(measurement.accel_y) / CTRL.gravity)
        
        ################################# EKF Update ###########################
        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(measurement, self.estimated_state)

        # NOT estimating these parameters
        self.estimated_state.alpha = 0.0
        self.estimated_state.beta = 0.0
        
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        
        return self.estimated_state

############################## Low-Pass Filter #######################
class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0         # initial condition

    # u is the measurements
    def update(self, u):
        ##### TODO #####
        self.y = self.alpha * self.y + (1-self.alpha) * u
        return self.y

############################## Kalman Filter #########################
########################## Attitude Estimation #########################
class EkfAttitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        
        ##### TODO #####
        # Process noise covariance matrix 2x2
        self.Q = np.eye(2) * 1e-10                           # np.diag([0.0, 0.0])
        
        # Gyroscope noise covariance matrix 3x3
        self.Q_gyro = np.eye(3) * SENSOR.gyro_sigma**2       # np.diag([0.00, 0.00, 0.00])

        # Accelerometer noise covariance matrix 3x3
        self.R_accel = np.eye(3) * SENSOR.accel_sigma**2     # np.diag([0.0, 0.0, 0.0])

        self.N = 10  # number of prediction step per sample
        
        # Sampling time
        self.Ts = SIM.ts_control/self.N        
        
        # xhat =    np.array([[phi], [theta]])
        self.xhat = np.array([[0.0], [0.0]])                # initial state estimate: phi, theta
        self.P = np.diag([0, 0]) * np.pi**2                 # initial estimate covariance matrix  # np.eye(2) * 0.1
        self.P = np.diag([0.1, 0.1])
        
        self.gate_threshold = 2.0                           #stats.chi2.isf(q=0.99, df=2)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, xhat, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        # xhat=[phi, theta] and measurement=[p, q, r, Va]
        ##### TODO #####         
        
        # p = measurement.gyro_x
        # q = measurement.gyro_y
        # r = measurement.gyro_z
        # Va = np.sqrt(measurement.diff_pressure / (0.5*CTRL.rho))

        p = state.p
        q = state.q
        r = state.r
        
        phi = xhat.item(0)
        theta = xhat.item(1)
        
        # G = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        #               [0.0, np.cos(phi), -np.sin(phi)]])
        # u = np.array([[state.p, state.q, state.r]]).T
        # f_ = G @ u
        
        f_ = np.array([[p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)],
                       [q * np.cos(phi) - r * np.sin(phi)]])                # np.zeros((2,1))
                
        return f_

    def h(self, xhat, measurement, state):
        # measurement model y
        # xhat=[phi, theta] and measurement=[p, q, r, Va]
        ##### TODO #####
        phi = xhat.item(0)      # xhat[0]
        theta = xhat.item(1)    # xhat[1]
     
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        
        # h_ = np.array([[x-accel], [y-accel], [z-accel]])
        # h_ = np.array([[q * Va * np.sin(theta) + measurement.accel_x],                          # + CTRL.gravity * np.sin(theta)
        #                [r * Va * np.cos(theta) - p * Va * np.sin(theta) + measurement.accel_y], # - CTRL.gravity * np.cos(theta) * np.sin(phi)
        #                [-q * Va * np.cos(theta) + measurement.accel_z]])                        # - CTRL.gravity * np.cos(theta) * np.cos(phi)
        
        h_ = np.array([[q * Va * np.sin(theta) + CTRL.gravity * np.sin(theta)],
                       [r * Va * np.cos(theta) - p * Va * np.sin(theta) - CTRL.gravity * np.cos(theta) * np.sin(phi)],
                       [-q * Va * np.cos(theta) - CTRL.gravity * np.cos(theta) * np.cos(phi)]])

        return h_

    def propagate_model(self, measurement, state):
        # model propagation       
        ##### TODO #####
        Tp = self.Ts
    
        for i in range(0, self.N):
            # propagate model
            # self.xhat = np.zeros((2,1))
            f_ = self.f(self.xhat, measurement, state)
            self.xhat += self.Ts * f_
            
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)

            phi = self.xhat.item(0)
            theta = self.xhat.item(1)
            
            # compute G matrix for gyro noise           
            G = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi)*np.tan(theta)],
                          [0.0, np.cos(phi), -np.sin(phi)]])
            G_d = G * self.Ts
            # convert to discrete time models
            A_d = np.eye(2) + A * self.Ts + (A @ A) * (self.Ts**2.0) / 2.0

            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + G_d @ self.Q_gyro @ G.T + self.Q * self.Ts**2
            # self.P = self.P + (self.Ts/self.N) * (A @ self.P + self.P @ A.T + self.Q) #np.zeros((2,2))
            
    def measurement_update(self, measurement, state):
        # measurement updates
        ##### TODO #####        

        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T
        
        # # @ Matrix multiplication, outer is Vector outer product
        # S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_accel)      # np.zeros((3,3))
        
        # if (y - h).T @ S_inv @ (y - h) < self.gate_threshold:
        #     L = self.P @ C.T @ S_inv
        #     self.P = (np.eye(2) - L @ C) @ self.P               # np.zeros((2,2))
        #     self.xhat += L @ (y - h)                            # np.zeros((2,1))

        for i in range(0, 3):
            if np.abs(y.item(i)-h.item(i)) < self.gate_threshold:
                
                Ci = C[i,:]
                L = self.P @ Ci.T / (self.R_accel[i,i] + Ci @ self.P @ Ci.T)
                self.P = (np.eye(2) - np.outer(L, Ci)) @ self.P

                change = L*(y.item(i) - h.item(i))
                
                self.xhat += np.reshape(change, (2,1))
                # self.xhat += np.reshape(L,(2,1)) * (y.item(i) - h.item(i))
        
############################## Position Estimation #############################
class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):

        # Process noise covariance matrix
        self.Q = np.diag([
                    0.00000000000000001,  # pn
                    0.00000000000000001,  # pe
                    0.00100000000000000,  # Vg
                    0.01000000000000000,  # chi
                    0.00010000000000000,  # wn
                    0.00010000000000000,  # we
                    0.00100000000000000,  # psi #0.0001
                    ])

        self.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma**2,      #y_gps_n
                    SENSOR.gps_e_sigma**2,      # y_gps_e
                    SENSOR.gps_Vg_sigma**2,     # y_gps_Vg
                    SENSOR.gps_course_sigma**2, # y_gps_course
                    ])
        
        self.R_pseudo = np.diag([
                    2.0,  # pseudo measurement #1
                    2.0,  # pseudo measurement #2
                    ])
        
        self.R_pseudo = np.diag([0.01, 0.01]) # not sure what this should be exactly

        # self.Q = np.eye(7) * 0.00000000000000001
        # self.R_gps = np.eye(4) * 0.1
        # self.R_pseudo = np.eye(2) * 0.01
        
        self.N = 5  # number of prediction step per sample
                
        # Sampling time
        self.Ts = SIM.ts_control / self.N

        # xhat = np.array   ([[pn],  [pe],  [Vg],   [chi], [wn],  [we],  [psi]])
        self.xhat = np.array([[0.0], [0.0], [25.0], [0.0], [0.0], [0.0], [0.0]])
        
        self.P = np.eye(7) * 0.1    # np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.diag([10**4, 10**4, 1, np.pi**2, 1, 1, np.pi**2])
        self.P = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        self.gps_n_old = 9999           # 0
        self.gps_e_old = 9999           # 0
        self.gps_Vg_old = 9999          # 0
        self.gps_course_old = 9999      # 0
        
        self.pseudo_threshold = 0       # stats.chi2.isf(0.01)
        self.gps_threshold = 100000     # don't gate GPS
        
    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, xhat, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        # xhat = [pn, pe, Vg, chi, wn, we, psi]
        # measurement = [Va, q, r, phi, theta]
         
        # Va = measurement.item(0)
        # q = measurement.item(1)
        # r = measurement.item(2)
        # phi = measurement.item(3)
        # theta = measurement.item(4)
         
        Vg = xhat[2, 0]       # xhat.item(2)
        chi = xhat[3, 0]      # xhat.item(3)
        wn = xhat[4, 0]       # xhat.item(4)
        we = xhat[5, 0]       # xhat.item(5)
        psi = xhat[6, 0]      # xhat.item(6)
                
        Va = state.Va
        q = state.q
        r = state.r
        phi = state.phi
        theta = state.theta

        # psi_dot = state.p
        # Vg_dot = ((Va * np.cos(state.alpha)) - (CTRL.gravity * np.sin(theta - state.alpha)))

        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        
        if Vg == 0:
            Vg = 0.00000000001
        else:
            Vg = Vg
        
        Vg_dot = ((Va * np.cos(psi) + wn) * (-Va * psi_dot * np.sin(psi)) + (Va * np.sin(psi) + we) * (Va * psi_dot * np.cos(psi))) / Vg                
        
        f_ = np.array([[Vg * np.cos(chi)],
                       [Vg * np.sin(chi)],
                       [Vg_dot],
                       [(CTRL.gravity / Vg) * np.tan(phi) * np.cos(chi - psi) ],
                       [0.0],
                       [0.0],
                       [psi_dot],
                       ])
        return f_

    def h_gps(self, xhat, measurement, state):
        # measurement model for gps measurements
        # xhat = [pn, pe, Vg, chi, wn, we, psi]
        # measurement = [Va, q, r, phi, theta] 
        
        pn = xhat[0, 0]       # xhat.item(0)
        pe = xhat[1, 0]       # xhat.item(1)
        Vg = xhat[2, 0]       # xhat.item(2)
        chi = xhat[3, 0]      # xhat.item(3)
        
        h_ = xhat[0:4, :]

        h_ = np.array([[pn], [pe], [Vg], [chi]])
               
        return h_
   
    def h_pseudo(self, xhat, measurement, state):
        # pseudo measurement model for wind triangale pseudo measurement (p. 161)
        # the pseudo measurment values are equal to ZERO!
        # xhat = [pn, pe, Vg, chi, wn, we, psi]
        # measurement = [Va, q, r, phi, theta] 

        Vg = xhat[2, 0]       # xhat.item(2)
        chi = xhat[3, 0]      # xhat.item(3)
        wn = xhat[4, 0]       # xhat.item(4)
        we = xhat[5, 0]       # xhat.item(5)
        psi = xhat[6, 0]      # xhat.item(6)
        
        Va = state.Va
                       
        h_ = np.array([[Va * np.cos(psi) - Vg * np.cos(chi) + wn],  # y_wind_n
                       [Va * np.sin(psi) - Vg * np.sin(chi) + we],  # y_wind_e
                       ])
        return h_

    #############################################################
    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model          
            f_ = self.f(self.xhat, measurement, state)
            
            self.xhat += self.Ts * f_        # np.zeros((7,1))
            
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            
            # convert to discrete time models
            A_d = np.eye(7) + A * self.Ts + (A @ A * self.Ts**2.0) / 2.0
            
            # update P with discrete time model                     
            self.P = A_d @ self.P @ A_d.T + self.Q * self.Ts**2.0
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q) #np.zeros((7,7))
            
    ##############################################################
    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        
        # S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_pseudo) # np.zeros((2,2))
        
        # if (y - h).T @ S_inv @ (y - h) < self.pseudo_threshold:
        #     L = self.P @ C.T @ S_inv
        #     self.P = (np.eye(7) - L @ C) @ self.P               # np.zeros((7,7))
        #     self.xhat += L @ (y - h)                            # np.zeros((7,1))
        
        for i in range(0, 2):
            # if np.abs(y.item(i)-h.item(i)) < self.pseudo_threshold:
            Ci = C[i, :]
            L = self.P @ Ci.T / (self.R_pseudo[i, i] + Ci @ self.P @ Ci.T)
            self.P = (np.eye(7) - np.outer(L, Ci)) @ self.P
            
            change = L * (y.item(i) - h.item(i))
            self.xhat += np.reshape(change, (7, 1))
            
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            
            # wrap: modifying i-th element from current value to new value
            y_chi = wrap(measurement.gps_course, h[3, 0])
            
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T
            
            # S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_gps)    # np.zeros((4,4))
            # if (y - h).T @ S_inv @ (y - h) < self.gps_threshold:
            #     L = self.P @ C.T @ S_inv
            #     self.P = (np.eye(7) - L @ C) @ self.P               # np.zeros((7,7))
            #     self.xhat += L @ (y - h)                            # np.zeros((7,1))
            
            for i in range(0, 4):
                # if np.abs(y.item(i)-h.item(i)) < self.gps_threshold:
                Ci = C[i, :]
                L = self.P @ Ci.T / (self.R_gps[i, i] + Ci @ self.P @ Ci.T)
                self.P = (np.eye(7) - np.outer(L, Ci)) @ self.P
                change = L * (y.item(i) - h.item(i))
                self.xhat += np.reshape(change, (7, 1))
                
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


##################################################
def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J