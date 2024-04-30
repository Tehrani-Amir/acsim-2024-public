"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import numpy as np
from scipy import stats
import parameters.aerosonde_parameters as MAV
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors

class Observer:
    
    ############################## Initilaization ################################
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        # use alpha filters to low pass filter gyros and accels
        # initialized estimated state message
        self.estimated_state = MsgState()
                
        ################################ Low-Pass Filter #########################
        # TODO: tune α =alpha. If u is noisy α_lpf should be close to unity 
        # LPF for estimate p, q, r 
        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        
        # LPF for theta, phi estimation
        self.lpf_accel_x = AlphaFilter(alpha=0.8, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.8, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.8, y0=initial_measurements.accel_z)
        
        # LPF for estimate altitude, airspeed 
        self.lpf_abs = AlphaFilter(alpha=0.5, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.5, y0=initial_measurements.diff_pressure)
        
        ######################### Extended Kalman Filter #########################
        # EKF for estimate xhat = [phi and theta]
        self.attitude_ekf = EkfAttitude()
        
        # EKF for estimate xhat = [pn, pe, Vg, chi, wn, we, psi]
        self.position_ekf = EkfPosition()
        # self.position_ekf.xhat = np.array([[initial_measurements.gps_n], [initial_measurements.gps_e], [initial_measurements.gps_Vg], [initial_measurements.gps_course], [0.0], [0.0], [initial_measurements.gps_course]])

    ################################### UpDATE ###################################
    def update(self, measurement, true_state):

        ############################## LPF Update ################################
        # update p, q, r estimation with simple LPF of gyro minus bias
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)
     
        # invert sensor model to get and update altitude and airspeed
        P = self.lpf_abs.update(measurement.abs_pressure)
        P0 = 101325   # Standard Pressure at Sea Level (Pa=N/m^2)
        T0 = 288.15   # Standard Temperature at Sea Level (Kelvin)
        L0 = -0.0065  # Rate of Temperature Decrease (K/m)
        R = 8.31432   # Universal gas Constant for Air (N.m/mol.K)
        M = 0.0289644 # Standard Molar Mass of Atmosphere Air
                        
        self.estimated_state.altitude = (T0/L0) * (1- (P/P0)**((R*L0)/(MAV.gravity*M)))
        # self.estimated_state.altitude = (P0 - P) / (MAV.rho * MAV.gravity)
        self.estimated_state.Va = np.sqrt((2.0 / MAV.rho) * self.lpf_diff.update(measurement.diff_pressure))
                
        ############################## EKF Update ################################
        # update phi, theta estimation with simple EKF
        self.attitude_ekf.update(measurement, self.estimated_state)

        # update pn, pe, Vg, chi, wn, we, psi estimation with simple EKF
        self.position_ekf.update(measurement, self.estimated_state)

        self.estimated_state.altitude = true_state.altitude
        # NOT estimating these parameters
        self.estimated_state.gamma = true_state.gamma
        self.estimated_state.alpha = true_state.alpha
        self.estimated_state.beta = true_state.beta
        
        self.estimated_state.bx = true_state.bx
        self.estimated_state.by = true_state.by
        self.estimated_state.bz = true_state.bz
        
        return self.estimated_state

################################## ALPHA Filter ##################################
class AlphaFilter:
    # alpha filter implements a simple low pass filter
    
    def __init__(self, alpha=0.5, y0=0.0): 
        self.alpha = alpha      # filter parameter
        self.y = y0             # initial condition
 
    def update(self, u):        # u is the measurements
        self.y = self.alpha * self.y + (1-self.alpha) * u
        return self.y 
 
########################## Attitude Estimation Using EKF ##########################
class EkfAttitude: 
    # continous-discrete EKF to estimate xhat=[phi, theta] given state=[p, q, r, Va]

    def __init__(self): 
         # Process noise covariance matrix 2x2 for xhat
        self.Q = np.eye(2) * 0.5
        
        # Gyroscope noise covariance matrix 3x3
        self.Q_gyro = np.eye(3) * SENSOR.gyro_sigma**2
        
        # Accelerometer noise covariance matrix 3x3
        self.R_accel = np.eye(3) * SENSOR.accel_sigma**2
        
        self.N = 10  # number of prediction step per sample
        
        # Sampling time
        self.Ts = SIM.ts_control/self.N        
        
        # initial state estimate [phi, theta] matrix 2x1
        self.xhat = np.array([[0.0], [0.0]])
        
        # initial estimate covariance matrix 2x2
        self.P = np.eye(2)
        
        # put a limit on y - h for acceleromteres in EKFAttitude=[phi, theta]
        # continuous chi-squared probability distribution
        self.accel_threshold = stats.chi2.isf(q=0.99, df=3)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, xhat, measurement, state):
        # system dynamics for propagation model 
        
        phi = xhat.item(0)
        theta = xhat.item(1)
        
        # state traisition matrix 2x3
        G = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)]])
        
        # state variables 3x1
        u = np.array([[state.p, state.q, state.r]]).T
        
        # xdot=[phi_dot, theta_dot] = f(x, u) matrix 2x1
        f_ = G @ u
                
        return f_ 
 
    def h(self, xhat, measurement, state):
        # accelerometer measurement model y=h(xhat, u) matrix 3x1
        # where h=[accel_x, accel_y, accel_z].T 
        phi = xhat.item(0)
        theta = xhat.item(1)
     
        h_ = np.array([[state.q * state.Va * np.sin(theta) + MAV.gravity * np.sin(theta)],
                       [state.r * state.Va * np.cos(theta) - state.p * state.Va * np.sin(theta) - 
                        MAV.gravity * np.cos(theta) * np.sin(phi)],
                       [-state.q * state.Va * np.cos(theta) - MAV.gravity * np.cos(theta) * np.cos(phi)]])

        return h_

    def propagate_model(self, measurement, state):
        # model propagation       
            
        for i in range(0, self.N):
            # propagate model
            f_ = self.f(self.xhat, measurement, state)
            self.xhat += self.Ts * f_        # matrix 2×1
            
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)

            phi = self.xhat.item(0)
            theta = self.xhat.item(1)

            # convert continuous state matrix A to discrete time models using Euler series Expansion for exp(A*Ts)
            A_d = np.eye(2) + A * self.Ts + (A @ A) * (self.Ts**2.0) / 2.0
            
            # continuous transition noise matrix G matrix for gyro noise
            G = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi)*np.tan(theta)],
                          [0.0, np.cos(phi), -np.sin(phi)]])
            
            # discretized transition noise matrix G_d for gyro noise [p, q, r]
            G_d = G * self.Ts

            # update P with discrete time model matrix 2x2
            self.P = A_d @ self.P @ A_d.T + G_d @ self.Q_gyro @ G.T + self.Q * self.Ts**2
            
            # # continuous-discrete propagation matrix 2x2
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q)
            
    def measurement_update(self, measurement, state):
        # measurement updates

        h = self.h(self.xhat, measurement, state)               # matrix 3x1
        C = jacobian(self.h, self.xhat, measurement, state)     # matrix 3x3
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T
        
        S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_accel)  # matrix 3x1

        # Estimation_Error = (y - h)
        if (y - h).T @ S_inv @ (y - h) < self.accel_threshold:
            L = self.P @ C.T @ S_inv
            # update P matrix 2x2 Based on New Version of BOOK
            self.P = (np.eye(2) - L @ C) @ self.P @ (np.eye(2) - L @ C).T + L @ self.R_accel @ L.T
            self.xhat += L @ (y - h)                            # matrix 2x1
        
########################## Psition Estimation Using EKF ##########################
class EkfPosition:
    # implement continous-discrete EKF to estimate 
    # xhat=[pn, pe, Vg, chi, wn, we, psi] given state=[Va, q, r, phi, theta]

    def __init__(self):
        # Process noise covariance matrix 7x7 for xhat
        self.Q = np.eye(7) * 0.5
        
        # Sensor noise covariance matrix 4x4 for gps 
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma**2,      #y_gps_n
                    SENSOR.gps_e_sigma**2,      # y_gps_e
                    SENSOR.gps_Vg_sigma**2,     # y_gps_Vg
                    SENSOR.gps_course_sigma**2, # y_gps_course
                    ])

        # pseudo measurement covariance matrix 2x2
        self.R_pseudo = np.eye(2) * 1e-3
        
        self.N = 10  # number of prediction step per sample
                
        # Sampling time
        self.Ts = SIM.ts_control / self.N

        # initial state estimate xhat matrix 7x1
        self.xhat = np.array([[0.0], [0.0], [25.0], [0.0], [0.0], [0.0], [0.0]])
        
        # initial estimate covariance matrix 7x7
        self.P = np.zeros(7)

        self.gps_n_old = 9999
        self.gps_e_old = 9999 
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

        # continuous chi-squared probability distribution
        self.pseudo_threshold = stats.chi2.isf(q=0.99, df=2)
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
        # system dynamics for propagation model
        
        Vg = xhat.item(2)
        chi = xhat.item(3)
        wn = xhat.item(4)
        we = xhat.item(5)
        psi = xhat.item(6)
        
        # psi_dot, Vg_dot Calculation
        psi_dot = (state.q * np.sin(state.phi) / np.cos(state.theta) + 
                  state.r * np.cos(state.phi) / np.cos(state.theta))
        
        Vg_dot = ((state.Va * np.cos(state.psi) + wn) * (-state.Va * psi_dot * np.sin(state.psi)) + 
                  (state.Va * np.sin(state.psi) + we) * (+state.Va * psi_dot * np.cos(state.psi))) / Vg             
        
        # xdot = f(x, u) matrix 7×1
        f_ = np.array([[Vg * np.cos(chi)],
                       [Vg * np.sin(chi)],
                       [Vg_dot],
                       [MAV.gravity * np.tan(state.phi) * np.cos(chi - psi) / Vg],
                       [0.0],
                       [0.0],
                       [psi_dot],
                       ])
        return f_

    def h_gps(self, xhat, measurement, state):
        # gps measurement model y=h(xhat, u) matrix 4×1
        # where y_gps = [gps_n, gps_e, gps_Vg, gps_course]
        h_ = xhat[0:4, :]               
        return h_
   
    def h_pseudo(self, xhat, measurement, state):
        # pseudo measurement model for wind triangale pseudo measurement y wind matrxi 2×1

        Vg = xhat.item(2)
        chi = xhat.item(3)
        wn = xhat.item(4)
        we = xhat.item(5)
        psi = xhat.item(6)
                               
        h_ = np.array([[state.Va * np.cos(psi) - Vg * np.cos(chi) + wn],  # y_wind_n
                       [state.Va * np.sin(psi) - Vg * np.sin(chi) + we],  # y_wind_e
                       ])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
       
        for i in range(0, self.N):
            # propagate model          
            f_ = self.f(self.xhat, measurement, state)
            self.xhat += self.Ts * f_        # matrix 7×1
            
            # compute Jacobian 
            A = jacobian(self.f, self.xhat, measurement, state)
             
            # convert continuous state matrix A to discrete time models using Euler series Expansion for exp(A*Ts)
            A_d = np.eye(7) + A * self.Ts + (A @ A * self.Ts**2.0) / 2.0
             
            # update P with discrete time model matrix 7×7                     
            self.P = A_d @ self.P @ A_d.T + self.Q * self.Ts**2.0
            
            # # continuous-discrete propagation matrix 7x7
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q)
            
    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement   y_wind
        h = self.h_pseudo(self.xhat, measurement, state)            # matrix 2×1
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)  # matrox 2×7
        y = np.array([[0, 0]]).T                                    # matrix 2×2
        
        S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_pseudo)     # matrix 2×2
        
        # Estimation_Error = (y - h)
        if (y - h).T @ S_inv @ (y - h) < self.pseudo_threshold:
            L = self.P @ C.T @ S_inv                                # matrix 7×2
            # update P matrix 7x7 Based on New Version of BOOK
            self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + L @ self.R_pseudo @ L.T
            self.xhat += L @ (y - h)                                # matrix 7×1
            
        # only update GPS when one of the signals changes y_gps
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)           # matrix 4×1
            C = jacobian(self.h_gps, self.xhat, measurement, state) # matrix 4×7
            
            # modifying chi ∈ [-𝜋, +𝜋]
            y_chi = wrap(measurement.gps_course, h[3, 0])           # scaler
            
            # Update y_gps
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T                               # matrix 4×1
            
            S_inv = np.linalg.inv(C @ self.P @ C.T + self.R_gps)    # matrix 4×4
            
            # Estimation_Error = (y - h)
            if (y - h).T @ S_inv @ (y - h) < self.gps_threshold:    
                L = self.P @ C.T @ S_inv                            # matrix 7×4
                # update P matrix 7x7 Based on New Version of BOOK
                self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + L @ self.R_gps @ L.T
                self.xhat += L @ (y - h)                            # matrix 7×1
                
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

##########################################################################
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
