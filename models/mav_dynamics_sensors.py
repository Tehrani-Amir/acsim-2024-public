"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces

# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from message_types.msg_sensors import MsgSensors
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation, euler_to_quaternion

class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        
        self._ts_simulation = Ts
        
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        # self._state = np.array([[MAV.north0],  # (0)
        #                        [MAV.east0],   # (1)
        #                        [MAV.down0],   # (2)
        #                        [MAV.u0],    # (3)
        #                        [MAV.v0],    # (4)
        #                        [MAV.w0],    # (5)
        #                        [MAV.e0],    # (6)
        #                        [MAV.e1],    # (7)
        #                        [MAV.e2],    # (8)
        #                        [MAV.e3],    # (9)
        #                        [MAV.p0],    # (10)
        #                        [MAV.q0],    # (11)
        #                        [MAV.r0],    # (12)
        #                        [0],   # (13)
        #                        [0],   # (14)
        #                        ])
        
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        
        self.initialize_velocity(MAV.u0, 0., 0.)
        
        # initialize true_state message
        self.true_state = MsgState()
        
        # initialize the sensors message
        self._sensors = MsgSensors()
        
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        
        # timer so that gps only updates every ts_gps seconds
        # large value ensures gps updates at initial time.
        self._t_gps = 999.  
        
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
              
    ## Initialize velocity Function
    def initialize_velocity(self, Va, alpha, beta):
        self._Va = Va
        self._alpha = alpha
        self._beta = beta
        
        # calculate airspeed components (u=ur, v=vr, w=wr)
        self._state[3] = Va * np.cos(alpha) * np.cos(beta)
        self._state[4] = Va * np.sin(beta)
        self._state[5] = Va * np.sin(alpha) * np.cos(beta)
        
        # update velocity data and forces and moments
        self._update_velocity_data()
        # self._forces_moments(delta=MsgDelta())
        
        # update the message class for the true state
        self._update_true_state()

    ## Objective Longitudinal Trim Function
    def calculate_trim_output(self, x):
        alpha, elevator, throttle = x
        
        roll, pitch, yaw = quaternion_to_euler(self._state[6:10])
        
        # theta = alpha + gamma, in trim condition gamma is zero, so theta=alpha
        self._state[6:10] = euler_to_quaternion(roll, alpha, yaw)
        
        self.initialize_velocity(self._Va, alpha, self._beta)
        
        delta = MsgDelta()
        delta.elevator = elevator
        delta.throttle = throttle
        
        forces = self._forces_moments(delta=delta)
        
        # Cost Function
        return(forces[0]**2 + forces[2]**2 + forces[4]**2)

    #################################################################
    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
       
        # simulate rate gyros(units are rad / sec)
        # sensor_noise = np.radians(0.5)   # it can be used instead of SENSOR.gyro_sigma
        self._sensors.gyro_x = self.true_state.p + np.random.normal(0, SENSOR.gyro_sigma) # + SENSOR.gyro_x_bias
        self._sensors.gyro_y = self.true_state.q + np.random.normal(0, SENSOR.gyro_sigma) # + SENSOR.gyro_y_bias
        self._sensors.gyro_z = self.true_state.r + np.random.normal(0, SENSOR.gyro_sigma) # + SENSOR.gyro_z_bias
                
        # simulate accelerometers(units of g)
        self._sensors.accel_x = self.current_forces_moments[0][0] / MAV.mass + MAV.gravity * np.sin(self.true_state.theta) + np.random.normal(0, SENSOR.accel_sigma)
        self._sensors.accel_y = self.current_forces_moments[1][0] / MAV.mass - MAV.gravity * np.cos(self.true_state.theta) * np.sin(self.true_state.phi) + np.random.normal(0, SENSOR.accel_sigma)
        self._sensors.accel_z = self.current_forces_moments[2][0] / MAV.mass - MAV.gravity * np.cos(self.true_state.theta) * np.cos(self.true_state.phi) + np.random.normal(0, SENSOR.accel_sigma)

        # simulate magnetometers                
        # The magnetic inclination is the angle between the earth's surface and the magnetic field lines. 
        # Magnetic declination is the angle between the magnetic north of the compass and the true north.
        
        # Tulsa Magnetic inclination of 64 °degrees
        # Tulsa Magnetic declination of 2 °degrees
        # Tulsa Magnetic Strength of 50046 nT (nano-tesla)
        Magnetic_field_strength = 50046.60
        Magnetic_Declination = + np.radians(2)
        Magnetic_Inclination = + np.radians(64)
        
        M_h = np.sin(Magnetic_Inclination) * Magnetic_field_strength
        
        M_n = np.cos(self.true_state.psi - Magnetic_Declination) * M_h
        M_e = np.sin(self.true_state.psi - Magnetic_Declination) * M_h
        M_d = np.cos(Magnetic_Inclination) * Magnetic_field_strength
        
        m0_v1 = np.array([M_n, M_e, M_d])
        
        # Rotation Matrix from Body to Inertial Reference Frame
        R_b_v1 = euler_to_rotation(self.true_state.phi, self.true_state.theta, 0)        
        R_inv = np.linalg.inv(R_b_v1)
        m0_b = np.dot(R_inv, m0_v1)
        
        self._sensors.mag_x = m0_b[0] + np.random.normal(0, SENSOR.mag_sigma) # + SENSOR.mag_beta
        self._sensors.mag_y = m0_b[1] + np.random.normal(0, SENSOR.mag_sigma) # + SENSOR.mag_beta
        self._sensors.mag_z = m0_b[2] + np.random.normal(0, SENSOR.mag_sigma) # + SENSOR.mag_beta
        
        psi_mag = -np.arctan2(m0_v1[1], m0_v1[0])
        psi_out = psi_mag + Magnetic_Declination
        # print(np.rad2deg(psi_out)-np.rad2deg(self.true_state.psi))
        
        # simulate pressure sensors Based on Dr. Hook Formula
        P0 = 101325   # Standard Pressure at Sea Level (Pa=N/m^2)
        T0 = 288.15   # Standard Temperature at Sea Level (Kelvin)
        L0 = -0.0065  # the Laps Rate, Rate of Temperature Decrease in the lower atmposphere (K/m)
        R = 8.31432   # Universal gas Constant for Air (N.m/mol.K)
        M = 0.0289644 # Standard Molar Mass of Atmosphere Air
        
        P = P0 * (1 -(L0 * self.true_state.altitude / T0))**(MAV.gravity * M / (R*L0))
        self._sensors.abs_pressure = P + np.random.normal(0, SENSOR.abs_pres_sigma)
        self._sensors.diff_pressure = 0.5 * MAV.rho * self.true_state.Va**2 + np.random.normal(0, SENSOR.diff_pres_sigma)
                
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_n + np.random.normal(0, SENSOR.gps_n_sigma)
            self._gps_eta_e = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_e + np.random.normal(0, SENSOR.gps_e_sigma)
            self._gps_eta_h = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_h + np.random.normal(0, SENSOR.gps_h_sigma)
            
            self._sensors.gps_n = self.true_state.north + self._gps_eta_n
            self._sensors.gps_e = self.true_state.east + self._gps_eta_e
            self._sensors.gps_h = self.true_state.altitude + self._gps_eta_h
            
            self._sensors.gps_Vg = np.sqrt((self.true_state.Va * np.cos(self.true_state.psi) + self.true_state.wn)**2 + (self.true_state.Va * np.sin(self.true_state.psi) + self.true_state.we)**2) + np.random.normal(0, SENSOR.gps_Vg_sigma)
            self._sensors.gps_course = np.arctan2(self.true_state.Va * np.sin(self.true_state.psi) + self.true_state.we , 
                                                  self.true_state.Va * np.cos(self.true_state.psi) + self.true_state.wn) + np.random.normal(0, SENSOR.gps_course_sigma)
            
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    #################################################################

    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        
        super()._rk4_step(forces_moments)
        
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        
        # update the message class for the true state
        self._update_true_state()
    
    #################################################################

    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]
        
        ##### TODO #####
        # convert wind vector from world to body frame (self._wind = ?)
        # CHECK THIS SECTION (wind is in NED frame so it is required to convert into body frame)
        phi, theta, psi = quaternion_to_euler(self._state[6:10])

        # euler_to_rotation is Rb_i (from Body to Inertial), so we must tranpose it
        # CHECK THIS SECTION
        Ri_b = euler_to_rotation(phi, theta, psi).T
        steady_state = Ri_b*steady_state
                
        # define the ground velociyty in body frame
        Vg_b = self._state[3:6]
        
        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        Va_b = Vg_b - steady_state
        ur, vr, wr = Va_b[:,0]
        
        # compute airspeed (self._Va = ?)
        self._Va = np.linalg.norm(Va_b, axis=0)[0]
        
        # compute angle of attack 
        self._alpha = np.arctan2(wr, ur)
        
        # compute sideslip angle 
        self._beta = np.arcsin(vr/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        
        # import the rotation first at the top of the code, 
        # euler_to_rotation is Rb_i (from Body to Inertial), so we must tranpose it
        Ri_b = euler_to_rotation(phi, theta, psi).T

        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
                
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg_i = [[0], [0], [MAV.mass * MAV.gravity]]
        fg_b = np.matmul(Ri_b, fg_i)
 
        # compute Lift and Drag coefficients (CL, CD)
        M_minus = np.exp(-MAV.M*(self._alpha - MAV.alpha0))
        M_plus = np.exp(MAV.M*(self._alpha + MAV.alpha0))
        
        sigmoid = (1 + M_minus + M_plus)/((1 + M_minus) * (1 + M_plus))
        
        CL = (1-sigmoid)*(MAV.C_L_0 + MAV.C_L_alpha*self._alpha) + sigmoid*(2*np.sign(self._alpha)*np.sin(self._alpha)**2*np.cos(self._alpha))
        
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2/(np.pi*MAV.e*MAV.AR)
                
        # Define delta's: elevator, aileron, rudder, throttle
        delta_e = delta.elevator
        delta_a = delta.aileron
        delta_r = delta.rudder
        delta_t = delta.throttle
        
        q_bar = 0.5 * MAV.rho * (self._Va**2) * MAV.S_wing

        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = q_bar * (CL + MAV.C_L_delta_e * delta_e + MAV.C_L_q * MAV.c*q/(2*self._Va))
        F_drag = q_bar * (CD + MAV.C_D_delta_e * delta_e + MAV.C_D_q * MAV.c*q/(2*self._Va))
        
        # compute Longitudinal forces in body frame (fx, fz)
        fx_a = - np.cos(self._alpha)*F_drag + np.sin(self._alpha)*F_lift
        fz_a = - np.sin(self._alpha)*F_drag - np.cos(self._alpha)*F_lift 
        
        # compute Lateral force in body frame (fy)
        fy_a = q_bar * (MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*MAV.b*p/(2*self._Va) + MAV.C_Y_r*MAV.b*r/(2*self._Va) + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)
        
        # compute aerodynamic forces in body frame
        fa_b = [[fx_a], [fy_a], [fz_a]]
        
        # compute Logitudinal torque in body frame (My=m)
        m = q_bar * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*self._alpha + MAV.C_m_q*MAV.c*q/(2*self._Va) + MAV.C_m_delta_e*delta_e)

        # compute Lateral torques in body frame (Mx=l, Mz=n)
        l = q_bar * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p*MAV.b*p/(2*self._Va) + MAV.C_ell_r*MAV.b*r/(2*self._Va) + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        n = q_bar * MAV.b * (MAV.C_n_0 + MAV.C_n_beta*self._beta + MAV.C_n_p*MAV.b*p/(2*self._Va) + MAV.C_n_r*MAV.b*r/(2*self._Va) + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)
        
        # compute aerodynamic moments in body frame
        ma_b = np.array([[l], [m], [n]])
        
        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)
        
        fx_p = 0.5 * MAV.rho * MAV.S_prop * MAV.C_prop * ((MAV.k_motor * delta_t)**2 - self._Va**2)
        f_prop = [[fx_p], [0], [0]]
        
        Tp = -MAV.k_T_p * (MAV.K_omega * delta_t)**2
        Tp = 0
        m_prop = np.array([[Tp], [0], [0]])
        
        # compute total forces in body frame
        f =  fg_b + fa_b + f_prop
        fx = f[0][0]
        fy = f[1][0]
        fz = f[2][0]
        
        # compute total moments in body frame
        Moments = ma_b + m_prop
        Mx = Moments[0][0]
        My = Moments[1][0]
        Mz = Moments[2][0]
         
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        
        self.current_forces_moments = forces_moments
        
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        
        # compute thrust and torque due to propeller
        
        ##### TODO #####

        # Based on the New Version of Book
        thrust_prop = 0.5 * MAV.rho * MAV.S_prop * ((MAV.k_motor*delta_t)**2 - Va**2)
        torque_prop = 0

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information

        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        
        self.true_state.u = self._state.item(3)
        self.true_state.v = self._state.item(4)
        self.true_state.w = self._state.item(5)

        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
                
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]        
        self.true_state.Vg = np.linalg.norm(pdot)
        
        self.true_state.gamma = np.arcsin(-pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
                
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)

        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        
        self.true_state.camera_az = self._state.item(13)
        self.true_state.camera_el = self._state.item(14)
