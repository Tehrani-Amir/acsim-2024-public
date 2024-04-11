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
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation, euler_to_quaternion

class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        
        self.initialize_velocity(MAV.u0, 0., 0.)
      
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
        self._forces_moments(delta=MsgDelta())
        
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

    ###################################
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

    ###################################
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
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        
        # compute thrust and torque due to propeller
        
        ##### TODO #####
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max * delta_t

        # Angular speed of propeller (omega_p = ?)
        a = MAV.rho * (MAV.D_prop**5) * MAV.C_Q0 / (2*np.pi)**2
        b = MAV.rho * (MAV.D_prop**4) * MAV.C_Q1 * (self._Va) / (2*np.pi)
        c = MAV.rho * (MAV.D_prop**3) * MAV.C_Q2 * (self._Va)**2 + MAV.KQ * (- v_in /MAV.R_motor + MAV.i0)
        
        # Propeller Speed (rad/sec) - Operating Propeller Speed
        Omega_p = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        
        # n = Omega_p/(2*np.pi) is propeller speed (revolutions/sec)
        
        # the associated advanced ration (op means operating)
        # J_op = (2*np.pi * self._Va) / (Omega_p*MAV.D_prop)
        
        # thrust and torque due to propeller
        # CT = MAV.C_T0 + MAV.C_T1 * J_op + MAV.C_T2 * J_op**2
        # CQ = MAV.C_Q0 + MAV.C_Q1 * J_op + MAV.C_Q2 * J_op**2
        
        # Based on the pdf Book
        # thrust_prop = MAV.rho * (Omega_p/(2*np.pi)**2) * MAV.D_prop**4 * CT
        # torque_prop = MAV.rho * (Omega_p/(2*np.pi)**2) * MAV.D_prop**5 * CQ
                
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
        
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0