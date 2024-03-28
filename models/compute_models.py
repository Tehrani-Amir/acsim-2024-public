"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import quaternion_to_euler, euler_to_quaternion
from tools.rotations import quaternion_to_rotation, euler_to_rotation 
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta

# this function is used to save the data into model_coef.py
def compute_model(mav, trim_state, trim_input):
    
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()

# this function is used to calculate coefficients 
def compute_tf_model(mav, trim_state, trim_input):
    
    # trim values
    mav._update_velocity_data()
    mav._state = trim_state
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    ###### TODO ######
    
    # delta_e = trim_input.elevator, 
    # delta_a = trim_input.aileron, 
    # delta_r = trim_input.rudder, 
    # delta_t = trim_input.throttle
    delta_e, delta_a, delta_r, delta_t = trim_input
    
    # define transfer function constants, chapter 5.4
    a_phi1 = -0.5 * MAV.rho * MAV._Va2**2 * MAV.S_wing * MAV.b * MAV.C_p_p * MAV.b / (2*MAV._Va)
    a_phi2 = +0.5 * MAV.rho * MAV._Va2**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    
    a_theta1 = -0.5 * MAV.rho * MAV._Va2**2 * MAV.S_wing * MAV.c * MAV.C_m_q *MAV.c / (2*MAV._Va * MAV.Jy)
    a_theta2 = -0.5 * MAV.rho * MAV._Va2**2 * MAV.S_wing * MAV.c * MAV.C_m_alpha  / (MAV.Jy)
    a_theta3 = +0.5 * MAV.rho * MAV._Va2**2 * MAV.S_wing * MAV.c * MAV.C_m_delta_e  / (MAV.Jy)

    # Compute transfer function coefficients using new propulsion model
    a_V1 = (1/MAV.mass) * MAV.rho * Va_trim * MAV.S * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * delta_e) + (1/MAV.mass) * MAV.rho * Va_trim *MAV.S_prop * MAV.C_prop
    a_V2 = (1/MAV.mass) * MAV.rho  * MAV.S_prop * MAV.C_prop * MAV.k_motor**2 * delta_t
    
    chi_trim = 0
    a_V3 = MAV.gravity * np.cos(theta_trim - chi_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    
    ##### TODO #####
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    
    # extract longitudinal states (u, w, q, theta, pd)
    A_lon = np.zeros((5,5))
    
    A_lon[0,0] = A[3,3]
    A_lon[0,1] = A[3,5]
    A_lon[0,2] = A[3,10]
    A_lon[0,3] = A[3,7]
    A_lon[0,4] = A[3,2]
    A_lon[1,0] = A[5,3]
    A_lon[1,1] = A[5,5]
    A_lon[1,2] = A[5,10]
    A_lon[1,3] = A[5,7]
    A_lon[1,4] = A[5,2]
    A_lon[2,0] = A[10,3]
    A_lon[2,1] = A[10,5]
    A_lon[2,2] = A[10,10]
    A_lon[2,3] = A[10,7]
    A_lon[2,4] = A[10,2]
    A_lon[3,0] = A[7,3]
    A_lon[3,1] = A[7,5]
    A_lon[3,2] = A[7,10]
    A_lon[3,3] = A[7,7]
    A_lon[3,4] = A[7,2]
    A_lon[4,0] = A[2,3]
    A_lon[4,1] = A[2,5]
    A_lon[4,2] = A[2,10]
    A_lon[4,3] = A[2,7]
    A_lon[4,4] = A[2,2]

    print(A_lon)
    
    eigenvalues, eigenvectors = np.linalg.eig(A_lon)
    print(eigenvalues)
    
    B_lon = np.zeros((5,2))
    # change pd to h

    # extract lateral states (v, p, r, phi, psi)
    A_lat = np.zeros((5,5))
    B_lat = np.zeros((5,2))
    
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
        ##### TODO #####
    x_euler = np.zeros((12,1))

    # e0 = x_quat[6]
    # e1 = x_quat[7]
    # e2 = x_quat[8]
    # e3 = x_quat[9]
    # phi, theta, psi = quaternion_to_euler(np.matrix([e0, e1, e2, e3]))
    
    # x_euler[0] = x_quat[0]
    # x_euler[1] = x_quat[1]
    # x_euler[2] = x_quat[2]
    # x_euler[3] = x_quat[3]
    # x_euler[4] = x_quat[4]
    # x_euler[5] = x_quat[5]
    # x_euler[6] = phi
    # x_euler[7] = theta
    # x_euler[8] = psi
    # x_euler[9] = x_quat[10]
    # x_euler[10] = x_quat[11]
    # x_euler[11] = x_quat[12]
    
    pn = x_quat.item(0)
    pe = x_quat.item(1)
    pd = x_quat.item(2)
    u = x_quat.item(3)
    v = x_quat.item(4)
    w = x_quat.item(5)

    phi, theta, psi = quaternion_to_euler(x_quat[6:10])
    
    p = x_quat.item(9)
    q = x_quat.item(10)
    r = x_quat.item(11)
    
    x_euler = np.array([pn, pe, pd, u, v, w, phi, theta, psi, p, q, r])
    
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    ##### TODO #####
    # x_quat = np.zeros((13,1))

    pn =x_euler.item(0)
    pe = x_euler.item(1)
    pd = x_euler.item(2)
    u = x_euler.item(3)
    v = x_euler.item(4)
    w = x_euler.item(5)
    
    e0, e1, e2, e3 = euler_to_quaternion(x_euler[6:9])
    
    p = x_quat.item(10)
    q = x_quat.item(11)
    r = x_quat.item(12)
    
    x_quat = np.array([pn, pe, pd, u, v, w, e0, e1, e1, e2, p, q, r])
    
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of Quaternion2Euler(quat) with respect to quat
    # compute partial Quaternion2Euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    
    forces_moments = mav._forces_moments(delta)
    
    ##### TODO #####
    # f_euler_ = np.zeros((12,1))
    return_state = mav._f(x_quat, forces_moments)
    f_euler_ = euler_state(return_state)
    
    phi = x_euler[6]
    theta = x_euler[7]
    
    p = x_euler[9]
    q = x_euler[10]
    r = x_euler[11]
    
    phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = q * np.sin(phi) * (1/np.cos(theta)) + r * np.cos(phi) * (1/np.cos(theta))

    f_euler_[6] = phi_dot
    f_euler_[7] = theta_dot
    f_euler_[8] = psi_dot

    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation

    ##### TODO #####
    A = np.zeros((12, 12))  # Jacobian of f wrt x
    
    # A longitudinal 
    f_at_x = f_euler(mav, x_euler, delta)
    
    for i in range(0, 12):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps
        f_at_x_eps = f_euler(mav, x_eps, delta)
        df_dxi = (f_at_x_eps - f_at_x) / eps
        A[:,i] = df_dxi[:,0]
        
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # deviation

    ##### TODO #####
    B = np.zeros((12, 4))  # Jacobian of f wrt u
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01

    ##### TODO #####
    dT_dVa = 0
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01

    ##### TODO #####
    dT_ddelta_t = 0
    return dT_ddelta_t
