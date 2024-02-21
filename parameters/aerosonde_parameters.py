import numpy as np
from tools.rotations import euler_to_quaternion

######################################################################################
                #   Initial Conditions
######################################################################################
# Define the Initial conditions for MAV
north0 = 0.     # initial north position
east0 = 0.      # initial east position
down0 = -200.0  # initial down position

u0 = 25         # initial velocity along body x-axis
v0 = 0.1        # initial velocity along body y-axis
w0 = 0.1        # initial velocity along body z-axis

phi0 = 0        # initial roll angle
theta0 = 0      # initial pitch angle
psi0 = 0        # initial yaw angle

p0 = 0.1        # initial roll rate
q0 = 0.01       # initial pitch rate
r0 = 0.01       # initial yaw rate (0.5)

Va0 = np.sqrt(u0**2+v0**2+w0**2) #initial airspeed

#   Quaternion State
euler = euler_to_quaternion(phi0, theta0, psi0)
e0 = euler.item(0)
e1 = euler.item(1)
e2 = euler.item(2)
e3 = euler.item(3)

######################################################################################
                #   Physical Parameters
######################################################################################
mass = 11.              # mass (kg) should be 13.5 kg
Jx = 0.8244             # Inertia (kg.m^2)
Jy = 1.135              
Jz = 1.759              
Jxz = 0                 # 0.1204
S_wing = 0.55           # wing area
b = 2.8956              # wing span
c = 0.18994             # the mean chord of MAV
S_prop = 0.2027         # 
rho = 1.2682            # density
e = 0.9                 # Oswald efficiency factor
AR = (b**2) / S_wing    # wing aspect ratio
gravity = 9.81          # gravitional acceleration

######################################################################################
                #   Longitudinal Coefficients
######################################################################################
C_L_0 = 0.23            # 0.28 based on the Book, p 276, table E.2 foe Aerosonde UAV
C_D_0 = 0.043           # 0.03
C_m_0 = 0.0135          # -0.02338

C_L_alpha = 5.61        # 3.45
C_D_alpha = 0.03
C_m_alpha = -2.74       # -0.38

C_L_q = 7.95            # 0
C_D_q = 0.0
C_m_q = -38.21          # -3.6

C_L_delta_e = 0.13      # -0.36
C_D_delta_e = 0.0135    # 0
C_m_delta_e = -0.99     # -0.5

M = 50.0
alpha0 = 0.47           # 0.4712

epsilon = 0.16          # 0.1592
C_D_p = 0.0

######################################################################################
                #   Lateral Coefficients
######################################################################################
C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0

C_Y_beta = -0.98
C_ell_beta = -0.13      # -0.12
C_n_beta = 0.073        # 0.25

C_Y_p = 0.0
C_ell_p = -0.51         # -0.26
C_n_p = 0.069           # 0.022

C_Y_r = 0.0
C_ell_r = 0.25          # 0.14
C_n_r = -0.095          # -0.35

C_Y_delta_a = 0.075     # 0.0
C_ell_delta_a = 0.17    # 0.08
C_n_delta_a = -0.011    # 0.06

C_Y_delta_r = 0.19      # -0.17
C_ell_delta_r = 0.0024  # 0.105
C_n_delta_r = -0.069    # -0.032

######################################################################################
                #   Propeller thrust / torque parameters (see addendum by McLain)
######################################################################################
# Prop parameters
D_prop = 20*(0.0254)                              # prop diameter in m

# Motor parameters
KV_rpm_per_volt = 145.                            # Motor speed constant from datasheet in RPM/V
KV = (1. / KV_rpm_per_volt) * 60. / (2. * np.pi)  # Back-emf constant, KV in V-s/rad
KQ = KV                                           # Motor torque constant, KQ in N-m/A
R_motor = 0.042                                   # Resistance of the motot winding (ohms)
i0 = 1.5                                          # no-load (zero-torque) current (A)

# Inputs
ncells = 12.
V_max = 3.7 * ncells                              # max voltage for specified number of battery cells

# Aerodynamic Coeffiecients from prop_data fit
C_T0 = 0.09357
C_T1 = -0.06044
C_T2 = -0.1079

C_Q0 = 0.005230
C_Q1 = 0.004970
C_Q2 = -0.01664

######################################################################################
                #   Calculation Variables
######################################################################################
#   gamma parameters pulled from page 36 (dynamics)
gamma = Jx * Jz - (Jxz**2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz**2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz**2)) / gamma
gamma8 = Jx / gamma

#   C values defines on pag 62
C_p_0         = gamma3 * C_ell_0      + gamma4 * C_n_0
C_p_beta      = gamma3 * C_ell_beta   + gamma4 * C_n_beta
C_p_p         = gamma3 * C_ell_p      + gamma4 * C_n_p
C_p_r         = gamma3 * C_ell_r      + gamma4 * C_n_r
C_p_delta_a    = gamma3 * C_ell_delta_a + gamma4 * C_n_delta_a
C_p_delta_r    = gamma3 * C_ell_delta_r + gamma4 * C_n_delta_r
C_r_0         = gamma4 * C_ell_0      + gamma8 * C_n_0
C_r_beta      = gamma4 * C_ell_beta   + gamma8 * C_n_beta
C_r_p         = gamma4 * C_ell_p      + gamma8 * C_n_p
C_r_r         = gamma4 * C_ell_r      + gamma8 * C_n_r
C_r_delta_a    = gamma4 * C_ell_delta_a + gamma8 * C_n_delta_a
C_r_delta_r    = gamma4 * C_ell_delta_r + gamma8 * C_n_delta_r