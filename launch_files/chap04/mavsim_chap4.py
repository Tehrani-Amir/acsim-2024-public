"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac

#from python_tools.quit_listener import QuitListener
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from message_types.msg_delta import MsgDelta
from mystuff.process_control_inputs import process_control_inputs
from mystuff.trim import compute_trim

import keyboard

#quitter = QuitListener()

VIDEO = False
PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False

if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap4_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

#initialize the visualization
if ANIMATION or PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    mav_view = MavViewer(app=app)  # initialize the mav viewer
if PLOTS:
    # initialize view of data plots
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()

# create initialization parameters
Va0 = 30  
alpha0 = 0.
beta0 = 0.

# call the function
mav.initialize_velocity(Va0, alpha0, beta0)

# first guess of trim
delta.elevator = -0.1248
delta.aileron = 0.0
delta.rudder = 0.0
delta.throttle = 0.6768

# initialize the simulation time
sim_time = SIM.start_time
plot_time = sim_time
end_time = 100

# find trim output
alpha, elevator, throttle = compute_trim(mav, delta)
# re-initializing based on the trim input
mav.initialize_velocity(Va0, alpha, beta0)
delta.elevator = elevator 
delta.elevator = throttle

# This means that to maintain the defined airspeed the input should be:
print(alpha, delta.elevator, delta.throttle)
# exit()

alpha, elevator, throttle = compute_trim(mav, delta)
mav.initialize_velocity(Va0, alpha, beta0)
delta.elevator = elevator
delta.throttle = throttle

# main simulation loop
print("Press 'Esc' to exit...")

while sim_time < end_time:
    # ------- set control surfaces -------------
    
    # Update control surfaces based on keyboard input
    process_control_inputs(delta)
    
    if keyboard.is_pressed('esc'):
        break
    
    # ------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            None,  # commanded states
                            delta)  # inputs to aircraft
    if ANIMATION or PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation


if SAVE_PLOT_IMAGE:
    data_view.save_plot_image("ch4_plot")

if VIDEO is True:
    video.close()