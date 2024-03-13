
# keyboard input controller

import keyboard
from message_types.msg_delta import MsgDelta

def process_control_inputs(delta: MsgDelta) -> None:
    if keyboard.is_pressed('up'):
        delta.elevator += 0.001
    elif keyboard.is_pressed('down'):
        delta.elevator -= 0.001
        
    if keyboard.is_pressed('right'):
        delta.aileron += 0.001
    elif keyboard.is_pressed('left'):
        delta.aileron -= 0.001
        
    if keyboard.is_pressed('a'):
        delta.rudder += 0.0001
    elif keyboard.is_pressed('d'):
        delta.rudder -= 0.0001
        
    if keyboard.is_pressed('w'):
        delta.throttle += 0.001
    elif keyboard.is_pressed('s'):
        delta.throttle -= 0.001
    
    if keyboard.is_pressed("z"):
        delta.elevator = 0          # -0.1248
        delta.aileron = 0           # 0.001836
        delta.rudder = 0            # -0.0003026
        delta.throttle = 0.5        # 0.6768
        
    if keyboard.is_pressed("esc"):
        exit()