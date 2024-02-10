"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh

class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the Spacecraft.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.sc_points, self.sc_index, self.sc_meshColors = self.get_sc_points()
        self.sc_body = self.add_object(
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)
        window.addItem(self.sc_body)  # add spacecraft to plot     

    def update(self, state):
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.sc_body = self.update_object(
            self.sc_body,
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_sc_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the spacecraft following information in Appendix C.3
        """
        
        # Define the MAV parameters and Geometry
        Scale = 5
        
        wing_l= Scale* 0.3
        wing_w= Scale* 1.5
        
        fuse_l1= Scale* 0.3
        fuse_l2= Scale* 0.15
        fuse_l3= Scale* 1.2
        fuse_w= Scale* 0.2
        fuse_h= Scale* 0.2
        
        tailwing_l= Scale* 0.1
        tailwing_w= 0.3*wing_w 
        tail_h= Scale* 0.4
        
        # points are in XYZ coordinates
        # define the points on the spacecraft according to Appendix C.3
        points = self.unit_length * np.array([

            [fuse_l1, 0, 0],  # point 1 [0]
            [fuse_l2, 0.5*fuse_w, -0.5*fuse_h],  # point 2 [1]
            [fuse_l2, -0.5*fuse_w, -0.5*fuse_h],  # point 3 [2]
            [fuse_l2, -0.5*fuse_w, 0.5*fuse_h],  # point 4 [3]
            [fuse_l2, 0.5*fuse_w, 0.5*fuse_h],  # point 5 [4]
            [-fuse_l3,0,0], #point 6 [5]
            [0, 0.5*wing_w, 0],  # point 7 [6]
            [-wing_l, 0.5*wing_w, 0],  # point 8 [7]
            [-wing_l, -0.5*wing_w, 0],  # point 9 [8]
            [0, -0.5*wing_w, 0],  # point 10 [9]
            [-fuse_l3+tailwing_l, 0.5*tailwing_w, 0],  # point 11 [10]
            [-fuse_l3, 0.5*tailwing_w, 0] , # point 12
            [-fuse_l3, -0.5*tailwing_w, 0] , # point 13
            [-fuse_l3+tailwing_l, -0.5*tailwing_w, 0],  # point 14
            [-fuse_l3+tailwing_l, 0, 0] , # point 15
            [-fuse_l3, 0, -tail_h] , # point 16
            
            ]).T
        
        # point index that defines the mesh
        index = np.array([
            [0, 1, 2],  # Nose 1
            [0, 2, 3],  # Nose 2
            [0, 1, 4],  # Nose 3
            [0, 3, 4],  # Noes 4
            
            [1, 2, 5],  # Body 1
            [2, 3, 5],  # Body 2
            [3, 4, 5],  # Body 3
            [1, 4, 5],  # Body 4
            
            [6, 8, 9],  # Wing 1
            [6, 7, 8],  # Wing 2
            
            [10, 12, 13],  # Tail Wing 1
            [10, 11, 12],  # Tail Wing 2 
            
            [5,14,15] #Ruddar
            ])
        
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        
        meshColors[0] = green  # Nose 1
        meshColors[1] = green  # Nose 2
        meshColors[2] = green  # Nose 3
        meshColors[3] = green  # Nose 4
        
        meshColors[4] = blue  # Body 1
        meshColors[5] = blue  # Body 2
        meshColors[6] = blue  # Body 3
        meshColors[7] = blue  # Body 4
        
        meshColors[8] = red  # Wing 1
        meshColors[9] = red  # Wing 2
        
        meshColors[10] = red  # Tail Wing 1
        meshColors[11] = red  # Tail Wing 2
        
        meshColors[12] = blue  # Ruddar
        
        return points, index, meshColors

