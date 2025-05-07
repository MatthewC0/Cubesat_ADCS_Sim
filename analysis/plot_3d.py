import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def plot_3d(data, time, filename):
    # Initializing inputs and plotter object
    quaternion = data
    t = time
    pl = pv.Plotter()

    # Displaying body-fixed frame unit vector arrows
    cent = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    dir = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mag = 2
    pl.add_arrows(cent=cent, direction=dir, mag=mag)
    pl.add_axes(line_width=2)

    # Adding cubesat object to the scene
    cubesat = pv.Cube(center=(0, 0, 0), x_length=3.0, y_length=2.0, z_length=1.0)
    actor = pl.add_mesh(cubesat, color='r',show_edges=True)

    # Creating movie file using input and writing initial frame
    pl.open_movie(filename)
    pl.write_frame()

    # Looping and writing each frame using quaternion input data
    for q in quaternion:
        r = R.from_quat(q)
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = r.as_matrix()

        actor.user_matrix = rot_mat
        pl.write_frame()

    pl.close()
