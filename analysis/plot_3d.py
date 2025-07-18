import numpy as np
import pyvista as pv
import vtk
from scipy.spatial.transform import Rotation as R


def plot_3d(data, time, filename):
    # Initializing inputs and plotter object
    quaternion = data
    t = time
    pl = pv.Plotter()

    # Displaying INERTIAL-FIXED frame unit vector arrows
    cent = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    dir = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mag = 2
    pl.add_arrows(cent=cent, direction=dir, mag=mag, color='black')
    pl.add_axes(line_width=2)

    # Adding cubesat object to the scene
    cubesat = pv.Cube(center=(0, 0, 0), x_length=3.0, y_length=2.0, z_length=1.0)
    actor = pl.add_mesh(cubesat, color='r',show_edges=True)

    # Displaying BODY-FIXED frame unit vector arrows
    body_arrows = pl.add_arrows(cent=cent, direction=dir, mag=mag, color='blue')

    # Text displays
    q_text = pl.add_text('Quaternion: ', position=(5, 730), font_size=10)
    t_text = pl.add_text('Time (s): ', position=(5, 690), font_size=10)

    # Creating movie file using input and writing initial frame
    pl.open_movie(filename)
    pl.write_frame()

    # Looping and writing each frame using quaternion input data
    for i, q in enumerate(quaternion):
        r = R.from_quat(q)
        rot_mat = r.as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = rot_mat

        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(mat.T.flatten(order='F'))
        body_arrows.SetUserMatrix(vtk_matrix)
        actor.SetUserMatrix(vtk_matrix)

        q_display = f'Quaternion (x, y, z, w):\n[{q[0]: .3f}, {q[1]: .3f}, {q[2]: .3f}, {q[3]: .3f}]'
        t_display = f'Time (s):\n{t[i]:.2f}'
        q_text.SetInput(q_display)
        t_text.SetInput(t_display)

        pl.write_frame()

    pl.close()
