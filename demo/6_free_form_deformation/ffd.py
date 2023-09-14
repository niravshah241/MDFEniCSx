import ufl
import dolfinx

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import scipy

# MPI communicator variables
world_comm = MPI.COMM_WORLD
rank = world_comm.Get_rank()
size = world_comm.Get_size()

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

reference_coordinates = mesh.geometry.x.copy()
num_control_points = [4, 24, 51, 101]

for cp in num_control_points:
    control_points_x = np.linspace(0., 1., cp)
    coefficients_x = \
        np.vstack([np.sin(control_points_x[:-1]*2*np.pi),
                   np.sin(control_points_x[1:]*2*np.pi)])
    bp_x = scipy.interpolate.BPoly(coefficients_x, control_points_x)
    mesh.geometry.x[:, 1] += 0.1 * bp_x(mesh.geometry.x[:, 0])

    '''
    control_points_y = np.linspace(0., 1., 27)
    coefficients_y = \
        np.vstack([np.sin(control_points_y[:-1]*2*np.pi),
                np.sin(control_points_y[1:]*2*np.pi)])
    bp_y = scipy.interpolate.BPoly(coefficients_y, control_points_y)
    mesh.geometry.x[:, 0] += 0.02 * bp_x(mesh.geometry.x[:, 1])
    '''

    with dolfinx.io.XDMFFile(mesh.comm,
                            f"ffd/deformed_mesh_cp_{cp}.xdmf", "w") as mesh_file:
        mesh_file.write_mesh(mesh)
    mesh.geometry.x[:] = reference_coordinates
