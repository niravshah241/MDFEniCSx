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

control_points_x = np.linspace(1., 0., 61)
coefficients_x = list()
berstein_order = 3

for k in range(berstein_order+1):
    list_row = list()
    for i in range(len(control_points_x)-1):
        list_row.append(np.sin(control_points_x[i] * 2 * np.pi))
    coefficients_x.append(list_row)

control_points_y = np.linspace(0., 1., 3)
coefficients_y = [[1, 1], [1, 1]]

bp = scipy.interpolate.BPoly(coefficients_x, control_points_x)
print(bp(0.22))

reference_coordinates = mesh.geometry.x.copy()
mesh.geometry.x[:, 1] += 0.1 * bp(mesh.geometry.x[:, 0])

with dolfinx.io.XDMFFile(mesh.comm,
                         f"ffd/2D_deformed_mesh.xdmf", "w") as mesh_file:
    mesh_file.write_mesh(mesh)
