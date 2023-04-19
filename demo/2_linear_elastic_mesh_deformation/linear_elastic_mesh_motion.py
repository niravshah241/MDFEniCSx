from mdfenicsx.mesh_motion_classes import LinearElasticMeshMotion

import dolfinx
import numpy as np

from mpi4py import MPI


np.set_printoptions(formatter={'float_kind': "{:.3f}".format})

# Read mesh
# Mesh geometric dimensions
gdim = 2
# gmsh model rank
gmsh_model_rank = 0
# MPI communicator for mesh
mesh_comm = MPI.COMM_WORLD
# Read mesh from .msh file
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

# Store reference mesh
with dolfinx.io.XDMFFile(mesh.comm, "linear_elastic/reference_mesh.xdmf",
                         "w") as reference_mesh_file:
    reference_mesh_file.write_mesh(mesh)

if mesh.comm.Get_rank() == 0:
    print("Mesh points before deformation")
    print(mesh.geometry.x[:7, :])

# Young's modulus for each subdomain
E = [1100e6, 110e6, 1.1e6, 0.011e6]

# Poisson's ratio for each subdomain
nu = [0.3, 0.2, 0.1, 0.05]

# Boundary conditions for linear elastic mesh deformation


# Bottom boundaries
def bc_bottom(x):
    return (0. * x[0], 0.2 * np.sin(x[0] * 2 * np.pi))


# Top boundaries
def bc_top(x):
    return (0. * x[0], 0.1 * np.sin(x[0] * 2 * np.pi))


# Side boundaries
def bc_side(x):
    return (0. * x[0], 0. * x[1])


# Mesh deformation with is_deformation=True
with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                             [1, 5, 9, 12, 4, 6, 10, 11],
                             [bc_bottom, bc_bottom, bc_top,
                              bc_top, bc_side, bc_side,
                              bc_side, bc_side], E, nu,
                             reset_reference=True,
                             is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm,
                             "linear_elastic/deformed_mesh.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
    if mesh.comm.Get_rank() == 0:
        print("Mesh points after first deformation")
        print(mesh.geometry.x[:7, :])

if mesh.comm.Get_rank() == 0:
    print("Mesh points after exit from context with reset_reference=True")
    print(mesh.geometry.x[:7, :])

# Mesh deformation with is_deformation=False
with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                             [1, 5, 9, 12, 4, 6, 10, 11],
                             [bc_bottom, bc_bottom, bc_top,
                              bc_top, bc_side, bc_side,
                              bc_side, bc_side], E, nu,
                             reset_reference=False,
                             is_deformation=True):
    if mesh.comm.Get_rank() == 0:
        print("Mesh points after second deformation")
        print(mesh.geometry.x[:7, :])

if mesh.comm.Get_rank() == 0:
    print("Mesh points after exit from context with reset_reference=False")
    print(mesh.geometry.x[:7, :])

# ### Mesh deformation based on new coordinates
# boundary condition ###

# Read mesh
# Mesh geometric dimensions
gdim = 2
# gmsh model rank
gmsh_model_rank = 0
# MPI communicator for mesh
mesh_comm = MPI.COMM_WORLD
# Read mesh from .msh file
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

# Boundary conditions for linear elastic mesh deformation


# Bottom boundaries
def bc_bottom(x):
    return (x[0], x[1] + 0.2 * np.sin(x[0] * 2 * np.pi))


# Top boundaries
def bc_top(x):
    return (x[0], x[1] + 0.1 * np.sin(x[0] * 2 * np.pi))


# Side boundaries
def bc_side(x):
    return (x[0], x[1])


# Mesh deformation with is_deformation=False
with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                             [1, 5, 9, 12, 4, 6, 10, 11],
                             [bc_bottom, bc_bottom, bc_top,
                              bc_top, bc_side, bc_side,
                              bc_side, bc_side], E, nu,
                             reset_reference=True,
                             is_deformation=False):
    if mesh.comm.Get_rank() == 0:
        print("Mesh points with is_deformation=False")
        print(mesh.geometry.x[:7, :])
