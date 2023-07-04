from mdfenicsx.mesh_motion_classes \
    import HarmonicMeshMotion, LinearElasticMeshMotion

from mpi4py import MPI
import numpy as np

import dolfinx


np.set_printoptions(formatter={'float_kind': "{:.3f}".format})


# Read from mesh
gdim = 2
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)


# Store reference mesh
with dolfinx.io.XDMFFile(mesh_comm, "part_boundary/reference_mesh.xdmf",
                         "w") as reference_mesh_file:
    reference_mesh_file.write_mesh(mesh)


# Boundary conditions for mesh deformation


# Top boundary
def bc_top(x):
    indices = np.where(x[0] >= 0.5)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices] = 0.1 * np.sin(2. * (x[0][indices] - 0.5) * np.pi)
    return y


# Middle boundary
def bc_middle(x):
    indices = np.where(x[0] <= 0.5)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices] = - 0.1 * np.sin(2. * x[0][indices] * np.pi)
    return y


# Fixed boundaries

# Other boundaries (Zero displacement/fixed)
def bc_fixed(x):
    return (0. * x[0], 0. * x[1])


# Mesh deformation (Harmonic Mesh Motion)
with HarmonicMeshMotion(mesh, boundaries, [1, 2, 3, 4, 5, 6, 7],
                        [bc_fixed, bc_fixed, bc_fixed, bc_top,
                         bc_fixed, bc_fixed, bc_middle],
                        reset_reference=True, is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, "part_boundary/deformed_harmonic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)


# Mesh deformation (Linear Elastic Mesh Motion)
E = [1100e6, 0.011e6]  # Young's modulus for each of the subdomains
nu = [0.3, 0.1]  # Poisson's ratio for each of the subdomains
with LinearElasticMeshMotion(mesh, subdomains, boundaries, [1, 2, 3, 4, 5, 6, 7],
                             [bc_fixed, bc_fixed, bc_fixed, bc_top,
                              bc_fixed, bc_fixed, bc_middle], E, nu,
                        reset_reference=True, is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, "part_boundary/deformed_linear_elastic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
