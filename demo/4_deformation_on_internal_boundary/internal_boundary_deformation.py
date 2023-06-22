from mdfenicsx.mesh_motion_classes import \
    HarmonicMeshMotion, LinearElasticMeshMotion

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
with dolfinx.io.XDMFFile(mesh_comm, "internal_boundary/reference_mesh.xdmf",
                         "w") as reference_mesh_file:
    reference_mesh_file.write_mesh(mesh)

# Boundary conditions for mesh deformation

# Internal boundary


def bc_internal(x):
    return (0. * x[0], 0.1 * np.sin(x[0] * 2. * np.pi))


# External boundary


def bc_external(x):
    return (0. * x[0], 0. * x[1])


# Mesh deformation (Harmonic mesh motion)
with HarmonicMeshMotion(mesh, facet_tags, [1, 2, 3, 4, 5, 6, 7],
                        [bc_external, bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_internal],
                        reset_reference=True, is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm,
                             "internal_boundary/deformed_harmonic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)


# Mesh deformation (Linear elastic mesh motion)
E = [1100e6, 0.011e6]  # Young's modulus for each of the subdomains
nu = [0.3, 0.1]  # Poisson's ratio for each of the subdomains
with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                             [1, 2, 3, 4, 5, 6, 7],
                             [bc_external, bc_external,
                              bc_external, bc_external,
                              bc_external, bc_external,
                              bc_internal], E, nu,
                             reset_reference=True,
                             is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm,
                             "internal_boundary/deformed_linear_elastic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
