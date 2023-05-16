import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

import dolfinx

from mdfenicsx.mesh_motion_classes import LinearElasticMeshMotion

np.set_printoptions(formatter={'float_kind': "{:.3f}".format})

dir = os.path.dirname(__file__)

# Read mesh
# Mesh geometric dimensions
gdim = 2
# gmsh model rank
gmsh_model_rank = 0
# MPI communicator for mesh
mesh_comm = MPI.COMM_WORLD
# Read mesh from .msh file
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(os.path.join(dir,"mesh_data/mesh.msh"), mesh_comm, gmsh_model_rank, gdim=gdim)

# Store reference mesh
with dolfinx.io.XDMFFile(mesh.comm, os.path.join(dir,"mesh_data/reference_mesh.xdmf"),"w") as reference_mesh_file:
    reference_mesh_file.write_mesh(mesh)

# Mesh deformation based on displacement boundary of the obstacle
def bc_obstacle(x):
    # Scaling the obstacle in x direction with fixed starting point
    
    # Version 2: (0.2 * x[0], 0.0 * x[1])
    return (0.1 * x[0], 0.0 * x[1])

def bc_wall(x):
    return ( 0.0 * x[0], 0.0 * x[1])

if mesh.comm.Get_rank() == 0:
    print("Mesh points before deformation")
    print(mesh.geometry.x[:, :])

# Young's modulus for each subdomain
E = [1100e6]

# Poisson's ratio for each subdomain
nu = [0.3]

# Mesh deformation with reset_reference=True
with LinearElasticMeshMotion(mesh, cell_tags,facet_tags, [1,2],
                        [bc_wall,bc_obstacle],E,nu, reset_reference=True,
                        is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm, os.path.join(dir,"mesh_data/linear_elastic_deformed_mesh.xdmf"),
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)

    if mesh.comm.Get_rank() == 0:
        print("Mesh points after deformation")
        print(mesh.geometry.x[:, :])
