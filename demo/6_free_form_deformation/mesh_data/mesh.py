import gmsh

from mpi4py import MPI

import dolfinx

gmsh.initialize('', False)

lc = 0.1
gdim = 2

gmsh.model.geo.addPoint(0., 0., 0., lc, 1)
gmsh.model.geo.addPoint(0.5, 0., 0., lc, 2)
gmsh.model.geo.addPoint(1., 0., 0., lc, 3)
gmsh.model.geo.addPoint(0., 0.5, 0., lc, 4)
gmsh.model.geo.addPoint(0.5, 0.5, 0., lc, 5)
gmsh.model.geo.addPoint(1., 0.5, 0., lc, 6)
gmsh.model.geo.addPoint(0., 1., 0., lc, 7)
gmsh.model.geo.addPoint(0.5, 1., 0., lc, 8)
gmsh.model.geo.addPoint(1., 1., 0., lc, 9)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 5, 2)
gmsh.model.geo.addLine(5, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.addLine(2, 3, 5)
gmsh.model.geo.addLine(3, 6, 6)
gmsh.model.geo.addLine(6, 5, 7)

gmsh.model.geo.addCurveLoop([5, 6, 7, -2], 2)
gmsh.model.geo.addPlaneSurface([2], 2)

gmsh.model.geo.addLine(5, 8, 8)
gmsh.model.geo.addLine(8, 7, 9)
gmsh.model.geo.addLine(7, 4, 10)

gmsh.model.geo.addCurveLoop([8, 9, 10, -3], 3)
gmsh.model.geo.addPlaneSurface([3], 3)

gmsh.model.geo.addLine(6, 9, 11)
gmsh.model.geo.addLine(9, 8, 12)

gmsh.model.geo.addCurveLoop([11, 12, -8, -7], 4)
gmsh.model.geo.addPlaneSurface([4], 4)

gmsh.model.geo.synchronize()

# 8=Frontal-Delaunay for Quads
gmsh.option.setNumber("Mesh.Algorithm", 8)
# 2=simple full-quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
# Apply recombination algorithm
gmsh.option.setNumber("Mesh.RecombineAll", 1)
# Mesh subdivision algorithm
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# Mesh generation
gmsh.model.mesh.generate(gdim)
# Mesh order
gmsh.model.mesh.setOrder(1)
# Mesh optimisation
gmsh.model.mesh.optimize("Netgen")

# Extract edges and surfaces to add physical groups
surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getBoundary(surfaces)

# Subdomains markings
for i in range(1, len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim, [surfaces[i-1][1]], surfaces[i-1][1])
# External bounndaries markings
for i in range(1, len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [edges[i-1][1]], edges[i-1][1])
# Internal boundary markings
gmsh.model.addPhysicalGroup(gdim-1, [2], 2)
gmsh.model.addPhysicalGroup(gdim-1, [3], 3)
gmsh.model.addPhysicalGroup(gdim-1, [7], 7)
gmsh.model.addPhysicalGroup(gdim-1, [8], 8)

gmsh.write("mesh.msh")

try:
    gmsh.fltk.run()
except:
    pass

gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf",
                         "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)
