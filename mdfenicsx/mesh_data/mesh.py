import gmsh

from mpi4py import MPI

import dolfinx

gmsh.initialize('',False) 

lc = 0.1
gdim = 2

gmsh.model.geo.addPoint(0.,0.,0.,lc,1)
gmsh.model.geo.addPoint(0.5,0.,0.,lc,2)
gmsh.model.geo.addPoint(1.,0.,0.,lc,3)
gmsh.model.geo.addPoint(0.,0.5,0.,lc,4)
gmsh.model.geo.addPoint(0.5,0.5,0.,lc,5)
gmsh.model.geo.addPoint(1.,0.5,0.,lc,6)
gmsh.model.geo.addPoint(0.,1.,0.,lc,7)
gmsh.model.geo.addPoint(0.5,1.,0.,lc,8)
gmsh.model.geo.addPoint(1.,1.,0.,lc,9)

gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,5,2)
gmsh.model.geo.addLine(5,4,3)
gmsh.model.geo.addLine(4,1,4)

gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
gmsh.model.geo.addPlaneSurface([1],1)

gmsh.model.geo.addLine(2,3,5)
gmsh.model.geo.addLine(3,6,6)
gmsh.model.geo.addLine(6,5,7)

gmsh.model.geo.addCurveLoop([5,6,7,-2], 2)
gmsh.model.geo.addPlaneSurface([2],2)

gmsh.model.geo.addLine(5,8,8)
gmsh.model.geo.addLine(8,7,9)
gmsh.model.geo.addLine(7,4,10)

gmsh.model.geo.addCurveLoop([8,9,10,-3],3)
gmsh.model.geo.addPlaneSurface([3],3)

gmsh.model.geo.addLine(6,9,11)
gmsh.model.geo.addLine(9,8,12)

gmsh.model.geo.addCurveLoop([11,12,-8,-7],4)
gmsh.model.geo.addPlaneSurface([4],4)

gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 8) # 8=Frontal-Delaunay for Quads (See section 7.4,  https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # 2=simple full-quad (See section 7.4,  https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options)
gmsh.option.setNumber("Mesh.RecombineAll", 1) # Apply recombination algorithm to all surfaces, ignoring per-surface spec
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1) # Mesh subdivision algorithm (0: none, 1: all quadrangles, 2: all hexahedra, 3: barycentric)
gmsh.model.mesh.generate(gdim) # Mesh generation
gmsh.model.mesh.setOrder(2) # Mesh order
gmsh.model.mesh.optimize("Netgen") # Mesh optimisation or improving quality of mesh

# Extract edges and surfaces to add physical groups
surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getBoundary(surfaces)

print(surfaces,edges)

# Subdomains markings
for i in range(1,len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim,[surfaces[i-1][1]],surfaces[i-1][1])
# External bounndaries markings
for i in range(1,len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1,[edges[i-1][1]],edges[i-1][1])
#Internal boundary markings
gmsh.model.addPhysicalGroup(gdim-1,[2],2)
gmsh.model.addPhysicalGroup(gdim-1,[3],3)
gmsh.model.addPhysicalGroup(gdim-1,[7],7)
gmsh.model.addPhysicalGroup(gdim-1,[8],8)

# NOTE Remove gmsh markers as dolfinx.io.gmshio extract_geometry and extract_topology_and_markers expects gmsh to provide model with only physical markers and not point/edge markers.
#gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
#gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))

gmsh.write("mesh.msh")

gmsh.fltk.run()

gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags)
    mesh_file_xdmf.write_meshtags(facet_tags)
