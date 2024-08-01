import sys,os

import gmsh
import numpy as np

from mpi4py import MPI

import dolfinx

dir = os.path.dirname(__file__)

gmsh.initialize()

lc = 0.2
num_airfoil_refinement = 100   

L = 4
H = 1
gdim = 2

"""
    Defining the shape of the airfoil using Bézier Curves
    Geometrical parameters of the airfoil according to Xiaoqiang et al 
"""
c_1, c_2 = 0.5, 0.5                                     # Coefficients of camber-line-abscissa parameter equation
c_3, c_4 = 0.1, 0.05                                    # Coefficients of camber-line.ordinate parameter equation
coeff_thick = [0.2969,-0.126,-0.3516,0.2843,-0.1036]    # Coefficients of the thickness equation taken from the NACA 2412 Airfoil

# Bézier curves with control parameter k \in [0,1]
x_c = lambda k: 3 * c_1 * k * (1 - k)**2 + 3 * c_2 * (1-k) * k**2 + k**3    # Camber-line-abscissa
y_c = lambda k: 3 * c_3 * k * (1 - k)**2 + 3 * c_4 * (1-k) * k**2           # Camber-line-abscissa

# Thickness equation for a 6% thick airfoil
thickness = lambda x: 0.06 / 0.1 * (coeff_thick[0] * np.sqrt(x) + coeff_thick[1] * x + coeff_thick[2] * x**2 + coeff_thick[3] * x**3 + coeff_thick[4] * x**4)

# Position of the airfoil in the computational domain defined by the coordinates of the leading edge
leading_edge_x = np.min([L/8,0.5])
leading_edge_y = H/2

# Upper and lower surface of the airfoil
x_u = x_l = lambda k: leading_edge_x + x_c(k)
y_u = lambda k: leading_edge_y + y_c(k) + thickness(x_c(k))
y_l = lambda k: leading_edge_y + y_c(k) - thickness(x_c(k))

# Calculate maximal thickness of the airfoil
thickness_max = np.max([thickness(x) for x in np.linspace(0,1,num_airfoil_refinement)])

"""
    Meshing the airfoil using gmsh
"""

rectangle = gmsh.model.occ.addRectangle(0,0,0, L, H, tag=3)
# Define lower curve of the airfoil using the BSplines and given points
points_lower_curve=[]
for k in np.linspace(0,1,num_airfoil_refinement):
    points_lower_curve.append(gmsh.model.occ.addPoint(x_l(k), y_l(k), 0.0, lc))

# Define upper curve of the airfoil using the BSplines and given points
points_upper_curve=[points_lower_curve[0]]
for k in np.linspace(0,1,num_airfoil_refinement)[1:-1]:
    points_upper_curve.append(gmsh.model.occ.addPoint(x_u(k), y_u(k), 0.0, lc))
points_upper_curve.append(points_lower_curve[-1])

C1 = gmsh.model.occ.addBSpline(points_lower_curve, degree=3)
C2 = gmsh.model.occ.addBSpline(points_upper_curve, degree=3)

# Create airfoil and cut out of computational domain
W = gmsh.model.occ.addWire([C1,C2])
obstacle=gmsh.model.occ.addPlaneSurface([W])

# Remove points of the airfoil
for i in list(dict.fromkeys(points_lower_curve + points_upper_curve)):
    gmsh.model.occ.remove([(0, i)])

# Cut out airfoil from computational domain
fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
gmsh.model.occ.synchronize()

# Create a distance field to the airfoil
distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [C1, C2])
gmsh.model.mesh.field.setNumbers(distance_field,"PointsList", [points_lower_curve[0],points_lower_curve[-1]])
gmsh.model.mesh.field.setNumber(distance_field,"Sampling", num_airfoil_refinement*2)
# Create refined mesh using a threshold field
refinement= gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(refinement, "IField", distance_field)
# Set the refinement levels (LcMin for the mesh size in the refined region, LcMax for the mesh size far from the refined region)
gmsh.model.mesh.field.setNumber(refinement, "LcMin", lc/5)
gmsh.model.mesh.field.setNumber(refinement, "LcMax", lc*2)
# Set the threshold value where which refinement should be applied
gmsh.model.mesh.field.setNumber(refinement, "DistMin", thickness_max/2)
gmsh.model.mesh.field.setNumber(refinement, "DistMax", thickness_max)

# Set the field as background mesh
gmsh.model.mesh.field.setAsBackgroundMesh(refinement)

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



"""
    Defining boundary markers for the mesh
"""

fluid_marker, wall_marker, obstacle_marker = 1, 1, 2
wall, obstacle = [], []

surfaces = gmsh.model.getEntities(dim=gdim)
boundaries = gmsh.model.getBoundary(surfaces, oriented=False)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], fluid_marker)
gmsh.model.setPhysicalName(surfaces[0][0], fluid_marker, "Fluid")


for boundary in boundaries:
    center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    if np.allclose(center_of_mass, [0, H/2, 0]):
        wall.append(boundary[1])
    elif np.allclose(center_of_mass, [L, H/2, 0]):
        wall.append(boundary[1])
    elif np.allclose(center_of_mass, [L/2, H, 0]):
        wall.append(boundary[1])
    elif np.allclose(center_of_mass, [L/2, 0, 0]):
        wall.append(boundary[1])
    else:
        obstacle.append(boundary[1])

# Set physical markers for the boundaries
gmsh.model.addPhysicalGroup(gdim-1, wall, wall_marker)
gmsh.model.setPhysicalName(gdim-1, wall_marker, "wall")
gmsh.model.addPhysicalGroup(gdim-1, obstacle, obstacle_marker)
gmsh.model.setPhysicalName(gdim-1, obstacle_marker, "obstacle")

gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=0))

gmsh.write(os.path.join(dir,"mesh.msh"))

try:
    gmsh.fltk.run()
except Exception:
    pass

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(os.path.join(dir,"mesh.msh"), mesh_comm, gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf","w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)
