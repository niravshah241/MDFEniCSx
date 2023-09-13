## Mesh Deformation basics ##

### 1. Problem statement

Consider a square domain with vertices (0, 0) -- (0, 1) -- (1, 1) -- (1, 0) as shown below. We also call this domain $\Omega$ as reference domain, corresponding mesh as reference mesh and deform it to $\tilde{\Omega}$. The purpose of this excercise is to demonstrate Harmonic mesh extension (Shamanskiy, A. et. al.)[https://doi.org/10.1007/s00466-020-01950-x]. We solve Laplace's equation on the reference domain to calculate pointwise mesh deformation based on the specified mesh deformation on the boundary.

* **Reference domain** and **Reference subdomains**:

![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_deformation/mesh_data/domain.png)
![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_deformation/mesh_data/subdomains.png)

* **Reference mesh** and **Reference boundaries**: 

1, 5: Bottom boundaries ($\Gamma_1, \Gamma_5$)

9, 12: Top boundaries ($\Gamma_9, \Gamma_{12}$)

4, 10: Left boundaries ($\Gamma_4, \Gamma_{10}$)

6, 11: Right boundaries ($\Gamma_6, \Gamma_{11}$)

![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_deformation/mesh_data/boundaries.png)

We define below mesh **displacements**:

$$\text{On } \Gamma_1 \cup \Gamma_5: \ (0., \ 0.2 sin(2 \pi x))$$

$$\text{On } \Gamma_9 \cup \Gamma_{12}: \ (0., \ 0.1 sin(2 \pi x))$$

$$\text{On } \Gamma_4 \cup \Gamma_{10} \cup \Gamma_6 \cup \Gamma_{11}: \ (0., \ 0.)$$

### 2. Implementation

First we read the reference mesh.
```
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                     mesh_comm, gmsh_model_rank,
                                     gdim=gdim)
```

We also store mesh coordinates of the reference mesh.
```
reference_coordinates = mesh.geometry.x.copy()
```

We create relevant function spaces with same degree as mesh degree.
```
mesh_degree = mesh.geometry.cmaps[0].degree
V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", mesh_degree))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
```

Next, we define boundary conditions, arrange boundary markers and boundary deformation expression in relevant orders. It should be noted that, it is important to define ZERO DIRICHLET boundary conditions, else unspecified boundaries are considered as zero NEUMANN boundary conditions
```
# Define mesh deformation on boundaries (boundary conditions)
bc_list = []

# Bottom boundaries (Boundary markers 1 and 5)
def bc_bottom(x):
    return (0. * x[0], 0.2 * np.sin(x[0] * 2 * np.pi))


# Top boundaries (Boundary markers 9 and 12)
def bc_top(x):
    return (0. * x[0], 0.1 * np.sin(x[0] * 2 * np.pi))


# Side boundaries (Boundary markers 4, 6, 10 and 11)
def bc_side(x):
    return (0. * x[0], 0. * x[1])


# We gather boundary markers and boundary displacements
# in respective orders as list and specify boundary conditions
bc_markers = [1, 5, 9, 12, 4, 6, 10, 11]
bc_function_list = \
    [bc_bottom, bc_bottom, bc_top, bc_top, bc_side, bc_side, bc_side, bc_side]
for i in range(len(bc_markers)):
    dofs = boundaries.find(bc_markers[i])
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, gdim-1, dofs)
    bc_func = dolfinx.fem.Function(V)
    bc_func.interpolate(bc_function_list[i])
    bc_list.append(dolfinx.fem.dirichletbc(bc_func, bc_dofs))
# bc_list now contains boundary conditions
```

We then solve Laplace's equation on the REFERENCE mesh to compute pointwise displacement field.
```
a_form = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
l_form = \
    dolfinx.fem.form(ufl.inner(dolfinx.fem.Constant
                               (mesh, PETSc.ScalarType((0.,) * mesh.geometry.dim)),
                               v) * ufl.dx)

uh = dolfinx.fem.Function(V)
A = assemble_matrix(a_form, bcs=bc_list)
A.assemble()
F = assemble_vector(l_form)
dolfinx.fem.petsc.apply_lifting(F, [a_form], [bc_list])
F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(F, bc_list)
ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
ksp.solve(F, uh.vector)
uh.x.scatter_forward()
```

Displacement field ```uh``` now corresponds to the displacement of each node. We add displacement field to the reference mesh cordinates. The deformed domain \tilde{\Omega} is given by:
$$\tilde{x} = x + uh \ , \ \tilde{x} \in \tilde{\Omega} \ , \ x \in \Omega \ .$$
```
mesh.geometry.x[:, :mesh.geometry.dim] += \
    uh.x.array.reshape(reference_coordinates.shape[0], gdim)


with dolfinx.io.XDMFFile(mesh.comm,
                        f"deformed_mesh_data/deformed_mesh.xdmf",
                        "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains, mesh.geometry)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)
```

To reset to the REFERENCE mesh configuration. We overwrite mesh cordinates with ```reference_coordinates```.
```
mesh.geometry.x[:] = reference_coordinates
```

Instead of solving Laplace's equation, one can also solve linear elasticity equation.
