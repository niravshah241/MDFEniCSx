import dolfinx
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, \
    assemble_vector, set_bc
import ufl

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from abc import ABC, abstractmethod


class MeshDeformation(ABC):
    def __init__(self, mesh, boundaries, bc_markers_list, bc_function_list,
                 reset_reference=False, is_deformation=True):
        '''
        Mesh, Boundraies: Original mesh and boundaries;
        bc_markers_list, bc_function_list:
            list of boundary marker and CORRESSPONDING boundary functions
        reset_reference: Bool, True to return original mesh to reference
                         state at the end of each context __exit__,
                         default=False --> mesh remains deformed at __exit__
        is_deformation: Bool, False to compute new coordinate,
                        default=True --> deformation is computed
        '''
        assert len(bc_markers_list) == len(bc_function_list)
        assert len(mesh.geometry.cmaps) == 1
        # Store coordinates of original mesh
        self._reference_coordinates = mesh.geometry.x.copy()
        # Current mesh and boundaries
        self._mesh = mesh
        self._boundaries = boundaries
        # List of boundary markers
        self._bc_markers = bc_markers_list
        # List of boundary functions
        self._bc_function_list = bc_function_list
        self._reset_reference = reset_reference
        self._is_deformation = is_deformation
        # FunctionSpace for mesh deformation function with degree = mesh order
        function_space_degree = self._mesh.geometry.cmaps[0].degree
        self._function_space = \
            dolfinx.fem.VectorFunctionSpace(self._mesh,
                                            ("CG", function_space_degree))
        # Test function on mesh deformation function space
        self._test_function = ufl.TestFunction(self._function_space)
        # Trial function on mesh deformation function space
        self._trial_function = ufl.TrialFunction(self._function_space)

    def __enter__(self):
        gdim = self._mesh.geometry.dim
        # Compute shape parametrization
        self.shape_parametrization = self.solve()
        # Apply shape parametrization
        if self._is_deformation is True:
            self._mesh.geometry.x[:, :self._mesh.geometry.dim] += \
                self.shape_parametrization.x.array.\
                reshape(self._reference_coordinates.shape[0],
                        gdim)
        else:
            self._mesh.geometry.x[:, :self._mesh.geometry.dim] = \
                self.shape_parametrization.x.array.\
                reshape(self._reference_coordinates.shape[0],
                        gdim)
        return self

    def assemble_bcs(self):
        # Assemble BCs as list
        gdim = self._mesh.geometry.dim
        boundaries = self._boundaries
        bc_list = list()
        for i in range(len(self._bc_markers)):
            dofs = boundaries.find(self._bc_markers[i])
            bc_dofs = dolfinx.fem.locate_dofs_topological(self._function_space,
                                                          gdim-1, dofs)
            bc_func = dolfinx.fem.Function(self._function_space)
            # Interpolate BCs on mesh deformation functionspace
            bc_func.interpolate(self._bc_function_list[i])
            bc_list.append(dolfinx.fem.dirichletbc(bc_func, bc_dofs))
        return bc_list

    @abstractmethod
    def bilinear_form(self):
        # Bilinear form
        pass

    @abstractmethod
    def linear_form(self):
        # Linear form
        pass

    def solve(self):
        # Solve system of equation
        uh = dolfinx.fem.Function(self._function_space)
        bcs = self.assemble_bcs()
        a_form = self.bilinear_form()
        l_form = self.linear_form()
        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        F = assemble_vector(l_form)
        apply_lifting(F, [a_form], [bcs])
        F.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, bcs)
        ksp = PETSc.KSP()
        ksp.create(self._mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        ksp.solve(F, uh.vector)
        uh.x.scatter_forward()
        return uh

    def __exit__(self, exception_type, exception_value, traceback):
        # Exit of context
        if self._reset_reference is True:
            self._mesh.geometry.x[:] = self._reference_coordinates
        else:
            pass


class HarmonicMeshMotion(MeshDeformation):
    # Harmonic mesh motion class with methods bilinear_form and linear_form
    def __init__(self, mesh, boundaries, bc_markers_list, bc_function_list,
                 reset_reference=False, is_deformation=True):
        super().__init__(mesh, boundaries, bc_markers_list, bc_function_list,
                         reset_reference=reset_reference,
                         is_deformation=is_deformation)

    def bilinear_form(self):
        # Assemble bilinear form for harmonic mesh motion
        u = self._trial_function
        v = self._test_function
        return dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)

    def linear_form(self):
        # Assemble linear form for of harmonic mesh motion
        v = self._test_function
        return dolfinx.fem.form(ufl.inner(dolfinx.fem.Constant
                                          (self._mesh, PETSc.ScalarType
                                           ((0.,) * self._mesh.geometry.dim)),
                                          v) * ufl.dx)


class LinearElasticMeshMotion(MeshDeformation):
    # Linear elastic mesh motion class
    def __init__(self, mesh, subdomains, boundaries, bc_markers_list,
                 bc_function_list, young_modulus_list, poisson_ratio_list,
                 reset_reference=False, is_deformation=True):
        super().__init__(mesh, boundaries, bc_markers_list, bc_function_list,
                         reset_reference=reset_reference,
                         is_deformation=is_deformation)
        self._young_modulus_list = young_modulus_list
        self._poisson_ratio_list = poisson_ratio_list
        self._subdomains = subdomains

    def epsilon(self, u_func):
        # Strain computation at given displacement field u_func
        return ufl.sym(ufl.grad(u_func))

    def sigma(self, u_func):
        # Stress computation at given displacement field u_func
        # DG function space for discontinuous material property
        material_function_space = dolfinx.fem.FunctionSpace(self._mesh,
                                                            ("DG", 0))
        # Function for Lam\'e parameter lambda = E / (2 * (1 + \nu))
        lambda_ = dolfinx.fem.Function(material_function_space)
        # Function for Lam\'e parameter mu = (E * \nu) / ((1-2*\nu) * (1+\nu))
        mu_ = dolfinx.fem.Function(material_function_space)
        # Assemble discontinuous material property functions
        for i in range(len(self._young_modulus_list)):
            lambda_.x.array[self._subdomains.find(i+1)] = \
                np.full_like(self._subdomains.find(i+1),
                             self._young_modulus_list[i] /
                             (2 * (1 + self._poisson_ratio_list[i])),
                             dtype=PETSc.ScalarType)
            mu_.x.array[self._subdomains.find(i+1)] = \
                np.full_like(self._subdomains.find(i+1),
                             (self._young_modulus_list[i] *
                              self._poisson_ratio_list[i]) /
                             ((1 - 2 * self._poisson_ratio_list[i]) *
                              (1 + self._poisson_ratio_list[i])),
                             dtype=PETSc.ScalarType)
        return lambda_ * ufl.nabla_div(u_func) * \
            ufl.Identity(len(u_func)) + \
            2 * mu_ * self.epsilon(u_func)  # Linear elastic stress field

    def bilinear_form(self):
        # Assemble bilinear form for linear elastic equation
        u = self._trial_function
        v = self._test_function
        return dolfinx.fem.form(ufl.inner
                                (self.sigma(u), self.epsilon(v)) * ufl.dx)

    def linear_form(self):
        # Assemble linear form of linear elastic equation with zero source term
        v = self._test_function
        return dolfinx.fem.form(ufl.dot(dolfinx.fem.Constant
                                        (self._mesh, PETSc.ScalarType((0.,) *
                                         self._mesh.geometry.dim)), v) *
                                ufl.dx)


if __name__ == "__main__":
    # Import mesh in dolfinx
    gdim = 2  # Mesh geometric dimensions
    gmsh_model_rank = 0  # gmsh model rank
    mesh_comm = MPI.COMM_WORLD  # MPI communicator for mesh
    mesh, cell_tags, facet_tags = \
        dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                        mesh_comm, gmsh_model_rank,
                                        gdim=gdim)  # Read .msh mesh file
    # Files for storing harmonic mesh motion deformation vectors
    harmonic_deformation_vector_file_xdmf = \
        dolfinx.io.XDMFFile(mesh.comm,
                            "harmonic/deformation_vector.xdmf",
                            "w")
    harmonic_deformation_vector_file_xdmf.write_mesh(mesh)
    # Files for storing linear elastic mesh motion deformation vectors
    linear_elastic_deformation_vector_file_xdmf = \
        dolfinx.io.XDMFFile(
            mesh.comm,
            "linear_elastic/deformation_vector.xdmf",
            "w")
    linear_elastic_deformation_vector_file_xdmf.write_mesh(mesh)

    # Parameters of the problem
    x0_max = 1.  # Maximum x value
    x1_max = 1.  # Maximum y value

    # Time steps
    t = 0  # Minimum time
    T = 2  # Maximum time
    num_steps = 41  # Number of time steps
    dt = (T - t)/(num_steps - 1)  # Time step

    # Class for defining boundary condition on boundary 1 and 5
    class BcFunc0():
        def __init__(self, t, x0_max, gdim):
            self.t = t
            self.x0_max = x0_max
            self.gdim = gdim

        def __call__(self, x):
            x0_max = self.x0_max
            values = np.zeros((self.gdim, x.shape[1]),
                              dtype=PETSc.ScalarType)
            if self.t < 1.:
                values[0] = 0.
                values[1] = t * np.sin(2*np.pi*x[0]/x0_max) / 20.
            else:
                values[0] = 0.
                values[1] = np.sin(2*np.pi*(x[0] - 0.1*(t-1.))/x0_max) / 20.
            return values

    # Class for defining boundary condition on boundary 9 and 12
    class BcFunc1():
        def __init__(self, t, x0_max, gdim):
            self.t = t
            self.x0_max = x0_max
            self.gdim = gdim

        def __call__(self, x):
            x0_max = self.x0_max
            values = np.zeros((self.gdim, x.shape[1]),
                              dtype=PETSc.ScalarType)
            if self.t < 1.:
                values[0] = 0.
                values[1] = t * np.sin(2 * np.pi * x[0] / x0_max) / 20.
            else:
                values[0] = (t-1.) * x[1] / 20.
                values[1] = np.sin(2 * np.pi *
                                   (x[0] - 0.1 * (t-1.))
                                   / x0_max) / 20.
            return values

    # Class for defining boundary condition on boundary 4, 6, 10, 11
    class BcFunc2():
        def __init__(self, t, gdim):
            self.t = t
            self.gdim = gdim

        def __call__(self, x):
            values = np.zeros((self.gdim, x.shape[1]),
                              dtype=PETSc.ScalarType)
            if self.t < 1.:
                values[0] = 0.
                values[1] = 0.
            else:
                values[0] = (t - 1.) * x[1] / 20.
                values[1] = 0.
            return values

    # Start of time steps
    for i in range(num_steps):
        print(f"Time step: {i + 1} of {num_steps}")
        # boundary condition on boundary 1 and 5
        u_bc_func = BcFunc0(t, x0_max, gdim)
        # boundary condition on boundary 9 and 12
        u_bc_func1 = BcFunc1(t, x0_max, gdim)
        # boundary condition on boundary 4, 6, 10, and 11
        u_bc_func2 = BcFunc2(t, gdim)

        # Harmonic mesh motion starts
        harmonic_deformation_mesh_file_xdmf = \
            dolfinx.io.XDMFFile(
                mesh.comm,
                f"harmonic/2D_deformed_mesh_{t}.xdmf",
                "w")
        # File for storing deformed mesh

        # Enter harmonic mesh motion context
        with HarmonicMeshMotion(mesh, facet_tags, [1, 5, 9, 12, 4, 6, 10, 11],
                                [u_bc_func, u_bc_func, u_bc_func1, u_bc_func1,
                                 u_bc_func2, u_bc_func2, u_bc_func2,
                                 u_bc_func2],
                                reset_reference=True,
                                is_deformation=True
                                ) as mesh_class:
            harmonic_deformation_mesh_file_xdmf.write_mesh(mesh_class._mesh)
        # Exit harmonic mesh motion context :
        # reset_reference = True --> mesh is now at original state

        # Store deformation on original mesh since reset_reference=True
        harmonic_deformation_vector_file_xdmf.\
            write_function(mesh_class.shape_parametrization, t)

        # Harmonic mesh motion ends

        # Linear elastic mesh motion starts
        linear_elastic_deformation_mesh_file_xdmf = \
            dolfinx.io.XDMFFile(mesh.comm,
                                f"linear_elastic/2D_deformed_mesh_{t}.xdmf",
                                "w")

        # Enter linear elastic mesh motion context
        with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                                     [1, 5, 9, 12, 4, 6, 10, 11],
                                     [u_bc_func, u_bc_func, u_bc_func1,
                                      u_bc_func1, u_bc_func2, u_bc_func2,
                                      u_bc_func2, u_bc_func2],
                                     [1100e6, 110e6, 1.1e6, 0.011e6],
                                     [0.3, 0.2, 0.1, 0.05],
                                     reset_reference=True,
                                     is_deformation=True) as mesh_class:
            linear_elastic_deformation_mesh_file_xdmf.\
                write_mesh(mesh_class._mesh)
        # Exit linear elastic mesh motion context :
        # reset_reference = True --> mesh is now at original state

        # Store deformation on original mesh since reset_reference=True
        linear_elastic_deformation_vector_file_xdmf.\
            write_function(mesh_class.shape_parametrization, t)
        # Linear elastic mesh motion ends

        # Update time step for next iteration
        t += dt
