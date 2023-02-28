## Linear Elastic Mesh Deformation ##

### 1. Problem statement

Consider a square domain with vertices (0, 0) -- (0, 1) -- (1, 1) -- (1, 0) as shown below. We also call this domain as reference domain and corresponding mesh as reference mesh.

* **Reference domain** $\Omega$ and **Reference subdomains** $\lbrace \Omega_i \rbrace_{i=1}^{4}$:

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

Like ```HarmonicMeshMotion```, we can also define below **new coordinates** boundary condition:

$$\text{On } \Gamma_1 \cup \Gamma_5: \ (x, \ y + 0.2 sin(2 \pi x))$$

$$\text{On } \Gamma_9 \cup \Gamma_{12}: \ (x, \ y + 0.1 sin(2 \pi x))$$

$$\text{On } \Gamma_4 \cup \Gamma_{10} \cup \Gamma_6 \cup \Gamma_{11}: \ (x, \ y)$$

We define below mesh Young's modulus $(E)$ and Poisson's ratio $(\nu)$:
$$\text{In } \Omega_1: E = 1100e6 \ , \ \nu = 0.3$$

$$\text{In } \Omega_2: E = 110e6 \ , \ \nu = 0.2$$

$$\text{In } \Omega_3: E = 1.1e6 \ , \ \nu = 0.1$$

$$\text{In } \Omega_4: E = 0.011e6 \ , \ \nu = 0.05$$

### 2. Implementation

```LinearElasticMeshMotion``` takes the following arguments:
 * Reference mesh as dolfinx mesh
 * Facet tags
 * Cell tags
 * Boundary markers as List of integers
 * Deformation applied on each boundary as List of functions (Same order as boundary markers)
 * Young's modulus as list (In increasing order of subdomain number)
 * Poisson's ratio as list (In increasing order of subdomain number)
 * Keyword arguments ```reset_reference``` and ```is_deformation```

```
E = [1100e6, 110e6, 1.1e6, 0.011e6]
nu = [0.3, 0.2, 0.1, 0.05]

with LinearElasticMeshMotion(mesh, cell_tags, facet_tags,
                             [1, 5, 9, 12, 4, 6, 10, 11],
                             [bc_bottom, bc_bottom, bc_top,
                              bc_top, bc_side, bc_side,
                              bc_side, bc_side], E, nu,
                             reset_reference=True,
                             is_deformation=True):
```

The mesh generation file is given in ```mesh_data/mesh.py``` which stores the mesh in same directory. The linear elastic mesh motion implementation is given in ```linear_elastic_mesh_motion.py```. We print first few mesh points and set keyword argument ```is_deformation=True``` with **displacement** boundary conditions. When the code is run with ```mpiexec -n 1 python3 linear_elastic_mesh_motion.py```, following output is produced.

```
Mesh points before deformation
[[0.000 0.000 0.000]
 [0.125 0.000 0.000]
 [0.000 0.125 0.000]
 [0.125 0.125 0.000]
 [0.062 0.000 0.000]
 [0.000 0.063 0.000]
 [0.125 0.063 0.000]]
Mesh points after first deformation
[[0.000 0.000 0.000]
 [0.125 0.141 0.000]
 [0.000 0.125 0.000]
 [0.125 0.220 0.000]
 [0.062 0.077 0.000]
 [0.000 0.063 0.000]
 [0.122 0.178 0.000]]
Mesh points after exit from context with reset_reference=True
[[0.000 0.000 0.000]
 [0.125 0.000 0.000]
 [0.000 0.125 0.000]
 [0.125 0.125 0.000]
 [0.062 0.000 0.000]
 [0.000 0.063 0.000]
 [0.125 0.063 0.000]]
Mesh points after second deformation
[[0.000 0.000 0.000]
 [0.125 0.141 0.000]
 [0.000 0.125 0.000]
 [0.125 0.220 0.000]
 [0.062 0.077 0.000]
 [0.000 0.063 0.000]
 [0.122 0.178 0.000]]
Mesh points after exit from context with reset_reference=False
[[0.000 0.000 0.000]
 [0.125 0.141 0.000]
 [0.000 0.125 0.000]
 [0.125 0.220 0.000]
 [0.062 0.077 0.000]
 [0.000 0.063 0.000]
 [0.122 0.178 0.000]]
```

As can be observed, ```reset_reference``` restores the mesh upon exit from mesh deformation context. This behavior is similar to the ```HarmonicMeshMotion```. We observe the mesh deformation below. As can be noticed, the mesh is deformed in non-uniform manner, due to different material properties.


* **Deformed mesh**: 
![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/2_linear_elastic_mesh_deformation/deformed_mesh.png)


However, when **new coordinates** boundary conditions are used with ```is_deformation=False``` instead of **displacements** boundary conditions with ```is_deformation=True```, the behavior is different from the ```HarmonicMeshMotion```.

```
Mesh points with is_deformation=False
[[-0.000 -0.000 0.000]
 [0.125 0.141 0.000]
 [-0.000 0.125 0.000]
 [0.054 0.145 0.000]
 [0.062 0.077 0.000]
 [0.000 0.063 0.000]
 [0.080 0.137 0.000]]
```

This is due to the fact that different properties ($E, \nu$) are assigned in each of the subdomains. As an excercise to the users, it can be verified that when material properties are same, **new coordinates** boundary conditions with ```is_deformation=False``` produces same behavior as **displacements** boundary conditions with ```is_deformation=True```.
