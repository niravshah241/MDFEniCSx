## Harmonic Mesh Deformation ##

### 1. Problem statement

Consider a square domain with vertices (0, 0) -- (0, 1) -- (1, 1) -- (1, 0) as shown below. We also call this domain as reference domain and corresponding mesh as reference mesh.

* **Reference domain** and **Reference subdomains**:

![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_motion/mesh_data/domain.png)
![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_motion/mesh_data/subdomains.png)

* **Reference mesh** and **Reference boundaries**: 

1, 5: Bottom boundaries ($\Gamma_1, \Gamma_5$)

9, 12: Top boundaries ($\Gamma_9, \Gamma_{12}$)

4, 10: Left boundaries ($\Gamma_4, \Gamma_{10}$)

6, 11: Right boundaries ($\Gamma_6, \Gamma_{11}$)

![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_motion/mesh_data/boundaries.png)

We define below mesh **displacements**:

$$\text{On } \Gamma_1 \cup \Gamma_5: \ (0., 0.2 sin(2 \pi x))$$

$$\text{On } \Gamma_9 \cup \Gamma_{12}: \ (0., 0.1 sin(2 \pi x))$$

$$\text{On } \Gamma_4 \cup \Gamma_{10} \cup \Gamma_6 \cup \Gamma_{11}: \ (0., 0.)$$

Instead of **displacements**, we could also specify the **new coordinates** after deformation on the boundary:

$$\text{On } \Gamma_1 \cup \Gamma_5: \ (x, y + 0.2 sin(2 \pi x))$$

$$\text{On } \Gamma_9 \cup \Gamma_{12}: \ (x, y + 0.1 sin(2 \pi x))$$

$$\text{On } \Gamma_4 \cup \Gamma_{10} \cup \Gamma_6 \cup \Gamma_{11}: \ (x, y)$$

### 2. Implementation

```HarmonicMeshMotion``` takes the following arguments:
 * Reference mesh as dolfinx mesh
 * Facet tags
 * Boundary markers as List of integers
 * Deformation applied on each boundary as List of functions (Same order as boundary markers)
 * Keyword arguments ```reset_reference``` and ```is_deformation```

```
with HarmonicMeshMotion(mesh, facet_tags, [1, 5, 9, 12, 4, 6, 10, 11],
                        [bc_bottom, bc_bottom, bc_top, bc_top, bc_side,
                         bc_side, bc_side, bc_side], reset_reference=True,
                        is_deformation=True):
```

The mesh generation file is given in ```mesh_data/mesh.py``` which stores the mesh in same directory. The harmonic mesh motion implementation is given in ```harmonic_mesh_motion.py```. We first set the **displacements** boundary contions with ```is_deformation=True```. We print first few mesh points to observe an important difference. When the code is run with ```mpiexec -n 1 python3 harmonic_mesh_motion.py```, following output is produced.

```
Mesh points before deformation
[[0.     0.     0.    ]
 [0.125  0.     0.    ]
 [0.     0.125  0.    ]
 [0.125  0.125  0.    ]
 [0.0625 0.     0.    ]
 [0.     0.0625 0.    ]
 [0.125  0.0625 0.    ]]
Mesh points after first deformation
[[0.         0.         0.        ]
 [0.125      0.14142136 0.        ]
 [0.         0.125      0.        ]
 [0.125      0.18971448 0.        ]
 [0.0625     0.07653669 0.        ]
 [0.         0.0625     0.        ]
 [0.125      0.15808655 0.        ]]
Mesh points after exit from context with reset_reference=True
[[0.     0.     0.    ]
 [0.125  0.     0.    ]
 [0.     0.125  0.    ]
 [0.125  0.125  0.    ]
 [0.0625 0.     0.    ]
 [0.     0.0625 0.    ]
 [0.125  0.0625 0.    ]]
Mesh points after second deformation
[[0.         0.         0.        ]
 [0.125      0.14142136 0.        ]
 [0.         0.125      0.        ]
 [0.125      0.18971448 0.        ]
 [0.0625     0.07653669 0.        ]
 [0.         0.0625     0.        ]
 [0.125      0.15808655 0.        ]]
Mesh points after exit from context with reset_reference=False
[[0.         0.         0.        ]
 [0.125      0.14142136 0.        ]
 [0.         0.125      0.        ]
 [0.125      0.18971448 0.        ]
 [0.0625     0.07653669 0.        ]
 [0.         0.0625     0.        ]
 [0.125      0.15808655 0.        ]]

```

As can be observed, after first mesh deformation, the mesh returns to the reference mesh configuration upon exit from the mesh deformation context. While, after the second mesh deformation, the mesh remains deformed upon exit from the mesh deformation context. This difference can be explained by the keyword argument ```reset_reference```.

When ```reset_reference=True```, the mesh returns to the reference mesh configuration upon exit from the mesh deformation context. Instead, when ```reset_reference=False```, the mesh remains deformed and does not return to the reference mesh configuration upon exit from the mesh deformation context.

* **Deformed mesh**: 
![alt text](https://github.com/niravshah241/MDFEniCSx/blob/main/demo/1_harmonic_mesh_motion/deformed_mesh.png)

It should also be noted that the keyword argument ```is_deformation``` is set to ```True``` in both the cases. This is because, the **displacement** was specified on the boundary instead of **new coordinates** after deformation. If we apply the boundary condition corresponding to **new coordinates** after deformation we obtain the same result by specifying ```is_deformation=False```.

```
Mesh points with is_deformation=False
[[-2.94902991e-17 -2.42861287e-17  0.00000000e+00]
 [ 1.25000000e-01  1.41421356e-01  0.00000000e+00]
 [-2.94902991e-17  1.25000000e-01  0.00000000e+00]
 [ 1.25000000e-01  1.89714481e-01  0.00000000e+00]
 [ 6.25000000e-02  7.65366865e-02  0.00000000e+00]
 [ 5.20417043e-18  6.25000000e-02  0.00000000e+00]
 [ 1.25000000e-01  1.58086549e-01  0.00000000e+00]]
```
