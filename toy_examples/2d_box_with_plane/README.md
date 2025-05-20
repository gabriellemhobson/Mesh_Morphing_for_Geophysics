# Toy Example: 2D Box with Embedded Plane, Morphed to Have Different Dip Angles

### Setup 

The domain is a square box with an embedded plane. 
The reference geometry has a vertical plane, while the target geometry has a plane that is 20 degrees from vertical. 
Points along the reference geometry's plane and the target geometry's plane are used to compute displacements; these points are distinct from vertices on the reference mesh. 
However, vertices on the boundary of the reference mesh are also used in defining the displacement and are assigned zero displacement. 
This means only the embedded plane will be morphed. 

The reference geometry is defined by `mesh_2d.geo`, a GMSH file. 
This file is meshed and saved using version 2.2 ASCII format in `mesh_2d.msh`. 

![Figure 1](schematic_simple.png)
Figure 1. Far left: the reference geometry and reference mesh. Center left: the displacement field required to morph the reference mesh into a new configuration. Center right: the displacement field evaluated at each vertex of the reference mesh. Far right: the morphed mesh representative of the target geometry.


### Procedure

The example is contained in `example_2d.py` which can be run on the command line using `python3 example_2d.py`. 
It takes `mesh_2d.msh` as input, performs the morphing steps, and saves the morphed mesh in `mesh_2d_deformed.msh`. 
It also generates the plot shown above. 
Each section of the code is commented. 