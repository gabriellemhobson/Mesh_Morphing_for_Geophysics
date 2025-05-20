import numpy as np
import matplotlib.pyplot as plt
import meshio
import pyvista as pv
from scipy.interpolate import RBFInterpolator
from copy import deepcopy

pv.global_theme.transparent_background = True

def get_boundary_pts(mesh,tags):
    line_cells = mesh.cells_dict["line"]
    bp = []
    for tag in tags:
        cells = line_cells
        pts = mesh.points[np.unique(cells.ravel())]
        for k in range(pts.shape[0]):
            if np.abs(pts[k,0]) < 1e-14:
                continue
            bp.append(pts[k,:])
    boundary_pts = np.unique(np.array(bp),axis=0)
    return boundary_pts

def compute_displacement(f1,f2,bnd):
    df = f2-f1
    d = np.vstack([df,np.zeros(bnd.shape)])
    return d

# --------------------------------------------------------------
# Load the reference mesh and get the boundary points
# --------------------------------------------------------------
fnb = "mesh_2d"
ref_msh_fname = "mesh_2d.msh"
tags = [1]

ref_mesh = meshio.read(ref_msh_fname)
ref_bnd_pts = get_boundary_pts(ref_mesh,tags)

# --------------------------------------------------------------
# Define points along the curve where the reference plane lies 
# and points along the target curve, 
# and finally visualize the points
# --------------------------------------------------------------
fault_straight = np.linspace([0,1,0],[0,0,0],12)
fault_angled = np.linspace([0,1,0],[-0.499, 1-0.866, 0,], 12)

plt.figure()
plt.scatter(fault_straight[:,0],fault_straight[:,1], label='reference points')
plt.scatter(fault_angled[:,0],fault_angled[:,1], label='target points')
plt.scatter(ref_bnd_pts[:,0],ref_bnd_pts[:,1], label='ref mesh boundary points')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Create the angle to displacement field RBF interpolator
# The angle is relative to the vertical plane. 
# --------------------------------------------------------------
df = fault_angled - fault_straight
dd = np.vstack([df,np.zeros(ref_bnd_pts.shape)])
ndd = dd.shape
DD = np.vstack([ np.zeros(dd.ravel().shape),dd.ravel()])
all_angles = np.atleast_2d(np.array([0.0, 20.0]))

rbf1 = RBFInterpolator(all_angles.T,DD,kernel="linear")

# --------------------------------------------------------------
# Evaluate the first RBF interpolator at a 20 degree angle. 
# This is trivial since it was given data for a plane at a 20
# degree angle, but it allows us to check against a "known"
# plane location.  
# angle_eval can be changed to values between [0, 20]. 
# Then, define another RBF interpolant that takes points along 
# the reference plane location and the ref mesh boundary points. 
# --------------------------------------------------------------
angle_eval = np.array(20.0,ndmin=2) 
disp = rbf1(angle_eval)
disp = disp.reshape(ndd)
X = np.vstack([fault_straight,ref_bnd_pts])
rbf2 = RBFInterpolator(X,disp,kernel="linear")

# --------------------------------------------------------------
# Evaluate the RBF interpolant at all vertices of the reference 
# mesh, and apply the displacement to the vertices to obtain
# the morphed mesh. 
# --------------------------------------------------------------

disp2  = rbf2(ref_mesh.points)

def_mesh = deepcopy(ref_mesh)
X_d = def_mesh.points + disp2
def_mesh.points = X_d

def_mesh.write(fnb+"_deformed.msh",file_format="gmsh22",binary=False)

# --------------------------------------------------------------
# Plotting section
# Evaluate the displacement at a structured grid for plotting
# purposes, and make some schematic figures. 
# --------------------------------------------------------------
n = 14
X,Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
disp_grid = 0.6*rbf2( np.vstack([X.flatten(), Y.flatten(), np.zeros(Y.flatten().shape)]).T)
grid = pv.PolyData(np.vstack([X.flatten(), Y.flatten(), np.zeros(Y.flatten().shape)]).T)
grid["disp"] = disp_grid

ref = pv.read("mesh_2d.msh")
ref["disp"] = disp2

s1 = pv.read(fnb+"_deformed.msh")

lw = 16

pl = pv.Plotter(off_screen=True, window_size=[12000, 4000], shape=(1,4), border=False)
pl.subplot(0,0)
pl.add_mesh(ref.threshold(value=(1,5), scalars="gmsh:physical"), style='wireframe', color='black', lighting=False, line_width=lw*4)
pl.add_mesh(ref, style='wireframe', color='lightgray', lighting=False, line_width=lw)
pl.camera_position = 'xy'

pl.subplot(0,1)
pl.add_mesh(ref.threshold(value=(1,5), scalars="gmsh:physical"), style='wireframe', color='black', lighting=False, line_width=lw*4)
pl.add_mesh(grid.glyph(scale="disp", orient="disp", tolerance=0.05), color='crimson', lighting=False)
pl.camera_position = 'xy'

pl.subplot(0,2)
pl.add_mesh(ref.threshold(value=(1,5), scalars="gmsh:physical"), style='wireframe', color='black', lighting=False, line_width=lw*4)
pl.add_mesh(ref, style='wireframe', color='lightgray', lighting=False, line_width=lw)
pl.add_mesh(ref.glyph(scale="disp", orient="disp", tolerance=0.05), color='crimson', lighting=False)
pl.camera_position = 'xy'

pl.subplot(0,3)
pl.add_mesh(s1.threshold(value=(1,5), scalars="gmsh:physical"), style='wireframe', color='black', lighting=False, line_width=lw*4)
pl.add_mesh(s1, style='wireframe', color='crimson', lighting=False, line_width=lw)
pl.camera_position = 'xy'
pl.screenshot("schematic_simple.png")