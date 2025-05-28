import pyvista as pv
import numpy as np
import os
import matplotlib
from slab_profile import em_quadratic_profile

pv.global_theme.background = 'white'
pv.global_theme.font.family = 'times'
pv.global_theme.font.size = 20
pv.global_theme.font.title_size = 20
pv.global_theme.font.label_size = 20
pv.global_theme.font.color = 'black'

def get_mesh_qual_info(fname,quality_measure):
    mesh = pv.read(fname)
    qual = mesh.compute_cell_quality(quality_measure=quality_measure)
    threshed = qual.threshold(value=(16,19), scalars="gmsh:physical")
    return threshed

alpha_vals = np.hstack([0.00175, np.linspace(5e-4,3.5e-3,7)[::-1]])

cmap = matplotlib.cm.get_cmap('twilight')
color_vals = [cmap(r) for r in list(np.linspace(0.1,0.9,len(alpha_vals-1)))]

y = np.linspace(0.1e3, 300.0*1e3, 800)

pl = pv.Plotter(window_size=[7000,4000], off_screen=True)
for i in range(1, len(alpha_vals)):
    a = np.round(alpha_vals[i],8)
    fname = os.path.join(os.path.join("meshes_morphed", "alpha_{}".format(a)), "eval_test_alpha_adjusted3_{}.vtu".format(a))
    mesh = pv.read(fname).threshold(value=(20,34), scalars="gmsh:physical")
    pl.add_mesh(mesh, style='wireframe', line_width=12, color=color_vals[i], label=str(a*1e3))

    # add points along the interface
    profile = em_quadratic_profile(a)
    x = profile.x_from_zf(y)
    
    point_cloud = pv.PolyData(np.vstack([x/1e3,-y/1e3,np.zeros(x.shape[0])]).T[::20,:])
    pl.add_mesh(point_cloud, color=color_vals[i], point_size=60.0, render_points_as_spheres=False)
    
a = alpha_vals[0]
fname = os.path.join(os.path.join("mesh_reference", "alpha_{}".format(a)), "alpha_{}.msh".format(a))
mesh = pv.read(fname).threshold(value=(20,34), scalars="gmsh:physical")
pl.add_mesh(mesh, style='wireframe', line_width=16, color='k', label=str(a*1e3)+" (ref)")

# pl.add_legend(face='-')
pl.camera_position = 'xy'
pl.zoom_camera(1.75)
pl.screenshot("variability_alpha_wireframe.png")

