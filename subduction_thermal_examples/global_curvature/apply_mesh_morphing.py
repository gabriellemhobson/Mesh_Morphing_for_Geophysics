import meshio
import pandas as pd
import numpy as np
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from modules.mesh_morph_subduction import Mesh_Morph

from time import perf_counter

def get_mesh_qual_info(fname,quality_measure):
    mesh = pv.read(fname)
    qual = mesh.compute_cell_quality(quality_measure=quality_measure)
    threshed = qual.threshold(value=(16,19), scalars="gmsh:physical")
    return threshed

log = pd.read_csv("training_log.csv")

# load reference mesh
eval_fnb = "eval_test_alpha"
y_x_merged = "training_y_x_merged.csv"
ref_msh_fname = log["mesh_file"].loc[log["reference"]].values[0]
ref_alpha = log["alpha"].loc[log["reference"]].values[0]
print(ref_alpha)

ref_msh = meshio.read(ref_msh_fname)

# tag list for reference (see geo file)
# "surface", 20
# "overplate_right", 21
# "overplate_base", 22
# "slab_left", 24
# "slab_overplate_int", 25
# "slab_wedge_int", 27
# "slab_right", 28
# "slab_base", 29
# "wedge_base", 30
# "outflow_wedge", 31
# "inflow_wedge", 32

tag_dict = {"zero_displacement":[21, 24], \
            "slab_interface":[25, 27], \
            "physical_pt":[42], \
            "corner_pt":[43], \
            "vertical_only":[31,32], \
            "slab_base":[29], \
            "slab_left":[24], \
            "slab_right":[28], \
            "zero_displacement_under_base_adjustment":[20, 21, 22, 25, 27, 31, 32, 30], \
            "zero_displacement_under_slab_left_adjustment":[20, 21, 22, 25, 27, 28, 29, 30, 31, 32], \
            "zero_displacement_under_slab_right_adjustment":[20, 21, 22, 24, 25, 27, 29, 30, 31, 32]}

slab_thickness = 100.0
overplate_thickness = 2.0
kernel = 'linear'


out_dir = 'meshes_morphed'
MM = Mesh_Morph(slab_thickness, kernel, y_x_merged, ref_alpha, eval_fnb, ref_msh_fname, tag_dict, out_dir)

train_dirs = list(log["alpha"].loc[~log["reference"]].values)

params = np.atleast_2d(log["alpha"].loc[~log["reference"]].values).T

t1 = perf_counter()
ref_arr, td_arr, DD = MM.build_MM_RBF(len(train_dirs), train_dirs, params)
t2 = perf_counter()

print('time to build_MM_RBF', t2-t1)

sample = np.atleast_2d(np.linspace(5e-4,3.5e-3,13)).T

print('sample', sample)

qm = 'scaled_jacobian'
fname_ref = log[log["reference"]==True]["mesh_file"].values[0]
qual_ref = get_mesh_qual_info(fname_ref,qm)
print('Reference mesh cell quality using '+qm)
print('Min:', np.min(qual_ref['CellQuality']), ', Max:', np.max(qual_ref['CellQuality']))
print('Average:', np.mean(qual_ref['CellQuality']))

fs = 18

qual_minmax = np.zeros((3,sample.shape[0]+1))
qual_avg = np.zeros((3,sample.shape[0]+1))

for i in range(sample.shape[0]):
    s = sample[i]
    s = np.round(s, 5)
    print('alpha:', s)

    t1 = perf_counter()
    m_out, fname_out = MM.eval_RBF(s)
    t2 = perf_counter()
    print('Time to eval MM', t2 - t1)

    qual_mm_AR = get_mesh_qual_info(fname_out,"aspect_ratio")
    qual_avg[0,i] = np.mean(qual_mm_AR['CellQuality'])
    qual_minmax[0,i] = np.max(qual_mm_AR['CellQuality'])

    qual_mm_SJ = get_mesh_qual_info(fname_out,"scaled_jacobian")
    qual_avg[1,i] = np.mean(qual_mm_SJ['CellQuality'])
    qual_minmax[1,i] = np.min(qual_mm_SJ['CellQuality'])

    qual_mm_MA = get_mesh_qual_info(fname_out,"min_angle")
    qual_avg[2,i] = np.mean(qual_mm_MA['CellQuality'])
    qual_minmax[2,i] = np.min(qual_mm_MA['CellQuality'])

# add reference
qual_mm_AR = get_mesh_qual_info(fname_ref,"aspect_ratio")
qual_avg[0,-1] = np.mean(qual_mm_AR['CellQuality'])
qual_minmax[0,-1] = np.max(qual_mm_AR['CellQuality'])

qual_mm_SJ = get_mesh_qual_info(fname_ref,"scaled_jacobian")
qual_avg[1,-1] = np.mean(qual_mm_SJ['CellQuality'])
qual_minmax[1,-1] = np.min(qual_mm_SJ['CellQuality'])

qual_mm_MA = get_mesh_qual_info(fname_ref,"min_angle")
qual_avg[2,-1] = np.mean(qual_mm_MA['CellQuality'])
qual_minmax[2,-1] = np.min(qual_mm_MA['CellQuality'])

alpha_arr = np.vstack([sample, ref_alpha]).flatten()

wd = 0.0002

plt.figure(figsize=(8,9))
plt.bar(alpha_arr,qual_minmax[0,:], color='darkblue',label="Max AR", width=wd)
plt.bar(alpha_arr,qual_avg[0,:], color='royalblue',label="Avg AR", width=wd)
plt.bar(alpha_arr[-1],qual_minmax[0,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=6, width=wd-(wd/8))
plt.plot([np.min(params),np.max(params)],[1,1],linestyle='--', color='tab:green', label='Ideal AR', linewidth=4)
plt.xlim([0.3e-3,3.7e-3])
plt.xticks(ticks=alpha_arr,fontsize=fs, rotation=60, labels=np.round(alpha_arr*1e3, 6))
plt.yticks(fontsize=fs)
plt.xlabel(r'$\alpha$ x $10^{3}$',fontsize=fs)
plt.ylabel('AR',fontsize=fs)
plt.legend(fontsize=fs,loc='upper right')
plt.savefig("AR_bar.png",dpi=1000)

plt.figure(figsize=(8,9))
plt.bar(alpha_arr,qual_avg[1,:], width=wd, color='teal',label="Avg SJ")
plt.bar(alpha_arr,qual_minmax[1,:], width=wd, color='lightseagreen',label="Min SJ")
plt.bar(alpha_arr[-1],qual_avg[1,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=6, width=wd)
plt.plot([np.min(params),np.max(params)],[1,1], linestyle='--', color='tab:green',label='Ideal SJ', linewidth=4)
plt.xlim([0.3e-3,3.7e-3])
plt.xticks(ticks=alpha_arr,fontsize=fs, rotation=60, labels=np.round(alpha_arr*1e3, 6))
plt.yticks(fontsize=fs)
plt.xlabel(r'$\alpha$ x $10^{3}$',fontsize=fs)
plt.ylabel('SJ',fontsize=fs)
plt.legend(fontsize=fs,loc='center right')
plt.savefig("SJ_bar.png",dpi=1000)

plt.figure(figsize=(8,9))
plt.bar(alpha_arr,qual_avg[2,:], width=wd, color='goldenrod',label="Avg MA")
plt.bar(alpha_arr,qual_minmax[2,:], width=wd, color='gold',label="Min MA")
plt.bar(alpha_arr[-1],qual_avg[2,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=6, width=wd)
plt.plot([np.min(params),np.max(params)],[60,60],linestyle='--', color='tab:green',label='Ideal MA', linewidth=4)
plt.xlim([0.3e-3,3.7e-3])
plt.xticks(ticks=alpha_arr,fontsize=fs, rotation=60, labels=np.round(alpha_arr*1e3, 6))
plt.yticks(fontsize=fs)
plt.xlabel(r'$\alpha$ x $10^{3}$',fontsize=fs)
plt.ylabel('MA',fontsize=fs)
plt.legend(fontsize=fs,loc='center right')
plt.savefig("MA_bar.png",dpi=1000)

print('Average mesh quality:') 
print(qual_avg)
print('Min or max mesh quality:')
print(qual_minmax)
