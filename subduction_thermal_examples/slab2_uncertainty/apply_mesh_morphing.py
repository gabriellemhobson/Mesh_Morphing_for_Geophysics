import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import meshio
from modules.mesh_morph_subduction import Mesh_Morph
import pyvista as pv

def get_mesh_qual_info(fname,quality_measure):
    mesh = pv.read(fname)
    qual = mesh.compute_cell_quality(quality_measure=quality_measure)
    threshed = qual.threshold(value=(16,19), scalars="gmsh:physical")
    return threshed

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(np.abs(stop), 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 

profiles = ["cascadia_profile_B"]

N = 9
beta = np.linspace(-1.1, 1.1, N)
print('beta', beta)

for profile in profiles:
    xy = np.loadtxt(os.path.join('cascadia', profile+'.xy'))
    val = np.loadtxt(os.path.join(os.path.join('cascadia',profile), "profile_unc.csv"))

    min_y = -800
    for k in range(beta.shape[0]):
        b = beta[k]
        sh = np.vstack([xy[:,0], xy[:,1]+b*val]).T
        sh = sh[sh[:,1]<0,:]
        if np.min(sh[:,1]) > min_y:
            min_y = np.min(sh[:,1])
            print(min_y)

    xy_b = np.zeros((xy[xy[:,1]>min_y,1].shape[0], beta.shape[0]))

    df = pd.DataFrame(xy[xy[:,1]>min_y,1], columns=["y"])
    df["x_{}".format(0.0)] = xy[xy[:,1]>min_y,0]

    for k in range(beta.shape[0]):

        b = beta[k]
        sh = np.vstack([xy[:,0], xy[:,1]+b*val]).T
        sh = sh[sh[:,1]<-0.5,:]

        dist = 120.0
        I = np.argmin(np.abs(sh[:,0]-dist))
        plt.plot(sh[:,0], sh[:,1], color='m')

        sh[0:I, 0] = np.linspace(0.0,sh[I,0],I)
        sh[0:I, 1] = -powspace(0.0,sh[I,1], 2, I)
        
        f = interpolate.interp1d(sh[:,1], sh[:,0])

        xy_b[:,k] = f(xy[xy[:,1]>min_y,1]).flatten()

        df["x_{}".format(b)] = xy_b[:,k]

    df.to_csv("y_x_merged.csv", index=False)

    tag_dict = {"zero_displacement":[21, 24], \
                "slab_interface":[25, 27], \
                "physical_pt":[42], \
                "vertical_only":[31, 32], \
                "slab_base":[29], \
                "slab_left":[24], \
                "slab_right":[28], \
                "zero_displacement_under_base_adjustment":[20, 21, 22, 25, 27, 31, 32, 30], \
                "zero_displacement_under_slab_left_adjustment":[20, 21, 22, 25, 27, 28, 29, 30, 31, 32], \
                "zero_displacement_under_slab_right_adjustment":[20, 21, 22, 24, 25, 27, 29, 30, 31, 32]}

    slab_thickness = 40.0
    kernel = 'linear'

    ref_alpha = 0.0
    eval_fnb = "beta"
    ref_msh_fname = os.path.join(os.path.join("cascadia", profile), profile+".msh")
    ref_msh = meshio.read(ref_msh_fname)

    out_dir = "meshes_morphed"
    os.makedirs(out_dir, exist_ok=True)
    MM = Mesh_Morph(slab_thickness, kernel, "y_x_merged.csv", ref_alpha, eval_fnb,ref_msh_fname,tag_dict, out_dir)

    train_dirs = list(beta)

    params = np.atleast_2d(beta).T

    ref_arr, td_arr, DD = MM.build_MM_RBF(N, train_dirs, params)
    
    n_samples = 9
    
    sample = np.atleast_2d(np.linspace(-1.0, 1.0, n_samples)).T
    print('sample', sample)
    qual_minmax = np.zeros((3,sample.shape[0]+1))
    qual_avg = np.zeros((3,sample.shape[0]+1))

    fs = 18

    for i in range(sample.shape[0]):
        s = sample[i]
        m_out, fname_out = MM.eval_RBF(s)

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
    fname_ref = os.path.join(os.path.join('cascadia', profile), profile+'.msh')
    qual_mm_AR = get_mesh_qual_info(fname_ref,"aspect_ratio")
    qual_avg[0,-1] = np.mean(qual_mm_AR['CellQuality'])
    qual_minmax[0,-1] = np.max(qual_mm_AR['CellQuality'])

    qual_mm_SJ = get_mesh_qual_info(fname_ref,"scaled_jacobian")
    qual_avg[1,-1] = np.mean(qual_mm_SJ['CellQuality'])
    qual_minmax[1,-1] = np.min(qual_mm_SJ['CellQuality'])

    qual_mm_MA = get_mesh_qual_info(fname_ref,"min_angle")
    qual_avg[2,-1] = np.mean(qual_mm_MA['CellQuality'])
    qual_minmax[2,-1] = np.min(qual_mm_MA['CellQuality'])

    beta_arr = np.vstack([sample, 0.0]).flatten()

    wd = 0.15

    plt.figure(figsize=(8,8))
    plt.bar(beta_arr,qual_minmax[0,:], color='darkblue',label="Max AR", width=wd)
    plt.bar(beta_arr,qual_avg[0,:], color='royalblue',label="Avg AR", width=wd)
    plt.bar(beta_arr[-1],qual_minmax[0,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=3, width=wd)

    plt.plot([np.min(params),np.max(params)],[1,1],linestyle='--', color='tab:green',label='Ideal AR', linewidth=4)
    plt.xlim([-1.15,1.15])
    plt.xticks(ticks=sample[:,0],fontsize=fs,rotation=60)
    plt.yticks(fontsize=fs)
    plt.xlabel(r'$\beta$',fontsize=fs)
    plt.ylabel('AR',fontsize=fs)
    plt.legend(fontsize=fs,loc='upper left')
    plt.savefig("AR_bar.png",dpi=1000)

    plt.figure(figsize=(8,8))
    plt.bar(beta_arr,qual_avg[1,:], width=wd, color='teal',label="Avg SJ")
    plt.bar(beta_arr,qual_minmax[1,:], width=wd, color='lightseagreen',label="Min SJ")
    plt.bar(beta_arr[-1],qual_avg[1,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=3, width=wd)
    plt.plot([np.min(params),np.max(params)],[1,1],linestyle='--', color='tab:green',label='Ideal SJ', linewidth=4)
    plt.xlim([-1.15,1.15])

    plt.xticks(ticks=sample[:,0],fontsize=fs,rotation=60)
    plt.yticks(fontsize=fs)
    plt.xlabel(r'$\beta$',fontsize=fs)
    plt.ylabel('SJ',fontsize=fs)
    plt.legend(fontsize=fs,loc='center right')
    plt.savefig("SJ_bar.png",dpi=1000)

    plt.figure(figsize=(8,8))
    plt.bar(beta_arr,qual_avg[2,:], width=wd, color='goldenrod',label="Avg MA")
    plt.bar(beta_arr,qual_minmax[2,:], width=wd, color='gold',label="Min MA")
    plt.bar(beta_arr[-1],qual_avg[2,-1], facecolor=None, edgecolor='darkorange', fill=False, linestyle='-', linewidth=3, width=wd)
    plt.plot([np.min(params),np.max(params)],[60,60],linestyle='--', color='tab:green',label='Ideal MA', linewidth=4)
    plt.xlim([-1.15,1.15])
    plt.xticks(ticks=sample[:,0],fontsize=fs, rotation=60)
    plt.yticks(fontsize=fs)
    plt.xlabel(r'$\beta$',fontsize=fs)
    plt.ylabel('MA',fontsize=fs)
    plt.legend(fontsize=fs,loc='center right')
    plt.savefig("MA_bar.png",dpi=1000)

    print('Average:') 
    print(qual_avg)
    print('Min or max:')
    print(qual_minmax)
