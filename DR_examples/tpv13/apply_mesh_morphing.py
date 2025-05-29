import os
import numpy as np
import subprocess
from pathlib import Path
import meshio
from extrude_and_generate_vertices import Extrude_Fault
from modules.mesh_morph_DR import Mesh_Morph

from time import perf_counter

#------------------------------------
# Decide whether to create exact meshes
# for each evaluated dip value
#------------------------------------
WRITE_EXACT = False

#------------------------------------
# Function to update base seissol files
#------------------------------------

def update_base_files(ss_fnb, theta, fnb, short_fnb, out_dir, n_patch):
    msh_file = fnb+"_{}.msh".format(theta)
    command = "pumgen" + ' ' + "-s" + ' ' + "msh2" + ' ' + msh_file + "\n"
    f_commands.write(command)

    os.makedirs(out_dir,exist_ok=True)

    fault_fname = os.path.join(out_dir, "fault_{}.yaml".format(theta))
    init_stress_fname = os.path.join(out_dir, "initial_stress_{}.yaml".format(theta))
    material_fname = os.path.join(out_dir, "material_{}.yaml".format(theta))
    param_fname = os.path.join(out_dir, "parameters_{}.yaml".format(theta))
    mesh_fname = fnb+"_{}".format(theta) + ".puml.h5"
    receiver_fname = os.path.join(out_dir, "receivers_{}.dat".format(theta))

    # fault file
    fname_in = os.path.join(ss_fnb, "base_fault.yaml")
    with open(fname_in, 'r') as f:
        data = f.readlines()
    b1 = np.min(n_patch,axis=0).ravel(); b2 = np.max(n_patch,axis=0).ravel()
    data[1] = '[s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]: !Include ' + init_stress_fname + '\n'
    data[7] = '        x: [{}, {}]\n'.format(b1[0],b2[0])
    data[9] = '        z: [{}, {}]\n'.format(b1[2],b2[2])
    with open(fault_fname, 'w') as f:
        f.writelines( data )

    # parameters file
    fname_in = os.path.join(ss_fnb, "base_parameters.par")
    with open(fname_in, 'r') as f:
        data = f.readlines()
    data[2] = "MaterialFileName = \'" + material_fname + "\' \n"
    data[21] = "ModelFileName = \'" + fault_fname + "\'\n"
    data[57] = "MeshFile = \'" + mesh_fname + "\' \n"
    data[68] = "OutputFile = \'" + os.path.join(out_dir, short_fnb + "_{}".format(theta)) + "\' \n"
    data[86] = "RFileName = \'" + receiver_fname+ "\'      ! Record Points in extra file"
    with open(param_fname, 'w') as f:
        f.writelines( data )

    # initial stress file
    fname_in = os.path.join(ss_fnb, "base_initial_stress.yaml")
    with open(fname_in, 'r') as f:
        data = f.readlines()

    data[14] = '        depth:          [0, ' + str(np.abs(b1[2])+260.0) + '] \n'
    with open(init_stress_fname, 'w') as f:
        f.writelines( data )

    # material file
    fname_in = os.path.join(ss_fnb, "base_material.yaml")
    with open(fname_in, 'r') as f:
        data = f.readlines()
    data[8] = '[s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]: !Include ' + init_stress_fname + '\n'
    with open(material_fname, 'w') as f:
        f.writelines( data )

    # receiver file
    fname_in = os.path.join(ss_fnb, "base_receivers.dat")
    with open(fname_in, 'r') as f:
        data = f.readlines()
    with open(receiver_fname, 'w') as f:
        f.writelines( data )
    print('Updated base files, wrote to directory', out_dir)

    return param_fname

#------------------------------------
# Define reference geometry parameters
#------------------------------------

# Here we define some necessary inputs containing information 
# about the geometry we want to work with.
# This geometry follows the TPV13 geometry and mesh parameters. 

# The input file "trace_angle.csv" defines points along the trace of the TPV 13 fault. 
# Since TPV 13 is planar we only need the corner points, 
# but the code works for curved faults also. 

# We also define a reference dip, `dip_ref`, which we will use 
# to extrude the trace and build a reference `.geo` file. 
# This reference geometry will be meshed and then morphed 
# using an RBF trained on all the extruded faults. 

fname_tr = "trace_angle.csv"
dip_ref = 60.0 
L = 15000.0 # length of extruded fault
h_n_patch = 100
h_fine = 250 
h_max = 2500 

bb = np.array([[-45000.0, -36000.0, 0.0],\
               [45000.0, -36000.0, 0.0],\
               [-45000.0, 36000.0, 0.0],\
               [45000.0, 36000.0, 0.0],\
               [-45000.0, -36000.0, -42000.0],\
               [45000.0, -36000.0, -42000.0],\
               [-45000.0, 36000.0, -42000.0],\
               [45000.0, 36000.0, -42000.0]])

print('Reference dip value: ', dip_ref)
np.savetxt("dip_ref.csv",np.atleast_2d(dip_ref), header="dip_ref")

# nucleation patch info
 
# Hypocenter, z_n is here the depth along-dip
x_n = 0e3;
z_n = 12e3; # must be positive sign in my code
# size of the nucleation patch
l_n = 3e3;
w_n = 3e3;
npatch = [x_n, z_n, l_n, w_n]

#------------------------------------
#  Define training dip values
#------------------------------------

# We define `dip_train` values; we will extrude the trace at each of these values 
# and save the fault points to `.pkl` files.

# The dip_train values are logged to a csv file. 

dip_train = np.linspace(80.0,40.0,18)
np.savetxt("dip_train.csv",dip_train,header='dip_train')
print('Training dip values: ', dip_train)

#------------------------------------
# Make the reference geometry
#------------------------------------
t1 = perf_counter()
fpath = os.path.join(os.getcwd(),"reference")
Path(fpath).mkdir(parents=True, exist_ok=True)

ref_fnb = os.path.join(fpath,"reference")
ext = Extrude_Fault(fname_tr, ref_fnb, L, h_n_patch, h_fine, h_max, bb, npatch)
ext.constant_dip(dip_in=dip_ref, write_geo=True)

ref_geo = ref_fnb+"_{}.geo".format(dip_ref)
command =["gmsh", "-3", ref_geo, "-v", "0"]
p = subprocess.run(command)
t2 = perf_counter()
print('Time to create reference mesh:', t2 - t1)

# msh = meshio.read(ref_fnb+"_{}.msh".format(dip_ref))
# msh.write(ref_fnb+"_{}.vtk".format(dip_ref), )

#------------------------------------
# Extrude the fault for each training dip
#------------------------------------

fpath = os.path.join(os.getcwd(),"training")
Path(fpath).mkdir(parents=True, exist_ok=True)

train_fnb = os.path.join(fpath,"training")
ext = Extrude_Fault(fname_tr, train_fnb, L, h_n_patch, h_fine, h_max, bb, npatch)
for dip in dip_train:
    ext.constant_dip(dip_in=dip, write_geo=False)

#------------------------------------
# Define evaluated dips
#------------------------------------

dip_eval = np.linspace(40.0,80.0,9)
np.savetxt("dip_eval.csv",dip_eval,header='dip_eval')
print('Evaluated dip values: ', dip_eval)

#------------------------------------
# Create exact meshes for evaluated dips
#------------------------------------

if WRITE_EXACT:
    fpath = os.path.join(os.getcwd(),"exact")
    Path(fpath).mkdir(parents=True, exist_ok=True)

    exact_fnb = os.path.join(fpath,"exact")
    ext = Extrude_Fault(fname_tr, exact_fnb, L, h_n_patch, h_fine, h_max, bb, npatch)
    for dip in dip_eval:
        t1 = perf_counter()
        npc = ext.constant_dip(dip_in=dip, write_geo=True)
        print('Exact mesh nucleation patch corners, dip', dip)
        print(npc)

        exact_geo = exact_fnb+"_{}.geo".format(dip)
        command =["gmsh", "-3", exact_geo, "-v", "0"]
        p = subprocess.run(command)

        msh = meshio.read(exact_fnb+"_{}.msh".format(dip))
        msh.write(exact_fnb+"_{}.vtk".format(dip))
        t2 = perf_counter()
        print("Time to generate exact mesh:", t2 - t1)

#------------------------------------
# Build MM RBF
#------------------------------------

fpath = os.path.join(os.getcwd(),"evaluate")
Path(fpath).mkdir(parents=True, exist_ok=True)

eval_fnb = os.path.join(fpath,"evaluate")
ref_msh = ref_fnb+"_{}.msh".format(dip_ref)
tags = [1,5]


t1 = perf_counter()
MM = Mesh_Morph(ref_fnb,train_fnb,eval_fnb,ref_msh,tags,kernel='linear')
ref_fault_fn = ref_fnb+"_{}.pkl".format(dip_ref)
MM.build_MM_RBF(dip_ref,dip_train,ref_fault_fn)
t2 = perf_counter()
print("Time to build MM:", t2-t1)

#------------------------------------
# Evaluate MM RBF at all testing dips
#------------------------------------

n_patch_all = []
limits_all = []
for k in range(dip_eval.shape[0]):
    theta = dip_eval[k]
    print(theta)
    print('starting eval')
    t1 = perf_counter()
    n_patch,limits = MM.eval_RBF(theta)
    t2 = perf_counter()
    print("Time to eval MM:", t2-t1)
    print('done with eval')
    n_patch_all.append(n_patch)
    limits_all.append(limits)

#-------------------------------------
# Write commands to a script that will 
# be run in the seissol environment
#-------------------------------------

prep_run_file = "prep_run_seissol.sh"

ss_fnb = "seissol_base_files"

seissol_path = "$HOME/SeisSol/build-release-rome/SeisSol_Release_drome_4_elastic"

with open(prep_run_file, 'w') as f_commands:
    out_dir = "output/seissol_dip_ref_{}".format(dip_ref)
    msh_file = ref_fnb+"_{}.msh".format(dip_ref)
    refmesh_temp = meshio.read(msh_file)
    n_patch = refmesh_temp.points[refmesh_temp.cells_dict["vertex"]]
    param_fname = update_base_files(ss_fnb, dip_ref, ref_fnb, 'seissol_dip_ref', out_dir, n_patch)

    command = seissol_path + " " + param_fname + "\n"
    f_commands.write(command)
    
    for k in range(dip_eval.shape[0]):
        # eval jobs
        theta = dip_eval[k]
        out_dir = "output/seissol_dip_eval_{}".format(theta)
        n_patch = n_patch_all[k]
        param_fname = update_base_files(ss_fnb, theta, eval_fnb, "seissol_dip_eval", out_dir, n_patch)

        command = seissol_path + " " + param_fname + "\n"
        f_commands.write(command)

        # exact jobs
        # out_dir = "output/seissol_dip_exact_{}".format(theta)
        # msh_file = exact_fnb+"_{}.msh".format(theta)
        # exactmesh_temp = meshio.read(msh_file)
        # n_patch = exactmesh_temp.points[refmesh_temp.cells_dict["vertex"]]
        # param_fname = update_base_files(ss_fnb, theta, exact_fnb, 'seissol_dip_exact', out_dir, n_patch)

        # command = "$HOME/SeisSol/build-release-rome/SeisSol_Release_drome_4_elastic" + " " + param_fname + "\n"
        # f_commands.write(command)

#------------------------------------
# Check difference in nucleation patch area
#------------------------------------

for k in range(dip_eval.shape[0]):
    theta = dip_eval[k]
    n_patch = n_patch_all[k]
    aa = np.linalg.norm(n_patch[0] - n_patch[1])
    bb = np.linalg.norm(n_patch[0] - n_patch[2])

    print('Eval dip', theta, ' has n_patch area', np.round(aa*bb,2))
    print('n_patch', n_patch)