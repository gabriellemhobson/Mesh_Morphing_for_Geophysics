import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from slab_profile import em_quadratic_profile

# -------------------------------------------------------------------
# Setup choices: edit variables in this block to suit
# -------------------------------------------------------------------

step = "training"
n = 18
alpha_range = np.linspace(4.5e-4,4.0e-3,n+2)

fs = 16
mesh_output_path = "mesh_reference"
target_output_path = "target_data"

ref_val = 0.00175

# -------------------------------------------------------------------
# Set up log and x y arrays
# -------------------------------------------------------------------

log = pd.DataFrame(columns=["alpha", "reference", "mesh_dir", "mesh_file"])
log_fname = step+"_log.csv"

y = np.linspace(0.1e3, 300.0*1e3, 800)

x_arr_dense = np.zeros((y.shape[0],alpha_range.shape[0]+1))

# -------------------------------------------------------------------
# get the maximum of the along-slab lengths when alpha varies
# -------------------------------------------------------------------

max_x = 0
for i in range(len(alpha_range)):
    alpha = alpha_range[i]
    profile = em_quadratic_profile(alpha)
    x = profile.x_from_zf(y)
    if np.max(x) > max_x:
        max_x = np.max(x)

# -------------------------------------------------------------------
# Generate reference mesh profile info
# -------------------------------------------------------------------

alpha = ref_val
profile = em_quadratic_profile(alpha)
x = profile.x_from_zf(y)

x_arr_dense[:,0] = x

x_by_z = np.vstack([x/1e3, -y/1e3]).T

output_subfolder = os.path.join(mesh_output_path, "alpha_{}".format(np.round(alpha,8)))
os.makedirs(output_subfolder, exist_ok=True)
geo_filename = "alpha_{}.geo".format(np.round(alpha,8))

y_by_x = np.vstack([-y/1e3, x/1e3]).T
df = pd.DataFrame(y_by_x, columns=["y", "x_{}".format(np.round(alpha,8))])
df.to_csv(os.path.join(output_subfolder, "y_x.csv"), index=False)

new_log_row = pd.Series({"alpha":np.round(alpha,8), "reference":True, "mesh_dir":output_subfolder, "mesh_file":os.path.join(output_subfolder, geo_filename[:-4]+'.msh')})
log = pd.concat([log, new_log_row.to_frame().T], ignore_index=True)

# -------------------------------------------------------------------
# Generate the target point data
# -------------------------------------------------------------------

for i in range(len(alpha_range)):
    alpha = alpha_range[i]
    profile = em_quadratic_profile(alpha)
    x = profile.x_from_zf(y)

    x_arr_dense[:,i] = x

    x_by_z = np.vstack([x/1e3, -y/1e3]).T

    output_subfolder = os.path.join(target_output_path, "alpha_{}".format(np.round(alpha,8)))
    os.makedirs(output_subfolder, exist_ok=True)
    
    y_by_x = np.vstack([-y/1e3, x/1e3]).T
    df = pd.DataFrame(y_by_x, columns=["y", "x_{}".format(np.round(alpha,8))])
    df.to_csv(os.path.join(output_subfolder, "y_x.csv"), index=False)

    new_log_row = pd.Series({"alpha":np.round(alpha,8), "reference":False, "mesh_dir":output_subfolder, "mesh_file":False})
    log = pd.concat([log, new_log_row.to_frame().T], ignore_index=True)


log.to_csv(log_fname, index=False)

# -------------------------------------------------------------------
# Use pandas magic to clean up and concatenate all the x_y files 
# -------------------------------------------------------------------

df_list = []

# first get the reference mesh data
output_subfolder = os.path.join(mesh_output_path, "alpha_{}".format(np.round(ref_val,8)))
df = pd.read_csv(os.path.join(output_subfolder, "y_x.csv"), delimiter=",")
df_list.append(df)

# then get the target point data
for i in range(len(alpha_range)):
    alpha = alpha_range[i]
    output_subfolder = os.path.join(target_output_path, "alpha_{}".format(np.round(alpha,8)))

    df = pd.read_csv(os.path.join(output_subfolder, "y_x.csv"), delimiter=",")
    df_list.append(df)

df_new = df_list[0]
for i in range(1,len(df_list)):
    df_mid = df_list[i].drop_duplicates(subset=df_list[i].columns[1:], inplace=False)
    df_new = df_new.merge(df_list[i], how='inner', on='y')

df_new.to_csv(step+"_y_x_merged.csv", index=False)

# -------------------------------------------------------------------
# Optional: plot the points that were generated
# -------------------------------------------------------------------

plt.figure(figsize=(8, 8))
for i in range(alpha_range.shape[0]):
    plt.scatter(x_arr_dense[:,i]/1e3, -y/1e3, c='r', s=30)
plt.xlabel('x (km)', fontsize=fs)
plt.ylabel('zf (km)', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.axis('scaled')
plt.show()