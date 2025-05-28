import numpy as np
import meshio
import pandas as pd
import os
from copy import deepcopy
from scipy.interpolate import RBFInterpolator

class Mesh_Morph:
    def __init__(self,slab_thickness, kernel, y_x_merged, ref_alpha, eval_fnb, ref_msh, tag_dict, out_dir):
        self.slab_thickness = slab_thickness
        self.kernel = kernel
        self.y_x_merged = pd.read_csv(y_x_merged)
        self.ref_alpha = ref_alpha
        self.eval_fnb = eval_fnb
        self.ref_mesh = meshio.read(ref_msh)
        self.ref_bnd_pts = self.get_boundary_pts(self.ref_mesh,tag_dict["zero_displacement"])
        self.physical_pt = self.get_physical_pt(self.ref_mesh, tag_dict["physical_pt"])
        self.v_only_pts = self.get_boundary_pts(self.ref_mesh, tag_dict["vertical_only"])
        self.tag_dict = tag_dict
        self.out_dir = out_dir

    def load_arr_from_merged(self, alpha):
        '''
        Pulls x, y data from pandas dataframe self.y_x_merged. 
        Input
            - alpha: the param value we are extracting data for.
        Output
            - arr: the x,y data. Starts at row of index 1 to avoid divide by 0 issue. 

        '''
        arr = self.y_x_merged[["y", "x_{}".format(alpha)]].to_numpy()
        arr = arr[1:,::-1]

        return arr

    def get_boundary_pts(self,mesh,tags):
        '''
        Pulls out vertex coordinates for physical lines with given tags. 
        Note that the part mesh.cell_data["gmsh:physical"][1] needs to change
        when we go from 2D to 3D meshes. 
        Input
            - mesh: meshio object. 
            - tags: list of integers, physical tags from gmsh. 
        Output
            - boundary_pts: numpy ndarray of x,y coordinates. 
        '''
        triangle_cells = mesh.cells_dict["line"]
        bp = []
        for tag in tags:
            cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][1]==tag)]
            pts = mesh.points[np.unique(cells.ravel())]
            for k in range(pts.shape[0]):
                bp.append(pts[k,:])
        boundary_pts = np.unique(np.array(bp),axis=0)
        if np.max(np.abs( boundary_pts[:,-1]-0.0)) < 1e-14:
            boundary_pts = boundary_pts[:,0:2]

        return boundary_pts
    
    def get_physical_pt(self,mesh,tags):
        triangle_cells = mesh.cells_dict["vertex"]
        bp = []
        for tag in tags:
            cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][0]==tag)]
            pts = mesh.points[np.unique(cells.ravel())]
            for k in range(pts.shape[0]):
                bp.append(pts[k,:])
        curve_pts = np.unique(np.array(bp),axis=0)
        if np.max(np.abs( curve_pts[:,-1]-0.0)) < 1e-14:
            curve_pts = curve_pts[:,0:2]

        return curve_pts

    def compute_displacement(self,f1,f2):
        df = f2-f1

        d = np.vstack([df,np.zeros(self.ref_bnd_pts.shape)])
        
        # if self.physical_corner_gmsh:
        rbf_int = RBFInterpolator(f1,df,kernel="linear")
        d_slab_end_corner = rbf_int(self.physical_pt)
        d_v = np.linspace(np.array([0.0, d_slab_end_corner[0][1]]), np.zeros((2)),self.v_only_pts.shape[0])
        d = np.vstack([d, d_v])

        return d
    
    def adjust_slab_base(self, mpts, bpts):        
        d_r = np.zeros(bpts.shape)

        for j in range(bpts.shape[0]):
            pt = bpts[j,:]
            if j > 0 and j < bpts.shape[0] - 1:
                aa = (mpts[j+1,:] - mpts[j-1,:])
                nn = aa[::-1]*np.array([1,-1])
                pn = mpts[j,:] + (self.slab_thickness/np.linalg.norm(nn))*nn
            elif j == bpts.shape[0] - 1:
                aa = (mpts[j,:] - mpts[j-1,:])
                nn = aa[::-1]*np.array([1,-1])
                pn = mpts[j,:] + (self.slab_thickness/np.linalg.norm(nn))*nn
            else:
                aa = (mpts[j+1,:] - mpts[j,:])
                nn = aa[::-1]*np.array([1,-1])
                pn = mpts[j,:] + (self.slab_thickness/np.linalg.norm(nn))*nn

            d_r[j,:] = pn - pt

        return d_r
    
    def adjust_slab_inflow_outflow(self, pts):
        # reorder pts so they are in decreasing y order
        II = np.argsort(pts[:,1])
        II_unravel = np.argsort(II)
        pts = pts[II,:]

        aa = pts[np.argmin([pts[:,1]]),:]
        bb = pts[np.argmax([pts[:,1]]),:]
        
        # compute distance between sucessive nodes 
        # save total distance of curve and line
        # make vector between start and end of line
        # scale vector by distance between sucessive nodes, 
        #   scaled by ratio of distances
        d_i = np.zeros((pts.shape[0]))
        for k in range(1,pts.shape[0]):
            d_i[k] = np.linalg.norm(pts[k,:]-pts[k-1,:])
        dist_curve = np.sum(d_i)
        dist_line = np.linalg.norm(aa-bb)
        lvec = bb-aa
        build = np.zeros(pts.shape)
        for k in range(pts.shape[0]):
            build[k,:] = aa + (lvec)*(np.sum(d_i[0:k+1])/dist_curve)
        d_s = build-pts
    
        II_unravel = II_unravel[II_unravel!=0]
        II_unravel = II_unravel[II_unravel!=(pts.shape[0]-1)]
        d_s = d_s[II_unravel, :]
        
        pts_cut = pts[II_unravel, :]

        return d_s, pts_cut

    def build_MM_RBF(self, n, train_alphas, params):
        ref_arr = self.load_arr_from_merged(self.ref_alpha)
        self.ref_arr = ref_arr

        D = []
        for k in range(n):
            td = train_alphas[k]
            td_arr = self.load_arr_from_merged(td)
            
            dd = self.compute_displacement(ref_arr,td_arr)
            
            self.ndd = dd.shape
            D.append(dd.ravel())
        DD = np.array(D)

        self.X = np.vstack([self.ref_arr,self.ref_bnd_pts, self.v_only_pts])

        self.rbf1 = RBFInterpolator(params,DD,kernel=self.kernel)

        return ref_arr, td_arr, DD

    def eval_RBF(self,theta):
        disp = self.rbf1(np.array(theta,ndmin=2))
        disp = disp.reshape(self.ndd)

        X_norm = self.X
        d_norm = disp

        # remove duplicate values in X_norm
        if np.unique(X_norm, axis=0).shape[0] < X_norm.shape[0]:
            print('Duplicate values in X_norm removed ('+str(X_norm.shape[0]-np.unique(X_norm).shape[0])+' value(s) found)')
            X_norm,idx = np.unique(X_norm, axis=0, return_index=True)
            d_norm = d_norm[idx]

        rbf2 = RBFInterpolator(X_norm,d_norm,kernel=self.kernel)
        
        rmesh_norm = self.ref_mesh.points[:,0:2] 
        disp2  = rbf2(rmesh_norm)
    
        def_mesh = deepcopy(self.ref_mesh)
        X_d = def_mesh.points[:,0:2] + disp2
        def_mesh.points[:,0:2] = X_d
        
        # ------------------------------------------------------------------------
        # now apply morph to slab base to get it slab_thickness away from the interface
        # ------------------------------------------------------------------------
        mpts = self.get_boundary_pts(def_mesh, self.tag_dict["slab_interface"])
        bpts = self.get_boundary_pts(def_mesh, self.tag_dict["slab_base"])

        rbf_mpts = RBFInterpolator(np.atleast_2d(mpts[:,0]).T, np.atleast_2d(mpts[:,1]).T, kernel='linear')

        n_new = int(1e2)

        aa = np.min(mpts[:,0])
        bb = np.max(mpts[:,0])

        x_arr = np.linspace(aa,bb,n_new)
        mpts_alt_y = rbf_mpts(np.atleast_2d(x_arr).T)
        mpts_alt = np.hstack([np.atleast_2d(x_arr).T, mpts_alt_y])

        rbf_bpts = RBFInterpolator(np.atleast_2d(bpts[:,0]).T, np.atleast_2d(bpts[:,1]).T, kernel='linear')
        aa = np.min(bpts[:,0])
        bb = np.max(bpts[:,0])
        x_arr = np.linspace(aa,bb,n_new)
        bpts_alt_y = rbf_bpts(np.atleast_2d(x_arr).T)
        bpts_alt = np.hstack([np.atleast_2d(x_arr).T, bpts_alt_y])

        d_r = self.adjust_slab_base(mpts_alt, bpts_alt)

        # keep all other boundaries fixed except slab_base, slab_right, slab_left
        X_zero_disp = self.get_boundary_pts(def_mesh, self.tag_dict["zero_displacement_under_base_adjustment"])
        
        d = np.zeros(X_zero_disp.shape)
        d = np.vstack([d, d_r])
        
        # self.X_adjust = np.vstack([X_zero_disp, bpts])
        self.X_adjust = np.vstack([X_zero_disp, bpts_alt])

        # define displacement field between morphed slab base and corrected slab base
        rbf3 = RBFInterpolator(self.X_adjust,d,kernel="linear")
        disp3  = rbf3(def_mesh.points[:,0:2])

        def_mesh_adjust = deepcopy(def_mesh)
        adj = def_mesh_adjust.points[:,0:2] + disp3
        def_mesh_adjust.points[:,0:2] = adj
        

        # ------------------------------------------------------------------------
        # adjust slab left
        # ------------------------------------------------------------------------

        X_slab_left_in = self.get_boundary_pts(def_mesh_adjust, self.tag_dict["slab_left"])

        d_r_slab_left, X_slab_left = self.adjust_slab_inflow_outflow(X_slab_left_in)

        X_zero_disp = self.get_boundary_pts(def_mesh_adjust, self.tag_dict["zero_displacement_under_slab_left_adjustment"])

        d = np.zeros(X_zero_disp.shape)
        d = np.vstack([d, d_r_slab_left])
        
        self.X_adjust = np.vstack([X_zero_disp, X_slab_left])

        rbf4 = RBFInterpolator(self.X_adjust,d,kernel="linear")
        disp4  = rbf4(def_mesh_adjust.points[:,0:2])

        def_mesh_adjust2 = deepcopy(def_mesh_adjust)
        adj = def_mesh_adjust2.points[:,0:2] + disp4
        def_mesh_adjust2.points[:,0:2] = adj
        
        # ------------------------------------------------------------------------
        # adjust slab right
        # ------------------------------------------------------------------------
        X_slab_right_in = self.get_boundary_pts(def_mesh_adjust2, self.tag_dict["slab_right"])
        d_r_slab_right, X_slab_right = self.adjust_slab_inflow_outflow(X_slab_right_in)
        X_zero_disp = self.get_boundary_pts(def_mesh_adjust2, self.tag_dict["zero_displacement_under_slab_right_adjustment"])
        
        d = np.zeros(X_zero_disp.shape)
        d = np.vstack([d, d_r_slab_right])
        self.X_adjust = np.vstack([X_zero_disp, X_slab_right])

        rbf5 = RBFInterpolator(self.X_adjust,d,kernel="linear")
        disp5  = rbf5(def_mesh_adjust2.points[:,0:2])

        def_mesh_adjust3 = deepcopy(def_mesh_adjust2)
        adj = def_mesh_adjust3.points[:,0:2] + disp5
        def_mesh_adjust3.points[:,0:2] = adj

        # ------------------------------------------------------------------------
        # write to file 
        # ------------------------------------------------------------------------
        os.makedirs(os.path.join(self.out_dir, self.eval_fnb+"_{}".format(theta[0])), exist_ok=True)

        fname_out_vtu = os.path.join(os.path.join(self.out_dir, self.eval_fnb+"_{}".format(theta[0])), self.eval_fnb+"_adjusted3_{}.vtu".format(theta[0]))
        def_mesh_adjust3.write(fname_out_vtu, file_format="vtu")
        
        def_mesh_adjust3.write(os.path.join(os.path.join(self.out_dir, self.eval_fnb+"_{}".format(theta[0])), self.eval_fnb+"_{}.msh".format(theta[0])), \
                               file_format="gmsh22", binary=False)

        return def_mesh_adjust3, fname_out_vtu