import numpy as np
import matplotlib.pyplot as plt
import meshio
import pickle as pkl
import pyvista as pv
from copy import deepcopy
from scipy.interpolate import RBFInterpolator
# from RBFScaled import RBFScaled

class Mesh_Morph:
    def __init__(self,ref_fnb,train_fnb,eval_fnb,ref_msh,tags,kernel):
        self.ref_fnb = ref_fnb
        self.train_fnb = train_fnb
        self.eval_fnb = eval_fnb
        print('starting meshio.read(ref_msh)')
        self.ref_mesh = meshio.read(ref_msh)
        print('done reading mesh')
        # print('starting ref_mesh.write')
        # self.ref_mesh.write(ref_fnb+".vtk")
        self.tags = tags
        self.kernel = kernel
        self.ref_bnd_pts, idx = self.get_boundary_pts(self.ref_mesh,tags,2)

    def load(self,filename):
        file = open(filename, "rb")
        vector = pkl.load(file)
        file.close()
        return vector

    def get_boundary_pts(self,mesh,tags,level):
        # get boundary points
        triangle_cells = mesh.cells_dict["triangle"]
        bp = []
        idx = []
        for tag in tags:
            cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][level]==tag)]
            # cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][0]==tag)]
            pts = mesh.points[np.unique(cells.ravel())]
            idx_arr = np.unique(cells.ravel())
            for k in range(pts.shape[0]):
                bp.append(pts[k,:])
                idx.append(idx_arr[k])
        boundary_pts = np.unique(np.array(bp),axis=0)
        return boundary_pts, idx

    def get_n_patch_edges(self,mesh,tags,level):
        # get boundary points
        triangle_cells = mesh.cells_dict["line"]
        bp = []
        idx = []
        for tag in tags:
            cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][level]==tag)]
            # cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][0]==tag)]
            pts = mesh.points[np.unique(cells.ravel())]
            idx_arr = np.unique(cells.ravel())
            for k in range(pts.shape[0]):
                bp.append(pts[k,:])
                idx.append(idx_arr[k])
        boundary_pts = np.unique(np.array(bp),axis=0)
        return boundary_pts
    
    # def get_n_patch_idx(self,mesh,tags):
    #     triangle_cells = mesh.cells_dict["triangle"]
    #     np_idx = []
    #     for tag in tags:
    #         cells = triangle_cells[np.where(mesh.cell_data["gmsh:physical"][0]==tag)]
    #         pts = mesh.points[np.unique(cells.ravel())]
    #         for k in range(pts.shape[0]):
    #             np.append(pts[k,:])
    #     n_patch = np.unique(np.array(np),axis=0)
    #     return n_patch


    def compute_displacement(self,f1,f2,bnd):
        # df = np.sqrt(np.sum((f1-f2)**2,axis=1))
        df = f2-f1
        d = np.vstack([df,np.zeros(bnd.shape)])
        return d

    def build_MM_RBF(self,dip_ref,dip_train,ref_fault_fn):
        all_dips = np.atleast_2d(np.hstack([np.array(dip_ref),dip_train]))
        all_faults = [self.ref_fnb+"_{}.pkl".format(dip_ref)]
        for k in dip_train:
            all_faults.append(self.train_fnb+"_{}.pkl".format(k))
        self.ref_fault = self.load(ref_fault_fn)
        D = []
        for fname in all_faults:
            fault = self.load(fname)
            dd = self.compute_displacement(self.ref_fault,fault,self.ref_bnd_pts)
            # print('dd',dd)
            # plotx = np.vstack([fault,self.ref_bnd_pts])
            # y_min = np.min(dd); y_max = np.max(dd)
            # dd_c = (dd - y_min)/(y_max - y_min)
            # plt.figure()
            # fig = plt.figure(1, (5, 5))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(plotx[:,0], plotx[:,1], plotx[:,2], c=dd_c, marker='.')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z');
            # ax.set_xlim([-4.0,4.0])
            # ax.set_ylim([-4.0,4.0])
            # ax.set_zlim([-2.0,0.0])
            # plt.show()

            self.ndd = dd.shape
            D.append(dd.ravel())
            # nc = rbf._coeffs.shape
            # C.append(rbf._coeffs)
            # print(rbf._coeffs.ravel().shape)
        DD = np.array(D)
        self.rbf1 = RBFInterpolator(all_dips.T,DD,kernel=self.kernel)
        # self.rbf1 = RBFScaled(all_dips.T,DD,kernel="linear",\
        #                       normalize_y=(False,None,None))

    def eval_RBF(self,theta):

        
        print('evaluate rbf1')
        disp = self.rbf1(np.array(theta,ndmin=2))
        print('reshape displacement')
        disp = disp.reshape(self.ndd)
        print('vstack')
        X = np.vstack([self.ref_fault,self.ref_bnd_pts])

        
        print('create rbf2')
        rbf2 = RBFInterpolator(X,disp,kernel=self.kernel)# ,neighbors=100)
        
        print('evaluate rbf2')
        disp2  = rbf2(self.ref_mesh.points)
        
        print('deepcopy')
        def_mesh = deepcopy(self.ref_mesh)
        print('add displacements')

        # add displacement to internal points only
        def_bnd_pts, idx = self.get_boundary_pts(def_mesh,self.tags,2)
        X_d = def_mesh.points
        mask = np.ones(X_d.shape[0], dtype=bool)
        mask[idx] = False
        X_d[mask] += disp2[mask]
        def_mesh.points = X_d

        # ---------------------------------------------
        # enforce planarity
        # ---------------------------------------------
        fault_pts, f_idx = self.get_boundary_pts(def_mesh,[3],2)
        fmax = np.max(fault_pts,axis=0)
        fmin = np.min(fault_pts,axis=0)
        fault_corners = np.vstack([fmax, fmin, np.hstack([fmax[0], fmin[1:]]), np.hstack([fmin[0], fmax[1:]])])
        plane_ex, center_ex, normal_ex = pv.fit_plane_to_points(fault_corners, return_meta=True)
        def project_points_to_plane(points, plane_origin, plane_normal):
            """
            Project points to a plane.
            From pyvista doc: https://docs.pyvista.org/examples/98-common/project-points-tessellate.html 
            """
            vec = points - plane_origin
            dist = np.dot(vec, plane_normal)
            return points - np.outer(dist, plane_normal)

        projected_points = project_points_to_plane(fault_pts, center_ex, normal_ex)
        projected_points[np.abs(projected_points[:,2]) < 1, 1] = 0.0
        projected_points[np.abs(projected_points[:,2]) < 1, 2] = 0.0

        # only use non-surface points
        mask_f = np.ones(projected_points.shape[0], dtype=bool)
        mask_f[np.abs(projected_points[:,2]) < 1] = False
        

        bnd_planarity = self.get_boundary_pts(def_mesh, self.tags, 2)[0]
        disp_planarity = np.vstack([projected_points[mask_f] - fault_pts[mask_f], np.zeros(bnd_planarity.shape)])
        X_planarity = np.vstack([fault_pts[mask_f], bnd_planarity])
        rbf4 = RBFInterpolator(X_planarity,disp_planarity,kernel=self.kernel)
        
        disp4  = rbf4(def_mesh.points)

        X_d_new = deepcopy(X_d)
        X_d_new[mask] += disp4[mask]
        def_mesh.points = X_d_new
        
        # ----------------------------------------------------------
        print('write vtk')
        def_mesh.write(self.eval_fnb+"_{}.vtk".format(theta))
        print('write msh')
        def_mesh.write(self.eval_fnb+"_{}.msh".format(theta),file_format="gmsh22",binary=False)
    
        n_patch = def_mesh.points[def_mesh.cells_dict["vertex"]]
        print('Corrected n_patch', n_patch)
        limits = [np.min(def_mesh.points[:,1]),np.max(def_mesh.points[:,1])] # assume x y z
        return n_patch,limits