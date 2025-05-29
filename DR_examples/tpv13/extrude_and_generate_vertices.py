import os
import numpy as np
import pickle as pkl
from extrude import Extrude

class Extrude_Fault:
    def __init__(self, fname_tr, fnb, L, h_n_patch, h_fine, h_max, bb, n_patch):
        self.fname_tr = fname_tr
        self.fnb = fnb
        self.L = L
        self.h_n_patch = h_n_patch
        self.h_fine = h_fine
        self.h_max = h_max
        self.bb = bb
        self.n_patch = n_patch

    def write(self,vector,filename):
        file = open(filename, "wb")
        pkl.dump(vector, file)
        file.close()

    def load_trace_pl(self,filename):
        nodes = []
        with open(filename) as fid:
            lines = fid.readlines()
            for li in lines:
                if li.startswith("VRTX"):
                    lli = li.split()
                    nodes.append([float(lli[2]), float(lli[3]), float(lli[4])])
        nodes = np.asarray(nodes)
        return nodes

    def load_trace(self,filename):
        bn = os.path.basename(filename)
        ext = bn.split(".")[1]
        if ext == "pl":
            nodes = self.load_trace_pl(filename)
        else:
            nodes = np.loadtxt(filename)
            ndim = nodes.shape[1]
            if ndim == 2:
                nx = nodes.shape[0]
                b = np.zeros((nx, 1))
                nodes = np.append(nodes, b, axis=1)
        return nodes

    def resample_trace(self,nodes, nx, smoothingParameter):
        from scipy.interpolate import splprep, splev

        nnodes = nodes.shape[0]
        spline_deg = 3 if nnodes > 3 else (2 if nnodes > 2 else 1)
        tck, u = splprep([nodes[:, 0], nodes[:, 1], nodes[:, 2]], s=smoothingParameter, k=spline_deg)
        unew = np.linspace(0, 1, nx)
        new_points = splev(unew, tck)
        nNewNodes = np.shape(new_points[0])[0]
        nodes_ = np.zeros((nNewNodes, 3))
        nodes_[:, 0] = new_points[0]
        nodes_[:, 1] = new_points[1]
        nodes_[:, 2] = new_points[2]
        
        return nodes_
    
    def get_npatch_corners(self,dip_in):
        x_n, z_n, l_n, w_n = self.n_patch
        c_trace = np.array([[x_n - l_n/2, 0, 0],[x_n + l_n/2, 0, 0]])
        extruder = Extrude(c_trace)
        c12, fill = extruder.generate(z_n-(w_n/2), dip=dip_in, sign=-1, N=6)
        c34, fill = extruder.generate(z_n+(w_n/2), dip=dip_in, sign=-1, N=6)
        n_patch_corners = np.vstack([c12,c34])
        # print(n_patch_corners)
        return n_patch_corners

    def write_to_geo(self, fname, trace, extrude_tr, bb, n_patch_corners, h_n_patch, h_fine, h_max):
        # breakpoint()
        # mid = (np.max(trace,axis=0) + np.min(extrude_tr,axis=0))/2.0
        
        # n_patch = np.vstack([np.linspace(trace[15],extrude_tr[15],4)[1:-1],\
        #                     np.linspace(trace[16],extrude_tr[16],4)[1:-1]])
        with open(fname, 'w') as f:
            f.write('SetFactory("OpenCASCADE");'); f.write('\n')
            # write trace
            for k in range(trace.shape[0]):
                line = 'Point(' + str(int(k)) + ') = {' + str(trace[k,0]) + ', ' + str(trace[k,1]) + ', ' + str(trace[k,2]) + ', ' + str(h_fine) + '};'
                f.write(line); f.write('\n')
            # breakpoint()
            spline_line = 'BSpline(1) = ' + str({int(k) for k in np.arange(0,trace.shape[0])}) + ';'
            f.write(spline_line); f.write('\n')

            # write extruded trace
            for k in range(extrude_tr.shape[0]):
                line = 'Point(' + str(int(k+trace.shape[0])) + ') = {' + str(extrude_tr[k,0]) + ', ' + str(extrude_tr[k,1]) + ', ' + str(extrude_tr[k,2]) + ', ' + str(h_fine) + '};'
                f.write(line); f.write('\n')
            # breakpoint()
            # spline_line = 'BSpline(2) = ' + str({*np.arange(trace.shape[0],trace.shape[0]+extrude_tr.shape[0])}) + ';'
            spline_line = 'BSpline(2) = ' + str({int(k) for k in np.arange(trace.shape[0],trace.shape[0]+extrude_tr.shape[0])}) + ';'
            f.write(spline_line); f.write('\n')

            # connect the BSplines
            f.write('Line(3) = {0,' + str(trace.shape[0]) + '};'); f.write('\n')
            f.write('Line(4) = {' + str(trace.shape[0]-1) + ',' + str(trace.shape[0]+extrude_tr.shape[0]-1) + '};'); f.write('\n')

            # add bounding box
            for k in range(bb.shape[0]):
                line = 'Point(' + str(int(trace.shape[0]+extrude_tr.shape[0]+k)) + ') = {' + str(bb[k,0]) + ', ' + str(bb[k,1]) + ', ' + str(bb[k,2]) + ', ' + str(h_max) + '};'
                f.write(line); f.write('\n')

            # lines connecting bounding box
            # numbering of points will need to change
            f.write('Line(5) = {68, 69};'); f.write('\n')
            f.write('Line(6) = {69, 71};'); f.write('\n')
            f.write('Line(7) = {71, 70};'); f.write('\n')
            f.write('Line(8) = {70, 68};'); f.write('\n')
            f.write('Line(9) = {68, 64};'); f.write('\n')
            f.write('Line(10) = {64, 66};'); f.write('\n')
            f.write('Line(11) = {66, 70};'); f.write('\n')
            f.write('Line(12) = {71, 67};'); f.write('\n')
            f.write('Line(13) = {67, 65};'); f.write('\n')
            f.write('Line(14) = {65, 69};'); f.write('\n')
            f.write('Line(15) = {64, 65};'); f.write('\n')
            f.write('Line(16) = {67, 66};'); f.write('\n')

            npts = trace.shape[0]+extrude_tr.shape[0]+bb.shape[0]
            for k in range(n_patch_corners.shape[0]):
                line = 'Point(' + str(int(npts+k)) + ') = {' + str(n_patch_corners[k,0]) + ', ' + str(n_patch_corners[k,1]) + ', ' + str(n_patch_corners[k,2]) + ', ' + str(h_n_patch) + '};'
                f.write(line); f.write('\n')

            f.write('Line(17) = {72, 73};'); f.write('\n')
            f.write('Line(18) = {73, 75};'); f.write('\n')
            f.write('Line(19) = {75, 74};'); f.write('\n')
            f.write('Line(20) = {74, 72};'); f.write('\n')
            
            f.write('Curve Loop(1) = {1,4,-2,-3};'); f.write('\n')    
            f.write('Plane Surface(1) = {1};'); f.write('\n')
            
            f.write('Curve{17} In Surface{1};'); f.write('\n')
            f.write('Curve{18} In Surface{1};'); f.write('\n')
            f.write('Curve{19} In Surface{1};'); f.write('\n')
            f.write('Curve{20} In Surface{1};'); f.write('\n')

            # planes and physical surfaces
            f.write('Curve Loop(21) = {9, 15, 14, -5};'); f.write('\n')
            f.write('Curve Loop(22) = {13, 14, 6, 12};'); f.write('\n')
            f.write('Curve Loop(23) = {16, 11, -7, 12};'); f.write('\n')
            f.write('Curve Loop(24) = {10, 11, 8, 9};'); f.write('\n')
            f.write('Curve Loop(25) = {7, 8, 5, 6};'); f.write('\n')
            f.write('Curve Loop(26) = {10, -16, 13, -15};'); f.write('\n')
            f.write('Plane Surface(3) = {21};'); f.write('\n')
            f.write('Plane Surface(4) = {22};'); f.write('\n')
            f.write('Plane Surface(5) = {23};'); f.write('\n')
            f.write('Plane Surface(6) = {24};'); f.write('\n')
            f.write('Plane Surface(7) = {25};'); f.write('\n')
            f.write('Plane Surface(8) = {26};'); f.write('\n')
            
            f.write('Physical Surface(1) = {2};'); f.write('\n') # free surface
            f.write('Physical Surface("fault", 3) = {1};'); f.write('\n')
            f.write('Physical Surface(5) = {3,4,5,6,7};'); f.write('\n')
            
            f.write('Physical Point(24) = {72};'); f.write('\n')
            f.write('Physical Point(25) = {73};'); f.write('\n')
            f.write('Physical Point(26) = {74};'); f.write('\n')
            f.write('Physical Point(27) = {75};'); f.write('\n')

            f.write('Physical Curve(92) = {17};'); f.write('\n')
            f.write('Physical Curve(93) = {18};'); f.write('\n')
            f.write('Physical Curve(94) = {19};'); f.write('\n')
            f.write('Physical Curve(95) = {20};'); f.write('\n')
            
            f.write('Surface Loop(1) = {8, 6, 5, 7, 3, 4};'); f.write('\n')
            f.write('Volume(1) = {1};'); f.write('\n')
            f.write('Physical Volume(23) = {1};'); f.write('\n')
            f.write('v() = BooleanFragments{ Volume{1}; Delete; }{ Surface{1}; Delete; };'); f.write('\n')
            f.write('MeshSize{ PointsOf{Volume{1};} } = ' + str(h_max) + ';'); f.write('\n')
            f.write('MeshSize{ PointsOf{Surface{1};} } = ' + str(h_fine) + ';'); f.write('\n')
            f.write('Mesh.MshFileVersion = 2.2;'); f.write('\n')

            return

    def constant_dip(self,dip_in,write_geo=False,sign=-1):
        coord = self.load_trace(self.fname_tr)
        
        # resample
        nx = 32
        coord_s = self.resample_trace(coord, nx, 10.0)

        extruder = Extrude(coord_s)
        # v, fill = extruder.generate(0.6, dip=dip_in, sign=1, N=6)
        v, fill = extruder.generate(self.L, dip=dip_in, sign=sign, N=6)

        # plt.figure()
        # fig = plt.figure(1, (5, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(coord_s[:,0], coord_s[:,1], coord_s[:,2], c='r', marker='o')
        # ax.scatter(v[:,0], v[:,1], v[:,2], c='b', marker='o')
        # ax.scatter(fill[:,0], fill[:,1], fill[:,2], c='g', marker='o')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z');
        # ax.set_xlim([-4.0,4.0])
        # ax.set_ylim([-4.0,4.0])
        # ax.set_zlim([-2.0,0.0])
        # plt.show()

        # bb = np.array([[-4.0, -4.0, 0.0],[4.0, -4.0, 0.0],[-4.0, 4.0, 0.0],[4.0, 4.0, 0.0],\
        #                [-4.0, -4.0, -2.0],[4.0, -4.0, -2.0],[-4.0, 4.0, -2.0],[4.0, 4.0, -2.0]])
        
        self.write(fill, self.fnb+"_{}.pkl".format(dip_in))

        npc = None
        if write_geo:
            fname_geo = self.fnb+"_{}.geo".format(dip_in)
            n_patch_corners = self.get_npatch_corners(dip_in)
            self.write_to_geo(fname_geo, coord_s, v, self.bb, n_patch_corners, self.h_n_patch, self.h_fine, self.h_max)
            self.n_patch_corners = n_patch_corners
            npc = n_patch_corners
        return npc