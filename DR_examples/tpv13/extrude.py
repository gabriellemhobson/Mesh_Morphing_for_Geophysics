
import numpy as np

class Extrude:
  """
  Supports constant along strike dip and variable.
  Supports constant along strike length and variable.
  """
  def __init__(self, trace, **kwargs):
    self.trace = trace

  def generate(self, dL, dip=90.0, sign=1, N=6):
    dip = np.deg2rad(dip)
    depth_vector = np.array([0, 0, -1])
    nx = self.trace.shape[0]
    av0 = np.zeros((nx, 3))
    av1 = np.zeros((nx, 3))
    vertices = np.zeros(self.trace.shape)
    for i in range(nx):
      v0 = self.trace[min(i + 1, nx-1), :] - self.trace[max(i-1,0), :]
      v0[2] = 0
      v0 = v0 / np.linalg.norm(v0)
      av0[i, :] = v0[:]
      av1[i, :] = np.array([-v0[1], v0[0], 0])

      # Convert scalar values for dip, dL into a vector
      _dip = np.zeros(nx)
      _dip[:] = dip
      _dL = np.zeros(nx)
      _dL[:] = dL 
      one_over_tan_dip = 1.0 / np.tan(_dip)
      for i in range(nx):
        ud = -(one_over_tan_dip[i] * av1[i, :] + depth_vector)
        ud = ud / np.linalg.norm(ud)
        vertices[i, :] = self.trace[i, :] - sign * _dL[i] * ud
        # vertices[i, 0] = self.trace[i, 0] - 1 * _dL[i] * ud[0]
        vertices[i, 1] = self.trace[i, 1] - sign * _dL[i] * ud[1]
        vertices[i, 2] = self.trace[i, 2] - 1 * _dL[i] * ud[2]

    fill = np.linspace(self.trace[0,:],vertices[0,:],N)[1:]
    for k in range(1,self.trace.shape[0]):
      fill = np.vstack([fill,np.linspace(self.trace[k,:],vertices[k,:],N)[1:,:]])


    return vertices,fill