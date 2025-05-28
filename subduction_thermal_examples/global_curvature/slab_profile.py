
import numpy as np

class em_quadratic_profile:
    """
    Evaluates Eq (1) from England & May (2021).

    `a` should be in the range [5e-4, 3.5e-3] (1/km).
    """

    def __init__(self, a):
        self.a = a

    def zf(self, x):
        """
        Evaluates zf = a x^2.

        Input
        -----
          - x: float.
            x-coordinate (m).
        Output
        ------
          - depth: float / iterable.
            Depth (m).
        """

        depth = self.a * (x/1.0e3)**2
        return depth * 1.0e3

    def x_from_zf(self, y):
        x = ((y/1.0e3)/self.a)**(0.5)
        return x * 1.0e3

    def u(self, x, nsample_points=128):
        """
        Compute the along slab distance.

        Input
        -----
          - x: float / iterable.
            x-coordinate (m).
        """

        u_ = np.zeros(x.shape)
        for k in range(len(x)):
            xi = x[k]
            xs = np.linspace(0.0, xi, nsample_points)
            zfs = self.zf(xs)
            d_ = 0.0
            for i in range(1, nsample_points):
                d_ += np.sqrt( (xs[i]-xs[i-1])**2 + (zfs[i]-zfs[i-1])**2 )
            u_[k] = d_
        return u_

    def delta(self, x):
        """
        Compute the local dip.

        Input
        -----
          - x: float / iterable.
            x-coordinate (m).
        Output
        ------
          - angle: float.
            Local dip (radian).
        """

        dy = 2.0 * self.a * (x/1.0e3)
        angle = np.arctan(dy)
        return angle
