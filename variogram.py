import numpy as np
from scipy.spatial import distance_matrix, distance
from rotation import rotmat, rot_from_matrix


class VariogramModel:
    def __init__(
        self,
        vargstr=None,
        nst=None,
        c0=None,
        it=None,
        cc=None,
        ang1=None,
        ang2=None,
        ang3=None,
        ahmax=None,
        ahmin=None,
        avert=None,
        parsestring=True,
    ):
        self.nst = nst
        self.c0 = c0
        self.it = it
        self.cc = cc
        self.ang1 = ang1
        self.ang2 = ang2
        self.ang3 = ang3
        self.ahmax = ahmax
        self.ahmin = ahmin
        self.avert = avert
        self.vargstr = vargstr
        self.rotmat = None
        self.cmax = None
        if self.vargstr is not None and parsestring:
            self.parsestr()

    def parsestr(self, vargstr=None):
        """
        Parses a GSLIB-style variogram model string

        .. codeauthor:: Jared Deutsch - 2015-02-04
        """
        if vargstr is None:
            vargstr = self.vargstr
        # Split into separate lines
        varglines = vargstr.splitlines()
        # Read nugget effect + nst line
        self.nst = int(varglines[0].split()[0])
        self.c0 = float(varglines[0].split()[1])
        # Read structures
        self.it = []
        self.cc = []
        self.ang1 = []
        self.ang2 = []
        self.ang3 = []
        self.ahmax = []
        self.ahmin = []
        self.avert = []
        for st in range(self.nst):
            self.it.append(int(varglines[1 + st * 2].split()[0]))
            self.cc.append(float(varglines[1 + st * 2].split()[1]))
            self.ang1.append(float(varglines[1 + st * 2].split()[2]))
            self.ang2.append(float(varglines[1 + st * 2].split()[3]))
            self.ang3.append(float(varglines[1 + st * 2].split()[4]))
            self.ahmax.append(float(varglines[2 + st * 2].split()[0]))
            self.ahmin.append(float(varglines[2 + st * 2].split()[1]))
            self.avert.append(float(varglines[2 + st * 2].split()[2]))

        self.cmax = self.c0 + np.sum(self.cc)
        self.setcova()

    def setcova(self):
        # initialize the rotation matrix
        self.rotmat = np.ones((self.nst, 3, 3))

        # determine anisotropies
        self.anis1 = [0.0] * self.nst
        self.anis2 = [0.0] * self.nst

        for st in range(self.nst):
            self.anis1[st] = self.ahmin[st] / self.ahmax[st]
            self.anis2[st] = self.avert[st] / self.ahmax[st]

        # determine the rotation matrix for each structure
        for st in range(self.nst):
            self.rotmat[st, :, :] = rotmat(
                self.ang1[st],
                self.ang2[st],
                self.ang3[st],
                self.anis1[st],
                self.anis2[st],
            )

    def pairwisecova(self, points):

        # nst is the number of nested structures
        # it is a vector of variogram model types
        # cc is a vector of variance contributions where c0 + sum(cc) = sill
        # aa is a vector of ranges for each nested structure
        # rotmat is 3D where the frame index is nst

        assert isinstance(points, np.ndarray), "`points` must be a `np.ndarray`"
        n = points.shape[0]
        cova = np.zeros((n, n))

        for i in range(self.nst):

            # anisotropic distance matrix for current structure
            rot_xyz = rot_from_matrix(
                points[:, 0], points[:, 1], points[:, 2], self.rotmat[i],
            )
            dmat = distance.cdist(rot_xyz, rot_xyz)

            # spherical
            if self.it[i] == 1:
                h = dmat / self.ahmax[i]
                cova = cova + self.cc[i] * np.where(
                    h < 1, (1 - h * (1.5 - 0.5 * h ** 2)), 0
                )

            # exponential
            if self.it[i] == 2:
                h = dmat / self.ahmax[i]
                cova = cova + self.cc[i] * np.exp(-3 * h)

            # Gaussian
            if self.it[i] == 3:
                h = dmat / self.ahmax[i]
                cova = cova + self.cc[i] * np.exp(-3 * h ** 2)

        np.fill_diagonal(cova, self.cmax)

        return cova
