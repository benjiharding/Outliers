import numpy as np


def rotmat(ang1, ang2, ang3, anis1, anis2):
    """After GSLIB subroutine 'setrot' by CV Deutsch 1992"""

    pi = np.pi
    epsillon = 1e-10
    R = np.eye(3)

    if (ang1 > 0) & (ang1 < 270):
        alpha = (90 - ang1) * pi / 180
    else:
        alpha = (450 - ang1) * pi / 180

    beta = -1 * ang2 * pi / 180
    theta = ang3 * pi / 180

    sina = np.sin(alpha)
    sinb = np.sin(beta)
    sint = np.sin(theta)
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cost = np.cos(theta)

    afac1 = 1 / max(anis1, epsillon)
    afac2 = 1 / max(anis2, epsillon)

    R[0, 0] = cosb * cosa
    R[1, 0] = cosb * sina
    R[2, 0] = -sinb
    R[0, 1] = afac1 * (-cost * sina + sint * sinb * cosa)
    R[1, 1] = afac1 * (cost * cosa + sint * sinb * sina)
    R[2, 1] = afac1 * (sint * cosb)
    R[0, 2] = afac2 * (sint * sina + cost * sinb * cosa)
    R[1, 2] = afac2 * (-sint * cosa + cost * sinb * sina)
    R[2, 2] = afac2 * (cost * cosb)

    return R


def rot_from_matrix(x, y, z=None, rotmat=None, origin=(0, 0, 0)):
    """3D rotation from existing rotation matrix"""

    if z is None:
        z = np.zeros(len(x))

    ox = origin[0]
    oy = origin[1]
    oz = origin[2]

    ax = (x - ox).reshape(-1, 1)
    ay = (y - oy).reshape(-1, 1)
    az = (z - oz).reshape(-1, 1)

    adj_xyz = np.hstack((ax, ay, az))

    rot_xyz = adj_xyz @ rotmat

    rot_xyz[:, 0] = rot_xyz[:, 0] + ox
    rot_xyz[:, 1] = rot_xyz[:, 1] + oy
    rot_xyz[:, 2] = rot_xyz[:, 2] + oz

    return rot_xyz
