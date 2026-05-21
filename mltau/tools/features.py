import numpy as np


# @numba.njit
def deltaR_etaPhi(eta1, phi1, eta2, phi2):
    """Calculates the angular distance between two objects in the eta-phi coordinates.

    Args:
        eta1 : float
            The eta coordinate of the first object.
        phi1 : float
            The phi coordinate of the first object.
        eta2 : float
            The eta coordinate of the second object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dR : float
            The angular distance between the two objects
    """
    deta = np.abs(eta1 - eta2)
    dphi = deltaPhi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# @numba.njit
def deltaR_thetaPhi(theta1, phi1, theta2, phi2):
    """Calculates the angular distance between two objects in the theta-phi coordinates.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        phi1 : float
            The phi coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dR : float
            The angular distance between the two objects
    """
    dtheta = np.abs(theta1 - theta2)
    dphi = deltaPhi(phi1, phi2)
    return np.sqrt(dtheta**2 + dphi**2)


def angle3d(theta1, phi1, theta2, phi2):
    """Computes the true 3D opening angle between two directions in spherical
    coordinates via the dot product of unit vectors.

    Unlike the Euclidean approximation sqrt(Δθ² + Δφ²) used by deltaR_thetaPhi,
    this correctly accounts for the sin(θ) contraction of phi circles away from
    the equator and is exact for all separation sizes.
    """
    cos_angle = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(
        theta1
    ) * np.cos(theta2)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


# @numba.njit
def deltaPhi(phi1, phi2):
    """Calculates the difference in azimuthal angle of two objects.

    Args:
        phi1 : float
            The phi coordinate of the first object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dPhi : float
            The difference in azimuthal angle
    """
    diff = phi1 - phi2
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def signedDeltaPhi(phi1, phi2):
    """Calculates the signed wrapped azimuthal angle difference."""
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


def signedDeltaPhi_deg(phi1_deg, phi2_deg):
    """Degree-safe phi difference, wrapped to (−180, 180]."""
    diff = phi1_deg - phi2_deg
    return ((diff + 180.0) % 360.0) - 180.0


# @numba.njit()
def deltaTheta(theta1, theta2):
    """Calculates the difference in polar angle of two objects.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.

    Returns:
        dTheta : float
            The difference in polar angle
    """
    return np.abs(theta1 - theta2)


def signedDeltaTheta(theta1, theta2):
    """Calculates the difference in polar angle of two objects.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.

    Returns:
        dTheta : float
            The difference in polar angle
    """
    return theta1 - theta2


# @numba.njit()
def deltaEta(eta1, eta2):
    """Calculates the difference in pseudorapidity of two objects.

    Args:
        eta1 : float
            The eta coordinate of the first object.
        eta2 : float
            The eta coordinate of the second object.

    Returns:
        dEta : float
            The difference in pseudorapidity
    """
    return np.abs(eta1 - eta2)


def signedDeltaEta(eta1, eta2):
    """Calculates the signed pseudorapidity difference."""
    return eta1 - eta2


# @numba.njit()
# def angle3d(theta1, phi1, theta2, phi2):
#     """Calculates the 3D angle between two objects given the theta and phi coordinates for them.

#     Args:
#         theta1 : float
#             The theta coordinate of the first object.
#         phi1 : float
#             The phi coordinate of the first object.
#         theta2 : float
#             The theta coordinate of the second object.
#         phi2 : float
#             The phi coordinate of the second object.

#     Returns:
#         angle : float
#             The 3D angle between the two objects
#     """
#     u_x1 = np.cos(phi1) * np.sin(theta1)
#     u_x2 = np.cos(phi2) * np.sin(theta2)
#     u_y1 = np.sin(phi1) * np.sin(theta1)
#     u_y2 = np.sin(phi2) * np.sin(theta2)
#     u_z1 = np.cos(theta1)
#     u_z2 = np.cos(theta2)
#     cos_angle = u_x1 * u_x2 + u_y1 * u_y2 + u_z1 * u_z2
#     assert cos_angle >= -1.0 and cos_angle <= +1.0
#     angle = np.arccos(cos_angle)
#     return angle
