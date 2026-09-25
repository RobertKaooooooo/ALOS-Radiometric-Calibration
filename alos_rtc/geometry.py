"""
geometry.py

Core radiometric-terrain-correction geometry for ALOS PALSAR, ported
from the UAVSAR method of Simard, M.; Riel, B. V.; Denbina, M.;
Hensley, S. (2016), "Radiometric Correction of Airborne Radar Images
Over Forested Terrain With Topography," IEEE TGRS, 54(8), 4488-4500,
and the accompanying `uavsar_calib.cpp`
(github.com/simard-landscape-lab/UAVSAR-Radiometric-Calibration).

Where the underlying physical formula is identical to UAVSAR's (the
look-vector components l_s, l_c, l_h of Eq. 11, and the terrain-normal
construction from Horn's method), this module follows the same
equations but substitutes ALOS-specific inputs (P, V from the Platform
Position Data Record; theta_ih from the six-coefficient polynomial
instead of UAVSAR's triangle construction) in place of UAVSAR's
peg-geometry construction. This translation is the author's own code,
verified against the original C++ output (see docs/verification.md).

ELECTRONIC STEERING ANGLE (ESA) NOTE:
Eq. 11 is a single shared formula used by both UAVSAR and ALOS. ESA is
a required parameter of that one formula for both sensors -- the only
difference is where its value comes from. UAVSAR reads a non-zero
angle from the ann file. For ALOS, the value substituted is 0, because
this sensor uses satellite yaw steering (confirmed via the Leader
File's Yaw Steering Mode Flag, see leader_file.read_esa_fields) to
achieve the same zero-Doppler compensation that UAVSAR achieves with
antenna electronic steering -- so the compensation is already captured
in the yaw angle read from the Attitude Data Record, not silently
dropped from the formula.
"""

import numpy as np


def compute_theta_ih(coefficients, slant_range_km):
    """
    Pure geometric incidence angle (no terrain), evaluated from the
    six-coefficient polynomial in the Data Set Summary Record.

    theta_ih = a0 + a1*R + a2*R^2 + a3*R^3 + a4*R^4 + a5*R^5   (radians)

    Parameters
    ----------
    coefficients : sequence of 6 floats (a0..a5), from
        leader_file.read_six_coefficients()
    slant_range_km : float or ndarray, slant range in km

    Returns
    -------
    theta_ih_rad : same shape as slant_range_km
    """
    a0, a1, a2, a3, a4, a5 = coefficients
    R = slant_range_km
    return a0 + a1 * R + a2 * R**2 + a3 * R**3 + a4 * R**4 + a5 * R**5


def slant_range_km(near_range_m, pixel_spacing_m, range_index):
    """
    Slant range (km) at a given range-direction pixel index, built
    from the near-range distance and per-pixel spacing (both read from
    the Leader/IMG files).
    """
    return (near_range_m + range_index * pixel_spacing_m) / 1000.0


def compute_nE_horn(Z, dlon_m, dlat_m, heading_rad):
    """
    Ground-surface unit normal vector via Horn's method, identical in
    form to the UAVSAR implementation in uavsar_calib.cpp.

    Parameters
    ----------
    Z : (3,3) array-like
        3x3 DEM elevation window, indexed Z[row][col] with row 0 = north:
            Z1 Z2 Z3
            Z4 Z5 Z6
            Z7 Z8 Z9
        (Z5 is the center pixel; only the 8 neighbors are used here.)
    dlon_m, dlat_m : float
        Local ground spacing (meters) between adjacent DEM columns/rows.
    heading_rad : float
        Flight heading, radians. For ALOS this is the satellite
        heading derived from the velocity vector V (see main.py); for
        UAVSAR it is peg.heading from the ann file.

    Returns
    -------
    nE : (3,) ndarray, unit normal vector (local along-track / cross-
         track / vertical frame)
    """
    Z1, Z2, Z3 = Z[0]
    Z4, Z5, Z6 = Z[1]
    Z7, Z8, Z9 = Z[2]

    p = (Z3 + Z6 + Z9 - Z1 - Z4 - Z7) / (6.0 * dlon_m)
    q = (Z1 + Z2 + Z3 - Z7 - Z8 - Z9) / (6.0 * dlat_m)

    slope = np.arctan(np.sqrt(p**2 + q**2))
    # Matches uavsar_calib.cpp exactly: atan(q/p), NOT arctan2 (arctan2 adds
    # +/-pi when p<0, flipping the aspect of every west-rising facet).
    if p == 0:
        aspect = np.pi if q > 0 else 0.0
    else:
        aspect = np.pi - np.arctan(q / p) + (np.pi / 2) * np.sign(p)

    slope_r = np.tan(slope) * np.cos(aspect - heading_rad - np.pi / 2)
    slope_a = np.tan(slope) * np.cos(aspect - heading_rad)

    temp = -1.0 / np.sqrt(1.0 + slope_r**2 + slope_a**2)
    nEx = temp * slope_a
    nEy = temp * slope_r
    nEz = -temp
    return np.array([nEx, nEy, nEz])


def compute_nL_local_frame(theta_ih_rad, pitch_rad, yaw_rad, esa_rad=0.0):
    """
    Look-vector components in the SAME local (along-track / cross-track
    / vertical) frame used by compute_nE_horn -- NOT rotated into ECEF.

    This is Simard et al. 2016 Eq. 11, with the sign convention already
    verified (l_h negated) so that the result can be dotted directly
    against nE from compute_nE_horn without any coordinate-frame
    conversion.

    IMPORTANT: keep nL in this local frame. An earlier version of this
    pipeline rotated nL into ECEF (using unit vectors built from P and
    V) while leaving nE in the local frame -- dotting the two together
    silently mixed two different coordinate bases and produced a
    systematic 8-15 deg error even at genuinely flat terrain (see
    docs/verification.md, Section "ECEF/local-frame bug"). Do not
    reintroduce that conversion.

    Returns
    -------
    nL_local : (3,) ndarray = (l_s, l_c, -l_h)
    """
    l_s = (np.sin(esa_rad) * np.cos(pitch_rad) * np.cos(yaw_rad)
           + np.cos(esa_rad) * (np.sin(pitch_rad) * np.cos(theta_ih_rad) * np.cos(yaw_rad)
                                 + np.sin(theta_ih_rad) * np.sin(yaw_rad)))
    l_c = (-np.sin(esa_rad) * np.cos(pitch_rad) * np.sin(yaw_rad)
           + np.cos(esa_rad) * (-np.sin(pitch_rad) * np.cos(theta_ih_rad) * np.sin(yaw_rad)
                                 + np.sin(theta_ih_rad) * np.cos(yaw_rad)))
    l_h = (np.sin(esa_rad) * np.sin(pitch_rad)
           - np.cos(esa_rad) * np.cos(pitch_rad) * np.cos(theta_ih_rad))
    return np.array([l_s, l_c, -l_h])


def compute_theta_l(nE, nL_local):
    """
    Local incidence angle (radians), including terrain effect.

        theta_l = arccos(nE . nL_local)

    Both vectors must be in the same local frame (see
    compute_nL_local_frame docstring).
    """
    dotv = np.clip(np.dot(nE, nL_local), -1.0, 1.0)
    return np.arccos(dotv)
