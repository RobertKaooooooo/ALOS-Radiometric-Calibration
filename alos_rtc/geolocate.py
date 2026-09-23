"""
geolocate.py

Ground-to-radar geolocation for ALOS PALSAR: given a ground point's
latitude, longitude and height, finds the corresponding (azimuth,
range) position in the radar image.

This solves the standard SAR "range-Doppler" intersection problem:
for a ground point P, find the satellite time t at which the
zero-Doppler condition holds, i.e. (P_sat(t) - P) . V_sat(t) = 0
(the satellite velocity is perpendicular to the look direction at the
moment of imaging, per the standard broadside/zero-Doppler processing
assumption used for ALOS PALSAR L1.1 products). The satellite
position/velocity at time t are interpolated from the Platform
Position Data Record via leader_file.read_platform_position_record().

TIMING OFFSET NOTE:
The solved time t does not directly equal the image line time-stamp
(img_file.read_line_time_ms()) -- there is a small, consistent offset
between the Platform Position Data Record's time reference and the
Signal Data Record's per-line time-stamp reference. This offset was
empirically calibrated against four known ground-truth corner
coordinates (see docs/verification.md) at approximately 0.1295
seconds, and confirmed to transfer correctly across different scenes
processed by the same JAXA ADEN pipeline. It has not been traced to
an explicit documented field in the CEOS format; callers should treat
GEOLOCATION_TIME_OFFSET_S as an empirical constant, not a derived one.

Original contribution: all functions in this module (the zero-Doppler
solve, its vectorization, and the empirical timing-offset calibration)
were implemented and verified by the author.
"""

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

# Empirically calibrated offset between PPDR time reference and SDR
# line-time reference (seconds). See module docstring.
GEOLOCATION_TIME_OFFSET_S = 0.1295


def llh2ecef(lat_deg, lon_deg, h):
    """
    Convert geodetic latitude/longitude/height (WGS84) to ECEF
    Cartesian coordinates. Vectorized: lat_deg, lon_deg, h may be
    scalars or same-shaped ndarrays.

    Returns
    -------
    (..., 3) ndarray of ECEF (x, y, z) in meters.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def interpolate_position_velocity_vec(t_arr, t0, interval, positions, velocities):
    """
    Vectorized version of leader_file.interpolate_position_velocity():
    linearly interpolates satellite position/velocity to an array of
    times at once, using pre-loaded PPDR arrays (avoids re-reading the
    Leader File from disk on every call, which is required for
    acceptable performance at DEM-facet scale).

    Parameters
    ----------
    t_arr : (N,) ndarray, seconds of day
    t0, interval, positions, velocities : as returned by
        leader_file.read_platform_position_record()

    Returns
    -------
    P, V : (N,3) ndarrays, ECEF position (m) and velocity (m/s)
    """
    idx = ((t_arr - t0) / interval).astype(int)
    idx = np.clip(idx, 0, len(positions) - 2)
    frac = (t_arr - (t0 + idx * interval)) / interval
    P = positions[idx] * (1 - frac)[:, None] + positions[idx + 1] * frac[:, None]
    V = velocities[idx] * (1 - frac)[:, None] + velocities[idx + 1] * frac[:, None]
    return P, V


def solve_zero_doppler_vec(P_ground, t_lo, t_hi, t0, interval, positions, velocities, iters=40):
    """
    Vectorized bisection solve of the zero-Doppler condition
    (P_sat(t) - P_ground) . V_sat(t) = 0 for an array of ground
    points simultaneously.

    Parameters
    ----------
    P_ground : (N,3) ndarray, ECEF ground points
    t_lo, t_hi : float, search bracket (seconds of day); must bracket
        a sign change of the Doppler condition for the solve to
        converge (in practice, [line_time[0]-1, line_time[-1]+1] is a
        safe bracket for points within the scene footprint)
    t0, interval, positions, velocities : PPDR arrays, as for
        interpolate_position_velocity_vec()
    iters : int, bisection iterations (40 gives sub-microsecond
        convergence given the ~8.5s scene duration)

    Returns
    -------
    (N,) ndarray, solved time (seconds of day) for each ground point
    """
    t_lo_arr = np.full(len(P_ground), t_lo, dtype=np.float64)
    t_hi_arr = np.full(len(P_ground), t_hi, dtype=np.float64)

    def f(t):
        P, V = interpolate_position_velocity_vec(t, t0, interval, positions, velocities)
        return np.sum((P - P_ground) * V, axis=1)

    f_lo = f(t_lo_arr)
    for _ in range(iters):
        t_mid = 0.5 * (t_lo_arr + t_hi_arr)
        f_mid = f(t_mid)
        same_sign = (f_lo * f_mid) > 0
        t_lo_arr = np.where(same_sign, t_mid, t_lo_arr)
        f_lo = np.where(same_sign, f_mid, f_lo)
        t_hi_arr = np.where(same_sign, t_hi_arr, t_mid)
    return 0.5 * (t_lo_arr + t_hi_arr)


def ground_to_radar(lat_deg, lon_deg, h, t_first, t_last, n_azimuth,
                     near_range_m, pixel_spacing_m, t0, interval,
                     positions, velocities, time_offset_s=GEOLOCATION_TIME_OFFSET_S):
    """
    Full ground-point -> (azimuth_index, range_index) solve, combining
    the zero-Doppler time solve with the empirical timing-offset
    correction and the range-to-pixel conversion.

    Parameters
    ----------
    lat_deg, lon_deg, h : (N,) ndarrays, ground point coordinates
    t_first, t_last : float, line time-stamps of the first and last
        azimuth lines (from img_file.read_line_time_ms())
    n_azimuth : int, number of azimuth lines in the scene
    near_range_m, pixel_spacing_m : float, from
        img_file.read_near_range_m() and the known system range
        pixel spacing (4.6842572 m for this ALOS-1 PALSAR FBD mode)
    t0, interval, positions, velocities : PPDR arrays
    time_offset_s : float, empirical PPDR/SDR timing offset (see
        module docstring); defaults to the calibrated value

    Returns
    -------
    azimuth_idx, range_idx : (N,) ndarrays, fractional pixel
        coordinates in the radar image (not yet bounds-checked --
        caller should filter to [0, n_azimuth-1] / [0, n_range-1])
    """
    P_ground = llh2ecef(lat_deg, lon_deg, h)
    t_sol = solve_zero_doppler_vec(P_ground, t_first - 1.0, t_last + 1.0,
                                    t0, interval, positions, velocities)
    t_corr = t_sol - time_offset_s
    azimuth_idx = (t_corr - t_first) / (t_last - t_first) * (n_azimuth - 1)
    P_sat, V_sat = interpolate_position_velocity_vec(t_corr, t0, interval, positions, velocities)
    slant_range_m = np.linalg.norm(P_sat - P_ground, axis=1)
    range_idx = (slant_range_m - near_range_m) / pixel_spacing_m
    return azimuth_idx, range_idx
