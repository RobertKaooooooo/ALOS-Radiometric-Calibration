"""
geolocate.py

Ground-to-radar geolocation for ALOS PALSAR L1.1: given a ground point's
latitude, longitude and ellipsoidal height, find its (azimuth, range)
position in the radar image.

The imaging time t of a ground point G satisfies the Doppler equation

    (P(t) - G) . V(t) / |P(t) - G|  =  -lambda * f_dc(R) / 2

where P, V are the satellite ECR position/velocity and f_dc(R) is the
Doppler centroid the image was processed to. JAXA L1.1 images are NOT
zero-Doppler: f_dc(R) = c0 + c1 * R_km, read from the Leader File
(the two values directly after 'EXTRACTED CHIRP' in the Data Set
Summary Record; R_km is the absolute slant range in km).

Two bugs fixed here (verified against the ASF Hi-Res Terrain Corrected
product for ALPSRP237630880: residual 0 lines azimuth, 0-4 px range):
  1. Orbit interpolation: state vectors are 60 s apart; linear
     interpolation puts the satellite 2.5-3.4 km too low (chord vs arc).
     Now cubic Hermite using both P and V (sub-metre error).
  2. Imaging geometry: zero-Doppler was assumed; the image is focused to
     the Doppler centroid polynomial above (~125 Hz), which shifted
     targets by ~540 azimuth lines (~0.25 s).
With both fixes no empirical time offset or range calibration is needed.
"""

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014
WAVELENGTH_M = 0.2360571   # ALOS PALSAR L-band, from Leader File


def read_doppler_and_spacing(led_path):
    """
    Return (dop_c0, dop_c1, range_pixel_spacing_m) from the Leader File
    Data Set Summary Record. The range pixel spacing is the number
    immediately before 'EXTRACTED CHIRP' (4.6842572 for FBS, 9.3685143 for
    FBD); the Doppler centroid terms are the two numbers immediately after.
    """
    txt = open(led_path, 'rb').read(20000).decode('ascii', 'ignore')
    k = txt.find('EXTRACTED CHIRP')
    spacing = float(txt[k-40:k].split()[-1])
    tok = txt[k+15:k+90].split()
    return float(tok[0]), float(tok[1]), spacing


def llh2ecef(lat_deg, lon_deg, h):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def interpolate_position_velocity_vec(t_arr, t0, interval, positions, velocities):
    """Cubic Hermite interpolation of ECR state vectors (uses P and V)."""
    i = np.clip(((t_arr - t0) / interval).astype(int), 0, len(positions) - 2)
    s = (t_arr - (t0 + i * interval)) / interval
    h = interval
    P = ((2*s**3 - 3*s**2 + 1)[:, None] * positions[i]
         + ((s**3 - 2*s**2 + s) * h)[:, None] * velocities[i]
         + (-2*s**3 + 3*s**2)[:, None] * positions[i+1]
         + ((s**3 - s**2) * h)[:, None] * velocities[i+1])
    V = (((6*s**2 - 6*s) / h)[:, None] * positions[i]
         + (3*s**2 - 4*s + 1)[:, None] * velocities[i]
         + ((-6*s**2 + 6*s) / h)[:, None] * positions[i+1]
         + (3*s**2 - 2*s)[:, None] * velocities[i+1])
    return P, V


def solve_doppler_vec(P_ground, t_lo, t_hi, t0, interval, positions, velocities,
                      dop_c0, dop_c1, wavelength=WAVELENGTH_M, iters=40):
    """Bisection solve of the Doppler-centroid equation for many points."""
    def f(t):
        P, V = interpolate_position_velocity_vec(t, t0, interval, positions, velocities)
        D = P - P_ground
        R = np.linalg.norm(D, axis=1)
        return np.sum(D * V, axis=1) / R + wavelength * (dop_c0 + dop_c1 * R / 1000.0) / 2
    lo = np.full(len(P_ground), t_lo, dtype=np.float64)
    hi = np.full(len(P_ground), t_hi, dtype=np.float64)
    flo = f(lo)
    for _ in range(iters):
        tm = 0.5 * (lo + hi); fm = f(tm); same = (flo * fm) > 0
        lo = np.where(same, tm, lo); flo = np.where(same, fm, flo); hi = np.where(same, hi, tm)
    return 0.5 * (lo + hi)


def ground_to_radar(lat_deg, lon_deg, h, t_first, t_last, n_azimuth,
                    near_range_m, pixel_spacing_m, t0, interval, positions, velocities,
                    dop_c0, dop_c1, wavelength=WAVELENGTH_M):
    """Ground point(s) -> fractional (azimuth_idx, range_idx). No empirical offsets."""
    G = llh2ecef(lat_deg, lon_deg, h)
    t = solve_doppler_vec(G, t_first - 2.0, t_last + 2.0, t0, interval,
                          positions, velocities, dop_c0, dop_c1, wavelength)
    az = (t - t_first) / (t_last - t_first) * (n_azimuth - 1)
    P, _ = interpolate_position_velocity_vec(t, t0, interval, positions, velocities)
    rg = (np.linalg.norm(P - G, axis=1) - near_range_m) / pixel_spacing_m
    return az, rg
