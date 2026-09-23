"""
area_correction.py

Area-projection radiometric terrain correction for ALOS PALSAR,
mirroring the DEM-facet-driven method in Simard et al. 2016 and the
original uavsar_calib.cpp (area = area_ref / |nE . nI|, accumulated
onto the radar-coordinate grid via inverse-distance weighting to the
four nearest pixels of each facet's fractional (azimuth, range)
position).

This module assumes the ground-to-radar mapping for each DEM facet
(azimuth_idx, range_idx) has already been computed, e.g. via
geolocate.ground_to_radar() applied across the DEM grid.

geometry.compute_nE_horn() and geometry.compute_nL_local_frame() are
per-point functions; this module vectorizes the same formulas across
many DEM facets at once for performance (identical math, not a
different method).

Original contribution: the vectorized facet-area computation and the
IDW accumulation/gap-filling in this module are the author's own
code, built to mirror uavsar_calib.cpp's DEM-facet area-projection
logic (see docs/verification.md).
"""

import numpy as np
from scipy import ndimage

DEG_M = 111320.0  # meters per degree of latitude (approx, WGS84)


def compute_facet_area_and_theta_l(dem, rows, cols, heading_rad,
                                    pixels_per_deg, lat_top,
                                    theta_ih, pitch_rad, yaw_rad, esa_rad=0.0):
    """
    Vectorized computation of local incidence angle and terrain-facet
    area for a set of DEM grid points (rows, cols), following the
    same formulas as geometry.compute_nE_horn() /
    compute_nL_local_frame() / compute_theta_l().

    Parameters
    ----------
    dem : (H, W) ndarray, full DEM elevation grid (meters)
    rows, cols : (N,) int ndarrays, DEM grid indices of each facet
        (must not include the DEM's outermost border row/column,
        since a 3x3 window is needed)
    heading_rad : float, flight heading (radians) -- a single
        scene-average value, consistent with how UAVSAR's own
        peg.heading is used as one constant for the whole scene
    pixels_per_deg, lat_top : DEM grid geolocation parameters (e.g.
        3600, 45.0 for a 1-arcsecond GLO-30 tile)
    theta_ih : (N,) ndarray, flat-earth incidence angle (radians) at
        each facet, from geometry.compute_theta_ih()
    pitch_rad, yaw_rad : (N,) ndarrays, interpolated attitude at each
        facet's solved acquisition time
    esa_rad : float, electronic steering angle (0.0 for ALOS, which
        uses yaw steering instead -- see geometry.py docstring)

    Returns
    -------
    theta_l_deg : (N,) ndarray, local incidence angle (degrees)
    area : (N,) ndarray, terrain-facet ground area (m^2)
    """
    lat_rows = (lat_top - rows / pixels_per_deg).astype(np.float32)
    dlat_m = np.float32(DEG_M / pixels_per_deg)
    dlon_m = (DEG_M / pixels_per_deg * np.cos(np.radians(lat_rows))).astype(np.float32)

    Z1 = dem[rows - 1, cols - 1]; Z2 = dem[rows - 1, cols]; Z3 = dem[rows - 1, cols + 1]
    Z4 = dem[rows,     cols - 1];                            Z6 = dem[rows,     cols + 1]
    Z7 = dem[rows + 1, cols - 1]; Z8 = dem[rows + 1, cols]; Z9 = dem[rows + 1, cols + 1]

    p = (Z3 + Z6 + Z9 - Z1 - Z4 - Z7) / (6.0 * dlon_m)
    q = (Z1 + Z2 + Z3 - Z7 - Z8 - Z9) / (6.0 * dlat_m)
    slope = np.arctan(np.sqrt(p ** 2 + q ** 2))
    aspect = np.where(p == 0, np.where(q > 0, np.pi / 2, -np.pi / 2),
                       np.pi - np.arctan2(q, p) + (np.pi / 2) * np.sign(p))
    slope_r = np.tan(slope) * np.cos(aspect - heading_rad - np.pi / 2)
    slope_a = np.tan(slope) * np.cos(aspect - heading_rad)
    tempv = -1.0 / np.sqrt(1.0 + slope_r ** 2 + slope_a ** 2)
    nEx = tempv * slope_a
    nEy = tempv * slope_r
    nEz = -tempv

    l_s = (np.sin(esa_rad) * np.cos(pitch_rad) * np.cos(yaw_rad)
           + np.cos(esa_rad) * (np.sin(pitch_rad) * np.cos(theta_ih) * np.cos(yaw_rad)
                                 + np.sin(theta_ih) * np.sin(yaw_rad)))
    l_c = (-np.sin(esa_rad) * np.cos(pitch_rad) * np.sin(yaw_rad)
           + np.cos(esa_rad) * (-np.sin(pitch_rad) * np.cos(theta_ih) * np.sin(yaw_rad)
                                 + np.sin(theta_ih) * np.cos(yaw_rad)))
    l_h = (np.sin(esa_rad) * np.sin(pitch_rad)
           - np.cos(esa_rad) * np.cos(pitch_rad) * np.cos(theta_ih))
    nLx, nLy, nLz = l_s, l_c, -l_h

    theta_l_deg = np.degrees(np.arccos(np.clip(nEx * nLx + nEy * nLy + nEz * nLz, -1, 1)))

    # Imaging-plane normal, per uavsar_calib.cpp convention:
    # nI = (0, nL.z, -nL.y) in this local frame, i.e. (0, -l_h, -l_c)
    nIy = -l_h
    nIz = -l_c
    denom = np.abs(nEy * nIy + nEz * nIz)
    area_ref = dlon_m * dlat_m
    area = area_ref / np.maximum(denom, 1e-6)

    return theta_l_deg, area


def accumulate_area_rdc(area, azimuth_idx, range_idx, n_azimuth, n_range,
                         outlier_factor=20.0):
    """
    Inverse-distance-weighted accumulation of per-facet terrain area
    onto the radar-coordinate (azimuth, range) grid, distributing each
    facet's area to its four surrounding pixels -- the same scheme
    uavsar_calib.cpp uses for its areaRDC array.

    Parameters
    ----------
    area : (N,) ndarray, per-facet area (m^2), from
        compute_facet_area_and_theta_l()
    azimuth_idx, range_idx : (N,) ndarrays, fractional radar-pixel
        coordinates for each facet (from geolocate.ground_to_radar())
    n_azimuth, n_range : int, radar image dimensions
    outlier_factor : float, facets with area > outlier_factor *
        median(area) are dropped before accumulation (grazing-angle /
        near-layover geometries can produce pathologically large area
        values; this is a coarse proxy for shadow/layover masking,
        not a full geometric shadow test)

    Returns
    -------
    (n_azimuth, n_range) ndarray, weighted-average accumulated area
        per radar pixel (gaps where no facet contributed -- see
        fill_gaps_nearest())
    filled_mask : (n_azimuth, n_range) bool ndarray, True where at
        least one facet contributed
    """
    area_ref_nominal = np.median(area)
    keep = area < outlier_factor * area_ref_nominal
    area_k = area[keep]
    az_k = azimuth_idx[keep]
    rg_k = range_idx[keep]

    x1 = np.floor(rg_k).astype(int); x2 = x1 + 1
    y1 = np.floor(az_k).astype(int); y2 = y1 + 1
    inb = (x2 < n_range) & (y2 < n_azimuth) & (x1 >= 0) & (y1 >= 0)
    area_k, x1, x2, y1, y2, rg_k, az_k = (area_k[inb], x1[inb], x2[inb], y1[inb], y2[inb],
                                           rg_k[inb], az_k[inb])

    d11 = np.sqrt((x1 - rg_k) ** 2 + (y1 - az_k) ** 2) + 1e-6
    d12 = np.sqrt((x2 - rg_k) ** 2 + (y1 - az_k) ** 2) + 1e-6
    d21 = np.sqrt((x1 - rg_k) ** 2 + (y2 - az_k) ** 2) + 1e-6
    d22 = np.sqrt((x2 - rg_k) ** 2 + (y2 - az_k) ** 2) + 1e-6

    area_rdc = np.zeros((n_azimuth, n_range), dtype=np.float64)
    sum_wgt = np.zeros((n_azimuth, n_range), dtype=np.float64)
    np.add.at(area_rdc, (y1, x1), area_k / d11); np.add.at(sum_wgt, (y1, x1), 1.0 / d11)
    np.add.at(area_rdc, (y1, x2), area_k / d12); np.add.at(sum_wgt, (y1, x2), 1.0 / d12)
    np.add.at(area_rdc, (y2, x1), area_k / d21); np.add.at(sum_wgt, (y2, x1), 1.0 / d21)
    np.add.at(area_rdc, (y2, x2), area_k / d22); np.add.at(sum_wgt, (y2, x2), 1.0 / d22)

    filled_mask = sum_wgt > 0
    area_rdc[filled_mask] /= sum_wgt[filled_mask]
    return area_rdc, filled_mask


def fill_gaps_nearest(area_rdc, filled_mask):
    """
    Nearest-neighbor gap fill for the sparse areaRDC grid (the DEM's
    native resolution is coarser than the radar pixel spacing, so a
    substantial fraction of radar pixels receive no direct facet
    contribution from accumulate_area_rdc()).

    Returns
    -------
    (n_azimuth, n_range) ndarray, float32, gap-filled area grid
    """
    idx = ndimage.distance_transform_edt(~filled_mask, return_distances=False, return_indices=True)
    return area_rdc[tuple(idx)].astype(np.float32)


def apply_area_correction(backscatter_uncorrected, area_rdc_filled, area_ref_nominal=None):
    """
    Apply the area-projection correction to raw (calibrated but
    uncorrected) backscatter: sigma0_corrected = sigma0_raw *
    (area_ref_nominal / areaRDC).

    Parameters
    ----------
    backscatter_uncorrected : (n_azimuth, n_range) ndarray, linear
        power, from backscatter.read_polarization_backscatter()
    area_rdc_filled : (n_azimuth, n_range) ndarray, from
        fill_gaps_nearest()
    area_ref_nominal : float or None. If None, uses the median of
        area_rdc_filled as an empirical reference (this is NOT a
        theoretically-derived flat-terrain reference area -- see
        docs/verification.md for the known limitation this implies)

    Returns
    -------
    (n_azimuth, n_range) ndarray, corrected linear backscatter
    """
    if area_ref_nominal is None:
        area_ref_nominal = np.median(area_rdc_filled)
    return backscatter_uncorrected * (area_ref_nominal / area_rdc_filled)
