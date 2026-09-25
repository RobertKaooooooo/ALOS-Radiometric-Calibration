"""
geocode.py

Geocoding utilities:
  1. Corner-based bilinear lat/lon interpolation across the ALOS image
     frame (azimuth x range).
  2. DEM bilinear sampling at arbitrary lat/lon.
  3. Full-image batch driver that computes theta_l over the entire
     radar-coordinate grid, processing rows in batches to keep memory
     use bounded (see docs/verification.md, "memory limits" note --
     an earlier attempt to read an entire large array into memory at
     once was killed by the OOM killer on a 7.7 GiB machine).
  4. Radar-coordinate -> DEM-coordinate resampling of a corrected
     backscatter raster, using a UAVSAR-style (ranpix, azpix) complex64
     mapping table (see docs/verification.md, ".trans file format").

All functions in this file are the author's own implementation.
"""

import numpy as np
import time

from . import leader_file
from . import img_file
from . import geometry


def bilinear_corner_latlon(azimuth_index, range_index, n_azimuth, n_range,
                            corners):
    """
    Interpolate lat/lon for a pixel at (azimuth_index, range_index)
    using the four corner coordinates of the image frame.

    `corners` is a dict with keys 'near0','far0','nearN','farN', each
    an (lat, lon) tuple, for:
        near0 : azimuth=0,            range=near
        far0  : azimuth=0,            range=far
        nearN : azimuth=n_azimuth-1,  range=near
        farN  : azimuth=n_azimuth-1,  range=far

    These four corners are normally read from Signal Data Record byte
    193-216 (latitude/longitude of 1st/mid/last pixel, millionths of
    degrees) for the first and last azimuth lines. In this pipeline
    they are supplied as pre-extracted constants (see run_pipeline.py)
    for reuse without re-parsing the IMG file on every call.
    """
    a = azimuth_index / (n_azimuth - 1)
    r = range_index / (n_range - 1)

    lat_00, lon_00 = corners['near0']
    lat_0N, lon_0N = corners['far0']
    lat_M0, lon_M0 = corners['nearN']
    lat_MN, lon_MN = corners['farN']

    lat = (lat_00 * (1 - a) * (1 - r) + lat_0N * (1 - a) * r
           + lat_M0 * a * (1 - r) + lat_MN * a * r)
    lon = (lon_00 * (1 - a) * (1 - r) + lon_0N * (1 - a) * r
           + lon_M0 * a * (1 - r) + lon_MN * a * r)
    return lat, lon


def dem_bilinear(dem, lat, lon, dem_lat_top, dem_lon_left, pixels_per_deg=3600):
    """
    Bilinear elevation lookup in a DEM array covering `dem_lat_top`
    (north edge) to south and `dem_lon_left` (west edge) to east, at
    `pixels_per_deg` resolution (3600 = 1 arc-second, e.g. GLO-30).
    """
    row = (dem_lat_top - lat) * pixels_per_deg - 0.5
    col = (lon - dem_lon_left) * pixels_per_deg - 0.5
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    fr, fc = row - r0, col - c0
    r0 = min(max(r0, 0), dem.shape[0] - 2)
    c0 = min(max(c0, 0), dem.shape[1] - 2)
    z00, z01 = dem[r0, c0], dem[r0, c0 + 1]
    z10, z11 = dem[r0 + 1, c0], dem[r0 + 1, c0 + 1]
    return (z00 * (1 - fr) * (1 - fc) + z01 * (1 - fr) * fc
            + z10 * fr * (1 - fc) + z11 * fr * fc)


# DEPRECATED: process_full_image() uses four-corner bilinear geolocation and
# fixes yaw = 0.0 (LED records ~-2.8 deg). Superseded by geolocate.ground_to_radar()
# + area_correction.compute_facet_area_and_theta_l(), which use Hermite orbit
# interpolation, the Doppler-centroid geometry and the interpolated LED yaw.
def process_full_image(led_path, img_path, dem, dem_lat_top, dem_lon_left,
                        corners, n_azimuth, n_range, near_range_m,
                        pixel_spacing_m, coefficients, heading_rad,
                        out_theta_l_path, out_nE_path_prefix=None,
                        batch_size=500, pixels_per_deg=3600):
    """
    Compute theta_l (and optionally nE) over the full radar-coordinate
    grid (n_azimuth x n_range), writing results to disk with
    np.memmap so the whole array is never held in RAM at once.

    yaw is set to 0 for every pixel in this reference implementation
    (verified negligible: Attitude Data Record shows <0.06 deg yaw
    variation over a 21-second window for this product). Replace with
    a per-line interpolated yaw if working with a product where yaw
    varies more.
    """
    theta_ih_per_column = geometry.compute_theta_ih(
        coefficients,
        slant_range_km=np.array([
            slant_range_km_for_col(near_range_m, pixel_spacing_m, c)
            for c in range(n_range)
        ])
    )

    att_times, att_pitch, _, _ = leader_file.read_attitude_record(led_path)

    # Pre-pass: read the real per-line acquisition time from the IMG file
    # for every azimuth line, and interpolate pitch once into a lookup
    # table. This mirrors the verified approach used during development
    # (building `pitches_per_line` up front is far faster than re-reading
    # the IMG file inside the per-pixel inner loop, and was cross-checked
    # against single-point hand calculations at the scene-center pixel).
    print('building per-line pitch lookup table...')
    pitch_per_line = np.zeros(n_azimuth)
    for ln in range(n_azimuth):
        t_line = img_file.read_line_time_ms(img_path, ln, n_range)
        pitch_per_line[ln] = leader_file.interpolate_pitch(
            led_path, t_line, att_cache=(att_times, att_pitch))
    print('pitch lookup table built')

    theta_l_out = np.memmap(out_theta_l_path, dtype=np.float32, mode='w+',
                             shape=(n_azimuth, n_range))
    nE_out = None
    if out_nE_path_prefix is not None:
        nE_out = {
            axis: np.memmap(f'{out_nE_path_prefix}_{axis}.bin', dtype=np.float32,
                             mode='w+', shape=(n_azimuth, n_range))
            for axis in ('x', 'y', 'z')
        }

    dlat_m = 111320.0 / pixels_per_deg
    t_start = time.time()

    for start in range(0, n_azimuth, batch_size):
        end = min(start + batch_size, n_azimuth)
        for ln in range(start, end):
            pitch = pitch_per_line[ln]

            r = np.arange(n_range) / (n_range - 1)
            a = ln / (n_azimuth - 1)
            lat_row, lon_row = _row_latlon(a, r, corners)

            row_f = (dem_lat_top - lat_row) * pixels_per_deg - 0.5
            col_f = (lon_row - dem_lon_left) * pixels_per_deg - 0.5
            r0 = np.clip(np.floor(row_f).astype(int), 0, dem.shape[0] - 2)
            c0 = np.clip(np.floor(col_f).astype(int), 0, dem.shape[1] - 2)
            fr = row_f - r0
            fc = col_f - c0

            def bilin(dr, dc):
                return dem[np.clip(r0 + dr, 0, dem.shape[0] - 1),
                           np.clip(c0 + dc, 0, dem.shape[1] - 1)].astype(np.float32)

            Z5 = (dem[r0, c0] * (1 - fr) * (1 - fc) + dem[r0, c0 + 1] * (1 - fr) * fc
                  + dem[r0 + 1, c0] * fr * (1 - fc) + dem[r0 + 1, c0 + 1] * fr * fc)
            Z2, Z8 = bilin(-1, 0), bilin(1, 0)
            Z4, Z6 = bilin(0, -1), bilin(0, 1)
            Z1, Z3 = bilin(-1, -1), bilin(-1, 1)
            Z7, Z9 = bilin(1, -1), bilin(1, 1)

            dlon_m = 111320.0 * np.cos(np.radians(lat_row)) / pixels_per_deg
            p = (Z3 + Z6 + Z9 - Z1 - Z4 - Z7) / (6 * dlon_m)
            q = (Z1 + Z2 + Z3 - Z7 - Z8 - Z9) / (6 * dlat_m)
            slope = np.arctan(np.sqrt(p**2 + q**2))
            # Matches uavsar_calib.cpp exactly: atan(q/p), NOT arctan2.
            p_safe = np.where(p == 0, 1.0, p)
            aspect = np.where(p == 0, np.where(q > 0, np.pi, 0.0),
                              np.pi - np.arctan(q / p_safe) + (np.pi / 2) * np.sign(p_safe))
            slope_r = np.tan(slope) * np.cos(aspect - heading_rad - np.pi / 2)
            slope_a = np.tan(slope) * np.cos(aspect - heading_rad)
            temp = -1.0 / np.sqrt(1.0 + slope_r**2 + slope_a**2)
            nEx, nEy, nEz = temp * slope_a, temp * slope_r, -temp

            yaw = 0.0
            l_s = (np.sin(pitch) * np.cos(theta_ih_per_column) * np.cos(yaw)
                   + np.sin(theta_ih_per_column) * np.sin(yaw))
            l_c = (-np.sin(pitch) * np.cos(theta_ih_per_column) * np.sin(yaw)
                   + np.sin(theta_ih_per_column) * np.cos(yaw))
            l_h = -np.cos(pitch) * np.cos(theta_ih_per_column)
            dotv = nEx * l_s + nEy * l_c + nEz * (-l_h)
            theta_l_out[ln, :] = np.degrees(np.arccos(np.clip(dotv, -1, 1)))

            if nE_out is not None:
                nE_out['x'][ln, :] = nEx
                nE_out['y'][ln, :] = nEy
                nE_out['z'][ln, :] = nEz

        if start % 5000 == 0:
            print('line %d/%d, elapsed %.1fs' % (start, n_azimuth, time.time() - t_start))

    theta_l_out.flush()
    if nE_out is not None:
        for arr in nE_out.values():
            arr.flush()
    print('DONE. total elapsed %.1f sec' % (time.time() - t_start))


def slant_range_km_for_col(near_range_m, pixel_spacing_m, col):
    return geometry.slant_range_km(near_range_m, pixel_spacing_m, col)


def _row_latlon(a, r, corners):
    lat_00, lon_00 = corners['near0']
    lat_0N, lon_0N = corners['far0']
    lat_M0, lon_M0 = corners['nearN']
    lat_MN, lon_MN = corners['farN']
    lat = (lat_00 * (1 - a) * (1 - r) + lat_0N * (1 - a) * r
           + lat_M0 * a * (1 - r) + lat_MN * a * r)
    lon = (lon_00 * (1 - a) * (1 - r) + lon_0N * (1 - a) * r
           + lon_M0 * a * (1 - r) + lon_MN * a * r)
    return lat, lon


def resample_backscatter_to_dem_grid(trans_path, backscatter_path,
                                      n_dem_rows, n_dem_cols,
                                      n_range, n_azimuth,
                                      out_path, batch_size=500):
    """
    Resample a radar-coordinate corrected-backscatter raster onto the
    DEM coordinate grid, using a (ranpix, azpix) mapping table stored
    as complex64 (real=ranpix, imag=azpix), shape (n_dem_rows,
    n_dem_cols). This mapping table is the `gc_out` output described
    in Simard's uavsar_calib.cpp (Category A, reused logic) --
    generating it is UAVSAR/C++ side work, not part of this Python
    package; this function only consumes it.

    Invalid/out-of-footprint pixels are written as -1.0 in the output.
    """
    trans = np.memmap(trans_path, dtype=np.complex64, mode='r',
                       shape=(n_dem_rows, n_dem_cols))
    backscatter = np.memmap(backscatter_path, dtype=np.float32, mode='r',
                             shape=(n_azimuth, n_range))
    out = np.memmap(out_path, dtype=np.float32, mode='w+',
                     shape=(n_dem_rows, n_dem_cols))

    t_start = time.time()
    for row_start in range(0, n_dem_rows, batch_size):
        row_end = min(row_start + batch_size, n_dem_rows)
        geo_chunk = np.asarray(trans[row_start:row_end, :])
        ranpix = geo_chunk.real
        azpix = geo_chunk.imag
        valid = (ranpix >= 0) & (ranpix < n_range - 1) & (azpix >= 0) & (azpix < n_azimuth - 1)

        r0 = np.clip(np.floor(ranpix).astype(int), 0, n_range - 2)
        a0 = np.clip(np.floor(azpix).astype(int), 0, n_azimuth - 2)
        fr = ranpix - r0
        fa = azpix - a0

        out_chunk = np.full(geo_chunk.shape, -1.0, dtype=np.float32)
        vr, vc = np.where(valid)
        for i in range(len(vr)):
            rr, cc = vr[i], vc[i]
            r0v, a0v = r0[rr, cc], a0[rr, cc]
            frv, fav = fr[rr, cc], fa[rr, cc]
            v00 = backscatter[a0v, r0v]
            v01 = backscatter[a0v, r0v + 1]
            v10 = backscatter[a0v + 1, r0v]
            v11 = backscatter[a0v + 1, r0v + 1]
            out_chunk[rr, cc] = (v00 * (1 - frv) * (1 - fav) + v01 * frv * (1 - fav)
                                 + v10 * (1 - frv) * fav + v11 * frv * fav)

        out[row_start:row_end, :] = out_chunk
        if row_start % 2000 == 0:
            print('row %d/%d, elapsed %.1f sec' % (row_start, n_dem_rows, time.time() - t_start))

    out.flush()
    print('DONE. total elapsed %.1f min' % ((time.time() - t_start) / 60))
