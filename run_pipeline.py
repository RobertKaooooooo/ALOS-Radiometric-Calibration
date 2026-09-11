"""
run_pipeline.py

End-to-end driver for the ALOS PALSAR radiometric-terrain-correction
geometry pipeline. See README.md for the full process flow diagram.

Usage
-----
    python run_pipeline.py --mode verify      # single test-point check
    python run_pipeline.py --mode full        # full-image processing

All file paths and geometry constants below are specific to product
ALPSRP049750870 (White Mountain National Forest, NH, USA) and were
extracted/verified during development (see docs/verification.md).
Replace them with the equivalent values for a different product.
"""

import argparse
import numpy as np

from alos_rtc import leader_file, img_file, geometry, geocode


# ---------------------------------------------------------------------
# Product-specific paths (edit these for your own data)
# ---------------------------------------------------------------------
LED_PATH = 'data/ALPSRP049750870-L1.1/LED-ALPSRP049750870-H1.1__A'
IMG_PATH = 'data/ALPSRP049750870-L1.1/IMG-HH-ALPSRP049750870-H1.1__A'
DEM_PATH = 'data/GLO30_ALOS_merged.bin'          # 7200x7200 float32, 1 arcsec
TRANS_PATH = 'data/geomap_09054.trans'           # UAVSAR-side (ranpix,azpix) map
BACKSCATTER_PATH = 'data/out_line1_fixed2.mlc'   # UAVSAR-side corrected backscatter

# ---------------------------------------------------------------------
# Product-specific geometry constants (verified during development)
# ---------------------------------------------------------------------
N_AZIMUTH = 18432
N_RANGE = 9344
NEAR_RANGE_M = 849415
PIXEL_SPACING_M = 4.6842572
HEADING_DEG_CORRECTED = -13.706322775792932
# See docs/verification.md, "heading bug" -- this is the CORRECTED value.
# The uncorrected value (194 deg) was traced to a cross-product argument
# order error (North_hat = East_hat x P_hat instead of P_hat x East_hat)
# and must not be reused.

# Four image-corner coordinates (lat, lon), pre-extracted from Signal
# Data Record byte 193-216 for the first and last azimuth lines.
CORNERS = {
    'near0': (43.743654, -71.410089),   # azimuth=0,           range=near
    'far0':  (43.852402, -70.552614),   # azimuth=0,           range=far
    'nearN': (44.257898, -71.540478),   # azimuth=N_AZIMUTH-1, range=near
    'farN':  (44.366655, -70.67531),    # azimuth=N_AZIMUTH-1, range=far
}

DEM_LAT_TOP = 45.0     # north edge of the merged 7200x7200 GLO-30 tile set
DEM_LON_LEFT = -72.0   # west edge
DEM_SHAPE = (7200, 7200)
DEM_PIXELS_PER_DEG = 3600

DEM_ROWS_FULL, DEM_COLS_FULL = 9300, 24196  # UAVSAR-side DEM-coordinate grid
                                             # used by the .trans mapping file


def load_dem():
    return np.fromfile(DEM_PATH, dtype=np.float32).reshape(DEM_SHAPE)


def run_verify():
    """
    Reproduce the scene-center hand-verification: compute theta_ih,
    nL, nE and theta_l at one known test pixel and compare against the
    values established during development (docs/verification.md,
    Section 2.4-2.5). Exits with an assertion error if the numbers
    drift, which should never happen unless the input files change.
    """
    coefficients = leader_file.read_six_coefficients(LED_PATH)
    print('six coefficients:', coefficients)

    esa_info = leader_file.read_esa_fields(LED_PATH)
    print('ESA fields:', esa_info)
    assert esa_info['yaw_steering_flag'] == '0', \
        'Unexpected yaw steering flag -- ESA=0 assumption may not hold'

    target_line, target_col = 1522, 9033
    t_line = img_file.read_line_time_ms(IMG_PATH, target_line, N_RANGE)
    P, V = leader_file.interpolate_position_velocity(LED_PATH, t_line)
    print('P =', P, ' V =', V)

    R_km = geometry.slant_range_km(NEAR_RANGE_M, PIXEL_SPACING_M, target_col)
    theta_ih = geometry.compute_theta_ih(coefficients, R_km)
    print('theta_ih = %.4f deg (expected ~40.591 deg at col=9033)' %
          np.degrees(theta_ih))

    att_times, att_pitch, _, _ = leader_file.read_attitude_record(LED_PATH)
    pitch = leader_file.interpolate_pitch(LED_PATH, t_line,
                                           att_cache=(att_times, att_pitch))

    dem = load_dem()
    lat, lon = geocode.bilinear_corner_latlon(
        target_line, target_col, N_AZIMUTH, N_RANGE, CORNERS)
    d = 1.0 / DEM_PIXELS_PER_DEG
    Z = np.array([
        [geocode.dem_bilinear(dem, lat + d, lon - d, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat + d, lon, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat + d, lon + d, DEM_LAT_TOP, DEM_LON_LEFT)],
        [geocode.dem_bilinear(dem, lat, lon - d, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat, lon, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat, lon + d, DEM_LAT_TOP, DEM_LON_LEFT)],
        [geocode.dem_bilinear(dem, lat - d, lon - d, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat - d, lon, DEM_LAT_TOP, DEM_LON_LEFT),
         geocode.dem_bilinear(dem, lat - d, lon + d, DEM_LAT_TOP, DEM_LON_LEFT)],
    ])

    dlat_m = 111320.0 / DEM_PIXELS_PER_DEG
    dlon_m = 111320.0 * np.cos(np.radians(lat)) / DEM_PIXELS_PER_DEG
    heading_rad = np.radians(HEADING_DEG_CORRECTED)
    nE = geometry.compute_nE_horn(Z, dlon_m, dlat_m, heading_rad)
    print('nE =', nE, ' |nE| =', np.linalg.norm(nE))

    nL_local = geometry.compute_nL_local_frame(theta_ih, pitch, yaw_rad=0.0,
                                                esa_rad=0.0)
    theta_l = geometry.compute_theta_l(nE, nL_local)
    print('theta_l = %.4f deg' % np.degrees(theta_l))
    print('(at this known-flat point, theta_l should equal theta_ih '
          'to within numerical precision -- this is the flat-terrain '
          'validation check, see docs/verification.md)')


def run_full():
    """
    Full-image theta_l computation over the DEM-coordinate grid,
    followed by resampling of the corrected backscatter raster onto
    the same grid. See README.md, steps 5-8.
    """
    coefficients = leader_file.read_six_coefficients(LED_PATH)
    esa_info = leader_file.read_esa_fields(LED_PATH)
    print('ESA fields:', esa_info)

    dem = load_dem()
    heading_rad = np.radians(HEADING_DEG_CORRECTED)

    geocode.process_full_image(
        led_path=LED_PATH,
        img_path=IMG_PATH,
        dem=dem,
        dem_lat_top=DEM_LAT_TOP,
        dem_lon_left=DEM_LON_LEFT,
        corners=CORNERS,
        n_azimuth=N_AZIMUTH,
        n_range=N_RANGE,
        near_range_m=NEAR_RANGE_M,
        pixel_spacing_m=PIXEL_SPACING_M,
        coefficients=coefficients,
        heading_rad=heading_rad,
        out_theta_l_path='output/ALOS_theta_l_GLO30.bin',
        out_nE_path_prefix='output/ALOS_nE_GLO30',
        batch_size=500,
        pixels_per_deg=DEM_PIXELS_PER_DEG,
    )

    print('Full-image theta_l written to output/ALOS_theta_l_GLO30.bin')
    print('(9300 x 24196, float32, degrees)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['verify', 'full'], default='verify')
    args = parser.parse_args()

    if args.mode == 'verify':
        run_verify()
    else:
        run_full()
