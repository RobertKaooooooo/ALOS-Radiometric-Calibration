"""
leader_file.py

Reads ALOS PALSAR CEOS Leader File records needed for the look-vector (nL)
and pure-geometric incidence-angle (theta_ih) computation.

Original contribution (no UAVSAR counterpart): all byte-offset parsing in
this file was implemented by the author based on the official JAXA CEOS
format specification (NEB-070062B, "ALOS/PALSAR Level 1.1/1.5 product
Format description").

Record layout notes (verified against real product ALPSRP049750870):
  - File Descriptor Record: fixed 720 bytes at the start of the file.
  - Data Set Summary Record: starts immediately after (absolute file
    offset 720). Field byte numbers below are "record byte" numbers as
    given in the JAXA spec; absolute file offset = 720 + record_byte - 1.
  - Platform Position Data Record: absolute file offset determined
    empirically to start at byte 4816 for this product (28 points,
    60 s interval, ECR/ECEF coordinates).
  - Attitude Data Record: absolute file offset 720 + 4096 + 4680,
    ASCII-encoded fields (not binary).
"""

import struct
import numpy as np


DSR_BASE = 720          # Data Set Summary Record starts right after the
                         # 720-byte File Descriptor Record.
PPDR_BASE = 4816         # Platform Position Data Record absolute offset
                         # (empirically verified for ALPSRP049750870).
ADR_BASE = 720 + 4096 + 4680  # Attitude Data Record absolute offset.


def read_six_coefficients(led_path):
    """
    Read the six incidence-angle polynomial coefficients from the Data
    Set Summary Record (record byte 1887-2006, 6 x 20-char ASCII floats).

    theta_ih(R) = a0 + a1*R + a2*R^2 + a3*R^3 + a4*R^4 + a5*R^5
    where R is slant range in km and theta_ih is returned in radians.
    """
    with open(led_path, 'rb') as f:
        f.seek(DSR_BASE + 1887 - 1)
        coeffs = [float(f.read(20).decode('ascii').strip()) for _ in range(6)]
    return coeffs


def read_esa_fields(led_path):
    """
    Read the fields needed to verify whether ALOS needs an ESA
    (electronic steering angle) term in the shared look-vector formula
    (Simard et al. 2016, Eq. 11).

    - Electronic boresight: record byte 899-914
    - Mechanical boresight: record byte 915-930
    - Yaw Steering Mode Flag: record byte 1831-1834
        '0' = yaw steering active   -> ESA = 0 in the shared formula
        '1' = no yaw steering       -> ESA term would need a real value
    """
    with open(led_path, 'rb') as f:
        f.seek(DSR_BASE + 899 - 1)
        electronic_boresight = float(f.read(16).decode('ascii').strip())
        f.seek(DSR_BASE + 915 - 1)
        mechanical_boresight = float(f.read(16).decode('ascii').strip())
        f.seek(DSR_BASE + 1831 - 1)
        yaw_steering_flag = f.read(4).decode('ascii').strip()
    return {
        'electronic_boresight_deg': electronic_boresight,
        'mechanical_boresight_deg': mechanical_boresight,
        'yaw_steering_flag': yaw_steering_flag,
        'esa_value_deg': 0.0 if yaw_steering_flag == '0' else None,
    }


def read_platform_position_record(led_path):
    """
    Read the full Platform Position Data Record: 28 discrete (position,
    velocity) points in ECEF/ECR coordinates, one every 60 seconds.

    Returns
    -------
    t0 : float          seconds of day for the first point
    interval : float    seconds between consecutive points
    positions : (N,3) ndarray, meters
    velocities : (N,3) ndarray, m/s
    """
    with open(led_path, 'rb') as f:
        f.seek(PPDR_BASE + 141 - 1)
        n_points = int(f.read(4).decode('ascii').strip())
        f.seek(PPDR_BASE + 161 - 1)
        t0 = float(f.read(22).decode('ascii').strip())
        f.seek(PPDR_BASE + 183 - 1)
        interval = float(f.read(22).decode('ascii').strip())

        f.seek(PPDR_BASE + 387 - 1)
        positions, velocities = [], []
        for _ in range(n_points):
            vals = [float(f.read(22).decode('ascii').strip()) for _ in range(6)]
            positions.append(vals[:3])
            velocities.append(vals[3:])

    return t0, interval, np.array(positions), np.array(velocities)


def interpolate_position_velocity(led_path, t_line):
    """
    Linearly interpolate the satellite position (P) and velocity (V)
    vectors to an arbitrary time-of-day `t_line` (seconds), by
    bracketing between the two nearest of the 28 recorded orbit points.

    This is the author's own code (the 28 raw points themselves are
    JAXA's precise orbit determination output, not computed here).
    """
    t0, interval, positions, velocities = read_platform_position_record(led_path)
    n_points = len(positions)

    idx = int((t_line - t0) / interval)
    idx = max(0, min(idx, n_points - 2))
    frac = (t_line - (t0 + idx * interval)) / interval

    P = positions[idx] * (1 - frac) + positions[idx + 1] * frac
    V = velocities[idx] * (1 - frac) + velocities[idx + 1] * frac
    return P, V


def read_attitude_record(led_path):
    """
    Read the full Attitude Data Record: per-point day-of-year,
    millisecond-of-day, pitch, roll, yaw (all ASCII-encoded, not binary).

    Returns
    -------
    times : (N,) ndarray, seconds of day
    pitch_deg, roll_deg, yaw_deg : (N,) ndarrays, degrees
    """
    with open(led_path, 'rb') as f:
        f.seek(ADR_BASE + 13 - 1)
        n_att = int(f.read(4).decode('ascii').strip())

        times, pitch, roll, yaw = [], [], [], []
        for i in range(n_att):
            off = ADR_BASE + 17 - 1 + i * 120
            f.seek(off)
            f.read(4)  # day of year, unused here
            ms = int(f.read(8).decode('ascii').strip())
            f.seek(off + 24)
            p = float(f.read(14).decode('ascii').strip())
            r = float(f.read(14).decode('ascii').strip())
            y = float(f.read(14).decode('ascii').strip())
            times.append(ms / 1000.0)
            pitch.append(p)
            roll.append(r)
            yaw.append(y)

    return np.array(times), np.array(pitch), np.array(roll), np.array(yaw)


def interpolate_pitch(led_path, t_line, att_cache=None):
    """
    Linearly interpolate the pitch angle (radians) to time `t_line`.

    `att_cache` may be a pre-loaded (times, pitch, roll, yaw) tuple from
    read_attitude_record() to avoid re-reading the file for every line
    when processing a full image (see geocode.py).
    """
    if att_cache is None:
        times, pitch_deg, _, _ = read_attitude_record(led_path)[:2]
    else:
        times, pitch_deg = att_cache[0], att_cache[1]

    n = len(times)
    idx = int(np.searchsorted(times, t_line) - 1)
    idx = max(0, min(idx, n - 2))
    frac = (t_line - times[idx]) / (times[idx + 1] - times[idx])
    pitch_interp_deg = pitch_deg[idx] * (1 - frac) + pitch_deg[idx + 1] * frac
    return np.radians(pitch_interp_deg)
