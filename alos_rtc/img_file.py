"""
img_file.py

Reads fields from the ALOS PALSAR CEOS Signal Data Record (the IMG file)
needed for per-pixel slant-range computation and per-line acquisition
timing.

Original contribution: all byte-offset parsing in this file was
implemented by the author based on the JAXA CEOS format specification
(NEB-070062B).
"""

import struct

import numpy as np


FDR_SIZE = 720  # File Descriptor Record, fixed size at the start of the file.


def record_length(n_range):
    """
    Signal Data Record length in bytes for a Level 1.1 product:
    412-byte prefix + n_range complex I/Q samples, 4 bytes each
    (8 bytes per complex sample).
    """
    return 412 + n_range * 8


def read_line_time_ms(img_path, line_index, n_range):
    """
    Read the millisecond-of-day timestamp for a given azimuth line,
    from Signal Data Record byte 45-48 (4-byte big-endian binary integer).
    NOTE: despite the function name, the return value is in SECONDS of day
    (the stored milliseconds are divided by 1000). Do not divide again.
    """
    rl = record_length(n_range)
    with open(img_path, 'rb') as f:
        f.seek(FDR_SIZE + line_index * rl + 45 - 1)
        ms = struct.unpack('>I', f.read(4))[0]
    return ms / 1000.0  # seconds of day


def read_near_range_m(img_path, line_index, n_range):
    """
    Read "slant range to 1st data sample" for a given azimuth line,
    from Signal Data Record byte 117-120 (4-byte big-endian binary
    integer, meters). In practice this value is constant across all
    lines for a given product (verified for ALPSRP049750870).
    """
    rl = record_length(n_range)
    with open(img_path, 'rb') as f:
        f.seek(FDR_SIZE + line_index * rl + 117 - 1)
        near_range_m = struct.unpack('>I', f.read(4))[0]
    return near_range_m


def read_line_times_fit(img_path, n_range, n_azimuth, step=7):
    """
    Sub-millisecond azimuth time of the first and last line.

    The per-line timestamp (bytes 45-48) is an integer millisecond of day,
    rounded to the nearest millisecond, while the line interval is 1/PRF
    (0.464 ms for ALPSRP237630880). Using the stamps of the first and last
    line directly therefore puts the time origin up to 0.5 ms (about one
    line) off. Here a straight line t = a + b*i is fitted to the stamps of
    every `step`-th line; the rounding error averages out.

    Returns (t_first, t_last, dt) in SECONDS of day.
    For ALPSRP237630880: t_first = 11233.3584335 s (stamp 11233.358 s),
    dt = 0.464000 ms = 1/PRF (PRF 2155.172 Hz), maximum residual 0.498 ms.
    """
    rl = record_length(n_range)
    idx = np.arange(0, n_azimuth, step)
    ms = np.empty(len(idx))
    with open(img_path, 'rb') as f:
        for j, i in enumerate(idx):
            f.seek(FDR_SIZE + int(i) * rl + 45 - 1)
            ms[j] = struct.unpack('>I', f.read(4))[0]
    b, a = np.polyfit(idx.astype(float), ms, 1)
    return a / 1000.0, (a + b * (n_azimuth - 1)) / 1000.0, b / 1000.0
