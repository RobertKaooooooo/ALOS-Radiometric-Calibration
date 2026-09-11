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
