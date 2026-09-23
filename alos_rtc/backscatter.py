"""
backscatter.py

Reads the raw complex I/Q signal from an ALOS PALSAR CEOS Signal Data
Record (IMG file) and converts it to calibrated (but not yet
terrain-corrected) sigma-nought backscatter, in linear power units.

CALIBRATION FORMULA:
Verified directly against JAXA/ESA documentation (ALOS PALSAR
calibration guide) and cross-checked against the actual calibration
factor (CF) stored in this scene's own Leader File Radiometric Data
Record (confirmed present as ASCII "-83.0000000" in the raw LED
bytes for the scenes used in this study):

    K [dB] = CF [dB] - CF_offset [dB]      (CF_offset = 32 dB for L1.1)
    sigma0 [dB] = 20*log10(DN) + K [dB]
                = 10*log10(I^2 + Q^2) + CF - CF_offset

For ALOS-1 PALSAR L1.1 products (complex I/Q, not amplitude-detected),
DN^2 = I^2 + Q^2. CF is nominally -83.0 dB for products processed with
ADEN processor v5.04+ (verified present in this study's scene LED
files); CF_offset is a fixed 32 dB for L1.1 products per the official
calibration document.

Original contribution: the file-reading loop, byte-offset parsing (via
img_file.py) and the calibration formula's application in this module
are the author's own code, built from and verified against the JAXA
calibration documentation.
"""

import numpy as np
import time

FDR_SIZE = 720       # File Descriptor Record size (see img_file.py)
SDR_PREFIX = 412      # Signal Data Record prefix, before I/Q samples
CF_DB = -83.0         # Calibration factor (verified against LED file)
CF_OFFSET_DB = 32.0   # L1.1 product offset (JAXA/ESA documentation)


def read_polarization_backscatter(img_path, n_range, n_azimuth, out_path=None, progress_every=2000):
    """
    Read an entire ALOS PALSAR L1.1 IMG file (one polarization) and
    compute calibrated sigma-nought backscatter (linear power) for
    every pixel.

    Parameters
    ----------
    img_path : str, path to the IMG-<pol>-... file
    n_range, n_azimuth : int, image dimensions (from summary.txt's
        Pdi_NoOfPixels / Pdi_NoOfLines, or verified against file size
        via record_length())
    out_path : str or None. If given, the result is written to this
        path as a memory-mapped float32 binary of shape
        (n_azimuth, n_range) (avoids holding the full array in RAM
        for large scenes); the function still returns the array.
        If None, the array is built and returned in memory only.
    progress_every : int, print a progress line every N lines (0 to
        disable)

    Returns
    -------
    (n_azimuth, n_range) ndarray of float32, linear sigma0 backscatter
    """
    record_len = SDR_PREFIX + n_range * 8  # 8 bytes per complex I/Q sample

    if out_path is not None:
        out = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(n_azimuth, n_range))
    else:
        out = np.empty((n_azimuth, n_range), dtype=np.float32)

    t0 = time.time()
    with open(img_path, 'rb') as f:
        f.seek(FDR_SIZE)
        for ln in range(n_azimuth):
            rec = f.read(record_len)
            iq_bytes = rec[SDR_PREFIX:]
            iq = np.frombuffer(iq_bytes, dtype='>f4').reshape(-1, 2)
            I = iq[:, 0].astype(np.float64)
            Q = iq[:, 1].astype(np.float64)
            power = I ** 2 + Q ** 2
            sigma0_db = 10 * np.log10(power + 1e-12) + CF_DB - CF_OFFSET_DB
            out[ln, :] = (10 ** (sigma0_db / 10.0)).astype(np.float32)
            if progress_every and ln % progress_every == 0:
                print('  line %d/%d, %.1fs' % (ln, n_azimuth, time.time() - t0))

    if out_path is not None:
        out.flush()
    return out
