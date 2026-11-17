"""
prepare_scene.py -- inputs of final_eq14.py for one ALOS PALSAR FBD L1.1 scene

  1. calibrated, terrain-uncorrected sigma0 (linear power, float32, NA x NR) for HV and HH
     (backscatter.read_polarization_backscatter, Eq. 3.1)
  2. geolocation of every GLO-30 facet inside the scene (geolocate.ground_to_radar, Eq. 3.3):
     rows, cols (DEM grid index), az, rg (fractional radar coordinates)

The conventions are those of facet_results_FINAL2.npz used for ALPSRP237630880, so that
final_eq14.py applies its corrections identically to every scene:
  * heights: GLO-30 values used as ellipsoidal heights (final_eq14.py adds the EGM2008 range shift)
  * azimuth: (t - t_first)/(t_last - t_first)*(NA - 1) with the integer-ms stamps of lines 0 and NA-1
    (final_eq14.py re-maps to the fitted line times)
  * no empirical offset

Run   : cd "/mnt/c/grd file" && RTC_SID=ALPSRP183950880 python3 scripts/prepare_scene.py
Check : cd "/mnt/c/grd file" && RTC_CHECK=1 python3 scripts/prepare_scene.py
        (reference scene: recomputes 200,000 facets of facet_results_FINAL2.npz and reports the differences)
Output: ALOS_HV_<SID>.bin, ALOS_HH_<SID>.bin, facets_<SID>.npz in the base folder; prepare_<SID>_log.txt
"""
import os, sys, time
import numpy as np

BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geolocate, backscatter  # noqa: E402

SID = os.environ.get('RTC_SID', 'ALPSRP237630880')
D = os.path.join(BASE, SID + '-L1.1') + '/'
LED = D + 'LED-' + SID + '-H1.1__A'
IMGS = {pol: D + 'IMG-%s-%s-H1.1__A' % (pol, SID) for pol in ('HV', 'HH')}
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
LOG = open(os.path.join(BASE, 'scripts', 'prepare_%s_log.txt' % SID), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()

def scene_size(img):
    """NR from the record length in the file descriptor (bytes 187-192), NA from the file size."""
    rl = int(open(img, 'rb').read(720)[186:192])
    return (os.path.getsize(img) - 720)//rl, (rl - 412)//8

NA, NR = scene_size(IMGS['HV'])
assert scene_size(IMGS['HH']) == (NA, NR)
t0, hstep, P, V = leader_file.read_platform_position_record(LED)
c0, c1, DR = geolocate.read_doppler_and_spacing(LED)
tf = img_file.read_line_time_ms(IMGS['HH'], 0, NR); te = img_file.read_line_time_ms(IMGS['HH'], NA-1, NR)
R0 = float(img_file.read_near_range_m(IMGS['HH'], 0, NR))
say('[scene] %s: NA %d, NR %d, range spacing %.7f m, near range %.0f m, Doppler %.4f + %.6f R_km Hz, lines %.3f-%.3f s'
    % (SID, NA, NR, DR, R0, c0, c1, tf, te))
nd = int(round(np.sqrt(os.path.getsize(DEMFILE)/4)))
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(nd, nd)

def locate(r, c):
    lat = LAT_TOP - r/PPD; lon = LON_LEFT + c/PPD; h = dem[r, c].astype(np.float64)
    return geolocate.ground_to_radar(lat, lon, h, tf, te, NA, R0, DR, t0, hstep, P, V, c0, c1)

def locate_chunked(r, c, chunk=400000):
    az = np.empty(len(r)); rg = np.empty(len(r))
    for i in range(0, len(r), chunk):
        az[i:i+chunk], rg[i:i+chunk] = locate(r[i:i+chunk], c[i:i+chunk])
    return az, rg

if os.environ.get('RTC_CHECK'):
    g = np.load(os.path.join(BASE, 'facet_results_FINAL2.npz'))
    idx = np.random.default_rng(0).choice(len(g['rows']), 200000, replace=False)
    az, rg = locate_chunked(g['rows'][idx], g['cols'][idx])
    say('[check] 200,000 facets of facet_results_FINAL2.npz: |d az| max %.2e mean %.2e lines; |d rg| max %.2e mean %.2e px'
        % (np.abs(az - g['az'][idx]).max(), np.abs(az - g['az'][idx]).mean(), np.abs(rg - g['rg'][idx]).max(), np.abs(rg - g['rg'][idx]).mean()))
    say('[check] dtypes in FINAL2: rows %s cols %s az %s rg %s' % (g['rows'].dtype, g['cols'].dtype, g['az'].dtype, g['rg'].dtype))
    sys.exit(0)

# ---- 1. backscatter ----
for pol, img in IMGS.items():
    out = os.path.join(BASE, 'ALOS_%s_%s.bin' % (pol, SID))
    if os.path.exists(out) and os.path.getsize(out) == NA*NR*4:
        say('[sigma0] %s exists, skipped' % out); continue
    t = time.time(); backscatter.read_polarization_backscatter(img, NR, NA, out_path=out, progress_every=0)
    say('[sigma0] %s written (%.0f s)' % (out, time.time() - t))

# ---- 2. facet geolocation ----
t = time.time()
rr, cc = np.meshgrid(np.arange(1, nd-1, 20), np.arange(1, nd-1, 20), indexing='ij'); rr = rr.ravel(); cc = cc.ravel()
az, rg = locate_chunked(rr, cc)
inside = (az >= 0) & (az <= NA-1) & (rg >= 0) & (rg <= NR-1)
r0, r1 = max(rr[inside].min() - 40, 1), min(rr[inside].max() + 40, nd-2)
c0_, c1_ = max(cc[inside].min() - 40, 1), min(cc[inside].max() + 40, nd-2)
say('[facets] coarse search: DEM rows %d-%d, cols %d-%d (lat %.3f-%.3f, lon %.3f-%.3f)'
    % (r0, r1, c0_, c1_, LAT_TOP - r1/PPD, LAT_TOP - r0/PPD, LON_LEFT + c0_/PPD, LON_LEFT + c1_/PPD))
keep = {'rows': [], 'cols': [], 'az': [], 'rg': []}
for rs in range(r0, r1 + 1, 100):
    rA, cA = np.meshgrid(np.arange(rs, min(rs + 100, r1 + 1)), np.arange(c0_, c1_ + 1), indexing='ij')
    rA = rA.ravel().astype(np.int32); cA = cA.ravel().astype(np.int32)
    a, g_ = locate_chunked(rA, cA)
    m = (a >= 0) & (a <= NA-1) & (g_ >= 0) & (g_ <= NR-1) & np.isfinite(dem[rA, cA])
    for k_, v in zip(('rows', 'cols', 'az', 'rg'), (rA[m], cA[m], a[m], g_[m])): keep[k_].append(v)
out = {k_: np.concatenate(v) for k_, v in keep.items()}
np.savez(os.path.join(BASE, 'facets_%s.npz' % SID), **out)
say('[facets] %d facets inside the image (%.0f s) -> facets_%s.npz' % (len(out['rows']), time.time() - t, SID))
LOG.close()
