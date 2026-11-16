"""
poster_maps.py  -- before / after maps for the TASTI poster, final method
before: sigma0 (flat ellipsoid) averaged in 31 x 7 windows
after : plan A, Eq. 14 window integration, gamma0 = sigma0_terrain / cos^0.5(theta_loc)
Run   : cd "/mnt/c/grd file" && RTC_GEOID_SHIFT=-28 python3 scripts/poster_maps.py
Output: scripts/out_ellip/poster_maps_eq14.png, poster_maps_eq14.npz, poster_maps_log.txt
"""
import os, sys, gc, glob
import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion

BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geometry, geolocate, area_correction  # noqa: E402
import rasterio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import glob as _glob                                              # poster typography: text TASA Explorer, maths Cambria (as PowerPoint equations)
from matplotlib import font_manager as _fm
_TH = bool(os.environ.get('RTC_THESIS'))                            # RTC_THESIS=1: serif fonts and plain labels for the thesis; output names get the suffix _thesis
for _f in _glob.glob(os.path.expanduser('~/.fonts/*.tt[fc]')) + (_glob.glob('/mnt/c/Windows/Fonts/times*.ttf') if _TH else []): _fm.fontManager.addfont(_f)
plt.rcParams.update({'font.family': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix'} if _TH else
                    {'font.family': ['TASA Explorer', 'DejaVu Sans'], 'mathtext.fontset': 'custom', 'mathtext.rm': 'Cambria',
                     'mathtext.it': 'Cambria:italic', 'mathtext.bf': 'Cambria:bold', 'mathtext.fallback': 'cm'})
_SFX = '_thesis' if _TH else ''

D    = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID  = 'ALPSRP237630880'
LED  = D + 'LED-' + SID + '-H1.1__A'
IMG  = D + 'IMG-HH-' + SID + '-H1.1__A'
NA, NR = int(os.environ.get('RTC_NA', 18432)), int(os.environ.get('RTC_NR', 4704))
BSFILE  = os.path.join(BASE, 'ALOS_HV_uncorrected_new.bin')
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
CCIFILE = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
WA, WR = 31, 7
INC_MIN, INC_MAX = 15.0, 70.0
GEOID_N = os.environ.get('RTC_GEOID_SHIFT')   # EGM2008 undulation (m): heights -> ellipsoid, range shift -N*cos(theta_inc)/DR
OUT = os.path.join(BASE, 'scripts', 'out_ellip' if GEOID_N else 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'poster_maps_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()
db = lambda x: 10*np.log10(x)
np.seterr(invalid='ignore', divide='ignore')

# ================= geometry (identical to final_checks.py) =================
txt = open(LED, 'rb').read(20000).decode('ascii', 'ignore')
k = txt.find('EXTRACTED CHIRP'); sp = txt[k-40:k].split()
DAZ, DR = float(sp[-2]), float(sp[-1]); AREF = DAZ*DR
t0, h, P, V = leader_file.read_platform_position_record(LED)
tf = img_file.read_line_time_ms(IMG, 0, NR); te = img_file.read_line_time_ms(IMG, NA-1, NR)
R0 = float(img_file.read_near_range_m(IMG, 0, NR))
Pm, Vm = geolocate.interpolate_position_velocity_vec(np.array([0.5*(tf+te)]), t0, h, P, V); Pm = Pm[0]; Vm = Vm[0]
la = np.arctan2(Pm[2], np.hypot(Pm[0], Pm[1])*(1-geolocate.WGS84_E2)); lo = np.arctan2(Pm[1], Pm[0])
HD = np.arctan2(Vm @ np.array([-np.sin(lo), np.cos(lo), 0]),
                Vm @ np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)]))
g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg0 = g['rg']
if not os.environ.get('RTC_NO_TIMEFIX'):   # fitted line-time origin (img_file.read_line_times_fit), identical to final_eq14.py
    tf_fit, te_fit, _dt = img_file.read_line_times_fit(IMG, NR, NA)
    az = ((tf + az*(te - tf)/(NA - 1)) - tf_fit)/(te_fit - tf_fit)*(NA - 1)
    say('[time] line-time origin %.7f s (stamp %.3f s): azimuth shift %+.3f lines' % (tf_fit, tf, -(tf_fit - tf)/_dt))
    tf, te = tf_fit, te_fit
AZ_SHIFT = float(os.environ.get('RTC_AZ_SHIFT', 4.17))   # azimuth timing calibration (+4.17 lines vs ASF RT1), identical to final_eq14.py; RTC_AZ_SHIFT=0 disables
if AZ_SHIFT:
    az = az + AZ_SHIFT; say('[az] constant azimuth shift %+.3f lines applied to all facets' % AZ_SHIFT)
coef = leader_file.read_six_coefficients(LED)
if GEOID_N:
    _d = -float(GEOID_N)*np.cos(geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0))/DR
    rg0 = rg0 + _d
    say('[geoid] N = %+.2f m -> range shift %+.2f to %+.2f px; outputs in %s' % (float(GEOID_N), _d.min(), _d.max(), OUT)); del _d
nd = int(round(np.sqrt(os.path.getsize(DEMFILE)/4)))
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(nd, nd)
coef = leader_file.read_six_coefficients(LED)
th_col = geometry.compute_theta_ih(coef, (R0 + np.arange(NR)*DR)/1000.0)
at, pd, rd, yd = leader_file.read_attitude_record(LED); o = np.argsort(at)
tt = tf + az/(NA-1)*(te-tf)
pitch = np.radians(np.interp(tt, at[o], pd[o])); yaw = np.radians(np.interp(tt, at[o], yd[o])); del tt
lat_rows = (LAT_TOP - rows/PPD).astype(np.float32)
dlat_m = np.float32(area_correction.DEG_M/PPD)
dlon_m = (area_correction.DEG_M/PPD*np.cos(np.radians(lat_rows))).astype(np.float32)
Z1 = dem[rows-1, cols-1]; Z2 = dem[rows-1, cols]; Z3 = dem[rows-1, cols+1]
Z4 = dem[rows,   cols-1];                         Z6 = dem[rows,   cols+1]
Z7 = dem[rows+1, cols-1]; Z8 = dem[rows+1, cols]; Z9 = dem[rows+1, cols+1]
p = (Z3+Z6+Z9-Z1-Z4-Z7)/(6.0*dlon_m); q = (Z1+Z2+Z3-Z7-Z8-Z9)/(6.0*dlat_m)
del Z1, Z2, Z3, Z4, Z6, Z7, Z8, Z9; gc.collect()
slope = np.arctan(np.sqrt(p**2 + q**2))
p_safe = np.where(p == 0, 1.0, p)
aspect = np.where(p == 0, np.where(q > 0, np.pi, 0.0), np.pi - np.arctan(q/p_safe) + (np.pi/2)*np.sign(p_safe))
tau_rho = -np.degrees(np.arctan(np.tan(slope)*np.cos(aspect - HD - np.pi/2)))
surf = (dlon_m*dlat_m*np.sqrt(1.0 + p**2 + q**2)).astype(np.float64)
del p, q, p_safe, aspect, slope, lat_rows, dlon_m; gc.collect()


nwa, nwr = NA//WA, NR//WR
th = geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0)
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
del area, pitch, yaw; gc.collect()
th_W = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]
A_flat = WA*WR*AREF/np.sin(th_W)
wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg0/WR).astype(np.int64)
inwin = (az >= 0) & (rg0 >= 0) & (wa < nwa) & (wr < nwr)
N_W = np.bincount((wa*nwr + wr)[inwin], minlength=nwa*nwr).reshape(nwa, nwr)
cover = binary_erosion(N_W > 0, structure=np.ones((3, 3)))
s_ = inwin & (tl < 90.0)
a_, r_ = az[s_], rg0[s_]; a0 = np.floor(a_).astype(np.int64); r0 = np.floor(r_).astype(np.int64); fa = a_ - a0; fr = r_ - r0
A = np.zeros(nwa*nwr); T = np.zeros(nwa*nwr)
for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
    pa = a0 + da; pr = r0 + dr; ok_ = (pa < nwa*WA) & (pr < nwr*WR)
    wid = np.where(ok_, (pa//WA)*nwr + pr//WR, 0); ww = np.where(ok_, w, 0.0)
    A += np.bincount(wid, weights=ww*surf[s_], minlength=nwa*nwr); T += np.bincount(wid, weights=ww*(surf*tl)[s_], minlength=nwa*nwr)
A = A.reshape(nwa, nwr); TL = T.reshape(nwa, nwr)/np.maximum(A, 1e-9)
okW = cover & (A > 0) & (TL >= INC_MIN) & (TL <= INC_MAX)
bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
bsW = bs[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)).astype(np.float64); del bs
before = np.where(cover, db(bsW/np.sqrt(np.cos(th_W))), np.nan)        # same quantity as after: gamma0, n = 0.5, theta_inc (no DEM)
after = np.where(okW, db(bsW*A_flat/np.maximum(A, 1e-9)/np.sqrt(np.cos(np.radians(TL)))), np.nan)
# rasterise on the DEM grid (1 arcsec), then 3 x 3 block mean (~3 arcsec)
f_ok = inwin
r0_, c0_ = rows[f_ok].min(), cols[f_ok].min(); H_ = rows[f_ok].max() - r0_ + 1; W_ = cols[f_ok].max() - c0_ + 1
def raster(Wv):
    img = np.full((H_, W_), np.nan, np.float32)
    img[rows[f_ok] - r0_, cols[f_ok] - c0_] = Wv[wa[f_ok], wr[f_ok]]
    h3, w3 = H_//3*3, W_//3*3
    return np.nanmean(img[:h3, :w3].reshape(h3//3, 3, w3//3, 3), axis=(1, 3))
import warnings; warnings.simplefilter('ignore')
B = raster(before); Aft = raster(after)
ext = [LON_LEFT + c0_/PPD, LON_LEFT + (c0_ + W_//3*3)/PPD, LAT_TOP - (r0_ + H_//3*3)/PPD, LAT_TOP - r0_/PPD]
np.savez_compressed(os.path.join(OUT, 'poster_maps_eq14.npz'), before=B, after=Aft, extent=np.array(ext))
both = np.isfinite(B) & np.isfinite(Aft)
say('[maps] grid %d x %d (~3 arcsec); before median %.2f std %.2f | after (Eq.14, n=0.5) median %.2f std %.2f dB'
    % (B.shape[0], B.shape[1], np.nanmedian(B[both]), np.nanstd(B[both]), np.nanmedian(Aft[both]), np.nanstd(Aft[both])))
ZB = [-71.33, -71.17, 44.28, 44.44]          # zoom box (lon0, lon1, lat0, lat1)
vmin, vmax = -19, -9
fig, axs = plt.subplots(2, 2, figsize=(11, 10.5), gridspec_kw=dict(height_ratios=[1, 1]))
for a, im, t in [(axs[0, 0], B, '(a) Before correction: HV $\\gamma^0$, $n$ = 0.5 ($\\theta_{inc}$, no DEM)'), (axs[0, 1], Aft, '(b) After correction: window integration, $n$ = 0.5')]:
    m = a.imshow(im, extent=ext, origin='upper', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    a.add_patch(plt.Rectangle((ZB[0], ZB[2]), ZB[1]-ZB[0], ZB[3]-ZB[2], fill=False, ec='red', lw=1.5))
    a.set_title(t, fontsize=11); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); a.set_aspect(1/np.cos(np.radians(44.5)))
for a, im, t in [(axs[1, 0], B, '(c) Before correction, area in the red frame'), (axs[1, 1], Aft, '(d) After correction, area in the red frame')]:
    a.imshow(im, extent=ext, origin='upper', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_xlim(ZB[0], ZB[1]); a.set_ylim(ZB[2], ZB[3]); a.set_title(t, fontsize=11)
    a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); a.set_aspect(1/np.cos(np.radians(44.5)))
cb = fig.colorbar(m, ax=axs, shrink=0.6, pad=0.02); cb.set_label('HV $\\gamma^0$, $n$ = 0.5, 100 m (dB)')
png = os.path.join(OUT, 'poster_maps_eq14%s.png' % _SFX); plt.savefig(png, dpi=200, facecolor='white', bbox_inches='tight')
say('[saved] %s' % png); LOG.close()
