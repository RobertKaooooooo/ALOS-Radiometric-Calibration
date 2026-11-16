"""poster_fig4.py -- poster version of section 4: area correction (sigma0_terrain - sigma0_flat), 5.99 M GLO-30 facets
Run: cd "/mnt/c/grd file" && RTC_GEOID_SHIFT=-28 python3 scripts/poster_fig4.py
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
LOG = open(os.path.join(OUT, 'poster_fig4_log.txt'), 'w')
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
    print('[geoid] N = %+.2f m -> range shift %+.2f to %+.2f px; outputs in %s' % (float(GEOID_N), _d.min(), _d.max(), OUT)); del _d
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


th = geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0)
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
thd = np.degrees(th); lit = tl < 90.0
AC = 10*np.log10(AREF/(np.sin(th)*area))                      # sigma0_terrain - sigma0_flat (dB), area term only
flat = lit & (np.abs(tau_rho) < 1.0)
say('[fig4] lit facets %d | flat (|tau_rho|<1) AC mean %+.3f dB | theta_inc %.1f-%.1f' % (lit.sum(), AC[flat].mean(), thd.min(), thd.max()))
for t in (-20, 20):
    m = lit & (np.abs(tau_rho - t) < 1)
    say('[fig4] tau_rho %+d +-1 deg: AC median %+.2f dB (theory at mean theta_inc %+.2f)' % (t, np.median(AC[m]), 10*np.log10(np.sin(np.radians(thd.mean()-t))/np.sin(np.radians(thd.mean())))))
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
tim = thd.mean(); YL = (-10, 4)
m = lit & np.isfinite(AC)
a = axs[0]
a.hexbin(tau_rho[m], AC[m], gridsize=120, extent=(-40, 40, YL[0], YL[1]), bins='log', cmap='Greys', mincnt=1)
tr = np.linspace(-40, 36, 300)
a.plot(tr, 10*np.log10(np.sin(np.radians(tim - tr))/np.sin(np.radians(tim))), color='#d62728', lw=2.5, label='theory, $\\theta_{inc}$ = %.1f°' % tim)
e = np.arange(-40, 41, 2.0); bi = np.digitize(tau_rho[m], e) - 1; acm = AC[m]
med = [np.median(acm[bi == i]) if (bi == i).sum() > 30 else np.nan for i in range(len(e) - 1)]
a.plot(0.5*(e[:-1] + e[1:]), med, 'o', color='#ff7f0e', ms=6, label='data median')
a.axhline(0, color='k', lw=0.8); a.axvline(0, color='k', lw=0.5, ls=':')
for t, txt, xy in ([] if _TH else [(0, 'flat ground: 0 dB', (6, 2.3)), (20, 'facing the radar\n20°: −2.9 dB → darkened', (-12, -6.0)), (-20, 'facing away\n20°: +1.4 dB → brightened', (-37, -3.2))]):
    v = 10*np.log10(np.sin(np.radians(tim - t))/np.sin(np.radians(tim)))
    a.annotate(txt, (t, v), xytext=xy, fontsize=13, arrowprops=dict(arrowstyle='->', lw=1.2))
a.set_xlim(-40, 40); a.set_ylim(*YL)
a.set_xlabel('Range slope $\\tau_\\rho$ (°), positive towards the radar'); a.set_ylabel('Area correction (dB)')
a.set_title('(a) Area correction versus range slope, $\\theta_{inc}$ = %.1f°' % tim, loc='left'); a.legend(loc='lower left', fontsize=13, frameon=False)
a = axs[1]
a.hexbin(tl[m], AC[m], gridsize=120, extent=(10, 75, YL[0], YL[1]), bins='log', cmap='Greys', mincnt=1)
tt = np.linspace(10, 75, 300)
for t_, c in [(thd.min(), '#1f77b4'), (thd.max(), '#2ca02c')]:
    a.plot(tt, 10*np.log10(np.sin(np.radians(tt))/np.sin(np.radians(t_))), color=c, lw=2.5, label='theory, $\\theta_{inc}$ = %.1f°' % t_)
a.axhline(0, color='k', lw=0.8); a.axvspan(thd.min(), thd.max(), color='#fde68a', alpha=0.5)
a.set_xlim(10, 75); a.set_ylim(*YL)
a.set_xlabel('Local incidence angle $\\theta_{loc}$ (°)'); a.set_ylabel('Area correction (dB)')
a.set_title('(b) Area correction versus local incidence angle', loc='left', fontsize=14)
ti = thd.max()                                                  # examples on the green line
for t_ in (thd.min(), thd.max()):
    a.plot(t_, 0, 'o', ms=11, mfc='white', mec='k', mew=2, zorder=5)
if not _TH: a.annotate('flat ground:\n$\\theta_{loc}$ = $\\theta_{inc}$ $\\rightarrow$ 0 dB', (ti, 0), xytext=(47, 2.6), fontsize=13, arrowprops=dict(arrowstyle='->', lw=1.4))
for tl_, xy in [(10.0, (13, -9.3)), (60.0, (51, -3.9))]:
    v = 10*np.log10(np.sin(np.radians(tl_))/np.sin(np.radians(ti)))
    sl_ = ti - tl_
    lab = '$\\theta_{loc}$ %d°: %.0f° slope\n%s $\\rightarrow$ %+.1f dB' % (tl_, abs(sl_), 'facing the radar' if sl_ > 0 else 'facing away', v)
    a.plot(tl_, v, 'o', ms=9, color='#2ca02c', mec='k', zorder=5)
    if not _TH: a.annotate(lab, (tl_, v), xytext=xy, fontsize=12.5, arrowprops=dict(arrowstyle='->', lw=1.4))
    say('[fig4] example on theta_inc %.1f line: theta_loc %.0f -> slope %.1f deg, %+.2f dB' % (ti, tl_, sl_, v)); a.legend(loc='upper left', fontsize=13, frameon=False)
plt.tight_layout()
png = os.path.join(OUT, 'poster_fig4%s.png' % _SFX); plt.savefig(png, dpi=220, facecolor='white')
say('[saved] %s' % png); LOG.close()
