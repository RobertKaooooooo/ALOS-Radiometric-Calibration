"""
angle_inc_vs_slope.py

Purpose (advisor question, 10/26): is the correction / analysis driven by the
local incidence angle theta_loc, or by the flat-ground incidence angle
theta_inc?  Separate the two variables instead of binning by theta_loc only.

  1. Distribution of theta_inc (all facets, flat facets)
  2. Correction vs range slope tau_rho, coloured by theta_inc
  3. Table: mean correction by (tau_rho bin x theta_inc bin)
  4. How much of the theta_loc variation comes from theta_inc vs slope

Per-facet correction = what a radar pixel receives from one facet in the
current code (area_correction.compute_gamma0_ratio):
  corr = 10log10( (1/sin theta_inc) * A_ref / A_facet ) - 10log10(cos theta_loc)
       = area term (DEM normal, projection angle) + cos term (theta_loc)

Scene : ALPSRP237630880 (11 Jul 2010, HV)
Inputs: same as fig_slope_vs_correction.py
Run   : cd "/mnt/c/grd file" && python3 scripts/angle_inc_vs_slope.py
Output: scripts/out/angle_inc_vs_slope.png
        scripts/out/angle_inc_vs_slope_log.txt
        scripts/out/angle_inc_vs_slope_table.csv
"""

import os, sys, gc
import numpy as np

BASE = '/mnt/c/grd file'
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geometry, geolocate, area_correction  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D    = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID  = 'ALPSRP237630880'
LED  = D + 'LED-' + SID + '-H1.1__A'
IMG  = D + 'IMG-HH-' + SID + '-H1.1__A'
NA, NR = 18432, 4704
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
PPD, LAT_TOP = 3600, 45.0
FLAT_SLOPE_DEG = 3.0
GEOID_N = os.environ.get('RTC_GEOID_SHIFT')   # EGM2008 undulation (m): range shift -N*cos(theta_inc)/DR
OUT = os.path.join(BASE, 'scripts', 'out_ellip' if GEOID_N else 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'angle_inc_vs_slope_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()

# ---------------- geometry (identical to fig_slope_vs_correction.py) ----------------
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
g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg = g['rg']
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(7200, 7200)
coef = leader_file.read_six_coefficients(LED)
if GEOID_N:
    _d = -float(GEOID_N)*np.cos(geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0))/DR
    rg = rg + _d
    print('[geoid] N = %+.2f m -> range shift %+.2f to %+.2f px; outputs in %s' % (float(GEOID_N), _d.min(), _d.max(), OUT)); del _d
th = geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0)
th_col = geometry.compute_theta_ih(coef, (R0 + np.arange(NR)*DR)/1000.0)
tt = tf + az/(NA-1)*(te-tf)
at, pd, rd, yd = leader_file.read_attitude_record(LED); o = np.argsort(at)
pitch = np.radians(np.interp(tt, at[o], pd[o])); yaw = np.radians(np.interp(tt, at[o], yd[o]))
del tt; gc.collect()
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
del pitch, yaw; gc.collect()

# slope terms, copied from area_correction.py lines 65-81
lat_rows = (LAT_TOP - rows/PPD).astype(np.float32)
dlat_m = np.float32(area_correction.DEG_M/PPD)
dlon_m = (area_correction.DEG_M/PPD*np.cos(np.radians(lat_rows))).astype(np.float32)
Z1 = dem[rows-1, cols-1]; Z2 = dem[rows-1, cols]; Z3 = dem[rows-1, cols+1]
Z4 = dem[rows,   cols-1];                         Z6 = dem[rows,   cols+1]
Z7 = dem[rows+1, cols-1]; Z8 = dem[rows+1, cols]; Z9 = dem[rows+1, cols+1]
p = (Z3+Z6+Z9-Z1-Z4-Z7)/(6.0*dlon_m); q = (Z1+Z2+Z3-Z7-Z8-Z9)/(6.0*dlat_m)
del Z1, Z2, Z3, Z4, Z6, Z7, Z8, Z9, dem; gc.collect()
slope = np.arctan(np.sqrt(p**2 + q**2))
p_safe = np.where(p == 0, 1.0, p)
aspect = np.where(p == 0, np.where(q > 0, np.pi, 0.0), np.pi - np.arctan(q/p_safe) + (np.pi/2)*np.sign(p_safe))
slope_r = np.tan(slope)*np.cos(aspect - HD - np.pi/2)
slope_a = np.tan(slope)*np.cos(aspect - HD)
S_deg   = np.degrees(slope)
tau_rho = -np.degrees(np.arctan(slope_r))     # + = facing the radar
tau_s   = np.degrees(np.arctan(slope_a))
del p, q, p_safe, aspect, slope, slope_r, slope_a, lat_rows, dlon_m; gc.collect()

inc = np.degrees(th)                            # theta_inc per facet (deg)
lit = tl < 90.0
area_term = 10*np.log10((1.0/np.sin(th))*AREF/area)
cos_term  = np.where(lit, -10*np.log10(np.cos(np.radians(np.minimum(tl, 89.9)))), np.nan)
corr = area_term + cos_term
flat = S_deg < FLAT_SLOPE_DEG
del area; gc.collect()

# ---------------- 1. theta_inc distribution ----------------
say('[1] theta_inc across the image (LED polynomial, near to far range): %.2f - %.2f deg' % (np.degrees(th_col.min()), np.degrees(th_col.max())))
say('    theta_inc of facets: p1/p50/p99 = %s deg | flat facets (S<%g): p1/p50/p99 = %s deg'
    % (np.round(np.percentile(inc, [1, 50, 99]), 2), FLAT_SLOPE_DEG, np.round(np.percentile(inc[flat], [1, 50, 99]), 2)))
say('    theta_loc of facets: p1/p50/p99 = %s deg | flat facets: p1/p50/p99 = %s deg'
    % (np.round(np.nanpercentile(tl, [1, 50, 99]), 1), np.round(np.percentile(tl[flat], [1, 50, 99]), 1)))
say('    if flat ground had theta_inc = 10 deg (does not exist here): correction = area 0 dB + cos %+.2f dB'
    % (-10*np.log10(np.cos(np.radians(10)))))

# ---------------- 4. where does the theta_loc variation come from ----------------
m = lit
X = np.column_stack([inc[m], tau_rho[m], tau_s[m], np.ones(m.sum())])
b, *_ = np.linalg.lstsq(X, tl[m], rcond=None)
say('\n[4] theta_loc = %.3f*theta_inc %+.3f*tau_rho %+.3f*tau_s %+.2f   (least squares, lit facets)' % tuple(b))
say('    std: theta_loc %.2f | theta_inc %.2f | tau_rho %.2f | tau_s %.2f deg' % (tl[m].std(), inc[m].std(), tau_rho[m].std(), tau_s[m].std()))
for nm, x in [('theta_inc', inc), ('tau_rho', tau_rho), ('tau_s', tau_s), ('total slope S', S_deg)]:
    say('    r(theta_loc, %-13s) = %+.3f   r(correction, %-13s) = %+.3f' % (nm, np.corrcoef(tl[m], x[m])[0, 1], nm, np.corrcoef(corr[m], x[m])[0, 1]))
v_inc = (b[0]*inc[m]).var(); v_rho = (b[1]*tau_rho[m]).var(); v_s = (b[2]*tau_s[m]).var()
say('    variance of theta_loc explained: theta_inc %.1f%% | tau_rho %.1f%% | tau_s %.1f%% (of the three linear terms)'
    % tuple(100*np.array([v_inc, v_rho, v_s])/(v_inc + v_rho + v_s)))

# ---------------- 3. table: tau_rho x theta_inc ----------------
ie = np.arange(np.floor(np.percentile(inc, 0.5)), np.ceil(np.percentile(inc, 99.5)) + 1, 1.0)
re = np.arange(-30, 41, 5)
say('\n[3] mean correction (dB) by range slope tau_rho (rows, + facing radar) and theta_inc (columns); "-" = fewer than 200 facets')
hdr = 'tau_rho \\ inc ' + ''.join('%9s' % ('%d-%d' % (ie[j], ie[j+1])) for j in range(len(ie)-1))
say(hdr)
csv = ['tau_rho_lo,tau_rho_hi,' + ','.join('inc_%d_%d' % (ie[j], ie[j+1]) for j in range(len(ie)-1))]
ri_ = np.digitize(tau_rho, re) - 1; ii_ = np.digitize(inc, ie) - 1
for i in range(len(re)-1):
    cells = []
    for j in range(len(ie)-1):
        mm = lit & (ri_ == i) & (ii_ == j)
        cells.append(corr[mm].mean() if mm.sum() >= 200 else np.nan)
    say('%+4d..%+4d     ' % (re[i], re[i+1]) + ''.join('%9s' % ('-' if np.isnan(c) else '%+.2f' % c) for c in cells))
    csv.append('%d,%d,' % (re[i], re[i+1]) + ','.join('' if np.isnan(c) else '%.3f' % c for c in cells))
open(os.path.join(OUT, 'angle_inc_vs_slope_table.csv'), 'w').write('\n'.join(csv) + '\n')
mm = flat
say('\n    flat facets by theta_inc: ' + ' | '.join(
    '%d-%d: %+.2f dB (1/cos %+.2f)' % (ie[j], ie[j+1], corr[mm & (ii_ == j)].mean(), (-10*np.log10(np.cos(np.radians(inc[mm & (ii_ == j)])))).mean())
    for j in range(len(ie)-1) if (mm & (ii_ == j)).sum() >= 200))

# ---------------- figure ----------------
rng = np.random.default_rng(0)
idx = np.flatnonzero(lit); idx = rng.choice(idx, min(80000, idx.size), replace=False)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1, 1.25, 1.1]})
a = axs[0]; hb = np.arange(ie[0] - 1, ie[-1] + 1.01, 0.2)
a.hist(inc, hb, density=True, color='0.55', alpha=0.7, label='all facets (n = %d)' % inc.size)
a.hist(inc[flat], hb, density=True, histtype='step', color='#d62728', lw=1.6, label='flat facets, S < %g$\\degree$ (n = %d)' % (FLAT_SLOPE_DEG, flat.sum()))
a.set_xlabel('$\\theta_{inc}$ (deg)'); a.set_ylabel('Density'); a.legend(fontsize=8.5, frameon=False)
a.set_title('(a) Flat-ground incidence $\\theta_{inc}$ in this scene', loc='left', fontsize=11)
a = axs[1]
sc = a.scatter(tau_rho[idx], corr[idx], c=inc[idx], s=2, cmap='viridis', lw=0, vmin=ie[0], vmax=ie[-1])
cb = fig.colorbar(sc, ax=a); cb.set_label('$\\theta_{inc}$ (deg)')
a.axhline(0, color='k', lw=0.6); a.set_ylim(-16, 8); a.set_xlim(-40, 45)
a.set_xlabel('Range slope $\\tau_\\rho$ (deg)  [+ = facing the radar]'); a.set_ylabel('Correction $\\gamma^0-\\sigma^0$ (dB)')
a.set_title('(b) Correction vs range slope, colour = $\\theta_{inc}$', loc='left', fontsize=11)
a = axs[2]; cmap = plt.get_cmap('viridis')
for j in range(len(ie)-1):
    sel = lit & (ii_ == j); med = []
    for i in range(len(re)-1):
        mm = sel & (ri_ == i); med.append(np.median(corr[mm]) if mm.sum() >= 200 else np.nan)
    a.plot(0.5*(re[:-1] + re[1:]), med, '-o', ms=3, color=cmap((j + 0.5)/(len(ie)-1)), label='$\\theta_{inc}$ %d-%d$\\degree$' % (ie[j], ie[j+1]))
a.axhline(0, color='k', lw=0.6); a.set_ylim(-16, 8)
a.set_xlabel('Range slope $\\tau_\\rho$ (deg)'); a.set_ylabel('Median correction (dB)')
a.set_title('(c) Median correction per $\\theta_{inc}$ class', loc='left', fontsize=11)
a.legend(fontsize=8, frameon=False)
fig.suptitle('%s: separating flat-ground incidence $\\theta_{inc}$ from terrain slope (one DEM facet per point)' % SID, fontsize=11)
plt.tight_layout()
png = os.path.join(OUT, 'angle_inc_vs_slope.png'); plt.savefig(png, dpi=200, facecolor='white')
say('\n[saved] %s' % png); LOG.close()
