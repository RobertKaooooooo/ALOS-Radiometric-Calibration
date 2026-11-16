"""
rtc_window_compare.py

Purpose (follow-up to the 10/26 advisor meeting):
  1. Current method (area_correction.compute_gamma0_ratio, HPA-like):
     each single-look radar pixel gets the IDW-averaged area of ~1 DEM facet,
     then the 31 x 7 moving window and the 100 m CCI cells.
     This reproduces the abstract / poster pipeline (history_recovered.txt).
  2. New method "window integration" (Simard 2016 Sec. II-B idea, done on a
     multilooked grid): for each non-overlapping 31 x 7 radar window
     (~100 m), SUM the true surface area of every illuminated DEM facet
     that maps into it:  A_W = sum( dx*dy*sqrt(1+p^2+q^2) )
     flat reference:     A_flat = 31*7*A_ref / sin(theta_inc)
     sigma0_terrain = sigma0_flat * A_flat / A_W
     gamma0         = sigma0_terrain / cos(theta_loc_W)^n ,  n = 0, 0.5, 1
     theta_loc_W = area-weighted mean theta_loc of the facets in the window.
     Facets with theta_loc >= 90 deg (shadow, not illuminated) are excluded;
     layover facets are kept and simply add up, which is the physically
     correct behaviour (several slopes in one window).
  3. Writes a traceable 100 m cell file (replaces summer_hv_cells.npz):
     ctl = mean theta_loc of the facets in the cell (definition fixed here).
  4. Counts facets in the layover region (tau_rho >= theta_inc) that the
     current code still marks valid.

Scene : ALPSRP237630880 (11 Jul 2010, HV)
Inputs: same as fig_slope_vs_correction.py and history_recovered.txt
Run   : cd "/mnt/c/grd file" && python3 scripts/rtc_window_compare.py
Output: scripts/out/rtc_window_compare_log.txt
        scripts/out/rtc_window_compare.png
        scripts/out/cells_100m_HV_20100711.npz
"""

import os, sys, gc
import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion

BASE = '/mnt/c/grd file'
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geometry, geolocate, area_correction  # noqa: E402
import rasterio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ---------------- inputs ----------------
D    = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID  = 'ALPSRP237630880'
LED  = D + 'LED-' + SID + '-H1.1__A'
IMG  = D + 'IMG-HH-' + SID + '-H1.1__A'          # used only for line times / near range (as in history)
NA, NR = 18432, 4704
BSFILE  = os.path.join(BASE, 'ALOS_HV_uncorrected_new.bin')   # sigma0 (flat), linear, float32, NA x NR
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
CCIFILE = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
WA, WR = 31, 7                    # window: azimuth lines x range pixels (as in history_recovered.txt)
INC_MIN, INC_MAX = 15.0, 70.0     # same mask as the current code
OUT = os.path.join(BASE, 'scripts', 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'rtc_window_compare_log.txt'), 'w')
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
say('[input] %s  A_ref %.3f m2  heading %.3f deg  window %d x %d' % (SID, AREF, np.degrees(HD), WA, WR))

g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg = g['rg']
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(7200, 7200)
coef = leader_file.read_six_coefficients(LED)
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
tau_rho = -np.degrees(np.arctan(slope_r))                     # + = facing the radar
surf = (dlon_m*dlat_m*np.sqrt(1.0 + p**2 + q**2)).astype(np.float64)   # true facet surface area (m2)
del p, q, p_safe, aspect, slope, slope_r, lat_rows, dlon_m; gc.collect()

# ---------------- layover count for the current code ----------------
lay = tau_rho >= np.degrees(th)
say('[layover] facets with tau_rho >= theta_inc: %d (%.3f%% of all facets)' % (lay.sum(), 100*lay.mean()))

# ---------------- method 1: current code ----------------
ratio, valid = area_correction.compute_gamma0_ratio(area, tl, az, rg, NA, NR, th_col, AREF)
del area; gc.collect()
ai = np.clip(np.round(az).astype(int), 0, NA-1); ri = np.clip(np.round(rg).astype(int), 0, NR-1)
say('[layover] of those, valid in the current code: %d' % valid[ai[lay], ri[lay]].sum())
bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
vf = valid.astype(np.float32); del valid
gam = bs*ratio*vf; del ratio; gc.collect()
d_old = uniform_filter(vf, (WA, WR))[ai, ri].astype(float); del vf
g_old = uniform_filter(gam, (WA, WR))[ai, ri].astype(float); del gam; gc.collect()
u_all = uniform_filter(bs, (WA, WR))[ai, ri].astype(float)          # uncorrected, all pixels
g_old = g_old/np.maximum(d_old, 1e-6)
ok_old = d_old > 1e-3

# ---------------- method 2: window integration ----------------
nwa, nwr = NA//WA, NR//WR
wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg/WR).astype(np.int64)
inwin = (az >= 0) & (rg >= 0) & (wa < nwa) & (wr < nwr)
lit = inwin & (tl < 90.0)
wid = wa*nwr + wr
A_W   = np.bincount(wid[lit], weights=surf[lit], minlength=nwa*nwr).reshape(nwa, nwr)
TL_W  = (np.bincount(wid[lit], weights=(surf*tl)[lit], minlength=nwa*nwr).reshape(nwa, nwr)
         / np.maximum(A_W, 1e-9))
N_W   = np.bincount(wid[inwin], minlength=nwa*nwr).reshape(nwa, nwr)
cover = binary_erosion(N_W > 0, structure=np.ones((3, 3)))       # drop windows at the footprint edge
th_W  = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]    # theta_inc per window column (rad)
A_flat = WA*WR*AREF/np.sin(th_W)
okW = cover & (A_W > 0) & (TL_W >= INC_MIN) & (TL_W <= INC_MAX)
bsW = bs[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)).astype(np.float64)
del bs; gc.collect()
area_fac = np.where(okW, A_flat/np.maximum(A_W, 1e-9), np.nan)
cosW = np.cos(np.radians(TL_W))
say('[window] windows %d x %d | valid %.1f%% | facets per window median %d (p5 %d, p95 %d) | A_W/A_flat median %.3f'
    % (nwa, nwr, 100*okW.mean(), np.median(N_W[cover]), np.percentile(N_W[cover], 5), np.percentile(N_W[cover], 95),
       np.nanmedian(1/area_fac)))
win_area = np.where(okW, bsW*area_fac, np.nan)                      # sigma0 terrain (n = 0)
win = {n: win_area/np.where(okW, cosW**n, np.nan) for n in (0.0, 0.5, 1.0)}

# sample window values at the facets
wa_c = np.clip(wa, 0, nwa-1); wr_c = np.clip(wr, 0, nwr-1)
ok_new = inwin & okW[wa_c, wr_c]
new_f = {n: v[wa_c, wr_c] for n, v in win.items()}
unc_f = bsW[wa_c, wr_c]

# ---------------- 100 m CCI cells ----------------
src = rasterio.open(CCIFILE); BIO = src.read(1); T = src.transform
say('[cci] %s  crs %s  res %.6f deg  tags %s' % (os.path.basename(CCIFILE), src.crs, T.a, dict(src.tags())))
br = ((LAT_TOP - rows/PPD - T.f)/T.e).astype(int); bc = ((LON_LEFT + cols/PPD - T.c)/T.a).astype(int)
bio = BIO[br, bc].astype(float)
ok = ok_old & ok_new & (bio > 0) & (bio < 600) & (u_all > 0) & (g_old > 0) & np.isfinite(new_f[1.0]) & (new_f[1.0] > 0)
say('[cells] facets used (valid in BOTH methods, 0 < AGB < 600): %d' % ok.sum())
key = br[ok]*BIO.shape[1] + bc[ok]
uu, inv = np.unique(key, return_inverse=True); nf = np.bincount(inv); kk = nf >= 5
cb = BIO.ravel()[uu].astype(float)[kk]
cmean = lambda v: (np.bincount(inv, weights=v[ok])/nf)[kk]
ctl = cmean(tl)                                   # definition: mean theta_loc of the facets in the cell
cells = {'cb': cb, 'ctl': ctl,
         'unc (sigma0 flat)':           10*np.log10(cmean(u_all)),
         'old gamma0 (current code)':   10*np.log10(cmean(g_old)),
         'new sigma0 terrain (n=0)':    10*np.log10(cmean(np.nan_to_num(new_f[0.0]))),
         'new gamma0 n=0.5':            10*np.log10(cmean(np.nan_to_num(new_f[0.5]))),
         'new gamma0 n=1':              10*np.log10(cmean(np.nan_to_num(new_f[1.0])))}
np.savez(os.path.join(OUT, 'cells_100m_HV_20100711.npz'), cb=cb, ctl=ctl, nfacet=nf[kk],
         cu=cells['unc (sigma0 flat)'], cg_old=cells['old gamma0 (current code)'],
         cs_new=cells['new sigma0 terrain (n=0)'], cg_new05=cells['new gamma0 n=0.5'], cg_new1=cells['new gamma0 n=1'])

# ---------------- statistics ----------------
m15 = (ctl >= INC_MIN) & (ctl <= INC_MAX)
say('\n[table] 100 m cells: %d (theta 15-70: %d). Regression on all cells; angle terms on theta 15-70 cells.' % (len(cb), m15.sum()))
say('%-28s %8s %8s %8s %10s %9s %7s %10s %13s' % ('quantity', 'median', 'std', 'r(AGB)', 'slope', 'intercept', 'RMSE', 'r(dB,th)', 'resid dB/deg'))
for name in ['unc (sigma0 flat)', 'old gamma0 (current code)', 'new sigma0 terrain (n=0)', 'new gamma0 n=0.5', 'new gamma0 n=1']:
    y = cells[name]; b = np.polyfit(cb, y, 1); r = y - np.polyval(b, cb)
    say('%-28s %+8.2f %8.2f %8.3f %10.5f %+9.3f %7.3f %+10.3f %+13.4f' % (
        name, np.median(y[m15]), y[m15].std(), np.corrcoef(cb, y)[0, 1], b[0], b[1], np.sqrt(np.mean(r**2)),
        np.corrcoef(y[m15], ctl[m15])[0, 1], np.polyfit(ctl[m15], r[m15], 1)[0]))
say('(r(AGB) with linear power: unc %.3f | old %.3f | new n=0.5 %.3f)' % tuple(
    np.corrcoef(cb, 10**(cells[n]/10))[0, 1] for n in ['unc (sigma0 flat)', 'old gamma0 (current code)', 'new gamma0 n=0.5']))

# ---------------- figure ----------------
tb = np.arange(15, 71, 5); tc = 0.5*(tb[:-1] + tb[1:]); ti = np.digitize(ctl, tb) - 1
bb = np.arange(0, 241, 20); bcn = 0.5*(bb[:-1] + bb[1:]); bi = np.digitize(cb, bb) - 1
sty = {'unc (sigma0 flat)': ('0.45', '-'), 'old gamma0 (current code)': ('#1f77b4', '-'),
       'new sigma0 terrain (n=0)': ('#2ca02c', '--'), 'new gamma0 n=0.5': ('#d62728', '-'), 'new gamma0 n=1': ('#9467bd', ':')}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for name, (c, ls) in sty.items():
    y = cells[name]
    axs[0].plot(tc, [np.median(y[ti == i]) if (ti == i).sum() > 50 else np.nan for i in range(len(tc))], color=c, ls=ls, lw=1.8, label=name)
    axs[1].plot(bcn, [y[bi == i].mean() if (bi == i).sum() > 50 else np.nan for i in range(len(bcn))], color=c, ls=ls, lw=1.8,
                label='%s (r = %.2f)' % (name, np.corrcoef(cb, y)[0, 1]))
axs[0].set_xlabel('Cell-mean local incidence angle $\\theta_{loc}$ (deg)'); axs[0].set_ylabel('Median backscatter (dB)')
axs[0].set_title('(a) Terrain dependence, 100 m cells', loc='left')
axs[1].set_xlabel('CCI AGB (Mg/ha)'); axs[1].set_ylabel('Mean backscatter (dB)')
axs[1].set_title('(b) Biomass dependence, 100 m cells', loc='left')
for a in axs: a.legend(fontsize=8, frameon=False)
fig.suptitle('%s HV: current code (single-look, ~1 facet per pixel) vs window integration (%d x %d, sum of facet areas)' % (SID, WA, WR), fontsize=10)
plt.tight_layout()
png = os.path.join(OUT, 'rtc_window_compare.png'); plt.savefig(png, dpi=200, facecolor='white')
say('\n[saved] %s' % png); LOG.close()
