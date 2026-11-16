"""
final_checks.py  --  remaining figures/tables for the 10/26 advisor questions

  F1  Regression (advisor Q1): CCI 2010 AGB vs backscatter, scatter + line + formula,
      for (a) uncorrected sigma0, (b) current code, (c) window integration n = 0.5
  F2  Like-for-like distributions (advisor Q11):
      (a) sigma0 vs sigma0 : flat-ellipsoid sigma0 (before)  vs  terrain sigma0 (window integration, area only)
      (b) gamma0 vs gamma0 : paper definition gamma0 = sigma0 / cos(theta_inc) (Simard Eq. 15 text),
          before (ellipsoid) vs after (terrain)
  F3  Area-method comparison, analogue of Simard Fig. 4 (advisor Q3b, Q8):
      per 31 x 7 window, area factor of
        current code (single-look, IDW of ~1 facet, then window mean)
        local-incidence method, Simard Eq. (5): sin(theta_loc)/sin(theta_inc)
      minus the window-integration reference (sum of facet areas), in dB
  F4  Residual angle trend, analogue of Simard Fig. 10 (advisor Q9):
      in 1-deg theta_inc bins, slope of backscatter (dB) vs theta_loc, 100 m cells

All numbers are written to scripts/out/final_checks_log.txt.

Definitions (same as rtc_window_compare.py):
  sigma0_flat           : JAXA L1.1 sigma0 (flat ellipsoid), input file
  current code ratio    : area_correction.py lines 224-229, rebuilt here from the same
                          accumulate_area_rdc() calls so the area part can be separated
  window integration    : A_W = sum over lit facets in the window of dx*dy*sqrt(1+p^2+q^2)
                          A_flat = 31*7*A_ref/sin(theta_inc);  sigma0_terrain = sigma0_flat*A_flat/A_W
  gamma0 (code, n)      : sigma0_terrain / cos(theta_loc)^n
  gamma0 (paper Eq. 15) : sigma0 / cos(theta_inc)

Scene : ALPSRP237630880 (11 Jul 2010, HV);  CCI AGB v7.0, year 2010
Run   : cd "/mnt/c/grd file" && python3 scripts/final_checks.py
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

D    = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID  = 'ALPSRP237630880'
LED  = D + 'LED-' + SID + '-H1.1__A'
IMG  = D + 'IMG-HH-' + SID + '-H1.1__A'
NA, NR = 18432, 4704
BSFILE  = os.path.join(BASE, 'ALOS_HV_uncorrected_new.bin')
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
CCIFILE = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
WA, WR = 31, 7
INC_MIN, INC_MAX = 15.0, 70.0
GEOID_N = os.environ.get('RTC_GEOID_SHIFT')   # EGM2008 undulation (m): range shift -N*cos(theta_inc)/DR
OUT = os.path.join(BASE, 'scripts', 'out_ellip' if GEOID_N else 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'fig4_m2_log.txt' if os.environ.get('RTC_FIG4M2') else 'final_checks_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()
db = lambda x: 10*np.log10(x)

# ---------------- geometry (identical to the other scripts) ----------------
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
lat_rows = (LAT_TOP - rows/PPD).astype(np.float32)
dlat_m = np.float32(area_correction.DEG_M/PPD)
dlon_m = (area_correction.DEG_M/PPD*np.cos(np.radians(lat_rows))).astype(np.float32)
Z1 = dem[rows-1, cols-1]; Z2 = dem[rows-1, cols]; Z3 = dem[rows-1, cols+1]
Z4 = dem[rows,   cols-1];                         Z6 = dem[rows,   cols+1]
Z7 = dem[rows+1, cols-1]; Z8 = dem[rows+1, cols]; Z9 = dem[rows+1, cols+1]
p = (Z3+Z6+Z9-Z1-Z4-Z7)/(6.0*dlon_m); q = (Z1+Z2+Z3-Z7-Z8-Z9)/(6.0*dlat_m)
del Z1, Z2, Z3, Z4, Z6, Z7, Z8, Z9, dem; gc.collect()
surf = (dlon_m*dlat_m*np.sqrt(1.0 + p**2 + q**2)).astype(np.float64)
del p, q, lat_rows, dlon_m; gc.collect()
say('[input] %s  A_ref %.3f m2  heading %.3f deg  window %d x %d  CCI %s' % (SID, AREF, np.degrees(HD), WA, WR, os.path.basename(CCIFILE)))

# ---------------- current code, rebuilt (area_correction.py lines 224-229) ----------------
aR, fill = area_correction.accumulate_area_rdc(area, az, rg, NA, NR, outlier_factor=1e30)
aT, _    = area_correction.accumulate_area_rdc(np.radians(tl), az, rg, NA, NR, outlier_factor=1e30)
valid = fill & (aT <= np.radians(INC_MAX)) & (aT >= np.radians(INC_MIN))
area_fe = 1.0/np.sin(th_col.astype(np.float64))
afac_px = (area_fe[None, :]*(AREF/np.where(fill, aR, 1.0)))                     # area part only
ratio = np.clip(afac_px/np.cos(aT), 0.001, 1000.0).astype(np.float32)          # == compute_gamma0_ratio
del aR, aT, area; gc.collect()
vf = valid.astype(np.float32)
# window means of the current code's area factor (over valid pixels)
afac_px = np.where(valid, afac_px, 0.0).astype(np.float32)
nwa, nwr = NA//WA, NR//WR
blk = lambda a: a[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).sum(axis=(1, 3)).astype(np.float64)
cnt_W = blk(vf); cur_afac_W = blk(afac_px)/np.maximum(cnt_W, 1); del afac_px; gc.collect()

bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
ai = np.clip(np.round(az).astype(int), 0, NA-1); ri = np.clip(np.round(rg).astype(int), 0, NR-1)
gam = bs*ratio*vf; del ratio; gc.collect()
d_old = uniform_filter(vf, (WA, WR))[ai, ri].astype(float)
g_old = uniform_filter(gam, (WA, WR))[ai, ri].astype(float)/np.maximum(d_old, 1e-6); del gam; gc.collect()
u_all = uniform_filter(bs, (WA, WR))[ai, ri].astype(float)
del valid, vf; gc.collect()

# ---------------- window integration ----------------
wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg/WR).astype(np.int64)
inwin = (az >= 0) & (rg >= 0) & (wa < nwa) & (wr < nwr)
lit = inwin & (tl < 90.0)
wid = wa*nwr + wr
# Simard Eq. 14: each facet split over its 4 neighbouring radar pixels, then each pixel to its window
A_W = np.zeros(nwa*nwr); T_ = np.zeros(nwa*nwr); E_ = np.zeros(nwa*nwr); C_ = np.zeros(nwa*nwr)
a_, r_ = az[lit], rg[lit]; a0 = np.floor(a_).astype(np.int64); r0 = np.floor(r_).astype(np.int64); fa = a_ - a0; fr = r_ - r0
e5 = (np.sin(np.radians(tl))/np.sin(th))[lit]
for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
    pa = a0 + da; pr = r0 + dr; ok_ = (pa < nwa*WA) & (pr < nwr*WR)
    wid2 = np.where(ok_, (pa//WA)*nwr + pr//WR, 0); ww = np.where(ok_, w, 0.0)
    A_W += np.bincount(wid2, weights=ww*surf[lit], minlength=nwa*nwr); T_ += np.bincount(wid2, weights=ww*(surf*tl)[lit], minlength=nwa*nwr)
    E_ += np.bincount(wid2, weights=ww*e5, minlength=nwa*nwr); C_ += np.bincount(wid2, weights=ww, minlength=nwa*nwr)
del a_, r_, a0, r0, fa, fr, e5
A_W = A_W.reshape(nwa, nwr); TL_W = T_.reshape(nwa, nwr)/np.maximum(A_W, 1e-9)
eq5 = E_.reshape(nwa, nwr)/np.maximum(C_.reshape(nwa, nwr), 1e-9)          # Simard Eq. 5 factor, facet mean (Eq. 14 weights)
N_W  = np.bincount(wid[inwin], minlength=nwa*nwr).reshape(nwa, nwr)
cover = binary_erosion(N_W > 0, structure=np.ones((3, 3)))
th_W = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]
A_flat = WA*WR*AREF/np.sin(th_W)
okW = cover & (A_W > 0) & (TL_W >= INC_MIN) & (TL_W <= INC_MAX) & (cnt_W > 0)
new_afac_W = A_flat/np.maximum(A_W, 1e-9)
bsW = bs[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)).astype(np.float64)
del bs; gc.collect()

# ---------------- F3: area factors per window (analogue of Simard Fig. 4) ----------------
dcur = db(cur_afac_W[okW]) - db(new_afac_W[okW])
deq5 = db(eq5[okW]) - db(new_afac_W[okW])
say('\n[F3] area factor minus window integration, %d windows (dB)' % okW.sum())
for nm, dd in [('current code (single-look IDW, ~1 facet/pixel)', dcur), ('local incidence, Simard Eq. 5', deq5)]:
    say('    %-48s mean %+.3f  median %+.3f  std %.3f  p5/p95 %+.2f/%+.2f  within +-0.5 dB %.1f%%  within +-1 dB %.1f%%'
        % (nm, dd.mean(), np.median(dd), dd.std(), np.percentile(dd, 5), np.percentile(dd, 95),
           100*np.mean(np.abs(dd) <= 0.5), 100*np.mean(np.abs(dd) <= 1.0)))
TLok = TL_W[okW]
say('    by window theta_loc (median difference, dB):')
for lo_, hi_ in [(15, 25), (25, 35), (35, 45), (45, 55), (55, 70)]:
    m = (TLok >= lo_) & (TLok < hi_)
    if m.sum() > 50:
        say('      %2d-%2d deg  n %6d  current %+.3f   Eq.5 %+.3f' % (lo_, hi_, m.sum(), np.median(dcur[m]), np.median(deq5[m])))

# ---------------- Simard Fig. 4 analogue in m^2 (RTC_FIG4M2=1): area per radar pixel, DEM integration minus method ----------------
if os.environ.get('RTC_FIG4M2'):
    import glob as _g
    from matplotlib import font_manager as _fm
    for _f in _g.glob(os.path.expanduser('~/.fonts/*.tt[fc]')): _fm.fontManager.addfont(_f)
    plt.rcParams.update({'font.family': ['TASA Explorer', 'DejaVu Sans'], 'mathtext.fontset': 'custom', 'mathtext.rm': 'Cambria',
                         'mathtext.it': 'Cambria:italic', 'mathtext.bf': 'Cambria:bold', 'mathtext.fallback': 'cm'})
    NPX = WA*WR
    A_int = A_W[okW]/NPX
    A_hpa = (A_flat/cur_afac_W)[okW]/NPX
    A_e5  = (A_flat/eq5)[okW]/NPX
    A_fl  = np.broadcast_to(A_flat, okW.shape)[okW]/NPX
    d_hpa = A_int - A_hpa; d_e5 = A_int - A_e5
    say('\n[Fig4-m2] area per radar pixel (m^2), %d windows of %d pixels; A_ref = %.2f m^2' % (okW.sum(), NPX, AREF))
    say('    flat-ground area A_ref/sin(theta_inc): median %.2f  (5-95%%: %.2f-%.2f)' % (np.median(A_fl), *np.percentile(A_fl, [5, 95])))
    say('    DEM integration A_W/%d:               median %.2f  (5-95%%: %.2f-%.2f)' % (NPX, np.median(A_int), *np.percentile(A_int, [5, 95])))
    for nm, dd in [('integration minus projection angle (current code)', d_hpa), ('integration minus local incidence (Eq. 5)', d_e5)]:
        say('    %-52s mean %+.2f  median %+.2f  std %.2f  p5/p95 %+.2f/%+.2f  |d|<=2 m2 %.1f%%  |d|<=5 m2 %.1f%%'
            % (nm, dd.mean(), np.median(dd), dd.std(), *np.percentile(dd, [5, 95]), 100*np.mean(np.abs(dd) <= 2), 100*np.mean(np.abs(dd) <= 5)))
    say('    by window theta_loc: median difference (m^2 per pixel)')
    TLok = TL_W[okW]; edges = np.arange(15, 71, 5); cen = []; mh = []; me = []; qh = []; qe = []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (TLok >= lo_) & (TLok < hi_)
        if m.sum() > 50:
            cen.append((lo_+hi_)/2); mh.append(np.median(d_hpa[m])); me.append(np.median(d_e5[m]))
            qh.append(np.percentile(d_hpa[m], [25, 75])); qe.append(np.percentile(d_e5[m], [25, 75]))
            say('      %2d-%2d deg  n %6d  projection angle %+.2f   Eq.5 %+.2f' % (lo_, hi_, m.sum(), mh[-1], me[-1]))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    a = axs[0]; hb = np.arange(-20, 20.01, 0.25)
    for dd, c, lab in [(d_hpa, '#d62728', 'Projection angle (current code)'), (d_e5, '#1f77b4', 'Local incidence angle, Eq. (5)')]:
        a.hist(dd, hb, density=True, histtype='step', lw=2, color=c, label='%s: median %+.2f m$^2$' % (lab, np.median(dd)))
    a.axvline(0, color='k', lw=0.8); a.set_xlim(-20, 20)
    a.set_xlabel('Area difference, integration minus method (m$^2$ per radar pixel)'); a.set_ylabel('Density')
    a.set_title('(a) Distribution, %d windows of 31 $\\times$ 7 pixels' % okW.sum(), loc='left')
    a.legend(frameon=False, fontsize=11)
    a = axs[1]; cen = np.array(cen)
    for med, q, c, lab in [(mh, qh, '#d62728', 'Projection angle (current code)'), (me, qe, '#1f77b4', 'Local incidence angle, Eq. (5)')]:
        q = np.array(q); a.fill_between(cen, q[:, 0], q[:, 1], color=c, alpha=0.18, lw=0); a.plot(cen, med, 'o-', color=c, lw=2, label=lab + ' (median, 25$-$75%)')
    a.axhline(0, color='k', lw=0.8); a.set_xlabel('Window mean local incidence angle $\\theta_{loc}$ ($\\degree$)')
    a.set_ylabel('Area difference (m$^2$ per radar pixel)'); a.set_title('(b) Difference versus local incidence angle', loc='left')
    a.legend(frameon=False, fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4_area_m2.png'), dpi=220, facecolor='white'); plt.close()
    say('\n[saved] fig4_area_m2.png, fig4_m2_log.txt in %s' % OUT); LOG.close(); sys.exit(0)

# ---------------- sample at facets, 100 m cells ----------------
wa_c = np.clip(wa, 0, nwa-1); wr_c = np.clip(wr, 0, nwr-1)
okf = inwin & okW[wa_c, wr_c]
s_ter = np.where(okW, bsW*new_afac_W, np.nan)[wa_c, wr_c]                      # terrain sigma0 (n = 0)
cosW = np.cos(np.radians(TL_W))[wa_c, wr_c]
src = rasterio.open(CCIFILE); BIO = src.read(1); T = src.transform
br = ((LAT_TOP - rows/PPD - T.f)/T.e).astype(int); bc = ((LON_LEFT + cols/PPD - T.c)/T.a).astype(int)
bio = BIO[br, bc].astype(float)
ok = (d_old > 1e-3) & okf & (bio > 0) & (bio < 600) & (u_all > 0) & (g_old > 0) & np.isfinite(s_ter) & (s_ter > 0)
key = br[ok]*BIO.shape[1] + bc[ok]
uu, inv = np.unique(key, return_inverse=True); nf = np.bincount(inv); kk = nf >= 5
cb = BIO.ravel()[uu].astype(float)[kk]
cm = lambda v: (np.bincount(inv, weights=v[ok])/nf)[kk]
ctl = cm(tl); cinc = cm(np.degrees(th))
C = {'unc':      db(cm(u_all)),
     'cur':      db(cm(g_old)),
     'new_s':    db(cm(s_ter)),
     'new_g05':  db(cm(s_ter/np.sqrt(cosW))),
     'new_g1':   db(cm(s_ter/cosW)),
     'unc_gp':   db(cm(u_all/np.cos(th))),          # gamma0 paper def., before (ellipsoid)
     'new_gp':   db(cm(s_ter/np.cos(th)))}          # gamma0 paper def., after (terrain)
NAME = {'unc': 'uncorrected sigma0 (flat)', 'cur': 'current code gamma0 (cos theta_loc)',
        'new_s': 'window integration sigma0 (n=0)', 'new_g05': 'window integration gamma0 n=0.5',
        'new_g1': 'window integration gamma0 n=1', 'unc_gp': 'gamma0 paper def. before',
        'new_gp': 'gamma0 paper def. after'}
np.savez(os.path.join(OUT, 'cells_100m_HV_20100711_final.npz'), cb=cb, ctl=ctl, cinc=cinc, nfacet=nf[kk], **C)
m15 = (ctl >= INC_MIN) & (ctl <= INC_MAX)
say('\n[cells] %d cells (theta_loc 15-70: %d), CCI 2010' % (len(cb), m15.sum()))
say('%-38s %8s %7s %8s %10s %10s %7s %9s %11s' % ('quantity', 'median', 'std', 'r(AGB)', 'slope', 'intercept', 'RMSE', 'r(dB,th)', 'resid/deg'))
FIT = {}
for kx in ['unc', 'cur', 'new_s', 'new_g05', 'new_g1', 'unc_gp', 'new_gp']:
    y = C[kx]; b = np.polyfit(cb, y, 1); r = y - np.polyval(b, cb); FIT[kx] = (b, np.sqrt(np.mean(r**2)), np.corrcoef(cb, y)[0, 1])
    say('%-38s %+8.2f %7.2f %8.3f %10.5f %+10.3f %7.3f %+9.3f %+11.4f' % (NAME[kx], np.median(y[m15]), y[m15].std(), FIT[kx][2], b[0], b[1], FIT[kx][1],
        np.corrcoef(y[m15], ctl[m15])[0, 1], np.polyfit(ctl[m15], r[m15], 1)[0]))

# ---------------- F4: analogue of Simard Fig. 10 ----------------
say('\n[F4] slope of backscatter (dB) vs theta_loc inside 1-deg theta_inc bins (100 m cells, theta_loc 15-70)')
say('     Simard Fig. 10 (UAVSAR, N1-corrected gamma0): +0.078 dB/deg at 34-35 deg look angle, +0.050 at 44-45 deg')
ib = np.arange(np.floor(cinc.min()), np.ceil(cinc.max()) + 1, 1.0)
say('     theta_inc   n_cells   ' + '  '.join('%14s' % kx for kx in ['unc', 'cur', 'new_s', 'new_g05', 'new_g1']))
for j in range(len(ib)-1):
    m = m15 & (cinc >= ib[j]) & (cinc < ib[j+1])
    if m.sum() < 1000: continue
    say('     %4.0f-%-4.0f  %8d   ' % (ib[j], ib[j+1], m.sum()) + '  '.join('%+14.4f' % np.polyfit(ctl[m], C[kx][m], 1)[0] for kx in ['unc', 'cur', 'new_s', 'new_g05', 'new_g1']))

# ---------------- figures ----------------
rng = np.random.default_rng(0); sel = rng.choice(len(cb), min(6000, len(cb)), replace=False)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for a, kx, t in zip(axs, ['unc', 'cur', 'new_g05'], ['(a) Uncorrected $\\sigma^0$', '(b) Current code $\\gamma^0$', '(c) Window integration, n = 0.5']):
    b, rm, rr = FIT[kx]; a.scatter(cb[sel], C[kx][sel], s=3, alpha=0.35, color='#3a7cc4', lw=0)
    xs = np.linspace(0, cb.max(), 50); a.plot(xs, np.polyval(b, xs), color='#d9534f', lw=2)
    a.set_title(t, loc='left', fontsize=11)
    a.text(0.03, 0.97, 'y = %.3f + %.5f x\nr = %.3f, RMSE = %.3f dB\nn = %d cells' % (b[1], b[0], rr, rm, len(cb)),
           transform=a.transAxes, va='top', fontsize=9, bbox=dict(fc='white', ec='0.8'))
    a.set_xlabel('CCI AGB 2010 (Mg/ha)')
axs[0].set_ylabel('Backscatter, HV, 100 m (dB)')
fig.suptitle('F1  One regression line per method, fitted on all 100 m cells (dB vs AGB)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F1_regression.png'), dpi=200, facecolor='white'); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6)); hb = np.arange(-22, -2, 0.2)
for a, (k1, k2, t) in zip(axs, [('unc', 'new_s', '(a) $\\sigma^0$ vs $\\sigma^0$ (no cosine term)'),
                                ('unc_gp', 'new_gp', '(b) $\\gamma^0$ vs $\\gamma^0$, $\\gamma^0=\\sigma^0/\\cos\\theta_{inc}$ (Simard Eq. 15)')]):
    for kx, c, lab in [(k1, '#1f77b4', 'before (flat ellipsoid)'), (k2, '#d62728', 'after (terrain, window integration)')]:
        y = C[kx][m15]; a.hist(y, hb, density=True, alpha=0.5, color=c, label='%s: median %.2f, std %.2f dB' % (lab, np.median(y), y.std()))
        a.axvline(np.median(y), color=c, ls='--', lw=1.2)
    a.set_title(t, loc='left', fontsize=11); a.set_xlabel('Backscatter, HV, 100 m (dB)'); a.legend(fontsize=8.5, frameon=False)
axs[0].set_ylabel('Probability density')
fig.suptitle('F2  Same quantity before and after terrain correction (%d cells, $\\theta_{loc}$ 15-70$\\degree$)' % m15.sum(), fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F2_distribution.png'), dpi=200, facecolor='white'); plt.close()

fig, a = plt.subplots(figsize=(7.5, 4.6)); hb = np.arange(-4, 4.01, 0.05)
for dd, c, lab in [(dcur, '#1f77b4', 'current code (~1 facet per pixel)'), (deq5, '#ff7f0e', 'local incidence, Simard Eq. 5')]:
    a.hist(dd, hb, density=True, histtype='step', lw=1.8, color=c, label='%s: within $\\pm$0.5 dB %.0f%%' % (lab, 100*np.mean(np.abs(dd) <= 0.5)))
a.axvline(0, color='k', lw=0.6); a.set_xlabel('Area factor minus window integration (dB)'); a.set_ylabel('Density')
a.set_title('F3  Area estimate vs DEM integration (analogue of Simard Fig. 4), %d windows' % okW.sum(), fontsize=10, loc='left')
a.legend(fontsize=8.5, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F3_area_methods.png'), dpi=200, facecolor='white'); plt.close()
say('\n[saved] F1_regression.png, F2_distribution.png, F3_area_methods.png, cells_100m_HV_20100711_final.npz in %s' % OUT)
LOG.close()
