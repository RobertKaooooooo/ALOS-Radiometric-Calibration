"""
v2_v3_checks.py  (follow-up to the 10/27 advisor meeting)

V4  The three factors of area_correction.py line 228, printed for sample pixels
    (flat / facing slope / back slope):
        ratio = (1/sin theta_inc) * (A_ref/areaRDC) * (1/cos theta_loc)
                 [area correction ..........]   [sigma0 -> gamma0]
V2  "Correction" redefined as AREA ONLY (flat ground = 0 dB):
        AC = sigma0_terrain - sigma0_flat = 10 log10( A_ref / (sin theta_inc * A) )
    facet level (A = facet area, line 106) and window level (A_flat / A_W).
V5  Theory for a pure range slope (derived from lines 79-106, tau_s = 0):
        |nE . nI| = sin(theta_inc - tau_rho) = sin(theta_loc)
        AC_theory = 10 log10( sin(theta_inc - tau_rho) / sin(theta_inc) )
    overlaid on the data; residual data - theory reported.
V3  Window integration (plan A), two ways of putting a facet into the 31 x 7 windows:
        floor    : whole facet area to the window of floor(az/31), floor(rg/7)  (previous run)
        bilinear : facet area split over the 4 neighbouring radar pixels with the
                   weights of Simard 2016 Eq. 14 ((1-dr)(1-da), ...), then each pixel to its window
    + paper-like HPA on a DEM densified 5x (GLO-30 -> ~6 m, like the UAVSAR HGT which is
      SRTM 30 m interpolated to 0.0000556 deg), run on a 64-window azimuth band, to show
      what "no upsampling" costs.

Scene : ALPSRP237630880 (11 Jul 2010, HV), CCI AGB 2010 v7.0
Run   : cd "/mnt/c/grd file" && python3 scripts/v2_v3_checks.py
Output: scripts/out/v2_v3_log.txt, fig_V2_area_correction.png, fig_V3_window_methods.png
"""

import os, sys, gc
import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates, uniform_filter

BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
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
NA, NR = int(os.environ.get('RTC_NA', 18432)), int(os.environ.get('RTC_NR', 4704))
BSFILE  = os.path.join(BASE, 'ALOS_HV_uncorrected_new.bin')
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
CCIFILE = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
WA, WR = 31, 7
INC_MIN, INC_MAX = 15.0, 70.0
UP = 5                      # DEM densification factor for the paper-like HPA test
NBAND = int(os.environ.get('RTC_NBAND', 64))   # windows (azimuth) in the HPA test band
GEOID_N = os.environ.get('RTC_GEOID_SHIFT')   # EGM2008 undulation (m): heights -> ellipsoid, range shift -N*cos(theta_inc)/DR
OUT = os.path.join(BASE, 'scripts', 'out_ellip' if GEOID_N else 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'v2_v3_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()
db = lambda x: 10*np.log10(x)

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
g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg = g['rg']
coef = leader_file.read_six_coefficients(LED)
if GEOID_N:
    _d = -float(GEOID_N)*np.cos(geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0))/DR
    rg = rg + _d
    print('[geoid] N = %+.2f m -> range shift %+.2f to %+.2f px; outputs in %s' % (float(GEOID_N), _d.min(), _d.max(), OUT)); del _d
nd = int(round(np.sqrt(os.path.getsize(DEMFILE)/4)))
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(nd, nd)
coef = leader_file.read_six_coefficients(LED)
th = geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0)
th_col = geometry.compute_theta_ih(coef, (R0 + np.arange(NR)*DR)/1000.0)
at, pd, rd, yd = leader_file.read_attitude_record(LED); o = np.argsort(at)
def attitude(a_idx):
    tt = tf + a_idx/(NA-1)*(te-tf)
    return np.radians(np.interp(tt, at[o], pd[o])), np.radians(np.interp(tt, at[o], yd[o]))
pitch, yaw = attitude(az)
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
del pitch, yaw; gc.collect()
# slopes, copied from area_correction.py lines 65-81
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
tau_rho = -np.degrees(np.arctan(np.tan(slope)*np.cos(aspect - HD - np.pi/2)))    # + = facing the radar
tau_s   =  np.degrees(np.arctan(np.tan(slope)*np.cos(aspect - HD)))
S_deg = np.degrees(slope)
surf = (dlon_m*dlat_m*np.sqrt(1.0 + p**2 + q**2)).astype(np.float64)
del p, q, p_safe, aspect, slope, lat_rows, dlon_m; gc.collect()
thd = np.degrees(th)
say('[input] %s  A_ref %.3f m2  heading %.3f deg  facets %d  theta_inc %.2f-%.2f deg' % (SID, AREF, np.degrees(HD), len(az), thd.min(), thd.max()))

# ================= V4: the three factors of line 228 =================
aR, fill = area_correction.accumulate_area_rdc(area, az, rg, NA, NR, outlier_factor=1e30)
aT, _    = area_correction.accumulate_area_rdc(np.radians(tl), az, rg, NA, NR, outlier_factor=1e30)
valid = fill & (aT <= np.radians(INC_MAX)) & (aT >= np.radians(INC_MIN))
ratio_fn, valid_fn = area_correction.compute_gamma0_ratio(area, tl, az, rg, NA, NR, th_col, AREF)
ai = np.clip(np.round(az).astype(int), 0, NA-1); ri = np.clip(np.round(rg).astype(int), 0, NR-1)
say('\n[V4] factors of area_correction.py line 228 at sample radar pixels (pixel = nearest to the facet)')
say('     F1 = 1/sin(theta_inc) (line 227)   F2 = A_ref/areaRDC   F3 = 1/cos(theta_loc)   ratio = F1*F2*F3, then clip (line 229)')
say('%-24s %7s %7s %7s %7s %8s | %7s %7s %7s | %8s %8s %8s | %10s %10s' % ('case', 'tau_rho', 'tau_s', 'th_inc', 'th_loc', 'areaRDC',
    'F1', 'F2', 'F3', 'F1F2 dB', 'F3 dB', 'ratio dB', 'line228 dB', 'diff dB'))
rng = np.random.default_rng(1)
cases = [('flat (S<1 deg)', S_deg < 1.0), ('facing, tau_rho 18-22', (tau_rho > 18) & (tau_rho < 22) & (np.abs(tau_s) < 3)),
         ('back, tau_rho -22..-18', (tau_rho > -22) & (tau_rho < -18) & (np.abs(tau_s) < 3))]
for nm, m in cases:
    idx = np.flatnonzero(m & valid[ai, ri])
    for i in (rng.choice(idx, min(3, idx.size), replace=False) if idx.size else []):
        a_, r_ = ai[i], ri[i]
        F1 = 1/np.sin(th_col[r_]); F2 = AREF/aR[a_, r_]; F3 = 1/np.cos(aT[a_, r_])
        say('%-24s %+7.2f %+7.2f %7.2f %7.2f %8.1f | %7.4f %7.4f %7.4f | %+8.3f %+8.3f %+8.3f | %+10.3f %+10.4f' % (
            nm, tau_rho[i], tau_s[i], np.degrees(th_col[r_]), np.degrees(aT[a_, r_]), aR[a_, r_], F1, F2, F3,
            db(F1*F2), db(F3), db(F1*F2*F3), db(ratio_fn[a_, r_]), db(F1*F2*F3) - db(ratio_fn[a_, r_])))
say('     ratio rebuilt vs compute_gamma0_ratio on all valid pixels: max |diff| = %.2e dB, valid masks identical: %s'
    % (np.nanmax(np.abs(db(np.clip(np.where(valid, (1/np.sin(th_col))[None, :]*AREF/np.where(fill, aR, 1.0)/np.cos(aT), 1.0), 1e-3, 1e3))
                         - db(np.where(valid, ratio_fn, 1.0)))), bool((valid == valid_fn).all())))
del ratio_fn, valid_fn; gc.collect()
# current code's area factor (F1*F2) per pixel, used later
afac_px = np.where(valid, (1/np.sin(th_col))[None, :]*AREF/np.where(fill, aR, 1.0), 0.0).astype(np.float32)
v_px = valid[ai, ri]
del aR, aT, fill; gc.collect()

# ================= V2 + V5: area-only correction, facet level =================
lit = tl < 90.0
AC = db(AREF/(np.sin(th)*area))                                     # sigma0_terrain - sigma0_flat (dB), area only
x = np.radians(thd - tau_rho)
AC_th = np.where(np.sin(x) > 0, db(np.maximum(np.sin(x), 1e-9)/np.sin(th)), np.nan)   # pure range slope theory
COS = -db(np.cos(np.radians(tl)))
res = AC - AC_th
say('\n[V2] area-only correction AC = 10log10(A_ref/(sin theta_inc * A)), facet level (flat ground should be 0)')
fl = S_deg < 3.0
say('     flat facets (S<3 deg) n %d: AC mean %+.3f dB (std %.3f) | cos term mean %+.3f | old metric gamma0-sigma0 mean %+.3f'
    % (fl.sum(), AC[fl].mean(), AC[fl].std(), COS[fl].mean(), (AC + COS)[fl].mean()))
say('theta_loc   n_facets   AC mean   AC p5/p95        theory mean   cos term   old metric   code-valid')
for lo_, hi_ in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 35), (35, 40), (40, 45), (45, 60), (60, 70), (70, 90)]:
    m = (tl >= lo_) & (tl < hi_)
    if m.sum() == 0: continue
    say('%2d-%2d deg  %9d   %+6.2f   %+6.2f/%+6.2f    %+6.2f        %+6.2f     %+6.2f      %5.1f%%' % (lo_, hi_, m.sum(), AC[m].mean(),
        *np.percentile(AC[m], [5, 95]), np.nanmean(AC_th[m]), COS[m].mean(), (AC + COS)[m].mean(), 100*v_px[m].mean()))
say('\n[V5] data - theory (pure range slope), lit facets with theta_inc - tau_rho > 0')
for nm, m in [('|tau_s| < 2 deg', lit & np.isfinite(res) & (np.abs(tau_s) < 2)), ('|tau_s| < 5 deg', lit & np.isfinite(res) & (np.abs(tau_s) < 5)),
              ('all', lit & np.isfinite(res))]:
    say('     %-16s n %9d  mean %+.4f  median %+.4f  p5/p95 %+.3f/%+.3f  |res|<0.1 dB %.1f%%' % (nm, m.sum(), res[m].mean(), np.median(res[m]),
        *np.percentile(res[m], [5, 95]), 100*np.mean(np.abs(res[m]) < 0.1)))
tim = thd.mean()
say('     theory at theta_inc = %.1f deg (scene mean):  ' % tim + '  '.join('tau_rho %+d: %+.2f dB' % (t, db(np.sin(np.radians(tim - t))/np.sin(np.radians(tim))))
                                                              for t in (-30, -20, -10, 0, 10, 20, 30)))
say('     layover (tau_rho >= theta_inc): %d facets, %d still valid in the code' % ((tau_rho >= thd).sum(), (v_px & (tau_rho >= thd)).sum()))

# ---------- figure V2 ----------
def binmed(xv, yv, e):
    bi = np.digitize(xv, e) - 1
    return 0.5*(e[:-1] + e[1:]), np.array([np.median(yv[bi == i]) if (bi == i).sum() >= 30 else np.nan for i in range(len(e) - 1)])
plt.rcParams.update({'font.size': 10.5})
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
mm = lit & np.isfinite(AC)
YL = (-12, 6)
a = axs[0]
hb = a.hexbin(tau_rho[mm], AC[mm], gridsize=160, extent=(-40, 45, YL[0], YL[1]), bins='log', cmap='Greys', mincnt=1)
tr = np.linspace(-40, 45, 400)
for t_, c in [(thd.min(), '#1f77b4'), (tim, '#d62728'), (thd.max(), '#2ca02c')]:
    xx = np.radians(t_ - tr); ok_ = np.sin(xx) > 0.01
    a.plot(tr[ok_], db(np.sin(xx[ok_])/np.sin(np.radians(t_))), color=c, lw=1.6, label='theory, $\\theta_{inc}$ = %.1f$\\degree$' % t_)
xc, med = binmed(tau_rho[mm], AC[mm], np.arange(-40, 46, 2.0))
a.plot(xc, med, 'o', ms=3.5, color='#ff7f0e', label='data median (2$\\degree$ bins)')
a.axhline(0, color='k', lw=0.7); a.axvline(0, color='k', lw=0.4, ls=':')
a.axvline(tim, color='#9467bd', ls='--', lw=1.1); a.text(tim - 0.8, YL[1] - 0.4, 'layover\n$\\tau_\\rho=\\theta_{inc}$', color='#9467bd', ha='right', va='top', fontsize=9)
for t_ in (-20, 20):
    v = db(np.sin(np.radians(tim - t_))/np.sin(np.radians(tim)))
    a.annotate('%s 20$\\degree$: %+.1f dB' % ('facing' if t_ > 0 else 'back', v), (t_, v), xytext=((-12, -7) if t_ > 0 else (-36, 4.3)),
               arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=9)
a.annotate('flat: 0 dB', (0, 0), xytext=(4, 3.0), arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=9)
a.set_ylim(*YL); a.set_xlim(-40, 45)
a.set_xlabel('Range slope $\\tau_\\rho$ (deg)  [+ = facing the radar]'); a.set_ylabel('Area correction $\\sigma^0_{terrain}-\\sigma^0_{flat}$ (dB)')
a.set_title('(a) Area correction vs range slope', loc='left'); a.legend(fontsize=8, frameon=False, loc='lower left')
a = axs[1]
a.hexbin(tl[mm], AC[mm], gridsize=160, extent=(0, 90, YL[0], YL[1]), bins='log', cmap='Greys', mincnt=1)
tt_ = np.linspace(1, 89, 300)
for t_, c in [(thd.min(), '#1f77b4'), (tim, '#d62728'), (thd.max(), '#2ca02c')]:
    a.plot(tt_, db(np.sin(np.radians(tt_))/np.sin(np.radians(t_))), color=c, lw=1.6, label='$10\\log_{10}(\\sin\\theta_{loc}/\\sin\\theta_{inc})$, $\\theta_{inc}$ = %.1f$\\degree$' % t_)
a.axhline(0, color='k', lw=0.7); a.axvspan(INC_MIN, INC_MAX, color='#2ca02c', alpha=0.05)
a.text(INC_MIN + 1, YL[1] - 0.6, 'code keeps $\\theta_{loc}$ 15-70$\\degree$ (line 226)', fontsize=8.5, color='#2ca02c')
a.set_ylim(*YL); a.set_xlim(0, 90)
a.set_xlabel('Local incidence angle $\\theta_{loc}$ (deg)'); a.set_ylabel('Area correction (dB)')
a.set_title('(b) Area correction vs $\\theta_{loc}$: 0 dB where $\\theta_{loc}=\\theta_{inc}$', loc='left'); a.legend(fontsize=8, frameon=False, loc='lower right')
a = axs[2]
hbins = np.arange(-0.40, 0.1001, 0.005)
for nm, m, c in [('|$\\tau_s$| < 2$\\degree$', lit & np.isfinite(res) & (np.abs(tau_s) < 2), '#1f77b4'), ('all lit facets', lit & np.isfinite(res), '0.4')]:
    a.hist(np.clip(res[m], -0.399, 0.099), hbins, density=True, histtype='step', lw=1.6, color=c,
           label='%s: median %+.3f dB, %.0f%% within $\\pm$0.1 dB' % (nm, np.median(res[m]), 100*np.mean(np.abs(res[m]) < 0.1)))
a.axvline(0, color='k', lw=0.6); a.set_xlim(-0.4, 0.1); a.set_xlabel('Data - theory (dB), clipped to [-0.4, 0.1]'); a.set_ylabel('Density')
a.set_title('(c) Check: code vs pure-range-slope theory', loc='left'); a.legend(fontsize=8, frameon=False, loc='upper left')
fig.suptitle('%s HV, one GLO-30 facet per point (n = %d lit facets). Area term only; the 1/cos$\\theta_{loc}$ ($\\sigma^0\\rightarrow\\gamma^0$) term is excluded' % (SID, mm.sum()), fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_V2_area_correction.png'), dpi=200, facecolor='white'); plt.close()
del hb, AC_th, res, COS, x; gc.collect()

# ================= V3: window integration, floor vs bilinear (Eq. 14) =================
nwa, nwr = NA//WA, NR//WR
blk = lambda arr: arr[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).sum(axis=(1, 3)).astype(np.float64)
vf = valid.astype(np.float32)
cnt_W = blk(vf); cur_afac_W = blk(afac_px)/np.maximum(cnt_W, 1)
del afac_px; gc.collect()

def window_sum(weights_fn):
    """returns sum over windows of surf*w and of surf*w*tl, plus facet count, for the lit facets"""
    A = np.zeros(nwa*nwr); T = np.zeros(nwa*nwr)
    for wid, w in weights_fn():
        A += np.bincount(wid, weights=w*surf[sel_l], minlength=nwa*nwr)
        T += np.bincount(wid, weights=w*(surf*tl)[sel_l], minlength=nwa*nwr)
    return A.reshape(nwa, nwr), T.reshape(nwa, nwr)

wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg/WR).astype(np.int64)
inwin = (az >= 0) & (rg >= 0) & (wa < nwa) & (wr < nwr)
N_W = np.bincount((wa*nwr + wr)[inwin], minlength=nwa*nwr).reshape(nwa, nwr)
cover = binary_erosion(N_W > 0, structure=np.ones((3, 3)))
sel_l = inwin & lit
def w_floor():
    yield (wa*nwr + wr)[sel_l], np.ones(sel_l.sum())
def w_bilinear():                       # Simard 2016 Eq. 14 weights on the 4 neighbouring radar pixels
    a_, r_ = az[sel_l], rg[sel_l]
    a0 = np.floor(a_).astype(np.int64); r0 = np.floor(r_).astype(np.int64); fa = a_ - a0; fr = r_ - r0
    for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
        pa = a0 + da; pr = r0 + dr
        ok_ = (pa < nwa*WA) & (pr < nwr*WR)
        yield np.where(ok_, (pa//WA)*nwr + pr//WR, 0), np.where(ok_, w, 0.0)
th_W = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]
A_flat = WA*WR*AREF/np.sin(th_W)
WIN = {}
for nm, fn in [('floor', w_floor), ('bilinear', w_bilinear)]:
    A_W, T_W = window_sum(fn); TL_W = T_W/np.maximum(A_W, 1e-9)
    okW = cover & (A_W > 0) & (TL_W >= INC_MIN) & (TL_W <= INC_MAX) & (cnt_W > 0)
    WIN[nm] = dict(A=A_W, TL=TL_W, ok=okW, afac=A_flat/np.maximum(A_W, 1e-9))
both = WIN['floor']['ok'] & WIN['bilinear']['ok']
d_fb = db(WIN['bilinear']['afac'][both]) - db(WIN['floor']['afac'][both])
say('\n[V3] window integration, bilinear (Eq. 14) minus floor, %d windows: area factor mean %+.4f  median %+.4f  std %.4f  p1/p99 %+.3f/%+.3f dB  |diff|<0.1 dB %.1f%%'
    % (both.sum(), d_fb.mean(), np.median(d_fb), d_fb.std(), *np.percentile(d_fb, [1, 99]), 100*np.mean(np.abs(d_fb) < 0.1)))
say('     theta_loc_W bilinear - floor: median %+.3f deg, p1/p99 %+.2f/%+.2f' % (np.median((WIN['bilinear']['TL'] - WIN['floor']['TL'])[both]),
    *np.percentile((WIN['bilinear']['TL'] - WIN['floor']['TL'])[both], [1, 99])))
ACW = db(WIN['bilinear']['afac'])
say('     window area correction AC_W = 10log10(A_flat/A_W) (bilinear), by window theta_loc:')
for lo_, hi_ in [(15, 25), (25, 35), (35, 40), (40, 45), (45, 55), (55, 70)]:
    m = both & (WIN['bilinear']['TL'] >= lo_) & (WIN['bilinear']['TL'] < hi_)
    if m.sum() > 50:
        say('       %2d-%2d deg  n %6d  AC_W median %+.2f dB  (theory sin(th_loc)/sin(th_inc) at window means %+.2f)' % (lo_, hi_, m.sum(), np.median(ACW[m]),
            np.median(db(np.sin(np.radians(WIN['bilinear']['TL'][m]))/np.sin(np.broadcast_to(th_W, ACW.shape)[m])))))

# ---------- cells: unc / current / floor / bilinear (same recipe as final_checks.py) ----------
bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
ratio, _ = area_correction.compute_gamma0_ratio(area, tl, az, rg, NA, NR, th_col, AREF)
gam = bs*ratio*vf; del ratio; gc.collect()
d_old = uniform_filter(vf, (WA, WR))[ai, ri].astype(float)
g_old = uniform_filter(gam, (WA, WR))[ai, ri].astype(float)/np.maximum(d_old, 1e-6); del gam; gc.collect()
u_all = uniform_filter(bs, (WA, WR))[ai, ri].astype(float)
bsW = bs[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)).astype(np.float64)
wa_c = np.clip(wa, 0, nwa-1); wr_c = np.clip(wr, 0, nwr-1)
src = rasterio.open(CCIFILE); BIO = src.read(1); Tr = src.transform
br = ((LAT_TOP - rows/PPD - Tr.f)/Tr.e).astype(int); bc = ((LON_LEFT + cols/PPD - Tr.c)/Tr.a).astype(int)
bio = BIO[br, bc].astype(float)
sF = np.where(WIN['floor']['ok'], bsW*WIN['floor']['afac'], np.nan)[wa_c, wr_c]
sB = np.where(WIN['bilinear']['ok'], bsW*WIN['bilinear']['afac'], np.nan)[wa_c, wr_c]
cF = np.cos(np.radians(WIN['floor']['TL']))[wa_c, wr_c]; cB = np.cos(np.radians(WIN['bilinear']['TL']))[wa_c, wr_c]
okf = inwin & WIN['floor']['ok'][wa_c, wr_c]
ok0 = (d_old > 1e-3) & okf & (bio > 0) & (bio < 600) & (u_all > 0) & (g_old > 0) & np.isfinite(sF) & (sF > 0)
ok = ok0 & np.isfinite(sB) & (sB > 0)
key = br[ok]*BIO.shape[1] + bc[ok]
uu, inv = np.unique(key, return_inverse=True); nf = np.bincount(inv); kk = nf >= 5
cb = BIO.ravel()[uu].astype(float)[kk]
cm = lambda v: (np.bincount(inv, weights=v[ok])/nf)[kk]
ctl = cm(tl)
C = {'uncorrected sigma0': db(cm(u_all)), 'current code gamma0': db(cm(g_old)),
     'floor   sigma0 n=0': db(cm(sF)), 'floor   gamma0 n=0.5': db(cm(sF/np.sqrt(cF))),
     'bilinear sigma0 n=0': db(cm(sB)), 'bilinear gamma0 n=0.5': db(cm(sB/np.sqrt(cB))), 'bilinear gamma0 n=1': db(cm(sB/cB))}
m15 = (ctl >= INC_MIN) & (ctl <= INC_MAX)
say('\n[V3 cells] %d cells (facets valid in floor AND bilinear; final_checks used floor only: 561,753 cells, unc r 0.564, current 0.621, n=0.5 0.658)' % len(cb))
say('%-24s %8s %7s %8s %10s %10s %7s %9s %11s' % ('method', 'median', 'std', 'r(AGB)', 'slope', 'intercept', 'RMSE', 'r(dB,th)', 'resid/deg'))
FIT = {}
for nm, y in C.items():
    b = np.polyfit(cb, y, 1); r = y - np.polyval(b, cb); FIT[nm] = (b, np.sqrt(np.mean(r**2)), np.corrcoef(cb, y)[0, 1])
    say('%-24s %+8.2f %7.2f %8.3f %10.5f %+10.3f %7.3f %+9.3f %+11.4f' % (nm, np.median(y[m15]), y[m15].std(), FIT[nm][2], b[0], b[1], FIT[nm][1],
        np.corrcoef(y[m15], ctl[m15])[0, 1], np.polyfit(ctl[m15], r[m15], 1)[0]))
del d_old, g_old, u_all, sF, sB, cF, cB, okf, ok0, ok, inv, key; gc.collect()

# ================= V3b: paper-like HPA on a DEM densified UP x (one azimuth band) =================
WB0 = max(nwa//2 - NBAND//2, 0); WB1 = min(WB0 + NBAND, nwa)
A0, A1 = WB0*WA, WB1*WA
pick = (az >= A0 - 40) & (az < A1 + 40)
r0_, r1_ = rows[pick].min(), rows[pick].max(); c0_, c1_ = cols[pick].min(), cols[pick].max()
AZg = np.full((r1_ - r0_ + 1, c1_ - c0_ + 1), np.nan); RGg = np.full_like(AZg, np.nan)
AZg[rows[pick] - r0_, cols[pick] - c0_] = az[pick]; RGg[rows[pick] - r0_, cols[pick] - c0_] = rg[pick]
fy = np.arange(0, (r1_ - r0_)*UP + 1)/UP; fx = np.arange(0, (c1_ - c0_)*UP + 1)/UP
FY, FX = np.meshgrid(fy, fx, indexing='ij')
Df = map_coordinates(dem, [FY + r0_, FX + c0_], order=1).astype(np.float32)          # DEM densified (bilinear)
AZf = map_coordinates(AZg, [FY, FX], order=1, cval=np.nan); RGf = map_coordinates(RGg, [FY, FX], order=1, cval=np.nan)
del FY, FX, AZg, RGg; gc.collect()
inner = np.zeros(Df.shape, bool); inner[1:-1, 1:-1] = True
use = inner & np.isfinite(AZf) & np.isfinite(RGf) & (AZf >= A0) & (AZf < A1 - 1) & (RGf >= 0) & (RGf < NR - 1)
fr_, fc_ = np.nonzero(use); azf = AZf[use]; rgf = RGf[use]
del AZf, RGf, use, inner; gc.collect()
lat_top_f = LAT_TOP - r0_/PPD
thf = geometry.compute_theta_ih(coef, (R0 + rgf*DR)/1000.0)
tlf = np.empty(azf.size, np.float32); arf = np.empty(azf.size, np.float32)
for s in range(0, azf.size, 4_000_000):
    e = slice(s, s + 4_000_000); pf, yf = attitude(azf[e])
    tlf[e], arf[e] = area_correction.compute_facet_area_and_theta_l(Df, fr_[e], fc_[e], HD, PPD*UP, lat_top_f, thf[e], pf, yf, area_ref_m2=AREF)
del Df, fr_, fc_, thf; gc.collect()
NB = A1 - A0
aRf, fillf = area_correction.accumulate_area_rdc(arf.astype(np.float64), azf - A0, rgf, NB, NR, outlier_factor=1e30)
aTf, _ = area_correction.accumulate_area_rdc(np.radians(tlf).astype(np.float64), azf - A0, rgf, NB, NR, outlier_factor=1e30)
validf = fillf & (aTf <= np.radians(INC_MAX)) & (aTf >= np.radians(INC_MIN))
afac_f = np.where(validf, (1/np.sin(th_col))[None, :]*AREF/np.where(fillf, aRf, 1.0), 0.0)
blkb = lambda arr: arr[:, :nwr*WR].reshape(WB1 - WB0, WA, nwr, WR).sum(axis=(1, 3)).astype(np.float64)
cnt_f = blkb(validf.astype(np.float32)); hpa_f_W = blkb(afac_f)/np.maximum(cnt_f, 1)
okb = both[WB0:WB1] & (cnt_f > 0)
# fill rates inside the footprint of the band (pixels whose window is covered)
covpx = np.repeat(np.repeat(cover[WB0:WB1], WA, axis=0), WR, axis=1)
fill_c = valid[A0:A1, :nwr*WR][covpx].mean(); fill_f = validf[:, :nwr*WR][covpx].mean()
ref = WIN['bilinear']['afac'][WB0:WB1][okb]
d_cur = db(cur_afac_W[WB0:WB1][okb]) - db(ref); d_hpa = db(hpa_f_W[okb]) - db(ref)
say('\n[V3b] paper-like HPA with DEM densified %dx (%.1f x %.1f m, %d fine facets), azimuth lines %d-%d (%d windows)'
    % (UP, area_correction.DEG_M/PPD/UP*np.cos(np.radians(LAT_TOP - (r0_ + r1_)/2/PPD)), area_correction.DEG_M/PPD/UP, azf.size, A0, A1, okb.sum()))
say('     radar pixels valid (have a facet, theta_loc 15-70): original DEM %.1f%% | densified DEM %.1f%%' % (100*fill_c, 100*fill_f))
for nm, dd in [('current code, original DEM', d_cur), ('HPA, densified DEM', d_hpa)]:
    say('     %-28s minus window integration: mean %+.3f  median %+.3f  std %.3f  within +-0.5 dB %.1f%%  within +-1 dB %.1f%%'
        % (nm, dd.mean(), np.median(dd), dd.std(), 100*np.mean(np.abs(dd) <= 0.5), 100*np.mean(np.abs(dd) <= 1.0)))

# ---------- figure V3 ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), layout='constrained')
a = axs[0]
a.hist(np.clip(d_fb, -1.995, 1.995), np.arange(-2, 2.001, 0.02), color='#1f77b4', alpha=0.8)
a.set_yscale('log'); a.axvline(0, color='k', lw=0.6)
a.set_xlabel('Area factor: bilinear (Eq. 14) - floor (dB), clipped to $\\pm$2'); a.set_ylabel('Windows (log scale)')
a.set_title('(a) Facet-to-window assignment, %d windows\nstd %.2f dB, |diff| < 0.1 dB: %.1f%%' % (both.sum(), d_fb.std(), 100*np.mean(np.abs(d_fb) < 0.1)), loc='left', fontsize=10)
a = axs[1]
hbins = np.arange(-1.5, 1.5001, 0.025)
for dd, c, lab in [(d_cur, '#1f77b4', 'current code, GLO-30 as is (valid px %.0f%%)' % (100*fill_c)),
                   (d_hpa, '#d62728', 'HPA, GLO-30 densified %dx (valid px %.0f%%)' % (UP, 100*fill_f))]:
    a.hist(np.clip(dd, -1.49, 1.49), hbins, density=True, histtype='step', lw=1.8, color=c, label='%s\nwithin $\\pm$0.5 dB: %.0f%%' % (lab, 100*np.mean(np.abs(dd) <= 0.5)))
a.axvline(0, color='k', lw=0.6); a.set_xlabel('Area factor minus window integration (dB), clipped to $\\pm$1.5'); a.set_ylabel('Density')
a.set_title('(b) Effect of DEM densification (paper-like), %d windows' % okb.sum(), loc='left', fontsize=10); a.legend(fontsize=8, frameon=False, loc='upper left')
a = axs[2]
nms = ['uncorrected sigma0', 'current code gamma0', 'floor   gamma0 n=0.5', 'bilinear gamma0 n=0.5']
labs = ['uncorrected $\\sigma^0$', 'current code $\\gamma^0$', 'window, floor, n=0.5', 'window, Eq. 14, n=0.5']
ys = np.arange(len(nms))[::-1]
rv = [FIT[n][2] for n in nms]; ev = [FIT[n][1] for n in nms]
a.plot(rv, ys, 'o', color='#1f77b4', ms=8)
for y_, v in zip(ys, rv): a.text(v, y_ + 0.22, 'r = %.3f' % v, ha='center', fontsize=9, color='#1f77b4')
a.set_yticks(ys); a.set_yticklabels(labs, fontsize=9); a.set_xlabel('r (dB vs CCI AGB 2010)', color='#1f77b4')
a.set_xlim(min(rv) - 0.03, max(rv) + 0.03); a.set_ylim(-0.6, len(nms) - 0.4); a.grid(axis='x', lw=0.4, alpha=0.5)
a2 = a.twiny()
a2.plot(ev, ys - 0.0, 's', color='#ff7f0e', ms=7, mfc='none', mew=1.6)
for y_, v in zip(ys, ev): a2.text(v, y_ - 0.38, 'RMSE %.3f' % v, ha='center', fontsize=9, color='#ff7f0e')
a2.set_xlim(max(ev) + 0.1, min(ev) - 0.1); a2.set_xlabel('RMSE of linear fit (dB), reversed: right = better', color='#ff7f0e')
a.set_title('(c) 100 m cells, n = %d (right = better on both axes)' % len(cb), loc='left', fontsize=10)
fig.suptitle('%s HV: window integration (plan A) checks' % SID, fontsize=11)
plt.savefig(os.path.join(OUT, 'fig_V3_window_methods.png'), dpi=200, facecolor='white'); plt.close()
say('\n[saved] v2_v3_log.txt, fig_V2_area_correction.png, fig_V3_window_methods.png in %s' % OUT)
LOG.close()
