"""
final_eq14.py  (all remaining checks before the meeting, plan A = window integration with Simard Eq. 14 weights)

T1  F1 regression figure, Eq. 14 version (slide 5)
T2  Residual angle trend per 1-deg theta_inc bin + figure next to Simard Fig. 10 values (slides 10-11)
T3  Test of the inference "current code r is lower because only 24.5% of pixels are used":
      plan A, n=1, but sigma0 averaged ONLY over the pixels valid in the current code
T4  Layover: drop pixels / windows that receive a facet with tau_rho >= theta_inc, re-run statistics
T5  Second data set for n: HH of the same scene (if ALOS_HH_uncorrected*.bin exists), n = 0 ... 1
T6  DEM height reference: GLO-30 heights are EGM2008 (geoid); geolocation needs ellipsoid heights.
      First-order correction: ellipsoid h = H + N  ->  slant range changes by -N*cos(theta_inc)
      rg_new = rg - N*cos(theta_inc)/DR   (azimuth unchanged, zero-Doppler)
      N from pyproj (EGM2008 grid, PROJ network) or env RTC_GEOID_N (metres)

Run   : cd "/mnt/c/grd file" && python3 scripts/final_eq14.py
Output: scripts/out/final_eq14_log.txt, F1_regression_eq14.png, F4_angle_trend_eq14.png
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
LOG = open(os.path.join(OUT, ('az%+.2f_' % float(os.environ['RTC_AZ_SHIFT']) if os.environ.get('RTC_AZ_SHIFT') else '') + ('final_eq14_GEOCHK_log.txt' if os.environ.get('RTC_GEOCHK') else 'final_eq14_OVS_log.txt' if os.environ.get('RTC_OVS') else 'final_eq14_ASF_log.txt' if os.environ.get('RTC_ASF') else 'final_eq14_SAME_log.txt' if os.environ.get('RTC_SAME') else 'final_eq14_T6_log.txt' if os.environ.get('RTC_ONLY_T6') else 'final_eq14_FIGS_log.txt' if os.environ.get('RTC_FIGS_ONLY') else 'final_eq14_log.txt')), 'w')
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
if not os.environ.get('RTC_NO_TIMEFIX'):   # the facet file was geolocated with the integer-ms stamps of lines 0 and NA-1; re-map to fitted line times
    tf_fit, te_fit, _dt = img_file.read_line_times_fit(IMG, NR, NA)
    az = ((tf + az*(te - tf)/(NA - 1)) - tf_fit)/(te_fit - tf_fit)*(NA - 1)
    say('[time] line-time origin %.7f s (stamp %.3f s): azimuth shift %+.3f lines' % (tf_fit, tf, -(tf_fit - tf)/_dt))
    tf, te = tf_fit, te_fit
AZ_SHIFT = float(os.environ.get('RTC_AZ_SHIFT', 4.17))   # azimuth timing calibration (lines): +4.17 estimated vs ASF RT1 via GEOCHK cross-correlation, uniform across 3x3 image blocks (4.11-4.24); set RTC_AZ_SHIFT=0 to disable
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

src = rasterio.open(CCIFILE); BIO = src.read(1); Tr = src.transform
br = ((LAT_TOP - rows/PPD - Tr.f)/Tr.e).astype(int); bc = ((LON_LEFT + cols/PPD - Tr.c)/Tr.a).astype(int)
bio = BIO[br, bc].astype(float)
nwa, nwr = NA//WA, NR//WR
blk = lambda arr: arr[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).sum(axis=(1, 3)).astype(np.float64)
th_W = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]
A_flat = WA*WR*AREF/np.sin(th_W)
say('[input] %s  A_ref %.3f m2  facets %d  CCI %s' % (SID, AREF, len(az), os.path.basename(CCIFILE)))


def run(rg, bs, extra=False, tag=''):
    """current code + plan A (Eq. 14) on facet-sampled arrays; extra = T3/T4 variants"""
    th = geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0)
    tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
    ai = np.clip(np.round(az).astype(int), 0, NA-1); ri = np.clip(np.round(rg).astype(int), 0, NR-1)
    lay = tau_rho >= np.degrees(th)
    R = {'tl': tl, 'th': th}
    ratio, valid = area_correction.compute_gamma0_ratio(area, tl, az, rg, NA, NR, th_col, AREF)
    del area; gc.collect()
    vf = valid.astype(np.float32)
    gam = bs*ratio*vf; del ratio; gc.collect()
    R['d_old'] = uniform_filter(vf, (WA, WR))[ai, ri].astype(float)
    R['g_old'] = uniform_filter(gam, (WA, WR))[ai, ri].astype(float)/np.maximum(R['d_old'], 1e-6)
    R['u_all'] = uniform_filter(bs, (WA, WR))[ai, ri].astype(float)
    if os.environ.get('RTC_SAME'):   # per-pixel RTC without / with half of the cosine term (theta_l = IDW average, as in compute_gamma0_ratio)
        aT, _ = area_correction.accumulate_area_rdc(np.radians(tl), az, rg, NA, NR, outlier_factor=1e30)
        cT = np.clip(np.cos(aT), 0.0, 1.0).astype(np.float32); del aT; gc.collect()   # clip: theta_l > 90 deg in void pixels gave sqrt(<0) = NaN
        R['s_old'] = uniform_filter(gam*cT, (WA, WR))[ai, ri].astype(float)/np.maximum(R['d_old'], 1e-6)
        R['h_old'] = uniform_filter(gam*np.sqrt(cT), (WA, WR))[ai, ri].astype(float)/np.maximum(R['d_old'], 1e-6)
        del cT; gc.collect()
    if extra:   # T4: pixels that receive a layover facet (the 4 IDW neighbours) are removed from the current code
        lp = np.zeros((NA, NR), bool)
        a0 = np.floor(az[lay]).astype(int); r0 = np.floor(rg[lay]).astype(int)
        for da in (0, 1):
            for dr in (0, 1):
                lp[np.clip(a0 + da, 0, NA-1), np.clip(r0 + dr, 0, NR-1)] = True
        vfl = (valid & ~lp).astype(np.float32); del lp
        R['d_nl'] = uniform_filter(vfl, (WA, WR))[ai, ri].astype(float)
        R['g_nl'] = uniform_filter(gam*vfl, (WA, WR))[ai, ri].astype(float)/np.maximum(R['d_nl'], 1e-6)
        R['cnt_layfac_valid'] = int(valid[ai[lay], ri[lay]].sum()); del vfl
    del gam; gc.collect()
    # ---- plan A, Eq. 14 weights ----
    wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg/WR).astype(np.int64)
    inwin = (az >= 0) & (rg >= 0) & (wa < nwa) & (wr < nwr)
    N_W = np.bincount((wa*nwr + wr)[inwin], minlength=nwa*nwr).reshape(nwa, nwr)
    cover = binary_erosion(N_W > 0, structure=np.ones((3, 3)))
    s = inwin & (tl < 90.0)
    a_, r_ = az[s], rg[s]
    a0 = np.floor(a_).astype(np.int64); r0 = np.floor(r_).astype(np.int64); fa = a_ - a0; fr = r_ - r0
    A = np.zeros(nwa*nwr); T = np.zeros(nwa*nwr); L = np.zeros(nwa*nwr)
    for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
        pa = a0 + da; pr = r0 + dr; ok_ = (pa < nwa*WA) & (pr < nwr*WR)
        wid = np.where(ok_, (pa//WA)*nwr + pr//WR, 0); ww = np.where(ok_, w, 0.0)
        A += np.bincount(wid, weights=ww*surf[s], minlength=nwa*nwr)
        T += np.bincount(wid, weights=ww*(surf*tl)[s], minlength=nwa*nwr)
        L += np.bincount(wid, weights=ww*lay[s], minlength=nwa*nwr)
    A = A.reshape(nwa, nwr); TL = T.reshape(nwa, nwr)/np.maximum(A, 1e-9); L = L.reshape(nwa, nwr)
    cntW = blk(vf)
    okW = cover & (A > 0) & (TL >= INC_MIN) & (TL <= INC_MAX) & (cntW > 0)
    afac = A_flat/np.maximum(A, 1e-9)
    bsW = bs[:nwa*WA, :nwr*WR].reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)).astype(np.float64)
    wa_c = np.clip(wa, 0, nwa-1); wr_c = np.clip(wr, 0, nwr-1)
    R['okf'] = inwin & okW[wa_c, wr_c]
    R['sB'] = np.where(okW, bsW*afac, np.nan)[wa_c, wr_c]
    R['cB'] = np.cos(np.radians(TL))[wa_c, wr_c]
    if extra:
        bsWv = blk(bs*vf)/np.maximum(cntW, 1)                                    # T3: only code-valid pixels
        R['sB_v'] = np.where(okW, bsWv*afac, np.nan)[wa_c, wr_c]
        R['sB_nl'] = np.where(okW & (L == 0), bsW*afac, np.nan)[wa_c, wr_c]      # T4: windows without layover facets
        R['n_win_lay'] = int((okW & (L > 0)).sum()); R['n_win'] = int(okW.sum())
    del vf, valid; gc.collect()
    return R


def cells(R, series):
    """series: name -> facet-level linear power; all series must be valid in the same facets"""
    ok = (R['d_old'] > 1e-3) & R['okf'] & (bio > 0) & (bio < 600) & (R['u_all'] > 0) & (R['g_old'] > 0)
    for v in series.values():
        ok &= np.isfinite(v) & (v > 0)
    key = br[ok]*BIO.shape[1] + bc[ok]
    uu, inv = np.unique(key, return_inverse=True); nf = np.bincount(inv); kk = nf >= 5
    cm = lambda v: (np.bincount(inv, weights=v[ok])/nf)[kk]
    out = {nm: db(cm(v)) for nm, v in series.items()}
    cells.key = uu[kk]
    return BIO.ravel()[uu].astype(float)[kk], cm(R['tl']), cm(np.degrees(R['th'])), out


def table(title, cb, ctl, C):
    m15 = (ctl >= INC_MIN) & (ctl <= INC_MAX); F = {}
    say('\n%s  (%d cells)' % (title, len(cb)))
    say('%-34s %8s %7s %8s %10s %10s %7s %9s %11s' % ('method', 'median', 'std', 'r(AGB)', 'slope', 'intercept', 'RMSE', 'r(dB,th)', 'resid/deg'))
    for nm, y in C.items():
        b = np.polyfit(cb, y, 1); r = y - np.polyval(b, cb); F[nm] = (b, np.sqrt(np.mean(r**2)), np.corrcoef(cb, y)[0, 1], r)
        say('%-34s %+8.2f %7.2f %8.3f %10.5f %+10.3f %7.3f %+9.3f %+11.4f' % (nm, np.median(y[m15]), y[m15].std(), F[nm][2], b[0], b[1], F[nm][1],
            np.corrcoef(y[m15], ctl[m15])[0, 1], np.polyfit(ctl[m15], r[m15], 1)[0]))
    return F


# ================= HV, main run =================
bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
if not os.environ.get('RTC_ONLY_T6'):
    R = run(rg0, bs, extra=True)
    S = {'uncorrected sigma0': R['u_all'], 'current code gamma0': R['g_old'],
         'Eq14 sigma0 n=0': R['sB'], 'Eq14 gamma0 n=0.5': R['sB']/np.sqrt(R['cB']), 'Eq14 gamma0 n=1': R['sB']/R['cB']}
    cb, ctl, cinc, C = cells(R, S)
    F = table('[main] HV, CCI 2010, plan A = Eq. 14 (expected: unc 0.564, current 0.622, n=0.5 0.667)', cb, ctl, C)
    if os.environ.get('RTC_GEOCHK'):   # geolocation check: correlate ASF RTC (geocoded, independent) with our radar image at each facet, over integer offsets
        from rasterio.warp import transform as wtf
        AD = '/mnt/c/grd file/AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1'
        ah = rasterio.open(AD + '_HV.tif'); AV = ah.read(1); LS = rasterio.open(AD + '.ls_map.tif').read(1)
        ex, ny = wtf('EPSG:4326', ah.crs, (LON_LEFT + cols/PPD).tolist(), (LAT_TOP - rows/PPD).tolist())
        ai = ((np.array(ny) - ah.transform.f)/ah.transform.e).astype(int); aj = ((np.array(ex) - ah.transform.c)/ah.transform.a).astype(int)
        okA = (ai >= 0) & (ai < AV.shape[0]) & (aj >= 0) & (aj < AV.shape[1]); vA = np.zeros(len(rows))
        vA[okA] = np.where(LS[ai[okA], aj[okA]] == 1, AV[ai[okA], aj[okA]], 0.0); okA &= vA > 0; lv = np.log(vA[okA])
        Sm = np.log(np.maximum(uniform_filter(np.asarray(bs, dtype=np.float32), (9, 3)), 1e-6)); azo = az[okA]
        for name, rgv in (('no geoid correction (GLO-30 heights used as ellipsoidal)', np.asarray(g['rg'])[okA]), ('with geoid correction N = %s m' % GEOID_N, rg0[okA])):
            C = {}
            for da in range(-8, 9, 2):
                ia = np.clip(np.round(azo).astype(int) + da, 0, NA-1)
                for dr in range(-6, 7):
                    C[(da, dr)] = np.corrcoef(lv, Sm[ia, np.clip(np.round(rgv).astype(int) + dr, 0, NR-1)])[0, 1]
            (ba, br_), cb = max(C.items(), key=lambda kv: kv[1])
            par = lambda m, c0, p: 0.5*(m - p)/(m - 2*c0 + p) if (m - 2*c0 + p) != 0 else 0.0
            fr = br_ + (par(C[(ba, br_-1)], cb, C[(ba, br_+1)]) if abs(br_) < 6 else 0.0)
            fa = ba + (2*par(C[(ba-2, br_)], cb, C[(ba+2, br_)]) if abs(ba) < 8 else 0.0)
            say('[GEOCHK] %s: %d facets; best offset azimuth %+d lines (sub-px %+.2f), range %+d px (sub-px %+.2f, %+.1f m slant); r = %.4f; r at (0,0) = %.4f' % (name, okA.sum(), ba, fa, br_, fr, fr*DR, cb, C[(0, 0)]))
            say('          r vs range offset at best azimuth: ' + ' '.join('%+d:%.4f' % (d, C[(ba, d)]) for d in range(-6, 7)))
        rr = np.round(rgv).astype(int); aa = np.round(azo).astype(int)
        qa = np.quantile(aa, [0, 1/3, 2/3, 1]); qr = np.quantile(rr, [0, 1/3, 2/3, 1])
        for bi in range(3):
            for bj in range(3):
                m = (aa >= qa[bi]) & (aa <= qa[bi+1]) & (rr >= qr[bj]) & (rr <= qr[bj+1])
                B = {}
                for da2 in range(-8, 9):
                    B[da2] = np.corrcoef(lv[m], Sm[np.clip(aa[m] + da2, 0, NA-1), np.clip(rr[m], 0, NR-1)])[0, 1]
                bb = max(B, key=B.get)
                fb = bb + (par(B[bb-1], B[bb], B[bb+1]) if abs(bb) < 8 else 0.0)
                say('[GEOCHK3x3] %s: az-band %d, rg-band %d: %d facets; best az %+d (sub %+.2f); r = %.4f' % (name, bi, bj, m.sum(), bb, fb, B[bb]))
        raise SystemExit
    if os.environ.get('RTC_OVS'):   # baseline: per-pixel RTC on K x K subdivided (planar-oversampled) facets, then window mean
        K = int(os.environ.get('RTC_OVS')); nF = len(rows); tlF = R['tl']
        idx = np.full((rows.max() + 2, cols.max() + 2), -1, np.int64); idx[rows, cols] = np.arange(nF)
        nb_ = lambda dr, dc: (lambda j: np.where(j >= 0, j, np.arange(nF)))(idx[np.clip(rows + dr, 0, None), np.clip(cols + dc, 0, None)])
        dn, up, rt, lf = nb_(1, 0), nb_(-1, 0), nb_(0, 1), nb_(0, -1)
        Jar, Jrr = (az[dn] - az[up])/2, (rg0[dn] - rg0[up])/2; Jac, Jrc = (az[rt] - az[lf])/2, (rg0[rt] - rg0[lf])/2
        del idx, dn, up, rt, lf; gc.collect()
        sv = tlF < 90.0; NP_ = nwa*WA*nwr*WR; Ap = np.zeros(NP_); Tp = np.zeros(NP_)
        for i_ in range(K):
            for j_ in range(K):
                du, dv = (i_ + 0.5)/K - 0.5, (j_ + 0.5)/K - 0.5
                a_ = (az + du*Jar + dv*Jac)[sv]; r_ = (rg0 + du*Jrr + dv*Jrc)[sv]; wsub = surf[sv]/K**2; tsub = tlF[sv]
                a0 = np.floor(a_).astype(np.int64); r0 = np.floor(r_).astype(np.int64); fa = a_ - a0; fr = r_ - r0
                for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
                    pa = a0 + da; pr = r0 + dr; ok_ = (pa >= 0) & (pr >= 0) & (pa < nwa*WA) & (pr < nwr*WR)
                    pid = pa[ok_]*(nwr*WR) + pr[ok_]
                    Ap += np.bincount(pid, weights=(w*wsub)[ok_], minlength=NP_); Tp += np.bincount(pid, weights=(w*wsub*tsub)[ok_], minlength=NP_)
        Ap = Ap.reshape(nwa*WA, nwr*WR); TLp = (Tp.reshape(nwa*WA, nwr*WR)/np.maximum(Ap, 1e-9)).astype(np.float32); del Tp; gc.collect()
        hp = Ap > 0; pc = hp.reshape(nwa, WA, nwr, WR).mean(axis=(1, 3)); pc = pc[pc > 0]
        say('[OVS] K = %d: per-window share of pixels with facet area > 0: median %.3f, 5th pct %.3f; pixel area sum / facet area sum = %.4f' % (K, np.median(pc), np.percentile(pc, 5), Ap.sum()/surf[sv].sum()))
        sp = np.where(hp, bs[:nwa*WA, :nwr*WR]*(AREF/np.sin(th_col[:nwr*WR]))[None, :]/np.maximum(Ap, 1e-9), np.nan).astype(np.float32)
        cp = np.clip(np.cos(np.radians(TLp)), 1e-3, 1).astype(np.float32); del TLp, Ap, hp; gc.collect()
        wa_c = np.clip(np.floor(az/WA).astype(np.int64), 0, nwa-1); wr_c = np.clip(np.floor(rg0/WR).astype(np.int64), 0, nwr-1)
        wm = lambda v: np.nanmean(v.reshape(nwa, WA, nwr, WR), axis=(1, 3))[wa_c, wr_c]
        SO = {'per-pixel OVS sigma0 n=0': wm(sp), 'per-pixel OVS gamma0 n=0.5': wm(sp/np.sqrt(cp)), 'per-pixel OVS gamma0 n=1': wm(sp/cp),
              'window integration sigma0 n=0': R['sB'], 'window integration gamma0 n=0.5': R['sB']/np.sqrt(R['cB']), 'window integration gamma0 n=1': R['sB']/R['cB']}
        cbO, ctlO, _, CO = cells(R, SO)
        table('[OVS] HV, per-pixel RTC on %dx%d subdivided facets vs window integration, identical cells' % (K, K), cbO, ctlO, CO)
        raise SystemExit
    if os.environ.get('RTC_ASF'):   # benchmark: ASF RTC (GAMMA, area-based gamma0, 12.5 m) sampled at the same facets
        from rasterio.warp import transform as wtf
        AD = '/mnt/c/grd file/AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1'
        ah = rasterio.open(AD + '_HV.tif'); AV = ah.read(1); LS = rasterio.open(AD + '.ls_map.tif').read(1)
        flat_ = LAT_TOP - rows/PPD; flon_ = LON_LEFT + cols/PPD
        ex, ny = wtf('EPSG:4326', ah.crs, flon_.tolist(), flat_.tolist())
        ai = ((np.array(ny) - ah.transform.f)/ah.transform.e).astype(int)
        aj = ((np.array(ex) - ah.transform.c)/ah.transform.a).astype(int)
        inA = (ai >= 0) & (ai < AV.shape[0]) & (aj >= 0) & (aj < AV.shape[1])
        asf = np.full(flat_.shape, np.nan)
        asf[inA] = np.where(LS[ai[inA], aj[inA]] == 1, AV[ai[inA], aj[inA]], np.nan)
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import Affine
        r0, r1, c0, c1 = br.min(), br.max() + 1, bc.min(), bc.max() + 1
        AC = np.zeros((r1 - r0, c1 - c0), np.float32)
        reproject(np.where(LS == 1, AV, 0).astype(np.float32), AC, src_transform=ah.transform, src_crs=ah.crs, src_nodata=0, dst_transform=Tr*Affine.translation(c0, r0), dst_crs='EPSG:4326', dst_nodata=0, resampling=Resampling.average)
        asf_c = AC[br - r0, bc - c0].astype(float); asf_c[asf_c <= 0] = np.nan
        say('[ASF] facets in frame %d / %d, valid (ls_map = 1) %d' % (inA.sum(), inA.size, np.isfinite(asf).sum()))
        SA = {'uncorrected gamma0 (theta_inc)': R['u_all']/np.cos(R['th']), 'per-pixel RTC gamma0 n=1': R['g_old'],
              'window integration gamma0 n=1': R['sB']/R['cB'], 'window integration gamma0 n=0.5': R['sB']/np.sqrt(R['cB']),
              'ASF RTC gamma0 (GAMMA)': asf, 'ASF RTC gamma0, cell average': asf_c}
        cbA, ctlA, _, CA = cells(R, SA)
        table('[ASF benchmark] HV, CCI 2010, identical cells and facets', cbA, ctlA, CA)
        if os.environ.get('RTC_BOOT'):   # spatial block bootstrap (95% CI) and spatial cross-validation of n
            key = cells.key; ncol = BIO.shape[1]; rng = np.random.default_rng(0); x = cbA; names = list(CA.keys())
            rr = lambda S: (lambda n, sx, sy, sxx, syy, sxy: (n*sxy - sx*sy)/np.sqrt((n*sxx - sx**2)*(n*syy - sy**2)))(*S.T)
            for Bk in (50, 100):   # block size in 100 m cells: 5 km and 10 km
                bi = np.unique((key//ncol//Bk)*100000 + (key % ncol)//Bk, return_inverse=True)[1]; nb = bi.max() + 1
                SS = {nm: np.stack([np.bincount(bi, w, minlength=nb) for w in (np.ones_like(x), x, CA[nm], x*x, CA[nm]**2, x*CA[nm])], 1) for nm in names}
                W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(1000)]).astype(float)
                Rb = {nm: rr(W @ SS[nm]) for nm in names}; ref = 'window integration gamma0 n=1'
                say('\n[BOOT] blocks of %d cells (%.0f km): %d blocks, 1000 resamples, r and 95%% CI' % (Bk, Bk/10, nb))
                for nm in names:
                    say('  %-34s r = %.3f  [%.3f, %.3f]' % ((nm, rr(SS[nm].sum(0)[None])[0]) + tuple(np.percentile(Rb[nm], [2.5, 97.5]))))
                for nm in names:
                    if nm != ref:
                        d = Rb[ref] - Rb[nm]; say('  r(window n=1) - r(%s): %+.3f  [%+.3f, %+.3f]' % ((nm, d.mean()) + tuple(np.percentile(d, [2.5, 97.5]))))
            cbN, ctlN, _, CN = cells(R, {'0': R['sB'], '0.5': R['sB']/np.sqrt(R['cB']), '1': R['sB']/R['cB']})
            kN = cells.key; fold = ((kN//ncol)//50 + (kN % ncol)//50) % 2; m15 = (ctlN >= INC_MIN) & (ctlN <= INC_MAX)
            for f in (0, 1):
                tr = (fold == f) & m15; te = fold != f; t15 = te & m15
                sl = {n_: np.polyfit(ctlN[tr], CN[n_][tr] - np.polyval(np.polyfit(cbN[tr], CN[n_][tr], 1), cbN[tr]), 1)[0] for n_ in CN}
                bn = min(sl, key=lambda n_: abs(sl[n_])); y = CN[bn]; b = np.polyfit(cbN[te], y[te], 1); res = y - np.polyval(b, cbN)
                say('[CV] fold %d (5 km checkerboard): train slopes %s -> n = %s; test r = %.3f, RMSE = %.3f dB, resid slope = %+.4f dB/deg' % (f, ' '.join('n=%s:%+.4f' % kv for kv in sl.items()), bn, np.corrcoef(cbN[te], y[te])[0, 1], np.sqrt(np.mean(res[te]**2)), np.polyfit(ctlN[t15], res[t15], 1)[0]))
        raise SystemExit
    if os.environ.get('RTC_SAME'):   # same quantity in all three methods; uncorrected has no DEM, so its cosine uses theta_inc
        ct = np.cos(R['th'])
        rows_ = []
        for n_, lab, pp in [(0, 'sigma0, area term only (n = 0)', R['s_old']), (0.5, 'gamma0, n = 0.5', R['h_old']), (1, 'gamma0, n = 1', R['g_old'])]:
            S2 = {'uncorrected / cos^n(theta_inc)': R['u_all']/ct**n_, 'per-pixel RTC': pp, 'window integration': R['sB']/R['cB']**n_}
            cb2, ctl2, _, C2 = cells(R, S2)
            rows_.append((n_, cb2, C2, table('[SAME] HV, same quantity in all methods: ' + lab, cb2, ctl2, C2)))
        # figure: rows = quantity, columns = method, same 6000 cells in every panel
        RL = {0: '$\\sigma^0$ (area term only, $n$ = 0)', 0.5: '$\\gamma^0$, $n$ = 0.5', 1: '$\\gamma^0$, $n$ = 1'}
        CL = ['Uncorrected', 'Per-pixel RTC', 'Window integration']
        rng = np.random.default_rng(0); sel = rng.choice(len(rows_[0][1]), min(6000, len(rows_[0][1])), replace=False)
        fig, axs = plt.subplots(3, 3, figsize=(15, 12.5), sharex=True, sharey=True)
        for i, (n_, cb2, C2, F2) in enumerate(rows_):
            for j, kx in enumerate(C2):
                a = axs[i, j]; b, rm, rr, _ = F2[kx]
                a.scatter(cb2[sel], C2[kx][sel], s=3, alpha=0.35, color='#3a7cc4', lw=0)
                xs = np.linspace(0, cb2.max(), 50); a.plot(xs, np.polyval(b, xs), color='#d9534f', lw=2)
                a.text(0.03, 0.97, 'r = %.3f\nRMSE = %.3f dB' % (rr, rm), transform=a.transAxes, va='top', fontsize=11, bbox=dict(fc='white', ec='0.8'))
                if i == 0: a.set_title(CL[j], fontsize=13)
                if j == 0: a.set_ylabel(RL[n_] + '\nHV, 100 m cell (dB)', fontsize=11)
                if i == 2: a.set_xlabel('CCI AGB 2010 (Mg/ha)')
        nP, cbP, CP, FP = rows_[1]                                         # poster panel: gamma0, n = 0.5
        with plt.rc_context({'font.size': 16, 'axes.titlesize': 17}):
            figP, axP = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
            for j, kx in enumerate(CP):
                b, rm, rr, _ = FP[kx]; axP[j].scatter(cbP[sel], CP[kx][sel], s=4, alpha=0.35, color='#3a7cc4', lw=0)
                xs = np.linspace(0, cbP.max(), 50); axP[j].plot(xs, np.polyval(b, xs), color='#d9534f', lw=3)
                axP[j].set_title(['(a) Uncorrected', '(b) Per-pixel RTC', '(c) Window integration'][j], loc='left')
                axP[j].text(0.04, 0.96, '$r$ = %.3f\nRMSE = %.3f dB' % (rr, rm), transform=axP[j].transAxes, va='top', fontsize=16, bbox=dict(fc='white', ec='0.7'))
                axP[j].set_xlabel('CCI AGB 2010 (Mg ha$^{-1}$)')
            axP[0].set_ylabel('HV $\\gamma^0$, $n$ = 0.5 (dB)')
            figP.tight_layout(); figP.savefig(os.path.join(OUT, 'poster_F1_same%s.png' % _SFX), dpi=220, facecolor='white'); plt.close(figP)
        fig.suptitle('Same quantity in every method: rows = quantity, columns = method; %d cells fitted, %d shown' % (len(rows_[0][1]), len(sel)), fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F1_same_quantity.png'), dpi=200, facecolor='white'); plt.close()
        say('\n[saved] final_eq14_SAME_log.txt, F1_same_quantity.png, poster_F1_same.png in %s' % OUT); LOG.close(); sys.exit(0)

    # ---- T1 figure ----
    rng = np.random.default_rng(0); sel = rng.choice(len(cb), min(6000, len(cb)), replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for a, kx, t in zip(axs, ['uncorrected sigma0', 'current code gamma0', 'Eq14 gamma0 n=0.5'],
                        ['(a) Uncorrected $\\sigma^0$', '(b) Per-pixel RTC $\\gamma^0$ (n = 1)', '(c) Window integration (Eq. 14), n = 0.5']):
        b, rm, rr, _ = F[kx]; a.scatter(cb[sel], C[kx][sel], s=3, alpha=0.35, color='#3a7cc4', lw=0)
        xs = np.linspace(0, cb.max(), 50); a.plot(xs, np.polyval(b, xs), color='#d9534f', lw=2)
        a.set_title(t, loc='left', fontsize=11)
        a.text(0.03, 0.97, 'y = %.3f + %.5f x\nr = %.3f, RMSE = %.3f dB\nn = %d cells' % (b[1], b[0], rr, rm, len(cb)),
               transform=a.transAxes, va='top', fontsize=9, bbox=dict(fc='white', ec='0.8'))
        a.set_xlabel('CCI AGB 2010 (Mg/ha)')
    axs[0].set_ylabel('Backscatter, HV, 100 m cell (dB)')
    fig.suptitle('F1  One regression line per method, fitted on all 100 m cells (dB vs AGB); 6000 cells shown', fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F1_regression_eq14.png'), dpi=200, facecolor='white'); plt.close()

    # ---- T2 residual angle trend per theta_inc bin ----
    m15 = (ctl >= INC_MIN) & (ctl <= INC_MAX)
    keys = ['uncorrected sigma0', 'current code gamma0', 'Eq14 sigma0 n=0', 'Eq14 gamma0 n=0.5', 'Eq14 gamma0 n=1']
    say('\n[T2] slope of AGB-regression residual (dB) vs theta_loc inside 1-deg theta_inc bins (dB/deg)')
    say('     Simard Fig. 10 (UAVSAR HV, N1 gamma0 vs theta_i): +0.078 (34-35), +0.059 (39-40), +0.050 (44-45)')
    ib = np.arange(np.floor(cinc.min()), np.ceil(cinc.max()) + 1, 1.0); rows_t = []
    say('     theta_inc   n_cells  ' + ''.join('%22s' % kx for kx in keys))
    for j in range(len(ib)-1):
        m = m15 & (cinc >= ib[j]) & (cinc < ib[j+1])
        if m.sum() < 1000: continue
        v = [np.polyfit(ctl[m], F[kx][3][m], 1)[0] for kx in keys]; rows_t.append((ib[j] + 0.5, v))
        say('     %4.0f-%-4.0f  %8d  ' % (ib[j], ib[j+1], m.sum()) + ''.join('%+22.4f' % x for x in v))
    tb = np.arange(15, 71, 5); tc = 0.5*(tb[:-1] + tb[1:]); ti = np.digitize(ctl, tb) - 1
    sty = {'uncorrected sigma0': ('0.45', '-', 'o'), 'current code gamma0': ('#1f77b4', '-', 's'), 'Eq14 sigma0 n=0': ('#2ca02c', '--', '^'),
           'Eq14 gamma0 n=0.5': ('#d62728', '-', 'D'), 'Eq14 gamma0 n=1': ('#9467bd', ':', 'v')}
    LAB = {'uncorrected sigma0': 'Uncorrected $\\sigma^0$', 'current code gamma0': 'Per-pixel RTC $\\gamma^0$ ($n$ = 1)',
           'Eq14 sigma0 n=0': 'Window integration $\\sigma^0$ ($n$ = 0)', 'Eq14 gamma0 n=0.5': 'Window integration $\\gamma^0$ ($n$ = 0.5)',
           'Eq14 gamma0 n=1': 'Window integration $\\gamma^0$ ($n$ = 1)'}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for kx in keys:
        c, ls, mk = sty[kx]
        axs[0].plot(tc, [np.median(F[kx][3][ti == i]) if (ti == i).sum() > 50 else np.nan for i in range(len(tc))], color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=LAB[kx])
        axs[1].plot([r_[0] for r_ in rows_t], [r_[1][keys.index(kx)] for r_ in rows_t], color=c, ls=ls, marker=mk, ms=5, lw=1.6, label=LAB[kx])
    axs[1].plot([34.5, 39.5, 44.5], [0.078, 0.059, 0.050], 'k*', ms=12, label='Simard Fig. 10 (UAVSAR, N1)')
    for a in axs: a.axhline(0, color='k', lw=0.6)
    axs[0].set_xlabel('Cell-mean local incidence angle $\\theta_{loc}$ (deg)'); axs[0].set_ylabel('Median residual of the AGB regression (dB)')
    axs[0].set_title('(a) Residual vs $\\theta_{loc}$, 100 m cells', loc='left', fontsize=10)
    axs[1].set_xlabel('$\\theta_{inc}$ bin centre (deg)  [paper: look angle]'); axs[1].set_ylabel('Residual slope vs $\\theta_{loc}$ (dB/deg)')
    axs[1].set_title('(b) Residual slope per 1$\\degree$ bin, compared with the paper', loc='left', fontsize=10)
    axs[1].set_xlim(33, 46)
    axs[0].legend(fontsize=8, frameon=False); axs[1].legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('F4  %s HV: angle dependence left after correction (0 = no dependence)' % SID, fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'F4_angle_trend_eq14.png'), dpi=200, facecolor='white'); plt.close()

    if os.environ.get('RTC_POSTER'):
        # ---------- poster versions (large fonts) ----------
        CG = cells(R, {'u': R['u_all']/np.cos(R['th']), 'g': R['sB']/np.cos(R['th']), 'su': R['u_all'], 'sg': R['sB']})
        ctlP = CG[1]; mP = (ctlP >= INC_MIN) & (ctlP <= INC_MAX)
        plt.rcParams.update({'font.size': 16, 'axes.titlesize': 17, 'legend.fontsize': 14})
        # P1 regression
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
        for a_, kx, t in zip(axs, ['uncorrected sigma0', 'current code gamma0', 'Eq14 gamma0 n=0.5'],
                             ['(a) Uncorrected σ$^0$', '(b) Per-pixel RTC, n = 1', '(c) Window integration, n = 0.5']):
            b, rm, rr, _ = F[kx]; a_.scatter(cb[sel], C[kx][sel], s=4, alpha=0.35, color='#3a7cc4', lw=0)
            xs = np.linspace(0, cb.max(), 50); a_.plot(xs, np.polyval(b, xs), color='#d9534f', lw=3)
            a_.set_title(t, loc='left')
            a_.text(0.04, 0.96, 'r = %.3f\nRMSE = %.3f dB' % (rr, rm), transform=a_.transAxes, va='top', fontsize=16, bbox=dict(fc='white', ec='0.7'))
            a_.set_xlabel('CCI AGB 2010 (Mg ha$^{-1}$)')
        axs[0].set_ylabel('HV backscatter (dB)')
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'poster_F1%s.png' % _SFX), dpi=220, facecolor='white'); plt.close()
        # P2 distributions
        fig, axs = plt.subplots(1, 2, figsize=(16, 5.4)); hb = np.arange(-20, -4, 0.2)
        for a_, (k1, k2, t) in zip(axs, [('su', 'sg', '(a) $\\sigma^0$'), ('u', 'g', '(b) $\\gamma^0$ = $\\sigma^0$ / $\\cos\\theta_{inc}$')]):
            for kx, c, lab in [(k1, '#1f77b4', 'Before correction'), (k2, '#d62728', 'After correction')]:
                y_ = CG[3][kx][mP]; a_.hist(y_, hb, density=True, alpha=0.5, color=c, label='%s: median %.2f, std %.2f dB' % (lab, np.median(y_), y_.std()))
                a_.axvline(np.median(y_), color=c, ls='--', lw=1.5)
            a_.set_title(t, loc='left'); a_.set_xlabel('HV backscatter, 100 m (dB)'); a_.legend(frameon=False, loc='upper left', fontsize=13)
        axs[0].set_ylabel('Probability density')
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'poster_F2%s.png' % _SFX), dpi=220, facecolor='white'); plt.close()
        say('[poster F2] %d cells: sigma0 %.2f/%.2f -> %.2f/%.2f | gamma0 %.2f/%.2f -> %.2f/%.2f (median/std)' % ((mP.sum(),) + tuple(
            v for kx in ('su', 'sg', 'u', 'g') for v in (np.median(CG[3][kx][mP]), CG[3][kx][mP].std()))))
        # P4 angle trend
        PL = {'uncorrected sigma0': 'Uncorrected', 'current code gamma0': 'Per-pixel RTC, $n$ = 1',
              'Eq14 sigma0 n=0': 'Window integration, $n$ = 0', 'Eq14 gamma0 n=0.5': 'Window integration, $n$ = 0.5', 'Eq14 gamma0 n=1': 'Window integration, $n$ = 1'}
        fig, axs = plt.subplots(1, 2, figsize=(16, 6.0))
        for kx in keys:
            c, ls, mk = sty[kx]
            axs[0].plot(tc, [np.median(F[kx][3][ti == i]) if (ti == i).sum() > 50 else np.nan for i in range(len(tc))], color=c, ls=ls, marker=mk, ms=7, lw=2.5, label=PL[kx])
            axs[1].plot([r_[0] for r_ in rows_t], [r_[1][keys.index(kx)] for r_ in rows_t], color=c, ls=ls, marker=mk, ms=8, lw=2.5)
        axs[1].plot([34.5, 39.5, 44.5], [0.078, 0.059, 0.050], 'k*', ms=18, label='Simard et al. (2016), Fig. 10')
        for a_ in axs: a_.axhline(0, color='k', lw=0.8)
        axs[1].axhline(0, color='k', lw=2.2)
        if not _TH: axs[1].text(33.4, 0.006, '0 = no angle dependence (ideal)', fontsize=15, va='bottom', fontweight='bold')
        axs[0].set_xlabel('Local incidence angle $\\theta_{loc}$ (°)'); axs[0].set_ylabel('Residual (dB)')
        axs[0].set_title('(a) Residual of the AGB regression', loc='left')
        axs[1].set_xlabel('$\\theta_{inc}$ bin (°)'); axs[1].set_ylabel('Residual slope (dB/°)')
        axs[1].set_title('(b) Residual slope per 1° bin of $\\theta_{inc}$', loc='left'); axs[1].set_xlim(33, 46)
        axs[0].legend(frameon=False, loc='lower left', fontsize=13); axs[1].legend(frameon=False, loc='lower right', fontsize=13)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'poster_F4%s.png' % _SFX), dpi=220, facecolor='white'); plt.close()
        say('[saved] poster_F1.png, poster_F2.png, poster_F4.png in %s' % OUT)
    if os.environ.get('RTC_FIGS_ONLY'):
        say('\n[saved] F1_regression_eq14.png, F4_angle_trend_eq14.png (figures only) in %s' % OUT); LOG.close(); sys.exit(0)
    # ---- T3 ----
    cbT, ctlT, _, CT = cells(R, {'current code gamma0 (n=1, 24.5% px)': R['g_old'], 'Eq14 n=1, all pixels': R['sB']/R['cB'],
                                 'Eq14 n=1, only code-valid pixels': R['sB_v']/R['cB']})
    table('[T3] is the r gap due to using only the code-valid pixels?', cbT, ctlT, CT)
    # ---- T4 ----
    say('\n[T4] layover facets valid in the current code: %d ; plan A windows with a layover facet: %d of %d (%.3f%%)'
        % (R['cnt_layfac_valid'], R['n_win_lay'], R['n_win'], 100*R['n_win_lay']/max(R['n_win'], 1)))
    cbL, ctlL, _, CL = cells(R, {'current code gamma0': R['g_old'], 'current code, layover px removed': R['g_nl'],
                                 'Eq14 n=0.5': R['sB']/np.sqrt(R['cB']), 'Eq14 n=0.5, layover windows removed': R['sB_nl']/np.sqrt(R['cB'])})
    table('[T4] effect of masking layover', cbL, ctlL, CL)
    del R, S, C, F, CT, CL; gc.collect()

    # ---- T5 HH ----
    hh = sorted(glob.glob(os.path.join(BASE, 'ALOS_HH_uncorrected*.bin')))
    if hh and os.path.getsize(hh[0]) == NA*NR*4:
        bsH = np.asarray(np.memmap(hh[0], dtype=np.float32, mode='r', shape=(NA, NR)))
        RH = run(rg0, bsH)
        SH = {'uncorrected sigma0': RH['u_all'], 'current code gamma0': RH['g_old']}
        for n in (0.0, 0.25, 0.5, 0.75, 1.0):
            SH['Eq14 n=%.2f' % n] = RH['sB']/RH['cB']**n
        cbH, ctlH, _, CH = cells(RH, SH)
        table('[T5] HH (%s): which n removes the angle trend?' % os.path.basename(hh[0]), cbH, ctlH, CH)
        del RH, SH, CH, bsH; gc.collect()
    else:
        say('\n[T5] no HH file ALOS_HH_uncorrected*.bin (NA x NR float32) in %s ; .bin files present: %s'
            % (BASE, [os.path.basename(f) for f in glob.glob(os.path.join(BASE, '*.bin'))]))

# ---- T6 geoid ----
lat_c = float(LAT_TOP - rows.mean()/PPD); lon_c = float(LON_LEFT + cols.mean()/PPD); N = None
if os.environ.get('RTC_GEOID_N'):
    N = float(os.environ['RTC_GEOID_N']); srcN = 'RTC_GEOID_N'
else:
    try:
        import pyproj
        pyproj.network.set_network_enabled(True)
        t_ = pyproj.Transformer.from_crs('EPSG:4326+3855', 'EPSG:4979', always_xy=True)
        z = t_.transform(lon_c, lat_c, 0.0)[2]
        if np.isfinite(z) and abs(z) > 1e-3: N = float(z); srcN = 'pyproj EGM2008'
    except Exception as e:
        say('\n[T6] pyproj geoid lookup failed: %s' % e)
if N is None:
    say('[T6] geoid undulation unknown; re-run with RTC_GEOID_N=<metres> (EGM2008 at %.3f N, %.3f E)' % (lat_c, lon_c))
else:
    thc = geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0)
    drg = -N*np.cos(thc)/DR
    say('\n[T6] EGM2008 N = %+.2f m at %.3f N %.3f E (%s): range shift %+.2f to %+.2f pixels (%+.1f m slant)'
        % (N, lat_c, lon_c, srcN, drg.min(), drg.max(), drg.mean()*DR))
    RG = run(rg0 + drg, bs); del thc, drg
    cbG, ctlG, _, CG = cells(RG, {'uncorrected sigma0': RG['u_all'], 'current code gamma0': RG['g_old'],
                                  'Eq14 sigma0 n=0': RG['sB'], 'Eq14 gamma0 n=0.5': RG['sB']/np.sqrt(RG['cB'])})
    table('[T6] geoid shift applied twice (2N) on top of RTC_GEOID_SHIFT: consistency check, expected to be worse' if GEOID_N else '[T6] ellipsoid heights (geoid-corrected range positions)', cbG, ctlG, CG)
say('\n[saved] final_eq14_log.txt, F1_regression_eq14.png, F4_angle_trend_eq14.png in %s' % OUT)
LOG.close()
