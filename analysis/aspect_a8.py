#!/usr/bin/env python3
"""A8: residual of the HV gamma0 (n = 0.5) vs CCI AGB 2010 regression against terrain aspect relative to the radar look
direction and against slope, for the three methods of the [SAME] n = 0.5 table (uncorrected / cos^0.5(theta_inc),
per-pixel RTC, window integration).

Input : out_ellip/a7_cells.npz (cell dB values, same cells as agb_sat_a7.py), written by
        RTC_GEOID_SHIFT=-27.32 RTC_SAME=1 RTC_A7=1 python3 final_eq14.py
        facet_results_FINAL2.npz (facet DEM rows/cols), GLO30_ALOS_merged.bin, leader file (heading)
Output: out_ellip/aspect_a8_log.txt (all numbers), out_ellip/aspect_a8_bins.csv (1-D and 2-D bin tables)

Terrain per CCI cell (100 m): DEM gradient p = dZ/dx (east), q = dZ/dy (north) of every facet, from the same 3 x 3
unweighted difference (/6) as final_eq14.py; cell gradient = mean of the facet gradients over all facets mapped to the cell
(planar fit). Cell slope = arctan |mean gradient|, cell aspect = downslope direction of the mean gradient (clockwise from north).
Relative aspect phi = aspect - (look azimuth + 180 deg), wrapped to (-180, 180]: phi = 0 faces the radar, |phi| = 180 faces away
(right-looking, look azimuth = heading + 90 deg). Range-direction slope tau_r = arctan(tan(s) cos(phi)) (> 0 towards the radar),
azimuth-direction slope tau_a = arctan(tan(s) sin(phi)).
Residual: y - (a + b AGB), OLS of each method over all cells (as [CHK] in agb_sat_a7_log.txt).
Facing - away difference D = mean residual (|phi| < 90) - mean residual (|phi| > 90) (also |phi| < 45 vs > 135), cells with
slope >= 5 deg and per 5-deg slope class. Uncertainty: spatial block bootstrap, 100 x 100 CCI cells (the "10 km" blocks of
final_eq14.py [BOOT]; 8.9 km N-S x 6.3 km E-W at 44.5 N), 1000 resamples, seed 0; the AGB regression is refitted in each
resample from block sums. 50-cell ("5 km") blocks as sensitivity.
"""
import os
import sys
import numpy as np

BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geolocate, area_correction  # noqa: E402

SID = 'ALPSRP237630880'
D_ = os.path.join(BASE, SID + '-L1.1') + '/'
LED = D_ + 'LED-' + SID + '-H1.1__A'
IMG = D_ + 'IMG-HH-' + SID + '-H1.1__A'
_rl = int(open(IMG, 'rb').read(720)[186:192])
NA, NR = (os.path.getsize(IMG) - 720)//_rl, (_rl - 412)//8
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
OUT = os.path.join(BASE, 'scripts', os.environ.get('RTC_OUTDIR', 'out_ellip'))
FN = os.path.join(OUT, 'a7_cells.npz')
LOG = open(os.path.join(OUT, 'aspect_a8_log.txt'), 'w')


def say(s=''):
    print(s); LOG.write(s + '\n'); LOG.flush()


# ---------------------------------------------------------------- heading (as final_eq14.py, fitted line times)
t0, h, P, V = leader_file.read_platform_position_record(LED)
tf, te, _dt = img_file.read_line_times_fit(IMG, NR, NA)
Pm, Vm = geolocate.interpolate_position_velocity_vec(np.array([0.5*(tf + te)]), t0, h, P, V); Pm = Pm[0]; Vm = Vm[0]
la = np.arctan2(Pm[2], np.hypot(Pm[0], Pm[1])*(1 - geolocate.WGS84_E2)); lo = np.arctan2(Pm[1], Pm[0])
HD = np.arctan2(Vm @ np.array([-np.sin(lo), np.cos(lo), 0]),
                Vm @ np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)]))
LOOK = HD + np.pi/2
say('[A8] scene %s; heading %.3f deg, look azimuth (right-looking) %.3f deg, facing-radar aspect %.3f deg (clockwise from N)'
    % (SID, np.degrees(HD), np.degrees(LOOK) % 360, (np.degrees(LOOK) + 180) % 360))

# ---------------------------------------------------------------- facet gradients (identical stencil to final_eq14.py)
g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; del g
nd = int(round(np.sqrt(os.path.getsize(DEMFILE)/4)))
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(nd, nd)
lat_rows = (LAT_TOP - rows/PPD).astype(np.float32)
dlat_m = np.float32(area_correction.DEG_M/PPD)
dlon_m = (area_correction.DEG_M/PPD*np.cos(np.radians(lat_rows))).astype(np.float32)
Z = {k: dem[rows + dr, cols + dc] for k, (dr, dc) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], 1)}
zmin = np.minimum.reduce([Z[k] for k in Z] + [dem[rows, cols]])
p = (Z[3] + Z[5] + Z[8] - Z[1] - Z[4] - Z[6])/(6.0*dlon_m)       # Z3+Z6+Z9-Z1-Z4-Z7 in final_eq14.py numbering
q = (Z[1] + Z[2] + Z[3] - Z[6] - Z[7] - Z[8])/(6.0*dlat_m)       # Z1+Z2+Z3-Z7-Z8-Z9
del Z, dem
okD = zmin > -100.0
say('[A8] facets %d; facets with a DEM value <= -100 m in the 3 x 3 stencil (excluded): %d' % (len(rows), (~okD).sum()))

# aspect convention check against final_eq14.py (np.pi - arctan(q/p) + pi/2 sign(p))
ps = np.where(p == 0, 1.0, p)
asp_f = np.where(p == 0, np.where(q > 0, np.pi, 0.0), np.pi - np.arctan(q/ps) + (np.pi/2)*np.sign(ps))
asp_m = np.arctan2(-p, -q) % (2*np.pi)
dA = np.abs(np.angle(np.exp(1j*(asp_f - asp_m))))
flat = (p == 0) & (q == 0)
say('[CHK] facet aspect: final_eq14.py formula vs arctan2(-p, -q) (downslope, clockwise from N): max |diff| %.2e rad over %d non-flat facets'
    % (dA[~flat].max(), (~flat).sum()))
del ps, asp_f, asp_m, dA, flat

# ---------------------------------------------------------------- facets -> CCI cells (same mapping as final_eq14.py)
C = np.load(FN)
tr = C['tr']; ncol = int(C['ncol'])
br = ((LAT_TOP - rows/PPD - tr[5])/tr[4]).astype(int); bc = ((LON_LEFT + cols/PPD - tr[2])/tr[0]).astype(int)
fkey = (br*ncol + bc)[okD]; p = p[okD].astype(np.float64); q = q[okD].astype(np.float64)
sf = np.degrees(np.arctan(np.hypot(p, q)))
uk, inv = np.unique(fkey, return_inverse=True); nfc = np.bincount(inv)
mp = np.bincount(inv, p)/nfc; mq = np.bincount(inv, q)/nfc; msf = np.bincount(inv, sf)/nfc
mg = np.bincount(inv, np.hypot(p, q))/nfc
del p, q, sf, inv, fkey, rows, cols, br, bc, lat_rows, dlon_m, zmin, okD

key = C['key']; ix = np.searchsorted(uk, key); ix = np.clip(ix, 0, len(uk) - 1)
found = uk[ix] == key
say('[A8] cells in a7_cells.npz %d; matched to DEM facets %d (unmatched %d)' % (len(key), found.sum(), (~found).sum()))
names = [str(s) for s in C['names']]
SHORT = dict(zip(names, ['uncorrected', 'per-pixel RTC', 'window integration']))
x = C['agb'].astype(float)[found]; Y = {nm: C['Y'][i].astype(float)[found] for i, nm in enumerate(names)}
key = key[found]; ix = ix[found]; tl = C['tl'].astype(float)[found]; N = len(x)
gx, gy = mp[ix], mq[ix]
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = np.degrees(np.arctan2(-gx, -gy)) % 360
phi = (aspect - (np.degrees(LOOK) + 180.0) + 180.0) % 360 - 180.0           # (-180, 180], 0 = facing the radar
phi = np.where(phi == -180.0, 180.0, phi)
tau_r = np.degrees(np.arctan(np.tan(np.radians(slope))*np.cos(np.radians(phi))))
tau_a = np.degrees(np.arctan(np.tan(np.radians(slope))*np.sin(np.radians(phi))))
coh = np.hypot(gx, gy)/np.maximum(mg[ix], 1e-12)
say('[A8] cell slope (planar fit) median %.2f, p90 %.2f, p99 %.2f, max %.2f deg; mean facet slope median %.2f deg; corr(planar, mean facet) %.3f'
    % (np.median(slope), np.percentile(slope, 90), np.percentile(slope, 99), slope.max(), np.median(msf[ix]), np.corrcoef(slope, msf[ix])[0, 1]))
say('[A8] aspect coherence |mean grad| / mean |grad|: median %.3f (slope >= 5 deg: %.3f); facets per cell median %d'
    % (np.median(coh), np.median(coh[slope >= 5]), np.median(nfc[ix])))
say('[CHK] corr(cell tau_r, cell-mean theta_loc) = %.3f (expected strongly negative: facing slopes have small theta_loc)' % np.corrcoef(tau_r, tl)[0, 1])

# ---------------------------------------------------------------- residuals
RES = {}
say('\n[CHK] full-range OLS (must equal [CHK] in agb_sat_a7_log.txt when no cell is unmatched)')
for nm in names:
    b = np.polyfit(x, Y[nm], 1); RES[nm] = Y[nm] - np.polyval(b, x)
    say('  %-20s r = %.3f  slope = %.5f  intercept = %+.3f  RMSE = %.3f' % (SHORT[nm], np.corrcoef(x, Y[nm])[0, 1], b[0], b[1], np.sqrt(np.mean(RES[nm]**2))))

csv = open(os.path.join(OUT, 'aspect_a8_bins.csv'), 'w')
csv.write('table,bin_lo,bin_hi,bin2_lo,bin2_hi,n,agb_mean,slope_mean,' + ','.join('%s_mean_dB,%s_sd_dB' % ((SHORT[nm].replace(' ', '_'),)*2) for nm in names) + '\n')


def bintab(tag, v, edges, sel=None, unit='deg', v2=None, e2=None):
    """1-D (or 2-D with v2/e2) bin table of residual mean / sd (ddof = 1) per method; n = cells"""
    sel = np.ones(N, bool) if sel is None else sel
    loop2 = [(None, None)] if v2 is None else list(zip(e2[:-1], e2[1:]))
    for lo2, hi2 in loop2:
        m2 = sel if lo2 is None else sel & (v2 >= lo2) & (v2 < hi2)
        if lo2 is not None:
            say('  -- slope %g-%g deg' % (lo2, hi2))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = m2 & (v >= lo) & (v < hi); n = int(m.sum())
            if n == 0:
                continue
            row = '  %7.1f to %-7.1f %8d %8.1f %6.2f' % (lo, hi, n, x[m].mean(), slope[m].mean())
            cr = '%s,%g,%g,%s,%s,%d,%.2f,%.3f' % (tag, lo, hi, '' if lo2 is None else '%g' % lo2, '' if hi2 is None else '%g' % hi2, n, x[m].mean(), slope[m].mean())
            for nm in names:
                r = RES[nm][m]; sd = r.std(ddof=1) if n > 1 else np.nan
                row += '   %+7.3f / %5.3f' % (r.mean(), sd); cr += ',%.4f,%.4f' % (r.mean(), sd)
            say(row + ('   (n < 100)' if n < 100 else '')); csv.write(cr + '\n')


HDR = '  %-18s %8s %8s %6s' % ('bin (%s)', 'n', 'AGB', 'slope') + ''.join('   %-17s' % ('%s m/sd' % SHORT[nm][:12]) for nm in names)
phi_b = np.where(phi >= 165.0, phi - 360.0, phi)              # put 165..180 with -195..-165 so that the away bin is |phi| > 165
E_PHI = np.arange(-195.0, 165.0 + 1e-9, 30.0)
smax = np.ceil(slope.max()/5.0)*5.0
E_S = np.arange(0.0, smax + 1e-9, 5.0)

say('\n[ASPECT] relative aspect phi, 30-deg bins centred on 0 (facing radar) ... +/-180 (away); bin -195 to -165 = |phi| > 165 (away). '
    'residual mean / sd (dB); cells with slope >= 5 deg')
say(HDR % 'phi'); bintab('aspect_slope>=5', phi_b, E_PHI, slope >= 5)
say('\n[ASPECT] same, all cells (aspect of near-flat cells is noise)')
say(HDR % 'phi'); bintab('aspect_all', phi_b, E_PHI)
say('\n[SLOPE] cell slope, 5-deg bins, all cells')
say(HDR % 'slope'); bintab('slope', slope, E_S)
say('\n[TAU_R] range-direction slope tau_r (> 0 = towards the radar), 5-deg bins, all cells')
er = np.arange(np.floor(tau_r.min()/5)*5, np.ceil(tau_r.max()/5)*5 + 1e-9, 5.0)
say(HDR % 'tau_r'); bintab('tau_r', tau_r, er)
say('\n[TAU_A] azimuth-direction slope tau_a (sign of sin(phi); phi > 0 = aspect clockwise of the facing direction), 5-deg bins, all cells')
ea = np.arange(np.floor(tau_a.min()/5)*5, np.ceil(tau_a.max()/5)*5 + 1e-9, 5.0)
say(HDR % 'tau_a'); bintab('tau_a', tau_a, ea)
E_S2 = np.array([0, 5, 10, 15, 20, 25, max(30.0, smax)])
say('\n[2D] relative aspect (30 deg) x slope class; residual mean / sd (dB)')
say(HDR % 'phi'); bintab('aspect_x_slope', phi_b, E_PHI, None, v2=slope, e2=E_S2)
csv.close()

# ---------------------------------------------------------------- harmonic fit (descriptive): resid = c0 + c1 cos(phi) + c2 sin(phi)
say('\n[HARM] OLS resid = c0 + c1 cos(phi) + c2 sin(phi) per slope class (c1 > 0: facing slopes brighter than the regression); dB')
cph, sph = np.cos(np.radians(phi)), np.sin(np.radians(phi))
for lo, hi in [(0, 5), (5, 1e9), (5, 10), (10, 15), (15, 20), (20, 25), (25, 1e9)]:
    m = (slope >= lo) & (slope < hi); X = np.stack([np.ones(m.sum()), cph[m], sph[m]], 1)
    s = '  slope %2g-%-4s n = %6d:' % (lo, '%g' % hi if hi < 1e9 else 'max', m.sum())
    for nm in names:
        c, *_ = np.linalg.lstsq(X, RES[nm][m], rcond=None)
        s += '   %s c1 %+.3f c2 %+.3f' % (SHORT[nm][:12], c[1], c[2])
    say(s)

# ---------------------------------------------------------------- facing - away difference, spatial block bootstrap
CLS = [('slope >= 5', 5, 1e9), ('slope 0-5 (control)', 0, 5), ('slope 5-10', 5, 10), ('slope 10-15', 10, 15),
       ('slope 15-20', 15, 20), ('slope 20-25', 20, 25), ('slope >= 25', 25, 1e9)]
DEF = [('|phi|<90 vs >90', np.abs(phi) < 90, np.abs(phi) > 90), ('|phi|<45 vs >135', np.abs(phi) < 45, np.abs(phi) > 135)]
pairs = [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]


def dstat(S, nm, cl, df):
    """D from (resampled) sums; regression refitted from all-cell sums"""
    n, sx, sxx, sy, sxy = S['n'], S['x'], S['xx'], S['y' + nm], S['xy' + nm]
    b = (n*sxy - sx*sy)/(n*sxx - sx**2)
    mf = (S['yF' + cl + df + nm] - b*S['xF' + cl + df])/S['nF' + cl + df]
    ma = (S['yA' + cl + df + nm] - b*S['xA' + cl + df])/S['nA' + cl + df]
    a = (sy - b*sx)/n
    return (mf - a) - (ma - a), mf - a, ma - a


for Bk in (100, 50):
    bi = np.unique((key//ncol//Bk)*100000 + (key % ncol)//Bk, return_inverse=True)[1]; nb = bi.max() + 1
    bcn = lambda w: np.bincount(bi, w, minlength=nb).astype(float)
    S = {'n': bcn(np.ones(N)), 'x': bcn(x), 'xx': bcn(x*x)}
    for nm in names:
        S['y' + nm] = bcn(Y[nm]); S['xy' + nm] = bcn(x*Y[nm])
    for cl, lo, hi in CLS:
        ms = (slope >= lo) & (slope < hi)
        for df, mF, mA in DEF:
            for side, mm in (('F', ms & mF), ('A', ms & mA)):
                S['n' + side + cl + df] = bcn(mm.astype(float)); S['x' + side + cl + df] = bcn(x*mm)
                for nm in names:
                    S['y' + side + cl + df + nm] = bcn(Y[nm]*mm)
    rng = np.random.default_rng(0)
    Wt = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(1000)]).astype(float)
    S1 = {k: v.sum() for k, v in S.items()}; Sb = {k: Wt @ v for k, v in S.items()}
    say('\n[BOOT] facing - away mean residual D (dB); blocks of %d x %d CCI cells (%s): %d blocks, 1000 resamples, seed 0; '
        'estimate [2.5, 97.5 percentile]; AGB regression refitted per resample' % (Bk, Bk, '"10 km"' if Bk == 100 else '"5 km"', nb))
    for df, _, _ in DEF:
        say('  definition %s' % df)
        for cl, lo, hi in CLS:
            nF, nA = int(S1['nF' + cl + df]), int(S1['nA' + cl + df])
            if min(nF, nA) == 0:
                say('    %-20s nF %d nA %d: empty side' % (cl, nF, nA)); continue
            s = '    %-20s nF %6d nA %6d:' % (cl, nF, nA)
            Db = {}
            for nm in names:
                d0, f0, a0 = dstat(S1, nm, cl, df); Db[nm] = dstat(Sb, nm, cl, df)[0]
                s += '  %s %+.3f [%+.3f, %+.3f] (F %+.3f A %+.3f)' % (SHORT[nm][:12], d0, *np.nanpercentile(Db[nm], [2.5, 97.5]), f0, a0)
            say(s)
            s = '    %-20s paired |D| reduction:' % ''
            for a_, c_ in pairs:
                dd = np.abs(Db[a_]) - np.abs(Db[c_]); d1 = abs(dstat(S1, a_, cl, df)[0]) - abs(dstat(S1, c_, cl, df)[0])
                s += '  |D %s| - |D %s| %+.3f [%+.3f, %+.3f]' % (SHORT[a_][:12], SHORT[c_][:12], d1, *np.nanpercentile(dd, [2.5, 97.5]))
            say(s)
say('\n[A8] done')
LOG.close()
