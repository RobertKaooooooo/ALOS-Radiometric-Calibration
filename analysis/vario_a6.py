#!/usr/bin/env python3
"""A6: empirical variogram of the AGB-regression residual of window-integration gamma0 (n = 0.5), HV, CCI 2010 100 m cells,
and a check of the block size (5 km, 10 km) used by the spatial block bootstrap in final_eq14.py (RTC_BOOT).

Input : out_ellip/vario_cells.npz written by
        RTC_GEOID_SHIFT=-27.32 RTC_ASF=1 RTC_VARIO=1 python3 final_eq14.py
Output: out_ellip/vario_a6_log.txt (all numbers), out_ellip/vario_a6_table.csv (binned variogram)

Variogram: all pairs on the CCI grid via FFT (Marcotte 1996, Comput. Geosci. 22(10), 1175-1186),
gamma(h) = sum I(x)I(x+h)[z(x+h)-z(x)]^2 / (2 N(h)); lags converted to metres at the mean latitude
(CCI pixel 1/1125 deg: about 99 m N-S and 70 m E-W at 44.5 N).
"""
import os
import sys
import numpy as np
from scipy.optimize import curve_fit

OUT = os.path.join(os.environ.get('RTC_BASE', '/mnt/c/grd file'), 'scripts', os.environ.get('RTC_OUTDIR', 'out_ellip'))
D = np.load(os.path.join(OUT, 'vario_cells.npz'))
LOG = open(os.path.join(OUT, 'vario_a6_log.txt'), 'w')


def say(s=''):
    print(s); LOG.write(s + '\n'); LOG.flush()


ncol = int(D['ncol']); tr = D['tr']
names = [str(s) for s in D['names_asf']]


def resid(agb, y):
    b = np.polyfit(agb, y, 1)
    return y - np.polyval(b, agb), b


def metric(key):
    r = key//ncol; c = key % ncol
    lat0 = np.radians(np.mean(tr[5] + (r + 0.5)*tr[4]))
    mlat = 111132.954 - 559.822*np.cos(2*lat0) + 1.175*np.cos(4*lat0)          # m per degree latitude
    mlon = 111412.84*np.cos(lat0) - 93.5*np.cos(3*lat0)                          # m per degree longitude
    return r, c, abs(tr[4])*mlat, abs(tr[0])*mlon, np.degrees(lat0)


def fft_vario(key, z):
    r, c, my, mx, lat0 = metric(key)
    r = r - r.min(); c = c - c.min(); ny, nx = r.max() + 1, c.max() + 1
    Z = np.zeros((ny, nx)); I = np.zeros((ny, nx)); Z[r, c] = z; I[r, c] = 1.0
    s = (2*ny, 2*nx); F = lambda a: np.fft.rfft2(a, s=s); iF = lambda a: np.fft.irfft2(a, s=s)
    fI, fZ, fZ2 = F(I), F(I*Z), F(I*Z*Z)
    N = np.rint(iF(fI*np.conj(fI)))
    G = iF(fZ2*np.conj(fI) + fI*np.conj(fZ2) - 2*fZ*np.conj(fZ))
    dy = np.fft.fftfreq(s[0], 1.0/s[0])[:, None]; dx = np.fft.fftfreq(s[1], 1.0/s[1])[None, :]
    H = np.hypot(dy*my, dx*mx)/1000.0                                            # km
    return dict(N=N, G=G, H=H, dy=dy, dx=dx, my=my, mx=mx, lat0=lat0, ny=ny, nx=nx)


EDGES = np.array([0.0, 0.15, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40])


def binned(V):
    m = (V['N'] > 0) & (V['H'] > 0)
    h = V['H'][m]; n = V['N'][m]; g = V['G'][m]
    k = np.digitize(h, EDGES) - 1; ok = (k >= 0) & (k < len(EDGES) - 1)
    nb = len(EDGES) - 1
    sN = np.bincount(k[ok], n[ok], nb); sG = np.bincount(k[ok], g[ok], nb); sH = np.bincount(k[ok], n[ok]*h[ok], nb)
    v = sN > 0
    return sH[v]/sN[v], sG[v]/(2*sN[v]), sN[v]/2                                 # mean lag (km), gamma, pairs (each pair counted once)


def axis(V, dim, kmax):
    """gamma along one grid axis (dim='ns' or 'ew'), lags 1..kmax cells"""
    out = []
    for k_ in range(1, kmax + 1):
        iy, ix = (k_, 0) if dim == 'ns' else (0, k_)
        out.append(((k_*(V['my'] if dim == 'ns' else V['mx']))/1000.0, V['G'][iy, ix]/(2*V['N'][iy, ix]), V['N'][iy, ix]))
    return np.array(out)


expo = lambda h, c0, c1, a: c0 + c1*(1 - np.exp(-3*h/a))                         # a = practical range (95 % of sill)
sph = lambda h, c0, c1, a: np.where(h < a, c0 + c1*(1.5*h/a - 0.5*(h/a)**3), c0 + c1)
dexp = lambda h, c0, c1, a1, c2, a2: c0 + c1*(1 - np.exp(-3*h/a1)) + c2*(1 - np.exp(-3*h/a2))


def fit(h, g, n, hmax):
    m = h <= hmax; out = {}
    w = g[m]/np.sqrt(n[m])                                                       # sigma ~ gamma/sqrt(N) (Cressie 1985 weights)
    for nm, f, p0, lo, hi in (('exponential', expo, [0.1, g.max(), 3], [0, 0, 0.05], [np.inf, np.inf, 200]),
                              ('spherical', sph, [0.1, g.max(), 5], [0, 0, 0.05], [np.inf, np.inf, 200]),
                              ('double exponential', dexp, [0.1, g.max()/2, 1, g.max()/2, 15], [0, 0, 0.05, 0, 0.05], [np.inf]*2 + [200, np.inf, 400])):
        try:
            p, _ = curve_fit(f, h[m], g[m], p0=p0, sigma=w, bounds=(lo, hi), maxfev=20000)
            rms = np.sqrt(np.mean((f(h[m], *p) - g[m])**2))
            out[nm] = (p, rms)
        except Exception as e:                                                   # report, do not hide
            out[nm] = (None, str(e))
    return out


def report(tag, key, agb, y):
    res, b = resid(agb, y); var = res.var()
    V = fft_vario(key, res)
    say('\n[%s] %d cells; OLS dB = %+.3f + %.5f AGB; residual variance %.4f dB^2 (sd %.3f dB)' % (tag, len(res), b[1], b[0], var, np.sqrt(var)))
    say('[%s] grid %d x %d cells, mean latitude %.3f N, cell %.1f m (N-S) x %.1f m (E-W)' % (tag, V['ny'], V['nx'], V['lat0'], V['my'], V['mx']))
    h, g, n = binned(V)
    say('[%s] omnidirectional variogram (lag bin mean km, gamma dB^2, gamma/variance, correlation 1-gamma/var, pairs)' % tag)
    for hh, gg, nn in zip(h, g, n):
        say('  %6.2f km  %.4f  %.3f  %+.3f  %d' % (hh, gg, gg/var, 1 - gg/var, nn))
    for thr in (0.5, 0.8, 0.9, 0.95):
        i = np.nonzero(g >= thr*var)[0]
        say('[%s] first lag bin with gamma >= %.0f%% of variance: %s' % (tag, 100*thr, '%.2f km' % h[i[0]] if len(i) else 'not reached within %.0f km' % h[-1]))
    for hmax in (20.0, 40.0):
        for nm, (p, rms) in fit(h, g, n, hmax).items():
            if p is None:
                say('[%s] fit %-18s (lags <= %.0f km): failed: %s' % (tag, nm, hmax, rms)); continue
            if nm == 'double exponential':
                say('[%s] fit %-18s (lags <= %.0f km): nugget %.4f, sill1 %.4f range1 %.2f km, sill2 %.4f range2 %.2f km; total sill %.4f (%.3f of var); rms %.4f'
                    % (tag, nm, hmax, p[0], p[1], p[2], p[3], p[4], p[0] + p[1] + p[3], (p[0] + p[1] + p[3])/var, rms))
            else:
                say('[%s] fit %-18s (lags <= %.0f km): nugget %.4f, partial sill %.4f, practical range %.2f km; sill %.4f (%.3f of var), nugget/sill %.3f; rms %.4f'
                    % (tag, nm, hmax, p[0], p[1], p[2], p[0] + p[1], (p[0] + p[1])/var, p[0]/(p[0] + p[1]), rms))
    for dim, nm in (('ns', 'N-S (azimuth-like)'), ('ew', 'E-W (range-like)')):
        A = axis(V, dim, 400)
        sel = sorted({int(np.argmin(abs(A[:, 0] - t))) for t in (0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25)})
        say('[%s] %s axis lags: ' % (tag, nm) + '  '.join('%.2fkm:%.3f' % (A[i, 0], A[i, 1]/var) for i in sel) + '   (gamma/variance)')
        for thr in (0.9, 0.95):
            i = np.nonzero(A[:, 1] >= thr*var)[0]
            say('[%s] %s: first lag with gamma >= %.0f%% of variance: %s' % (tag, nm, 100*thr, '%.2f km' % A[i[0], 0] if len(i) else 'not reached within %.1f km' % A[-1, 0]))
    return h, g, n, var, V


# ---------------------------------------------------------------- variograms
say('[A6] residual variogram, window-integration gamma0 n = 0.5, HV, CCI 2010 (input %s)' % os.path.join(OUT, 'vario_cells.npz'))
iA = names.index('window integration gamma0 n=0.5')
hA, gA, nA, varA, VA = report('ASF cell set', D['key_asf'], D['agb_asf'], D['Y_asf'][iA])
hM, gM, nM, varM, VM = report('main cell set', D['key_main'], D['agb_main'], D['y_main'])
with open(os.path.join(OUT, 'vario_a6_table.csv'), 'w') as f:
    f.write('set,lag_km,gamma_dB2,gamma_over_var,pairs\n')
    for tag, h, g, n, var in (('asf', hA, gA, nA, varA), ('main', hM, gM, nM, varM)):
        for row in zip(h, g, n):
            f.write('%s,%.4f,%.6f,%.6f,%d\n' % (tag, row[0], row[1], row[1]/var, row[2]))

# ---------------------------------------------------------------- block bootstrap: reproduction and block-size sensitivity
key = D['key_asf']; x = D['agb_asf']; Y = {nm: D['Y_asf'][i] for i, nm in enumerate(names)}
rr = lambda S: (lambda n, sx, sy, sxx, syy, sxy: (n*sxy - sx*sy)/np.sqrt((n*sxx - sx**2)*(n*syy - sy**2)))(*S.T)
ref = 'window integration gamma0 n=1'; cmp_ = ['window integration gamma0 n=0.5', 'ASF RTC gamma0 (GAMMA)', 'ASF RTC gamma0, cell average']
my, mx = VA['my'], VA['mx']


def boot(br_, bc_, rng, nrep=1000):
    bi = np.unique((key//ncol//br_)*100000 + (key % ncol)//bc_, return_inverse=True)[1]; nb = bi.max() + 1
    SS = {nm: np.stack([np.bincount(bi, w, minlength=nb) for w in (np.ones_like(x), x, Y[nm], x*x, Y[nm]**2, x*Y[nm])], 1) for nm in [ref] + cmp_}
    W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(nrep)]).astype(float)
    Rb = {nm: rr(W @ SS[nm]) for nm in SS}
    return nb, Rb


say('\n[BOOTCHK] reproduction of final_eq14.py RTC_BOOT (rng seed 0, 50 then 100 cells, 1000 resamples)')
rng = np.random.default_rng(0)
for Bk in (50, 100):
    nb, Rb = boot(Bk, Bk, rng)
    lo, hi = np.percentile(Rb['window integration gamma0 n=0.5'], [2.5, 97.5]); d = Rb[ref] - Rb['ASF RTC gamma0, cell average']
    say('  %3d cells: %d blocks; r(n=0.5) [%.3f, %.3f]; r(n=1) - r(ASF cell average) %+.3f [%+.3f, %+.3f]' % ((Bk, nb, lo, hi, d.mean()) + tuple(np.percentile(d, [2.5, 97.5]))))

say('\n[BOOTSENS] block-size sensitivity, 1000 resamples, rng seed = 1000 + rows (block rows x cols in CCI cells; km N-S x E-W)')
say('  %-11s %-15s %7s  %-24s %-30s %-30s' % ('cells', 'km', 'blocks', 'r(n=0.5) 95% CI (width)', 'r(n=1)-r(ASF cell avg) CI (width)', 'r(n=1)-r(n=0.5) CI (width)'))
cfg = [(b, b) for b in (5, 10, 20, 30, 50, 75, 100, 150, 200)] + [(b, int(round(b*my/mx))) for b in (20, 30, 50, 75, 100, 150)]
for b_r, b_c in cfg:
    nb, Rb = boot(b_r, b_c, np.random.default_rng(1000 + b_r + (0 if b_r == b_c else 100000)))
    c1 = np.percentile(Rb['window integration gamma0 n=0.5'], [2.5, 97.5])
    d2 = Rb[ref] - Rb['ASF RTC gamma0, cell average']; c2 = np.percentile(d2, [2.5, 97.5])
    d3 = Rb[ref] - Rb['window integration gamma0 n=0.5']; c3 = np.percentile(d3, [2.5, 97.5])
    say('  %3d x %-5d %5.1f x %-7.1f %7d  [%.3f, %.3f] (%.3f)     %+.3f [%+.3f, %+.3f] (%.3f)    %+.3f [%+.3f, %+.3f] (%.3f)%s'
        % (b_r, b_c, b_r*my/1000, b_c*mx/1000, nb, c1[0], c1[1], c1[1] - c1[0], d2.mean(), c2[0], c2[1], c2[1] - c2[0], d3.mean(), c3[0], c3[1], c3[1] - c3[0],
           '   square' if b_r != b_c else ''))
say('\n[A6] done')
