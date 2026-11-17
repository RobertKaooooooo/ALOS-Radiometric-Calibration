#!/usr/bin/env python3
"""A7: saturation of HV gamma0 (n = 0.5) against CCI AGB 2010, 100 m cells, for the three methods of the [SAME] n = 0.5 table
(uncorrected / cos^0.5(theta_inc), per-pixel RTC, window integration).

Input : out_ellip/a7_cells.npz written by
        RTC_GEOID_SHIFT=-27.32 RTC_SAME=1 RTC_A7=1 python3 final_eq14.py
Output: out_ellip/agb_sat_a7_log.txt (all numbers), out_ellip/agb_sat_a7_bins.csv (25 Mg/ha bin table)

Models fitted to all cells (dB vs AGB, ordinary least squares):
  linear        y = a + b x
  hinge         y = a + b1 x + d (x - x0)+          pre-break slope b1, post-break slope b1 + d, breakpoint x0 by grid search (1 Mg/ha)
  plateau       y = a + b1 min(x, x0)               linear-plateau (post-break slope fixed at 0), x0 by grid search (1 Mg/ha)
  logarithmic   y = a + b ln x                      no intrinsic saturation point; local slope b/x reported
  exponential   y = a - c exp(-x/k)                 x95 = 3k (AGB at 95 % of the modelled dynamic range)
Uncertainty: spatial block bootstrap (10 km and 5 km blocks of CCI cells, 1000 resamples, seed 0), breakpoints and slopes
re-estimated by the same grid search in every resample (exact OLS from block sums); paired differences between methods.
"""
import os
import numpy as np
from scipy.optimize import curve_fit

OUT = os.path.join(os.environ.get('RTC_BASE', '/mnt/c/grd file'), 'scripts', os.environ.get('RTC_OUTDIR', 'out_ellip'))
FN = os.path.join(OUT, 'a7_cells.npz')
D = np.load(FN)
LOG = open(os.path.join(OUT, 'agb_sat_a7_log.txt'), 'w')


def say(s=''):
    print(s); LOG.write(s + '\n'); LOG.flush()


names = [str(s) for s in D['names']]
SHORT = dict(zip(names, ['uncorrected', 'per-pixel RTC', 'window integration']))
x = D['agb'].astype(float); Y = {nm: D['Y'][i].astype(float) for i, nm in enumerate(names)}
key = D['key']; ncol = int(D['ncol']); N = len(x)
say('[A7] HV gamma0 n = 0.5 vs CCI AGB 2010, %d cells (input %s)' % (N, FN))
say('[A7] AGB range %.1f - %.1f Mg/ha; percentiles 50/90/95/99/99.5: %s' % ((x.min(), x.max(), ' / '.join('%.1f' % v for v in np.percentile(x, [50, 90, 95, 99, 99.5])))))

# ---------------------------------------------------------------- consistency with [SAME] n = 0.5
say('\n[CHK] full-range OLS (compare with [SAME] gamma0 n = 0.5 in final_eq14_SAME_log.txt)')
for nm in names:
    b = np.polyfit(x, Y[nm], 1); r = Y[nm] - np.polyval(b, x)
    say('  %-20s r = %.3f  slope = %.5f dB/(Mg/ha)  intercept = %+.3f  RMSE = %.3f' % (SHORT[nm], np.corrcoef(x, Y[nm])[0, 1], b[0], b[1], np.sqrt(np.mean(r**2))))

# ---------------------------------------------------------------- 25 Mg/ha bins
W25 = 25.0
edges = np.arange(0, np.ceil(x.max()/W25)*W25 + W25, W25)
ib = np.digitize(x, edges) - 1
say('\n[BIN] 25 Mg/ha bins: mean, standard deviation (ddof = 1) and standard error of cell dB; n = cells per bin (same cells for all methods)')
hdr = '  %-11s %8s %9s' % ('AGB bin', 'n', 'mean AGB') + ''.join('   %-30s' % ('%s mean/sd/se' % SHORT[nm]) for nm in names)
say(hdr)
csv = open(os.path.join(OUT, 'agb_sat_a7_bins.csv'), 'w')
csv.write('agb_lo,agb_hi,n,agb_mean,' + ','.join('%s_mean_dB,%s_sd_dB,%s_se_dB' % ((SHORT[nm].replace(' ', '_'),)*3) for nm in names) + '\n')
BM = {nm: [] for nm in names}; BC = []; BN = []
for j in range(len(edges) - 1):
    m = ib == j; n = int(m.sum())
    if n == 0:
        continue
    row = '  %4.0f-%-6.0f %8d %9.1f' % (edges[j], edges[j+1], n, x[m].mean()); cr = '%.0f,%.0f,%d,%.2f' % (edges[j], edges[j+1], n, x[m].mean())
    for nm in names:
        v = Y[nm][m]; sd = v.std(ddof=1) if n > 1 else np.nan; se = sd/np.sqrt(n)
        row += '   %+7.3f / %5.3f / %6.4f        ' % (v.mean(), sd, se); cr += ',%.4f,%.4f,%.5f' % (v.mean(), sd, se)
        BM[nm].append(v.mean())
    BC.append(x[m].mean()); BN.append(n)
    say(row + ('   (n < 100)' if n < 100 else '')); csv.write(cr + '\n')
csv.close(); BC = np.array(BC); BN = np.array(BN)
say('[BIN] increment of bin mean between successive bins (dB per 25 Mg/ha):')
for nm in names:
    d = np.diff(BM[nm])
    say('  %-20s ' % SHORT[nm] + ' '.join('%+.3f' % v for v in d))

# ---------------------------------------------------------------- segment statistics (A7 item in the handbook: < 100 and >= 100 Mg/ha)
say('\n[SEG] r and OLS slope inside AGB segments')
for lo, hi in ((0, 100), (100, 1e9), (0, 150), (150, 1e9), (0, 200), (200, 1e9)):
    m = (x >= lo) & (x < hi)
    say('  AGB %3.0f-%-5s n = %6d: ' % (lo, '%.0f' % hi if hi < 1e9 else 'max', m.sum()) +
        '  '.join('%s r = %.3f slope = %.5f' % (SHORT[nm], np.corrcoef(x[m], Y[nm][m])[0, 1], np.polyfit(x[m], Y[nm][m], 1)[0]) for nm in names))

# ---------------------------------------------------------------- breakpoint models from (block) sums
X0 = np.arange(10.0, np.floor(np.percentile(x, 99.5)) + 1.0, 1.0)
say('\n[GRID] breakpoint grid %.0f - %.0f Mg/ha, step 1 (upper end = 99.5th percentile of AGB); %d candidates' % (X0[0], X0[-1], len(X0)))


def block_sums(bi, nb):
    """sums per block: base (nb, 3 + 3*nm) and x0-dependent (nb, nx0, 3 + nm)"""
    bc = lambda w: np.bincount(bi, w, minlength=nb)
    base = {'n': bc(np.ones(N)), 'x': bc(x), 'xx': bc(x*x)}
    for nm in names:
        base['y' + nm] = bc(Y[nm]); base['xy' + nm] = bc(x*Y[nm]); base['yy' + nm] = bc(Y[nm]**2)
    dep = {k: np.zeros((nb, len(X0))) for k in ['h', 'hh', 'xh'] + ['yh' + nm for nm in names]}
    for i, x0 in enumerate(X0):
        h = np.maximum(x - x0, 0.0)
        dep['h'][:, i] = bc(h); dep['hh'][:, i] = bc(h*h); dep['xh'][:, i] = bc(x*h)
        for nm in names:
            dep['yh' + nm][:, i] = bc(Y[nm]*h)
    return base, dep


def solve(S, nm):
    """S: dict of summed statistics, base arrays (...), dependent arrays (..., nx0). Returns hinge and plateau fits over the grid."""
    n, sx, sxx = S['n'][..., None], S['x'][..., None], S['xx'][..., None]
    sy, sxy, syy = S['y' + nm][..., None], S['xy' + nm][..., None], S['yy' + nm][..., None]
    sh, shh, sxh, syh = S['h'], S['hh'], S['xh'], S['yh' + nm]
    sh, shh, sxh, syh = np.broadcast_arrays(sh, shh, sxh, syh)
    n, sx, sxx, sy, sxy, syy = [np.broadcast_to(a, sh.shape) for a in (n, sx, sxx, sy, sxy, syy)]
    A = np.stack([np.stack([n, sx, sh], -1), np.stack([sx, sxx, sxh], -1), np.stack([sh, sxh, shh], -1)], -2)
    c = np.stack([sy, sxy, syh], -1)
    bH = np.linalg.solve(A, c[..., None])[..., 0]; sseH = syy - (bH*c).sum(-1)
    sm, smm, smy = sx - sh, sxx - 2*sxh + shh, sxy - syh                                  # m = min(x, x0) = x - (x - x0)+
    det = n*smm - sm**2
    bP1 = (n*smy - sm*sy)/det; bP0 = (sy - bP1*sm)/n; sseP = syy - bP0*sy - bP1*smy
    return bH, sseH, np.stack([bP0, bP1], -1), sseP


def best(bH, sseH, bP, sseP):
    iH = np.argmin(sseH, -1); iP = np.argmin(sseP, -1)
    take = lambda a, i: np.take_along_axis(a, i[..., None], -1)[..., 0]
    tH = lambda k: np.take_along_axis(bH[..., k], iH[..., None], -1)[..., 0]
    return dict(x0H=X0[iH], b1H=tH(1), postH=tH(1) + tH(2), aH=tH(0), sseH=take(sseH, iH), iH=iH,
                x0P=X0[iP], b1P=np.take_along_axis(bP[..., 1], iP[..., None], -1)[..., 0], aP=np.take_along_axis(bP[..., 0], iP[..., None], -1)[..., 0],
                sseP=take(sseP, iP), iP=iP)


bi0 = np.zeros(N, int)
base, dep = block_sums(bi0, 1)
Sall = {**{k: v[0] for k, v in base.items()}, **{k: v[0] for k, v in dep.items()}}
bic = lambda sse, k: N*np.log(sse/N) + k*np.log(N)
FIT = {}
say('\n[FIT] all cells; slopes in dB per Mg/ha (x 100 = dB per 100 Mg/ha); BIC relative to the linear model (negative = better)')
for nm in names:
    y = Y[nm]
    bL = np.polyfit(x, y, 1); sseL = np.sum((y - np.polyval(bL, x))**2)
    bH, sseH, bP, sseP = solve(Sall, nm); B = best(bH, sseH, bP, sseP); FIT[nm] = B
    bG = np.polyfit(np.log(x), y, 1); sseG = np.sum((y - np.polyval(bG, np.log(x)))**2)
    try:
        pE, _ = curve_fit(lambda t, a, c, k: a - c*np.exp(-t/k), x, y, p0=[y.mean() + 1, 3.0, 80.0], maxfev=20000)
        sseE = np.sum((y - (pE[0] - pE[1]*np.exp(-x/pE[2])))**2)
    except Exception as e:                                                                  # report, do not hide
        pE, sseE = None, str(e)
    rm = lambda sse: np.sqrt(sse/N)
    say('\n  [%s]' % SHORT[nm])
    say('    linear      : y = %+.3f + %.5f x; RMSE %.4f' % (bL[1], bL[0], rm(sseL)))
    say('    hinge       : x0 = %.0f Mg/ha; pre slope %.5f, post slope %+.5f (post/pre %.3f); intercept %+.3f; level at x0 %+.3f dB; RMSE %.4f; dBIC %+.1f'
        % (B['x0H'], B['b1H'], B['postH'], B['postH']/B['b1H'], B['aH'], B['aH'] + B['b1H']*B['x0H'], rm(B['sseH']), bic(B['sseH'], 4) - bic(sseL, 2)))
    say('    plateau     : x0 = %.0f Mg/ha; pre slope %.5f; intercept %+.3f; plateau level %+.3f dB; RMSE %.4f; dBIC %+.1f'
        % (B['x0P'], B['b1P'], B['aP'], B['aP'] + B['b1P']*B['x0P'], rm(B['sseP']), bic(B['sseP'], 3) - bic(sseL, 2)))
    say('    logarithmic : y = %+.3f + %.4f ln x; RMSE %.4f; dBIC %+.1f; local slope b/x at 25/50/100/150/200/300 Mg/ha: %s'
        % (bG[1], bG[0], rm(sseG), bic(sseG, 2) - bic(sseL, 2), ' '.join('%.5f' % (bG[0]/v) for v in (25, 50, 100, 150, 200, 300))))
    if pE is None:
        say('    exponential : fit failed: %s' % sseE)
    else:
        say('    exponential : y = %+.3f - %.3f exp(-x/%.1f); x95 = 3k = %.0f Mg/ha; initial slope c/k %.5f; RMSE %.4f; dBIC %+.1f'
            % (pE[0], pE[1], pE[2], 3*pE[2], pE[1]/pE[2], rm(sseE), bic(sseE, 3) - bic(sseL, 2)))
    prof = sseH - B['sseH']
    inside = X0[prof <= 3.84*B['sseH']/(N - 4)]                                             # profile F/chi2 1-df interval (independent-cell assumption)
    say('    hinge x0 profile interval assuming independent cells (dSSE <= 3.84 s^2): %.0f - %.0f Mg/ha (too narrow if cells are spatially correlated)' % (inside.min(), inside.max()))

say('\n[FIT] bin-mean check: hinge fitted to 25 Mg/ha bin means (weighted by n), same grid')
for nm in names:
    yb = np.array(BM[nm]); w = BN.astype(float); best_ = None
    for x0 in X0:
        Xd = np.stack([np.ones_like(BC), BC, np.maximum(BC - x0, 0)], 1)
        bb, *_ = np.linalg.lstsq(Xd*np.sqrt(w)[:, None], yb*np.sqrt(w), rcond=None); s = np.sum(w*(yb - Xd @ bb)**2)
        if best_ is None or s < best_[0]:
            best_ = (s, x0, bb)
    say('  %-20s x0 = %.0f Mg/ha, pre slope %.5f, post slope %+.5f' % (SHORT[nm], best_[1], best_[2][1], best_[2][1] + best_[2][2]))

# ---------------------------------------------------------------- spatial block bootstrap
pairs = [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]
for Bk in (100, 50):
    bi = np.unique((key//ncol//Bk)*100000 + (key % ncol)//Bk, return_inverse=True)[1]; nb = bi.max() + 1
    base, dep = block_sums(bi, nb)
    rng = np.random.default_rng(0)
    Wt = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(1000)]).astype(float)
    Sb = {**{k: Wt @ v for k, v in base.items()}, **{k: Wt @ v for k, v in dep.items()}}
    Bb = {nm: best(*solve(Sb, nm)) for nm in names}
    say('\n[BOOT] blocks of %d cells (%.0f km): %d blocks, 1000 resamples; median [2.5, 97.5 percentile]; share of resamples with x0 at grid edge' % (Bk, Bk/10, nb))
    ci = lambda v: '%.4g [%.4g, %.4g]' % (np.median(v), *np.percentile(v, [2.5, 97.5]))
    for nm in names:
        b = Bb[nm]; edge = lambda v: np.mean((v <= X0[0]) | (v >= X0[-1]))
        say('  %-20s hinge x0 %s (edge %.3f); pre slope %s; post slope %s' % (SHORT[nm], ci(b['x0H']), edge(b['x0H']), ci(b['b1H']), ci(b['postH'])))
        say('  %-20s plateau x0 %s (edge %.3f); pre slope %s' % ('', ci(b['x0P']), edge(b['x0P']), ci(b['b1P'])))
    say('  paired differences (same resample):')
    for a, c in pairs:
        A_, C_ = Bb[a], Bb[c]
        say('  %s - %s: hinge dx0 %s; d pre slope %s; d post slope %s; plateau dx0 %s; d pre slope %s'
            % (SHORT[a], SHORT[c], ci(A_['x0H'] - C_['x0H']), ci(A_['b1H'] - C_['b1H']), ci(A_['postH'] - C_['postH']),
               ci(A_['x0P'] - C_['x0P']), ci(A_['b1P'] - C_['b1P'])))
        say('      ratio of pre slopes (hinge) %s; P(dx0 hinge > 0) %.3f, P(dx0 hinge < 0) %.3f'
            % (ci(A_['b1H']/C_['b1H']), np.mean(A_['x0H'] > C_['x0H']), np.mean(A_['x0H'] < C_['x0H'])))
say('\n[A7] done')
