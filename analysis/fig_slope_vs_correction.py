"""
fig_slope_vs_correction.py

Purpose (advisor request, 10/26 meeting):
  Within the pixels whose local incidence angle theta_loc is 10-20 deg,
  plot terrain slope (x) against the RTC correction (y), so that one can
  see whether flat ground with a small incidence angle is being corrected
  the same way as a slope facing the radar.
  Also reports how many DEM facets fall in each radar pixel.

Scene : ALPSRP237630880 (11 Jul 2010, HV), White Mountain NF
Inputs: the same files and calls as the abstract-figure script recovered
        from ~/.bash_history (lines ~1490-1790, saved as
        history_recovered.txt), i.e.
          facet_results_FINAL2.npz            rows, cols, az, rg per DEM facet
          GLO30_ALOS_merged.bin               7200 x 7200 float32, 1 arcsec, lat_top 45.0, lon_left -72.0
          LED-ALPSRP237630880-H1.1__A         spacing, orbit, attitude, theta_inc polynomial
          IMG-HH-ALPSRP237630880-H1.1__A      line times, near range
          alos_rtc/area_correction.py         compute_facet_area_and_theta_l(), compute_gamma0_ratio()

Correction definition (same as area_correction.compute_gamma0_ratio):
  ratio = (1/sin theta_inc) * (A_ref / A) / cos theta_loc
  correction_dB = 10*log10(ratio) = gamma0(dB) - sigma0(dB)
  area term (dB) = 10*log10((1/sin theta_inc) * A_ref / A)
  cos  term (dB) = -10*log10(cos theta_loc)

Range slope sign: tau_rho > 0 means the slope faces the radar.
  The code's slope_r (area_correction.py line 80) is positive for slopes
  facing AWAY (checked: theta_loc increases with slope_r), so
  tau_rho = -atan(slope_r). The script checks this sign on the data.

Outputs (in OUT):
  slope_vs_correction.png
  slope_vs_correction_log.txt     every number printed below
  slope_vs_correction_bins.npz    the binned tables

Run:  cd "/mnt/c/grd file" && python3 scripts/fig_slope_vs_correction.py
"""

import os, sys, gc
import numpy as np

BASE = '/mnt/c/grd file'
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geometry, geolocate, area_correction  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ---------------- inputs ----------------
D    = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID  = 'ALPSRP237630880'
LED  = D + 'LED-' + SID + '-H1.1__A'
IMG  = D + 'IMG-HH-' + SID + '-H1.1__A'
NA, NR = 18432, 4704
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
PPD, LAT_TOP = 3600, 45.0
FLAT_SLOPE_DEG = 3.0          # "flat" = total DEM slope below this
OUT = os.path.join(BASE, 'scripts', 'out')
os.makedirs(OUT, exist_ok=True)

LOG = open(os.path.join(OUT, 'slope_vs_correction_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()

# ---------------- geometry (identical to the recovered script) ----------------
txt = open(LED, 'rb').read(20000).decode('ascii', 'ignore')
k = txt.find('EXTRACTED CHIRP'); sp = txt[k-40:k].split()
DAZ, DR = float(sp[-2]), float(sp[-1]); AREF = DAZ * DR
t0, h, P, V = leader_file.read_platform_position_record(LED)
tf = img_file.read_line_time_ms(IMG, 0, NR); te = img_file.read_line_time_ms(IMG, NA-1, NR)
R0 = float(img_file.read_near_range_m(IMG, 0, NR))
Pm, Vm = geolocate.interpolate_position_velocity_vec(np.array([0.5*(tf+te)]), t0, h, P, V); Pm = Pm[0]; Vm = Vm[0]
la = np.arctan2(Pm[2], np.hypot(Pm[0], Pm[1])*(1-geolocate.WGS84_E2)); lo = np.arctan2(Pm[1], Pm[0])
HD = np.arctan2(Vm @ np.array([-np.sin(lo), np.cos(lo), 0]),
                Vm @ np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)]))
say('[input] %s  DAZ %.7f m  DR %.7f m  A_ref %.3f m2  heading %.3f deg' % (SID, DAZ, DR, AREF, np.degrees(HD)))

g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg = g['rg']
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(7200, 7200)
coef = leader_file.read_six_coefficients(LED)
th = geometry.compute_theta_ih(coef, (R0 + rg*DR)/1000.0)            # theta_inc per facet (rad)
tt = tf + az/(NA-1)*(te-tf)
at, pd, rd, yd = leader_file.read_attitude_record(LED); o = np.argsort(at)
pitch = np.radians(np.interp(tt, at[o], pd[o])); yaw = np.radians(np.interp(tt, at[o], yd[o]))
del tt; gc.collect()

# code's own theta_loc and facet area (area_correction.py lines 32-108)
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP,
                                                         th, pitch, yaw, area_ref_m2=AREF)
del pitch, yaw; gc.collect()

# slope, range slope, azimuth slope: copied from area_correction.py lines 65-81
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
del p, q, p_safe, aspect; gc.collect()

S_deg   = np.degrees(slope)                    # total slope
tau_rho = -np.degrees(np.arctan(slope_r))      # range slope, + = facing radar
tau_s   = np.degrees(np.arctan(slope_a))       # azimuth slope
c_sign = np.corrcoef(tau_rho, tl)[0, 1]
say('[check] corr(range slope tau_rho, theta_loc) = %+.3f  (must be negative: facing radar -> smaller theta_loc)' % c_sign)

# per-facet correction (one facet = what a radar pixel gets when it receives one facet)
area_term = 10*np.log10((1.0/np.sin(th))*AREF/area)
cos_term  = -10*np.log10(np.cos(np.radians(tl)))
corr_f    = area_term + cos_term

# correction actually applied by the code on the radar grid, sampled at each facet
ratio, valid = area_correction.compute_gamma0_ratio(
    area, tl, az, rg, NA, NR, geometry.compute_theta_ih(coef, (R0+np.arange(NR)*DR)/1000.0), AREF)
ai = np.clip(np.round(az).astype(int), 0, NA-1); ri = np.clip(np.round(rg).astype(int), 0, NR-1)
v_px = valid[ai, ri]; corr_px = 10*np.log10(ratio[ai, ri].astype(float))
say('[grid] radar pixels valid in compute_gamma0_ratio: %.1f%% of the full image' % (100*valid.mean()))
del ratio, valid; gc.collect()

# ---------------- facets per radar pixel ----------------
ok = (az >= 0) & (az < NA-1) & (rg >= 0) & (rg < NR-1)
a0 = np.floor(az[ok]).astype(np.int64); r0 = np.floor(rg[ok]).astype(np.int64)
cnt = sum(np.bincount((a0+i)*NR + (r0+j), minlength=NA*NR) for i in (0, 1) for j in (0, 1)).reshape(NA, NR)
box = cnt[a0.min():a0.max(), r0.min():r0.max()]
say('[facets] facets %d | radar px in footprint %d | px with >=1 facet %.1f%% | mean facets per filled px %.2f | max %d | radar px per facet %.1f'
    % (ok.sum(), box.size, 100*(box > 0).mean(), box[box > 0].mean(), box.max(), box.size/ok.sum()))
del cnt, box, a0, r0; gc.collect()

# ---------------- tables ----------------
flat = S_deg < FLAT_SLOPE_DEG
say('\n[table 1] by theta_loc bin (all facets)')
say('theta_loc   n_facets  flat(S<%g)  S p5/p50/p95 (deg)   tau_rho p50   corr_facet mean   area term   cos term   code-valid' % FLAT_SLOPE_DEG)
for lo_, hi_ in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 35), (35, 40), (40, 45), (45, 60), (60, 70), (70, 90)]:
    m = (tl >= lo_) & (tl < hi_)
    if m.sum() == 0:
        say('%2d-%2d deg  none' % (lo_, hi_)); continue
    s5, s50, s95 = np.percentile(S_deg[m], [5, 50, 95])
    say('%2d-%2d deg  %9d  %8.2f%%   %5.1f/%5.1f/%5.1f     %+6.1f        %+6.2f dB       %+6.2f    %+6.2f    %5.1f%%'
        % (lo_, hi_, m.sum(), 100*flat[m].mean(), s5, s50, s95, np.median(tau_rho[m]), corr_f[m].mean(),
           area_term[m].mean(), cos_term[m].mean(), 100*v_px[m].mean()))

fm = flat
say('\n[table 2] flat facets (S < %g deg): n %d | theta_loc p1/p5/p50/p95/p99 %s deg | corr_facet mean %+.3f dB (std %.3f) | 10log10(1/cos theta_inc) mean %+.3f dB'
    % (FLAT_SLOPE_DEG, fm.sum(), np.round(np.percentile(tl[fm], [1, 5, 50, 95, 99]), 1), corr_f[fm].mean(), corr_f[fm].std(),
       (-10*np.log10(np.cos(th[fm]))).mean()))

sel = (tl >= 10) & (tl < 20)
edges = np.arange(-10, 71, 5)
say('\n[table 3] theta_loc 10-20 deg only: correction by range slope tau_rho (deg, + facing radar)')
say('tau_rho bin   n        corr_facet mean   std    code-valid')
rows_out = []
for i in range(len(edges)-1):
    m = sel & (tau_rho >= edges[i]) & (tau_rho < edges[i+1])
    if m.sum() < 20: continue
    say('%+4d..%+4d  %8d   %+6.2f dB        %.2f   %5.1f%%' % (edges[i], edges[i+1], m.sum(), corr_f[m].mean(), corr_f[m].std(), 100*v_px[m].mean()))
    rows_out.append((edges[i], edges[i+1], m.sum(), corr_f[m].mean(), corr_f[m].std(), v_px[m].mean()))
np.savez(os.path.join(OUT, 'slope_vs_correction_bins.npz'), tau_rho_bins=np.array(rows_out))

# ---------------- figure ----------------
rng = np.random.default_rng(0)
def sub(m, n=20000):
    idx = np.flatnonzero(m); return idx if idx.size <= n else rng.choice(idx, n, replace=False)
ref = (tl >= 35) & (tl < 40)
flat_val = (-10*np.log10(np.cos(th[fm]))).mean()

YLO, YHI = -16.0, 3.0
inc_mean = np.degrees(th).mean()
sel_v = sel & v_px; sel_x = sel & ~v_px
plt.rcParams.update({'font.size': 11})
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, x, xl in [(axs[0], tau_rho, 'Range-direction slope $\\tau_\\rho$ (deg)  [+ = facing the radar]'),
                  (axs[1], S_deg,   'Total terrain slope S (deg)')]:
    i_ref = sub(ref); i_v = sub(sel_v); i_x = sub(sel_x)
    ax.scatter(x[i_ref], corr_f[i_ref], s=2, alpha=0.25, color='0.6', lw=0,
               label='$\\theta_{loc}$ 35-40$\\degree$ (n = %d)' % ref.sum())
    ax.scatter(x[i_v], corr_f[i_v], s=3, alpha=0.5, color='#1f77b4', lw=0,
               label='$\\theta_{loc}$ 10-20$\\degree$, valid in code (n = %d)' % sel_v.sum())
    ax.scatter(x[i_x], corr_f[i_x], s=3, alpha=0.5, color='#ff7f0e', lw=0,
               label='$\\theta_{loc}$ 10-20$\\degree$, void in code (n = %d)' % sel_x.sum())
    # binned median of the 10-20 deg facets
    e = np.arange(np.floor(x[sel].min()), np.ceil(x[sel].max()) + 1, 1.0)
    bi = np.digitize(x[sel], e) - 1; ys = corr_f[sel]
    med = np.array([np.median(ys[bi == i]) if (bi == i).sum() >= 30 else np.nan for i in range(len(e) - 1)])
    ax.plot(0.5*(e[:-1] + e[1:]), med, color='k', lw=1.4, label='median of 10-20$\\degree$ (1$\\degree$ bins)')
    ax.axhline(flat_val, color='#d62728', lw=1.2, ls='--', label='flat ground, 1/cos $\\theta_{inc}$ = %+.2f dB' % flat_val)
    ax.axhline(0, color='k', lw=0.6)
    n_out = int((sel & (corr_f < YLO)).sum())
    ax.text(0.99, 0.02, '%d facets below %g dB not shown' % (n_out, YLO), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='0.3')
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(xl); ax.set_ylabel('Correction $\\gamma^0 - \\sigma^0$ (dB)')
axs[0].axvline(inc_mean, color='#9467bd', lw=1.2, ls=':')
axs[0].text(inc_mean - 0.5, YHI - 0.6, 'layover onset\n$\\tau_\\rho$ = $\\theta_{inc}$ = %.1f$\\degree$' % inc_mean,
            fontsize=9, color='#9467bd', va='top', ha='right')
axs[0].legend(fontsize=8.5, frameon=False, markerscale=4, loc='lower left')
axs[0].set_title('(a) Correction vs range slope', loc='left')
axs[1].set_title('(b) Correction vs total slope', loc='left')
fig.suptitle('%s, one DEM facet (GLO-30, 1") per point; flat = S < %g$\\degree$: %.2f%% of the 10-20$\\degree$ facets'
             % (SID, FLAT_SLOPE_DEG, 100*flat[sel].mean()), fontsize=11)
plt.tight_layout()
png = os.path.join(OUT, 'slope_vs_correction.png')
plt.savefig(png, dpi=200, facecolor='white')
say('\n[saved] %s' % png)
LOG.close()
