"""
thesis_figs.py  -- thesis figures in one consistent style (serif font, no in-figure titles, panel labels only)

Run   : cd "/mnt/c/grd file" && RTC_GEOID_SHIFT=-27.32 python3 scripts/thesis_figs.py
        add RTC_GEOCHKFIG=1 to also compute the geolocation correlation surface (needs the ASF product, ~2 min more)
Output: scripts/out_ellip/thesis_*.png and thesis_figs_log.txt

Figures
  thesis_study_area.png      hillshade of GLO-30 with the scene footprint, scale bar and north arrow       (Fig. 3.2)
  thesis_cci.png             ESA CCI AGB 2010 inside the footprint                                        (Fig. 3.3)
  thesis_theta_loc_map.png   local incidence angle and range slope on the DEM grid, layover facets marked (new)
  thesis_slope_correction.png mean area correction versus range and azimuth slope, theta_loc contours      (Fig. 4.2)
  thesis_window_facets.png   facets per 31 x 7 window and per single-look pixel                           (new, supports 24.5 %)
  thesis_line_time.png       integer-ms line stamps minus the linear fit                                  (new, supports 0.93 line)
  thesis_correction_map.png  A_flat / A_W in dB on the DEM grid                                           (new)
  thesis_hh_n_sweep.png      HH: r and residual slope versus n, from final_eq14_log.txt [T5]              (new)
  thesis_geochk_surface.png  r versus azimuth / range offset (RTC_GEOCHKFIG=1)                            (new)
Geometry is identical to final_eq14.py (fitted line times, azimuth calibration, EGM2008 shift).
"""
import os, sys, gc, glob, re
import numpy as np
from scipy.ndimage import uniform_filter

BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
sys.path.insert(0, os.path.join(BASE, 'ALOS-Radiometric-Calibration'))
from alos_rtc import leader_file, img_file, geometry, geolocate, area_correction  # noqa: E402
import rasterio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
for _f in glob.glob('/mnt/c/Windows/Fonts/times*.ttf'): _fm.fontManager.addfont(_f)
plt.rcParams.update({'font.family': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 10,
                     'axes.titlesize': 10.5, 'axes.labelsize': 10, 'legend.fontsize': 8.5})

D = os.path.join(BASE, 'ALPSRP237630880-L1.1') + '/'
SID = 'ALPSRP237630880'
LED = D + 'LED-' + SID + '-H1.1__A'; IMG = D + 'IMG-HH-' + SID + '-H1.1__A'
NA, NR = int(os.environ.get('RTC_NA', 18432)), int(os.environ.get('RTC_NR', 4704))
BSFILE = os.path.join(BASE, 'ALOS_HV_uncorrected_new.bin')
GEOFILE = os.path.join(BASE, 'facet_results_FINAL2.npz')
DEMFILE = os.path.join(BASE, 'GLO30_ALOS_merged.bin')
CCIFILE = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
PPD, LAT_TOP, LON_LEFT = 3600, 45.0, -72.0
WA, WR = 31, 7
GEOID_N = os.environ.get('RTC_GEOID_SHIFT')
OUT = os.path.join(BASE, 'scripts', 'out_ellip' if GEOID_N else 'out'); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'thesis_figs_log.txt'), 'w')
def say(s):
    print(s); LOG.write(s + '\n'); LOG.flush()
def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=220, facecolor='white', bbox_inches='tight'); plt.close(fig); say('[saved] %s' % p)
np.seterr(invalid='ignore', divide='ignore')
ASPECT = 1/np.cos(np.radians(44.5))

# ================= geometry (identical to final_eq14.py) =================
txt = open(LED, 'rb').read(20000).decode('ascii', 'ignore')
k = txt.find('EXTRACTED CHIRP'); sp = txt[k-40:k].split()
DAZ, DR = float(sp[-2]), float(sp[-1]); AREF = DAZ*DR
t0, h, P, V = leader_file.read_platform_position_record(LED)
tf = img_file.read_line_time_ms(IMG, 0, NR); te = img_file.read_line_time_ms(IMG, NA-1, NR)
R0 = float(img_file.read_near_range_m(IMG, 0, NR))
Pm, Vm = geolocate.interpolate_position_velocity_vec(np.array([0.5*(tf+te)]), t0, h, P, V); Pm = Pm[0]; Vm = Vm[0]
la = np.arctan2(Pm[2], np.hypot(Pm[0], Pm[1])*(1-geolocate.WGS84_E2)); lo = np.arctan2(Pm[1], Pm[0])
HD = np.arctan2(Vm @ np.array([-np.sin(lo), np.cos(lo), 0]), Vm @ np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)]))
g = np.load(GEOFILE); rows = g['rows']; cols = g['cols']; az = g['az']; rg0 = g['rg']
tf_fit, te_fit, _dt = img_file.read_line_times_fit(IMG, NR, NA)
az = ((tf + az*(te - tf)/(NA - 1)) - tf_fit)/(te_fit - tf_fit)*(NA - 1)
say('[time] line-time origin %.7f s (stamp %.3f s): azimuth shift %+.3f lines' % (tf_fit, tf, -(tf_fit - tf)/_dt)); tf, te = tf_fit, te_fit
AZ_SHIFT = float(os.environ.get('RTC_AZ_SHIFT', 4.17))
if AZ_SHIFT: az = az + AZ_SHIFT; say('[az] constant azimuth shift %+.3f lines' % AZ_SHIFT)
coef = leader_file.read_six_coefficients(LED)
if GEOID_N:
    _d = -float(GEOID_N)*np.cos(geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0))/DR; rg0 = rg0 + _d
    say('[geoid] N = %+.2f m -> range shift %+.2f to %+.2f px' % (float(GEOID_N), _d.min(), _d.max())); del _d
nd = int(round(np.sqrt(os.path.getsize(DEMFILE)/4)))
dem = np.fromfile(DEMFILE, dtype=np.float32).reshape(nd, nd)
th_col = geometry.compute_theta_ih(coef, (R0 + np.arange(NR)*DR)/1000.0)
at, pd_, rd, yd = leader_file.read_attitude_record(LED); o = np.argsort(at)
tt = tf + az/(NA-1)*(te-tf)
pitch = np.radians(np.interp(tt, at[o], pd_[o])); yaw = np.radians(np.interp(tt, at[o], yd[o])); del tt
lat_rows = (LAT_TOP - rows/PPD).astype(np.float32)
dlat_m = np.float32(area_correction.DEG_M/PPD); dlon_m = (area_correction.DEG_M/PPD*np.cos(np.radians(lat_rows))).astype(np.float32)
Z1 = dem[rows-1, cols-1]; Z2 = dem[rows-1, cols]; Z3 = dem[rows-1, cols+1]; Z4 = dem[rows, cols-1]; Z6 = dem[rows, cols+1]
Z7 = dem[rows+1, cols-1]; Z8 = dem[rows+1, cols]; Z9 = dem[rows+1, cols+1]
p = (Z3+Z6+Z9-Z1-Z4-Z7)/(6.0*dlon_m); q = (Z1+Z2+Z3-Z7-Z8-Z9)/(6.0*dlat_m); del Z1, Z2, Z3, Z4, Z6, Z7, Z8, Z9; gc.collect()
slope = np.arctan(np.sqrt(p**2 + q**2)); p_safe = np.where(p == 0, 1.0, p)
aspect = np.where(p == 0, np.where(q > 0, np.pi, 0.0), np.pi - np.arctan(q/p_safe) + (np.pi/2)*np.sign(p_safe))
tau_rho = -np.degrees(np.arctan(np.tan(slope)*np.cos(aspect - HD - np.pi/2)))          # range slope, + facing the radar
tau_az = np.degrees(np.arctan(np.tan(slope)*np.cos(aspect - HD)))                      # azimuth slope
surf = (dlon_m*dlat_m*np.sqrt(1.0 + p**2 + q**2)).astype(np.float64)
del p, q, p_safe, aspect, slope, lat_rows, dlon_m; gc.collect()
th = geometry.compute_theta_ih(coef, (R0 + rg0*DR)/1000.0)
tl, area = area_correction.compute_facet_area_and_theta_l(dem, rows, cols, HD, PPD, LAT_TOP, th, pitch, yaw, area_ref_m2=AREF)
lit = tl < 90.0; lay = tau_rho >= np.degrees(th)
AC = 10*np.log10(AREF/(np.sin(th)*area))                                               # facet-level area correction (dB)
say('[input] facets %d, lit %d, layover %d (%.3f%%)' % (len(rows), lit.sum(), lay.sum(), 100*lay.mean()))

# footprint on the DEM grid
r0_, c0_ = rows.min(), rows.max() + 1,
r0_, r1_, c0_, c1_ = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
foot = np.zeros((r1_ - r0_, c1_ - c0_), bool); foot[rows - r0_, cols - c0_] = True
ext = [LON_LEFT + c0_/PPD, LON_LEFT + c1_/PPD, LAT_TOP - r1_/PPD, LAT_TOP - r0_/PPD]
def grid(v, blk=3):
    """facet values -> DEM-grid image (NaN outside), then blk x blk block mean"""
    img = np.full(foot.shape, np.nan, np.float32); img[rows - r0_, cols - c0_] = v
    hh, ww = img.shape[0]//blk*blk, img.shape[1]//blk*blk
    return np.nanmean(img[:hh, :ww].reshape(hh//blk, blk, ww//blk, blk), axis=(1, 3))
def scalebar(a, km=10):
    x0 = ext[0] + 0.05*(ext[1] - ext[0]); y0 = ext[2] + 0.05*(ext[3] - ext[2]); dx = km/(111.32*np.cos(np.radians(44.5)))
    a.plot([x0, x0 + dx], [y0, y0], 'k-', lw=2.5); a.text(x0 + dx/2, y0 + 0.012, '%d km' % km, ha='center', fontsize=8.5)
    a.annotate('N', xy=(ext[1] - 0.06*(ext[1] - ext[0]), ext[3] - 0.05*(ext[3] - ext[2])), xytext=(ext[1] - 0.06*(ext[1] - ext[0]), ext[3] - 0.16*(ext[3] - ext[2])),
               ha='center', fontsize=9, arrowprops=dict(arrowstyle='-|>', lw=1.2))
import warnings; warnings.simplefilter('ignore')

# ---------- Fig 3.2 study area: hillshade + footprint ----------
sub = dem[r0_ - 60:r1_ + 60, c0_ - 60:c1_ + 60].astype(float)
gy, gx = np.gradient(sub, 30.0, 22.0)
alt, azm = np.radians(45), np.radians(315)
sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
hs = np.sin(alt)*np.cos(sl) + np.cos(alt)*np.sin(sl)*np.cos(azm - asp)
ext2 = [LON_LEFT + (c0_ - 60)/PPD, LON_LEFT + (c1_ + 60)/PPD, LAT_TOP - (r1_ + 60)/PPD, LAT_TOP - (r0_ - 60)/PPD]
fig, a = plt.subplots(figsize=(6.8, 5.4))
a.imshow(hs, extent=ext2, origin='upper', cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
a.contour(np.flipud(foot.astype(float)), levels=[0.5], colors='red', linewidths=1.3, extent=ext, origin='lower')
a.set_aspect(ASPECT); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); scalebar(a)
a.text(0.02, 0.98, 'Scene footprint (red): ALOS PALSAR %s' % SID, transform=a.transAxes, va='top', fontsize=8.5, bbox=dict(fc='white', ec='none', alpha=0.85))
save(fig, 'thesis_study_area.png')

# ---------- Fig 3.3 CCI ----------
src = rasterio.open(CCIFILE); T = src.transform
br = ((LAT_TOP - rows/PPD - T.f)/T.e).astype(int); bc = ((LON_LEFT + cols/PPD - T.c)/T.a).astype(int)
cr0, cr1, cc0, cc1 = br.min(), br.max() + 1, bc.min(), bc.max() + 1
B = src.read(1, window=rasterio.windows.Window(cc0, cr0, cc1 - cc0, cr1 - cr0)).astype(float)
fb = np.zeros(B.shape, bool); fb[br - cr0, bc - cc0] = True; B = np.where(fb & (B > 0), B, np.nan)
extc = [T.c + cc0*T.a, T.c + cc1*T.a, T.f + cr1*T.e, T.f + cr0*T.e]
fig, a = plt.subplots(figsize=(6.8, 5.4))
m = a.imshow(B, extent=extc, origin='upper', cmap='YlGn', vmin=0, vmax=200, interpolation='nearest')
a.set_aspect(ASPECT); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); scalebar(a)
cb = fig.colorbar(m, ax=a, pad=0.02, shrink=0.85); cb.set_label('AGB (Mg ha$^{-1}$)')
say('[cci] median %.1f, p95 %.1f Mg/ha, %d cells' % (np.nanmedian(B), np.nanpercentile(B, 95), np.isfinite(B).sum()))
save(fig, 'thesis_cci.png')

# ---------- new: theta_loc map and range slope with layover ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
m0 = axs[0].imshow(grid(np.where(lit, tl, np.nan)), extent=ext, origin='upper', cmap='viridis', vmin=10, vmax=75, interpolation='nearest')
cb = fig.colorbar(m0, ax=axs[0], pad=0.02, shrink=0.8); cb.set_label('Local incidence angle $\\theta_{loc}$ (°)')
m1 = axs[1].imshow(grid(tau_rho), extent=ext, origin='upper', cmap='RdBu_r', vmin=-40, vmax=40, interpolation='nearest')
cb = fig.colorbar(m1, ax=axs[1], pad=0.02, shrink=0.8); cb.set_label('Range slope $\\tau_\\rho$ (°), positive towards the radar')
if lay.any():
    axs[1].plot(LON_LEFT + cols[lay]/PPD, LAT_TOP - rows[lay]/PPD, '.', color='k', ms=1.2, label='layover facets (%d)' % lay.sum()); axs[1].legend(loc='lower right', frameon=True)
for a, t in zip(axs, ['(a) Local incidence angle', '(b) Range slope and layover facets']):
    a.set_aspect(ASPECT); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); a.set_title(t, loc='left')
save(fig, 'thesis_theta_loc_map.png')

# ---------- Fig 4.2 mean correction versus range and azimuth slope ----------
e = np.arange(-65, 66, 5.0); ir = np.digitize(tau_rho, e) - 1; ia = np.digitize(tau_az, e) - 1
ok = lit & (ir >= 0) & (ir < len(e)-1) & (ia >= 0) & (ia < len(e)-1)
S = np.full((len(e)-1, len(e)-1), np.nan); C = np.zeros_like(S)
np.add.at(C, (ia[ok], ir[ok]), 1); tmp = np.zeros_like(S); np.add.at(tmp, (ia[ok], ir[ok]), AC[ok]); S = np.where(C >= 200, tmp/np.maximum(C, 1), np.nan)
TLm = np.zeros_like(S); np.add.at(TLm, (ia[ok], ir[ok]), tl[ok]); TLm = np.where(C >= 200, TLm/np.maximum(C, 1), np.nan)
fig, a = plt.subplots(figsize=(7.2, 5.6)); ec = 0.5*(e[:-1] + e[1:])
from matplotlib.colors import TwoSlopeNorm
m = a.pcolormesh(e, e, S, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-8, vcenter=0, vmax=3), shading='flat')
cs = a.contour(ec, ec, TLm, levels=[20, 30, 40, 50, 60, 70], colors='k', linewidths=0.8); a.clabel(cs, fmt='%d°', fontsize=8)
a.plot(0, 0, 'k+', ms=10, mew=1.5); a.set_xlabel('Range slope $\\tau_\\rho$ (°), positive towards the radar'); a.set_ylabel('Azimuth slope (°)')
cb = fig.colorbar(m, ax=a, pad=0.02); cb.set_label('Mean area correction $\\sigma^0_{terrain} - \\sigma^0_{flat}$ (dB)')
a.text(0.02, 0.98, 'contours: mean $\\theta_{loc}$; bins with $\\geq$ 200 facets', transform=a.transAxes, va='top', fontsize=8.5)
save(fig, 'thesis_slope_correction.png')

# ---------- new: facets per window and per pixel ----------
nwa, nwr = NA//WA, NR//WR
wa = np.floor(az/WA).astype(np.int64); wr = np.floor(rg0/WR).astype(np.int64)
inwin = (az >= 0) & (rg0 >= 0) & (wa < nwa) & (wr < nwr)
N_W = np.bincount((wa*nwr + wr)[inwin], minlength=nwa*nwr)
ai = np.floor(az).astype(np.int64); ri = np.floor(rg0).astype(np.int64); inp = (ai >= 0) & (ri >= 0) & (ai < NA) & (ri < NR)
N_P = np.bincount((ai*NR + ri)[inp], minlength=NA*NR)
frac_px = (N_P > 0).mean()
N_P4 = np.zeros(NA*NR, np.int32)                                                     # original per-pixel method: each facet spread to its 4 IDW neighbours
for da in (0, 1):
    for dr in (0, 1):
        pa = ai + da; pr = ri + dr; ok4 = (pa >= 0) & (pr >= 0) & (pa < NA) & (pr < NR)
        N_P4 += np.bincount((pa*NR + pr)[ok4], minlength=NA*NR).astype(np.int32)
frac_px4 = (N_P4 > 0).mean()
say('[window] facets per window: median %d, p5 %d, p95 %d; share of windows with >= 1 facet %.3f; share of single-look pixels with >= 1 facet %.4f' %
    (np.median(N_W[N_W > 0]), np.percentile(N_W[N_W > 0], 5), np.percentile(N_W[N_W > 0], 95), (N_W > 0).mean(), frac_px))
say('[window] share of single-look pixels receiving IDW weight from >= 1 facet (4 neighbours, original method) %.4f' % frac_px4)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].hist(N_W[N_W > 0], bins=np.arange(0, 40, 1), color='#4c72b0', edgecolor='white'); axs[0].set_xlabel('DEM facets per 31 x 7 window'); axs[0].set_ylabel('Windows')
axs[0].axvline(np.median(N_W[N_W > 0]), color='k', ls='--', lw=1); axs[0].set_title('(a) Window integration: facets per window', loc='left')
cnt = np.bincount(np.minimum(N_P, 4), minlength=5); cnt4 = np.bincount(np.minimum(N_P4, 4), minlength=5)
axs[1].bar(np.arange(5) - 0.2, cnt/cnt.sum(), 0.4, color='#4c72b0', edgecolor='white', label='nearest pixel (%.1f %% > 0)' % (100*frac_px))
axs[1].bar(np.arange(5) + 0.2, cnt4/cnt4.sum(), 0.4, color='#dd8452', edgecolor='white', label='4 IDW neighbours (%.1f %% > 0)' % (100*frac_px4)); axs[1].legend(frameon=False)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(['0', '1', '2', '3', '$\\geq$ 4']); axs[1].set_xlabel('DEM facets per single-look pixel'); axs[1].set_ylabel('Share of pixels')
axs[1].set_title('(b) Per-pixel assignment of DEM facets', loc='left')
save(fig, 'thesis_window_facets.png')

# ---------- new: line-time residual ----------
rl = 412 + NR*8; idx = np.arange(0, NA, 7); ms = np.empty(len(idx))
import struct
with open(IMG, 'rb') as f:
    for j, i in enumerate(idx):
        f.seek(720 + int(i)*rl + 44); ms[j] = struct.unpack('>I', f.read(4))[0]
b_, a_ = np.polyfit(idx.astype(float), ms, 1); res = ms - (a_ + b_*idx)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(idx[:300], res[:300], '.', ms=3, color='#4c72b0'); axs[0].axhline(0, color='k', lw=0.8); axs[0].set_xlabel('Azimuth line'); axs[0].set_ylabel('Stamp $-$ fit (ms)')
axs[0].set_title('(a) First 2,100 lines: rounding of the integer-millisecond stamp', loc='left')
axs[1].hist(res, bins=np.arange(-0.6, 0.61, 0.04), color='#4c72b0', edgecolor='white'); axs[1].set_xlabel('Stamp $-$ fit (ms)'); axs[1].set_ylabel('Lines')
axs[1].set_title('(b) All lines: residual within $\\pm$0.5 ms; fitted interval %.6f ms = 1/PRF' % b_, loc='left')
say('[time] fit: origin %.4f ms, interval %.6f ms, max |residual| %.3f ms, origin minus first stamp %+.4f ms' % (a_, b_, np.abs(res).max(), a_ - ms[0]))
save(fig, 'thesis_line_time.png')

# ---------- new: correction map A_flat / A_W (dB) ----------
A = np.zeros(nwa*nwr); s_ = inwin & lit
a0 = np.floor(az[s_]).astype(np.int64); r0 = np.floor(rg0[s_]).astype(np.int64); fa = az[s_] - a0; fr = rg0[s_] - r0
for da, dr, w in [(0, 0, (1-fa)*(1-fr)), (0, 1, (1-fa)*fr), (1, 0, fa*(1-fr)), (1, 1, fa*fr)]:
    pa = a0 + da; pr = r0 + dr; ok_ = (pa < nwa*WA) & (pr < nwr*WR)
    A += np.bincount(np.where(ok_, (pa//WA)*nwr + pr//WR, 0), weights=np.where(ok_, w, 0.0)*surf[s_], minlength=nwa*nwr)
A = A.reshape(nwa, nwr); th_W = th_col[:nwr*WR].reshape(nwr, WR).mean(axis=1)[None, :]; A_flat = WA*WR*AREF/np.sin(th_W)
corrW = np.where(A > 0, 10*np.log10(A_flat/np.maximum(A, 1e-9)), np.nan)
wa_c = np.clip(wa, 0, nwa-1); wr_c = np.clip(wr, 0, nwr-1); cf = np.where(inwin, corrW[wa_c, wr_c], np.nan)
fig, a = plt.subplots(figsize=(6.8, 5.4))
m = a.imshow(grid(cf), extent=ext, origin='upper', cmap='RdBu_r', vmin=-6, vmax=6, interpolation='nearest')
a.set_aspect(ASPECT); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)'); scalebar(a)
cb = fig.colorbar(m, ax=a, pad=0.02, shrink=0.85); cb.set_label('Window area correction $10\\log_{10}(A_{flat}/A_W)$ (dB)')
say('[corr map] window correction: median %+.2f dB, p5 %+.2f, p95 %+.2f' % tuple(np.nanpercentile(corrW, [50, 5, 95])))
save(fig, 'thesis_correction_map.png')

# ---------- new: HH n sweep from the main log ----------
lg = os.path.join(OUT, 'final_eq14_log.txt')
if os.path.exists(lg):
    L = open(lg).read().split('[T5]')[-1].splitlines(); ns, rs, ss_ = [], [], []
    for ln in L:
        mm = re.match(r'Eq14 n=([0-9.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', ln)
        if mm: ns.append(float(mm.group(1))); rs.append(float(mm.group(4))); ss_.append(float(mm.group(9)))
    if ns:
        fig, a = plt.subplots(figsize=(6.2, 3.8)); a2 = a.twinx()
        a.plot(ns, rs, 'o-', color='#4c72b0', label='$r$ with AGB'); a2.plot(ns, ss_, 's--', color='#c44e52', label='residual slope')
        a2.axhline(0, color='k', lw=0.6); a.set_xlabel('Exponent $n$'); a.set_ylabel('$r$', color='#4c72b0'); a2.set_ylabel('Residual slope (dB/°)', color='#c44e52')
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, loc='center right', frameon=False)
        a.set_title('HH, window integration: dependence on $n$', loc='left'); save(fig, 'thesis_hh_n_sweep.png')

# ---------- optional: geolocation correlation surface ----------
if os.environ.get('RTC_GEOCHKFIG'):
    from rasterio.warp import transform as wtf
    AD = os.path.join(BASE, 'AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1/AP_23763_FBD_F0880_RT1')
    ah = rasterio.open(AD + '_HV.tif'); AV = ah.read(1); LS = rasterio.open(AD + '.ls_map.tif').read(1)
    ex, ny = wtf('EPSG:4326', ah.crs, (LON_LEFT + cols/PPD).tolist(), (LAT_TOP - rows/PPD).tolist())
    aI = ((np.array(ny) - ah.transform.f)/ah.transform.e).astype(int); aJ = ((np.array(ex) - ah.transform.c)/ah.transform.a).astype(int)
    okA = (aI >= 0) & (aI < AV.shape[0]) & (aJ >= 0) & (aJ < AV.shape[1]); vA = np.zeros(len(rows))
    vA[okA] = np.where(LS[aI[okA], aJ[okA]] == 1, AV[aI[okA], aJ[okA]], 0.0); okA &= vA > 0; lv = np.log(vA[okA])
    bs = np.asarray(np.memmap(BSFILE, dtype=np.float32, mode='r', shape=(NA, NR)))
    Sm = np.log(np.maximum(uniform_filter(bs, (9, 3)), 1e-6)); del bs
    aa = np.round(az[okA]).astype(int); rr = np.round(rg0[okA]).astype(int)
    das = np.arange(-8, 9); drs = np.arange(-6, 7); Cs = np.zeros((len(das), len(drs)))
    for i, da in enumerate(das):
        ia = np.clip(aa + da, 0, NA-1)
        for j, dr in enumerate(drs): Cs[i, j] = np.corrcoef(lv, Sm[ia, np.clip(rr + dr, 0, NR-1)])[0, 1]
    bi, bj = np.unravel_index(Cs.argmax(), Cs.shape)
    say('[geochk] best offset azimuth %+d lines, range %+d px, r = %.4f; r at (0,0) = %.4f' % (das[bi], drs[bj], Cs.max(), Cs[8, 6]))
    fig, a = plt.subplots(figsize=(6.4, 5.2))
    m = a.pcolormesh(np.append(drs, 7) - 0.5, np.append(das, 9) - 0.5, Cs, cmap='viridis', shading='flat')
    a.plot(drs[bj], das[bi], 'r+', ms=14, mew=2); a.set_xlabel('Range offset (pixels)'); a.set_ylabel('Azimuth offset (lines)')
    cb = fig.colorbar(m, ax=a, pad=0.02); cb.set_label('$r$ (log ASF $\\gamma^0$ vs log smoothed $\\sigma^0_{flat}$)')
    a.text(0.02, 0.98, 'maximum at (%+d, %+d), $r$ = %.3f' % (drs[bj], das[bi], Cs.max()), transform=a.transAxes, va='top', color='white', fontsize=9)
    save(fig, 'thesis_geochk_surface.png')
LOG.close()
