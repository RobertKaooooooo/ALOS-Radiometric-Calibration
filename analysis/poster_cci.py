"""poster_cci.py -- ESA CCI AGB 2010 (v7.0) over the ALOS scene footprint, for the poster panel 2(b)
Run   : cd "/mnt/c/grd file" && python3 scripts/poster_cci.py
Output: scripts/out_ellip/poster_cci2010.png
"""
import os, numpy as np, rasterio, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE = os.environ.get('RTC_BASE', '/mnt/c/grd file')
CCI = os.path.join(BASE, 'N50W080_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif')
g = np.load(os.path.join(BASE, 'facet_results_FINAL2.npz')); rows, cols = g['rows'], g['cols']
src = rasterio.open(CCI); T = src.transform
print('[cci] %s | tags %s' % (os.path.basename(CCI), dict(src.tags())))
br = ((45.0 - rows/3600 - T.f)/T.e).astype(int); bc = ((-72.0 + cols/3600 - T.c)/T.a).astype(int)
r0, r1, c0, c1 = br.min(), br.max() + 1, bc.min(), bc.max() + 1
B = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
foot = np.zeros(B.shape, bool); foot[br - r0, bc - c0] = True
B = np.where(foot & (B > 0), B, np.nan)
ext = [T.c + c0*T.a, T.c + c1*T.a, T.f + r1*T.e, T.f + r0*T.e]
print('[cci] window %d x %d | AGB median %.1f Mg/ha | p95 %.1f' % (B.shape[0], B.shape[1], np.nanmedian(B), np.nanpercentile(B, 95)))
fig, a = plt.subplots(figsize=(7.4, 5.7))
m = a.imshow(B, extent=ext, origin='upper', cmap='YlGn', vmin=0, vmax=200, interpolation='nearest')
a.set_aspect(1/np.cos(np.radians(44.5))); a.set_xlabel('Longitude (°)'); a.set_ylabel('Latitude (°)')
a.set_title('ESA CCI above-ground biomass, 2010 (v7.0, 100 m)')
cb = fig.colorbar(m, ax=a, pad=0.02); cb.set_label('AGB (Mg ha$^{-1}$)')
out = os.path.join(BASE, 'scripts', 'out_ellip'); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, 'poster_cci2010.png'), dpi=250, facecolor='white', bbox_inches='tight'); print('[saved] %s' % os.path.join(out, 'poster_cci2010.png'))
