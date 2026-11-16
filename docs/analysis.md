# Analysis scripts

The scripts in `analysis/` produce every figure and table of the master's thesis
and the TASTI 2026 poster from the `alos_rtc` package. They are copied unchanged
from the author's working directory; each file begins with a docstring that
states its purpose, inputs and outputs.

No SAR, DEM or biomass data are distributed with this repository. ALOS PALSAR
products are subject to the JAXA and ASF terms of use, which do not permit
redistribution. Users must obtain the input data listed below themselves.

## Inputs

| Input | Source | Used for |
|---|---|---|
| ALOS PALSAR L1.1, ALPSRP237630880 (11 July 2010, FBD) | JAXA / ASF | Image to be corrected |
| `ALOS_HV_uncorrected_new.bin`, `ALOS_HH_uncorrected_new.bin` | Derived from the L1.1 product with `alos_rtc.backscatter` | sigma0 (flat ellipsoid), linear, float32, 18432 x 4704 |
| Copernicus GLO-30 DEM (`GLO30_ALOS_merged.bin`) | ESA / Copernicus | Facet geometry; heights refer to EGM2008 |
| `facet_results_FINAL2.npz` | Geolocation with `alos_rtc.geolocate` | Radar coordinates of each DEM facet |
| ESA CCI Biomass v7.0, AGB 2010, tile N50W080 | CEDA | Reference AGB, 100 m |
| ASF Hi-Res RTC product `AP_23763_FBD_F0880_RT1` | ASF | Geolocation check and operational benchmark (GAMMA, NED13 DEM) |

## Base path

All scripts locate the data through a base directory and import `alos_rtc` from
`<BASE>/ALOS-Radiometric-Calibration`. The default is `/mnt/c/grd file`
(Windows + WSL). Scripts that read `RTC_BASE` accept another location:

```
RTC_BASE=/path/to/data python3 analysis/final_eq14.py
```

`fig_slope_vs_correction.py`, `angle_inc_vs_slope.py`, `final_checks.py` and
`rtc_window_compare.py` still use the fixed default.

## Vertical datum

GLO-30 heights are orthometric (EGM2008). The range-Doppler geolocation needs
ellipsoidal heights, h = H + N. The analysis applies a constant undulation via
`RTC_GEOID_SHIFT` (metres), which shifts each facet in slant range by
-N cos(theta_inc) / DR. The value used for the thesis is the EGM2008 undulation
at the scene centre (44.559 N, 71.111 W), obtained with pyproj 3.8.0
(EPSG:4326+3855 to EPSG:4979):

| Location | N (m) |
|---|---|
| Scene centre | -27.32 |
| Corners | -26.78 to -27.95 |

The across-scene variation (about 1.2 m) changes the range shift by less than
0.1 pixel, so a single constant is used. N is fixed from the geoid model and is
not tuned against the evaluation data. With `RTC_GEOID_SHIFT` set, outputs go
to `scripts/out_ellip/`; otherwise to `scripts/out/`.

## final_eq14.py: run modes

Main analysis: window integration with the bilinear weights of
Simard et al. (2016) Eq. 14 on 31 x 7 single-look windows (about 100 m),
evaluated against CCI AGB on 100 m cells.

| Environment variable | Purpose | Log file |
|---|---|---|
| (none) | Main run: tables T1 to T5, figures F1 and F4 | `final_eq14_log.txt` |
| `RTC_SAME=1` | Same physical quantity before and after correction (sigma0, gamma0 n = 0.5, gamma0 n = 1); the uncorrected reference uses cos^n(theta_inc) | `final_eq14_SAME_log.txt` |
| `RTC_ASF=1` | Benchmark against the ASF RTC product, sampled at the same facets and averaged to the same 100 m cells (layover/shadow pixels excluded with the ASF `ls_map`) | `final_eq14_ASF_log.txt` |
| `RTC_ASF=1 RTC_BOOT=1` | Adds a spatial block bootstrap (5 km and 10 km blocks, 1000 resamples, paired) for r and r differences, and a 2-fold 5 km checkerboard cross-validation of the exponent n | `final_eq14_ASF_log.txt`, sections `[BOOT]` and `[CV]` |
| `RTC_OVS=K` | Baseline: per-pixel correction on DEM facets subdivided K x K (planar oversampling), then window mean; compared with window integration on identical cells | `final_eq14_OVS_log.txt` |
| `RTC_ONLY_T6=1` | Geoid lookup only (T6) | `final_eq14_T6_log.txt` |
| `RTC_POSTER=1`, `RTC_FIGS_ONLY=1` | Poster figure variants | `final_eq14_log.txt` |
| `RTC_GEOID_SHIFT=<m>` | Constant geoid undulation N (see above) | output folder `out_ellip/` |

Commands used for the thesis (WSL):

```
cd "/mnt/c/grd file/scripts"
export RTC_GEOID_SHIFT=-27.32
python3 final_eq14.py
RTC_SAME=1 python3 final_eq14.py
RTC_ASF=1 RTC_BOOT=1 python3 final_eq14.py
for K in 4 8 16; do RTC_OVS=$K python3 final_eq14.py; cp out_ellip/final_eq14_OVS_log.txt out_ellip/final_eq14_OVS_K${K}_log.txt; done
```

Peak memory of the `RTC_OVS` runs is about 5.8 GB; K = 16 takes roughly 40 min
on the author's laptop.

## Other scripts

| Script | Purpose | Main output |
|---|---|---|
| `final_checks.py` | Regression figure, like-for-like distributions, Simard Fig. 4 and Fig. 10 analogues; `RTC_FIG4M2=1` gives the area comparison in m^2 per radar pixel | `final_checks_log.txt`, `fig4_m2_log.txt` |
| `rtc_window_compare.py` | First comparison of the per-pixel code and window integration | `rtc_window_compare_log.txt` |
| `v2_v3_checks.py` | Area-only correction, theory for a pure range slope, floor vs bilinear allocation | `v2_v3_log.txt` |
| `fig_slope_vs_correction.py` | Slope vs correction for 10 to 20 deg local incidence angle | `out/` |
| `angle_inc_vs_slope.py` | Separates the effects of theta_inc and range slope | `angle_inc_vs_slope_log.txt` |
| `poster_fig4.py` | Area correction vs range slope and theta_loc (5.99 M facets) | poster figure |
| `poster_maps.py` | Before/after maps | `poster_maps_log.txt` |
| `poster_cci.py` | CCI AGB 2010 over the scene | `poster_cci2010.png` |

## Results used in the thesis (HV, program output)

Window integration, gamma0 n = 0.5, versus CCI AGB 2010, 561,758 cells of 100 m
(`final_eq14_SAME_log.txt`):

| Quantity | Uncorrected | Per-pixel RTC | Window integration |
|---|---|---|---|
| sigma0 (n = 0), r / RMSE (dB) | 0.551 / 1.410 | 0.643 / 1.011 | 0.681 / 0.876 |
| gamma0 n = 0.5, r / RMSE (dB) | 0.551 / 1.409 | 0.660 / 0.963 | 0.702 / 0.824 |
| gamma0 n = 1, r / RMSE (dB) | 0.550 / 1.408 | 0.649 / 0.988 | 0.686 / 0.858 |

Benchmark and statistics (`final_eq14_ASF_log.txt`, 555,154 cells): ASF RTC
gamma0, cell average, r = 0.670 [0.654, 0.686]; window integration gamma0 n = 1,
r = 0.686 [0.669, 0.703] (95 % CI, 10 km blocks). Both cross-validation folds
select n = 0.5.

Convergence of the oversampled per-pixel baseline (`final_eq14_OVS_K8_log.txt`, `final_eq14_OVS_K16_log.txt`; the K = 4 values were taken from the terminal output and its log still has to be saved by re-running):
K = 4 is biased by about 2 dB (median sigma0 -10.39 dB); K = 8 and K = 16 agree
with window integration (median -12.44 dB, identical r). The two implementations
are therefore equivalent in accuracy once the facets are dense enough.
