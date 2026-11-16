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
| `RTC_GEOCHK=1` | Geolocation check against the ASF product: correlation over azimuth/range offsets at every facet, with and without the geoid correction | `final_eq14_GEOCHK_log.txt` |
| `RTC_AZ_SHIFT=<lines>` | Constant azimuth offset added to all facets (sensitivity test); log names get the prefix `az<value>_` | `az<value>_final_eq14_*_log.txt` |
| `RTC_NO_TIMEFIX=1` | Use the integer-millisecond line-time origin (earlier results) | as above |
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

All results below use the EGM2008 undulation (N = -27.32 m) and the fitted
line-time origin (`img_file.read_line_times_fit`, default since 2026-11-16;
`RTC_NO_TIMEFIX=1` reproduces the earlier integer-millisecond origin, whose
logs are kept in `scripts/out_ellip_v1/`).

Same quantity before and after correction, versus CCI AGB 2010, 561,785 cells
of 100 m (`final_eq14_SAME_log.txt`), r / RMSE (dB):

| Quantity | Uncorrected | Per-pixel RTC | Window integration |
|---|---|---|---|
| sigma0 (n = 0) | 0.549 / 1.412 | 0.641 / 1.016 | 0.679 / 0.880 |
| gamma0 n = 0.5 | 0.549 / 1.411 | 0.658 / 0.967 | 0.699 / 0.828 |
| gamma0 n = 1 | 0.549 / 1.411 | 0.646 / 0.993 | 0.683 / 0.861 |

Benchmark (`final_eq14_ASF_log.txt`, 555,176 cells, 95 % CI from 10 km blocks):
ASF RTC gamma0, cell average, r = 0.670 [0.654, 0.686]; window integration
gamma0 n = 1, r = 0.684 [0.667, 0.701]; paired difference +0.014
[+0.009, +0.019]. Both cross-validation folds select n = 0.5 (test r 0.693 and
0.706).

Convergence (`final_eq14_OVS_K4_log.txt`, `final_eq14_OVS_K8_log.txt`): K = 4
is biased by about 2 dB (median sigma0 -10.38 dB versus -12.44 dB); K = 8
agrees with window integration (r 0.700 versus 0.699 for gamma0 n = 0.5).
K = 16 was run before the line-time fix (`out_ellip_v1`) and agrees with K = 8.

Geolocation (`final_eq14_GEOCHK_log.txt`): per-facet correlation with the ASF
product gives a range offset of +2.32 pixels without and +0.04 pixel with the
geoid correction, and an azimuth offset of +4.17 lines (about 13 m), whose
cause is not identified. Shifting all facets by -0.93, 0 and +3.3 lines
(relative to the earlier origin) changes r (gamma0 n = 0.5) to 0.699, 0.702
and 0.706; the ranking of the methods is unchanged.
