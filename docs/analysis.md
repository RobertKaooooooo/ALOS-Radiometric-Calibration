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
| `RTC_AZ_SHIFT=<lines>` | Constant azimuth timing calibration added to all facets; default 4.17 (estimated by `RTC_GEOCHK`), 0 disables it; a non-zero value prefixes log names with `az<value>_` | `az<value>_final_eq14_*_log.txt` |
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

All results below include the azimuth timing calibration of +4.17 lines
(default since 2026-11-16; see Geolocation).

Same quantity before and after correction, versus CCI AGB 2010, 561,792 cells
of 100 m (`final_eq14_SAME_log.txt`), r / RMSE (dB):

| Quantity | Uncorrected | Per-pixel RTC | Window integration |
|---|---|---|---|
| sigma0 (n = 0) | 0.553 / 1.406 | 0.648 / 1.004 | 0.685 / 0.870 |
| gamma0 n = 0.5 | 0.553 / 1.405 | 0.665 / 0.955 | 0.706 / 0.817 |
| gamma0 n = 1 | 0.552 / 1.405 | 0.654 / 0.981 | 0.690 / 0.850 |

Benchmark (`final_eq14_ASF_log.txt`, 555,186 cells, 95 % CI from 10 km blocks):
ASF RTC gamma0, cell average, r = 0.670 [0.654, 0.686]; window integration
gamma0 n = 1, r = 0.691 [0.674, 0.708]; paired difference +0.021
[+0.015, +0.026]. Because the azimuth calibration is estimated from the ASF
product, the difference without it is also reported: +0.014 [+0.009, +0.019].
Both cross-validation folds select n = 0.5 (test r 0.700 and 0.713).

Convergence (`final_eq14_OVS_K4_log.txt`, `final_eq14_OVS_K8_log.txt`): K = 4
is biased by about 2 dB (median sigma0 -10.39 dB versus -12.44 dB); K = 8
agrees with window integration (r 0.707 versus 0.706 for gamma0 n = 0.5).
K = 16 (rerun with the line-time fix and the azimuth calibration) agrees with K = 8 to 0.001 in r; the earlier run before both fixes
is kept in `out_ellip_v1`.

Geolocation (`final_eq14_GEOCHK_log.txt`): per-facet correlation with the ASF
product gives a range offset of +2.32 pixels without and +0.04 pixel with the
geoid correction. Without the azimuth calibration (`RTC_AZ_SHIFT=0`) the
azimuth offset is +4.17 lines; estimated separately in a 3 x 3 split of the
image (`[GEOCHK3x3]`) it is +4.11 to +4.24 lines, constant in range. With the
default calibration the offset is 0.00 lines overall and -0.06 to +0.08 lines
in the nine blocks, and r at zero offset equals the maximum (0.670).


## Three scenes (path 124, frame 880, ascending)

Prepared with `analysis/prepare_scene.py` (recomputing 200,000 facets of the
reference scene reproduces `facet_results_FINAL2.npz` exactly) and run with
`RTC_SID=<scene> RTC_AZ_SHIFT=4.17 RTC_GEOID_SHIFT=-27.32`. Outputs are in
`out_ellip_<scene>/`.

| | ALPSRP237630880 | ALPSRP230920880 | ALPSRP183950880 |
|---|---|---|---|
| Date | 2010-07-11 | 2010-05-26 | 2009-07-08 |
| Doppler centroid at 871 km | 125.9 Hz | 69.7 Hz | 122.3 Hz |
| Line-time fit correction | -0.93 line | +0.80 line | -0.07 line |
| Azimuth offset, no calibration | +4.17 | +4.04 | +4.45 |
| Range offset | +0.04 px | -0.01 px | -0.20 px |
| gamma0 n = 0.5, r / RMSE: uncorrected | 0.553 / 1.405 | 0.531 / 1.469 | 0.515 / 1.480 |
| per-pixel RTC | 0.665 / 0.955 | 0.649 / 0.982 | 0.625 / 1.001 |
| window integration | 0.706 / 0.817 | 0.688 / 0.849 | 0.667 / 0.864 |
| window n = 1 minus ASF (10 km bootstrap) | +0.021 [+0.015, +0.026] | +0.017 [+0.011, +0.023] | +0.025 [+0.019, +0.032] |
| CV-selected n | 0.5, 0.5 | 0.5, 0.5 | 0.5, 0.5 |

CCI AGB 2010 is the reference for all three scenes. The range offset of
ALPSRP183950880 (-0.20 px) is not explained.
