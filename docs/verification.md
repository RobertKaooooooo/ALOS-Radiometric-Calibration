# Verification Record

Every stage of the UAVSAR-to-ALOS Python translation in this package
was independently hand-derived at least once at a real test pixel and
compared against the running code's own output, with agreement to
within numerical precision (typically <0.0001 in vector components,
<0.001 deg in angles) except where a genuine implementation bug was
found -- in which case the bug was traced to its root cause, fixed,
and re-verified. This document records the specific numbers.

Test product: ALPSRP049750870 (White Mountain National Forest, NH, USA).

---

## 1. UAVSAR nL -- hand vs. code, scene-center test point

```
r_x1=6398597.508756  r_x2=6386822.681328  slt_range=19135.283220
r_l3=0.616275  r_look=0.906171
peg_ra=6386102.427056  gavgalt=12495.081700
ESA=0.0048769986  pitch=0.0323842078  yaw=-0.0300436262

hand-calculated nL = (-0.001188, -0.787530, 0.616275)
code-output nL     = (-0.001188, -0.787530, 0.616275)
difference = 0.000000
```

## 2. UAVSAR nE -- hand vs. code, real coordinate (ii=4650, jj=12098)

```
Z1=722.1772 Z2=721.9381 Z3=721.6991
Z4=720.5333 Z5=720.2543 Z6=719.9752
Z7=718.8894 Z8=718.5704 Z9=718.2513
p=-0.062609  q=0.272768

hand-calculated nE (with heading rotation applied) matches code nE
to within 0.000022 Euclidean distance. (Initial mismatch of 0.0334
without the rotation term was traced to a missing heading correction
-- root cause identified and corrected during verification.)
```

## 3. UAVSAR theta_l -- final local incidence angle

```
theta_l (from nE.nL) = 36.4291 deg
theta_l (program output) = 36.4291 deg
difference = 0.000003 deg
```

## 4. ALOS nL -- full derivation chain, scene center, hand vs. code

```
Step 0: f = (11201.251-11160.0)/(11220.0-11160.0) = 0.687517
Step 1: P=(1164806.94,-5039037.02,4814132.52) V=(-2922.32,4480.82,5382.86)
Step 2: |P|=7,065,730.054 m  h_hat=(-0.16485,0.71317,-0.68134)
Step 3: |V|=7,588.997 m/s  s_hat=(-0.38507,0.59044,0.70930)
Step 4: c_hat=(-0.90813,-0.37929,-0.17729)
Step 5: pitch=0.0000 deg  yaw=-2.7970 deg  theta_ih=38.7030 deg
Step 6: l_s=-0.030512  l_c=0.624539  l_h=-0.780398

hand nL = (-0.68406575, 0.30165369, -0.66407656)
code nL = (-0.68406575, 0.30165369, -0.66407656)
difference = 0.00000000
|nL| = 0.99996929
```

## 5. ALOS nE -- Horn's method, same scene-center pixel

```
p=-0.261660  q=-0.125826
slope=16.1902 deg  aspect=64.3182 deg  heading=-13.7063 deg
slope_r=0.284023  slope_a=0.060244

hand nE = (-0.05785466, -0.27275867, 0.96034137)
|nE| = 1.00000000
theta_l = arccos(nE . (-nL)) = 47.1218 deg
(cross-checked against independent 3-point validation table: 47.122 deg)
```

## 6. ESA algebra verification -- alpha=0 substitution into Eq. 11

```
l_s = sin(ESA)*cos(pitch)*cos(yaw) + cos(ESA)*(sin(pitch)*cos(theta_ih)*cos(yaw)+sin(theta_ih)*sin(yaw))
  at ESA=0: l_s = sin(pitch)*cos(theta_ih)*cos(yaw)+sin(theta_ih)*sin(yaw)  -- matches formula in use

l_h = sin(ESA)*sin(pitch) - cos(ESA)*cos(pitch)*cos(theta_ih)
  at ESA=0: l_h = -cos(pitch)*cos(theta_ih)  -- matches formula in use
```

Confirmed via Leader File: Yaw Steering Mode Flag = '0' (yaw steering
active), Electronic boresight = Mechanical boresight = 47.6000000 deg
(no residual electronic correction applied).

## 7. ECEF/local-frame bug -- flat-point diff before and after fix

```
Before fix, 8 known-flat points (DEM slope=0.00):
  diff (theta_l - theta_ih) range: 8.44 to 10.24 deg (should be 0)

After fix, same 8 points:
  diff = 0.000 for all (one exception at slope=11.44 deg, correctly non-zero)
```

Root cause: nL was rotated into ECEF using s_hat/c_hat/h_hat built from
P and V, while nE remained in the local (along-track/cross-track/
vertical) frame. Dotting the two together silently mixed coordinate
bases. Fix: keep nL in the same local frame as nE using
`geometry.compute_nL_local_frame` (Eq. 11 components directly, no ECEF
conversion) -- this also removes the need for V at all in the theta_l
computation itself (V is only used for the heading estimate).

## 8. Invalid-pixel bug (UAVSAR-side C++ code) -- before/after

This bug was found in `uavsar_calib.cpp` (Category A code, not part of
this Python package, but the diagnostic that found it was run against
this package's ALOS output for comparison). Eight per-row output
arrays (nEx/nEy/nEz, nLx/nLy/nLz, look_array, slope_array) were reused
across the outer row loop without being reset when a pixel failed the
RDC-range validity check -- invalid pixels silently inherited stale
values from the previous row.

```
Before fix: r_look 10-15 deg bin, n=1,752,374, mean_diff=60.977 deg (implausible)
After fix:  r_look 10-15 deg bin, n=1,598 (corrupted pixels removed), mean_diff=14.730 deg
After fix:  r_look 15-90 deg, mean_diff stable at -0.6 to +1.6 deg throughout
```

Fix: explicit `-100.0f` reset added at both `continue` points in the
C++ source, mirroring the pattern already used for `simsar[jj]`.

## 9. Heading sign bug -- ascending-orbit sanity check

```
Uncorrected heading: 194 deg (implies southbound flight)
Time-direction indicator (Data Set Summary Record): "ASCEND" (northbound)
  -- contradiction identified
Root cause: North_hat = cross(East_hat, P_hat) -- argument order reversed
  (cross product is anticommutative: A x B = -(B x A))
Fix: North_hat = cross(P_hat, East_hat)
Corrected heading: -13.706322775792932 deg (~346.3 deg, northbound) --
  consistent with ASCEND
```
