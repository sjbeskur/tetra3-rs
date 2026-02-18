# tetra3-rs Algorithm

## 1. Star Detection (`centroids.rs`)

- Subtract local background (25x25 mean filter) from the image
- Estimate noise level (RMS of all pixels)
- Threshold at `sigma x noise_std` to create a binary mask
- Clean the mask with morphological opening (3x3 cross) to remove hot pixels
- Label connected components (lutz algorithm) -- each blob is a candidate star
- Filter blobs by area (5-100 pixels) and compute intensity-weighted centroid (center of mass)
- Sort by brightness, brightest first

## 2. Pattern Hashing (the core idea)

The key insight: any group of 4 stars forms 6 pairwise angular distances. Divide the 5 smallest by the largest to get 5 **edge ratios** in [0,1]. These ratios are **rotation/scale invariant** -- they only depend on the geometric shape of the 4-star pattern, not on where the camera is pointing.

The database pre-computes these edge ratios for millions of known 4-star patterns from the star catalog and stores them in a hash table (`hash_table.rs`). Each ratio is quantized into bins (e.g., 50 bins), and the bin indices form the hash key.

## 3. Solving (`solver.rs`)

**Pattern matching loop:**

- Take the top N brightest centroids (default 8 = `pattern_checking_stars`)
- Try all C(N, 4) combinations (70 for N=8)
- For each 4-star combo:
  - Convert pixel positions -> unit vectors using a pinhole camera model (`geometry.rs`) with the estimated FOV
  - Compute pairwise angles -> edge ratios
  - Quantize ratios to bin ranges (with +/-`pattern_max_error` tolerance)
  - Generate all hash keys in that range and look them up in the database

**For each hash hit (candidate catalog pattern):**

- Recompute catalog edge ratios and verify they're within bounds (filters ~99% of false hash collisions)
- Estimate the FOV from the ratio of catalog vs image largest edge
- Check FOV is within the expected range
- Sort both image and catalog vectors by distance from centroid (establishes correspondence)
- Compute the **rotation matrix** from image frame -> sky frame using SVD (Kabsch algorithm)

**Verification:**

- Use the rotation to project all catalog stars in the FOV back to pixel coordinates
- Match projected catalog centroids against all detected image centroids within a pixel radius
- Run a **binomial statistical test**: "If the match were random, what's the probability of getting this many coincidences?"
  - `prob_single = num_nearby_stars x match_radius^2` (chance a random point matches one star)
  - Apply Bonferroni correction: divide threshold by total number of patterns in the database
  - If probability < threshold -> accept the match

## 4. Refinement

Once accepted:

- Recompute the rotation matrix using **all** matched stars (not just the initial 4)
- Extract RA, Dec, Roll from the rotation matrix
- Refine the FOV by comparing pairwise angles between matched image/catalog vectors
- Compute RMSE residual in arcseconds

## 5. Calibration (`calibration.rs`)

When a camera calibration file is provided:

- Undistort raw centroids using the OpenCV5 model (iterative fixed-point inversion)
- Re-center them so the principal point maps to image center
- Reproject using `fx` for both axes (so the solver's single-FOV model works)
- Supply the exact FOV from `2 * atan(width / (2 * fx))`
