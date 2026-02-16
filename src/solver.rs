//! Star pattern matching and plate solving.
//!
//! Implements the core tetra3 solving algorithm:
//! 1. Take star centroids from an image
//! 2. Form 4-star patterns from the brightest stars
//! 3. Compute edge-ratio fingerprints and look them up in the hash table
//! 4. For each candidate match, estimate rotation and verify by matching all visible stars
//! 5. Accept if the match probability is below threshold

use std::time::Instant;

use nalgebra::{Matrix3, Vector3};

use crate::database::StarDatabase;
use crate::geometry;
use crate::hash_table;

/// Parameters for the solve operation.
#[derive(Debug, Clone)]
pub struct SolveParams {
    /// Estimated field of view in degrees (speeds up solving significantly).
    pub fov_estimate: Option<f64>,
    /// Maximum FOV error from estimate in degrees.
    pub fov_max_error: Option<f64>,
    /// Number of brightest stars to form patterns from (default 8).
    pub pattern_checking_stars: usize,
    /// Maximum match distance as a fraction of FOV (default 0.01).
    pub match_radius: f64,
    /// Maximum false-positive probability to accept (default 1e-3).
    pub match_threshold: f64,
    /// Timeout in milliseconds (None = no timeout).
    pub solve_timeout: Option<f64>,
    /// Known distortion coefficient (None = unknown, will not correct).
    pub distortion: Option<f64>,
}

impl Default for SolveParams {
    fn default() -> Self {
        Self {
            fov_estimate: None,
            fov_max_error: None,
            pattern_checking_stars: 8,
            match_radius: 0.01,
            match_threshold: 1e-3,
            solve_timeout: None,
            distortion: None,
        }
    }
}

/// Result of a successful plate solve.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Right ascension of image center in degrees.
    pub ra: f64,
    /// Declination of image center in degrees.
    pub dec: f64,
    /// Roll angle relative to north celestial pole in degrees.
    pub roll: f64,
    /// Calculated horizontal field of view in degrees.
    pub fov: f64,
    /// Calculated distortion coefficient (if distortion was enabled).
    pub distortion: Option<f64>,
    /// RMS residual of matched stars in arcseconds.
    pub rmse: f64,
    /// Number of stars matched between image and catalog.
    pub num_matches: usize,
    /// Probability that the match is a false positive.
    pub prob: f64,
    /// Time spent solving in milliseconds.
    pub t_solve_ms: f64,
    /// The rotation matrix from camera frame to celestial frame.
    pub rotation_matrix: Matrix3<f64>,
}

/// Find matching pairs of centroids within a pixel radius.
///
/// Returns pairs of `(image_index, catalog_index)`. Matches are unique 1-to-1:
/// each image centroid matches at most one catalog centroid and vice versa.
pub fn find_centroid_matches(
    image_centroids: &[(f64, f64)],
    catalog_centroids: &[(f64, f64)],
    radius: f64,
) -> Vec<(usize, usize)> {
    let mut matches: Vec<(usize, usize, f64)> = Vec::new();

    for (i, &(iy, ix)) in image_centroids.iter().enumerate() {
        for (j, &(cy, cx)) in catalog_centroids.iter().enumerate() {
            let dy = iy - cy;
            let dx = ix - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            if dist < radius {
                matches.push((i, j, dist));
            }
        }
    }

    // Sort by distance so closest matches are preferred
    matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Enforce unique 1-to-1 matching
    let mut used_image = vec![false; image_centroids.len()];
    let mut used_catalog = vec![false; catalog_centroids.len()];
    let mut result = Vec::new();

    for (i, j, _) in matches {
        if !used_image[i] && !used_catalog[j] {
            used_image[i] = true;
            used_catalog[j] = true;
            result.push((i, j));
        }
    }

    result
}

/// Compute pairwise distances between unit vectors, returned as angles in radians.
///
/// Returns sorted edge angles for a set of vectors (ascending).
/// For N vectors, returns N*(N-1)/2 distances.
fn pairwise_angles(vectors: &[Vector3<f64>]) -> Vec<f64> {
    let n = vectors.len();
    let mut angles = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (vectors[i] - vectors[j]).norm();
            angles.push(2.0 * (0.5 * dist).asin());
        }
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles
}

/// Compute the edge-ratio pattern from sorted edge angles.
///
/// Divides all edges by the largest, drops the largest (always 1.0),
/// returning `N-1` ratios in `[0, 1]`.
fn edge_ratios(sorted_angles: &[f64]) -> Vec<f64> {
    let largest = *sorted_angles.last().unwrap();
    sorted_angles[..sorted_angles.len() - 1]
        .iter()
        .map(|&a| a / largest)
        .collect()
}

/// Sort vectors by distance from their centroid (for unique pattern ordering).
fn sort_by_centroid_distance(vectors: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
    let centroid: Vector3<f64> =
        vectors.iter().copied().sum::<Vector3<f64>>() / vectors.len() as f64;
    let mut indexed: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, (v - centroid).norm()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed.iter().map(|&(i, _)| vectors[i]).collect()
}

/// Generate all C(n, k) combinations of indices.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    fn enumerate(combo: &mut Vec<usize>, start: usize, remaining: usize, n: usize, total_k: usize, result: &mut Vec<Vec<usize>>) {
        if remaining == 0 {
            result.push(combo.clone());
            return;
        }
        for i in start..=(n - remaining) {
            let idx = total_k - remaining;
            combo[idx] = i;
            enumerate(combo, i + 1, remaining - 1, n, total_k, result);
        }
    }
    enumerate(&mut combo, 0, k, n, k, &mut result);
    result
}

/// Generate all hash codes in a range grid and return unique sorted keys.
///
/// For each of the `p` edge ratios, we have a `[min, max]` bin range.
/// This produces the Cartesian product of all bin ranges, then sorts
/// each key ascending and deduplicates.
fn hash_code_grid(mins: &[u64], maxs: &[u64]) -> Vec<Vec<u16>> {
    let p = mins.len();
    let mut codes: Vec<Vec<u16>> = vec![vec![]];

    for i in 0..p {
        let mut new_codes = Vec::new();
        for code in &codes {
            for bin in mins[i]..=maxs[i] {
                let mut c = code.clone();
                c.push(bin as u16);
                new_codes.push(c);
            }
        }
        codes = new_codes;
    }

    // Sort each code ascending and deduplicate
    for code in &mut codes {
        code.sort();
    }
    codes.sort();
    codes.dedup();
    codes
}

/// Binomial CDF: P(X <= k) where X ~ Binomial(n, p).
///
/// Used for the statistical verification test. Computes using the
/// regularized incomplete beta function approximation.
fn binom_cdf(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    // Sum from x=0 to k of C(n,x) * p^x * (1-p)^(n-x)
    // Use log-space for numerical stability
    let mut cdf = 0.0;
    let mut log_pmf = n as f64 * (1.0 - p).ln(); // log(C(n,0) * p^0 * (1-p)^n)
    cdf += log_pmf.exp();

    for x in 1..=k {
        // log_pmf += log(C(n,x)/C(n,x-1)) + log(p/(1-p))
        log_pmf += ((n - x + 1) as f64 / x as f64).ln() + (p / (1.0 - p)).ln();
        cdf += log_pmf.exp();
    }

    cdf.min(1.0)
}

/// Solve for the sky location from a list of star centroids.
///
/// This is the core solver. It tries combinations of the brightest stars,
/// hashes their edge-ratio patterns, looks up candidates in the database,
/// and verifies matches statistically.
pub fn solve_from_centroids(
    db: &dyn StarDatabase,
    star_centroids: &[(f64, f64)],
    size: (u32, u32),
    params: &SolveParams,
) -> Option<SolveResult> {
    let props = db.properties();
    let (height, width) = (size.0, size.1);
    let p_size = props.pattern_size;
    let p_bins = props.pattern_bins;
    let p_max_err = props.pattern_max_error;
    let presorted = props.presort_patterns;
    let num_stars_to_use = props.verification_stars_per_fov;

    let num_patterns = db.pattern_catalog().num_rows() / 2;
    let match_threshold = params.match_threshold / num_patterns as f64;

    let fov_initial = if let Some(est) = params.fov_estimate {
        est.to_radians()
    } else {
        ((props.max_fov + props.min_fov) / 2.0).to_radians()
    };

    let fov_estimate_rad = params.fov_estimate.map(|f| f.to_radians());
    let fov_max_error_rad = params.fov_max_error.map(|f| f.to_radians());

    // Trim centroids to verification star count
    let max_centroids = star_centroids.len().min(num_stars_to_use);
    let image_centroids = &star_centroids[..max_centroids];

    // Apply known distortion correction if provided
    let image_centroids_undist: Vec<(f64, f64)> = if let Some(k) = params.distortion {
        if k != 0.0 {
            geometry::undistort_centroids(image_centroids, size, k)
        } else {
            image_centroids.to_vec()
        }
    } else {
        image_centroids.to_vec()
    };

    let t0 = Instant::now();
    let pattern_check_count = image_centroids
        .len()
        .min(params.pattern_checking_stars);

    // Edge ratio pattern length: C(p_size, 2) - 1
    let _pattlen = p_size * (p_size - 1) / 2 - 1;

    // Try all combinations of p_size stars from the brightest pattern_checking_stars
    for pattern_indices in combinations(pattern_check_count, p_size) {
        // Check timeout
        if let Some(timeout_ms) = params.solve_timeout {
            if t0.elapsed().as_secs_f64() * 1000.0 > timeout_ms {
                break;
            }
        }

        // Get pattern centroids and compute vectors with initial FOV estimate
        let pattern_centroids: Vec<(f64, f64)> = pattern_indices
            .iter()
            .map(|&i| image_centroids_undist[i])
            .collect();
        let pattern_vectors = geometry::compute_vectors(
            &pattern_centroids,
            width,
            height,
            fov_initial.to_degrees(),
        );

        // Calculate edge angles and ratios
        let edge_angles = pairwise_angles(&pattern_vectors);
        let image_largest_edge = *edge_angles.last().unwrap();
        let image_ratios = edge_ratios(&edge_angles);

        // Build min/max ratio bounds with tolerance
        let ratio_mins: Vec<f64> = image_ratios.iter().map(|r| r - p_max_err).collect();
        let ratio_maxs: Vec<f64> = image_ratios.iter().map(|r| r + p_max_err).collect();

        // Convert to bin ranges
        let bin_mins: Vec<u64> = ratio_mins
            .iter()
            .map(|&r| (r * p_bins as f64).max(0.0) as u64)
            .collect();
        let bin_maxs: Vec<u64> = ratio_maxs
            .iter()
            .map(|&r| (r * p_bins as f64).min(p_bins as f64) as u64)
            .collect();

        // Generate all hash codes to look up
        let hash_codes = hash_code_grid(&bin_mins, &bin_maxs);

        // Hash each code and look up in pattern catalog
        let catalog = db.pattern_catalog();
        let hash_indices = hash_table::key_to_index_batch(&hash_codes, p_bins, catalog.num_rows() as u64);

        for &hash_index in &hash_indices {
            let match_rows = hash_table::get_table_indices_from_hash(hash_index, catalog);
            if match_rows.is_empty() {
                continue;
            }

            // Filter by FOV if we have pattern_largest_edge data
            let match_rows = if let (Some(largest_edges), Some(fov_est), Some(fov_err)) =
                (db.pattern_largest_edge(), fov_estimate_rad, fov_max_error_rad)
            {
                match_rows
                    .into_iter()
                    .filter(|&row| {
                        let le = largest_edges[row];
                        let fov2 = le / 1000.0 / image_largest_edge * fov_initial;
                        (fov2 - fov_est).abs() < fov_err
                    })
                    .collect::<Vec<_>>()
            } else {
                match_rows
            };

            for &row in &match_rows {
                let catalog_star_ids = catalog.get_row(row);

                // Get catalog star vectors for this pattern
                let catalog_pattern_vectors: Vec<Vector3<f64>> = catalog_star_ids
                    .iter()
                    .map(|&id| {
                        let v = db.star_vector(id as usize);
                        Vector3::new(v[0], v[1], v[2])
                    })
                    .collect();

                // Compute catalog edge ratios and verify they match
                let cat_angles = pairwise_angles(&catalog_pattern_vectors);
                let cat_largest = *cat_angles.last().unwrap();
                let cat_ratios = edge_ratios(&cat_angles);

                // Check all ratios are within bounds
                let ratios_match = cat_ratios.iter().zip(ratio_mins.iter().zip(ratio_maxs.iter()))
                    .all(|(&cr, (&mn, &mx))| cr > mn && cr < mx);
                if !ratios_match {
                    continue;
                }

                // Estimate FOV from pattern
                let fov = if fov_estimate_rad.is_some() {
                    cat_largest / image_largest_edge * fov_initial
                } else {
                    // Calculate from camera projection
                    let max_pixel_dist = pattern_centroids
                        .iter()
                        .flat_map(|a| pattern_centroids.iter().map(move |b| {
                            let dy = a.0 - b.0;
                            let dx = a.1 - b.1;
                            (dy * dy + dx * dx).sqrt()
                        }))
                        .fold(0.0f64, f64::max);
                    let f = max_pixel_dist / 2.0 / (cat_largest / 2.0).tan();
                    2.0 * (width as f64 / 2.0 / f).atan()
                };

                // FOV check
                if let (Some(fov_est), Some(fov_err)) = (fov_estimate_rad, fov_max_error_rad) {
                    if (fov - fov_est).abs() > fov_err {
                        continue;
                    }
                }

                // Recompute image vectors with refined FOV and sort by centroid distance
                let refined_pattern_vectors = geometry::compute_vectors(
                    &pattern_centroids,
                    width,
                    height,
                    fov.to_degrees(),
                );
                let sorted_image_vectors = sort_by_centroid_distance(&refined_pattern_vectors);

                // Sort catalog vectors the same way (unless presorted)
                let sorted_catalog_vectors = if presorted {
                    catalog_pattern_vectors.clone()
                } else {
                    sort_by_centroid_distance(&catalog_pattern_vectors)
                };

                // Compute rotation matrix from pattern match
                let rotation_matrix = match geometry::find_rotation_matrix(
                    &sorted_image_vectors,
                    &sorted_catalog_vectors,
                ) {
                    Some(r) => r,
                    None => continue,
                };

                // Find all catalog stars within the diagonal FOV
                let image_center_vector = [
                    rotation_matrix[(0, 0)],
                    rotation_matrix[(0, 1)],
                    rotation_matrix[(0, 2)],
                ];
                let fov_diagonal = fov
                    * ((width as f64).powi(2) + (height as f64).powi(2)).sqrt()
                    / width as f64;
                let nearby_inds = db.get_nearby_stars(&image_center_vector, fov_diagonal / 2.0);

                // Derotate nearby stars to image frame and compute centroids
                let mut nearby_centroids = Vec::new();
                let mut nearby_vectors = Vec::new();
                let mut nearby_kept_inds = Vec::new();

                for &star_idx in &nearby_inds {
                    let sv = db.star_vector(star_idx);
                    let v = Vector3::new(sv[0], sv[1], sv[2]);
                    let derot = rotation_matrix * v;
                    // Only keep if in front of camera (z > 0 in camera frame...
                    // actually the Python uses _compute_centroids with trim)
                    if derot.x <= 0.0 {
                        continue; // behind camera (x is boresight in Python convention)
                    }
                    // Convert to centroid using the Python convention
                    // In Python: centroids[:, 2:0:-1] means [z, y] mapped from vector [x, y, z]
                    // scale_factor = -width/2/tan(fov/2)
                    let scale_factor = -(width as f64) / 2.0 / (fov / 2.0).tan();
                    let cy = scale_factor * derot.z / derot.x + height as f64 / 2.0;
                    let cx = scale_factor * derot.y / derot.x + width as f64 / 2.0;

                    // Trim to within image bounds
                    if cy > 0.0 && cy < height as f64 && cx > 0.0 && cx < width as f64 {
                        nearby_centroids.push((cy, cx));
                        nearby_vectors.push(v);
                        nearby_kept_inds.push(star_idx);
                    }
                }

                // Only keep as many as image centroids
                let keep_count = nearby_centroids.len().min(image_centroids.len());
                nearby_centroids.truncate(keep_count);
                nearby_vectors.truncate(keep_count);
                nearby_kept_inds.truncate(keep_count);

                // Match image centroids to projected catalog centroids
                let match_radius_px = width as f64 * params.match_radius;
                let matched = find_centroid_matches(
                    &image_centroids_undist,
                    &nearby_centroids,
                    match_radius_px,
                );

                let num_extracted = image_centroids.len();
                let num_nearby = nearby_centroids.len();
                let num_matched = matched.len();

                // Statistical verification (binomial test)
                let prob_single = num_nearby as f64 * params.match_radius.powi(2);
                let prob_mismatch = binom_cdf(
                    num_extracted - (num_matched.saturating_sub(2)),
                    num_extracted,
                    1.0 - prob_single,
                );

                if prob_mismatch < match_threshold {
                    // Match accepted! Refine the solution.

                    // Get matched vectors for refinement
                    let matched_image_centroids: Vec<(f64, f64)> =
                        matched.iter().map(|&(i, _)| image_centroids[i]).collect();
                    let matched_image_vectors = geometry::compute_vectors(
                        &matched_image_centroids,
                        width,
                        height,
                        fov.to_degrees(),
                    );
                    let matched_catalog_vectors: Vec<Vector3<f64>> = matched
                        .iter()
                        .map(|&(_, j)| nearby_vectors[j])
                        .collect();

                    // Recompute rotation matrix with all matches
                    let rotation_matrix = match geometry::find_rotation_matrix(
                        &matched_image_vectors,
                        &matched_catalog_vectors,
                    ) {
                        Some(r) => r,
                        None => continue,
                    };

                    // Extract RA, Dec, Roll
                    let ra = rotation_matrix[(0, 1)]
                        .atan2(rotation_matrix[(0, 0)])
                        .to_degrees()
                        .rem_euclid(360.0);
                    let dec = rotation_matrix[(0, 2)]
                        .atan2(
                            (rotation_matrix[(1, 2)].powi(2) + rotation_matrix[(2, 2)].powi(2))
                                .sqrt(),
                        )
                        .to_degrees();
                    let roll = rotation_matrix[(1, 2)]
                        .atan2(rotation_matrix[(2, 2)])
                        .to_degrees()
                        .rem_euclid(360.0);

                    // Refine FOV
                    let fov_final = if params.distortion.is_none() {
                        // Compare mutual angles to scale FOV
                        let angles_camera = pairwise_angles(&matched_image_vectors);
                        let angles_catalog = pairwise_angles(&matched_catalog_vectors);
                        let scale: f64 = angles_catalog
                            .iter()
                            .zip(angles_camera.iter())
                            .map(|(c, i)| c / i)
                            .sum::<f64>()
                            / angles_catalog.len() as f64;
                        fov * scale
                    } else {
                        fov
                    };

                    let distortion_out = params.distortion;

                    // Undistort matched centroids for final residual calculation
                    let matched_centroids_undist = if let Some(k) = params.distortion {
                        if k != 0.0 {
                            geometry::undistort_centroids(&matched_image_centroids, size, k)
                        } else {
                            matched_image_centroids.clone()
                        }
                    } else {
                        matched_image_centroids.clone()
                    };

                    // Compute final vectors and residual
                    let final_vectors = geometry::compute_vectors(
                        &matched_centroids_undist,
                        width,
                        height,
                        fov_final.to_degrees(),
                    );
                    // Rotate to sky frame
                    let final_sky_vectors: Vec<Vector3<f64>> = final_vectors
                        .iter()
                        .map(|v| rotation_matrix.transpose() * v)
                        .collect();

                    // RMSE in arcseconds
                    let mut sum_sq = 0.0;
                    for (fv, cv) in final_sky_vectors.iter().zip(matched_catalog_vectors.iter()) {
                        let dist = (fv - cv).norm();
                        let angle = 2.0 * (0.5 * dist).asin();
                        sum_sq += angle * angle;
                    }
                    let rmse = (sum_sq / num_matched as f64).sqrt().to_degrees() * 3600.0;

                    let t_solve = t0.elapsed().as_secs_f64() * 1000.0;

                    return Some(SolveResult {
                        ra,
                        dec,
                        roll,
                        fov: fov_final.to_degrees(),
                        distortion: distortion_out,
                        rmse,
                        num_matches: num_matched,
                        prob: prob_mismatch * num_patterns as f64,
                        t_solve_ms: t_solve,
                        rotation_matrix,
                    });
                }
            }
        }
    }

    None // Failed to solve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_angles_three_orthogonal() {
        let vecs = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let angles = pairwise_angles(&vecs);
        // All pairs are 90 degrees apart
        assert_eq!(angles.len(), 3);
        for &a in &angles {
            assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_edge_ratios_equal_angles() {
        // All equal angles -> all ratios should be 1.0 (except dropped largest)
        let angles = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let ratios = edge_ratios(&angles);
        assert_eq!(ratios.len(), 5);
        for &r in &ratios {
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_find_centroid_matches_exact() {
        let image = vec![(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)];
        let catalog = vec![(100.5, 200.5), (300.5, 400.5), (800.0, 800.0)];
        let matches = find_centroid_matches(&image, &catalog, 2.0);
        assert_eq!(matches.len(), 2);
        // Should match 0->0 and 1->1
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    #[test]
    fn test_find_centroid_matches_unique() {
        // Two image centroids close to the same catalog centroid
        let image = vec![(100.0, 100.0), (100.5, 100.5)];
        let catalog = vec![(100.2, 100.2)];
        let matches = find_centroid_matches(&image, &catalog, 5.0);
        // Should only match one (the closer one)
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_combinations() {
        let combos = combinations(5, 3);
        assert_eq!(combos.len(), 10); // C(5,3) = 10
        // First should be [0,1,2], last [2,3,4]
        assert_eq!(combos[0], vec![0, 1, 2]);
        assert_eq!(combos[9], vec![2, 3, 4]);
    }

    #[test]
    fn test_combinations_4_from_8() {
        let combos = combinations(8, 4);
        assert_eq!(combos.len(), 70); // C(8,4) = 70
    }

    #[test]
    fn test_binom_cdf_certain() {
        // P(X <= n) with any p should be 1.0
        assert!((binom_cdf(10, 10, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_binom_cdf_fair_coin() {
        // P(X <= 0) for 1 trial with p=0.5 should be 0.5
        assert!((binom_cdf(0, 1, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hash_code_grid_simple() {
        let codes = hash_code_grid(&[1, 2], &[2, 3]);
        // (1,2), (1,3), (2,2), (2,3) -> sorted: (1,2), (1,3), (2,2), (2,3)
        // After sorting each ascending and dedup: same 4
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_sort_by_centroid_distance() {
        let vecs = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let sorted = sort_by_centroid_distance(&vecs);
        // All are equidistant from centroid, so order is stable
        assert_eq!(sorted.len(), 3);
    }
}
