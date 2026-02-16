//! Core geometric primitives: pinhole camera model, lens distortion, and rotation math.
//!
//! # Distortion Model
//!
//! The distortion functions use the tetra3 radial distortion model, which differs from the
//! naive `r' = r(1 + k*r²)` form commonly seen in tutorials. The tetra3 model is:
//!
//! ```text
//! r_undistorted = r_distorted * (1 - k' * r_distorted²) / (1 - k)
//! ```
//!
//! where `k' = k * (2 / width)²`. The single coefficient `k` represents the amount of
//! distortion at a distance of `width/2` from the image center (negative = barrel,
//! positive = pincushion). The `(1 - k)` denominator normalizes so that `k=0` is the
//! identity transform.
//!
//! Because this model is not analytically invertible, `distort_centroids` uses
//! Newton-Raphson iteration to recover distorted positions from undistorted ones.
//! The derivative used is `dr_u/dr_d = (1 - 3*k'*r_d²) / (1 - k)`.
//!
//! Centroids follow the Python convention: `(y, x)` pixel coordinates with `(0.5, 0.5)`
//! at the top-left pixel center, and `size` is `(height, width)`.

use nalgebra::{Matrix3, Vector3};

/// Convert pixel centroids to 3D unit vectors using the pinhole camera model.
///
/// Given star positions in `(y, x)` pixel coordinates and camera parameters,
/// produces unit vectors in the camera reference frame.
///
/// The camera frame follows the Python tetra3 convention:
/// - `v.x` (index 0) = boresight (optical axis)
/// - `v.y` (index 1) = horizontal (image x direction)
/// - `v.z` (index 2) = vertical (image y direction)
pub fn compute_vectors(
    centroids: &[(f64, f64)],
    image_width: u32,
    image_height: u32,
    fov: f64,
) -> Vec<Vector3<f64>> {
    let cx = image_width as f64 / 2.0;
    let cy = image_height as f64 / 2.0;
    let scale = (fov.to_radians() / 2.0).tan() / cx;

    centroids
        .iter()
        .map(|&(y, x)| {
            let v = Vector3::new(
                1.0,
                (cx - x) * scale,
                (cy - y) * scale,
            );
            v.normalize()
        })
        .collect()
}

/// Convert 3D unit vectors back to `(y, x)` pixel centroids (inverse of compute_vectors).
///
/// Vectors use the Python tetra3 convention: `v.x` = boresight, `v.y` = horizontal,
/// `v.z` = vertical.
pub fn compute_centroids(
    vectors: &[Vector3<f64>],
    image_width: u32,
    image_height: u32,
    fov: f64,
) -> Vec<(f64, f64)> {
    let cx = image_width as f64 / 2.0;
    let cy = image_height as f64 / 2.0;
    let scale_factor = -cx / (fov.to_radians() / 2.0).tan();

    vectors
        .iter()
        .map(|v| {
            let y = scale_factor * v.z / v.x + cy;
            let x = scale_factor * v.y / v.x + cx;
            (y, x)
        })
        .collect()
}

/// Apply radial undistortion to centroids.
///
/// Implements `r_u = r_d * (1 - k' * r_d^2) / (1 - k)` where `k' = k * (2/width)^2`.
/// Centroids are `(y, x)` pixel coordinates with `(0.5, 0.5)` at the top-left pixel center.
/// `k` is the distortion at `width/2` from center (negative = barrel, positive = pincushion).
/// `size` is `(height, width)` in pixels.
pub fn undistort_centroids(
    centroids: &[(f64, f64)],
    size: (u32, u32),
    k: f64,
) -> Vec<(f64, f64)> {
    let (height, width) = (size.0 as f64, size.1 as f64);
    let cy = height / 2.0;
    let cx = width / 2.0;

    centroids
        .iter()
        .map(|&(y, x)| {
            let dy = y - cy;
            let dx = x - cx;
            let r_norm = (dy * dy + dx * dx).sqrt() / width * 2.0;
            let scale = (1.0 - k * r_norm * r_norm) / (1.0 - k);
            (cy + dy * scale, cx + dx * scale)
        })
        .collect()
}

/// Apply radial distortion to centroids (inverse of undistort).
///
/// Uses Newton-Raphson iteration to invert `r_u = r_d * (1 - k' * r_d^2) / (1 - k)`.
/// Centroids are `(y, x)` pixel coordinates. `size` is `(height, width)`.
pub fn distort_centroids(
    centroids: &[(f64, f64)],
    size: (u32, u32),
    k: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<(f64, f64)> {
    let (height, width) = (size.0 as f64, size.1 as f64);
    let cy = height / 2.0;
    let cx = width / 2.0;

    centroids
        .iter()
        .map(|&(y, x)| {
            let dy = y - cy;
            let dx = x - cx;
            let r_undist = (dy * dy + dx * dx).sqrt() / width * 2.0;

            // Newton-Raphson: solve for r_dist given r_undist
            let mut r_dist = r_undist;
            for _ in 0..max_iter {
                let r_undist_est = r_dist * (1.0 - k * r_dist * r_dist) / (1.0 - k);
                let dru_drd = (1.0 - 3.0 * k * r_dist * r_dist) / (1.0 - k);
                let error = r_undist - r_undist_est;
                r_dist += error / dru_drd;

                if error.abs() < tol {
                    break;
                }
            }

            let ratio = if r_undist > 0.0 {
                r_dist / r_undist
            } else {
                1.0
            };
            (cy + dy * ratio, cx + dx * ratio)
        })
        .collect()
}

/// Find the rotation matrix that best maps one set of unit vectors to another.
///
/// Uses the Kabsch algorithm (SVD-based) to find the optimal rotation.
pub fn find_rotation_matrix(
    from: &[Vector3<f64>],
    to: &[Vector3<f64>],
) -> Option<Matrix3<f64>> {
    if from.len() != to.len() || from.len() < 2 {
        return None;
    }

    // Build the cross-covariance matrix H
    let mut h = Matrix3::zeros();
    for (a, b) in from.iter().zip(to.iter()) {
        h += a * b.transpose();
    }

    // SVD decomposition
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    // Python does U @ V directly (no determinant correction)
    Some(u * v_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vectors_centroids_roundtrip() {
        let width = 1024;
        let height = 768;
        let fov = 20.0;
        // Centroids are (y, x) format
        let centroids = vec![(384.0, 512.0), (300.0, 600.0), (500.0, 200.0)];

        let vectors = compute_vectors(&centroids, width, height, fov);
        let recovered = compute_centroids(&vectors, width, height, fov);

        for (orig, rec) in centroids.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig.0, rec.0, epsilon = 1e-10);
            assert_relative_eq!(orig.1, rec.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_center_pixel_maps_to_boresight() {
        let width = 1024;
        let height = 768;
        let fov = 20.0;
        // Center pixel in (y, x) format: (height/2, width/2)
        let centroids = vec![(384.0, 512.0)];

        let vectors = compute_vectors(&centroids, width, height, fov);
        // Boresight is along v.x (index 0) in tetra3 convention
        assert_relative_eq!(vectors[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(vectors[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vectors[0].z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_identity() {
        let vecs = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        let rot = find_rotation_matrix(&vecs, &vecs).unwrap();
        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn test_undistort_zero_distortion_is_identity() {
        let centroids = vec![(300.0, 600.0), (500.0, 200.0)];

        let result = undistort_centroids(&centroids, (768, 1024), 0.0);
        for (orig, res) in centroids.iter().zip(result.iter()) {
            assert_relative_eq!(orig.0, res.0, epsilon = 1e-10);
            assert_relative_eq!(orig.1, res.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_distort_undistort_roundtrip() {
        let size = (768, 1024);
        let k = -0.15;
        let centroids = vec![(300.0, 600.0), (500.0, 200.0)];

        let distorted = distort_centroids(&centroids, size, k, 1e-10, 50);
        let recovered = undistort_centroids(&distorted, size, k);
        for (orig, rec) in centroids.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig.0, rec.0, epsilon = 1e-6);
            assert_relative_eq!(orig.1, rec.1, epsilon = 1e-6);
        }
    }
}
