//! Camera calibration support for external intrinsics (e.g., from MrCal).
//!
//! Loads camera intrinsics and OpenCV5 distortion coefficients from a TOML file,
//! then pre-processes centroids so the standard solver (which assumes a centered
//! principal point and single FOV) produces correct results.
//!
//! # TOML format
//!
//! ```toml
//! [camera]
//! fx = 1761.0
//! fy = 1761.0
//! cx = 1965.0
//! cy = 1087.0
//! width = 4000
//! height = 2200
//!
//! [distortion]
//! model = "opencv5"
//! coefficients = [-0.02, 0.03, 0.0002, 0.0005, 0.0196]
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use tetra3::calibration::CameraCalibration;
//!
//! let cal = CameraCalibration::from_toml_file(Path::new("camera.toml")).unwrap();
//! let raw_centroids = vec![(500.0, 300.0), (600.0, 400.0)];
//! let prepared = cal.prepare_centroids(&raw_centroids);
//! // Use prepared.centroids with solver, setting fov_estimate = prepared.fov
//! ```

use std::io;
use std::path::Path;

/// Camera intrinsic parameters and distortion model from external calibration.
#[derive(Debug, Clone)]
pub struct CameraCalibration {
    /// Focal length in pixels (horizontal).
    pub fx: f64,
    /// Focal length in pixels (vertical).
    pub fy: f64,
    /// Principal point x (pixels from left edge).
    pub cx: f64,
    /// Principal point y (pixels from top edge).
    pub cy: f64,
    /// Expected image width in pixels.
    pub width: u32,
    /// Expected image height in pixels.
    pub height: u32,
    /// Distortion coefficients [k1, k2, p1, p2, k3] (OpenCV5 model).
    pub distortion_coefficients: [f64; 5],
}

/// Result of preparing centroids with calibration data.
pub struct PreparedCentroids {
    /// Undistorted centroids shifted so the principal point is at image center.
    /// Format: (y, x) pixel coordinates.
    pub centroids: Vec<(f64, f64)>,
    /// Exact horizontal FOV in degrees, computed as `2 * atan(width / (2 * fx))`.
    pub fov: f64,
    /// Recommended FOV error bound in degrees.
    pub fov_max_error: f64,
}

impl CameraCalibration {
    /// Load calibration from a TOML file.
    pub fn from_toml_file(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    /// Parse calibration from a TOML string.
    pub fn from_toml_str(content: &str) -> io::Result<Self> {
        let value: toml::Value = content
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("TOML parse error: {}", e)))?;

        let camera = value
            .get("camera")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing [camera] section"))?;

        let fx = get_f64(camera, "fx")?;
        let fy = get_f64(camera, "fy")?;
        let cx = get_f64(camera, "cx")?;
        let cy = get_f64(camera, "cy")?;
        let width = get_u32(camera, "width")?;
        let height = get_u32(camera, "height")?;

        let dist = value
            .get("distortion")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing [distortion] section"))?;

        let model = dist
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing distortion.model"))?;

        if model != "opencv5" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported distortion model '{}', expected 'opencv5'", model),
            ));
        }

        let coeffs_array = dist
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing distortion.coefficients array")
            })?;

        if coeffs_array.len() != 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected 5 distortion coefficients, got {}", coeffs_array.len()),
            ));
        }

        let mut distortion_coefficients = [0.0f64; 5];
        for (i, v) in coeffs_array.iter().enumerate() {
            distortion_coefficients[i] = v
                .as_float()
                .or_else(|| v.as_integer().map(|n| n as f64))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("coefficient {} is not a number", i),
                    )
                })?;
        }

        Ok(CameraCalibration {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            distortion_coefficients,
        })
    }

    /// Horizontal field of view in degrees, derived from focal length.
    pub fn fov_degrees(&self) -> f64 {
        2.0 * (self.width as f64 / (2.0 * self.fx)).atan().to_degrees()
    }

    /// Undistort and re-center centroids for the standard solver.
    ///
    /// Takes raw `(y, x)` pixel centroids from the image and returns corrected
    /// centroids in a coordinate space where the principal point is at image center
    /// and the pixel scale is uniform (`1/fx` rad/pixel in both axes).
    pub fn prepare_centroids(&self, centroids: &[(f64, f64)]) -> PreparedCentroids {
        let [k1, k2, p1, p2, k3] = self.distortion_coefficients;
        let w = self.width as f64;
        let h = self.height as f64;

        let corrected: Vec<(f64, f64)> = centroids
            .iter()
            .map(|&(py, px)| {
                // Normalize to camera coordinates using true intrinsics
                let x_d = (px - self.cx) / self.fx;
                let y_d = (py - self.cy) / self.fy;

                // Undistort in normalized coordinates
                let (x_u, y_u) = undistort_point_opencv5(x_d, y_d, k1, k2, p1, p2, k3);

                // Reproject using fx for BOTH axes so the solver's single-scale
                // model (derived from horizontal FOV) works correctly
                let px_corrected = x_u * self.fx + w / 2.0;
                let py_corrected = y_u * self.fx + h / 2.0;

                (py_corrected, px_corrected)
            })
            .collect();

        PreparedCentroids {
            centroids: corrected,
            fov: self.fov_degrees(),
            fov_max_error: 0.1,
        }
    }
}

/// Undistort a single point from distorted to ideal normalized coordinates
/// using the OpenCV5 model (k1, k2, p1, p2, k3) with fixed-point iteration.
fn undistort_point_opencv5(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
) -> (f64, f64) {
    let mut x_u = x_d;
    let mut y_u = y_d;

    for _ in 0..20 {
        let r2 = x_u * x_u + y_u * y_u;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let dx_tang = 2.0 * p1 * x_u * y_u + p2 * (r2 + 2.0 * x_u * x_u);
        let dy_tang = p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * p2 * x_u * y_u;

        x_u = (x_d - dx_tang) / radial;
        y_u = (y_d - dy_tang) / radial;
    }

    (x_u, y_u)
}

fn get_f64(table: &toml::Value, key: &str) -> io::Result<f64> {
    table
        .get(key)
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing or invalid '{}'", key)))
}

fn get_u32(table: &toml::Value, key: &str) -> io::Result<u32> {
    table
        .get(key)
        .and_then(|v| v.as_integer())
        .map(|i| i as u32)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing or invalid '{}'", key)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undistort_zero_distortion_is_identity() {
        let (x, y) = undistort_point_opencv5(0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((x - 0.1).abs() < 1e-12);
        assert!((y - (-0.2)).abs() < 1e-12);
    }

    #[test]
    fn test_undistort_center_is_identity() {
        let (x, y) = undistort_point_opencv5(0.0, 0.0, -0.1, 0.05, 0.001, 0.002, 0.01);
        assert!(x.abs() < 1e-12);
        assert!(y.abs() < 1e-12);
    }

    #[test]
    fn test_undistort_roundtrip() {
        let k1 = -0.02;
        let k2 = 0.03;
        let p1 = 0.0002;
        let p2 = 0.0005;
        let k3 = 0.0196;

        let x_u_orig = 0.15;
        let y_u_orig = -0.10;

        // Forward: apply distortion to get distorted coordinates
        let r2 = x_u_orig * x_u_orig + y_u_orig * y_u_orig;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let x_d = x_u_orig * radial + 2.0 * p1 * x_u_orig * y_u_orig + p2 * (r2 + 2.0 * x_u_orig * x_u_orig);
        let y_d = y_u_orig * radial + p1 * (r2 + 2.0 * y_u_orig * y_u_orig) + 2.0 * p2 * x_u_orig * y_u_orig;

        // Inverse: undistort should recover original
        let (x_u, y_u) = undistort_point_opencv5(x_d, y_d, k1, k2, p1, p2, k3);
        assert!((x_u - x_u_orig).abs() < 1e-10, "x: {} vs {}", x_u, x_u_orig);
        assert!((y_u - y_u_orig).abs() < 1e-10, "y: {} vs {}", y_u, y_u_orig);
    }

    #[test]
    fn test_fov_from_fx() {
        let cal = CameraCalibration {
            fx: 1761.0,
            fy: 1761.0,
            cx: 2000.0,
            cy: 1100.0,
            width: 4000,
            height: 2200,
            distortion_coefficients: [0.0; 5],
        };
        let expected = 2.0 * (4000.0_f64 / (2.0 * 1761.0)).atan().to_degrees();
        assert!((cal.fov_degrees() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_prepare_centroids_center_no_distortion() {
        // When PP is at image center and no distortion, center pixel stays at center
        let cal = CameraCalibration {
            fx: 1000.0,
            fy: 1000.0,
            cx: 500.0,
            cy: 384.0,
            width: 1000,
            height: 768,
            distortion_coefficients: [0.0; 5],
        };
        let centroids = vec![(384.0, 500.0)];
        let result = cal.prepare_centroids(&centroids);
        assert!((result.centroids[0].0 - 384.0).abs() < 1e-10);
        assert!((result.centroids[0].1 - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_prepare_centroids_pp_shift() {
        // Off-center PP: pixel at PP should map to image center
        let cal = CameraCalibration {
            fx: 1000.0,
            fy: 1000.0,
            cx: 520.0,
            cy: 400.0,
            width: 1000,
            height: 768,
            distortion_coefficients: [0.0; 5],
        };
        // A pixel at the principal point (y=400, x=520) is normalized (0, 0)
        // After reprojection to centered space: (h/2, w/2) = (384, 500)
        let centroids = vec![(400.0, 520.0)];
        let result = cal.prepare_centroids(&centroids);
        assert!((result.centroids[0].0 - 384.0).abs() < 1e-10);
        assert!((result.centroids[0].1 - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_toml() {
        let toml_str = r#"
[camera]
fx = 1761.0
fy = 1761.0
cx = 1965.0
cy = 1087.0
width = 4000
height = 2200

[distortion]
model = "opencv5"
coefficients = [-0.02, 0.03, 0.0002, 0.0005, 0.0196]
"#;
        let cal = CameraCalibration::from_toml_str(toml_str).unwrap();
        assert!((cal.fx - 1761.0).abs() < 1e-10);
        assert!((cal.cy - 1087.0).abs() < 1e-10);
        assert_eq!(cal.width, 4000);
        assert!((cal.distortion_coefficients[0] - (-0.02)).abs() < 1e-10);
        assert!((cal.distortion_coefficients[4] - 0.0196).abs() < 1e-10);
    }

    #[test]
    fn test_parse_toml_integers() {
        // TOML integers (no decimal point) should be accepted for float fields
        let toml_str = r#"
[camera]
fx = 1761
fy = 1761
cx = 1965
cy = 1087
width = 4000
height = 2200

[distortion]
model = "opencv5"
coefficients = [0, 0, 0, 0, 0]
"#;
        let cal = CameraCalibration::from_toml_str(toml_str).unwrap();
        assert!((cal.fx - 1761.0).abs() < 1e-10);
    }
}
