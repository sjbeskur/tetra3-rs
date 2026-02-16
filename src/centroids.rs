//! Star centroid extraction from images.
//!
//! Extracts star positions from grayscale images using:
//! 1. Background subtraction (local mean filter)
//! 2. Noise estimation and thresholding
//! 3. Binary morphological opening (3x3 cross)
//! 4. Connected component labeling via the [`lutz`] crate (single-pass algorithm)
//! 5. Intensity-weighted center-of-mass centroid computation
//! 6. Filtering by area, brightness, and shape
//!
//! Returns centroids as `(y, x)` pixel coordinates sorted brightest-first,
//! matching the Python `get_centroids_from_image` convention.

use crate::img_filter::{binary_opening, median_value, uniform_filter};

/// Parameters for centroid extraction.
#[derive(Debug, Clone)]
pub struct CentroidParams {
    /// Number of noise standard deviations to threshold at (default 2.0).
    pub sigma: f64,
    /// Fixed threshold (bypasses sigma-based thresholding if set).
    pub image_th: Option<f64>,
    /// Size of the mean filter for background subtraction (must be odd, default 25).
    pub filter_size: usize,
    /// Background subtraction mode.
    pub bg_sub_mode: BgSubMode,
    /// Noise estimation mode.
    pub sigma_mode: SigmaMode,
    /// Apply binary opening to clean the mask (default true).
    pub binary_open: bool,
    /// Maximum spot area in pixels (default 100).
    pub max_area: usize,
    /// Minimum spot area in pixels (default 5).
    pub min_area: usize,
    /// Maximum spot brightness sum (None = no limit).
    pub max_sum: Option<f64>,
    /// Minimum spot brightness sum (None = no limit).
    pub min_sum: Option<f64>,
    /// Maximum major/minor axis ratio (None = no limit).
    pub max_axis_ratio: Option<f64>,
    /// Maximum number of centroids to return (None = all).
    pub max_returned: Option<usize>,
}

impl Default for CentroidParams {
    fn default() -> Self {
        Self {
            sigma: 2.0,
            image_th: None,
            filter_size: 25,
            bg_sub_mode: BgSubMode::LocalMean,
            sigma_mode: SigmaMode::GlobalRootSquare,
            binary_open: true,
            max_area: 100,
            min_area: 5,
            max_sum: None,
            min_sum: None,
            max_axis_ratio: None,
            max_returned: None,
        }
    }
}

/// Background subtraction mode.
#[derive(Debug, Clone, Copy)]
pub enum BgSubMode {
    /// Subtract local mean (uniform filter of `filter_size`). Default.
    LocalMean,
    /// Subtract global mean of all pixels.
    GlobalMean,
    /// Subtract global median of all pixels.
    GlobalMedian,
    /// No background subtraction.
    None,
}

/// Noise standard deviation estimation mode.
#[derive(Debug, Clone, Copy)]
pub enum SigmaMode {
    /// sqrt(mean(pixel²)) over all pixels. Default.
    GlobalRootSquare,
    /// median(|pixel|) * 1.48 over all pixels.
    GlobalMedianAbs,
}

/// A detected spot with its statistics.
#[derive(Debug, Clone)]
pub struct SpotInfo {
    /// Centroid y position (subpixel, 0.5 = top-left pixel center).
    pub y: f64,
    /// Centroid x position (subpixel, 0.5 = top-left pixel center).
    pub x: f64,
    /// Sum of pixel intensities in the spot (zeroth moment).
    pub sum: f64,
    /// Spot area in pixels.
    pub area: usize,
}

/// Adapter to use a thresholded image with the `lutz` crate.
struct ThresholdedImage<'a> {
    mask: &'a [bool],
    width: u32,
    height: u32,
}

impl lutz::Image for ThresholdedImage<'_> {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn has_pixel(&self, x: u32, y: u32) -> bool {
        self.mask[(y * self.width + x) as usize]
    }
}

/// Extract star centroids from a grayscale image.
///
/// `pixels` is row-major f32 pixel data, `width`/`height` are image dimensions.
/// Returns `(y, x)` centroid positions sorted brightest-first.
pub fn get_centroids_from_image(
    pixels: &[f32],
    width: u32,
    height: u32,
    params: &CentroidParams,
) -> Vec<(f64, f64)> {
    let blobs = extract_blobs(pixels, width, height, params);
    blobs.iter().map(|s| (s.y, s.x)).collect()
}

/// Extract blobs with full statistics from a grayscale image.
///
/// `pixels` is row-major f32 pixel data. Returns blobs sorted brightest-first.
pub fn extract_blobs(
    pixels: &[f32],
    width: u32,
    height: u32,
    params: &CentroidParams,
) -> Vec<SpotInfo> {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels.len(), w * h);

    // 1. Background subtraction
    let image: Vec<f32> = match params.bg_sub_mode {
        BgSubMode::LocalMean => {
            let bg = uniform_filter(pixels, w, h, params.filter_size);
            pixels.iter().zip(bg.iter()).map(|(p, b)| p - b).collect()
        }
        BgSubMode::GlobalMean => {
            let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
            pixels.iter().map(|p| p - mean).collect()
        }
        BgSubMode::GlobalMedian => {
            let median = median_value(pixels);
            pixels.iter().map(|p| p - median).collect()
        }
        BgSubMode::None => pixels.to_vec(),
    };

    // 2. Compute threshold
    let threshold = if let Some(th) = params.image_th {
        th as f32
    } else {
        let noise_std = match params.sigma_mode {
            SigmaMode::GlobalRootSquare => {
                let mean_sq = image.iter().map(|&v| v * v).sum::<f32>() / image.len() as f32;
                mean_sq.sqrt()
            }
            SigmaMode::GlobalMedianAbs => {
                let abs_values: Vec<f32> = image.iter().map(|v| v.abs()).collect();
                median_value(&abs_values) * 1.48
            }
        };
        noise_std * params.sigma as f32
    };

    // 3. Create binary mask
    let mut mask: Vec<bool> = image.iter().map(|&v| v > threshold).collect();

    // 4. Binary opening with 3x3 cross structuring element
    if params.binary_open {
        mask = binary_opening(&mask, w, h);
    }

    // 5. Connected component labeling via lutz
    let lutz_image = ThresholdedImage {
        mask: &mask,
        width,
        height,
    };

    let mut blobs: Vec<SpotInfo> = Vec::new();

    for component in lutz::lutz::<_, Vec<lutz::Pixel>>(&lutz_image) {
        let area = component.len();

        // Area filter
        if area < params.min_area || area > params.max_area {
            continue;
        }

        // Compute intensity-weighted centroid (center-of-mass)
        let mut sum = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;

        for pixel in &component {
            let px = pixel.x as usize;
            let py = pixel.y as usize;
            let val = image[py * w + px].max(0.0) as f64;
            sum += val;
            sum_x += px as f64 * val;
            sum_y += py as f64 * val;
        }

        if sum <= 0.0 {
            continue;
        }

        // Sum filter
        if let Some(min_sum) = params.min_sum {
            if sum < min_sum {
                continue;
            }
        }
        if let Some(max_sum) = params.max_sum {
            if sum > max_sum {
                continue;
            }
        }

        let cx = sum_x / sum;
        let cy = sum_y / sum;

        // Axis ratio filter (second moments)
        if let Some(max_ratio) = params.max_axis_ratio {
            let mut m2_xx = 0.0f64;
            let mut m2_yy = 0.0f64;
            let mut m2_xy = 0.0f64;
            for pixel in &component {
                let val = image[pixel.y as usize * w + pixel.x as usize].max(0.0) as f64;
                let dx = pixel.x as f64 - cx;
                let dy = pixel.y as f64 - cy;
                m2_xx += dx * dx * val;
                m2_yy += dy * dy * val;
                m2_xy += dx * dy * val;
            }
            m2_xx /= sum;
            m2_yy /= sum;
            m2_xy /= sum;
            let disc = ((m2_xx - m2_yy).powi(2) + 4.0 * m2_xy.powi(2)).sqrt();
            let major = (2.0 * (m2_xx + m2_yy + disc)).sqrt();
            let minor = (2.0 * (m2_xx + m2_yy - disc).max(0.0)).sqrt();
            if minor <= 0.0 || major / minor > max_ratio {
                continue;
            }
        }

        // +0.5 offset: Python convention where (0.5, 0.5) is top-left pixel center
        blobs.push(SpotInfo {
            y: cy + 0.5,
            x: cx + 0.5,
            sum,
            area,
        });
    }

    // Sort by brightness (largest sum first)
    blobs.sort_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap());

    // Limit count
    if let Some(max) = params.max_returned {
        blobs.truncate(max);
    }

    blobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_blobs_bright_dot() {
        // Create a 20x20 image with a bright spot (3x3) near center
        let mut pixels = vec![0.0f32; 400];
        for dy in 0..3 {
            for dx in 0..3 {
                pixels[(8 + dy) * 20 + (8 + dx)] = 100.0;
            }
        }

        let params = CentroidParams {
            sigma: 1.0,
            filter_size: 5,
            min_area: 1,
            max_area: 20,
            binary_open: false, // Don't open, spot is small
            ..Default::default()
        };

        let blobs = extract_blobs(&pixels, 20, 20, &params);
        assert!(!blobs.is_empty(), "Should detect at least one spot");

        let spot = &blobs[0];
        // Centroid should be near (9.5, 9.5) - center of 3x3 block at (8,8)
        assert!((spot.y - 9.5).abs() < 1.0, "y={} should be near 9.5", spot.y);
        assert!((spot.x - 9.5).abs() < 1.0, "x={} should be near 9.5", spot.x);
        assert_eq!(spot.area, 9);
    }

    #[test]
    fn test_get_centroids_sorted_brightest() {
        // 50x50 image with two well-separated blobs
        let mut pixels = vec![0.0f32; 2500];
        // Dim spot at (10,10), 3x3
        for dy in 0..3 {
            for dx in 0..3 {
                pixels[(9 + dy) * 50 + (9 + dx)] = 50.0;
            }
        }
        // Bright spot at (35,35), 3x3
        for dy in 0..3 {
            for dx in 0..3 {
                pixels[(34 + dy) * 50 + (34 + dx)] = 500.0;
            }
        }

        let params = CentroidParams {
            sigma: 1.0,
            filter_size: 5,
            min_area: 1,
            max_area: 20,
            binary_open: false,
            bg_sub_mode: BgSubMode::None,
            ..Default::default()
        };

        let centroids = get_centroids_from_image(&pixels, 50, 50, &params);
        assert!(centroids.len() >= 2, "Should find at least 2 blobs, found {}", centroids.len());
        // First centroid should be the brighter one (near y=35.5)
        assert!(centroids[0].0 > 30.0, "Brightest spot y={} should be near 35.5", centroids[0].0);
    }

    #[test]
    fn test_area_filter() {
        // Spot with area 1 should be rejected with min_area=5
        let mut pixels = vec![0.0f32; 100];
        pixels[55] = 1000.0; // Single bright pixel

        let params = CentroidParams {
            sigma: 1.0,
            filter_size: 5,
            min_area: 5,
            binary_open: false,
            ..Default::default()
        };

        let blobs = extract_blobs(&pixels, 10, 10, &params);
        assert!(blobs.is_empty(), "Single pixel spot should be rejected by min_area=5");
    }

}
