//! Image filtering and preprocessing utilities.
//!
//! General-purpose operations on row-major f32 grayscale images:
//! uniform (box/mean) filter, median, and binary morphological opening.

/// Uniform (box/mean) filter over a 2D image (separable).
///
/// Matches `scipy.ndimage.uniform_filter` behavior: the filter window is
/// always `size` wide, and edges are handled by clamping (nearest).
pub fn uniform_filter(data: &[f32], width: usize, height: usize, size: usize) -> Vec<f32> {
    let half = (size / 2) as isize;
    let sz = size as f32;

    // Horizontal pass
    let mut temp = vec![0.0f32; data.len()];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0f32;
            for dx in -half..=half {
                let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                sum += data[row + sx];
            }
            temp[row + x] = sum / sz;
        }
    }

    // Vertical pass
    let mut output = vec![0.0f32; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for dy in -half..=half {
                let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                sum += temp[sy * width + x];
            }
            output[y * width + x] = sum / sz;
        }
    }

    output
}

/// Median of a slice of f32 values.
pub fn median_value(data: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Binary opening with a 3x3 cross structuring element.
///
/// Opening = erosion followed by dilation. Removes isolated pixels
/// and thin protrusions while preserving larger shapes.
pub fn binary_opening(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    // Erosion: pixel is true only if it and all 4-connected neighbors are true
    let mut eroded = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !mask[idx] {
                continue;
            }
            let up = y == 0 || mask[(y - 1) * width + x];
            let down = y == height - 1 || mask[(y + 1) * width + x];
            let left = x == 0 || mask[y * width + x - 1];
            let right = x == width - 1 || mask[y * width + x + 1];
            eroded[idx] = up && down && left && right;
        }
    }

    // Dilation: pixel is true if it or any 4-connected neighbor is true in eroded
    let mut dilated = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if eroded[idx]
                || (y > 0 && eroded[(y - 1) * width + x])
                || (y < height - 1 && eroded[(y + 1) * width + x])
                || (x > 0 && eroded[y * width + x - 1])
                || (x < width - 1 && eroded[y * width + x + 1])
            {
                dilated[idx] = true;
            }
        }
    }

    dilated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_value() {
        assert_eq!(median_value(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median_value(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_uniform_filter_constant_image() {
        let pixels = vec![42.0f32; 100];
        let filtered = uniform_filter(&pixels, 10, 10, 5);
        for &v in &filtered {
            assert!(
                (v - 42.0).abs() < 1.0,
                "Uniform filter of constant should be constant, got {}",
                v
            );
        }
    }

    #[test]
    fn test_binary_opening_removes_single_pixel() {
        let mask = vec![
            false, false, false, false, false,
            false, false, false, false, false,
            false, false, true,  false, false,
            false, false, false, false, false,
            false, false, false, false, false,
        ];
        let result = binary_opening(&mask, 5, 5);
        assert!(result.iter().all(|&v| !v));
    }

    #[test]
    fn test_binary_opening_preserves_cross() {
        let mask = vec![
            false, false, true,  false, false,
            false, false, true,  false, false,
            true,  true,  true,  true,  true,
            false, false, true,  false, false,
            false, false, true,  false, false,
        ];
        let result = binary_opening(&mask, 5, 5);
        assert!(result[12]); // (2,2) center pixel
    }
}
