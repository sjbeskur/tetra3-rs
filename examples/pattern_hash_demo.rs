//! Demonstrates the pattern hashing pipeline step by step.
//!
//! Shows how 4 star centroids become an edge-ratio hash key that can be
//! looked up in the pattern database. This is the core of the tetra3 algorithm.
//!
//! Usage:
//!   cargo run --example pattern_hash_demo

use nalgebra::{Matrix3, Vector3};

fn main() {
    // ── Step 0: Start with 4 star centroids (y, x) in a 1024x768 image ──
    //
    // These are pixel positions of the 4 brightest detected stars.
    let centroids: Vec<(f64, f64)> = vec![
        (200.0, 150.0),  // Star A (upper-left)
        (180.0, 600.0),  // Star B (upper-right)
        (550.0, 300.0),  // Star C (lower-left)
        (500.0, 700.0),  // Star D (lower-right)
    ];
    let image_width: u32 = 1024;
    let image_height: u32 = 768;
    let fov_deg: f64 = 15.0; // estimated field of view in degrees

    println!("=== Pattern Hashing Demo ===\n");
    println!("Image: {}x{}, FOV: {:.1} deg", image_width, image_height, fov_deg);
    println!("\n4 star centroids (y, x) in pixels:");
    for (i, (y, x)) in centroids.iter().enumerate() {
        println!("  Star {}: ({:.1}, {:.1})", (b'A' + i as u8) as char, y, x);
    }

    // ── Step 1: Convert pixel positions to unit vectors ──
    //
    // The pinhole camera model maps each pixel to a direction in 3D space.
    // The camera frame is: x = boresight, y = horizontal, z = vertical.
    let vectors = tetra3::geometry::compute_vectors(
        &centroids,
        image_width,
        image_height,
        fov_deg,
    );

    println!("\n── Step 1: Pixel -> Unit Vectors (pinhole camera model) ──");
    println!("  scale = tan(FOV/2) / (width/2) = tan({:.1}) / {} = {:.6} rad/px",
        fov_deg / 2.0,
        image_width / 2,
        (fov_deg.to_radians() / 2.0).tan() / (image_width as f64 / 2.0),
    );
    for (i, v) in vectors.iter().enumerate() {
        println!("  Star {}: [{:+.6}, {:+.6}, {:+.6}]",
            (b'A' + i as u8) as char, v.x, v.y, v.z);
    }

    // ── Step 2: Compute pairwise angular distances ──
    //
    // For 4 stars there are C(4,2) = 6 pairs.
    // Angular distance = 2 * arcsin(|v1 - v2| / 2)
    let labels = ['A', 'B', 'C', 'D'];
    let mut edges: Vec<(char, char, f64)> = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dist = (vectors[i] - vectors[j]).norm();
            let angle = 2.0 * (0.5 * dist).asin();
            edges.push((labels[i], labels[j], angle));
        }
    }
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    println!("\n── Step 2: Pairwise Angular Distances (6 edges, sorted) ──");
    for (i, (a, b, angle)) in edges.iter().enumerate() {
        let tag = if i == 5 { " <-- largest" } else { "" };
        println!("  {}-{}: {:8.4} deg ({:.6} rad){}",
            a, b, angle.to_degrees(), angle, tag);
    }

    // ── Step 3: Compute edge ratios ──
    //
    // Divide each edge by the largest. Drop the largest (always 1.0).
    // This gives 5 ratios in [0, 1] that are scale/rotation invariant.
    let largest = edges[5].2;
    let ratios: Vec<f64> = edges[..5].iter().map(|e| e.2 / largest).collect();

    println!("\n── Step 3: Edge Ratios (5 values, normalized by largest) ──");
    println!("  Largest edge ({}-{}): {:.4} deg", edges[5].0, edges[5].1, largest.to_degrees());
    for (i, (a, b, angle)) in edges[..5].iter().enumerate() {
        println!("  {}-{}: {:.4} / {:.4} = {:.6}",
            a, b, angle.to_degrees(), largest.to_degrees(), ratios[i]);
    }
    println!("\n  Edge ratio fingerprint: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]);

    // ── Step 4: Quantize into bins ──
    //
    // Each ratio is multiplied by bin_factor to get a bin index.
    // The database typically uses 50 bins.
    let bin_factor: u64 = 50;
    let pattern_max_error: f64 = 0.005;

    let bins: Vec<u16> = ratios.iter()
        .map(|&r| (r * bin_factor as f64) as u16)
        .collect();

    // With tolerance, each ratio maps to a range of bins
    let bin_mins: Vec<u16> = ratios.iter()
        .map(|&r| ((r - pattern_max_error) * bin_factor as f64).max(0.0) as u16)
        .collect();
    let bin_maxs: Vec<u16> = ratios.iter()
        .map(|&r| ((r + pattern_max_error) * bin_factor as f64).min(bin_factor as f64) as u16)
        .collect();

    println!("\n── Step 4: Quantize into {} bins (tolerance: +/-{}) ──", bin_factor, pattern_max_error);
    for (i, &r) in ratios.iter().enumerate() {
        println!("  ratio {:.4} -> bin {} (range {}-{})",
            r, bins[i], bin_mins[i], bin_maxs[i]);
    }
    println!("\n  Hash key (center): {:?}", bins);

    // ── Step 5: Hash the key ──
    //
    // The key is hashed using a polynomial + Knuth multiplicative hash:
    //   raw = sum(key[i] * bin_factor^i)
    //   index = (raw * 2654435761) % table_size
    let table_size: u64 = 8_652_092; // example: same as the fov40 database
    let hash_index = tetra3::hash_table::key_to_index(&bins, bin_factor, table_size);

    println!("\n── Step 5: Hash -> Table Index ──");
    println!("  raw = {}*50^0 + {}*50^1 + {}*50^2 + {}*50^3 + {}*50^4",
        bins[0], bins[1], bins[2], bins[3], bins[4]);
    let raw: u64 = bins.iter().enumerate()
        .map(|(i, &b)| (b as u64) * bin_factor.pow(i as u32))
        .sum();
    println!("      = {}", raw);
    println!("  index = ({} * 2654435761) % {} = {}", raw, table_size, hash_index);

    // ── Step 6: Show the search grid ──
    //
    // Because of tolerance, we search multiple hash keys (Cartesian product of bin ranges).
    let num_keys: usize = bin_mins.iter().zip(bin_maxs.iter())
        .map(|(&mn, &mx)| (mx - mn + 1) as usize)
        .product();

    println!("\n── Step 6: Search Grid ──");
    println!("  With +/-{} tolerance, each ratio spans 1-2 bins.", pattern_max_error);
    println!("  Total keys to look up: {} (Cartesian product of bin ranges)", num_keys);

    // ── Summary ──
    println!("\n=== Summary ===");
    println!("  4 star centroids");
    println!("  -> 6 pairwise angles");
    println!("  -> 5 edge ratios (rotation/scale invariant fingerprint)");
    println!("  -> quantized into {} bins each", bin_factor);
    println!("  -> hashed to table index {}", hash_index);
    println!("  -> look up {} nearby keys to find candidate catalog patterns", num_keys);
    println!("\n  If a catalog pattern has the same edge ratios, it's the same");
    println!("  geometric shape -- meaning those 4 image stars match those 4");
    println!("  catalog stars regardless of camera orientation.");

    // ── Bonus: show invariance ──
    println!("\n=== Bonus: Rotation Invariance ===");
    println!("  Rotating all vectors by an arbitrary rotation...");

    // Apply a random rotation (30 deg about z-axis)
    let angle = 30.0_f64.to_radians();
    let rot = Matrix3::new(
        angle.cos(), -angle.sin(), 0.0,
        angle.sin(),  angle.cos(), 0.0,
        0.0,          0.0,         1.0,
    );
    let rotated_vectors: Vec<Vector3<f64>> = vectors.iter()
        .map(|v| (rot * v).normalize())
        .collect();

    // Recompute edge ratios from rotated vectors
    let mut rotated_edges: Vec<f64> = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dist = (rotated_vectors[i] - rotated_vectors[j]).norm();
            rotated_edges.push(2.0 * (0.5 * dist).asin());
        }
    }
    rotated_edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rotated_largest = *rotated_edges.last().unwrap();
    let rotated_ratios: Vec<f64> = rotated_edges[..5].iter()
        .map(|&a| a / rotated_largest).collect();

    let rotated_bins: Vec<u16> = rotated_ratios.iter()
        .map(|&r| (r * bin_factor as f64) as u16)
        .collect();
    let rotated_hash = tetra3::hash_table::key_to_index(&rotated_bins, bin_factor, table_size);

    println!("  Original ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]);
    println!("  Rotated  ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        rotated_ratios[0], rotated_ratios[1], rotated_ratios[2], rotated_ratios[3], rotated_ratios[4]);
    println!("  Original hash: {}", hash_index);
    println!("  Rotated  hash: {}", rotated_hash);
    println!("  Match: {}", if hash_index == rotated_hash { "YES -- same pattern!" } else { "no (rounding)" });
}
