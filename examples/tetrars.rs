//! tetra3 command-line plate solver.
//!
//! Usage:
//!   cargo run --example tetrars -- <database_dir> <image> [options]
//!
//! Options:
//!   --fov <degrees>          Estimated field of view
//!   --fov-error <degrees>    Max FOV error from estimate (default: 5.0 if fov given)
//!   --distortion <k>         Known distortion coefficient
//!   --timeout <ms>           Solve timeout in milliseconds
//!   --stars <n>              Pattern checking stars (default: 8)
//!   --cal <path>             TOML camera calibration file
//!   --match-threshold <p>    Match acceptance threshold (default: 1e-3)
//!
//! Example:
//!   cargo run --example tetrars -- data/tetra_db/ stars.png --fov 12.0

use std::path::Path;
use std::time::Instant;

use tetra3::centroids::{self, CentroidParams};
use tetra3::database::TetraDatabase;
use tetra3::solver::{self, SolveParams};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <database_dir> <image> [options]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --fov <degrees>          Estimated field of view");
        eprintln!("  --fov-error <degrees>    Max FOV error (default: 5.0 if fov given)");
        eprintln!("  --distortion <k>         Known distortion coefficient");
        eprintln!("  --timeout <ms>           Solve timeout in milliseconds");
        eprintln!("  --stars <n>              Pattern checking stars (default: 8)");
        eprintln!("  --cal <path>             TOML camera calibration file");
        eprintln!("  --match-threshold <p>    Match acceptance threshold (default: 1e-3)");
        std::process::exit(1);
    }

    let db_path = &args[1];
    let img_path = &args[2];

    // Parse optional arguments
    let mut solve_params = SolveParams::default();
    let mut cal_path: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--fov" => {
                i += 1;
                solve_params.fov_estimate = Some(args[i].parse().expect("invalid --fov value"));
                if solve_params.fov_max_error.is_none() {
                    solve_params.fov_max_error = Some(5.0);
                }
            }
            "--fov-error" => {
                i += 1;
                solve_params.fov_max_error = Some(args[i].parse().expect("invalid --fov-error value"));
            }
            "--distortion" => {
                i += 1;
                solve_params.distortion = Some(args[i].parse().expect("invalid --distortion value"));
            }
            "--timeout" => {
                i += 1;
                solve_params.solve_timeout = Some(args[i].parse().expect("invalid --timeout value"));
            }
            "--stars" => {
                i += 1;
                solve_params.pattern_checking_stars = args[i].parse().expect("invalid --stars value");
            }
            "--cal" => {
                i += 1;
                cal_path = Some(args[i].clone());
            }
            "--match-threshold" => {
                i += 1;
                solve_params.match_threshold = args[i].parse().expect("invalid --match-threshold value");
            }
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load database
    eprintln!("Loading database: {}", db_path);
    let t0 = Instant::now();
    let db = TetraDatabase::load(Path::new(db_path)).unwrap_or_else(|e| {
        eprintln!("Failed to load database: {}", e);
        std::process::exit(1);
    });
    eprintln!(
        "  {} stars, {} pattern rows, loaded in {:.1}ms",
        db.num_stars(),
        db.pattern_catalog().num_rows(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Load image
    eprintln!("Loading image: {}", img_path);
    let img = image::open(img_path).unwrap_or_else(|e| {
        eprintln!("Failed to load image: {}", e);
        std::process::exit(1);
    });
    let gray = img.to_luma8();
    let width = gray.width();
    let height = gray.height();
    let pixels: Vec<f32> = gray.as_raw().iter().map(|&v| v as f32).collect();
    eprintln!("  {}x{} pixels", width, height);

    // Extract centroids
    eprintln!("Extracting centroids...");
    let centroid_params = CentroidParams::default();
    let star_centroids = centroids::get_centroids_from_image(&pixels, width, height, &centroid_params);
    eprintln!("  Found {} stars", star_centroids.len());

    if star_centroids.len() < 4 {
        eprintln!("Error: need at least 4 stars to solve, found {}", star_centroids.len());
        std::process::exit(1);
    }

    // Apply calibration if provided
    let (final_centroids, final_solve_params) = if let Some(ref cal_file) = cal_path {
        use tetra3::calibration::CameraCalibration;
        eprintln!("Loading calibration: {}", cal_file);
        let cal = CameraCalibration::from_toml_file(Path::new(cal_file)).unwrap_or_else(|e| {
            eprintln!("Failed to load calibration: {}", e);
            std::process::exit(1);
        });

        if width != cal.width || height != cal.height {
            eprintln!(
                "Warning: image {}x{} does not match calibration {}x{}",
                width, height, cal.width, cal.height
            );
        }

        let prepared = cal.prepare_centroids(&star_centroids);
        eprintln!("  FOV from calibration: {:.4} deg", prepared.fov);

        let mut params = solve_params.clone();
        params.fov_estimate = Some(prepared.fov);
        params.fov_max_error = Some(prepared.fov_max_error);
        params.distortion = None;
        (prepared.centroids, params)
    } else {
        (star_centroids, solve_params)
    };

    // Solve
    eprintln!("Solving...");
    let size = (height, width);

    use tetra3::database::StarDatabase;
    let result = solver::solve_from_centroids(&db, &final_centroids, size, &final_solve_params);

    match result {
        Some(sol) => {
            println!("SOLVE OK");
            println!("  RA:       {:.4} deg", sol.ra);
            println!("  Dec:      {:.4} deg", sol.dec);
            println!("  Roll:     {:.4} deg", sol.roll);
            println!("  FOV:      {:.4} deg", sol.fov);
            println!("  Matches:  {}", sol.num_matches);
            println!("  RMSE:     {:.2} arcsec", sol.rmse);
            println!("  Prob:     {:.2e}", sol.prob);
            println!("  Time:     {:.1} ms", sol.t_solve_ms);
            if let Some(k) = sol.distortion {
                println!("  Distort:  {:.6}", k);
            }
        }
        None => {
            println!("SOLVE FAILED - no match found");
            std::process::exit(2);
        }
    }
}
