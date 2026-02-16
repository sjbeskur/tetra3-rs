//! Database types and traits for star pattern databases.
//!
//! The solver depends on a loaded database through the [`StarDatabase`] trait.
//! [`TetraDatabase`] is the concrete implementation that owns the data and
//! supports saving/loading via a directory-based format (CSV + raw binary).

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;

use crate::hash_table::PatternCatalog;

/// A single star entry from the star table.
///
/// Columns in the Python star_table are:
///   [0] ra (radians), [1] dec (radians),
///   [2] x = cos(ra)*cos(dec), [3] y = sin(ra)*cos(dec), [4] z = sin(dec),
///   [5] magnitude
#[derive(Debug, Clone, Copy)]
pub struct StarEntry {
    pub ra: f64,
    pub dec: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub magnitude: f64,
}

/// Database properties matching the Python `_db_props` dictionary.
#[derive(Debug, Clone)]
pub struct DatabaseProperties {
    pub pattern_mode: String,
    pub pattern_size: usize,
    pub pattern_bins: u64,
    pub pattern_max_error: f64,
    pub max_fov: f64,
    pub min_fov: f64,
    pub verification_stars_per_fov: usize,
    pub pattern_stars_per_fov: usize,
    pub star_max_magnitude: f64,
    pub star_catalog: String,
    pub epoch_equinox: Option<f64>,
    pub epoch_proper_motion: Option<f64>,
    pub presort_patterns: bool,
    pub simplify_pattern: bool,
    pub range_ra: Option<(f64, f64)>,
    pub range_dec: Option<(f64, f64)>,
}

/// Trait that the solver requires from a loaded database.
pub trait StarDatabase {
    /// Database configuration properties.
    fn properties(&self) -> &DatabaseProperties;

    /// The pattern catalog (hash table of star ID patterns).
    fn pattern_catalog(&self) -> &PatternCatalog;

    /// Largest edge angle for each pattern in milliradians, if available.
    /// Indexed by the same row as the pattern catalog.
    fn pattern_largest_edge(&self) -> Option<&[f64]>;

    /// Get a star entry by index.
    fn star(&self, index: usize) -> &StarEntry;

    /// Get the unit vector (x, y, z) for a star by index.
    fn star_vector(&self, index: usize) -> [f64; 3] {
        let s = self.star(index);
        [s.x, s.y, s.z]
    }

    /// Number of stars in the star table.
    fn num_stars(&self) -> usize;

    /// Find star indices within `radius` radians of the given unit vector.
    ///
    /// Default implementation does a brute-force scan. Override with a
    /// KDTree-based implementation for better performance.
    fn get_nearby_stars(&self, vector: &[f64; 3], radius: f64) -> Vec<usize> {
        let max_dist = 2.0 * (radius / 2.0).sin();
        let cos_radius = radius.cos();

        let mut result = Vec::new();
        for i in 0..self.num_stars() {
            let s = self.star(i);
            // Bounding box pre-filter
            if (s.x - vector[0]).abs() > max_dist
                || (s.y - vector[1]).abs() > max_dist
                || (s.z - vector[2]).abs() > max_dist
            {
                continue;
            }
            // Dot product check (angle < radius)
            let dot = vector[0] * s.x + vector[1] * s.y + vector[2] * s.z;
            if dot > cos_radius {
                result.push(i);
            }
        }
        result
    }
}

/// Concrete star database backed by owned data.
///
/// Saved as a directory containing CSV metadata and raw binary pattern data.
/// The KD-tree spatial index is rebuilt automatically on construction and after loading.
pub struct TetraDatabase {
    properties: DatabaseProperties,
    star_table: Vec<StarEntry>,
    pattern_catalog: PatternCatalog,
    pattern_largest_edge: Option<Vec<f64>>,
    /// KD-tree for fast spatial queries — rebuilt on load.
    kdtree: Option<ImmutableKdTree<f64, 3>>,
}

impl TetraDatabase {
    /// Create a database from its constituent parts.
    ///
    /// Builds the KD-tree spatial index automatically.
    pub fn new(
        properties: DatabaseProperties,
        star_table: Vec<StarEntry>,
        pattern_catalog: PatternCatalog,
        pattern_largest_edge: Option<Vec<f64>>,
    ) -> Self {
        let kdtree = Self::build_kdtree(&star_table);
        Self {
            properties,
            star_table,
            pattern_catalog,
            pattern_largest_edge,
            kdtree: Some(kdtree),
        }
    }

    /// Save the database to a directory.
    ///
    /// Creates the directory if it does not exist, then writes:
    /// - `properties.csv` (single-row CSV with headers)
    /// - `stars.csv` (CSV with headers: ra,dec,x,y,z,magnitude)
    /// - `patterns.bin` (raw little-endian binary)
    /// - `largest_edges.bin` (raw little-endian binary, only if present)
    pub fn save(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;

        self.write_properties_csv(&dir.join("properties.csv"))?;
        self.write_stars_csv(&dir.join("stars.csv"))?;

        {
            let file = File::create(dir.join("patterns.bin"))?;
            let mut writer = BufWriter::new(file);
            self.pattern_catalog.write_binary(&mut writer)?;
        }

        if let Some(ref edges) = self.pattern_largest_edge {
            let file = File::create(dir.join("largest_edges.bin"))?;
            let mut writer = BufWriter::new(file);
            writer.write_all(&(edges.len() as u64).to_le_bytes())?;
            for &val in edges {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load a database from a directory.
    ///
    /// Expects `properties.csv`, `stars.csv`, `patterns.bin`, and
    /// optionally `largest_edges.bin`. Rebuilds the KD-tree after loading.
    pub fn load(dir: &Path) -> io::Result<Self> {
        let properties = Self::read_properties_csv(&dir.join("properties.csv"))?;
        let star_table = Self::read_stars_csv(&dir.join("stars.csv"))?;

        let pattern_catalog = {
            let file = File::open(dir.join("patterns.bin"))?;
            let mut reader = BufReader::new(file);
            PatternCatalog::read_binary(&mut reader)?
        };

        let largest_edges_path = dir.join("largest_edges.bin");
        let pattern_largest_edge = if largest_edges_path.exists() {
            let file = File::open(&largest_edges_path)?;
            let mut reader = BufReader::new(file);
            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            let count = u64::from_le_bytes(buf8) as usize;
            let byte_count = count * 8;
            let mut bytes = vec![0u8; byte_count];
            reader.read_exact(&mut bytes)?;
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Some(data)
        } else {
            None
        };

        let kdtree = Self::build_kdtree(&star_table);
        Ok(Self {
            properties,
            star_table,
            pattern_catalog,
            pattern_largest_edge,
            kdtree: Some(kdtree),
        })
    }

    /// Build an immutable KD-tree from star unit vectors.
    fn build_kdtree(star_table: &[StarEntry]) -> ImmutableKdTree<f64, 3> {
        let points: Vec<[f64; 3]> = star_table
            .iter()
            .map(|s| [s.x, s.y, s.z])
            .collect();
        ImmutableKdTree::new_from_slice(&points)
    }

    fn write_properties_csv(&self, path: &Path) -> io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let p = &self.properties;
        wtr.write_record([
            "pattern_mode", "pattern_size", "pattern_bins", "pattern_max_error",
            "max_fov", "min_fov", "verification_stars_per_fov", "pattern_stars_per_fov",
            "star_max_magnitude", "star_catalog", "epoch_equinox", "epoch_proper_motion",
            "presort_patterns", "simplify_pattern",
            "range_ra_min", "range_ra_max", "range_dec_min", "range_dec_max",
        ]).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let opt_f64 = |v: Option<f64>| v.map_or(String::new(), |x| x.to_string());
        let (ra_min, ra_max) = match p.range_ra {
            Some((min, max)) => (min.to_string(), max.to_string()),
            None => (String::new(), String::new()),
        };
        let (dec_min, dec_max) = match p.range_dec {
            Some((min, max)) => (min.to_string(), max.to_string()),
            None => (String::new(), String::new()),
        };

        wtr.write_record([
            &p.pattern_mode,
            &p.pattern_size.to_string(),
            &p.pattern_bins.to_string(),
            &p.pattern_max_error.to_string(),
            &p.max_fov.to_string(),
            &p.min_fov.to_string(),
            &p.verification_stars_per_fov.to_string(),
            &p.pattern_stars_per_fov.to_string(),
            &p.star_max_magnitude.to_string(),
            &p.star_catalog,
            &opt_f64(p.epoch_equinox),
            &opt_f64(p.epoch_proper_motion),
            &p.presort_patterns.to_string(),
            &p.simplify_pattern.to_string(),
            &ra_min, &ra_max, &dec_min, &dec_max,
        ]).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        wtr.flush().map_err(Into::into)
    }

    fn read_properties_csv(path: &Path) -> io::Result<DatabaseProperties> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let headers = rdr.headers()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            .clone();

        let record = rdr.records()
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty properties CSV"))?
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let get = |name: &str| -> io::Result<&str> {
            let idx = headers.iter().position(|h| h == name)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                    format!("missing CSV column: {}", name)))?;
            Ok(record.get(idx).unwrap_or(""))
        };

        let parse_f64 = |name: &str| -> io::Result<f64> {
            get(name)?.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                    format!("{}: {}", name, e)))
        };
        let parse_usize = |name: &str| -> io::Result<usize> {
            get(name)?.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                    format!("{}: {}", name, e)))
        };
        let parse_u64 = |name: &str| -> io::Result<u64> {
            get(name)?.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                    format!("{}: {}", name, e)))
        };
        let parse_bool = |name: &str| -> io::Result<bool> {
            get(name)?.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                    format!("{}: {}", name, e)))
        };
        let parse_opt_f64 = |name: &str| -> io::Result<Option<f64>> {
            let s = get(name)?;
            if s.is_empty() { Ok(None) } else {
                s.parse().map(Some)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                        format!("{}: {}", name, e)))
            }
        };

        let range_ra = match (parse_opt_f64("range_ra_min")?, parse_opt_f64("range_ra_max")?) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        };
        let range_dec = match (parse_opt_f64("range_dec_min")?, parse_opt_f64("range_dec_max")?) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        };

        Ok(DatabaseProperties {
            pattern_mode: get("pattern_mode")?.to_string(),
            pattern_size: parse_usize("pattern_size")?,
            pattern_bins: parse_u64("pattern_bins")?,
            pattern_max_error: parse_f64("pattern_max_error")?,
            max_fov: parse_f64("max_fov")?,
            min_fov: parse_f64("min_fov")?,
            verification_stars_per_fov: parse_usize("verification_stars_per_fov")?,
            pattern_stars_per_fov: parse_usize("pattern_stars_per_fov")?,
            star_max_magnitude: parse_f64("star_max_magnitude")?,
            star_catalog: get("star_catalog")?.to_string(),
            epoch_equinox: parse_opt_f64("epoch_equinox")?,
            epoch_proper_motion: parse_opt_f64("epoch_proper_motion")?,
            presort_patterns: parse_bool("presort_patterns")?,
            simplify_pattern: parse_bool("simplify_pattern")?,
            range_ra,
            range_dec,
        })
    }

    fn write_stars_csv(&self, path: &Path) -> io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        wtr.write_record(["ra", "dec", "x", "y", "z", "magnitude"])
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        for star in &self.star_table {
            wtr.write_record([
                star.ra.to_string(),
                star.dec.to_string(),
                star.x.to_string(),
                star.y.to_string(),
                star.z.to_string(),
                star.magnitude.to_string(),
            ]).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }

        wtr.flush().map_err(Into::into)
    }

    fn read_stars_csv(path: &Path) -> io::Result<Vec<StarEntry>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mut stars = Vec::new();
        for result in rdr.records() {
            let record = result.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let parse = |idx: usize| -> io::Result<f64> {
                record.get(idx)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing column"))?
                    .parse()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            };
            stars.push(StarEntry {
                ra: parse(0)?,
                dec: parse(1)?,
                x: parse(2)?,
                y: parse(3)?,
                z: parse(4)?,
                magnitude: parse(5)?,
            });
        }
        Ok(stars)
    }
}

impl StarDatabase for TetraDatabase {
    fn properties(&self) -> &DatabaseProperties {
        &self.properties
    }

    fn pattern_catalog(&self) -> &PatternCatalog {
        &self.pattern_catalog
    }

    fn pattern_largest_edge(&self) -> Option<&[f64]> {
        self.pattern_largest_edge.as_deref()
    }

    fn star(&self, index: usize) -> &StarEntry {
        &self.star_table[index]
    }

    fn num_stars(&self) -> usize {
        self.star_table.len()
    }

    /// Find star indices within `radius` radians using the KD-tree.
    ///
    /// Converts the angular radius to squared Euclidean chord distance
    /// on the unit sphere: d² = 2 - 2·cos(θ).
    fn get_nearby_stars(&self, vector: &[f64; 3], radius: f64) -> Vec<usize> {
        if let Some(tree) = &self.kdtree {
            let sq_dist = 2.0 - 2.0 * radius.cos();
            tree.within::<SquaredEuclidean>(vector, sq_dist)
                .iter()
                .map(|nn| nn.item as usize)
                .collect()
        } else {
            // Fallback to brute-force if tree not built
            let cos_radius = radius.cos();
            let mut result = Vec::new();
            for i in 0..self.num_stars() {
                let s = &self.star_table[i];
                let dot = vector[0] * s.x + vector[1] * s.y + vector[2] * s.z;
                if dot > cos_radius {
                    result.push(i);
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let props = DatabaseProperties {
            pattern_mode: "edge_ratio".to_string(),
            pattern_size: 4,
            pattern_bins: 50,
            pattern_max_error: 0.005,
            max_fov: 20.0,
            min_fov: 5.0,
            verification_stars_per_fov: 30,
            pattern_stars_per_fov: 10,
            star_max_magnitude: 7.0,
            star_catalog: "hip_main".to_string(),
            epoch_equinox: Some(2000.0),
            epoch_proper_motion: Some(2024.0),
            presort_patterns: true,
            simplify_pattern: false,
            range_ra: None,
            range_dec: None,
        };

        let stars = vec![
            StarEntry {
                ra: 0.1, dec: 0.2,
                x: 0.1_f64.cos() * 0.2_f64.cos(),
                y: 0.1_f64.sin() * 0.2_f64.cos(),
                z: 0.2_f64.sin(),
                magnitude: 3.5,
            },
            StarEntry {
                ra: 1.0, dec: -0.5,
                x: 1.0_f64.cos() * (-0.5_f64).cos(),
                y: 1.0_f64.sin() * (-0.5_f64).cos(),
                z: (-0.5_f64).sin(),
                magnitude: 5.2,
            },
        ];

        let mut catalog = PatternCatalog::new(10, 4);
        catalog.set_row(3, &[0, 1, 0, 1]);

        let largest_edge = vec![0.0; 10];

        let db = TetraDatabase::new(
            props,
            stars,
            catalog,
            Some(largest_edge),
        );

        // Save to temp directory
        let tmp = std::env::temp_dir().join("tetra3_test_db");
        db.save(&tmp).expect("save failed");

        // Load
        let loaded = TetraDatabase::load(&tmp).expect("load failed");

        // Verify properties
        assert_eq!(loaded.properties().pattern_size, 4);
        assert_eq!(loaded.properties().pattern_bins, 50);
        assert_eq!(loaded.properties().star_catalog, "hip_main");
        assert_eq!(loaded.properties().presort_patterns, true);

        // Verify stars
        assert_eq!(loaded.num_stars(), 2);
        assert!((loaded.star(0).ra - 0.1).abs() < 1e-10);
        assert!((loaded.star(1).magnitude - 5.2).abs() < 1e-10);

        // Verify pattern catalog
        assert_eq!(loaded.pattern_catalog().get_row(3), &[0, 1, 0, 1]);
        assert!(loaded.pattern_catalog().is_row_empty(0));

        // Verify largest edge
        assert!(loaded.pattern_largest_edge().is_some());
        assert_eq!(loaded.pattern_largest_edge().unwrap().len(), 10);

        // Cleanup
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_kdtree_nearby_stars() {
        // Place stars at known positions on the unit sphere
        let stars: Vec<StarEntry> = [
            (0.0, 0.0),     // x-axis: (1, 0, 0)
            (0.0, 0.1),     // slightly above x-axis
            (0.0, 0.2),     // further above
            (1.0, 0.0),     // ~57 degrees away
            (3.14, 0.0),    // opposite side of sphere
        ]
        .iter()
        .map(|&(ra, dec)| StarEntry {
            ra,
            dec,
            x: ra.cos() * dec.cos(),
            y: ra.sin() * dec.cos(),
            z: dec.sin(),
            magnitude: 5.0,
        })
        .collect();

        let props = DatabaseProperties {
            pattern_mode: "edge_ratio".to_string(),
            pattern_size: 4,
            pattern_bins: 50,
            pattern_max_error: 0.005,
            max_fov: 20.0,
            min_fov: 5.0,
            verification_stars_per_fov: 30,
            pattern_stars_per_fov: 10,
            star_max_magnitude: 7.0,
            star_catalog: "test".to_string(),
            epoch_equinox: None,
            epoch_proper_motion: None,
            presort_patterns: false,
            simplify_pattern: false,
            range_ra: None,
            range_dec: None,
        };

        let catalog = PatternCatalog::new(10, 4);
        let db = TetraDatabase::new(props, stars, catalog, None);

        // Query near the x-axis with 0.25 radian radius
        // Should find stars 0 (at 0.0) and 1 (at 0.1 dec offset)
        // Star 2 (at 0.2 dec) is borderline, star 3 is too far
        let query = [1.0, 0.0, 0.0];
        let mut nearby = db.get_nearby_stars(&query, 0.15);
        nearby.sort();
        assert!(nearby.contains(&0), "Should find star at origin");
        assert!(nearby.contains(&1), "Should find star at 0.1 dec offset");
        assert!(!nearby.contains(&3), "Should not find star at 1.0 ra");
        assert!(!nearby.contains(&4), "Should not find star on opposite side");
    }
}
