# tetra3 (Rust)

A Rust port of [tetra3](https://github.com/esa/tetra3), ESA's fast lost-in-space plate solver for star trackers. Given a star field image, it identifies stars and determines the camera's pointing direction (RA, Dec, roll) without prior pointing information.

Based on the Tetra algorithm by Brown, Stubis, and Cahoy (AIAA/USU SmallSat 2017).

## Quick start

```bash
# Extract the star database
cd data/
tar xzf tetra_db.tar.gz
cd ..

# Solve an image
cargo run --release --example tetrars -- data/tetra_db/ data/2019-07-29T204726_Alt60_Azi-135_Try1.tiff

# With known FOV
cargo run --release --example tetrars -- data/tetra_db/ image.tiff --fov 12.0

# With camera calibration file
cargo run --release --example tetrars -- data/tetra_db/ image.tiff --cal camera.toml
```

## Differences from the Python version

### Centroid extraction: Lutz algorithm
The Python version uses `scipy.ndimage.label` for connected component labeling during centroid extraction. The Rust port uses the [lutz](https://crates.io/crates/lutz) crate, which implements a single-pass connected component labeling algorithm. This avoids pulling in a full image-processing framework and runs efficiently on embedded targets.

### Database format: CSV + raw binary
The Python version stores databases as `.npz` files (NumPy compressed archives). The Rust port uses a directory-based format:

- `properties.csv` -- human-readable database metadata (FOV range, pattern parameters, star catalog info)
- `stars.csv` -- star catalog subset (RA, Dec, unit vector, magnitude)
- `patterns.bin` -- raw little-endian binary for the pattern hash table (~189MB for the default database)
- `largest_edges.bin` -- optional raw binary for presorted pattern data

Use `tools/convert_npz.py` to convert a Python `.npz` database to this format.

### Camera calibration: OpenCV5 distortion model
The Python version supports a single radial distortion coefficient `k`. The Rust port adds support for external camera calibration via TOML files, including:

- Independent focal lengths `fx`, `fy` (non-square pixels)
- Off-center principal point `cx`, `cy`
- OpenCV5 distortion model with 5 coefficients: `k1, k2, p1, p2, k3` (radial + tangential)

Calibration is applied as a pre-processing step on centroids before the solver runs, so the solver core is unchanged. This is compatible with calibration tools like [MrCal](https://mrcal.secretsauce.net/).

Example calibration file (`camera.toml`):

```toml
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
```

### No Python/NumPy/SciPy dependency
The Rust port is a standalone binary with no runtime dependencies on Python. The KD-tree uses [kiddo](https://crates.io/crates/kiddo) instead of `scipy.spatial.KDTree`, and linear algebra uses [nalgebra](https://crates.io/crates/nalgebra).

## Architecture

| Module | Purpose |
|---|---|
| `centroids` | Star centroid extraction from grayscale images (filtering, thresholding, Lutz labeling, subpixel CoM) |
| `database` | Star database loading/saving (CSV + binary format), KD-tree construction |
| `solver` | Pattern matching and plate solving (hash lookup, verification, rotation extraction) |
| `geometry` | Pinhole camera model, lens distortion, Kabsch rotation |
| `calibration` | Camera calibration TOML parser, OpenCV5 undistortion, centroid pre-processing |
| `hash_table` | Pattern catalog with quadratic probing hash table |
| `img_filter` | Image filtering primitives (median, uniform, binary opening) |

## CLI options

```
Usage: tetrars <database_dir> <image> [options]

Options:
  --fov <degrees>          Estimated field of view
  --fov-error <degrees>    Max FOV error from estimate (default: 5.0 if fov given)
  --distortion <k>         Known distortion coefficient (single radial k)
  --timeout <ms>           Solve timeout in milliseconds
  --stars <n>              Pattern checking stars (default: 8)
  --cal <path>             TOML camera calibration file
```


e.g.
```
cargo run --release --example tetrars data/fov40/ outputs/pixelink.png --cal outputs/pixelink.toml --match-threshold 1e-2 --stars 16B
```


## License

Apache 2.0. See [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE) for attribution to the original ESA project.
