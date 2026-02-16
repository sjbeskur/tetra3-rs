# Tools

Utilities for working with tetra3 star databases.

## convert_npz.py

Converts a Python tetra3 `.npz` database (NumPy compressed format) into the directory-based format used by the Rust solver.

### Prerequisites

- Python 3.12+
- NumPy

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
cd tools/
uv sync
```

Or with pip:

```bash
pip install numpy
```

### Usage

```bash
python tools/convert_npz.py <input.npz> <output_dir/>
```

Example:

```bash
python tools/convert_npz.py tetra3_database.npz data/tetra_db/
```

### Output

The converter produces a directory containing:

| File                | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `properties.csv`    | Single-row CSV with database metadata (FOV range, pattern params)  |
| `stars.csv`         | Star catalog: RA, Dec, unit vector (x, y, z), magnitude            |
| `patterns.bin`      | Binary pattern hash table (little-endian u32 pattern size, u64 row count, flat u32 data) |
| `largest_edges.bin` | Optional presorted largest edge data (little-endian u64 count, flat f64 data) |

The output directory can be loaded directly by `TetraDatabase::load()` in the Rust library.

### Where to get .npz databases

Generate a `.npz` database using the original Python [tetra3](https://github.com/esa/tetra3) package, then convert it with this tool for use with the Rust solver.
