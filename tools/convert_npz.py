#!/usr/bin/env python3
"""Convert a tetra3 .npz database to the Rust directory-based format.

Usage:
    python convert_npz.py input.npz output_dir/

The output directory can be loaded by TetraDatabase::load() in Rust.
It contains:
  - properties.csv  (single-row CSV with headers)
  - stars.csv       (CSV: ra,dec,x,y,z,magnitude)
  - patterns.bin    (raw LE binary: u32 pattern_size + u64 num_rows + flat u32 data)
  - largest_edges.bin (raw LE binary: u64 count + flat f64 data, only if present)
"""

import csv
import os
import struct
import sys
import numpy as np


def convert(input_path, output_dir):
    print(f"Loading {input_path}...")
    with np.load(input_path) as data:
        star_table = data["star_table"]  # (N, 6) float32
        pattern_catalog = data["pattern_catalog"]  # (M, pattern_size) uint
        props_packed = data["props_packed"]

        pattern_largest_edge = None
        try:
            pattern_largest_edge = data["pattern_largest_edge"]
        except KeyError:
            pass

    # Unpack properties from structured array
    props = {}
    for key in [
        "pattern_mode", "pattern_size", "pattern_bins", "pattern_max_error",
        "max_fov", "min_fov", "star_catalog", "epoch_equinox",
        "epoch_proper_motion", "pattern_stars_per_fov",
        "verification_stars_per_fov", "star_max_magnitude",
        "simplify_pattern", "range_ra", "range_dec", "presort_patterns",
    ]:
        try:
            val = props_packed[key][()]
            # Convert numpy types to Python types
            if isinstance(val, np.generic):
                val = val.item()
            props[key] = val
        except (ValueError, KeyError):
            # Handle legacy key names
            if key == "verification_stars_per_fov":
                try:
                    props[key] = int(props_packed["catalog_stars_per_fov"][()])
                except (ValueError, KeyError):
                    props[key] = 20
            elif key == "star_max_magnitude":
                try:
                    props[key] = float(props_packed["star_min_magnitude"][()])
                except (ValueError, KeyError):
                    props[key] = 7.0
            elif key == "presort_patterns":
                props[key] = False
            elif key == "simplify_pattern":
                props[key] = False
            else:
                props[key] = None

    pattern_size = int(props["pattern_size"])

    print(f"  Stars: {star_table.shape[0]}")
    print(f"  Pattern rows: {pattern_catalog.shape[0]}")
    print(f"  Pattern size: {pattern_size}")
    print(f"  FOV range: {props['min_fov']:.1f} - {props['max_fov']:.1f} deg")
    if pattern_largest_edge is not None:
        print(f"  Largest edge data: yes ({len(pattern_largest_edge)} entries)")
    else:
        print("  Largest edge data: no")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing to {output_dir}/...")

    # 1. properties.csv
    write_properties_csv(os.path.join(output_dir, "properties.csv"), props)

    # 2. stars.csv
    write_stars_csv(os.path.join(output_dir, "stars.csv"), star_table)

    # 3. patterns.bin
    write_patterns_bin(os.path.join(output_dir, "patterns.bin"),
                       pattern_catalog, pattern_size)

    # 4. largest_edges.bin (optional)
    if pattern_largest_edge is not None:
        write_largest_edges_bin(
            os.path.join(output_dir, "largest_edges.bin"),
            pattern_largest_edge)

    print("Done!")


def unpack_range(val):
    """Unpack a range value to (min_str, max_str). Returns ("", "") for None/full-sky."""
    if val is None:
        return ("", "")
    arr = np.asarray(val, dtype=np.float64).flatten()
    if len(arr) == 2 and arr[0] == 0.0 and arr[1] == 0.0:
        return ("", "")
    if np.any(np.isnan(arr)):
        return ("", "")
    return (repr(float(arr[0])), repr(float(arr[1])))


def write_properties_csv(path, props):
    """Write properties as a single-row CSV with headers."""
    ra_min, ra_max = unpack_range(props.get("range_ra"))
    dec_min, dec_max = unpack_range(props.get("range_dec"))

    epoch_eq = props.get("epoch_equinox")
    epoch_pm = props.get("epoch_proper_motion")

    headers = [
        "pattern_mode", "pattern_size", "pattern_bins", "pattern_max_error",
        "max_fov", "min_fov", "verification_stars_per_fov",
        "pattern_stars_per_fov", "star_max_magnitude", "star_catalog",
        "epoch_equinox", "epoch_proper_motion",
        "presort_patterns", "simplify_pattern",
        "range_ra_min", "range_ra_max", "range_dec_min", "range_dec_max",
    ]
    values = [
        str(props["pattern_mode"]),
        str(int(props["pattern_size"])),
        str(int(props["pattern_bins"])),
        repr(float(props["pattern_max_error"])),
        repr(float(props["max_fov"])),
        repr(float(props["min_fov"])),
        str(int(props["verification_stars_per_fov"])),
        str(int(props["pattern_stars_per_fov"])),
        repr(float(props["star_max_magnitude"])),
        str(props["star_catalog"]),
        repr(float(epoch_eq)) if epoch_eq is not None else "",
        repr(float(epoch_pm)) if epoch_pm is not None else "",
        str(bool(props.get("presort_patterns", False))).lower(),
        str(bool(props.get("simplify_pattern", False))).lower(),
        ra_min, ra_max, dec_min, dec_max,
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(values)


def write_stars_csv(path, star_table):
    """Write star table as CSV: ra,dec,x,y,z,magnitude with full precision."""
    star_table_f64 = star_table.astype(np.float64)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ra", "dec", "x", "y", "z", "magnitude"])
        for row in star_table_f64:
            writer.writerow([repr(float(v)) for v in row])


def write_patterns_bin(path, pattern_catalog, pattern_size):
    """Write pattern catalog as raw LE binary."""
    flat = pattern_catalog.flatten().astype(np.uint32)
    num_rows = len(flat) // pattern_size
    with open(path, "wb") as f:
        f.write(struct.pack("<I", pattern_size))    # u32
        f.write(struct.pack("<Q", num_rows))        # u64
        f.write(flat.astype("<u4").tobytes())        # flat u32 data, explicit LE


def write_largest_edges_bin(path, largest_edge):
    """Write largest edge data as raw LE binary."""
    edge_f64 = largest_edge.astype(np.float64)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(edge_f64)))   # u64 count
        f.write(edge_f64.astype("<f8").tobytes())    # flat f64 data, explicit LE


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.npz output_dir/")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
