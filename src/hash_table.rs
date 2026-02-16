//! Pattern hash table with quadratic probing for star pattern lookup.
//!
//! The pattern catalog is a 2D array where each row stores the star IDs of one pattern
//! (typically 4 stars). Row 0 is reserved as the "empty" sentinel. Patterns are addressed
//! by hashing their edge-ratio key into a row index and resolving collisions with
//! quadratic probing (`index + c²`).
//!
//! # Hash function
//!
//! A pattern key is a list of `p` binned edge ratios (typically 5 values, each in
//! `0..bin_factor`). The hash is:
//!
//! ```text
//! raw = sum(key[i] * bin_factor^i)          (polynomial in bin_factor)
//! hash = (raw * MAGIC_RAND) % table_size    (Knuth multiplicative hash)
//! ```
//!
//! where `MAGIC_RAND = 2654435761` (the golden ratio constant `(√5-1)/2 * 2^32`,
//! rounded to the nearest prime). All arithmetic is wrapping `u64`.

/// Knuth multiplicative hashing constant (closest prime to `(√5-1)/2 * 2^32`).
const MAGIC_RAND: u64 = 2654435761;

/// Hash a pattern key (binned edge ratios) into a table index.
///
/// Computes `sum(key[i] * bin_factor^i) * MAGIC_RAND % max_index`.
/// Matches the Python `_key_to_index` for a single key.
pub fn key_to_index(key: &[u16], bin_factor: u64, max_index: u64) -> u64 {
    let mut raw: u64 = 0;
    let mut power: u64 = 1; // bin_factor^0
    for &k in key.iter() {
        raw = raw.wrapping_add((k as u64).wrapping_mul(power));
        power = power.wrapping_mul(bin_factor);
    }
    raw.wrapping_mul(MAGIC_RAND) % max_index
}

/// Batch-hash multiple keys at once (matches Python `_key_to_index` with 2D input).
pub fn key_to_index_batch(keys: &[Vec<u16>], bin_factor: u64, max_index: u64) -> Vec<u64> {
    keys.iter()
        .map(|key| key_to_index(key, bin_factor, max_index))
        .collect()
}

/// Look up a hash in the pattern catalog using quadratic probing.
///
/// Returns the row indices of all occupied slots in the probe chain starting at
/// `hash_index`. Stops at the first empty row (where all columns are zero).
///
/// `table` is the pattern catalog: rows are patterns, columns are star IDs.
/// A row is empty if all its values are zero.
pub fn get_table_indices_from_hash(hash_index: u64, table: &PatternCatalog) -> Vec<usize> {
    let max_ind = table.num_rows() as u64;
    let mut found = Vec::new();
    for c in 0u64.. {
        let i = ((hash_index.wrapping_add(c.wrapping_mul(c))) % max_ind) as usize;
        if table.is_row_empty(i) {
            break;
        }
        found.push(i);
    }
    found
}

/// Insert a pattern into the catalog at the hashed position using quadratic probing.
///
/// Finds the first empty row in the probe chain and writes the pattern there.
/// Returns the row index where the pattern was inserted.
pub fn insert_at_index(
    pattern: &[u32],
    hash_index: u64,
    table: &mut PatternCatalog,
) -> usize {
    let max_ind = table.num_rows() as u64;
    for c in 0u64.. {
        let i = ((hash_index.wrapping_add(c.wrapping_mul(c))) % max_ind) as usize;
        if table.is_row_empty(i) {
            table.set_row(i, pattern);
            return i;
        }
    }
    unreachable!("table is full")
}

/// Storage for the pattern catalog (hash table of star patterns).
///
/// Each row contains `pattern_size` star IDs. Row 0 is the empty sentinel
/// (all zeros means "no pattern here").
pub struct PatternCatalog {
    /// Flat storage: `data[row * pattern_size .. (row+1) * pattern_size]`
    data: Vec<u32>,
    /// Number of star IDs per pattern (typically 4)
    pattern_size: usize,
}

impl PatternCatalog {
    /// Create an empty catalog with the given dimensions (all zeros).
    pub fn new(num_rows: usize, pattern_size: usize) -> Self {
        Self {
            data: vec![0u32; num_rows * pattern_size],
            pattern_size,
        }
    }

    /// Create a catalog from existing flat data.
    pub fn from_data(data: Vec<u32>, pattern_size: usize) -> Self {
        assert_eq!(data.len() % pattern_size, 0);
        Self { data, pattern_size }
    }

    pub fn num_rows(&self) -> usize {
        self.data.len() / self.pattern_size
    }

    pub fn pattern_size(&self) -> usize {
        self.pattern_size
    }

    /// Check if a row is empty (all zeros = no pattern stored).
    pub fn is_row_empty(&self, row: usize) -> bool {
        let start = row * self.pattern_size;
        self.data[start..start + self.pattern_size]
            .iter()
            .all(|&v| v == 0)
    }

    /// Get the star IDs for a pattern at the given row.
    pub fn get_row(&self, row: usize) -> &[u32] {
        let start = row * self.pattern_size;
        &self.data[start..start + self.pattern_size]
    }

    /// Write a pattern's star IDs into the given row.
    pub fn set_row(&mut self, row: usize, pattern: &[u32]) {
        assert_eq!(pattern.len(), self.pattern_size);
        let start = row * self.pattern_size;
        self.data[start..start + self.pattern_size].copy_from_slice(pattern);
    }

    /// Get patterns at multiple row indices.
    pub fn get_rows(&self, indices: &[usize]) -> Vec<Vec<u32>> {
        indices.iter().map(|&i| self.get_row(i).to_vec()).collect()
    }

    /// Write the pattern catalog as raw little-endian binary.
    ///
    /// Format: u32 pattern_size, u64 num_rows, then flat u32 data.
    pub fn write_binary(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writer.write_all(&(self.pattern_size as u32).to_le_bytes())?;
        writer.write_all(&(self.num_rows() as u64).to_le_bytes())?;
        // Write flat data as raw LE bytes in bulk
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * 4,
            )
        };
        // On little-endian systems, write directly; this is the common case
        #[cfg(target_endian = "little")]
        writer.write_all(byte_slice)?;
        #[cfg(target_endian = "big")]
        for &val in &self.data {
            writer.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Read the pattern catalog from raw little-endian binary.
    pub fn read_binary(reader: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        let pattern_size = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf8)?;
        let num_rows = u64::from_le_bytes(buf8) as usize;

        let total = num_rows * pattern_size;
        let byte_count = total * 4;
        let mut bytes = vec![0u8; byte_count];
        reader.read_exact(&mut bytes)?;
        let data: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(Self { data, pattern_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_to_index_deterministic() {
        let key = [100u16, 200, 300, 400, 500];
        let idx1 = key_to_index(&key, 50, 1_000_000);
        let idx2 = key_to_index(&key, 50, 1_000_000);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_key_to_index_within_bounds() {
        let key = [65535u16, 65535, 65535, 65535, 65535];
        let max_index = 1_000_000u64;
        let idx = key_to_index(&key, 50, max_index);
        assert!(idx < max_index);
    }

    #[test]
    fn test_key_to_index_matches_python() {
        // Python: key = [10, 20, 30, 40, 25], bin_factor = 50, max_index = 1000000
        // raw = 10*1 + 20*50 + 30*2500 + 40*125000 + 25*6250000
        //     = 10 + 1000 + 75000 + 5000000 + 156250000 = 161326010
        // hash = (161326010 * 2654435761) % 1000000
        let key = [10u16, 20, 30, 40, 25];
        let raw: u64 = 10 + 20 * 50 + 30 * 2500 + 40 * 125000 + 25 * 6250000;
        let expected = raw.wrapping_mul(MAGIC_RAND) % 1_000_000;
        let result = key_to_index(&key, 50, 1_000_000);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut catalog = PatternCatalog::new(100, 4);
        let pattern = [1u32, 2, 3, 4];
        let hash = 10u64;

        let inserted_at = insert_at_index(&pattern, hash, &mut catalog);
        assert_eq!(inserted_at, 10);
        assert_eq!(catalog.get_row(10), &pattern);

        let found = get_table_indices_from_hash(hash, &catalog);
        assert_eq!(found, vec![10]);
    }

    #[test]
    fn test_insert_collision_quadratic_probe() {
        let mut catalog = PatternCatalog::new(100, 4);
        let hash = 10u64;

        // Insert first pattern at index 10
        insert_at_index(&[1, 2, 3, 4], hash, &mut catalog);
        // Second pattern with same hash probes to 10+1=11
        insert_at_index(&[5, 6, 7, 8], hash, &mut catalog);
        // Third pattern probes to 10+4=14
        insert_at_index(&[9, 10, 11, 12], hash, &mut catalog);

        let found = get_table_indices_from_hash(hash, &catalog);
        assert_eq!(found, vec![10, 11, 14]);

        assert_eq!(catalog.get_row(10), &[1, 2, 3, 4]);
        assert_eq!(catalog.get_row(11), &[5, 6, 7, 8]);
        assert_eq!(catalog.get_row(14), &[9, 10, 11, 12]);
    }

    #[test]
    fn test_empty_lookup() {
        let catalog = PatternCatalog::new(100, 4);
        let found = get_table_indices_from_hash(42, &catalog);
        assert!(found.is_empty());
    }

    #[test]
    fn test_batch_hash() {
        let keys = vec![
            vec![10u16, 20, 30, 40, 25],
            vec![1, 2, 3, 4, 5],
        ];
        let results = key_to_index_batch(&keys, 50, 1_000_000);
        assert_eq!(results[0], key_to_index(&keys[0], 50, 1_000_000));
        assert_eq!(results[1], key_to_index(&keys[1], 50, 1_000_000));
    }
}
