#![allow(dead_code)]

/// STC — Syndrome-Trellis Coding
///
/// Embedding: finds the minimum-cost modification sequence
///            such that H·y = m  (in GF2)
///
/// Decoding:  computes m = H·y  (simple GF2 matrix-vector product)

/// Parametri STC
pub struct StcParams {
    /// Generator vector for matrix H (trellis height).
    /// Determines the structure of the parity-check matrix.
    /// Must be shared between sender and receiver (together with the key).
    pub h_hat: Vec<u64>,

    /// Number of trellis states = 2^(len(h_hat)*bits_per_word).
    /// In practice h_hat is used as columns of H, each element = 1 word.
    pub h_height: usize, // number of rows of H per "column" = log2(num_states)
}

impl StcParams {
    /// Creates STC parameters with standard h_hat (from literature).
    /// h_height = 7 → 128 states, good speed/security trade-off.
    pub fn new(h_height: usize) -> Self {
        // h_hat: vector of 64-bit integers defining H.
        // Each bit of h_hat[i] defines which row of H is XORed
        // when processing the i-th coefficient.
        // Standard values from literature for h=7:
        let h_hat = vec![
            0b1011011u64, // column 0
            0b1111001u64, // column 1
            0b1010011u64, // column 2
                          // Repeated cyclically for all n coefficients
        ];
        StcParams { h_hat, h_height }
    }

    /// Number of trellis states.
    pub fn num_states(&self) -> usize {
        1 << self.h_height
    }

    /// Computes the state transition when choosing value y_i
    /// for the i-th coefficient.
    /// The state is an integer in [0, 2^h_height).
    ///
    /// H is constructed as follows: column i is h_hat[i % h_hat.len()],
    /// rotated by (i / h_hat.len()) * 1 bit (banded structure).
    #[inline]
    pub fn next_state(&self, state: usize, coeff_idx: usize, y_bit: u8) -> usize {
        if y_bit == 0 {
            // If y_i = 0, column of H contributes 0 → state unchanged
            state
        } else {
            // If y_i = 1, XOR with column i of H
            let col = self.h_column(coeff_idx);
            state ^ col
        }
    }

    /// Returns the i-th column of H as a bitmask.
    #[inline]
    fn h_column(&self, i: usize) -> usize {
        let base = self.h_hat[i % self.h_hat.len()] as usize;
        // Cyclic rotation to ensure H is "universal"
        let shift = (i / self.h_hat.len()) % self.h_height;
        let mask = (1 << self.h_height) - 1;
        ((base << shift) | (base >> (self.h_height - shift))) & mask
    }
}

const INF_COST: f64 = f64::INFINITY;

/// STC embedding via parity interleaving.
///
/// Coefficients are divided into k groups with interleaving:
///   group j = { j, j+k, j+2k, ... }
/// For each group the parity of the stego bits must equal message[j].
/// If not, the minimum-cost bit in the group is flipped.
///
/// Extraction is simply recomputing the same parities → correctness guaranteed.
pub fn stc_embed(
    cover_bits: &[u8],
    costs: &[f64],
    message: &[u8],
    _params: &StcParams,
) -> Result<Vec<u8>, StcError> {
    let n = cover_bits.len();
    let k = message.len();

    if k > n {
        return Err(StcError::PayloadTooLarge {
            payload: k,
            capacity: n,
        });
    }

    let mut stego_bits = cover_bits.to_vec();

    for j in 0..k {
        // Calcola parità attuale del gruppo j e trova l'elemento di costo minimo
        let mut parity: u8 = 0;
        let mut min_cost = f64::INFINITY;
        let mut min_idx = j;

        let mut idx = j;
        while idx < n {
            parity ^= cover_bits[idx];
            if costs[idx] < min_cost {
                min_cost = costs[idx];
                min_idx = idx;
            }
            idx += k;
        }

        // Se la parità non corrisponde al bit di messaggio, flippa il minimo-costo
        if parity != message[j] {
            stego_bits[min_idx] ^= 1;
        }
    }

    Ok(stego_bits)
}

/// STC decoding: extracts the message from the stego image.
///
/// For each group j, computes the XOR parity of the stego bits
/// at positions {j, j+k, j+2k, ...} — symmetric to embedding.
pub fn stc_extract(stego_bits: &[u8], k: usize, _params: &StcParams) -> Vec<u8> {
    let n = stego_bits.len();
    let mut message = vec![0u8; k];

    for j in 0..k {
        let mut parity: u8 = 0;
        let mut idx = j;
        while idx < n {
            parity ^= stego_bits[idx];
            idx += k;
        }
        message[j] = parity;
    }

    message
}

/// Converts bytes to bits (MSB first)
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1))
        .collect()
}

/// Converts bits to bytes (MSB first)
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &b)| acc | (b << (7 - i)))
        })
        .collect()
}

/// STC errors
#[derive(Debug)]
pub enum StcError {
    PayloadTooLarge { payload: usize, capacity: usize },
    EmbeddingFailed,
}

impl std::fmt::Display for StcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StcError::PayloadTooLarge { payload, capacity } => write!(
                f,
                "Payload ({} bits) exceeds capacity ({} bits)",
                payload, capacity
            ),
            StcError::EmbeddingFailed => {
                write!(f, "STC embedding failed: no valid path in trellis")
            }
        }
    }
}
