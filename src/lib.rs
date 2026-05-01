/// juniward-rs — J-UNIWARD + STC steganography library
///
/// # Quick start
///
/// ```no_run
/// use juniward::{embed, extract};
///
/// let cover = std::fs::read("cover.jpg").unwrap();
/// let message = b"secret message";
///
/// // Embed
/// let stego = embed(&cover, message, Default::default()).unwrap();
///
/// // Extract
/// let recovered = extract(&stego, message.len()).unwrap();
/// assert_eq!(recovered, message);
/// ```
pub mod stc;
pub mod uniward;

mod jpeg;

pub use stc::{StcError, StcParams};

use jpeg::{read_jpeg_dct, write_jpeg_dct};
use stc::{bits_to_bytes, bytes_to_bits, stc_embed, stc_extract};
use uniward::compute_jwuniward_costs;

// ─── Public error type ────────────────────────────────────────────────────────

/// Errors returned by the library.
#[derive(Debug)]
pub enum JuniwardError {
    /// The JPEG could not be decoded.
    InvalidJpeg(String),
    /// The message is too long for the cover image.
    PayloadTooLarge {
        payload_bits: usize,
        max_bits: usize,
    },
    /// STC embedding failed.
    EmbeddingFailed(StcError),
}

impl std::fmt::Display for JuniwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JuniwardError::InvalidJpeg(s) => write!(f, "Invalid JPEG: {s}"),
            JuniwardError::PayloadTooLarge {
                payload_bits,
                max_bits,
            } => write!(
                f,
                "Payload too large: {payload_bits} bits requested, max safe payload is {max_bits} bits"
            ),
            JuniwardError::EmbeddingFailed(e) => write!(f, "Embedding failed: {e}"),
        }
    }
}

impl std::error::Error for JuniwardError {}

impl From<StcError> for JuniwardError {
    fn from(e: StcError) -> Self {
        JuniwardError::EmbeddingFailed(e)
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the embedding pipeline.
pub struct EmbedConfig {
    /// Wavelet-domain regularisation constant (default: 1e-10).
    /// Smaller values → costs more sensitive to very smooth areas.
    pub sigma: f64,

    /// STC trellis height (default: 7 → 128 states).
    /// Higher values increase security at the cost of speed.
    pub stc_h_height: usize,

    /// Maximum embedding rate in bits-per-non-zero-AC coefficient (default: 0.4).
    /// Used only as a safety guard; the actual rate is determined by the message length.
    pub max_bpnzac: f64,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            sigma: 1e-10,
            stc_h_height: 7,
            max_bpnzac: 0.4,
        }
    }
}

// ─── Cost statistics helper ───────────────────────────────────────────────────

/// Statistics about J-UNIWARD costs.
#[derive(Debug, Clone)]
pub struct CostStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl CostStats {
    fn compute(costs: &[f64]) -> Self {
        let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = costs.iter().sum::<f64>() / costs.len() as f64;
        Self { min, max, mean }
    }
}

// ─── Embed ────────────────────────────────────────────────────────────────────

/// Embeds `message` bytes into a JPEG cover image using J-UNIWARD + STC.
///
/// Returns the stego JPEG as a byte vector.
///
/// # Errors
/// - [`JuniwardError::InvalidJpeg`] if the input bytes are not a valid JPEG.
/// - [`JuniwardError::PayloadTooLarge`] if the message exceeds the safe capacity.
/// - [`JuniwardError::EmbeddingFailed`] if STC cannot find a valid path.
pub fn embed(cover: &[u8], message: &[u8], cfg: EmbedConfig) -> Result<Vec<u8>, JuniwardError> {
    let params = StcParams::new(cfg.stc_h_height);
    embed_with_params(cover, message, cfg, &params)
}

/// Same as [`embed`] but with custom [`StcParams`].
pub fn embed_with_params(
    cover: &[u8],
    message: &[u8],
    cfg: EmbedConfig,
    params: &StcParams,
) -> Result<Vec<u8>, JuniwardError> {
    let jpeg = unsafe { read_jpeg_dct(cover) };

    // J-UNIWARD costs
    let costs = compute_jwuniward_costs(
        &jpeg.blocks,
        jpeg.width_blocks,
        jpeg.height_blocks,
        cfg.sigma,
    );

    // Count non-zero AC coefficients for capacity check
    let nz_ac: usize = jpeg
        .blocks
        .iter()
        .enumerate()
        .filter(|(i, v)| i % 64 != 0 && **v != 0)
        .count();
    let max_bits = (nz_ac as f64 * cfg.max_bpnzac) as usize;
    let message_bits = bytes_to_bits(message);

    if message_bits.len() > max_bits {
        return Err(JuniwardError::PayloadTooLarge {
            payload_bits: message_bits.len(),
            max_bits,
        });
    }

    // Build cover-bit and cost views over AC coefficients only
    let cover_bits: Vec<u8> = jpeg
        .blocks
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .map(|(_, v): (_, &i16)| (v.unsigned_abs() & 1) as u8)
        .collect();

    let ac_costs: Vec<f64> = costs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .zip(
            jpeg.blocks
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 64 != 0)
                .map(|(_, v)| v),
        )
        .map(|((_, &c), &v)| {
            if v == 0 || v == 1 || v == -1 {
                f64::INFINITY
            } else {
                c
            }
        })
        .collect();

    // STC embedding
    let stego_bits = stc_embed(&cover_bits, &ac_costs, &message_bits, params)?;

    // Reconstruct DCT blocks with flipped LSBs
    let mut stego_blocks = jpeg.blocks.clone();
    let mut stego_idx = 0usize;
    for (coeff_idx, coeff) in stego_blocks.iter_mut().enumerate() {
        if coeff_idx % 64 == 0 {
            continue;
        }
        let orig_bit = ((*coeff as i16).unsigned_abs() & 1) as u8;
        let new_bit = stego_bits[stego_idx];
        stego_idx += 1;

        if orig_bit != new_bit {
            if *coeff > 0 {
                *coeff ^= 1;
            } else if *coeff < 0 {
                let abs_new = (coeff.unsigned_abs() ^ 1) as i16;
                *coeff = -abs_new;
            }
        }
    }

    let stego_data =
        unsafe { write_jpeg_dct(cover, &stego_blocks, jpeg.width_blocks, jpeg.height_blocks) };

    Ok(stego_data)
}

// ─── Extract ──────────────────────────────────────────────────────────────────

/// Extracts a hidden message from a stego JPEG.
///
/// `message_len` is the expected message length **in bytes** — the same value
/// used during embedding. Both sender and receiver must agree on this value
/// (together with the STC parameters).
pub fn extract(stego: &[u8], message_len: usize) -> Result<Vec<u8>, JuniwardError> {
    extract_with_params(stego, message_len, &StcParams::new(7))
}

/// Same as [`extract`] but with custom [`StcParams`].
pub fn extract_with_params(
    stego: &[u8],
    message_len: usize,
    params: &StcParams,
) -> Result<Vec<u8>, JuniwardError> {
    let jpeg = unsafe { read_jpeg_dct(stego) };

    let stego_bits: Vec<u8> = jpeg
        .blocks
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .map(|(_, v): (_, &i16)| (v.unsigned_abs() & 1) as u8)
        .collect();

    let message_bits_len = message_len * 8;
    let recovered_bits = stc_extract(&stego_bits, message_bits_len, params);
    Ok(bits_to_bytes(&recovered_bits))
}

// ─── Lower-level helpers ──────────────────────────────────────────────────────

/// Computes J-UNIWARD costs for a raw JPEG, returned as a flat `Vec<f64>`
/// with one entry per DCT coefficient (same layout as the coefficient array).
///
/// Useful for analysis or custom embedding pipelines.
pub fn compute_costs(jpeg_data: &[u8], sigma: f64) -> (Vec<f64>, CostStats) {
    let jpeg = unsafe { read_jpeg_dct(jpeg_data) };
    let costs = compute_jwuniward_costs(&jpeg.blocks, jpeg.width_blocks, jpeg.height_blocks, sigma);
    let stats = CostStats::compute(&costs);
    (costs, stats)
}
