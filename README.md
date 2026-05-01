# juniward

**J-UNIWARD + STC steganography for JPEG images, written in Rust.**

juniward implements the [J-UNIWARD](https://doi.org/10.1186/1687-417X-2014-1) (JPEG Universal Wavelet Relative Distortion) cost function together with Syndrome-Trellis Coding (STC) to embed and extract secret messages inside JPEG files with minimal perceptual distortion.

---

## Features

- **J-UNIWARD cost function** — computes per-coefficient embedding costs in the DB8 wavelet domain; low-cost regions (textures) absorb changes invisibly.
- **STC embedding** — minimum-distortion LSB assignment via parity-interleaved syndrome coding.
- **In-place DCT modification** — reads and writes the raw DCT coefficients through `mozjpeg-sys`; no re-compression quality loss.
- **Parallel computation** — wavelet and cost stages use [Rayon](https://github.com/rayon-rs/rayon) for multi-core speed.
- **Simple API** — three public functions cover the common use cases.

---

## Requirements

- Rust 1.80+ (edition 2024)
- A C toolchain (`gcc` / `clang`) for building `mozjpeg-sys`
- `libjpeg`-compatible headers (provided automatically by `mozjpeg-sys`)

---

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
juniward = { path = "path/to/juniward" }
```

### Embed a message

```rust
use juniward::{embed, EmbedConfig};

let cover = std::fs::read("cover.jpg")?;
let message = b"Secret payload";

// Default config: sigma=1e-10, STC height=7, max rate=0.4 bpnzAC
let stego = embed(&cover, message, EmbedConfig::default())?;
std::fs::write("stego.jpg", &stego)?;
```

### Extract a message

```rust
use juniward::extract;

let stego = std::fs::read("stego.jpg")?;
let recovered = extract(&stego, 14 /* message length in bytes */)?;

assert_eq!(recovered, b"Secret payload");
```

### Analyse embedding costs (without modifying the image)

```rust
use juniward::compute_costs;

let data = std::fs::read("cover.jpg")?;
let (costs, stats) = compute_costs(&data, 1e-10);

println!("min={:.4}  max={:.4}  mean={:.4}", stats.min, stats.max, stats.mean);
```

---

## API reference

### `embed`

```rust
pub fn embed(cover: &[u8], message: &[u8], cfg: EmbedConfig) -> Result<Vec<u8>, JuniwardError>
```

Embeds `message` bytes into a JPEG cover image and returns the stego JPEG.

**Errors**
| Variant | Cause |
|---|---|
| `JuniwardError::InvalidJpeg` | Input bytes cannot be decoded as JPEG |
| `JuniwardError::PayloadTooLarge` | Message exceeds the safe embedding capacity |
| `JuniwardError::EmbeddingFailed` | STC could not find a valid embedding path |

### `extract`

```rust
pub fn extract(stego: &[u8], message_len: usize) -> Result<Vec<u8>, JuniwardError>
```

Extracts a `message_len`-byte message from a stego JPEG.  
The sender and receiver must agree on `message_len` (and `StcParams` if non-default).

### `extract_with_params`

```rust
pub fn extract_with_params(
    stego: &[u8],
    message_len: usize,
    params: &StcParams,
) -> Result<Vec<u8>, JuniwardError>
```

Same as `extract` but accepts custom `StcParams`.

### `compute_costs`

```rust
pub fn compute_costs(jpeg_data: &[u8], sigma: f64) -> (Vec<f64>, CostStats)
```

Returns J-UNIWARD costs for every DCT coefficient and summary statistics.  
Useful for analysis pipelines that implement their own embedding strategy.

---

## Configuration

```rust
pub struct EmbedConfig {
    /// Wavelet regularisation constant. Default: 1e-10.
    pub sigma: f64,

    /// STC trellis height (log₂ of the number of states). Default: 7 (128 states).
    pub stc_h_height: usize,

    /// Maximum payload in bits-per-non-zero-AC-coefficient. Default: 0.4.
    pub max_bpnzac: f64,
}
```

`EmbedConfig::default()` gives conservative, publication-standard settings.  
Increase `stc_h_height` for stronger security; decrease `max_bpnzac` for lower detectability.

---

## How it works

```
Cover JPEG
    │
    ▼
mozjpeg-sys ──► DCT coefficients  (Y channel, all blocks)
    │
    ▼
IDCT  ──► spatial domain  ──► DB8 DWT (HL, LH, HH subbands)
    │
    ▼
J-UNIWARD cost per coefficient
    c(i) = Σ_{k∈{HL,LH,HH}} |Δ W_k(i)| / (σ + |W_k(i)|)
    │
    ▼
STC: find LSB assignment  y  that minimises  Σ c(i)·|y_i - x_i|
     subject to  H·y = m  (GF2 parity constraints)
    │
    ▼
mozjpeg-sys ──► Stego JPEG (lossless DCT round-trip, no re-compression)
```

DC coefficients and ±1 AC coefficients are never modified to preserve JPEG RLE entropy coding stability.

---

## Security notes

- **This library is a research implementation.** Do not use it as the sole layer of security for sensitive communications.
- The STC generator vector `h_hat` and `message_len` act as implicit shared secrets — both must match between sender and receiver.
- Statistical steganalysis (e.g., SRM, J-STRUCT) may still detect embedding at high payload rates.  
  Keeping `max_bpnzac ≤ 0.4` is the recommended safe threshold from the literature.

---

## License

This project is licensed under the MIT License.
