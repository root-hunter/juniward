/// J-UNIWARD: JPEG Universal Wavelet Relative Distortion
///
/// Computes the embedding cost for each DCT coefficient.
/// Low costs = textured areas (safe to modify)
/// High costs = smooth areas (modifications are visible)

/// Daubechies 8-tap (DB8) wavelet filters in 3 directions:
/// - HH (diagonal)
/// - HL (horizontal)
/// - LH (vertical)
/// Each filter is separable: applied first on rows then on columns.
const DB8_LO: [f64; 8] = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.187034811718881,
    -0.027983769416983849,
    0.630880767929590,
    0.714846570552542,
    0.230377813308855,
];

const DB8_HI: [f64; 8] = [
    -0.230377813308855,
    0.714846570552542,
    -0.630880767929590,
    -0.027983769416983849,
    0.187034811718881,
    0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
];

/// Applies 1D convolution with reflect padding.
fn convolve1d(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let k = kernel.len();
    let pad = k / 2;
    let mut out = vec![0.0f64; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            // Index with reflect padding
            let idx = i as isize + j as isize - pad as isize;
            let idx = reflect_index(idx, n);
            sum += signal[idx] * kernel[k - 1 - j];
        }
        out[i] = sum;
    }
    out
}

fn reflect_index(idx: isize, n: usize) -> usize {
    let n = n as isize;
    let idx = if idx < 0 { -idx - 1 } else { idx };
    let idx = if idx >= n { 2 * n - idx - 1 } else { idx };
    idx.clamp(0, n - 1) as usize
}

/// Applies a separable 2D wavelet filter to a rows×cols matrix.
/// Returns the filtered matrix linearized in row-major order.
fn apply_wavelet_2d(
    img: &[f64],
    rows: usize,
    cols: usize,
    row_filter: &[f64],
    col_filter: &[f64],
) -> Vec<f64> {
    // First pass: filter each row
    let mut tmp = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let row_slice = &img[r * cols..(r + 1) * cols];
        let filtered = convolve1d(row_slice, col_filter);
        tmp[r * cols..(r + 1) * cols].copy_from_slice(&filtered);
    }

    // Second pass: filter each column
    let mut out = vec![0.0f64; rows * cols];
    for c in 0..cols {
        let col_vec: Vec<f64> = (0..rows).map(|r| tmp[r * cols + c]).collect();
        let filtered = convolve1d(&col_vec, row_filter);
        for r in 0..rows {
            out[r * cols + c] = filtered[r];
        }
    }
    out
}

/// Reconstructs the spatial image from DCT blocks.
/// Uses a manual 8x8 IDCT (direct formula).
pub fn idct_image(dct_blocks: &[i16], width_blocks: usize, height_blocks: usize) -> Vec<f64> {
    let width = width_blocks * 8;
    let height = height_blocks * 8;
    let mut spatial = vec![0.0f64; width * height];

    for br in 0..height_blocks {
        for bc in 0..width_blocks {
            let block_idx = (br * width_blocks + bc) * 64;
            let block = &dct_blocks[block_idx..block_idx + 64];

            // IDCT 8x8
            for y in 0..8 {
                for x in 0..8 {
                    let mut sum = 0.0f64;
                    for v in 0..8usize {
                        for u in 0..8usize {
                            let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                            let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                            let coef = block[v * 8 + u] as f64;
                            sum += cu
                                * cv
                                * coef
                                * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0)
                                    .cos()
                                * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0)
                                    .cos();
                        }
                    }
                    let px_row = br * 8 + y;
                    let px_col = bc * 8 + x;
                    spatial[px_row * width + px_col] = sum / 4.0 + 128.0;
                }
            }
        }
    }
    spatial
}

/// Computes J-UNIWARD costs for every modifiable DCT coefficient.
///
/// Exploits convolution linearity: W(cover + delta) - W(cover) = W(delta).
/// The wavelet deltas for all 64 DCT basis functions are precomputed once,
/// avoiding repeated calls to apply_wavelet_2d for every block.
///
/// Returns a cost vector with one entry per DCT coefficient (n_blocks * 64).
pub fn compute_jwuniward_costs(
    dct_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
    sigma: f64, // numerical stabilizer, typically 1e-10
) -> Vec<f64> {
    let n_blocks = width_blocks * height_blocks;
    let width = width_blocks * 8;
    let height = height_blocks * 8;

    // 1. Reconstruct spatial image from cover
    let spatial = idct_image(dct_blocks, width_blocks, height_blocks);

    // 2. Precompute the 3 cover wavelet residuals (HL, LH, HH) — 3 calls total
    let w_hl = apply_wavelet_2d(&spatial, height, width, &DB8_LO, &DB8_HI);
    let w_lh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_LO);
    let w_hh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_HI);
    let wavelets_cover: [&Vec<f64>; 3] = [&w_hl, &w_lh, &w_hh];

    // 3. Precompute the 64 DCT basis functions
    let basis = precompute_dct_basis();

    // 4. Precompute wavelet deltas for each of the 64 DCT coefficients.
    //    By convolution linearity: W(cover+delta) - W(cover) = W(delta).
    //    The delta is the DCT basis function / 4 on an 8x8 block embedded in a zero context.
    //    Influence area: the 8-tap filter extends support by pad=8 pixels on each side.
    //    → context: (8 + 2*pad) × (8 + 2*pad) = 24×24 pixels.
    //    This replaces 691,200 calls to apply_wavelet_2d with just 192.
    let pad = DB8_LO.len(); // 8
    let ctx_size = 8 + 2 * pad; // 24

    // delta_w[coeff_idx][subband] = ctx_size×ctx_size vector with the wavelet response of the delta
    let mut delta_w: Vec<[Vec<f64>; 3]> = Vec::with_capacity(64);
    for coeff_idx in 0..64usize {
        let mut ctx = vec![0.0f64; ctx_size * ctx_size];
        for y in 0..8 {
            for x in 0..8 {
                ctx[(y + pad) * ctx_size + (x + pad)] = basis[coeff_idx][y * 8 + x] / 4.0;
            }
        }
        let dw_hl = apply_wavelet_2d(&ctx, ctx_size, ctx_size, &DB8_LO, &DB8_HI);
        let dw_lh = apply_wavelet_2d(&ctx, ctx_size, ctx_size, &DB8_HI, &DB8_LO);
        let dw_hh = apply_wavelet_2d(&ctx, ctx_size, ctx_size, &DB8_HI, &DB8_HI);
        delta_w.push([dw_hl, dw_lh, dw_hh]);
    }

    // 5. Compute costs: for each block and coefficient, accumulate the relative variation
    //    using the precomputed wavelet deltas.
    let mut costs = vec![0.0f64; n_blocks * 64];

    for br in 0..height_blocks {
        for bc in 0..width_blocks {
            let block_base = (br * width_blocks + bc) * 64;
            let px_r = br * 8;
            let px_c = bc * 8;

            for coeff_idx in 0..64usize {
                let dw = &delta_w[coeff_idx];
                let mut cost = 0.0f64;

                for k in 0..3usize {
                    let w_cover = wavelets_cover[k];
                    let dw_k = &dw[k];

                    // The context is centered on the block:
                    // ctx[r_ctx][c_ctx] maps to image position
                    // (px_r - pad + r_ctx, px_c - pad + c_ctx)
                    for r_ctx in 0..ctx_size {
                        let r_img = px_r as isize - pad as isize + r_ctx as isize;
                        if r_img < 0 || r_img >= height as isize {
                            continue;
                        }
                        let r_img = r_img as usize;

                        for c_ctx in 0..ctx_size {
                            let c_img = px_c as isize - pad as isize + c_ctx as isize;
                            if c_img < 0 || c_img >= width as isize {
                                continue;
                            }
                            let c_img = c_img as usize;

                            let delta = dw_k[r_ctx * ctx_size + c_ctx];
                            if delta == 0.0 {
                                continue;
                            }
                            let w_orig = w_cover[r_img * width + c_img];
                            cost += delta.abs() / (sigma + w_orig.abs());
                        }
                    }
                }

                costs[block_base + coeff_idx] = cost;
            }
        }
    }

    costs
}

/// Precomputes the 64 DCT 8x8 basis functions.
/// basis[coeff_idx][pixel_idx] = value of the basis function at that pixel.
fn precompute_dct_basis() -> Vec<Vec<f64>> {
    let mut basis = vec![vec![0.0f64; 64]; 64];
    for v in 0..8usize {
        for u in 0..8usize {
            let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            for y in 0..8usize {
                for x in 0..8usize {
                    basis[v * 8 + u][y * 8 + x] = cu
                        * cv
                        * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                        * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                }
            }
        }
    }
    basis
}
