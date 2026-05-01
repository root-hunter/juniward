/// J-UNIWARD: JPEG Universal Wavelet Relative Distortion
///
/// Computes the embedding cost for each DCT coefficient.
/// Low costs = textured areas (safe to modify)
/// High costs = smooth areas (modifications are visible)
use rayon::prelude::*;
use std::sync::LazyLock;

// ─── DB8 wavelet filters ───────────────────────────────────────────────────────

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

// ─── Precomputed IDCT cosine table ────────────────────────────────────────────

/// COS8[k][n] = cos(π·(2n+1)·k / 16), for k,n in 0..8.
static COS8: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
    let mut t = [[0.0f64; 8]; 8];
    for k in 0..8usize {
        for n in 0..8usize {
            t[k][n] = ((2 * n + 1) as f64 * k as f64 * std::f64::consts::PI / 16.0).cos();
        }
    }
    t
});

const ISQRT2: f64 = 0.7071067811865476_f64;
const WAVELET_PAD: usize = DB8_LO.len();
const DELTA_CTX_SIZE: usize = 8 + 2 * WAVELET_PAD;
const DELTA_CTX_AREA: usize = DELTA_CTX_SIZE * DELTA_CTX_SIZE;

static IDCT8: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
    let cos = &*COS8;
    let mut t = [[0.0f64; 8]; 8];
    for k in 0..8usize {
        let c = if k == 0 { ISQRT2 } else { 1.0 };
        for n in 0..8usize {
            t[k][n] = c * cos[k][n] * 0.5;
        }
    }
    t
});

// ─── Separable 8×8 IDCT ───────────────────────────────────────────────────────

#[inline]
fn idct8x8(block: &[i16], out: &mut [f64; 64]) {
    let idct = &*IDCT8;
    let mut tmp = [0.0f64; 64];

    // Row pass
    for v in 0..8usize {
        for x in 0..8usize {
            let mut s = 0.0f64;
            for u in 0..8usize {
                s += block[v * 8 + u] as f64 * idct[u][x];
            }
            tmp[v * 8 + x] = s;
        }
    }

    // Column pass
    for x in 0..8usize {
        for y in 0..8usize {
            let mut s = 0.0f64;
            for v in 0..8usize {
                s += tmp[v * 8 + x] * idct[v][y];
            }
            out[y * 8 + x] = s;
        }
    }
}

// ─── Spatial image reconstruction ─────────────────────────────────────────────

pub fn idct_image(dct_blocks: &[i16], width_blocks: usize, height_blocks: usize) -> Vec<f64> {
    let width = width_blocks * 8;
    let mut spatial = vec![0.0f64; width * height_blocks * 8];

    spatial
        .par_chunks_mut(width * 8)
        .enumerate()
        .for_each(|(br, pixel_rows)| {
            let mut block_out = [0.0f64; 64];
            for bc in 0..width_blocks {
                let block_idx = (br * width_blocks + bc) * 64;
                idct8x8(&dct_blocks[block_idx..block_idx + 64], &mut block_out);
                for y in 0..8 {
                    for x in 0..8 {
                        pixel_rows[y * width + bc * 8 + x] = block_out[y * 8 + x] + 128.0;
                    }
                }
            }
        });

    spatial
}

// ─── 1D convolution with reflect padding ──────────────────────────────────────

#[inline]
fn reflect_index(idx: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = idx;
    if i < 0 {
        i = -i - 1;
    }
    if i >= n {
        i = 2 * n - i - 1;
    }
    i.clamp(0, n - 1) as usize
}

#[inline]
fn convolve1d_into(signal: &[f64], kernel: &[f64], out: &mut [f64]) {
    let n = signal.len();
    let k = kernel.len();
    let pad = k / 2;
    let center_end = if n + pad >= k { n - k + pad + 1 } else { 0 };

    for i in 0..pad.min(n) {
        let mut sum = 0.0;
        for j in 0..k {
            sum += signal[reflect_index(i as isize + j as isize - pad as isize, n)]
                * kernel[k - 1 - j];
        }
        out[i] = sum;
    }
    for i in pad..center_end {
        let base = i - pad;
        let mut sum = 0.0;
        for j in 0..k {
            sum += signal[base + j] * kernel[k - 1 - j];
        }
        out[i] = sum;
    }
    for i in center_end.max(pad)..n {
        let mut sum = 0.0;
        for j in 0..k {
            sum += signal[reflect_index(i as isize + j as isize - pad as isize, n)]
                * kernel[k - 1 - j];
        }
        out[i] = sum;
    }
}

// ─── 2D separable wavelet filter ──────────────────────────────────────────────

fn apply_wavelet_2d(
    img: &[f64],
    rows: usize,
    cols: usize,
    row_filter: &[f64],
    col_filter: &[f64],
) -> Vec<f64> {
    let mut tmp = vec![0.0f64; rows * cols];
    tmp.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        convolve1d_into(&img[r * cols..(r + 1) * cols], col_filter, row);
    });

    let mut out = vec![0.0f64; rows * cols];
    out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let k = row_filter.len();
        let pad = k / 2;

        for j in 0..k {
            let src_r = reflect_index(r as isize + j as isize - pad as isize, rows);
            let weight = row_filter[k - 1 - j];
            let src = &tmp[src_r * cols..(src_r + 1) * cols];
            for c in 0..cols {
                row[c] += src[c] * weight;
            }
        }
    });
    out
}

// ─── DCT basis functions ───────────────────────────────────────────────────────

fn precompute_dct_basis() -> [[f64; 64]; 64] {
    let cos = &*COS8;
    let mut basis = [[0.0f64; 64]; 64];
    for v in 0..8usize {
        for u in 0..8usize {
            let cu = if u == 0 { ISQRT2 } else { 1.0 };
            let cv = if v == 0 { ISQRT2 } else { 1.0 };
            for y in 0..8usize {
                for x in 0..8usize {
                    basis[v * 8 + u][y * 8 + x] = cu * cv * cos[u][x] * cos[v][y];
                }
            }
        }
    }
    basis
}

static DCT_BASIS: LazyLock<[[f64; 64]; 64]> = LazyLock::new(precompute_dct_basis);

static DELTA_WAVELETS: LazyLock<Vec<[Box<[f64; DELTA_CTX_AREA]>; 3]>> = LazyLock::new(|| {
    let mut delta_w = Vec::with_capacity(64);

    for coeff_idx in 0..64usize {
        let mut ctx = vec![0.0f64; DELTA_CTX_AREA];
        for y in 0..8 {
            for x in 0..8 {
                ctx[(y + WAVELET_PAD) * DELTA_CTX_SIZE + (x + WAVELET_PAD)] =
                    DCT_BASIS[coeff_idx][y * 8 + x] / 4.0;
            }
        }

        let dw_hl = apply_wavelet_2d(&ctx, DELTA_CTX_SIZE, DELTA_CTX_SIZE, &DB8_LO, &DB8_HI);
        let dw_lh = apply_wavelet_2d(&ctx, DELTA_CTX_SIZE, DELTA_CTX_SIZE, &DB8_HI, &DB8_LO);
        let dw_hh = apply_wavelet_2d(&ctx, DELTA_CTX_SIZE, DELTA_CTX_SIZE, &DB8_HI, &DB8_HI);

        let mut coeff_delta = [
            Box::new([0.0f64; DELTA_CTX_AREA]),
            Box::new([0.0f64; DELTA_CTX_AREA]),
            Box::new([0.0f64; DELTA_CTX_AREA]),
        ];
        for i in 0..DELTA_CTX_AREA {
            coeff_delta[0][i] = dw_hl[i].abs();
            coeff_delta[1][i] = dw_lh[i].abs();
            coeff_delta[2][i] = dw_hh[i].abs();
        }
        delta_w.push(coeff_delta);
    }

    delta_w
});

// ─── J-UNIWARD cost computation ───────────────────────────────────────────────

/// Computes J-UNIWARD costs for every DCT coefficient.
///
/// Exploits convolution linearity: only 192 wavelet calls total (64 × 3),
/// then reuses delta responses across all blocks.
/// The outer block loop is parallelized with rayon.
pub fn compute_jwuniward_costs(
    dct_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
    sigma: f64,
) -> Vec<f64> {
    compute_jwuniward_costs_impl(dct_blocks, width_blocks, height_blocks, sigma, 0)
}

/// Computes J-UNIWARD costs for AC coefficients only, in block-major order.
///
/// Layout per block is coefficients 1..64, so the returned vector has
/// `width_blocks * height_blocks * 63` entries. Embedding never changes DC
/// coefficients, so this avoids roughly 1/64 of the innermost cost work and
/// avoids allocating a full 64-coefficient cost plane for the hot path.
pub fn compute_jwuniward_ac_costs(
    dct_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
    sigma: f64,
) -> Vec<f64> {
    compute_jwuniward_costs_impl(dct_blocks, width_blocks, height_blocks, sigma, 1)
}

fn compute_jwuniward_costs_impl(
    dct_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
    sigma: f64,
    coeff_start: usize,
) -> Vec<f64> {
    let n_blocks = width_blocks * height_blocks;
    let width = width_blocks * 8;
    let height = height_blocks * 8;
    let coeffs_per_block = 64 - coeff_start;

    let spatial = idct_image(dct_blocks, width_blocks, height_blocks);

    let w_hl = apply_wavelet_2d(&spatial, height, width, &DB8_LO, &DB8_HI);
    let w_lh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_LO);
    let w_hh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_HI);
    let inv_wavelets = [
        inverse_abs_denominator(w_hl, sigma),
        inverse_abs_denominator(w_lh, sigma),
        inverse_abs_denominator(w_hh, sigma),
    ];

    let delta_w = &*DELTA_WAVELETS;

    let mut costs = vec![0.0f64; n_blocks * coeffs_per_block];

    costs
        .par_chunks_mut(width_blocks * coeffs_per_block)
        .enumerate()
        .for_each(|(br, chunk)| {
            let px_r = br * 8;
            for bc in 0..width_blocks {
                let px_c = bc * 8;
                let block_base = bc * coeffs_per_block;

                let r_lo = px_r.saturating_sub(WAVELET_PAD);
                let r_hi = (px_r + 8 + WAVELET_PAD).min(height);
                let r_ctx_off = WAVELET_PAD.saturating_sub(px_r);

                let c_lo = px_c.saturating_sub(WAVELET_PAD);
                let c_hi = (px_c + 8 + WAVELET_PAD).min(width);
                let c_ctx_off = WAVELET_PAD.saturating_sub(px_c);

                for coeff_idx in coeff_start..64usize {
                    let dw = &delta_w[coeff_idx];
                    let mut cost = 0.0f64;

                    for k in 0..3usize {
                        let inv_cover = &inv_wavelets[k];
                        let dw_k = &dw[k];

                        let mut r_ctx = r_ctx_off;
                        for r_img in r_lo..r_hi {
                            let row_inv = &inv_cover[r_img * width..r_img * width + width];
                            let dw_row = &dw_k
                                [r_ctx * DELTA_CTX_SIZE..r_ctx * DELTA_CTX_SIZE + DELTA_CTX_SIZE];
                            let mut c_ctx = c_ctx_off;
                            for c_img in c_lo..c_hi {
                                let delta = dw_row[c_ctx];
                                if delta != 0.0 {
                                    cost += delta * row_inv[c_img];
                                }
                                c_ctx += 1;
                            }
                            r_ctx += 1;
                        }
                    }

                    chunk[block_base + coeff_idx - coeff_start] = cost;
                }
            }
        });

    costs
}

fn inverse_abs_denominator(mut wavelet: Vec<f64>, sigma: f64) -> Vec<f64> {
    wavelet.par_iter_mut().for_each(|value| {
        *value = 1.0 / (sigma + value.abs());
    });
    wavelet
}
