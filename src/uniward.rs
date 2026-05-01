/// J-UNIWARD: JPEG Universal Wavelet Relative Distortion
///
/// Calcola il costo di embedding per ogni coefficiente DCT.
/// Costi bassi = zone texturizzate (sicure da modificare)
/// Costi alti  = zone lisce (modifiche visibili)

/// Filtri wavelet di Daubechies 8-tap (DB8) nelle 3 direzioni:
/// - HH (diagonale)
/// - HL (orizzontale)
/// - LH (verticale)
/// Ogni filtro è separabile: applicato prima su righe poi su colonne.
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

/// Applica convoluzione 1D con padding "reflect" (bordi riflessi)
fn convolve1d(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let k = kernel.len();
    let pad = k / 2;
    let mut out = vec![0.0f64; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            // Indice con reflect padding
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

/// Applica filtro wavelet separabile 2D a una matrice rows×cols
/// Ritorna la matrice filtrata linearizzata (row-major)
fn apply_wavelet_2d(
    img: &[f64],
    rows: usize,
    cols: usize,
    row_filter: &[f64],
    col_filter: &[f64],
) -> Vec<f64> {
    // Prima passo: filtra ogni riga
    let mut tmp = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let row_slice = &img[r * cols..(r + 1) * cols];
        let filtered = convolve1d(row_slice, col_filter);
        tmp[r * cols..(r + 1) * cols].copy_from_slice(&filtered);
    }

    // Secondo passo: filtra ogni colonna
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

/// Ricostruisce l'immagine spaziale dai blocchi DCT.
/// Usa IDCT 8x8 manuale (formula diretta).
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
                            sum += cu * cv * coef
                                * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                                * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
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

/// Calcola i costi J-UNIWARD per ogni coefficiente DCT modificabile.
///
/// Sfrutta la linearità della convoluzione: W(cover + delta) - W(cover) = W(delta).
/// I delta wavelet per i 64 coefficienti DCT vengono precalcolati una volta sola,
/// evitando di chiamare apply_wavelet_2d per ogni blocco.
///
/// Ritorna un vettore di costi, uno per coefficiente DCT (n_blocks * 64).
pub fn compute_jwuniward_costs(
    dct_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
    sigma: f64, // stabilizzatore numerico, tipicamente 1e-10
) -> Vec<f64> {
    let n_blocks = width_blocks * height_blocks;
    let width = width_blocks * 8;
    let height = height_blocks * 8;

    // 1. Ricostruisce immagine spaziale dalla cover
    let spatial = idct_image(dct_blocks, width_blocks, height_blocks);

    // 2. Precalcola i 3 residui wavelet della cover (HL, LH, HH) — 3 chiamate totali
    let w_hl = apply_wavelet_2d(&spatial, height, width, &DB8_LO, &DB8_HI);
    let w_lh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_LO);
    let w_hh = apply_wavelet_2d(&spatial, height, width, &DB8_HI, &DB8_HI);
    let wavelets_cover: [&Vec<f64>; 3] = [&w_hl, &w_lh, &w_hh];

    // 3. Precalcola le funzioni base DCT 8x8
    let basis = precompute_dct_basis();

    // 4. Precalcola i delta wavelet per ognuno dei 64 coefficienti DCT.
    //    Per la linearità della convoluzione: W(cover+delta) - W(cover) = W(delta).
    //    Il delta è la funzione base DCT/4 su un blocco 8x8 immerso in un contesto zero.
    //    Area di influenza: il filtro da 8 tap estende il supporto di pad=8 pixel per lato.
    //    → contesto: (8 + 2*pad) × (8 + 2*pad) = 24×24 pixel.
    //    Questo sostituisce 691.200 chiamate a apply_wavelet_2d con sole 192.
    let pad = DB8_LO.len(); // 8
    let ctx_size = 8 + 2 * pad; // 24

    // delta_w[coeff_idx][subbanda] = vettore ctx_size×ctx_size con la risposta wavelet del delta
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

    // 5. Calcola i costi: per ogni blocco e coefficiente, accumula la variazione relativa
    //    usando i delta wavelet precalcolati.
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

                    // Il contesto è centrato sul blocco:
                    // ctx[r_ctx][c_ctx] corrisponde all'immagine in
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

/// Precomputa le 64 funzioni base DCT 8x8
/// basis[coeff_idx][pixel_idx] = valore della funzione base
fn precompute_dct_basis() -> Vec<Vec<f64>> {
    let mut basis = vec![vec![0.0f64; 64]; 64];
    for v in 0..8usize {
        for u in 0..8usize {
            let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            for y in 0..8usize {
                for x in 0..8usize {
                    basis[v * 8 + u][y * 8 + x] = cu * cv
                        * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                        * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                }
            }
        }
    }
    basis
}