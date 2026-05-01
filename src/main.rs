mod uniward;
mod stc;

use mozjpeg_sys::*;
use std::fs;
use stc::{StcParams, bytes_to_bits, bits_to_bytes, stc_embed, stc_extract};
use uniward::compute_jwuniward_costs;

// ─── Lettura coefficienti DCT da JPEG ────────────────────────────────────────

struct JpegDct {
    blocks: Vec<i16>,        // coefficienti DCT, interleaved a blocchi 64
    width_blocks: usize,
    height_blocks: usize,
    qt: Vec<u16>,            // tabella di quantizzazione (64 valori, per Y)
}

unsafe fn read_jpeg_dct(data: &[u8]) -> JpegDct {
    let mut cinfo: jpeg_decompress_struct = std::mem::zeroed();
    let mut err: jpeg_error_mgr = std::mem::zeroed();

    cinfo.common.err = jpeg_std_error(&mut err);
    jpeg_CreateDecompress(
        &mut cinfo,
        JPEG_LIB_VERSION,
        std::mem::size_of::<jpeg_decompress_struct>(),
    );

    jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as u64);
    jpeg_read_header(&mut cinfo, true as i32);

    let coef_arrays = jpeg_read_coefficients(&mut cinfo);

    // Legge solo componente Y (luminanza) — la più grande e usata da J-UNIWARD
    let comp = &*cinfo.comp_info;
    let width_blocks = comp.width_in_blocks as usize;
    let height_blocks = comp.height_in_blocks as usize;
    let n_blocks = width_blocks * height_blocks;

    let mut blocks = vec![0i16; n_blocks * 64];

    let virt_array = *coef_arrays;
    for block_row in 0..height_blocks {
        let row_array: JBLOCKARRAY = ((*cinfo.common.mem).access_virt_barray.unwrap())(
            &mut cinfo.common,
            virt_array,
            block_row as JDIMENSION,
            1,
            false as boolean,
        );
        let row: JBLOCKROW = *row_array;
        for block_col in 0..width_blocks {
            let block: &JBLOCK = &*row.add(block_col);
            let dst_start = (block_row * width_blocks + block_col) * 64;
            for i in 0..64 {
                blocks[dst_start + i] = block[i];
            }
        }
    }

    // Legge tabella quantizzazione per Y
    let qt_ptr = (*cinfo.comp_info).quant_table;
    let mut qt = vec![1u16; 64];
    if !qt_ptr.is_null() {
        for i in 0..64 {
            qt[i] = (*qt_ptr).quantval[i];
        }
    }

    jpeg_finish_decompress(&mut cinfo);
    jpeg_destroy_decompress(&mut cinfo);

    JpegDct { blocks, width_blocks, height_blocks, qt }
}

// ─── Scrittura JPEG con coefficienti modificati ───────────────────────────────

unsafe fn write_jpeg_dct(
    original_data: &[u8],
    modified_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
) -> Vec<u8> {
    // Decomprime l'originale per copiare la struttura
    let mut src_info: jpeg_decompress_struct = std::mem::zeroed();
    let mut src_err: jpeg_error_mgr = std::mem::zeroed();
    src_info.common.err = jpeg_std_error(&mut src_err);
    jpeg_CreateDecompress(
        &mut src_info,
        JPEG_LIB_VERSION,
        std::mem::size_of::<jpeg_decompress_struct>(),
    );
    jpeg_mem_src(&mut src_info, original_data.as_ptr(), original_data.len() as u64);
    jpeg_read_header(&mut src_info, true as i32);
    let coef_arrays = jpeg_read_coefficients(&mut src_info);

    // Scrive i nuovi coefficienti
    let virt_array = *coef_arrays;
    for block_row in 0..height_blocks {
        let row_array: JBLOCKARRAY = ((*src_info.common.mem).access_virt_barray.unwrap())(
            &mut src_info.common,
            virt_array,
            block_row as JDIMENSION,
            1,
            true as boolean, // writable!
        );
        let row: JBLOCKROW = *row_array;
        for block_col in 0..width_blocks {
            let block: &mut JBLOCK = &mut *row.add(block_col);
            let src_start = (block_row * width_blocks + block_col) * 64;
            for i in 0..64 {
                block[i] = modified_blocks[src_start + i];
            }
        }
    }

    // Comprime in memoria
    let mut dst_ptr: *mut u8 = std::ptr::null_mut();
    let mut dst_size: u64 = 0;

    let mut dst_info: jpeg_compress_struct = std::mem::zeroed();
    let mut dst_err: jpeg_error_mgr = std::mem::zeroed();
    dst_info.common.err = jpeg_std_error(&mut dst_err);
    jpeg_CreateCompress(
        &mut dst_info,
        JPEG_LIB_VERSION,
        std::mem::size_of::<jpeg_compress_struct>(),
    );
    jpeg_mem_dest(&mut dst_info, &mut dst_ptr, &mut dst_size);

    jpeg_copy_critical_parameters(&src_info, &mut dst_info);
    jpeg_write_coefficients(&mut dst_info, coef_arrays);

    jpeg_finish_compress(&mut dst_info);
    jpeg_destroy_compress(&mut dst_info);
    jpeg_finish_decompress(&mut src_info);
    jpeg_destroy_decompress(&mut src_info);

    // Copia il buffer in un Vec Rust
    let result = std::slice::from_raw_parts(dst_ptr, dst_size as usize).to_vec();
    libc_free(dst_ptr as *mut std::ffi::c_void);
    result
}

unsafe extern "C" {
    fn free(ptr: *mut std::ffi::c_void);
}
fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe { free(ptr) }
}

// ─── Pipeline principale ──────────────────────────────────────────────────────

fn main() {
    println!("=== J-UNIWARD + STC Steganography ===\n");

    // 1. Leggi immagine cover
    let data = fs::read("image.jpg").expect("Metti image.jpg nella directory corrente");
    println!("Cover JPEG letto: {} bytes", data.len());

    // 2. Estrai coefficienti DCT
    let jpeg = unsafe { read_jpeg_dct(&data) };
    let n_blocks = jpeg.width_blocks * jpeg.height_blocks;
    let n_coeffs = n_blocks * 64;
    println!("Dimensione: {}x{} blocchi, {} coefficienti DCT totali",
        jpeg.width_blocks, jpeg.height_blocks, n_coeffs);

    // 3. Calcola costi J-UNIWARD
    println!("\n[1/4] Calcolo costi J-UNIWARD...");
    let sigma = 1e-10;
    let costs = compute_jwuniward_costs(
        &jpeg.blocks,
        jpeg.width_blocks,
        jpeg.height_blocks,
        sigma,
    );

    // Statistiche costi
    let (min_c, max_c, mean_c) = cost_stats(&costs);
    println!("  Costo min: {:.4}, max: {:.4}, medio: {:.4}", min_c, max_c, mean_c);

    // 4. Prepara messaggio
    let message = "Ciao! Messaggio segreto nascosto con J-UNIWARD + STC.";
    println!("\n[2/4] Messaggio: \"{}\"", message);
    let message_bits = bytes_to_bits(message.as_bytes());
    println!("  Lunghezza: {} bytes = {} bit", message.len(), message_bits.len());

    // Verifica capacità (regola pratica: max ~0.4 bpnzAC)
    // Contiamo coefficienti AC non-zero
    let nz_ac: usize = jpeg.blocks.iter().enumerate()
        .filter(|&(ref i, &v)| i % 64 != 0 && v != 0)
        .count();
    let max_payload = (nz_ac as f64 * 0.4) as usize;
    println!("  Coefficienti AC non-zero: {}", nz_ac);
    println!("  Capacità massima sicura (0.4 bpnzAC): {} bit", max_payload);

    if message_bits.len() > max_payload {
        eprintln!("ERRORE: messaggio troppo lungo per embedding sicuro");
        return;
    }

    // 5. Estrai LSB dei coefficienti cover
    // Usiamo tutti i coefficienti AC (escludiamo DC a indice 0 di ogni blocco)
    // e li serializziamo in ordine zig-zag come da standard JPEG
    let cover_bits: Vec<u8> = jpeg.blocks.iter().enumerate()
        .filter(|(i, _)| i % 64 != 0) // escludi DC
        .map(|(_, &v)| (v.unsigned_abs() & 1) as u8)
        .collect();

    let ac_costs: Vec<f64> = costs.iter().enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .zip(jpeg.blocks.iter().enumerate().filter(|(i, _)| i % 64 != 0).map(|(_, v)| v))
        .map(|((_, &c), &v)| {
            // Non modificare coefficienti 0 (non si può) o ±1 (diventerebbero 0,
            // alterando la struttura RLE del JPEG e causando errori nel round-trip)
            if v == 0 || v == 1 || v == -1 {
                f64::INFINITY
            } else {
                c
            }
        })
        .collect();

    // 6. STC Embedding
    println!("\n[3/4] STC Embedding...");
    let params = StcParams::new(7); // h=7 → 128 stati trellis
    println!("  Parametri: h_height={}, num_states={}", params.h_height, params.num_states());

    let stego_bits = match stc_embed(&cover_bits, &ac_costs, &message_bits, &params) {
        Ok(bits) => {
            let n_changes: usize = cover_bits.iter().zip(bits.iter())
                .filter(|(a, b)| a != b).count();
            println!("  Modifiche effettuate: {} / {} ({:.2}%)",
                n_changes, cover_bits.len(),
                100.0 * n_changes as f64 / cover_bits.len() as f64);
            bits
        }
        Err(e) => {
            eprintln!("Embedding fallito: {}", e);
            return;
        }
    };

    // 7. Ricostruisce blocchi DCT stego
    let mut stego_blocks = jpeg.blocks.clone();
    let mut stego_idx = 0usize;
    for (coeff_idx, coeff) in stego_blocks.iter_mut().enumerate() {
        if coeff_idx % 64 == 0 {
            continue; // salta DC
        }
        let orig_bit = (coeff.unsigned_abs() & 1) as u8;
        let new_bit = stego_bits[stego_idx];
        stego_idx += 1;

        if orig_bit != new_bit {
            // Flip LSB mantenendo il segno
            if *coeff > 0 {
                *coeff ^= 1;
            } else if *coeff < 0 {
                // Per negativi: |coeff| XOR 1, poi rimetti segno
                let abs_new = (coeff.unsigned_abs() ^ 1) as i16;
                *coeff = -abs_new;
            }
        }
    }

    // 8. Scrivi JPEG stego
    println!("\n[4/4] Scrittura stego.jpg...");
    let stego_data = unsafe {
        write_jpeg_dct(&data, &stego_blocks, jpeg.width_blocks, jpeg.height_blocks)
    };
    fs::write("stego.jpg", &stego_data).expect("Impossibile scrivere stego.jpg");
    println!("  stego.jpg scritto: {} bytes (originale: {} bytes)",
        stego_data.len(), data.len());

    // ── Verifica: estrai il messaggio da stego ──
    println!("\n=== VERIFICA DECODING ===");

    let stego_jpeg = unsafe { read_jpeg_dct(&stego_data) };
    let extracted_bits: Vec<u8> = stego_jpeg.blocks.iter().enumerate()
        .filter(|(i, _)| i % 64 != 0)
        .map(|(_, &v)| (v.unsigned_abs() & 1) as u8)
        .collect();

    let recovered_bits = stc_extract(&extracted_bits, message_bits.len(), &params);
    let recovered_bytes = bits_to_bytes(&recovered_bits);
    let recovered_text = String::from_utf8_lossy(&recovered_bytes);

    println!("Messaggio recuperato: \"{}\"", recovered_text);

    let bit_errors: usize = message_bits.iter()
        .zip(recovered_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!("Bit errati: {} / {}", bit_errors, message_bits.len());

    if bit_errors == 0 {
        println!("\n✓ Embedding e decoding completati con successo!");
    } else {
        println!("\n✗ Attenzione: {} bit errati nel messaggio recuperato", bit_errors);
    }
}

fn cost_stats(costs: &[f64]) -> (f64, f64, f64) {
    let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = costs.iter().sum::<f64>() / costs.len() as f64;
    (min, max, mean)
}