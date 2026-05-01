use mozjpeg_sys::*;
use std::fs;

fn main() {
    let data = fs::read("image.jpg").expect("Impossibile leggere il file");

    unsafe {
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

        let num_components = cinfo.num_components as usize;
        println!("Componenti: {}", num_components);

        for ci in 0..num_components {
            let comp = &*cinfo.comp_info.add(ci);
            let width_in_blocks  = comp.width_in_blocks as usize;
            let height_in_blocks = comp.height_in_blocks as usize;

            println!(
                "Componente {}: {}x{} blocchi",
                ci, width_in_blocks, height_in_blocks
            );

            let virt_array = *coef_arrays.add(ci);

            for block_row in 0..height_in_blocks {
                // access_virt_barray ritorna JBLOCKARRAY = *mut JBLOCKROW
                let row_array: JBLOCKARRAY = ((*cinfo.common.mem).access_virt_barray.unwrap())(
                    &mut cinfo.common,
                    virt_array,
                    block_row as JDIMENSION,
                    1,
                    false as boolean,
                );

                // row_array[0] è la JBLOCKROW per questa riga (richiediamo 1 riga)
                // JBLOCKROW = *mut JBLOCK = *mut [JCOEF; 64]
                let row: JBLOCKROW = *row_array;

                for block_col in 0..width_in_blocks {
                    // row[block_col] è il JBLOCK = [JCOEF; 64]
                    let block: &JBLOCK = &*row.add(block_col);

                    let dc = block[0]; // DC è il primo coefficiente
                    let ac = &block[1..]; // AC[0..62] sono i restanti 63 coefficenti

                    print!("DC={:5} ", dc);
                }
                println!();
            }
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
    }
}