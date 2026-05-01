use mozjpeg_sys::*;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn bytes_to_human(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;

    while size >= 1024.0 && unit < units.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }

    format!("{:.2} {}", size, units[unit])
}

fn main() {
    let data = fs::read("image.jpg").expect("Failed to read file");
    let out_file = File::create("coefficients.txt").expect("Failed to create output file");
    let mut out = BufWriter::new(out_file);

    let message = "Secret message hidden in DCT coefficients!";
    writeln!(out, "Message: {}", message).unwrap();

    let message_bytes = message.as_bytes();
    let message_len = message_bytes.len();

    writeln!(out, "Message length: {} bytes", message_len).unwrap();

    let message_binary = message_bytes.iter()
        .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect::<Vec<u8>>();

    writeln!(out, "Message in binary ({} bits):", message_binary.len()).unwrap();

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
        writeln!(out, "Components: {}", num_components).unwrap();


        let color_space = match cinfo.jpeg_color_space {
            JCS_GRAYSCALE => "GRAYSCALE",
            JCS_RGB       => "RGB",
            JCS_YCbCr     => "YCbCr",
            JCS_CMYK      => "CMYK",
            JCS_YCCK      => "YCCK",
            _             => "Unknown",
        };

        let components = match cinfo.jpeg_color_space {
            JCS_GRAYSCALE => vec!["Y"],
            JCS_RGB       => vec!["R", "G", "B"],
            JCS_YCbCr     => vec!["Y", "Cb", "Cr"],
            JCS_CMYK      => vec!["C", "M", "Y", "K"],
            JCS_YCCK      => vec!["Y", "Cb", "Cr", "K"],
            _             => vec!["Unknown"],
        };

        writeln!(out, "Color space: {}", color_space).unwrap();
        writeln!(out, "Image width: {}", cinfo.image_width).unwrap();
        writeln!(out, "Image height: {}", cinfo.image_height).unwrap();

        for ci in 0..num_components {
            let comp = &*cinfo.comp_info.add(ci);
            let width_in_blocks  = comp.width_in_blocks as usize;
            let height_in_blocks = comp.height_in_blocks as usize;
            
            if ci == 0 {
                writeln!(out, "Image bits for encoding: {}", bytes_to_human((num_components*width_in_blocks*height_in_blocks) as u64)).unwrap();
            }

            let component_code = components.get(ci).unwrap_or(&"?");

            writeln!(out, "Component {} ({}): {}x{} blocks",
                ci, component_code, width_in_blocks, height_in_blocks).unwrap();

            let virt_array = *coef_arrays.add(ci);

            writeln!(out, "{} DC and AC coefficients:", component_code).unwrap();
            for block_row in 0..height_in_blocks {
                // access_virt_barray returns JBLOCKARRAY = *mut JBLOCKROW
                let row_array: JBLOCKARRAY = ((*cinfo.common.mem).access_virt_barray.unwrap())(
                    &mut cinfo.common,
                    virt_array,
                    block_row as JDIMENSION,
                    1,
                    false as boolean,
                );

                // row_array[0] is the JBLOCKROW for this row (we request 1 row)
                // JBLOCKROW = *mut JBLOCK = *mut [JCOEF; 64]
                let row: JBLOCKROW = *row_array;

                for block_col in 0..width_in_blocks {
                    // row[block_col] is the JBLOCK = [JCOEF; 64]
                    let block: &JBLOCK = &*row.add(block_col);

                    let dc = block[0]; // DC is the first coefficient
                    let _ac = &block[1..]; // AC[0..62] are the remaining 63 coefficients

                    write!(out, "DT({:3},{:3})={:5} ", block_row, block_col, dc).unwrap();
                }
                writeln!(out).unwrap();
            }
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
    }
}