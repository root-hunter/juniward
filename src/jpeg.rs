/// JPEG DCT I/O helpers (mozjpeg-sys wrappers).

use mozjpeg_sys::*;

pub struct JpegDct {
    pub blocks: Vec<i16>,
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub qt: Vec<u16>,
}

pub unsafe fn read_jpeg_dct(data: &[u8]) -> JpegDct {
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
}

pub unsafe fn write_jpeg_dct(
    original_data: &[u8],
    modified_blocks: &[i16],
    width_blocks: usize,
    height_blocks: usize,
) -> Vec<u8> {
    unsafe {
        let mut src_info: jpeg_decompress_struct = std::mem::zeroed();
        let mut src_err: jpeg_error_mgr = std::mem::zeroed();
        src_info.common.err = jpeg_std_error(&mut src_err);
        jpeg_CreateDecompress(
            &mut src_info,
            JPEG_LIB_VERSION,
            std::mem::size_of::<jpeg_decompress_struct>(),
        );
        jpeg_mem_src(
            &mut src_info,
            original_data.as_ptr(),
            original_data.len() as u64,
        );
        jpeg_read_header(&mut src_info, true as i32);
        let coef_arrays = jpeg_read_coefficients(&mut src_info);

        let virt_array = *coef_arrays;
        for block_row in 0..height_blocks {
            let row_array: JBLOCKARRAY = ((*src_info.common.mem).access_virt_barray.unwrap())(
                &mut src_info.common,
                virt_array,
                block_row as JDIMENSION,
                1,
                true as boolean,
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

        let result = std::slice::from_raw_parts(dst_ptr, dst_size as usize).to_vec();
        libc_free(dst_ptr as *mut std::ffi::c_void);
        result
    }
}

unsafe extern "C" {
    fn free(ptr: *mut std::ffi::c_void);
}
fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe { free(ptr) }
}
