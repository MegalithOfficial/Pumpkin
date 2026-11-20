use cudarc::driver::DeviceRepr;

#[repr(C)]
pub(crate) struct CudaLayer {
    pub(crate) offset_x: f64,
    pub(crate) offset_y: f64,
    pub(crate) offset_z: f64,
    pub(crate) xz_scale: f64,
    pub(crate) y_scale: f64,
    pub(crate) amplitude: f32,
    pub(crate) permutations: [u8; 256],
    pub(crate) padding: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<CudaLayer>() == 304);

// SAFETY: The CUDA Layer struct has the same layout and every byte is initialized.
unsafe impl DeviceRepr for CudaLayer {}
