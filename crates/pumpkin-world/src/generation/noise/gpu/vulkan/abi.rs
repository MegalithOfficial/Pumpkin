//! `std430` buffer layouts shared with `worldgen.comp`.

use vulkano::buffer::BufferContents;

pub(super) const MAX_PLANES: usize = 25;
pub(super) const MAX_GRID_CHUNKS: usize = 8;
pub(super) const MAX_BINARY_PAIRS: usize = 4;
pub(super) const MAX_SELECTIONS: usize = 3;
pub(super) const MAX_ALTERNATIVES: usize = 5;
pub(super) const MAX_COLUMN_STATES: usize = 256;

#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub(super) struct VkLayer {
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    pub xz_scale: f64,
    pub y_scale: f64,
    pub amplitude: f32,
    pub _pad0: u32,
    pub permutations: [u32; 64],
}

const _: () = assert!(std::mem::size_of::<VkLayer>() == 304);

impl VkLayer {
    pub(super) fn pack_permutations(bytes: &[u8; 256]) -> [u32; 64] {
        let mut packed = [0u32; 64];
        let (chunks, _) = bytes.as_chunks::<4>();
        for (dst, &chunk) in packed.iter_mut().zip(chunks) {
            *dst = u32::from_le_bytes(chunk);
        }
        packed
    }
}

#[derive(BufferContents, Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub(super) struct Operation {
    pub kind: i32,
    pub a: f32,
    pub b: f32,
}

const _: () = assert!(std::mem::size_of::<Operation>() == 12);

#[derive(BufferContents, Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub(super) struct Tail {
    pub ops: [Operation; 8],
    pub count: i32,
}

const _: () = assert!(std::mem::size_of::<Tail>() == 100);

#[derive(BufferContents, Copy, Clone, Default)]
#[repr(C)]
pub(super) struct PlaneDesc {
    pub layer_offset: u32,
    pub layer_count: u32,
    pub tail: Tail,
}

const _: () = assert!(std::mem::size_of::<PlaneDesc>() == 108);

#[derive(BufferContents, Copy, Clone, Default)]
#[repr(C)]
pub(super) struct BlendedDesc {
    pub layer_offset: u32,
    pub smear_offset: u32,
    pub main_count: u32,
    pub min_count: u32,
    pub total_count: u32,
    pub tail: Tail,
}

const _: () = assert!(std::mem::size_of::<BlendedDesc>() == 120);

#[derive(BufferContents, Copy, Clone, Default)]
#[repr(C)]
pub(super) struct SelectorDesc {
    pub threshold_count: u32,
    pub thresholds: [f32; 4],
    pub alternatives: [u32; MAX_ALTERNATIVES],
}

const _: () = assert!(std::mem::size_of::<SelectorDesc>() == 40);

#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub(super) struct BatchParams {
    pub plane_count: u32,
    pub size_x: i32,
    pub size_y: i32,
    pub size_z: i32,
    pub step_x: i32,
    pub step_y: i32,
    pub step_z: i32,
    pub _pad0: u32,
    pub origins: [i32; MAX_GRID_CHUNKS * 3],
    pub planes: [PlaneDesc; MAX_PLANES],
    pub blended: BlendedDesc,
    pub companions: [PlaneDesc; MAX_BINARY_PAIRS],
    pub binary_ops: [u32; MAX_BINARY_PAIRS],
    pub binary_slots: [u32; MAX_PLANES],
    pub selectors: [SelectorDesc; MAX_SELECTIONS],
    pub selector_slots: [u32; MAX_PLANES],
}

impl Default for BatchParams {
    fn default() -> Self {
        Self {
            plane_count: 0,
            size_x: 0,
            size_y: 0,
            size_z: 0,
            step_x: 0,
            step_y: 0,
            step_z: 0,
            _pad0: 0,
            origins: [0; MAX_GRID_CHUNKS * 3],
            planes: [PlaneDesc::default(); MAX_PLANES],
            blended: BlendedDesc::default(),
            companions: [PlaneDesc::default(); MAX_BINARY_PAIRS],
            binary_ops: [0; MAX_BINARY_PAIRS],
            binary_slots: [0; MAX_PLANES],
            selectors: [SelectorDesc::default(); MAX_SELECTIONS],
            selector_slots: [0; MAX_PLANES],
        }
    }
}
