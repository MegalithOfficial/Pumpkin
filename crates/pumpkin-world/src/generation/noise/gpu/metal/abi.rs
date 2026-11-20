//! Buffer layouts and exact `f64` 1D coordinate precomputation shared with `worldgen.metal`.
//!
//! Apple Metal shaders do not support hardware `f64`. Instead of losing precision, the host
//! precomputes the tiny 1D `X`, `Z`, and `Y` (including blended smear-Y) coordinate tables in
//! exact `f64`, and the Metal kernel runs pure `f32` Perlin interpolation with fast-math disabled.

pub(super) const MAX_PLANES: usize = 25;
pub(super) const MAX_GRID_CHUNKS: usize = 8;
pub(super) const MAX_BINARY_PAIRS: usize = 4;
pub(super) const MAX_SELECTIONS: usize = 3;
pub(super) const MAX_ALTERNATIVES: usize = 5;

/// Host-side `f64` metadata for one Perlin octave used to build 1D coordinate tables.
#[derive(Copy, Clone)]
pub(super) struct HostLayer {
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    pub xz_scale: f64,
    pub y_scale: f64,
    pub smear_scale: Option<f64>,
    pub permutations: [u8; 256],
}

/// GPU-side Perlin octave payload (`f32` amplitude + permutation table, no `f64`).
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct MetalLayer {
    pub amplitude: f32,
    pub permutations: [u8; 256],
}

const _: () = assert!(std::mem::size_of::<MetalLayer>() == 260);

/// Precomputed X-axis lookup for one `(layer, chunk, col_x)`.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct CoordX {
    pub x0: i32,
    pub x1: i32,
    pub rx: f32,
    pub tx: f32,
}

const _: () = assert!(std::mem::size_of::<CoordX>() == 16);

/// Precomputed Z-axis lookup for one `(layer, chunk, col_z)`.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct CoordZ {
    pub floor_z: i32,
    pub rz: f32,
    pub tz: f32,
}

const _: () = assert!(std::mem::size_of::<CoordZ>() == 12);

/// Precomputed Y-axis lookup for one `(layer, y_row, iy)` — used for both regular and smeared octaves.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct CoordY {
    pub floor_y: i32,
    pub ry: f32,
    pub ty: f32,
}

const _: () = assert!(std::mem::size_of::<CoordY>() == 12);

#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub(super) struct Operation {
    pub kind: i32,
    pub a: f32,
    pub b: f32,
}

const _: () = assert!(std::mem::size_of::<Operation>() == 12);

#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub(super) struct Tail {
    pub ops: [Operation; 8],
    pub count: i32,
}

const _: () = assert!(std::mem::size_of::<Tail>() == 100);

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct PlaneDesc {
    pub layer_offset: u32,
    pub layer_count: u32,
    pub tail: Tail,
}

const _: () = assert!(std::mem::size_of::<PlaneDesc>() == 108);

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct BlendedDesc {
    pub layer_offset: u32,
    pub main_count: u32,
    pub min_count: u32,
    pub total_count: u32,
    pub tail: Tail,
}

const _: () = assert!(std::mem::size_of::<BlendedDesc>() == 116);

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct SelectorDesc {
    pub threshold_count: u32,
    pub thresholds: [f32; 4],
    pub alternatives: [u32; MAX_ALTERNATIVES],
}

const _: () = assert!(std::mem::size_of::<SelectorDesc>() == 40);

#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct BatchParams {
    pub plane_count: u32,
    pub chunk_count: u32,
    pub y_row_count: u32,
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    pub y_rows: [u32; MAX_GRID_CHUNKS],
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
            chunk_count: 0,
            y_row_count: 0,
            size_x: 0,
            size_y: 0,
            size_z: 0,
            y_rows: [0; MAX_GRID_CHUNKS],
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

#[inline]
fn wrap_coord(x: f64) -> f64 {
    const ROUND_OFF: f64 = 33_554_432.0;
    const HALF_ROUND_OFF: f64 = f64::from_bits(0x416f_ffff_ffff_ffff);
    if (-HALF_ROUND_OFF..HALF_ROUND_OFF).contains(&x) {
        x
    } else {
        x - (x / ROUND_OFF + 0.5).floor() * ROUND_OFF
    }
}

#[inline]
fn smoothstep(x: f32) -> f32 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

#[inline]
pub(super) fn prepare_coord_x(layer: &HostLayer, block_x: i32) -> CoordX {
    let x = wrap_coord(f64::from(block_x) * layer.xz_scale) + layer.offset_x;
    let fx = x.floor();
    let floor_x = fx as i32;
    let rx = (x - fx) as f32;
    CoordX {
        x0: i32::from(layer.permutations[(floor_x & 255) as usize]),
        x1: i32::from(layer.permutations[((floor_x.wrapping_add(1)) & 255) as usize]),
        rx,
        tx: smoothstep(rx),
    }
}

#[inline]
pub(super) fn prepare_coord_z(layer: &HostLayer, block_z: i32) -> CoordZ {
    let z = wrap_coord(f64::from(block_z) * layer.xz_scale) + layer.offset_z;
    let fz = z.floor();
    let floor_z = fz as i32;
    let rz = (z - fz) as f32;
    CoordZ {
        floor_z,
        rz,
        tz: smoothstep(rz),
    }
}

#[inline]
pub(super) fn prepare_coord_y(layer: &HostLayer, block_y: i32) -> CoordY {
    let original_y = f64::from(block_y) * layer.y_scale;
    let y = wrap_coord(original_y) + layer.offset_y;
    let fy = y.floor();
    let floor_y = fy as i32;
    let relative_y = y - fy;
    layer.smear_scale.map_or_else(
        || {
            let ry = relative_y as f32;
            CoordY {
                floor_y,
                ry,
                ty: smoothstep(ry),
            }
        },
        |scale| {
            let limit = if original_y >= 0.0 && original_y < relative_y {
                original_y
            } else {
                relative_y
            };
            let fudge = (limit / scale + f64::from(1.0e-7f32)).floor() * scale;
            CoordY {
                floor_y,
                ry: (relative_y - fudge) as f32,
                ty: smoothstep(relative_y as f32),
            }
        },
    )
}
