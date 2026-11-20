use std::sync::{Arc, OnceLock, atomic::Ordering};

use cudarc::driver::{CudaSlice, DeviceRepr};

use super::{
    super::super::router::density_function::noise::InterpolatedNoiseSampler, Engine, TABLE_BYTES,
    Tail, abi::CudaLayer, disable,
};

pub type BlendedResident = OnceLock<Option<Arc<BlendedLayers>>>;

pub struct BlendedLayers {
    pub(super) layers: CudaSlice<CudaLayer>,
    pub(super) smear: CudaSlice<f64>,
    pub(super) main: u32,
    pub(super) min: u32,
}

pub(super) struct Plane {
    pub layers: Arc<BlendedLayers>,
    pub tail: Tail,
}

#[derive(Default)]
#[repr(C)]
pub(super) struct Descriptor {
    pub layers: u64,
    pub smear: u64,
    pub tail: Tail,
    pub main: u32,
    pub min: u32,
    pub total: u32,
}

const _: () = assert!(std::mem::size_of::<Descriptor>() == 128);
// SAFETY: Initialized scalar fields matching the CUDA BlendedDescriptor.
unsafe impl DeviceRepr for Descriptor {}

pub(super) fn resident(
    engine: &Engine,
    sampler: &InterpolatedNoiseSampler,
) -> Option<Arc<BlendedLayers>> {
    sampler
        .resident
        .cuda
        .get_or_init(|| {
            let stacks = sampler.volume_stacks();
            let main = stacks[0].0.layers.len();
            let min = stacks[1].0.layers.len();
            let mut layers = Vec::new();
            let mut smear = Vec::new();
            for (stack, xz_scale, y_scale) in stacks {
                for layer in &stack.layers {
                    let (noise, scale) = layer.noise.parts();
                    // The smear limit is below one and its quotient must fit a CUDA int.
                    if !scale.is_finite() || scale.abs() <= 1.0 / f64::from(i32::MAX) {
                        return None;
                    }
                    layers.push(CudaLayer {
                        offset_x: noise.offset_x,
                        offset_y: noise.offset_y,
                        offset_z: noise.offset_z,
                        xz_scale: xz_scale * layer.frequency,
                        y_scale: y_scale * layer.frequency,
                        amplitude: layer.amplitude,
                        permutations: *noise.permutations(),
                        padding: [0; 4],
                    });
                    smear.push(scale);
                }
            }
            if main == 0 || min == 0 || main + min == layers.len() || layers.len() * 28 > 48 * 1024
            {
                return None;
            }
            let stream = engine.context.default_stream();
            let upload = (|| {
                let device_layers = stream.clone_htod(&layers)?;
                let device_smear = stream.clone_htod(&smear)?;
                stream.synchronize()?;
                Ok::<_, cudarc::driver::DriverError>(BlendedLayers {
                    layers: device_layers,
                    smear: device_smear,
                    main: main as u32,
                    min: min as u32,
                })
            })();
            match upload {
                Ok(device) => {
                    let bytes = std::mem::size_of_val(layers.as_slice())
                        + std::mem::size_of_val(smear.as_slice());
                    TABLE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
                    Some(Arc::new(device))
                }
                Err(e) => {
                    disable(&engine.disabled, e);
                    None
                }
            }
        })
        .clone()
}
