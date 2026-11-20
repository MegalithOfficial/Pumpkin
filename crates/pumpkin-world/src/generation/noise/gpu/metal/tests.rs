use super::abi::{HostLayer, prepare_coord_x, prepare_coord_y, prepare_coord_z};
use pumpkin_util::noise::perlin::{Noise, PerlinNoise, SmearedPerlinNoise};
use pumpkin_util::noise::volume::DensityVolume;
use pumpkin_util::random::xoroshiro128::Xoroshiro;

const GRADIENTS: [[f32; 3]; 16] =
    include!("../../../../../../pumpkin-util/src/noise/perlin_gradients.in");

fn metal_sample_reference(layer: &HostLayer, block_x: i32, block_y: i32, block_z: i32) -> f32 {
    let cx = prepare_coord_x(layer, block_x);
    let cz = prepare_coord_z(layer, block_z);
    let cy = prepare_coord_y(layer, block_y);

    let perm = |i: i32| i32::from(layer.permutations[(i & 255) as usize]);
    let dot = |hash: i32, x: f32, y: f32, z: f32| {
        let g = GRADIENTS[(perm(hash) & 15) as usize];
        (g[0] * x + g[2] * z) + g[1] * y
    };
    let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);

    let xy00 = perm(cx.x0 + cy.floor_y);
    let xy10 = perm(cx.x1 + cy.floor_y);
    let xy01 = perm(cx.x0 + cy.floor_y + 1);
    let xy11 = perm(cx.x1 + cy.floor_y + 1);

    let z0y0 = lerp(
        cx.tx,
        dot(xy00 + cz.floor_z, cx.rx, cy.ry, cz.rz),
        dot(xy10 + cz.floor_z, cx.rx - 1.0, cy.ry, cz.rz),
    );
    let z0y1 = lerp(
        cx.tx,
        dot(xy01 + cz.floor_z, cx.rx, cy.ry - 1.0, cz.rz),
        dot(xy11 + cz.floor_z, cx.rx - 1.0, cy.ry - 1.0, cz.rz),
    );
    let z1y0 = lerp(
        cx.tx,
        dot(xy00 + cz.floor_z + 1, cx.rx, cy.ry, cz.rz - 1.0),
        dot(xy10 + cz.floor_z + 1, cx.rx - 1.0, cy.ry, cz.rz - 1.0),
    );
    let z1y1 = lerp(
        cx.tx,
        dot(xy01 + cz.floor_z + 1, cx.rx, cy.ry - 1.0, cz.rz - 1.0),
        dot(xy11 + cz.floor_z + 1, cx.rx - 1.0, cy.ry - 1.0, cz.rz - 1.0),
    );
    lerp(cz.tz, lerp(cy.ty, z0y0, z0y1), lerp(cy.ty, z1y0, z1y1))
}

#[test]
fn metal_1d_coordinate_tables_match_cpu_perlin_bit_for_bit() {
    let mut rng = Xoroshiro::from_seed(13579);
    let sampler = PerlinNoise::new(&mut rng);
    let volume = DensityVolume::new(5, 17, 5, -112, -64, 144, 4, 8, 4);
    let xz_scale = 0.25;
    let y_scale = 0.125;
    let mut cpu = vec![0.0f32; volume.size()];
    sampler.add_to_volume(&mut cpu, &volume, xz_scale, y_scale, 1.0);

    let layer = HostLayer {
        offset_x: sampler.offset_x,
        offset_y: sampler.offset_y,
        offset_z: sampler.offset_z,
        xz_scale,
        y_scale,
        smear_scale: None,
        permutations: *sampler.permutations(),
    };

    for iz in 0..volume.size_z {
        for ix in 0..volume.size_x {
            for iy in 0..volume.size_y {
                let idx = volume.index_unchecked(ix, iy, iz);
                let metal = metal_sample_reference(
                    &layer,
                    volume.block_x(ix),
                    volume.block_y(iy),
                    volume.block_z(iz),
                );
                assert_eq!(
                    cpu[idx].to_bits(),
                    metal.to_bits(),
                    "regular mismatch at ({ix}, {iy}, {iz})"
                );
            }
        }
    }

    let smeared = SmearedPerlinNoise::new(&mut rng, 0.0625);
    let (inner, smear_scale) = smeared.parts();
    let mut cpu_smeared = vec![0.0f32; volume.size()];
    smeared.add_to_volume(&mut cpu_smeared, &volume, xz_scale, y_scale, 1.0);
    let smeared_layer = HostLayer {
        offset_x: inner.offset_x,
        offset_y: inner.offset_y,
        offset_z: inner.offset_z,
        xz_scale,
        y_scale,
        smear_scale: Some(smear_scale),
        permutations: *inner.permutations(),
    };
    for iz in 0..volume.size_z {
        for ix in 0..volume.size_x {
            for iy in 0..volume.size_y {
                let idx = volume.index_unchecked(ix, iy, iz);
                let metal = metal_sample_reference(
                    &smeared_layer,
                    volume.block_x(ix),
                    volume.block_y(iy),
                    volume.block_z(iz),
                );
                assert_eq!(
                    cpu_smeared[idx].to_bits(),
                    metal.to_bits(),
                    "smeared mismatch at ({ix}, {iy}, {iz})"
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_live {
    use super::super::*;
    use crate::chunk_system::StagedChunkEnum;
    use crate::generation::{generator::WorldGenerator, get_world_gen, proto_chunk::ProtoChunk};
    use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};
    use pumpkin_data::dimension::Dimension;
    use pumpkin_util::world_seed::Seed;
    use std::sync::atomic::Ordering;

    struct CpuReferenceGuard;
    impl Drop for CpuReferenceGuard {
        fn drop(&mut self) {
            CHECKING_CPU.set(false);
        }
    }

    #[test]
    #[ignore = "requires macOS with a Metal GPU"]
    fn live_metal_noise_stage_matches_cpu() {
        initialize(WorldgenConfig {
            backend: WorldgenBackend::Metal,
            device: 0,
            streams: 0,
        });
        assert!(engine().is_some(), "Metal backend failed to initialize");
        let dispatches_before = DISPATCHES.load(Ordering::Relaxed);
        let hits_before = HITS.load(Ordering::Relaxed);

        for (dimension, coords) in [
            (Dimension::OVERWORLD, &[(-7, 9), (11, -13)][..]),
            (Dimension::THE_NETHER, &[(7, -4)][..]),
        ] {
            let world = get_world_gen(Seed(13579), dimension, false, Vec::new(), String::new());
            let WorldGenerator::Noise(generator) = &*world else {
                panic!("expected noise generator");
            };
            for &(cx, cz) in coords {
                let generate = || {
                    let _grid = grid::prefetch_chunk(generator, cx, cz);
                    let mut chunk = ProtoChunk::new(cx, cz, &world);
                    chunk.step_to_biomes(generator);
                    chunk.stage = StagedChunkEnum::StructureReferences;
                    chunk.step_to_noise(generator);
                    chunk
                };
                let cpu = {
                    CHECKING_CPU.set(true);
                    let _guard = CpuReferenceGuard;
                    generate()
                };
                let gpu = generate();
                for z in 0..16 {
                    for x in 0..16 {
                        for y in 0..i32::from(cpu.height()) {
                            assert_eq!(
                                cpu.get_block_state_raw(x, y, z),
                                gpu.get_block_state_raw(x, y, z),
                                "chunk ({cx}, {cz}) local ({x}, {y}, {z})"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            DISPATCHES.load(Ordering::Relaxed) > dispatches_before,
            "expected Metal compute dispatches"
        );
        assert!(
            HITS.load(Ordering::Relaxed) > hits_before,
            "expected prefetched Metal noise hits"
        );
    }
}
