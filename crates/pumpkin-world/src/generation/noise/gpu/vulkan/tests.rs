use super::*;
use crate::chunk_system::StagedChunkEnum;
use crate::generation::{generator::WorldGenerator, get_world_gen, proto_chunk::ProtoChunk};
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;

struct CpuReferenceGuard;
impl Drop for CpuReferenceGuard {
    fn drop(&mut self) {
        CHECKING_CPU.set(false);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU with shaderFloat64"]
fn live_vulkan_noise_stage_matches_cpu() {
    initialize(WorldgenConfig {
        backend: WorldgenBackend::Vulkan,
        device: 0,
        streams: 0,
    });
    assert!(engine().is_some(), "Vulkan backend failed to initialize");
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
        "expected Vulkan compute dispatches"
    );
    assert!(
        HITS.load(Ordering::Relaxed) > hits_before,
        "expected prefetched Vulkan noise hits"
    );
}
