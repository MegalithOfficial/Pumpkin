use crate::chunk_system::StagedChunkEnum;
use crate::generation::{generator::WorldGenerator, get_world_gen, proto_chunk::ProtoChunk};
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;

#[test]
fn noise_solid_fast_path_matches_reference() {
    for (dimension, coords) in [
        (Dimension::OVERWORLD, &[(0, 0), (-7, 9), (29, -13)][..]),
        (Dimension::THE_NETHER, &[(7, -4)][..]),
        (Dimension::THE_END, &[(18, 21)][..]),
    ] {
        let world = get_world_gen(Seed(42), dimension, false, Vec::new(), String::new());
        let WorldGenerator::Noise(generator) = world.as_ref() else {
            unreachable!();
        };
        for &(x, z) in coords {
            let generate = |fast_path| {
                let mut chunk = ProtoChunk::new(x, z, &world);
                chunk.step_to_biomes(generator);
                chunk.stage = StagedChunkEnum::StructureReferences;
                chunk.step_to_noise_with_solid_fast_path(generator, fast_path);
                chunk
            };
            let reference = generate(false);
            let fast = generate(true);
            let context = format!("{} chunk ({x}, {z})", generator.dimension.minecraft_name);
            assert_eq!(
                reference.flat_block_map.len(),
                fast.flat_block_map.len(),
                "{context} block storage length"
            );
            assert!(
                reference.flat_block_map == fast.flat_block_map,
                "{context} blocks"
            );
            assert_eq!(
                reference.flat_surface_height_map, fast.flat_surface_height_map,
                "{context} surface heightmap"
            );
            assert_eq!(
                reference.flat_ocean_floor_height_map, fast.flat_ocean_floor_height_map,
                "{context} ocean floor heightmap"
            );
            assert_eq!(
                reference.flat_motion_blocking_height_map, fast.flat_motion_blocking_height_map,
                "{context} motion blocking heightmap"
            );
            assert_eq!(
                reference.flat_motion_blocking_no_leaves_height_map,
                fast.flat_motion_blocking_no_leaves_height_map,
                "{context} no-leaves heightmap"
            );
        }
    }
}
