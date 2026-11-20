use super::surface_biomes;
use crate::chunk_system::StagedChunkEnum;
use crate::generation::{
    generator::{VanillaGenerator, WorldGenerator},
    get_world_gen,
    proto_chunk::ProtoChunk,
};
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;

#[test]
fn surface_threshold_cache_matches_reference() {
    for (dimension, seed, x, z) in [
        (Dimension::OVERWORLD, 42, 0, 0),
        (Dimension::OVERWORLD, 42, -7, 9),
        (Dimension::OVERWORLD, 0, -595, 544),
        (Dimension::THE_NETHER, 42, 7, -4),
        (Dimension::THE_END, 42, 18, 21),
    ] {
        let world = get_world_gen(Seed(seed), dimension, false, Vec::new(), String::new());
        let WorldGenerator::Noise(generator) = world.as_ref() else {
            unreachable!();
        };
        let biomes = surface_biomes(&world, x, z);
        let generate = |cache_thresholds| {
            let mut chunk = noise_chunk(&world, generator, x, z);
            chunk.step_to_surface_with_threshold_cache(generator, &biomes, cache_thresholds);
            chunk
        };
        let context = format!("{} chunk ({x}, {z})", generator.dimension.minecraft_name);
        assert_same_surface(&generate(false), &generate(true), &context);
    }
}

fn noise_chunk(world: &WorldGenerator, generator: &VanillaGenerator, x: i32, z: i32) -> ProtoChunk {
    let mut chunk = ProtoChunk::new(x, z, world);
    chunk.step_to_biomes(generator);
    chunk.stage = StagedChunkEnum::StructureReferences;
    chunk.step_to_noise_with_solid_fast_path(generator, true);
    chunk
}

fn assert_same_surface(reference: &ProtoChunk, generated: &ProtoChunk, context: &str) {
    assert!(
        reference.flat_block_map == generated.flat_block_map,
        "{context} blocks"
    );
    assert_eq!(
        reference.flat_surface_height_map, generated.flat_surface_height_map,
        "{context} surface heightmap"
    );
    assert_eq!(
        reference.flat_ocean_floor_height_map, generated.flat_ocean_floor_height_map,
        "{context} ocean floor heightmap"
    );
    assert_eq!(
        reference.flat_motion_blocking_height_map, generated.flat_motion_blocking_height_map,
        "{context} motion blocking heightmap"
    );
    assert_eq!(
        reference.flat_motion_blocking_no_leaves_height_map,
        generated.flat_motion_blocking_no_leaves_height_map,
        "{context} no-leaves heightmap"
    );
}
