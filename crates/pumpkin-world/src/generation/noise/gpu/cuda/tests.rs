use super::*;
use crate::{
    chunk_system::StagedChunkEnum,
    generation::{generator::WorldGenerator, get_world_gen, proto_chunk::ProtoChunk},
};
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;

#[test]
fn fused_parameters_distinguish_signed_zero() {
    let positive = Operation {
        kind: 2,
        a: 0.0,
        b: 0.0,
    };
    let negative = Operation {
        a: -0.0,
        ..positive
    };
    assert!(positive != negative);
}

#[test]
fn overworld_router_has_fusable_noise_branches_and_keeps_cache_boundaries() {
    use crate::generation::{
        GlobalRandomConfig,
        noise::router::{
            chunk_density_function::ChunkNoiseFunctionBuilderOptions,
            chunk_noise_router::ChunkNoiseRouter, proto_noise_router::ProtoNoiseRouters,
        },
    };
    use pumpkin_data::noise_router::OVERWORLD_BASE_NOISE_ROUTER;
    use std::collections::BTreeSet;
    let proto = ProtoNoiseRouters::generate(
        &OVERWORLD_BASE_NOISE_ROUTER,
        &GlobalRandomConfig::new(13579, false),
    );
    let options = ChunkNoiseFunctionBuilderOptions::new(vec![], vec![], None);
    let mut router = ChunkNoiseRouter::generate(&proto.noise, &options);
    let volume = DensityVolume::with_block_step(16, 384, 16, -112, -64, 144);
    let planes = grid::candidate_count(router.final_density_stack(), &volume);
    assert!(
        planes >= 3,
        "expected three or more eligible planes, found {planes}"
    );
    assert_eq!(
        grid::blended_grid(router.final_density_stack(), &volume),
        Some(DensityVolume::new(5, 49, 5, -112, -64, 144, 4, 8, 4)),
    );
    assert!(grid::fused_candidate_count(router.final_density_stack(), &volume) > 0);
    let (selections, _, planes) =
        grid::selection_counts(router.final_density_stack(), &volume, false);
    let (hybrid_selections, hybrid_pairs, hybrid_planes) =
        grid::selection_counts(router.final_density_stack(), &volume, true);
    assert!(selections >= 2);
    assert_eq!(hybrid_selections, selections);
    assert!(hybrid_pairs > 0);
    assert_eq!(hybrid_planes, planes + 1);
    let stack = router.component_stack_mut();
    let mut leaf_ids = BTreeSet::new();
    let mut fused = 0;
    for (i, node) in stack.iter().enumerate() {
        let plan = branch(&stack[..=i]);
        if matches!(node, Component::Chunk(_)) {
            assert!(plan.is_none());
        }
        if let Some((leaf, tail)) = plan {
            leaf_ids.insert(leaf.data.noise_id.id);
            fused += usize::from(tail.count > 0);
        }
    }
    assert!(leaf_ids.len() > 1 && fused > 0);
}

/// Run alone in a CUDA verification mode. An entirely CPU run cannot pass.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn live_cuda_noise_stage_matches_cpu() {
    initialize(pumpkin_config::world::WorldgenConfig {
        backend: pumpkin_config::world::WorldgenBackend::Cuda,
        device: 0,
    });
    let engine = engine().expect("CUDA backend");
    let before = cuda_worldgen_stats();
    for (dimension, coords) in [
        (Dimension::OVERWORLD, &[(-7, 9), (11, -13)][..]),
        (Dimension::THE_NETHER, &[(7, -4)][..]),
    ] {
        let world = get_world_gen(Seed(13579), dimension, false, Vec::new(), String::new());
        let WorldGenerator::Noise(generator) = &*world else {
            panic!("expected noise generator")
        };
        let before_dimension = cuda_worldgen_stats();
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
        if engine.cache_early_plan && coords.len() > 1 {
            let after_dimension = cuda_worldgen_stats();
            assert_eq!(
                after_dimension.cached_plan_builds - before_dimension.cached_plan_builds,
                1
            );
            assert_eq!(
                after_dimension.cached_early_grid_hits - before_dimension.cached_early_grid_hits,
                coords.len() as u64
            );
        }
    }
    let after = cuda_worldgen_stats();
    assert!(after.available && after.fused_volumes > before.fused_volumes);
    if engine.grid {
        assert!(after.grid_batches > before.grid_batches);
        assert!(after.prefetched_hits > before.prefetched_hits);
    }
    if engine.terrain {
        assert!(after.blended_hits > before.blended_hits);
    }
    if engine.fuse_density {
        assert!(after.smear_table_batches > before.smear_table_batches);
    }
    if engine.select_density {
        assert!(after.selected_density_hits > before.selected_density_hits);
    }
    if engine.reserve_binary {
        assert!(after.fused_binary_hits > before.fused_binary_hits);
    }
    if engine.prefetch_early {
        assert!(after.early_grid_hits > before.early_grid_hits);
    }
    if engine.pipeline {
        let first_reads = after.pipeline_ready_at_first_use + after.pipeline_waits;
        assert!(first_reads > 0);
        pipeline_readback_preserves_chunk_and_plane_order(engine);
    }
}

fn pipeline_readback_preserves_chunk_and_plane_order(engine: &Engine) {
    use crate::generation::noise::router::{
        chunk_density_function::ChunkNoiseFunctionBuilderOptions,
        chunk_noise_router::ChunkNoiseRouter,
    };

    let world = get_world_gen(
        Seed(13579),
        Dimension::OVERWORLD,
        false,
        Vec::new(),
        String::new(),
    );
    let WorldGenerator::Noise(generator) = &*world else {
        panic!("expected a noise generator")
    };
    let options = ChunkNoiseFunctionBuilderOptions::new(vec![], vec![], None);
    let mut router =
        ChunkNoiseRouter::generate_without_veins(&generator.base_router.noise, &options);
    let stack = router.component_stack_mut();
    let mut ids = std::collections::BTreeSet::new();
    let mut indices: Vec<_> = (0..stack.len())
        .filter(|&i| {
            branch(&stack[..=i])
                .is_some_and(|(noise, tail)| tail.count > 0 && ids.insert(noise.data.noise_id.id))
        })
        .take(3)
        .collect();
    assert_eq!(indices.len(), 3);
    if engine.fuse_density {
        let binary = (0..stack.len())
            .find(|&i| fusion::branch(&stack[..=i]).is_some())
            .expect("select a pair of noise branches");
        indices.push(binary);
    }
    if engine.select_density {
        let selection = (0..stack.len())
            .find(|&i| selection::branch(&stack[..=i]).is_some())
            .expect("select an interval branch");
        indices.push(selection);
    }
    let mut replies = Vec::new();
    let mut jobs = Vec::new();
    for (x, z) in [(-112, 144), (29_989_984, -29_989_984)] {
        let volume = DensityVolume::new(5, 49, 5, x, -64, z, 4, 8, 4);
        let planes = indices
            .iter()
            .map(|&i| {
                if engine.select_density
                    && let Some(selection) = selection::branch(&stack[..=i])
                {
                    return selection.upload(engine).expect("upload interval selection");
                }
                if engine.fuse_density
                    && let Some(pair) = fusion::branch(&stack[..=i])
                {
                    return grid::Plane {
                        layers: resident_layers(engine, pair.left.0).expect("upload left noise"),
                        tail: pair.left.1,
                        companion: Some(fusion::Operand {
                            layers: resident_layers(engine, pair.right.0)
                                .expect("upload right noise"),
                            tail: pair.right.1,
                            operation: pair.operation,
                        }),
                        selection: None,
                    };
                }
                let (noise, tail) = branch(&stack[..=i]).expect("selected a noise branch");
                let layers = resident_layers(engine, noise).expect("uploaded layers");
                grid::Plane {
                    layers,
                    tail,
                    companion: None,
                    selection: None,
                }
            })
            .collect();
        let (reply, rx) = bounded(1);
        jobs.push(grid::Job {
            volume,
            plan: Arc::new(grid::Plan {
                volume,
                indices: Vec::new(),
                fused_mask: 0,
                selection_mask: 0,
                planes,
                blended: None,
            }),
            reply,
            cancel: None,
        });
        replies.push(rx);
    }
    let module = engine
        .context
        .load_module(Ptx::from_binary(
            include_bytes!(concat!(env!("OUT_DIR"), "/worldgen.fatbin")).to_vec(),
        ))
        .expect("load worldgen PTX");
    let kernel = module
        .load_function(if engine.select_density {
            "fill_terrain_select"
        } else if engine.fuse_density {
            "fill_terrain_fused"
        } else {
            "fill_grid_pipeline"
        })
        .expect("load pipeline kernel");
    let mut lane = worker::Lane::new(
        engine.context.new_stream().expect("CUDA stream"),
        kernel,
        false,
        engine.fuse_density,
    );
    if engine.select_density {
        lane.enable_selection();
    }
    if engine.fuse_density {
        lane.enable_smear_table(
            module
                .load_function("prepare_terrain_y")
                .expect("load smear kernel"),
        );
    }
    lane.dispatch_grid(&jobs, true, engine.fuse_density)
        .expect("launch two different origins");
    CHECKING_CPU.set(true);
    let _guard = CpuReferenceGuard;
    for (job, rx) in jobs.iter().zip(replies) {
        let result = rx.recv().expect("read back the completed batch");
        let len = job.volume.size();
        let mut cpu = vec![0.0f32; len];
        for (plane, &i) in indices.iter().enumerate() {
            Component::sample_volume_from_stack(&mut stack[..=i], &mut cpu, &job.volume);
            let start = result.offset + plane * len;
            assert!(
                cpu.iter()
                    .zip(&result.values[start..start + len])
                    .all(|(expected, gpu)| expected.to_bits() == gpu.to_bits()),
                "mismatch for plane {plane} at {:?}",
                job.volume
            );
        }
    }
}
