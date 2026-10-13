#![allow(clippy::all, clippy::pedantic, clippy::nursery, clippy::restriction)]
// TEMPORARY replay of vanilla decoration order for issue 2362, not to be committed.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::generation::proto_chunk_test::stage_parity::{TestBlockRegistry, TestGridCache};
use crate::generation::{generator::WorldGenerator, get_world_gen, proto_chunk::ProtoChunk};
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;

use super::surface_biomes;

fn state_string(id: pumpkin_data::BlockStateId) -> String {
    let state = pumpkin_data::BlockState::from_id(id);
    let block = state.id.to_block();
    let mut s = format!("minecraft:{}", block.name);
    if let Some(props) = block.properties(id) {
        let mut p = props.to_props();
        p.sort();
        if !p.is_empty() {
            s.push('[');
            s.push_str(&p.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(","));
            s.push(']');
        }
    }
    s
}

fn dump(chunk: &ProtoChunk, out: &mut String) {
    let mut palette: Vec<u16> = Vec::new();
    let mut idx: HashMap<u16, usize> = HashMap::new();
    let mut data = Vec::new();
    for y in 0..chunk.height() as i32 {
        for z in 0..16 {
            for x in 0..16 {
                let st = chunk.get_block_state_raw(x, y, z);
                let i = *idx.entry(st.as_u16()).or_insert_with(|| {
                    palette.push(st.as_u16());
                    palette.len() - 1
                });
                data.push(i);
            }
        }
    }
    let _ = writeln!(out, "{} {} {} {}", chunk.x, chunk.z, chunk.bottom_y(), chunk.height());
    let names: Vec<String> = palette
        .iter()
        .map(|&p| state_string(pumpkin_data::BlockStateId::new(p).unwrap()))
        .collect();
    let _ = writeln!(out, "{}", names.join(" "));
    let _ = writeln!(out, "{}", data.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn issue_2362_replay() {
    let order_path = std::env::var("ORDER_FILE").unwrap();
    let out = std::env::var("DUMP_DIR").unwrap();
    let order: Vec<(i32, i32)> = std::fs::read_to_string(order_path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|v| v.parse::<i32>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    let decorated: HashSet<(i32, i32)> = order.iter().copied().collect();
    let mut carved = HashSet::new();
    for &(x, z) in &order {
        for dx in -1..=1 {
            for dz in -1..=1 {
                carved.insert((x + dx, z + dz));
            }
        }
    }
    let mut started = HashSet::new();
    for &(x, z) in &carved {
        for dx in -9..=9 {
            for dz in -9..=9 {
                started.insert((x + dx, z + dz));
            }
        }
    }

    let world_gen = get_world_gen(Seed(1785195618762295304), Dimension::OVERWORLD, false, Vec::new(), String::new());
    let WorldGenerator::Noise(generator) = &*world_gen else { unreachable!() };
    let mut chunks = HashMap::new();
    for &(x, z) in &started {
        let mut c = ProtoChunk::new(x, z, &world_gen);
        c.step_to_biomes(generator);
        c.set_structure_starts(generator);
        chunks.insert((x, z), c);
    }
    for &(x, z) in &carved {
        let c = chunks.get_mut(&(x, z)).unwrap();
        c.set_structure_references(generator);
    }
    for &(x, z) in &carved {
        let sb = surface_biomes(&world_gen, x, z);
        let c = chunks.get_mut(&(x, z)).unwrap();
        c.step_to_noise(generator);
        c.step_to_surface(generator, &sb);
        c.step_to_carvers(generator);
    }

    let registry = TestBlockRegistry;
    let mut grid = TestGridCache { center_pos: order[0], chunks };
    for &(x, z) in &order {
        grid.center_pos = (x, z);
        eprintln!("TRACE BEGIN {x} {z}");
        ProtoChunk::generate_features_and_structure(&mut grid, &registry, &generator.random_config);
        eprintln!("TRACE END {x} {z}");
    }

    let mut post = String::new();
    for &(x, z) in &order {
        let full = (-1..=1).all(|dx| (-1..=1).all(|dz| decorated.contains(&(x + dx, z + dz))));
        if full {
            dump(grid.chunks.get(&(x, z)).unwrap(), &mut post);
        }
    }
    std::fs::write(format!("{out}/replay.features.txt"), post).unwrap();
}
