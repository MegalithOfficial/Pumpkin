use crate::generation::proto_chunk::GenerationCache;
use pumpkin_data::{Block, BlockDirection, BlockState, HorizontalFacingExt};
use pumpkin_util::{
    math::{position::BlockPos, vector3::Vector3},
    random::{RandomGenerator, RandomImpl},
};

pub struct UnderwaterMagmaFeature {
    /// vertical search limit from origin for a water column floor
    pub floor_search_range: i32,
    /// horizontal radius from the floor spot to sample
    pub placement_radius: i32,
    /// chance each inspected block turns into magma
    pub placement_probability: f32,
}

impl UnderwaterMagmaFeature {
    fn is_water<T: GenerationCache>(chunk: &T, pos: &BlockPos) -> bool {
        GenerationCache::get_block_state(chunk, &pos.0).to_block_id() == Block::WATER
    }

    /// Floor half of vanilla `Column.scan`: the origin must be water, then scan down through it.
    fn get_floor_y<T: GenerationCache>(&self, chunk: &T, origin: BlockPos) -> Option<i32> {
        if !Self::is_water(chunk, &origin) {
            return None;
        }

        let mut pos = origin;
        let mut i = 1;
        while i < self.floor_search_range && Self::is_water(chunk, &pos) {
            pos = pos.down();
            i += 1;
        }

        (!Self::is_water(chunk, &pos)).then_some(pos.0.y)
    }

    fn is_valid_placement<T: GenerationCache>(chunk: &T, pos: &BlockPos) -> bool {
        let state = GenerationCache::get_block_state(chunk, &pos.0).to_state();
        if state.id.to_block_id() == Block::WATER || state.is_air() {
            return false;
        }
        if Self::is_visible_from_outside(chunk, &pos.down(), BlockDirection::Up) {
            return false;
        }
        !BlockDirection::horizontal().iter().any(|dir| {
            let dir = dir.to_block_direction();
            Self::is_visible_from_outside(chunk, &pos.offset(dir.to_offset()), dir.opposite())
        })
    }

    fn is_visible_from_outside<T: GenerationCache>(
        chunk: &T,
        pos: &BlockPos,
        covered_direction: BlockDirection,
    ) -> bool {
        !GenerationCache::get_block_state(chunk, &pos.0)
            .to_state()
            .is_face_occluded(covered_direction)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate<T: GenerationCache>(
        &self,
        chunk: &mut T,
        _min_y: i8,
        _height: u16,
        _feature_name: pumpkin_data::placed_feature::PlacedFeature,
        random: &mut RandomGenerator,
        pos: BlockPos,
    ) -> bool {
        let Some(floor_y) = self.get_floor_y(chunk, pos) else {
            return false;
        };

        let floor_pos = BlockPos::new(pos.0.x, floor_y, pos.0.z);
        let r = self.placement_radius;
        let mut placed = false;

        // Same order as `BlockPos.betweenClosed`: x changes fastest, then y, then z.
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    if random.next_f32() >= self.placement_probability {
                        continue;
                    }

                    let target_pos = floor_pos.offset(Vector3::new(dx, dy, dz));
                    if !Self::is_valid_placement(chunk, &target_pos) {
                        continue;
                    }

                    chunk.set_block_state(
                        &target_pos.0,
                        BlockState::from_id(Block::MAGMA_BLOCK.default_state.id),
                    );
                    placed = true;
                }
            }
        }

        placed
    }
}
