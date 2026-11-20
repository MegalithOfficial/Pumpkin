use super::{
    super::super::router::chunk_density_function::ChunkSpecificNoiseFunctionComponent, Component,
    Dep, Leaf, Noise, Tail, grid, resident_layers, sampler_branch,
};
use cudarc::driver::CudaSlice;
use std::sync::Arc;

pub(super) const MAX_SELECTIONS: usize = 3;
pub(super) const MAX_ALTERNATIVES: usize = 5;

pub(super) struct Selection {
    pub thresholds: &'static [f32],
    pub alternatives: Vec<Alternative>,
}

pub(super) struct Alternative {
    pub layers: Arc<CudaSlice<super::abi::CudaLayer>>,
    pub tail: Tail,
}

pub(super) struct Branch<'a, 'router> {
    pub selector: (&'a Noise, Tail),
    thresholds: &'static [f32],
    alternatives: &'static [usize],
    prefix: &'a [Component<'router>],
}

impl<'a, 'router> Branch<'a, 'router> {
    pub(super) fn sources(&self) -> usize {
        self.alternatives.len() + 1
    }

    pub(super) fn upload(&self, engine: &super::Engine) -> Option<grid::Plane> {
        let (selector, tail) = self.selector;
        let layers = resident_layers(engine, selector)?;
        let mut column_states = layers.len();
        let mut alternatives = Vec::with_capacity(self.alternatives.len());
        for &index in self.alternatives {
            let (Leaf::Noise(noise), tail) = sampler_branch(self.prefix.get(..=index)?, true)?
            else {
                return None;
            };
            let layers = resident_layers(engine, noise)?;
            column_states += layers.len();
            alternatives.push(Alternative { layers, tail });
        }
        if column_states > super::worker::MAX_COLUMN_STATES {
            return None;
        }
        Some(grid::Plane {
            layers,
            tail,
            companion: None,
            selection: Some(Selection {
                thresholds: self.thresholds,
                alternatives,
            }),
        })
    }
}

pub(super) fn branch<'a, 'router>(stack: &'a [Component<'router>]) -> Option<Branch<'a, 'router>> {
    let (Component::Dependent(Dep::IntervalSelect(node)), prefix) = stack.split_last()? else {
        return None;
    };
    if !(2..=MAX_ALTERNATIVES).contains(&node.functions_indices.len())
        || node.thresholds.len() + 1 != node.functions_indices.len()
    {
        return None;
    }

    // Cache the selector's pure volume on the CPU when other nodes need it.
    // The GPU can independently evaluate the same volume without warming that cache.
    let mut input = prefix.get(..=node.input_index)?;
    for _ in 0..8 {
        let (node, rest) = input.split_last()?;
        match node {
            Component::Chunk(ChunkSpecificNoiseFunctionComponent::Cache(cache)) => {
                input = rest.get(..=cache.input_index)?;
            }
            Component::PassThrough(pass) => input = rest.get(..=pass.input_index())?,
            _ => break,
        }
    }
    let (Leaf::Noise(selector), tail) = sampler_branch(input, true)? else {
        return None;
    };
    for &index in node.functions_indices {
        if !matches!(
            sampler_branch(prefix.get(..=index)?, true),
            Some((Leaf::Noise(_), _))
        ) {
            return None;
        }
    }
    Some(Branch {
        selector: (selector, tail),
        thresholds: node.thresholds,
        alternatives: node.functions_indices,
        prefix,
    })
}
