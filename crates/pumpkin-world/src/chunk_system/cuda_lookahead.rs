use super::{ChunkPos, HashMapType, dag::NodeKey};
use crate::generation::{
    generator::WorldGenerator,
    noise::gpu::cuda::{
        grid::{self, EarlyGridScope, PreparedGrid},
        scheduled_lookahead_enabled,
    },
};
use crate::level::Level;
use crossbeam::channel::{Receiver, Sender, bounded};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};

// GPU grids for nearby Noise tasks share the queue with demand-driven jobs.
const MAX_LOOKAHEAD: usize = 16;

pub(super) struct Lookahead {
    requests: Sender<Request>,
    pending: HashMapType<NodeKey, Ticket>,
    limit: usize,
}

struct Request {
    pos: ChunkPos,
    world: Arc<WorldGenerator>,
    cancel: Arc<AtomicBool>,
    reply: Sender<PreparedGrid>,
}

pub(super) struct Ticket {
    world: Arc<WorldGenerator>,
    cancel: Arc<AtomicBool>,
    ready: Receiver<PreparedGrid>,
}

impl Ticket {
    pub(super) fn activate(&self, level: &Level) -> Option<EarlyGridScope<'_>> {
        let scope = (|| {
            let current = level.world_gen.load();
            if !Arc::ptr_eq(&self.world, &current) {
                return None;
            }
            let WorldGenerator::Noise(generator) = self.world.as_ref() else {
                return None;
            };
            self.ready.try_recv().ok()?.activate(generator)
        })();
        if scope.is_none() {
            self.cancel.store(true, Ordering::Release);
        }
        scope
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl Lookahead {
    pub(super) fn spawn(max_in_flight: u16) -> Option<(Self, JoinHandle<()>)> {
        if !scheduled_lookahead_enabled() {
            return None;
        }
        let limit = usize::from(max_in_flight).min(MAX_LOOKAHEAD);
        let (requests, receiver) = bounded(limit);
        match thread::Builder::new()
            .name("CUDA-Lookahead".to_string())
            .spawn(move || prepare_grids(receiver))
        {
            Ok(thread) => Some((
                Self {
                    requests,
                    pending: HashMapType::default(),
                    limit,
                },
                thread,
            )),
            Err(error) => {
                tracing::warn!(%error, "CUDA worldgen lookahead thread unavailable");
                None
            }
        }
    }

    pub(super) fn schedule(&mut self, key: NodeKey, pos: ChunkPos, world: Arc<WorldGenerator>) {
        if self.pending.len() >= self.limit || self.pending.contains_key(&key) {
            return;
        }
        if !matches!(world.as_ref(), WorldGenerator::Noise(_)) {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let (reply, ready) = bounded(1);
        let request = Request {
            pos,
            world: world.clone(),
            cancel: cancel.clone(),
            reply,
        };
        if self.requests.try_send(request).is_ok() {
            self.pending.insert(
                key,
                Ticket {
                    world,
                    cancel,
                    ready,
                },
            );
        }
    }

    pub(super) fn take(&mut self, key: NodeKey) -> Option<Ticket> {
        self.pending.remove(&key)
    }
}

fn prepare_grids(requests: Receiver<Request>) {
    while let Ok(request) = requests.recv() {
        if request.cancel.load(Ordering::Acquire) {
            continue;
        }
        if let WorldGenerator::Noise(generator) = request.world.as_ref()
            && let Some(grid) = grid::prepare_chunk(
                generator,
                request.pos.x,
                request.pos.y,
                Some(request.cancel),
            )
        {
            let _ = request.reply.send(grid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::get_world_gen;
    use pumpkin_data::dimension::Dimension;
    use pumpkin_util::world_seed::Seed;
    use slotmap::SlotMap;

    #[test]
    fn pending_lookahead_is_bounded_and_cancels_dropped_tickets() {
        let world = Arc::from(get_world_gen(
            Seed(42),
            Dimension::OVERWORLD,
            false,
            Vec::new(),
            String::new(),
        ));
        let (requests, receiver) = bounded(2);
        let mut queue = Lookahead {
            requests,
            pending: HashMapType::default(),
            limit: 1,
        };
        let mut nodes = SlotMap::<NodeKey, ()>::with_key();
        let first = nodes.insert(());
        let second = nodes.insert(());
        queue.schedule(first, ChunkPos::new(0, 0), Arc::clone(&world));
        let first_request = receiver.try_recv().expect("first request");
        queue.schedule(second, ChunkPos::new(0, 1), Arc::clone(&world));
        assert!(receiver.try_recv().is_err());
        assert!(!first_request.cancel.load(Ordering::Acquire));
        drop(queue.take(first));
        assert!(first_request.cancel.load(Ordering::Acquire));
        queue.schedule(second, ChunkPos::new(0, 1), world);
        assert!(receiver.try_recv().is_ok());
    }
}
