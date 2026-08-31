use super::ChunkPos;
use crate::cylindrical_chunk_iterator::Cylindrical;
use crate::level::SyncChunk;
use crossbeam::atomic::AtomicCell;
use crossbeam::channel::{Receiver, Sender, TrySendError};
use std::sync::Arc;
use std::sync::{Mutex, Weak};
use tokio::sync::oneshot;

type GlobalChunkMessage = (ChunkPos, Weak<crate::chunk::ChunkData>);

const GLOBAL_LISTENER_CAPACITY: usize =
    Cylindrical::get_offsets(pumpkin_data::chunk_view_lut::MAX_VIEW_DISTANCE).len();

struct GlobalListenerRegistration {
    sender: Sender<GlobalChunkMessage>,
    watched: Arc<AtomicCell<Cylindrical>>,
}

pub struct GlobalChunkListener {
    receiver: Receiver<GlobalChunkMessage>,
    watched: Arc<AtomicCell<Cylindrical>>,
}

impl GlobalChunkListener {
    pub fn update_watched(&self, watched: Cylindrical) {
        self.watched.store(watched);
    }

    pub fn try_recv(&self) -> Result<GlobalChunkMessage, crossbeam::channel::TryRecvError> {
        self.receiver.try_recv()
    }
}

pub struct ChunkListener {
    single: Mutex<Vec<(ChunkPos, oneshot::Sender<SyncChunk>)>>,
    global: Mutex<Vec<Arc<GlobalListenerRegistration>>>,
}

impl Default for ChunkListener {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkListener {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            single: Mutex::new(Vec::new()),
            global: Mutex::new(Vec::new()),
        }
    }

    pub fn add_single_chunk_listener(&self, pos: ChunkPos) -> oneshot::Receiver<SyncChunk> {
        let (tx, rx) = oneshot::channel();
        self.single
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((pos, tx));
        rx
    }

    pub fn add_global_chunk_listener(&self, watched: Cylindrical) -> GlobalChunkListener {
        let (tx, rx) = crossbeam::channel::bounded(GLOBAL_LISTENER_CAPACITY);
        let watched = Arc::new(AtomicCell::new(watched));
        self.global
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::new(GlobalListenerRegistration {
                sender: tx,
                watched: watched.clone(),
            }));
        GlobalChunkListener {
            receiver: rx,
            watched,
        }
    }

    pub fn process_new_chunk(&self, pos: ChunkPos, chunk: &SyncChunk) {
        {
            let mut single = self
                .single
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut i = 0;
            let mut len = single.len();
            while i < len {
                if single[i].0 == pos {
                    let (_, send) = single.remove(i);
                    let _ = send.send(chunk.clone());
                    // log::debug!("single listener {i} send {pos:?}");
                    len -= 1;
                    continue;
                }
                i += 1;
            }
        }
        self.process_global(pos, &Arc::downgrade(chunk));
    }

    fn process_global(&self, pos: ChunkPos, chunk: &Weak<crate::chunk::ChunkData>) {
        let listeners = self
            .global
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut disconnected = Vec::new();

        for listener in listeners {
            if !listener.watched.load().is_within_distance(pos.x, pos.y) {
                continue;
            }
            match listener.sender.try_send((pos, chunk.clone())) {
                Ok(()) | Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => disconnected.push(listener),
            }
        }

        if !disconnected.is_empty() {
            self.global
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|listener| {
                    !disconnected
                        .iter()
                        .any(|closed| Arc::ptr_eq(listener, closed))
                });
        }
    }
}
