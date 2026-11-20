//! Programs a request derives from the model's own and would otherwise rebuild each time.
//!
//! The greedy argmax program ([`with_greedy_argmax`]) and the direct KV rows program
//! ([`with_direct_kv_rows`]) are each a function of the program they extend, the roots and
//! the vocabulary, never of the prompt. Rebuilding one is a `to_vec` over every op, a shape
//! inference and a SipHash of the result (measured 4.54 ms for the greedy program at the
//! start of a resident repeat request). [`ProgramCache`] keeps them between requests.
//!
//! Home: [`ResidentPlans`](super::resident_plans::ResidentPlans), the same store that keeps
//! the decode plans and the kv buffers. `BackendRuntime` takes the cache when it resumes the
//! context and puts it back when it parks, so the model's serving gate (one request at a
//! time) is the only exclusion needed and no lock is added.
//!
//! A hit needs every input the constructor read to be equal ([`GreedyKey`], and for the
//! direct rows the source hash, the source roots and the two cache formats). A miss rebuilds
//! and replaces; an entry is never reused under a different key.

use super::*;
use super::device_kv::{DirectKvRows, with_direct_kv_rows};
use super::greedy_device::with_greedy_argmax;

/// Greedy programs kept: the decode program and the prefill-cut program are the two a
/// request's greedy step alternates between, so one entry each is all a request can reuse.
const GREEDY_PROGRAMS_KEPT: usize = 2;
/// Direct rows programs kept at rest: a model has a decode, a verify and a prefill-cut
/// program; past this the map holds hoisted per-step variants and is dropped, not grown.
const DIRECT_KV_PROGRAMS_KEPT: usize = 8;

/// Everything [`with_greedy_argmax`] and the logits buffer's size read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GreedyKey {
    /// the program the argmax ops extend, by [`program_hash`]
    program_hash: u64,
    logits_root: NodeId,
    vocab: u32,
    suppressed_ids: Vec<u32>,
    /// whether the extended program is the prefill cut, which the loop alternates with decode
    from_prefill_cut: bool,
}

impl GreedyKey {
    pub(super) fn new(
        program_hash: u64,
        logits_root: NodeId,
        vocab: u32,
        suppressed_ids: &[u32],
        from_prefill_cut: bool,
    ) -> Self {
        Self {
            program_hash,
            logits_root,
            vocab,
            suppressed_ids: suppressed_ids.to_vec(),
            from_prefill_cut,
        }
    }

    pub(super) fn is_prefill_cut(&self) -> bool {
        self.from_prefill_cut
    }
}

/// A greedy program with the placed logits row its device step writes.
pub(super) struct GreedyProgram {
    key: GreedyKey,
    pub(super) program: Vec<Op>,
    pub(super) token: NodeId,
    pub(super) finite: NodeId,
    /// owned by the entry, so the row lives exactly as long as the program that writes it and
    /// is freed when the entry is evicted or the serving context clears its plans
    pub(super) logits: PlacedBuffer,
    pub(super) hash: u64,
}

impl GreedyProgram {
    pub(super) fn key(&self) -> &GreedyKey {
        &self.key
    }
    /// `None` when the program does not accept the argmax ops.
    fn build(key: GreedyKey, program: &[Op], symbols: &[u64]) -> Result<Option<Self>, InteropError> {
        let Some((extended, token, finite)) = with_greedy_argmax(
            program,
            key.logits_root,
            key.vocab,
            symbols,
            &key.suppressed_ids,
        ) else {
            return Ok(None);
        };
        let logits = allocate_placed_buffer(key.vocab as usize * core::mem::size_of::<f32>())?;
        let hash = program_hash(&extended);
        Ok(Some(Self {
            key,
            program: extended,
            token,
            finite,
            logits,
            hash,
        }))
    }
}

struct DirectKvEntry {
    /// the layer roots the rows were built from: part of the key beside the source hash
    source_roots: Vec<LayerCacheRoots>,
    built: Option<(u64, DirectKvRows)>,
}

/// [`with_direct_kv_rows`] results by source program hash, valid for one pair of cache formats.
pub(super) struct DirectKvPrograms {
    formats: (NumericFormat, NumericFormat),
    entries: BTreeMap<u64, DirectKvEntry>,
    #[cfg(test)]
    builds: usize,
}

impl DirectKvPrograms {
    fn new(formats: (NumericFormat, NumericFormat), #[cfg(test)] builds: usize) -> Self {
        Self {
            formats,
            entries: BTreeMap::new(),
            #[cfg(test)]
            builds,
        }
    }

    /// The rewritten program and its hash for `source`, built on first use and again whenever
    /// the roots differ from the ones the held entry was built from; `None` when neither
    /// plane's format takes a device row store.
    pub(super) fn rows_for(
        &mut self,
        source: &[Op],
        source_hash: u64,
        source_roots: &[LayerCacheRoots],
    ) -> Result<Option<(u64, &DirectKvRows)>, InteropError> {
        let fresh = self
            .entries
            .get(&source_hash)
            .is_some_and(|held| held.source_roots == source_roots);
        if !fresh {
            let built = with_direct_kv_rows(
                source,
                source_roots,
                (&self.formats.0, &self.formats.1),
            )?
            .map(|rows| (program_hash(&rows.program), rows));
            #[cfg(test)]
            {
                self.builds += 1;
            }
            self.entries.insert(
                source_hash,
                DirectKvEntry {
                    source_roots: source_roots.to_vec(),
                    built,
                },
            );
        }
        Ok(self
            .entries
            .get(&source_hash)
            .and_then(|held| held.built.as_ref())
            .map(|(hash, rows)| (*hash, rows)))
    }
}

/// The programs one model keeps between requests.
#[derive(Default)]
pub(super) struct ProgramCache {
    greedy: Vec<GreedyProgram>,
    direct_kv: Option<DirectKvPrograms>,
    #[cfg(test)]
    greedy_builds: usize,
}

impl ProgramCache {
    pub(super) fn is_empty(&self) -> bool {
        self.greedy.is_empty() && self.direct_kv.as_ref().is_none_or(|held| held.entries.is_empty())
    }

    /// The greedy program for `key`: a held one when its key is equal, else built from
    /// `program`. `None` when the program declines the argmax ops. `symbols` are the step's
    /// bound values: a build checks the extended program against them, a hit does not read
    /// them, because the appended ops are shaped by `[rows, vocab]` with `rows` symbolic.
    pub(super) fn greedy_for(
        &mut self,
        key: GreedyKey,
        program: &[Op],
        symbols: &[u64],
    ) -> Result<Option<GreedyProgram>, InteropError> {
        if let Some(position) = self.greedy.iter().position(|held| held.key == key) {
            return Ok(Some(self.greedy.remove(position)));
        }
        #[cfg(test)]
        {
            self.greedy_builds += 1;
        }
        GreedyProgram::build(key, program, symbols)
    }

    /// Keeps `program` for the next request, replacing an entry with its key and evicting the
    /// oldest past [`GREEDY_PROGRAMS_KEPT`].
    pub(super) fn keep_greedy(&mut self, program: GreedyProgram) {
        self.greedy.retain(|held| held.key != program.key);
        self.greedy.push(program);
        if self.greedy.len() > GREEDY_PROGRAMS_KEPT {
            drop(self.greedy.remove(0));
        }
    }

    /// The held direct rows when they were built for `formats`; otherwise an empty set, which
    /// replaces them.
    pub(super) fn take_direct_kv(&mut self, formats: (NumericFormat, NumericFormat)) -> DirectKvPrograms {
        match self.direct_kv.take() {
            Some(held) if held.formats == formats => held,
            #[cfg(test)]
            Some(held) => DirectKvPrograms::new(formats, held.builds),
            #[cfg(test)]
            None => DirectKvPrograms::new(formats, 0),
            #[cfg(not(test))]
            _ => DirectKvPrograms::new(formats),
        }
    }

    pub(super) fn keep_direct_kv(&mut self, held: DirectKvPrograms) {
        if held.entries.len() <= DIRECT_KV_PROGRAMS_KEPT {
            self.direct_kv = Some(held);
        }
    }

    #[cfg(test)]
    pub(super) fn greedy_builds(&self) -> usize {
        self.greedy_builds
    }

    #[cfg(test)]
    pub(super) fn direct_kv_builds(&self) -> usize {
        self.direct_kv.as_ref().map_or(0, |held| held.builds)
    }
}
