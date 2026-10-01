//! A radix trie over fixed token blocks, in a capacity-bounded
//! [`Arena`](super::arena::Arena), that names the cached entry sharing the
//! longest prefix with a prompt in one root walk
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R13).
//!
//! An entry's ids are cut into whole blocks of `block_tokens`; a node holds a
//! run of blocks that every entry through it shares, a divergence splits a
//! node in two, and a removal merges a node that stopped branching into its
//! only child, so a non-root node always has an entry ending on it or two
//! children. That bounds the nodes by two per entry, which is why the arena's
//! capacity can come from the cache's entry limit and an insert past it is a
//! refusal rather than a growth.
//!
//! A node does not copy its tokens. It names an entry that passes through it
//! (`owner`) and the block range of that entry's ids it stands for, and the
//! caller resolves the stamp to ids (`ids_of`). A node's block range is the
//! same for every entry through it, so when its owner is removed the node
//! hands the range to another entry through it. Copying would cost
//! `4 * block_tokens` bytes per block; naming costs none.
//!
//! A node's children are found by the hash of the block that starts their
//! run, through one fixed table keyed by (parent, block hash) and chained
//! through the nodes; a hit is verified against the owner's ids before it
//! counts, so a collision costs a comparison and never a wrong reuse. Keying
//! by first token instead was measured out: every conversation opens with the
//! same BOS and chat-header tokens, so the first token names no child.
//!
//! Compose it with [`super::prompt_cache::PromptCache`], which owns the
//! entries and the stamps; it is the trie's only caller. No `Box`, and after
//! the arena has grown to its high-water mark no operation allocates.

use super::arena::{Arena, Handle};

const HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;
const PARENT_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
const BUCKET_MIX: u64 = 0xff51_afd7_ed55_8ccd;

type NodeId = Handle<Node>;
type LinkId = Handle<Link>;

#[derive(Debug, Clone, Copy)]
struct Node {
    parent: Option<NodeId>,
    first_child: Option<NodeId>,
    next_sibling: Option<NodeId>,
    previous_sibling: Option<NodeId>,
    bucket_next: Option<NodeId>,
    terminals: Option<LinkId>,
    edge_hash: u64,
    owner: u64,
    start_block: u32,
    run_blocks: u32,
    through: u32,
}

impl Node {
    const fn fresh(owner: u64, start_block: u32, run_blocks: u32, edge_hash: u64) -> Self {
        Self {
            parent: None,
            first_child: None,
            next_sibling: None,
            previous_sibling: None,
            bucket_next: None,
            terminals: None,
            edge_hash,
            owner,
            start_block,
            run_blocks,
            through: 1,
        }
    }

    const fn end_block(&self) -> usize {
        (self.start_block + self.run_blocks) as usize
    }
}

#[derive(Debug, Clone, Copy)]
struct Link {
    stamp: u64,
    next: Option<LinkId>,
}

/// Which entries a lookup considers once the prompt shares less than one
/// whole block with every entry that has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SubBlock {
    /// Only entries shorter than a block: a shorter overlap cannot clear the
    /// similarity floor for any other.
    ShortOnly,
    /// Also the entries opening with the prompt's first token.
    FirstToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(super) enum TrieError {
    #[error("prefix index already holds its {entry_capacity} entries")]
    Full { entry_capacity: usize },
    #[error("prefix index lost the node entry {stamp} points through")]
    Inconsistent { stamp: u64 },
}

pub(super) struct PrefixTrie {
    block_tokens: usize,
    entry_capacity: usize,
    nodes: Arena<Node>,
    links: Arena<Link>,
    buckets: Vec<Option<NodeId>>,
    root: Option<NodeId>,
}

fn block_hash(block: &[u32]) -> u64 {
    let folded = block.iter().fold(HASH_SEED, |hash, token| {
        (hash.rotate_left(5) ^ u64::from(*token)).wrapping_mul(HASH_SEED)
    });
    (folded ^ (folded >> 32)).wrapping_mul(BUCKET_MIX)
}

fn keep_better(best: &mut Option<(u64, usize)>, stamp: u64, lcp: usize) {
    if best.is_none_or(|(held_stamp, held_lcp)| (lcp, stamp) > (held_lcp, held_stamp)) {
        *best = Some((stamp, lcp));
    }
}

impl PrefixTrie {
    /// A trie for up to `entry_capacity` entries at once, cut into blocks of
    /// `block_tokens`. The nodes are bounded by two per entry plus the root.
    pub(super) fn new(block_tokens: usize, entry_capacity: usize) -> Self {
        let node_capacity = 2 * (entry_capacity + 1) + 1;
        let buckets = node_capacity.next_power_of_two();
        Self {
            block_tokens: block_tokens.max(1),
            entry_capacity,
            nodes: Arena::new(node_capacity),
            links: Arena::new(entry_capacity + 1),
            buckets: vec![None; buckets],
            root: None,
        }
    }

    pub(super) const fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    pub(super) const fn entry_capacity(&self) -> usize {
        self.entry_capacity
    }

    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    fn edit<Out>(&mut self, id: NodeId, change: impl FnOnce(&mut Node) -> Out) -> Option<Out> {
        self.nodes.get_mut(id).map(change)
    }

    fn block<'tokens>(&self, tokens: &'tokens [u32], index: usize) -> Option<&'tokens [u32]> {
        tokens.get(index * self.block_tokens..(index + 1) * self.block_tokens)
    }

    fn bucket_of(&self, parent: NodeId, edge_hash: u64) -> usize {
        let mixed =
            (edge_hash ^ u64::from(parent.get()).wrapping_mul(PARENT_MIX)).wrapping_mul(BUCKET_MIX);
        (mixed >> 32) as usize & (self.buckets.len() - 1)
    }

    fn matching_blocks<'ids>(
        &self,
        node: &Node,
        tokens: &[u32],
        depth: usize,
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> usize {
        let Some(owner) = ids_of(node.owner) else {
            return 0;
        };
        (0..node.run_blocks as usize)
            .take_while(|offset| {
                let held = self.block(owner, node.start_block as usize + offset);
                held.is_some() && held == self.block(tokens, depth + offset)
            })
            .count()
    }

    fn find_child<'ids>(
        &self,
        parent: NodeId,
        tokens: &[u32],
        depth: usize,
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<(NodeId, usize)> {
        let hash = block_hash(self.block(tokens, depth)?);
        let mut cursor = *self.buckets.get(self.bucket_of(parent, hash))?;
        while let Some(id) = cursor {
            let node = self.node(id)?;
            if node.parent == Some(parent) && node.edge_hash == hash {
                let matched = self.matching_blocks(node, tokens, depth, ids_of);
                if matched > 0 {
                    return Some((id, matched));
                }
            }
            cursor = node.bucket_next;
        }
        None
    }

    fn link_child(&mut self, parent: NodeId, child: NodeId) -> Option<()> {
        let head = self.node(parent)?.first_child;
        let bucket = self.bucket_of(parent, self.node(child)?.edge_hash);
        let chain_head = *self.buckets.get(bucket)?;
        self.edit(child, |node| {
            node.parent = Some(parent);
            node.next_sibling = head;
            node.previous_sibling = None;
            node.bucket_next = chain_head;
        })?;
        if let Some(old_head) = head {
            self.edit(old_head, |node| node.previous_sibling = Some(child))?;
        }
        self.edit(parent, |node| node.first_child = Some(child))?;
        *self.buckets.get_mut(bucket)? = Some(child);
        Some(())
    }

    fn unlink_child(&mut self, child: NodeId) -> Option<()> {
        let node = *self.node(child)?;
        let parent = node.parent?;
        match node.previous_sibling {
            Some(previous) => self.edit(previous, |held| held.next_sibling = node.next_sibling)?,
            None => self.edit(parent, |held| held.first_child = node.next_sibling)?,
        }
        if let Some(next) = node.next_sibling {
            self.edit(next, |held| held.previous_sibling = node.previous_sibling)?;
        }
        self.unlink_from_bucket(child, parent, node)
    }

    fn unlink_from_bucket(&mut self, child: NodeId, parent: NodeId, node: Node) -> Option<()> {
        let bucket = self.bucket_of(parent, node.edge_hash);
        let head = *self.buckets.get(bucket)?;
        if head == Some(child) {
            *self.buckets.get_mut(bucket)? = node.bucket_next;
            return Some(());
        }
        let mut cursor = head;
        while let Some(id) = cursor {
            let held = *self.node(id)?;
            if held.bucket_next == Some(child) {
                return self.edit(id, |previous| previous.bucket_next = node.bucket_next);
            }
            cursor = held.bucket_next;
        }
        None
    }

    fn push_terminal(&mut self, node: NodeId, stamp: u64) -> Option<()> {
        let next = self.node(node)?.terminals;
        let link = self.links.insert(Link { stamp, next }).ok()?;
        self.edit(node, |held| held.terminals = Some(link))
    }

    fn pop_terminal(&mut self, node: NodeId, stamp: u64) -> Option<()> {
        let mut previous: Option<LinkId> = None;
        let mut cursor = self.node(node)?.terminals;
        while let Some(id) = cursor {
            let link = *self.links.get(id)?;
            if link.stamp == stamp {
                match previous {
                    Some(before) => self.links.get_mut(before)?.next = link.next,
                    None => self.edit(node, |held| held.terminals = link.next)?,
                }
                self.links.remove(id)?;
                return Some(());
            }
            previous = Some(id);
            cursor = link.next;
        }
        None
    }

    fn ensure_root(&mut self) -> Option<NodeId> {
        if self.root.is_none() {
            let mut root = Node::fresh(0, 0, 0, 0);
            root.through = 0;
            self.root = self.nodes.insert(root).ok();
        }
        self.root
    }

    /// Adds entry `stamp` with `ids`; `ids_of` resolves the stamps of the
    /// entries already held to their ids. At most two nodes and one link are
    /// made, and a trie at its capacity refuses before changing anything.
    pub(super) fn insert<'ids>(
        &mut self,
        stamp: u64,
        ids: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Result<(), TrieError> {
        let nodes_needed = if self.root.is_some() { 2 } else { 3 };
        if self.nodes.available() < nodes_needed || self.links.available() < 1 {
            return Err(TrieError::Full {
                entry_capacity: self.entry_capacity,
            });
        }
        self.insert_path(stamp, ids, ids_of)
            .ok_or(TrieError::Inconsistent { stamp })
    }

    fn insert_path<'ids>(
        &mut self,
        stamp: u64,
        ids: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<()> {
        let root = self.ensure_root()?;
        self.edit(root, |node| node.through += 1)?;
        let blocks = ids.len() / self.block_tokens;
        let (mut node, mut depth) = (root, 0);
        while depth < blocks {
            (node, depth) = match self.find_child(node, ids, depth, ids_of) {
                None => (self.new_leaf(node, stamp, ids, depth, blocks)?, blocks),
                Some((child, matched)) if matched == self.node(child)?.run_blocks as usize => {
                    self.edit(child, |held| held.through += 1)?;
                    (child, depth + matched)
                }
                Some((child, matched)) => {
                    (self.split(node, child, matched, ids_of)?, depth + matched)
                }
            };
        }
        self.push_terminal(node, stamp)
    }

    fn new_leaf(
        &mut self,
        parent: NodeId,
        stamp: u64,
        ids: &[u32],
        depth: usize,
        blocks: usize,
    ) -> Option<NodeId> {
        let edge_hash = block_hash(self.block(ids, depth)?);
        let start = u32::try_from(depth).ok()?;
        let run = u32::try_from(blocks - depth).ok()?;
        let leaf = self
            .nodes
            .insert(Node::fresh(stamp, start, run, edge_hash))
            .ok()?;
        self.link_child(parent, leaf)?;
        Some(leaf)
    }

    fn split<'ids>(
        &mut self,
        parent: NodeId,
        child: NodeId,
        matched: usize,
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<NodeId> {
        let lower = *self.node(child)?;
        let matched = u32::try_from(matched).ok()?;
        let lower_first = (lower.start_block + matched) as usize;
        let lower_hash = block_hash(self.block(ids_of(lower.owner)?, lower_first)?);
        self.unlink_child(child)?;
        let mut upper = Node::fresh(lower.owner, lower.start_block, matched, lower.edge_hash);
        upper.through = lower.through + 1;
        let upper = self.nodes.insert(upper).ok()?;
        self.link_child(parent, upper)?;
        self.edit(child, |node| {
            node.start_block += matched;
            node.run_blocks -= matched;
            node.edge_hash = lower_hash;
        })?;
        self.link_child(upper, child)?;
        Some(upper)
    }

    /// Removes entry `stamp`, which was inserted with `ids`; `ids_of`
    /// resolves the stamps of the entries still held, and the removed entry
    /// may already be gone from the caller's map (it can own a node, so its
    /// own ids answer for it). Nodes no entry passes through are freed for the
    /// next insert.
    pub(super) fn remove<'ids>(
        &mut self,
        stamp: u64,
        ids: &'ids [u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Result<(), TrieError> {
        let resolve = |held: u64| {
            if held == stamp {
                Some(ids)
            } else {
                ids_of(held)
            }
        };
        self.remove_path(stamp, ids, &resolve)
            .ok_or(TrieError::Inconsistent { stamp })
    }

    fn locate<'ids>(
        &self,
        ids: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<NodeId> {
        let blocks = ids.len() / self.block_tokens;
        let (mut node, mut depth) = (self.root?, 0);
        while depth < blocks {
            let (child, matched) = self.find_child(node, ids, depth, ids_of)?;
            if matched != self.node(child)?.run_blocks as usize {
                return None;
            }
            (node, depth) = (child, depth + matched);
        }
        Some(node)
    }

    fn remove_path<'ids>(
        &mut self,
        stamp: u64,
        ids: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<()> {
        let end = self.locate(ids, ids_of)?;
        self.pop_terminal(end, stamp)?;
        let mut lowest_survivor: Option<NodeId> = None;
        let mut cursor = Some(end);
        while let Some(id) = cursor {
            let node = *self.node(id)?;
            cursor = node.parent;
            if node.through <= 1 {
                self.release(id, node)?;
                continue;
            }
            self.edit(id, |held| held.through -= 1)?;
            if node.owner == stamp && node.parent.is_some() {
                let replacement = self.any_entry_below(id)?;
                self.edit(id, |held| held.owner = replacement)?;
            }
            lowest_survivor.get_or_insert(id);
        }
        lowest_survivor.map_or(Some(()), |id| self.merge_if_redundant(id))
    }

    fn release(&mut self, id: NodeId, node: Node) -> Option<()> {
        if node.parent.is_some() {
            self.unlink_child(id)?;
        } else {
            self.root = None;
        }
        self.nodes.remove(id).map(|_removed| ())
    }

    fn any_entry_below(&self, from: NodeId) -> Option<u64> {
        let mut cursor = from;
        loop {
            let node = self.node(cursor)?;
            if let Some(link) = node.terminals {
                return Some(self.links.get(link)?.stamp);
            }
            cursor = node.first_child?;
        }
    }

    fn merge_if_redundant(&mut self, id: NodeId) -> Option<()> {
        let node = *self.node(id)?;
        if node.parent.is_none() || node.terminals.is_some() {
            return Some(());
        }
        let only_child = node.first_child?;
        if self.node(only_child)?.next_sibling.is_some() {
            return Some(());
        }
        self.merge_into_child(id, node, only_child)
    }

    fn merge_into_child(&mut self, upper: NodeId, node: Node, child: NodeId) -> Option<()> {
        let parent = node.parent?;
        self.unlink_child(child)?;
        self.unlink_child(upper)?;
        self.edit(child, |held| {
            held.start_block = node.start_block;
            held.run_blocks += node.run_blocks;
            held.edge_hash = node.edge_hash;
        })?;
        self.link_child(parent, child)?;
        self.nodes.remove(upper).map(|_removed| ())
    }

    fn descend<'ids>(
        &self,
        prompt: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> Option<(NodeId, usize)> {
        let blocks = prompt.len() / self.block_tokens;
        let (mut node, mut depth) = (self.root?, 0);
        while depth < blocks {
            let Some((child, matched)) = self.find_child(node, prompt, depth, ids_of) else {
                break;
            };
            (node, depth) = (child, depth + matched);
            if matched < self.node(child)?.run_blocks as usize {
                break;
            }
        }
        Some((node, depth))
    }

    /// Whole blocks of `prompt` that some entry shares from the start.
    #[cfg(all(test, feature = "metal", target_os = "macos"))]
    pub(super) fn matched_blocks<'ids>(
        &self,
        prompt: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
    ) -> usize {
        self.descend(prompt, ids_of).map_or(0, |(_, depth)| depth)
    }

    fn step_over(&self, candidate: Option<NodeId>, skip: Option<NodeId>) -> Option<NodeId> {
        match candidate {
            Some(id) if Some(id) == skip => self.node(id).and_then(|node| node.next_sibling),
            other => other,
        }
    }

    fn visit_terminals(&self, node: NodeId, visit: &mut impl FnMut(u64)) -> Option<()> {
        let mut cursor = self.node(node)?.terminals;
        while let Some(id) = cursor {
            let link = self.links.get(id)?;
            visit(link.stamp);
            cursor = link.next;
        }
        Some(())
    }

    fn visit_below(
        &self,
        top: NodeId,
        skip: Option<NodeId>,
        visit: &mut impl FnMut(u64),
    ) -> Option<()> {
        let mut node = top;
        loop {
            self.visit_terminals(node, visit)?;
            let mut next = self.step_over(self.node(node)?.first_child, skip);
            let mut climb = node;
            while next.is_none() {
                if climb == top {
                    return Some(());
                }
                let current = self.node(climb)?;
                next = self.step_over(current.next_sibling, skip);
                climb = current.parent?;
            }
            node = next?;
        }
    }

    fn best_below(
        &self,
        top: NodeId,
        skip: Option<NodeId>,
        from_token: usize,
        offer: &mut impl FnMut(u64, usize) -> Option<usize>,
    ) -> Option<(u64, usize)> {
        let mut best = None;
        self.visit_below(top, skip, &mut |stamp| {
            if let Some(lcp) = offer(stamp, from_token) {
                keep_better(&mut best, stamp, lcp);
            }
        })?;
        best
    }

    /// The entry the caller's `offer` rates highest, deepest level first: the
    /// entries through the deepest block the prompt shares, then those that
    /// left the path one node up, and so on to the root, stopping at the first
    /// level with an acceptable entry. `offer(stamp, from_token)` returns the
    /// entry's common prefix with the prompt (it knows tokens before
    /// `from_token` already match) when the entry is acceptable. Among a
    /// level's entries the longest prefix wins, then the larger stamp.
    /// Strictly deeper levels share strictly longer prefixes, so the first
    /// level with an answer holds the longest one.
    ///
    /// Cost: one hash and one block comparison per node on the prompt's path,
    /// then the offers of the entries on the level that answers; no
    /// allocation.
    pub(super) fn best<'ids>(
        &self,
        prompt: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
        sub_block: SubBlock,
        mut offer: impl FnMut(u64, usize) -> Option<usize>,
    ) -> Option<(u64, usize)> {
        let root = self.root?;
        let (mut cursor, depth) = self.descend(prompt, ids_of)?;
        let (mut skip, mut from_block) = (None, depth);
        while cursor != root {
            if let Some(found) =
                self.best_below(cursor, skip, from_block * self.block_tokens, &mut offer)
            {
                return Some(found);
            }
            skip = Some(cursor);
            cursor = self.node(cursor)?.parent?;
            from_block = self.node(cursor)?.end_block();
        }
        self.best_sub_block(root, skip, prompt, ids_of, sub_block, &mut offer)
    }

    fn best_sub_block<'ids>(
        &self,
        root: NodeId,
        on_path: Option<NodeId>,
        prompt: &[u32],
        ids_of: &impl Fn(u64) -> Option<&'ids [u32]>,
        sub_block: SubBlock,
        offer: &mut impl FnMut(u64, usize) -> Option<usize>,
    ) -> Option<(u64, usize)> {
        let mut best = None;
        self.visit_terminals(root, &mut |stamp| {
            if let Some(lcp) = offer(stamp, 0) {
                keep_better(&mut best, stamp, lcp);
            }
        })?;
        if let (SubBlock::FirstToken, Some(first)) = (sub_block, prompt.first()) {
            let mut cursor = self.node(root)?.first_child;
            while let Some(id) = cursor {
                let node = self.node(id)?;
                cursor = node.next_sibling;
                let opens_alike = ids_of(node.owner)
                    .and_then(|owner| owner.get(node.start_block as usize * self.block_tokens))
                    == Some(first);
                if Some(id) != on_path
                    && opens_alike
                    && let Some((stamp, lcp)) = self.best_below(id, None, 0, offer)
                {
                    keep_better(&mut best, stamp, lcp);
                }
            }
        }
        best
    }

    #[cfg(test)]
    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    pub(super) fn link_count(&self) -> usize {
        self.links.len()
    }

    #[cfg(test)]
    pub(super) fn byte_len(&self) -> usize {
        self.nodes.byte_len()
            + self.links.byte_len()
            + self.buckets.capacity() * size_of::<Option<NodeId>>()
    }

    #[cfg(test)]
    pub(super) const fn node_bytes() -> usize {
        size_of::<Node>()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, TestRunner};

    use super::*;
    use crate::generate::alloc_probe::allocations_during;
    use crate::generate::prompt_cache::{entry_is_reusable, longest_common_prefix};

    const BLOCK: usize = 4;

    /// Checks every structural rule the lookups lean on, returning the number
    /// of entries held: counts through each node equal the entries beneath
    /// it, a non-root node has an entry ending on it or two children, every
    /// child is reachable through the table, and no node is orphaned.
    fn audit(trie: &PrefixTrie) -> usize {
        let Some(root) = trie.root else {
            assert_eq!(trie.nodes.len(), 0, "an empty trie holds no node");
            assert_eq!(trie.links.len(), 0, "an empty trie holds no link");
            return 0;
        };
        let (entries, nodes) = audit_node(trie, root);
        assert_eq!(nodes, trie.nodes.len(), "every node hangs off the root");
        assert_eq!(entries, trie.links.len(), "every entry ends on one node");
        assert!(
            trie.nodes.len() <= 2 * entries + 1,
            "at most two nodes per entry"
        );
        entries
    }

    fn audit_node(trie: &PrefixTrie, id: NodeId) -> (usize, usize) {
        let node = trie.node(id).expect("a linked node is live");
        let mut terminals = 0;
        trie.visit_terminals(id, &mut |_| terminals += 1)
            .expect("terminals are live");
        let (mut entries, mut nodes, mut children) = (terminals, 1, 0);
        let mut cursor = node.first_child;
        while let Some(child_id) = cursor {
            let child = trie.node(child_id).expect("a child is live");
            assert_eq!(child.parent, Some(id), "a child names its parent");
            assert!(child.run_blocks >= 1, "a non-root node holds a block");
            assert_eq!(child.start_block as usize, node.end_block());
            let hash = child.edge_hash;
            let mut chain = trie.buckets[trie.bucket_of(id, hash)];
            let mut reachable = false;
            while let Some(link) = chain {
                reachable |= link == child_id;
                chain = trie.node(link).expect("a chained node is live").bucket_next;
            }
            assert!(reachable, "a child is reachable through the table");
            let (below_entries, below_nodes) = audit_node(trie, child_id);
            entries += below_entries;
            nodes += below_nodes;
            children += 1;
            cursor = child.next_sibling;
        }
        assert_eq!(
            node.through as usize, entries,
            "through counts the entries below"
        );
        if node.parent.is_some() {
            assert!(
                terminals > 0 || children >= 2,
                "a node that stopped branching merges"
            );
        }
        (entries, nodes)
    }

    struct Held {
        trie: PrefixTrie,
        entries: BTreeMap<u64, Vec<u32>>,
        next_stamp: u64,
    }

    impl Held {
        fn with_capacity(entry_capacity: usize) -> Self {
            Self {
                trie: PrefixTrie::new(BLOCK, entry_capacity),
                entries: BTreeMap::new(),
                next_stamp: 0,
            }
        }

        fn add(&mut self, ids: &[u32]) -> Result<u64, TrieError> {
            let stamp = self.next_stamp;
            let entries = &self.entries;
            self.trie
                .insert(stamp, ids, &|held| entries.get(&held).map(Vec::as_slice))?;
            self.entries.insert(stamp, ids.to_vec());
            self.next_stamp += 1;
            Ok(stamp)
        }

        fn drop_entry(&mut self, stamp: u64) {
            let ids = self.entries.remove(&stamp).expect("the entry is held");
            let entries = &self.entries;
            self.trie
                .remove(stamp, &ids, &|held| entries.get(&held).map(Vec::as_slice))
                .expect("a held entry is indexed");
        }

        fn lookup(&self, prompt: &[u32], floor: u32, sub_block: SubBlock) -> Option<(u64, usize)> {
            let entries = &self.entries;
            self.trie.best(
                prompt,
                &|held| entries.get(&held).map(Vec::as_slice),
                sub_block,
                |stamp, from| {
                    let ids = entries.get(&stamp)?;
                    let lcp = from + longest_common_prefix(ids.get(from..)?, prompt.get(from..)?);
                    entry_is_reusable(lcp, ids.len(), prompt.len(), floor).then_some(lcp)
                },
            )
        }

        fn scan(&self, prompt: &[u32], floor: u32) -> Option<(u64, usize)> {
            self.entries
                .iter()
                .map(|(stamp, ids)| (*stamp, longest_common_prefix(ids, prompt), ids.len()))
                .filter(|&(_, lcp, len)| entry_is_reusable(lcp, len, prompt.len(), floor))
                .max_by_key(|&(stamp, lcp, _)| (lcp, stamp))
                .map(|(stamp, lcp, _)| (stamp, lcp))
        }

        fn sub_block_for(prompt_len: usize, floor: u32) -> SubBlock {
            if (BLOCK - 1) * 1000 > floor as usize * prompt_len {
                SubBlock::FirstToken
            } else {
                SubBlock::ShortOnly
            }
        }

        fn agrees_with_the_scan(&self, prompt: &[u32], floor: u32) {
            let sub_block = Self::sub_block_for(prompt.len(), floor);
            assert_eq!(
                self.lookup(prompt, floor, sub_block),
                self.scan(prompt, floor),
                "prompt {prompt:?} over {:?}",
                self.entries
            );
        }
    }

    fn run(start: u32, blocks: usize) -> Vec<u32> {
        (start..start + (blocks * BLOCK) as u32).collect()
    }

    fn joined(parts: &[&[u32]]) -> Vec<u32> {
        parts.iter().flat_map(|part| part.iter().copied()).collect()
    }

    #[test]
    fn a_prompt_extending_an_entry_finds_it_at_the_entrys_length() {
        let mut held = Held::with_capacity(4);
        let conversation = run(100, 6);
        let stamp = held.add(&conversation).unwrap();
        let prompt = joined(&[&conversation, &[7, 7, 7]]);

        let found = held.lookup(&prompt, 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((stamp, conversation.len())));
        assert_eq!(
            held.trie.node_count(),
            2,
            "the root and one run of six blocks"
        );
    }

    #[test]
    fn a_divergence_splits_the_shared_run_and_each_branch_finds_its_own_entry() {
        let mut held = Held::with_capacity(4);
        let shared = run(100, 3);
        let left = joined(&[&shared, &run(500, 2)]);
        let right = joined(&[&shared, &run(900, 2)]);
        let left_stamp = held.add(&left).unwrap();
        let right_stamp = held.add(&right).unwrap();

        let found_left = held.lookup(&left, 100, SubBlock::ShortOnly);
        let found_right = held.lookup(&right, 100, SubBlock::ShortOnly);

        assert_eq!(held.trie.node_count(), 4, "root, shared run, two branches");
        assert_eq!(found_left, Some((left_stamp, left.len())));
        assert_eq!(found_right, Some((right_stamp, right.len())));
        audit(&held.trie);
    }

    #[test]
    fn removing_the_entry_that_owns_a_shared_run_hands_the_run_to_the_survivor() {
        let mut held = Held::with_capacity(4);
        let shared = run(100, 3);
        let first = joined(&[&shared, &run(500, 2)]);
        let second = joined(&[&shared, &run(900, 2)]);
        let first_stamp = held.add(&first).unwrap();
        let second_stamp = held.add(&second).unwrap();

        held.drop_entry(first_stamp);
        let found = held.lookup(&second, 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((second_stamp, second.len())));
        assert_eq!(held.trie.node_count(), 2, "the run and the branch merged");
        assert_eq!(audit(&held.trie), 1);
    }

    #[test]
    fn a_node_that_an_entry_ends_on_survives_when_the_longer_entry_through_it_is_removed() {
        let mut held = Held::with_capacity(4);
        let short = run(100, 2);
        let long = joined(&[&short, &run(500, 2)]);
        let short_stamp = held.add(&short).unwrap();
        let long_stamp = held.add(&long).unwrap();

        held.drop_entry(long_stamp);
        let found = held.lookup(&long, 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((short_stamp, short.len())));
        assert_eq!(held.trie.node_count(), 2);
        audit(&held.trie);
    }

    #[test]
    fn the_same_block_behind_different_parents_names_different_nodes() {
        let mut held = Held::with_capacity(4);
        let tail = run(700, 1);
        let behind_a = joined(&[&run(100, 1), &tail]);
        let behind_b = joined(&[&run(200, 1), &tail]);
        held.add(&behind_a).unwrap();
        let stamp_b = held.add(&behind_b).unwrap();

        let found = held.lookup(&behind_b, 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((stamp_b, behind_b.len())));
        assert_eq!(
            held.trie.node_count(),
            3,
            "two single-run nodes under the root"
        );
    }

    #[test]
    fn an_entry_shorter_than_a_block_is_found_without_any_node() {
        let mut held = Held::with_capacity(4);
        let stamp = held.add(&[2, 9, 9]).unwrap();

        let found = held.lookup(&[2, 9, 9, 5, 5, 5, 5], 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((stamp, 3)));
        assert_eq!(held.trie.node_count(), 1, "only the root");
    }

    #[test]
    fn a_prompt_sharing_the_first_tokens_but_not_a_block_finds_the_entry_when_asked_to() {
        let mut held = Held::with_capacity(4);
        let stored = joined(&[&[2, 10, 11], &run(300, 4)]);
        let stamp = held.add(&stored).unwrap();
        let prompt = [2, 10, 11, 99, 98, 97];

        let with_first_token = held.lookup(&prompt, 0, SubBlock::FirstToken);
        let short_only = held.lookup(&prompt, 0, SubBlock::ShortOnly);

        assert_eq!(with_first_token, Some((stamp, 3)));
        assert_eq!(short_only, None);
    }

    #[test]
    fn a_trie_at_its_capacity_refuses_the_next_entry_and_takes_it_after_a_removal() {
        let mut held = Held::with_capacity(2);
        let first = held.add(&run(100, 2)).unwrap();
        held.add(&run(200, 2)).unwrap();
        held.add(&run(300, 2)).unwrap();

        let refused = held.add(&run(400, 2));
        held.drop_entry(first);
        let accepted = held.add(&run(400, 2));

        assert_eq!(refused, Err(TrieError::Full { entry_capacity: 2 }));
        assert!(accepted.is_ok());
        audit(&held.trie);
    }

    #[test]
    fn identical_entries_are_both_held_and_the_later_stamp_wins_the_tie() {
        let mut held = Held::with_capacity(4);
        let conversation = run(100, 3);
        held.add(&conversation).unwrap();
        let later = held.add(&conversation).unwrap();

        let found = held.lookup(&conversation, 100, SubBlock::ShortOnly);

        assert_eq!(found, Some((later, conversation.len())));
        assert_eq!(audit(&held.trie), 2);
    }

    #[derive(Debug, Clone)]
    enum Operation {
        Add(Vec<u32>),
        Drop(usize),
        Ask(Vec<u32>, u32),
    }

    fn operation() -> impl Strategy<Value = Operation> {
        let tokens = |longest: usize| vec(0_u32..4, 0..longest);
        prop_oneof![
            4 => (tokens(6), tokens(30)).prop_map(|(head, tail)| Operation::Add(joined(&[&head, &tail]))),
            2 => (0_usize..16).prop_map(Operation::Drop),
            4 => (tokens(6), tokens(30), 0_u32..400)
                .prop_map(|(head, tail, floor)| Operation::Ask(joined(&[&head, &tail]), floor)),
        ]
    }

    fn apply(held: &mut Held, operation: Operation) {
        match operation {
            Operation::Add(ids) if !ids.is_empty() => {
                if held.entries.len() < 8 {
                    held.add(&ids).expect("within capacity");
                }
            }
            Operation::Drop(pick) if !held.entries.is_empty() => {
                let stamp = *held
                    .entries
                    .keys()
                    .nth(pick % held.entries.len())
                    .expect("a pick inside the held entries");
                held.drop_entry(stamp);
            }
            Operation::Ask(prompt, floor) if !prompt.is_empty() => {
                held.agrees_with_the_scan(&prompt, floor);
            }
            _ => {}
        }
        assert_eq!(audit(&held.trie), held.entries.len());
    }

    #[test]
    fn the_trie_answers_what_the_scan_answers_over_10000_generated_operation_sequences() {
        const SEQUENCES: usize = 10_000;
        let mut runner = TestRunner::new(Config {
            cases: SEQUENCES as u32,
            failure_persistence: None,
            ..Config::default()
        });
        let executed = std::cell::Cell::new(0_usize);

        runner
            .run(&vec(operation(), 0..40), |operations| {
                let mut held = Held::with_capacity(8);
                operations
                    .into_iter()
                    .for_each(|step| apply(&mut held, step));
                let stamps: Vec<u64> = held.entries.keys().copied().collect();
                stamps.into_iter().for_each(|stamp| held.drop_entry(stamp));
                assert_eq!(held.trie.node_count(), 0);
                assert_eq!(held.trie.link_count(), 0);
                executed.set(executed.get() + 1);
                Ok(())
            })
            .expect("the trie must agree with the scan on every sequence");

        assert_eq!(executed.get(), SEQUENCES);
    }

    #[test]
    fn ten_thousand_long_run_operations_agree_with_the_scan_and_free_every_node() {
        const OPERATIONS: usize = 10_000;
        let mut rng = fastrand::Rng::with_seed(13);
        let mut held = Held::with_capacity(16);
        let tokens = |rng: &mut fastrand::Rng, longest: usize| -> Vec<u32> {
            (0..rng.usize(0..longest)).map(|_| rng.u32(0..3)).collect()
        };
        let mut asked = 0_usize;

        for _ in 0..OPERATIONS {
            let head = tokens(&mut rng, 10);
            let tail = tokens(&mut rng, 40);
            let ids = joined(&[&head, &tail]);
            match rng.u8(0..10) {
                0..=3 if !ids.is_empty() => {
                    if held.entries.len() >= 16 {
                        let oldest = *held
                            .entries
                            .keys()
                            .next()
                            .expect("a full cache holds entries");
                        held.drop_entry(oldest);
                    }
                    held.add(&ids).expect("within capacity");
                }
                4..=5 if !held.entries.is_empty() => {
                    let stamp = *held
                        .entries
                        .keys()
                        .nth(rng.usize(0..held.entries.len()))
                        .expect("a pick inside the held entries");
                    held.drop_entry(stamp);
                }
                _ if !ids.is_empty() => {
                    held.agrees_with_the_scan(&ids, rng.u32(0..400));
                    asked += 1;
                }
                _ => {}
            }
            assert_eq!(audit(&held.trie), held.entries.len());
        }
        let stamps: Vec<u64> = held.entries.keys().copied().collect();
        stamps.into_iter().for_each(|stamp| held.drop_entry(stamp));

        assert!(asked > 2_000, "only {asked} lookups ran");
        assert_eq!(held.trie.node_count(), 0);
        assert_eq!(held.trie.link_count(), 0);
    }

    #[test]
    fn insert_lookup_and_remove_never_reach_the_allocator_once_the_arena_has_grown() {
        let shared = run(100, 3);
        let conversations: Vec<Vec<u32>> = (0..4)
            .map(|branch| joined(&[&shared, &run(500 + branch * 100, 4)]))
            .collect();
        let ids_of = |stamp: u64| conversations.get((stamp % 4) as usize).map(Vec::as_slice);
        let mut trie = PrefixTrie::new(BLOCK, 8);
        let probe = joined(&[&conversations[2], &[1, 2, 3]]);
        let cycle = |trie: &mut PrefixTrie, base: u64| {
            for stamp in base..base + 4 {
                trie.insert(stamp, &conversations[(stamp % 4) as usize], &ids_of)
                    .expect("within capacity");
            }
            trie.best(&probe, &ids_of, SubBlock::ShortOnly, |_, _| Some(0));
            for stamp in base..base + 4 {
                trie.remove(stamp, &conversations[(stamp % 4) as usize], &ids_of)
                    .expect("held");
            }
        };
        cycle(&mut trie, 0);

        let allocations = allocations_during(|| {
            for round in 1..=25_000_u64 {
                cycle(&mut trie, round * 4);
            }
        });

        assert_eq!(allocations, 0);
        assert_eq!(trie.node_count(), 0);
    }

    #[test]
    fn a_trie_reports_its_block_size_its_capacity_and_the_bytes_it_holds() {
        let mut held = Held::with_capacity(8);
        let empty_bytes = held.trie.byte_len();
        held.add(&run(100, 3)).unwrap();
        held.add(&run(200, 3)).unwrap();

        assert_eq!(held.trie.block_tokens(), BLOCK);
        assert_eq!(held.trie.entry_capacity(), 8);
        assert!(held.trie.byte_len() > empty_bytes);
        println!(
            "PREFIX_TRIE node_bytes={} bytes_for_two_entries_of_three_blocks={}",
            PrefixTrie::node_bytes(),
            held.trie.byte_len()
        );
    }

    #[test]
    fn a_node_stays_within_one_cache_line() {
        assert!(
            PrefixTrie::node_bytes() <= 64,
            "{}",
            PrefixTrie::node_bytes()
        );
    }
}
