use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_log;
use rayon::prelude::*;
use regex::Regex;

// Linear-time regex (O(n), no backtracking) — safe for math/LaTeX text
static PRE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+")
        .expect("valid regex")
});

// Special token IDs
const SPECIAL_TOKENS: &[&str] = &[
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<|system|>",
    "[INST]",
    "[/INST]",
    "<|thought|>",
    "<|solution|>",
];
const NUM_BYTE_TOKENS: usize = 256;
const MAX_TRIE_MATCH_LEN: usize = 64;
const MAX_ENCODE_CACHE_ENTRIES: usize = 200_000;
const MAX_ENCODE_RESERVE: usize = 1_048_576;
const TRAIN_BUFFER_SIZE: usize = 1 << 20;
const TRAIN_MAX_WORD_TYPES_MULTIPLIER: usize = 64;
const TRAIN_MIN_WORD_TYPES: usize = 200_000;
const TRAIN_MAX_CANDIDATE_TYPES_MULTIPLIER: usize = 32;
const TRAIN_MIN_CANDIDATE_TYPES: usize = 300_000;
const TRAIN_PRUNE_WATERMARK_MULTIPLIER: usize = 2;
const TRAIN_MAX_SUBSTRING_LEN: usize = 20;
const TRAIN_MAX_WORD_LEN: usize = 128;

// Trie for fast token lookup
#[derive(Default)]
struct ByteTrieNode {
    children: Vec<Option<Box<ByteTrieNode>>>,
    value: Option<u32>,
}

impl ByteTrieNode {
    fn new() -> Self {
        Self {
            children: (0..256).map(|_| None).collect(),
            value: None,
        }
    }
}

#[derive(Default)]
struct ByteTrie {
    root: ByteTrieNode,
}

impl ByteTrie {
    fn new() -> Self {
        Self {
            root: ByteTrieNode::new(),
        }
    }

    fn insert(&mut self, piece: &[u8], id: u32) {
        if piece.is_empty() {
            self.root.value = Some(id);
            return;
        }

        let mut node = &mut self.root;
        for &b in piece {
            let idx = b as usize;
            if node.children[idx].is_none() {
                node.children[idx] = Some(Box::new(ByteTrieNode::new()));
            }
            node = node.children[idx].as_mut().unwrap();
        }
        node.value = Some(id);
    }

    fn longest_prefix_match(&self, text: &[u8], start: usize) -> Option<(u32, usize)> {
        let mut best = None;
        let mut node = &self.root;
        let end = text.len().min(start + MAX_TRIE_MATCH_LEN);

        for i in start..end {
            let b = text[i];
            if let Some(ref child) = node.children[b as usize] {
                if let Some(id) = child.value {
                    best = Some((id, i - start + 1));
                }
                node = child.as_ref();
            } else {
                break;
            }
        }
        best
    }

    fn clear(&mut self) {
        self.root = ByteTrieNode::new();
    }
}

#[pyclass]
pub struct UnigramTokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f64>,
    special_tokens: AHashMap<CompactString, u32>,
    compiled_pattern: Regex,
    encode_cache: Arc<RwLock<AHashMap<Box<[u8]>, Box<[u32]>>>>,
    trie: ByteTrie,
}

#[pymethods]
impl UnigramTokenizer {
    #[new]
    #[pyo3(signature = (vocab_file=None))]
    fn new(vocab_file: Option<String>) -> PyResult<Self> {
        let _ = pyo3_log::try_init();

        let mut tokenizer = Self {
            vocab: Vec::with_capacity(1024),
            scores: Vec::with_capacity(1024),
            special_tokens: AHashMap::new(),
            compiled_pattern: PRE_REGEX.clone(),
            encode_cache: Arc::new(RwLock::new(AHashMap::new())),
            trie: ByteTrie::new(),
        };

        tokenizer.initialize_base_vocab();

        if let Some(path) = vocab_file {
            tokenizer.load_vocab(&path)?;
        }

        Ok(tokenizer)
    }

    #[staticmethod]
    fn load(vocab_file: String) -> PyResult<Self> {
        Self::new(Some(vocab_file))
    }

    fn encode(&self, data: Vec<u8>) -> Vec<u32> {
        self.encode_internal(data)
    }

    fn encode_batch(&self, py: Python<'_>, batch: Vec<Vec<u8>>) -> Vec<Vec<u32>> {
        py.detach(|| {
            batch
                .into_par_iter()
                .map(|data| self.encode_internal(data))
                .collect()
        })
    }

    fn decode(&self, tokens: Vec<u32>) -> Vec<u8> {
        let mut bytes_out: Vec<u8> =
            Vec::with_capacity(tokens.len().saturating_mul(2).min(MAX_ENCODE_RESERVE));
        for &id in &tokens {
            if let Some(token_bytes) = self.vocab.get(id as usize) {
                bytes_out.extend_from_slice(token_bytes);
            }
        }
        bytes_out
    }

    fn save(&self, vocab_file: String) -> PyResult<()> {
        let mut out = File::create(&vocab_file)?;

        for (piece, score) in self.vocab.iter().zip(self.scores.iter()) {
            if let Ok(s) = std::str::from_utf8(piece) {
                writeln!(out, "{}\t{}", s, score)?;
            } else {
                write!(out, "{:x?}\t{}\n", piece, score)?;
            }
        }

        Ok(())
    }

    fn train(&mut self, file_path: String, vocab_size: u32, min_freq: u32) -> PyResult<()> {
        self.train_internal(file_path, vocab_size as usize, min_freq)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl UnigramTokenizer {
    fn initialize_base_vocab(&mut self) {
        // Add special tokens
        for (i, &tok) in SPECIAL_TOKENS.iter().enumerate() {
            let bytes = tok.as_bytes().to_vec();
            self.vocab.push(bytes.clone());
            self.scores.push(0.0);
            self.special_tokens
                .insert(CompactString::from(tok), i as u32);
            self.trie.insert(&bytes, i as u32);
        }

        // Add byte fallback tokens
        for i in 0..NUM_BYTE_TOKENS {
            let bytes = vec![i as u8];
            let id = (self.special_tokens.len() + i) as u32;
            self.vocab.push(bytes.clone());
            self.scores.push(-10.0);
            self.trie.insert(&bytes, id);
        }
    }

    fn base_vocab_size(&self) -> usize {
        self.special_tokens.len() + NUM_BYTE_TOKENS
    }

    fn viterbi_encode(&self, text: &[u8]) -> Vec<u32> {
        let n = text.len();
        if n == 0 {
            return Vec::new();
        }

        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_path_token = vec![0u32; n + 1];
        let mut best_path_len = vec![0usize; n + 1];
        best_score[0] = 0.0;
        let byte_token_base = self.special_tokens.len() as u32;
        let scores = &self.scores;

        for i in 0..n {
            if best_score[i] == f64::NEG_INFINITY {
                continue;
            }

            // Try to find longest matching token from trie
            if let Some((token_id, token_len)) = self.trie.longest_prefix_match(text, i) {
                let j = i + token_len;
                let score = best_score[i] + scores[token_id as usize];
                if score > best_score[j] {
                    best_score[j] = score;
                    best_path_token[j] = token_id;
                    best_path_len[j] = token_len;
                }
            }

            // Fallback: single byte token
            let byte_id = byte_token_base + text[i] as u32;
            let score = best_score[i] + scores[byte_id as usize];
            if score > best_score[i + 1] {
                best_score[i + 1] = score;
                best_path_token[i + 1] = byte_id;
                best_path_len[i + 1] = 1;
            }
        }

        // Reconstruct path
        let mut result = Vec::new();
        let mut pos = n;

        while pos > 0 {
            let token_id = best_path_token[pos];
            result.push(token_id);

            let mut token_len = best_path_len[pos];
            if token_len == 0 {
                token_len = self
                    .vocab
                    .get(token_id as usize)
                    .map(|piece| piece.len().max(1))
                    .unwrap_or(1);
            }
            pos -= token_len.min(pos);
        }

        result.reverse();
        result
    }

    fn encode_internal(&self, data: Vec<u8>) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::with_capacity(data.len().min(MAX_ENCODE_RESERVE));
        let text = String::from_utf8_lossy(&data);

        for mat in self.compiled_pattern.find_iter(&text) {
            let piece = mat.as_str();

            // Check for special tokens
            if let Some(&id) = self.special_tokens.get(piece) {
                result.push(id);
                continue;
            }

            let piece_bytes = piece.as_bytes();

            // Check cache
            {
                let cache_r = self.encode_cache.read();
                if let Some(cached) = cache_r.get(piece_bytes) {
                    result.extend_from_slice(cached);
                    continue;
                }
            }

            // Encode and cache
            let encoded = self.viterbi_encode(piece_bytes);
            result.extend_from_slice(&encoded);
            {
                let mut cache_w = self.encode_cache.write();
                if cache_w.len() >= MAX_ENCODE_CACHE_ENTRIES {
                    cache_w.clear();
                }
                cache_w.insert(
                    Box::<[u8]>::from(piece_bytes),
                    encoded.into_boxed_slice(),
                );
            }
        }
        result
    }

    fn prune_top_k_counts(map: &mut AHashMap<Arc<[u8]>, i64>, keep: usize) {
        if keep == 0 || map.len() <= keep {
            return;
        }

        let mut entries: Vec<(Arc<[u8]>, i64)> = map.drain().collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(keep);
        map.extend(entries);
    }

    fn merge_counts_bounded(
        dst: &mut AHashMap<Arc<[u8]>, i64>,
        src: AHashMap<Arc<[u8]>, i64>,
        cap: usize,
    ) {
        for (k, v) in src {
            *dst.entry(k).or_default() += v;
        }

        if dst.len() > cap.saturating_mul(TRAIN_PRUNE_WATERMARK_MULTIPLIER) {
            Self::prune_top_k_counts(dst, cap);
        }
    }

    fn count_words_batch(lines: &[String]) -> AHashMap<Arc<[u8]>, i64> {
        lines
            .par_iter()
            .fold(
                || AHashMap::new(),
                |mut local_counts: AHashMap<Arc<[u8]>, i64>, line| {
                    for mat in PRE_REGEX.find_iter(line.as_str()) {
                        let piece = mat.as_str().as_bytes();
                        if piece.is_empty() || piece.len() > TRAIN_MAX_WORD_LEN {
                            continue;
                        }

                        *local_counts.entry(Arc::<[u8]>::from(piece)).or_default() += 1;
                    }
                    local_counts
                },
            )
            .reduce(
                || AHashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_default() += v;
                    }
                    a
                },
            )
    }

    fn stream_word_counts(
        &self,
        file_path: &str,
        max_word_types: usize,
    ) -> PyResult<AHashMap<Arc<[u8]>, i64>> {
        let file = File::open(file_path)?;
        let reader = BufReader::with_capacity(TRAIN_BUFFER_SIZE, file);

        let mut global_counts: AHashMap<Arc<[u8]>, i64> = AHashMap::new();
        let mut batch: Vec<String> = Vec::with_capacity(8192);

        for line in reader.lines() {
            let line = line?;
            if !line.is_empty() {
                batch.push(line);
            }

            if batch.len() >= batch.capacity() {
                let local_counts = Self::count_words_batch(&batch);
                Self::merge_counts_bounded(&mut global_counts, local_counts, max_word_types);
                batch.clear();
            }
        }

        if !batch.is_empty() {
            let local_counts = Self::count_words_batch(&batch);
            Self::merge_counts_bounded(&mut global_counts, local_counts, max_word_types);
        }

        Self::prune_top_k_counts(&mut global_counts, max_word_types);
        Ok(global_counts)
    }

    fn collect_candidates(
        &self,
        words: &[(Arc<[u8]>, i64)],
        max_candidate_types: usize,
    ) -> AHashMap<Arc<[u8]>, i64> {
        let thread_count = rayon::current_num_threads().max(1);
        let per_thread_cap = (max_candidate_types / thread_count).max(10_000);

        let mut candidates: AHashMap<Arc<[u8]>, i64> = words
            .par_iter()
            .fold(
                || AHashMap::new(),
                |mut local_cands: AHashMap<Arc<[u8]>, i64>, (word, count)| {
                    let bytes = word.as_ref();
                    let max_len = bytes.len().min(TRAIN_MAX_SUBSTRING_LEN);
                    if max_len < 2 {
                        return local_cands;
                    }

                    for len in 2..=max_len {
                        for start in 0..=(bytes.len() - len) {
                            let span = &bytes[start..start + len];
                            *local_cands.entry(Arc::<[u8]>::from(span)).or_default() += *count;
                        }
                    }

                    if local_cands.len()
                        > per_thread_cap.saturating_mul(TRAIN_PRUNE_WATERMARK_MULTIPLIER)
                    {
                        Self::prune_top_k_counts(&mut local_cands, per_thread_cap);
                    }

                    local_cands
                },
            )
            .reduce(
                || AHashMap::new(),
                |mut a, b| {
                    Self::merge_counts_bounded(&mut a, b, max_candidate_types);
                    a
                },
            );

        Self::prune_top_k_counts(&mut candidates, max_candidate_types);
        candidates
    }

    fn reset_to_base_vocab(&mut self) {
        let base_size = self.base_vocab_size();
        self.vocab.truncate(base_size);
        self.scores.truncate(base_size);
        self.trie.clear();

        // Rebuild trie with base vocab
        for (id, piece) in self.vocab.iter().enumerate() {
            self.trie.insert(piece, id as u32);
        }

        self.encode_cache.write().clear();
    }

    fn compute_em_step(&self, words: &[(Arc<[u8]>, i64)]) -> Vec<f64> {
        let vocab_len = self.vocab.len();

        // Parallel fold+reduce: each rayon thread accumulates local freq vector
        let expected_freq: Vec<f64> = words
            .par_iter()
            .fold(
                || vec![0.0f64; vocab_len],
                |mut local_freq, (word, count)| {
                    let encoding = self.viterbi_encode(word);
                    for &token_id in &encoding {
                        local_freq[token_id as usize] += *count as f64;
                    }
                    local_freq
                },
            )
            .reduce(
                || vec![0.0f64; vocab_len],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += *y;
                    }
                    a
                },
            );

        let total: f64 = expected_freq.iter().sum();
        if total > 0.0 {
            expected_freq
                .iter()
                .map(|&freq| {
                    if freq > 0.0 {
                        (freq / total).ln()
                    } else {
                        -100.0
                    }
                })
                .collect()
        } else {
            vec![-100.0; vocab_len]
        }
    }

    fn prune_vocab(&mut self, target_size: usize, shrink_ratio: f64) {
        let base_size = self.base_vocab_size();
        let to_keep = ((self.vocab.len() as f64 * shrink_ratio) as usize)
            .max(base_size)
            .min(target_size);

        // Sort learnable tokens by score
        let mut token_scores: Vec<(usize, f64)> = (base_size..self.vocab.len())
            .map(|i| (i, self.scores[i]))
            .collect();

        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Keep base vocab + top scored tokens
        let mut keep_set: AHashSet<usize> = (0..base_size).collect();
        for &(idx, _) in token_scores.iter().take(to_keep - base_size) {
            keep_set.insert(idx);
        }

        // Rebuild vocabulary
        let mut new_vocab = Vec::new();
        let mut new_scores = Vec::new();
        let mut new_trie = ByteTrie::new();

        for (old_id, (piece, score)) in self.vocab.iter().zip(self.scores.iter()).enumerate() {
            if keep_set.contains(&old_id) {
                let new_id = new_vocab.len() as u32;
                new_vocab.push(piece.clone());
                new_scores.push(*score);
                new_trie.insert(piece, new_id);
            }
        }

        self.vocab = new_vocab;
        self.scores = new_scores;
        self.trie = new_trie;
    }

    fn train_internal(
        &mut self,
        file_path: String,
        target_vocab_size: usize,
        min_freq: u32,
    ) -> PyResult<()> {
        self.reset_to_base_vocab();

        let max_word_types = (target_vocab_size.saturating_mul(TRAIN_MAX_WORD_TYPES_MULTIPLIER))
            .max(TRAIN_MIN_WORD_TYPES);
        let max_candidate_types = (target_vocab_size
            .saturating_mul(TRAIN_MAX_CANDIDATE_TYPES_MULTIPLIER))
        .max(TRAIN_MIN_CANDIDATE_TYPES);

        let mut words: Vec<(Arc<[u8]>, i64)> = self
            .stream_word_counts(&file_path, max_word_types)?
            .into_iter()
            .filter(|(_, count)| *count >= min_freq as i64)
            .collect();

        if words.is_empty() {
            return Ok(());
        }

        words.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let candidates = self.collect_candidates(&words, max_candidate_types);

        // Sort candidates by frequency
        let mut candidates_vec: Vec<(Arc<[u8]>, f64)> = candidates
            .into_iter()
            .filter(|(_, freq)| *freq >= 2)
            .map(|(piece, freq)| (piece, (freq as f64).ln()))
            .collect();

        candidates_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Seed vocabulary — start with 4x target so EM+pruning can converge to target_vocab_size
        let base_size = self.base_vocab_size();
        let seed_size = (target_vocab_size * 4).saturating_sub(base_size);
        for (piece, score) in candidates_vec.into_iter().take(seed_size) {
            let id = self.vocab.len() as u32;
            self.vocab.push(piece.to_vec());
            self.scores.push(score);
            self.trie.insert(&piece, id);
        }

        // EM + pruning iterations
        while self.vocab.len() > target_vocab_size {
            // EM step
            self.scores = self.compute_em_step(&words);

            // Pruning step
            self.prune_vocab(target_vocab_size, 0.75);
        }

        // Final EM step
        self.scores = self.compute_em_step(&words);

        Ok(())
    }

    fn load_vocab(&mut self, vocab_file: &str) -> PyResult<()> {
        let file = File::open(vocab_file)?;
        let reader = BufReader::with_capacity(TRAIN_BUFFER_SIZE, file);

        self.vocab.clear();
        self.scores.clear();
        self.trie.clear();
        self.encode_cache.write().clear();

        // Collect special tokens data before mutating self
        let special_tokens_data: Vec<(Vec<u8>, u32)> = self
            .special_tokens
            .iter()
            .map(|(tok_name, &tok_id)| (tok_name.as_bytes().to_vec(), tok_id))
            .collect();

        // Restore special tokens
        for (bytes, tok_id) in special_tokens_data {
            self.ensure_vocab_size(tok_id as usize + 1);
            self.vocab[tok_id as usize] = bytes.clone();
            self.trie.insert(&bytes, tok_id);
        }

        // Restore byte tokens
        for i in 0..NUM_BYTE_TOKENS {
            let id = self.special_tokens.len() + i;
            let bytes = vec![i as u8];
            self.ensure_vocab_size(id + 1);
            self.vocab[id] = bytes.clone();
            self.scores[id] = -10.0;
            self.trie.insert(&[i as u8], id as u32);
        }

        // Load vocabulary from file
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.splitn(2, '\t');
            let Some(piece_str) = parts.next() else {
                continue;
            };
            let Some(score_str) = parts.next() else {
                continue;
            };

            let piece = piece_str.as_bytes().to_vec();
            let score: f64 = score_str.parse().unwrap_or(0.0);

            let id = self.vocab.len() as u32;
            self.vocab.push(piece.clone());
            self.scores.push(score);
            self.trie.insert(&piece, id);
        }

        Ok(())
    }

    fn ensure_vocab_size(&mut self, size: usize) {
        while self.vocab.len() < size {
            self.vocab.push(Vec::new());
            self.scores.push(0.0);
        }
    }
}

#[pymodule]
fn unitokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = pyo3_log::try_init();
    m.add_class::<UnigramTokenizer>()?;
    Ok(())
}
