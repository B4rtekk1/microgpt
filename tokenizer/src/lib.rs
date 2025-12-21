use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_log;

static PRE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+")
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

        for i in start..text.len().min(start + 64) {
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
    encode_cache: Arc<RwLock<AHashMap<Arc<[u8]>, Vec<u32>>>>,
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

    fn decode(&self, tokens: Vec<u32>) -> Vec<u8> {
        let mut bytes_out: Vec<u8> = Vec::with_capacity(tokens.len() * 2);
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

    fn get_byte_token_id(&self, byte: u8) -> u32 {
        self.special_tokens.len() as u32 + byte as u32
    }

    fn viterbi_encode(&self, text: &[u8]) -> Vec<u32> {
        let n = text.len();
        if n == 0 {
            return Vec::new();
        }

        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_path_token = vec![0u32; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            if best_score[i].is_infinite() && best_score[i] < 0.0 {
                continue;
            }

            // Try to find longest matching token from trie
            if let Some((token_id, token_len)) = self.trie.longest_prefix_match(text, i) {
                let j = i + token_len;
                let score = best_score[i] + self.scores[token_id as usize];
                if score > best_score[j] {
                    best_score[j] = score;
                    best_path_token[j] = token_id;
                }
            }

            // Fallback: single byte token
            let byte_id = self.get_byte_token_id(text[i]);
            let score = best_score[i] + self.scores.get(byte_id as usize).copied().unwrap_or(-10.0);
            if score > best_score[i + 1] {
                best_score[i + 1] = score;
                best_path_token[i + 1] = byte_id;
            }
        }

        // Reconstruct path
        let mut result = Vec::new();
        let mut pos = n;

        while pos > 0 {
            let token_id = best_path_token[pos];
            result.push(token_id);

            let token_len = self.vocab[token_id as usize].len();
            pos = pos.saturating_sub(token_len);
        }

        result.reverse();
        result
    }

    fn encode_internal(&self, data: Vec<u8>) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::with_capacity(data.len());
        let text = String::from_utf8_lossy(&data);

        for mat in self.compiled_pattern.find_iter(&text) {
            let piece = mat.expect("regex failed").as_str();
            let piece_cs = CompactString::from(piece);

            // Check for special tokens
            if let Some(&id) = self.special_tokens.get(&piece_cs) {
                result.push(id);
                continue;
            }

            let piece_bytes = piece.as_bytes();
            let piece_arc = Arc::<[u8]>::from(piece_bytes);

            // Check cache
            {
                let cache_r = self.encode_cache.read();
                if let Some(cached) = cache_r.get(&piece_arc) {
                    result.extend_from_slice(cached);
                    continue;
                }
            }

            // Encode and cache
            let encoded = self.viterbi_encode(piece_bytes);

            {
                let mut cache_w = self.encode_cache.write();
                cache_w.insert(piece_arc, encoded.clone());
            }

            result.extend(encoded);
        }
        result
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
        let mut expected_freq: Vec<f64> = vec![0.0; vocab_len];

        for (word, count) in words {
            let encoding = self.viterbi_encode(word);
            for &token_id in &encoding {
                expected_freq[token_id as usize] += *count as f64;
            }
        }

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

        // Load and count words
        let file = File::open(&file_path)?;
        let reader = BufReader::new(file);
        let mut word_counts: AHashMap<Arc<[u8]>, i64> = AHashMap::new();

        for line in reader.lines() {
            let line = line?;
            for mat in PRE_REGEX.find_iter(&line) {
                if let Ok(m) = mat {
                    let piece = m.as_str().as_bytes();
                    if !piece.is_empty() {
                        *word_counts.entry(Arc::<[u8]>::from(piece)).or_default() += 1;
                    }
                }
            }
        }

        let words: Vec<(Arc<[u8]>, i64)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_freq as i64)
            .collect();

        // Collect candidate substrings
        let mut candidates: AHashMap<Arc<[u8]>, i64> = AHashMap::new();
        for (word, count) in &words {
            for len in 2..=word.len().min(20) {
                for start in 0..=(word.len() - len) {
                    let substr = Arc::<[u8]>::from(&word[start..start + len]);
                    *candidates.entry(substr).or_default() += count;
                }
            }
        }

        // Sort candidates by frequency
        let mut candidates_vec: Vec<(Arc<[u8]>, f64)> = candidates
            .into_iter()
            .filter(|(_, freq)| *freq >= 2)
            .map(|(piece, freq)| (piece, (freq as f64).ln()))
            .collect();

        candidates_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Seed vocabulary
        let base_size = self.base_vocab_size();
        let seed_size = target_vocab_size.saturating_sub(base_size).min(8000);
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
        let reader = BufReader::new(file);

        self.vocab.clear();
        self.scores.clear();
        self.trie.clear();

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

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let piece = parts[0].as_bytes().to_vec();
            let score: f64 = parts[1].parse().unwrap_or(0.0);

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
