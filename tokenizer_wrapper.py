import os
import sys
import json

try:
    from unitokenizer import UnigramTokenizer
except ImportError:
    UnigramTokenizer = None  # Rust tokenizer not installed — will use HF fallback

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Special tokens shared across both backends
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<BOS>", "<EOS>",
    "<|system|>", "[INST]", "[/INST]",
    "<|thought|>", "<|solution|>",
]


class Tokenizer:
    """
    Unified tokenizer wrapper supporting two backends:
    1. unitokenizer (custom Rust Unigram) — preferred, requires maturin build
    2. HuggingFace tokenizers (BPE) — fallback, pip install tokenizers
    
    Backend is auto-selected: unitokenizer if available, otherwise HF tokenizers.
    Force a backend with: Tokenizer(backend="rust") or Tokenizer(backend="hf")
    """
    def __init__(self, vocab_path="tokenizer_vocab.txt", backend=None):
        self.vocab_path = vocab_path
        
        # Auto-select backend
        if backend == "rust" or (backend is None and UnigramTokenizer is not None):
            if UnigramTokenizer is None:
                raise ImportError("unitokenizer not available. Build with 'maturin develop --release'.")
            self._backend = "rust"
            self._init_rust(vocab_path)
        elif backend == "hf" or (backend is None and HF_AVAILABLE):
            if not HF_AVAILABLE:
                raise ImportError("Install HuggingFace tokenizers: pip install tokenizers")
            self._backend = "hf"
            self._init_hf(vocab_path)
        else:
            raise ImportError(
                "No tokenizer backend available.\n"
                "Install one of:\n"
                "  1. Build unitokenizer: cd tokenizer && maturin develop --release\n"
                "  2. pip install tokenizers"
            )
        
        print(f"Tokenizer backend: {self._backend} (vocab: {self.vocab_size})")

    def _init_rust(self, vocab_path):
        """Initialize Rust unitokenizer backend."""
        if os.path.exists(vocab_path):
            self.model = UnigramTokenizer.load(vocab_path)
            self._vocab_size = self.model.vocab_size()
        else:
            print(f"Warning: Vocab file '{vocab_path}' not found. Initializing empty tokenizer.")
            self.model = UnigramTokenizer()
            self._vocab_size = 0

    def _init_hf(self, vocab_path):
        """Initialize HuggingFace BPE tokenizer backend."""
        hf_path = vocab_path.replace(".txt", ".json")
        if os.path.exists(hf_path):
            self.model = HFTokenizer.from_file(hf_path)
            self._vocab_size = self.model.get_vocab_size()
        else:
            print(f"Warning: HF vocab '{hf_path}' not found. Initializing empty tokenizer.")
            self.model = HFTokenizer(BPE(unk_token="<UNK>"))
            self.model.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.model.decoder = ByteLevelDecoder()
            self._vocab_size = 0

    def train(self, data_file, vocab_size=8000, min_freq=2):
        """Train the tokenizer on the provided text file."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data file not found: {data_file}")
        
        print(f"Training tokenizer ({self._backend}) on {data_file} (Target Vocab: {vocab_size})...")
        
        if self._backend == "rust":
            self.model.train(data_file, vocab_size, min_freq)
            self._vocab_size = self.model.vocab_size()
        else:
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_freq,
                special_tokens=SPECIAL_TOKENS,
                show_progress=True,
            )
            self.model = HFTokenizer(BPE(unk_token="<UNK>"))
            self.model.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.model.decoder = ByteLevelDecoder()
            self.model.train([data_file], trainer)
            self._vocab_size = self.model.get_vocab_size()
        
        print(f"Training complete. Vocab size: {self._vocab_size}")

    def save(self, path=None):
        """Save the current vocabulary to a file."""
        save_path = path if path else self.vocab_path
        
        if self._backend == "rust":
            self.model.save(save_path)
        else:
            hf_path = save_path.replace(".txt", ".json")
            self.model.save(hf_path)
            # Also save a .txt version with token\tscore format for compatibility
            vocab = self.model.get_vocab()
            with open(save_path, "w", encoding="utf-8") as f:
                for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\t{idx}\n")
        
        print(f"Tokenizer saved to {save_path}")

    def encode(self, text):
        """Encode a string into a list of integers."""
        if self._backend == "rust":
            if isinstance(text, str):
                text_bytes = text.encode('utf-8')
            elif isinstance(text, bytes):
                text_bytes = text
            else:
                raise ValueError(f"Expected str or bytes, got {type(text)}")
            return self.model.encode(text_bytes)
        else:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            return self.model.encode(text).ids

    def encode_batch(self, texts, batch_size=4096):
        """
        Encode a list of strings in parallel batches.
        HF tokenizers uses multi-threaded Rust internally — much faster than one-by-one.
        For Rust backend, uses native Rust batch encoding when available.
        Returns: list of list[int]
        """
        if self._backend == "hf":
            all_ids = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                # Ensure all items are strings
                chunk = [t.decode('utf-8', errors='replace') if isinstance(t, bytes) else t for t in chunk]
                encoded = self.model.encode_batch(chunk)
                all_ids.extend([e.ids for e in encoded])
            return all_ids
        else:
            if hasattr(self.model, "encode_batch"):
                all_ids = []
                for i in range(0, len(texts), batch_size):
                    chunk = texts[i:i + batch_size]
                    chunk = [t.encode('utf-8') if isinstance(t, str) else t for t in chunk]
                    all_ids.extend(self.model.encode_batch(chunk))
                return all_ids
            return [self.encode(t) for t in texts]

    def decode(self, tokens):
        """Decode a list of integers back into a string."""
        if self._backend == "rust":
            decoded_bytes = self.model.decode(tokens)
            return decoded_bytes.decode('utf-8', errors='replace')
        else:
            return self.model.decode(tokens)

    @property
    def vocab_size(self):
        """Return the size of the vocabulary."""
        if self._backend == "rust":
            if hasattr(self.model, 'vocab_size'):
                return self.model.vocab_size()
        else:
            return self.model.get_vocab_size()
        return self._vocab_size

    @classmethod
    def from_pretrained(cls, path, backend=None):
        """Alternative constructor to load from a specific path."""
        return cls(vocab_path=path, backend=backend)
