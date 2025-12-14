import os
import sys

try:
    from unitokenizer import UnigramTokenizer
except ImportError:
    UnigramTokenizer = None

class Tokenizer:
    """
    A Python wrapper for the custom Rust-based UnigramTokenizer.
    """
    def __init__(self, vocab_path="tokenizer_vocab.txt"):
        if UnigramTokenizer is None:
            raise ImportError(
                "Could not import 'unitokenizer'. "
                "Please ensure the Rust extension is built and installed (e.g., 'maturin develop --release')."
            )
        
        self.vocab_path = vocab_path
        
        if os.path.exists(vocab_path):
            self.model = UnigramTokenizer.load(vocab_path)
            self._vocab_size = self.model.vocab_size()
        else:
            # Initialize an empty tokenizer if vocab doesn't exist (e.g. wrapper for training)
            print(f"Warning: Vocab file '{vocab_path}' not found. Initializing empty tokenizer.")
            self.model = UnigramTokenizer()
            self._vocab_size = 0

    def train(self, data_file, vocab_size=8000, min_freq=2):
        """
        Train the tokenizer on the provided text file.
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data file not found: {data_file}")
            
        print(f"Training tokenizer on {data_file} (Target Vocab: {vocab_size})...")
        self.model.train(data_file, vocab_size, min_freq)
        self._vocab_size = self.model.vocab_size()
        print(f"Training complete. Vocab size: {self._vocab_size}")

    def save(self, path=None):
        """
        Save the current vocabulary to a file.
        """
        save_path = path if path else self.vocab_path
        self.model.save(save_path)
        print(f"Tokenizer saved to {save_path}")

    def encode(self, text):
        """
        Encode a string into a list of integers.
        Handles string <-> bytes conversion required by the Rust lib.
        """
        if isinstance(text, str):
            text_bytes = text.encode('utf-8')
        elif isinstance(text, bytes):
            text_bytes = text
        else:
            raise ValueError(f"Expected str or bytes, got {type(text)}")
            
        return self.model.encode(text_bytes)

    def decode(self, tokens):
        """
        Decode a list of integers back into a string.
        """
        # Rust lib returns bytes, we decode to string
        decoded_bytes = self.model.decode(tokens)
        return decoded_bytes.decode('utf-8', errors='replace')

    @property
    def vocab_size(self):
        """
        Return the size of the vocabulary.
        """
        # Refresh in case it changed (e.g. after training)
        if hasattr(self.model, 'vocab_size'):
            return self.model.vocab_size()
        return self._vocab_size

    @classmethod
    def from_pretrained(cls, path):
        """
        Alternative constructor to load from a specific path.
        """
        return cls(vocab_path=path)
