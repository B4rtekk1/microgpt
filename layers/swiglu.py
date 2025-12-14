"""
SwiGLU - Swish-Gated Linear Unit activation function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_config as mc

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit activation function) - modern activation function for FFN.

    SwiGLU is variant of GLU (Gated Linear Unit) that uses the Swish/SiLU function as the gate instead of sigmoid.

    FFN architecture with SwiGLU:
        FFN(x) = W2 * (SiLU(W1 * x) ⊙ (W3 * x))
    
    Where:
        - SiLU(x) = x * sigmoid(x) (also called swish)
        - ⊙ denotes element-wise multiplication
    
    Hidden dimension:
        LLaMA uses hidden_dim = (2/3) * 4 * n_embd instead of 4 * n_embd because we have 3 projections instead of 2
    
    Attributes:
        n_embd: Input/output dimension
        hidden_dim: Intermediate (hidden) dimension
        w1: Gate projection (for SiLU activation)
        w2: Down projection (from hidden_dim to n_embd)  
        w3: Up projection (for element-wise multiplication with gate)
    
    Shape:
        - Input: (batch_size, seq_len, n_embd)
        - Output: (batch_size, seq_len, n_embd)
        
    References:
        - "GLU Variants Improve Transformer" (Shazeer, 2020)
        - https://arxiv.org/abs/2002.05202
    """
    def __init__(self, model_config: mc.ModelConfig):
        """
        Initialize SwiGLU FFN.

        Args:
            model_config: Model configuration containing n_embd
        """
        super().__init__()
        self.config = model_config
        self.n_embd = self.config.n_embd
        self.hidden_dim = int(2*(4*self.n_embd) / 3)
        self.hidden_dim = ((self.hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.n_embd, bias=False)
        self.w3 = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
    
    def forward(self, hidden_states: torch.Tensor):
        """Pass input through SwiGLU FFN.

        Steps:
            1. gate = Silu(W1 @ x)
            2. up = W3 @ x
            3. hidden = gate ⊙ up
            4. output = W2 @ hidden

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch, seq_len, n_embd).
        """
        gate = F.silu(self.w1(hidden_states))
        up = self.w3(hidden_states)
        hidden_states = gate * up
        hidden_states = self.w2(hidden_states)
        return hidden_states

    def extra_repr(self) -> str:
        return f"n_emb={self.n_embd}, hidden_dim={self.hidden_dim}"