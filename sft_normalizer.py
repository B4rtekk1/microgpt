"""
SFT Dataset Normalizer

Converts various SFT dataset formats to a unified instruction format:
[INST] user message [/INST] assistant response

Supported datasets:
- sft-ultra (HuggingFaceH4/ultrachat_200k): messages format
- sft-hermes (teknium/OpenHermes-2.5): conversations format  
- sft (OpenAssistant/oasst_top1): messages format
- math-metamath (meta-math/MetaMathQA): query/response format
- math-sft (openai/gsm8k): question/answer format
- math-sft-plus (AI-MO/NuminaMath-CoT): problem/solution format
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SFTNormalizer:
    """Normalizes various SFT dataset formats to a unified instruction format."""
    
    # Token definitions - must match tokenizer/src/lib.rs SPECIAL_TOKENS
    INST_START = "[INST]"
    INST_END = "[/INST]"
    SYS_START = "<|system|>"
    SYS_END = "<|system|>"
    
    # Dataset format mappings
    DATASET_FORMATS = {
        # Chat/conversation datasets
        "sft-ultra": "ultrachat",
        "sft-hermes": "openhermes",
        "sft": "openassistant",
        
        # Math datasets
        "math-metamath": "metamath",
        "math-sft": "gsm8k",
        "math-sft-plus": "numina",
        
        # Generic formats (auto-detect)
        "generic": "auto"
    }
    
    def __init__(self, add_eos: bool = True, eos_token: str = "<EOS>"):
        """
        Initialize the normalizer.
        
        Args:
            add_eos: Whether to add EOS token at the end of formatted text
            eos_token: The EOS token to use
        """
        self.add_eos = add_eos
        self.eos_token = eos_token
    
    def normalize(self, item: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
        """
        Normalize a single dataset item to instruction format.
        
        Args:
            item: Dictionary containing the dataset item
            dataset_name: Optional dataset name for format hints
            
        Returns:
            Formatted string in [INST] ... [/INST] format
        """
        # Try to get format from dataset name
        format_type = self.DATASET_FORMATS.get(dataset_name, "auto")
        
        # Route to appropriate formatter
        if format_type == "ultrachat":
            result = self._format_ultrachat(item)
        elif format_type == "openhermes":
            result = self._format_openhermes(item)
        elif format_type == "openassistant":
            result = self._format_openassistant(item)
        elif format_type == "metamath":
            result = self._format_metamath(item)
        elif format_type == "gsm8k":
            result = self._format_gsm8k(item)
        elif format_type == "numina":
            result = self._format_numina(item)
        else:
            result = self._auto_format(item)
        
        # Add EOS if configured
        if self.add_eos and result and not result.endswith(self.eos_token):
            result = result.strip() + " " + self.eos_token
            
        return result
    
    # =========================================================================
    # Chat/Conversation Formatters
    # =========================================================================
    
    def _format_ultrachat(self, item: Dict[str, Any]) -> str:
        """
        Format UltraChat 200k dataset.
        
        Expected format:
        {
            "prompt": "...",
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }
        """
        if 'messages' not in item:
            return self._auto_format(item)
            
        messages = item['messages']
        if not isinstance(messages, list):
            return self._auto_format(item)
            
        return self._format_message_list(messages)
    
    def _format_openhermes(self, item: Dict[str, Any]) -> str:
        """
        Format OpenHermes 2.5 dataset.
        
        Expected format:
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
                ...
            ],
            "system": "..." (optional)
        }
        
        OR ShareGPT format:
        {
            "conversations": [
                {"from": "system", "value": "..."},
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
            ]
        }
        """
        conversations = item.get('conversations', [])
        if not conversations:
            return self._auto_format(item)
        
        # Extract system message if present
        system_msg = item.get('system', '')
        
        formatted_parts = []
        current_user = None
        
        for turn in conversations:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))
            
            if role in ['system']:
                system_msg = content
            elif role in ['human', 'user']:
                current_user = content
            elif role in ['gpt', 'assistant'] and current_user:
                # Build instruction with optional system prompt
                if system_msg and not formatted_parts:  # Add system only at start
                    user_content = f"{self.SYS_START}\n{system_msg}\n{self.SYS_END}\n\n{current_user}"
                else:
                    user_content = current_user
                    
                formatted_parts.append(
                    f"{self.INST_START} {user_content} {self.INST_END} {content}"
                )
                current_user = None
        
        return " ".join(formatted_parts)
    
    def _format_openassistant(self, item: Dict[str, Any]) -> str:
        """
        Format OpenAssistant dataset.
        
        Expected format:
        {
            "text": "..." (pre-formatted)
            OR
            "messages": [{"role": "...", "content": "..."}, ...]
        }
        """
        # If already formatted as text
        if 'text' in item and isinstance(item['text'], str):
            return item['text']
            
        # Format from messages
        if 'messages' in item:
            return self._format_message_list(item['messages'])
            
        return self._auto_format(item)
    
    def _format_message_list(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format a standard messages list with role/content pairs.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            
        Returns:
            Formatted string
        """
        formatted_parts = []
        current_user = None
        system_msg = None
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                system_msg = content
            elif role == 'user':
                current_user = content
            elif role == 'assistant' and current_user:
                # Include system message in first turn
                if system_msg and not formatted_parts:
                    user_content = f"{self.SYS_START}\n{system_msg}\n{self.SYS_END}\n\n{current_user}"
                else:
                    user_content = current_user
                    
                formatted_parts.append(
                    f"{self.INST_START} {user_content} {self.INST_END} {content}"
                )
                current_user = None
        
        return " ".join(formatted_parts)
    
    # =========================================================================
    # Math Dataset Formatters
    # =========================================================================
    
    def _format_metamath(self, item: Dict[str, Any]) -> str:
        """
        Format MetaMathQA dataset.
        
        Expected format:
        {
            "query": "...",
            "response": "...",
            "type": "..." (optional, e.g., "GSM_Rephrased", "MATH_AnsAug")
        }
        """
        query = item.get('query', item.get('question', ''))
        response = item.get('response', item.get('answer', ''))
        
        if not query or not response:
            return self._auto_format(item)
            
        return f"{self.INST_START} {query.strip()} {self.INST_END} {response.strip()}"
    
    def _format_gsm8k(self, item: Dict[str, Any]) -> str:
        """
        Format GSM8K dataset.
        
        Expected format:
        {
            "question": "...",
            "answer": "..."
        }
        """
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        if not question or not answer:
            return self._auto_format(item)
            
        return f"{self.INST_START} {question.strip()} {self.INST_END} {answer.strip()}"
    
    def _format_numina(self, item: Dict[str, Any]) -> str:
        """
        Format NuminaMath-CoT dataset.
        
        Expected format:
        {
            "problem": "...",
            "solution": "...",
            "source": "..." (optional, e.g., "olympiads", "amc_aime")
        }
        """
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        if not problem or not solution:
            return self._auto_format(item)
        
        # Add chain-of-thought instruction prefix for math problems
        instruction = f"Solve this math problem step by step:\n\n{problem.strip()}"
        
        return f"{self.INST_START} {instruction} {self.INST_END} {solution.strip()}"
    
    # =========================================================================
    # Auto-detection Formatter
    # =========================================================================
    
    def _auto_format(self, item: Dict[str, Any]) -> str:
        """
        Auto-detect and format unknown dataset formats.
        
        Tries to detect format based on available keys.
        """
        # Already formatted text
        if 'text' in item and isinstance(item['text'], str):
            text = item['text']
            # Check if already in instruction format
            if self.INST_START in text:
                return text
            return text
        
        # Messages format (OpenAI/Anthropic style)
        if 'messages' in item and isinstance(item['messages'], list):
            return self._format_message_list(item['messages'])
        
        # Conversations format (ShareGPT style)
        if 'conversations' in item and isinstance(item['conversations'], list):
            return self._format_openhermes(item)
        
        # Question/Answer pairs
        if 'question' in item and 'answer' in item:
            return f"{self.INST_START} {item['question']} {self.INST_END} {item['answer']}"
        
        # Problem/Solution pairs
        if 'problem' in item and 'solution' in item:
            return f"{self.INST_START} {item['problem']} {self.INST_END} {item['solution']}"
        
        # Query/Response pairs (MetaMath style)
        if 'query' in item and 'response' in item:
            return f"{self.INST_START} {item['query']} {self.INST_END} {item['response']}"
        
        # Instruction/Output pairs (Alpaca style)
        if 'instruction' in item:
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item.get('output', item.get('response', ''))
            
            if input_text:
                instruction = f"{instruction}\n\nInput: {input_text}"
                
            if output:
                return f"{self.INST_START} {instruction} {self.INST_END} {output}"
            else:
                return f"{self.INST_START} {instruction} {self.INST_END}"
        
        # Prompt/Completion pairs (older format)
        if 'prompt' in item and 'completion' in item:
            return f"{self.INST_START} {item['prompt']} {self.INST_END} {item['completion']}"
        
        # Prompt/Response pairs
        if 'prompt' in item and 'response' in item:
            return f"{self.INST_START} {item['prompt']} {self.INST_END} {item['response']}"
        
        # Fallback: convert entire item to string
        logger.warning(f"Unknown dataset format. Keys: {list(item.keys())}")
        return str(item)


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_sft_dataset(
    items: List[Dict[str, Any]], 
    dataset_name: Optional[str] = None,
    add_eos: bool = True
) -> List[str]:
    """
    Normalize a list of SFT dataset items.
    
    Args:
        items: List of dataset items to normalize
        dataset_name: Optional dataset name for format detection
        add_eos: Whether to add EOS token
        
    Returns:
        List of normalized instruction strings
    """
    normalizer = SFTNormalizer(add_eos=add_eos)
    return [normalizer.normalize(item, dataset_name) for item in items]


def get_supported_datasets() -> List[str]:
    """Return list of explicitly supported dataset formats."""
    return list(SFTNormalizer.DATASET_FORMATS.keys())


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with different formats
    normalizer = SFTNormalizer(add_eos=False)
    
    # Test UltraChat format
    ultrachat_sample = {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."}
        ]
    }
    print("UltraChat format:")
    print(normalizer.normalize(ultrachat_sample, "sft-ultra"))
    print()
    
    # Test OpenHermes format
    openhermes_sample = {
        "conversations": [
            {"from": "human", "value": "Explain quantum computing."},
            {"from": "gpt", "value": "Quantum computing uses quantum mechanics..."}
        ]
    }
    print("OpenHermes format:")
    print(normalizer.normalize(openhermes_sample, "sft-hermes"))
    print()
    
    # Test MetaMath format
    metamath_sample = {
        "query": "If x + 5 = 10, what is x?",
        "response": "To solve this equation:\nx + 5 = 10\nx = 10 - 5\nx = 5"
    }
    print("MetaMath format:")
    print(normalizer.normalize(metamath_sample, "math-metamath"))
    print()
    
    # Test NuminaMath format
    numina_sample = {
        "problem": "Find all prime numbers less than 10.",
        "solution": "We check each number:\n2 is prime\n3 is prime\n5 is prime\n7 is prime\nAnswer: 2, 3, 5, 7"
    }
    print("NuminaMath format:")
    print(normalizer.normalize(numina_sample, "math-sft-plus"))
