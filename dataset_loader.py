"""
Dataset loader for downloading and managing datasets from HuggingFace.
Supports different training stages: base (pre-training), mid (knowledge), sft (instruction tuning).
Also supports loading local JSONL files from the dataset/ folder.

Usage:
    loader = DatasetLoader()
    
    # Load from HuggingFace (streaming)
    ds = loader.load("base", streaming=True)
    
    # Load from local JSONL file (prefix with "local:")
    ds = loader.load("local:fineweb2", streaming=True)
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Iterator, Generator
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to local datasets
LOCAL_DATASET_DIR = Path(__file__).parent / "dataset"


class LocalJSONLDataset:
    """Iterator for local JSONL files that mimics HuggingFace dataset behavior."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


class DatasetLoader:
    """
    Class for managing dataset downloads from HuggingFace for different training stages.
    
    Supports:
    - base: language learning (pre-training)
    - mid: factual knowledge (knowledge injection)
    - sft: chatbot (instruction tuning)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            cache_dir: Optional path to cache directory for datasets.
        """
        self.cache_dir = cache_dir
        self.datasets_config = {
            "base": {
                "path": "HuggingFaceFW/fineweb-edu",
                "name": "sample-10BT",
                "split": "train"
            },
            "wikitext": {
                "path": "wikitext",
                "name": "wikitext-103-raw-v1",
                "split": "train"
            },
            "fineweb-edu": {
                "path": "HuggingFaceFW/fineweb-edu",
                "name": "sample-10BT",
                "split": "train"
            },
            "smollm": {
                "path": "HuggingFaceTB/smollm-corpus",
                "name": "cosmopedia-v2",
                "split": "train"
            },
            "cosmopedia": {
                "path": "HuggingFaceTB/cosmopedia-v2",
                "name": "default",
                "split": "train"
            },
            "mid": {
                "path": "wikipedia",
                "name": "20220301.en", 
                "split": "train",
                "extra_args": {"trust_remote_code": True}
            },
            "sft": {
                "path": "OpenAssistant/oasst_top1_2023-08-25",
                "name": None,
                "split": "train"
            },
            "math-knowledge": {
                "path": "HuggingFaceTB/finemath",
                "name": "finemath-3plus",
                "split": "train"
            },
            "math-pretrain": {
                "path": "open-web-math/open-web-math",
                "name": None,
                "split": "train"
            },
            "math-sft": {
                "path": "openai/gsm8k",
                "name": "main",
                "split": "train"
            },
            "math-metamath": {
                "path": "meta-math/MetaMathQA",
                "name": None,
                "split": "train"
            },
            "math-sft-plus": {
                "path": "AI-MO/NuminaMath-CoT",
                "name": None,
                "split": "train"
            },
            "sft-ultra": {
                "path": "HuggingFaceH4/ultrachat_200k",
                "name": "default",
                "split": "train_sft"
            },
            "sft-hermes": {
                "path": "teknium/OpenHermes-2.5",
                "name": None,
                "split": "train"
            }
        }

    def load(self, stage: str, **kwargs) -> Union[Dataset, DatasetDict, LocalJSONLDataset]:
        """
        Download and return dataset for the given stage.

        Args:
            stage: Stage name like 'base', 'mid', 'sft', or 'local:filename' for local JSONL files.
            **kwargs: Additional arguments passed to load_dataset.

        Returns:
            Dataset, DatasetDict, or LocalJSONLDataset iterator.
        """
        # Handle local datasets (prefix with "local:")
        if stage.startswith("local:"):
            dataset_name = stage[6:]  # Remove "local:" prefix
            return self._load_local(dataset_name)
        
        if stage not in self.datasets_config:
            valid_keys = ", ".join(self.datasets_config.keys())
            local_files = [f.stem for f in LOCAL_DATASET_DIR.glob("*.jsonl")] if LOCAL_DATASET_DIR.exists() else []
            local_hint = f" Local datasets available: {', '.join(local_files)}" if local_files else ""
            raise ValueError(f"Unknown stage: '{stage}'. Available stages: {valid_keys}.{local_hint}")

        config = self.datasets_config[stage]
        logger.info(f"Loading dataset for stage: {stage} ({config['path']})...")

        try:
            # Prepare arguments for load_dataset
            load_args = {
                "path": config["path"],
                "name": config["name"],
                "split": config["split"],
                "cache_dir": self.cache_dir,
            }
            
            # Add specific arguments from config (e.g., trust_remote_code)
            if "extra_args" in config:
                load_args.update(config["extra_args"])

            # Add/override arguments passed to the function
            load_args.update(kwargs)
            
            # Remove 'name' if None (some datasets don't require configuration)
            if load_args["name"] is None:
                del load_args["name"]

            dataset = load_dataset(**load_args)
            
            logger.info(f"Successfully loaded dataset: {stage}")
            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset {stage}: {e}")
            raise e
    
    def _load_local(self, dataset_name: str) -> LocalJSONLDataset:
        """
        Load a local JSONL dataset from the dataset/ folder.
        
        Args:
            dataset_name: Name of the dataset file (without .jsonl extension)
            
        Returns:
            LocalJSONLDataset iterator
        """
        file_path = LOCAL_DATASET_DIR / f"{dataset_name}.jsonl"
        
        if not file_path.exists():
            available = [f.stem for f in LOCAL_DATASET_DIR.glob("*.jsonl")] if LOCAL_DATASET_DIR.exists() else []
            raise FileNotFoundError(
                f"Local dataset '{dataset_name}' not found at {file_path}. "
                f"Available local datasets: {', '.join(available) if available else 'none'}"
            )
        
        logger.info(f"Loading local dataset: {dataset_name} from {file_path}")
        return LocalJSONLDataset(file_path)

    def get_available_stages(self):
        """Return list of available training stages (including local datasets)."""
        stages = list(self.datasets_config.keys())
        
        # Add local datasets
        if LOCAL_DATASET_DIR.exists():
            local_datasets = [f"local:{f.stem}" for f in LOCAL_DATASET_DIR.glob("*.jsonl")]
            stages.extend(local_datasets)
        
        return stages
    
    def get_available_local_datasets(self):
        """Return list of available local datasets."""
        if not LOCAL_DATASET_DIR.exists():
            return []
        return [f.stem for f in LOCAL_DATASET_DIR.glob("*.jsonl")]


# Example usage (only when running file directly)
if __name__ == "__main__":
    loader = DatasetLoader()
    try:
        # Example loading (commented out to avoid downloading huge data on import)
        # print("Available stages:", loader.get_available_stages())
        # ds_base = loader.load("base", streaming=True)  # Streaming allows quick preview without full download
        # print("Base item:", next(iter(ds_base)))
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
