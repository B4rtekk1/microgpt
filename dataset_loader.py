"""
Dataset loader for downloading and managing datasets from HuggingFace.
Supports different training stages: base (pre-training), mid (knowledge), sft (instruction tuning).
"""

import logging
from typing import Optional, Dict, Any, Union
from datasets import load_dataset, DatasetDict, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                "path": "wikitext",
                "name": "wikitext-103-raw-v1",
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
            }
        }

    def load(self, stage: str, **kwargs) -> Union[Dataset, DatasetDict]:
        """
        Download and return dataset for the given stage.

        Args:
            stage: 'base', 'mid', or 'sft'.
            **kwargs: Additional arguments passed to load_dataset.

        Returns:
            Dataset or DatasetDict.
        """
        if stage not in self.datasets_config:
            valid_keys = ", ".join(self.datasets_config.keys())
            raise ValueError(f"Unknown stage: '{stage}'. Available stages: {valid_keys}")

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

    def get_available_stages(self):
        """Return list of available training stages."""
        return list(self.datasets_config.keys())


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
