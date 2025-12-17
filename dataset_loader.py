import logging
from typing import Optional, Dict, Any, Union
from datasets import load_dataset, DatasetDict, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Klasa do zarządzania pobieraniem datasetów z HuggingFace dla różnych etapów treningu.
    Obsługuje:
    - base: nauka języka (pre-training)
    - mid: fakty/wiedza (knowledge injection)
    - sft: chatbot (instruction tuning)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Inicjalizacja loadera.
        
        Args:
            cache_dir: Opcjonalna ścieżka do katalogu cache dla datasetów.
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
        Pobiera i zwraca dataset dla zadanego etapu.

        Args:
            stage: 'base', 'mid', lub 'sft'.
            **kwargs: Dodatkowe argumenty przekazywane do load_dataset.

        Returns:
            Dataset lub DatasetDict.
        """
        if stage not in self.datasets_config:
            valid_keys = ", ".join(self.datasets_config.keys())
            raise ValueError(f"Nieznany etap: '{stage}'. Dostępne etapy: {valid_keys}")

        config = self.datasets_config[stage]
        logger.info(f"Ładowanie datasetu dla etapu: {stage} ({config['path']})...")

        try:
            # Przygotowanie argumentów dla load_dataset
            load_args = {
                "path": config["path"],
                "name": config["name"],
                "split": config["split"],
                "cache_dir": self.cache_dir,
            }
            
            # Dodaj specyficzne argumenty z konfiguracji (np. trust_remote_code)
            if "extra_args" in config:
                load_args.update(config["extra_args"])

            # Dodaj/nadpisz argumenty przekazane do funkcji
            load_args.update(kwargs)
            
            # Usuwamy 'name' jeśli jest None (niektóre datasety nie wymagają konfiguracji)
            if load_args["name"] is None:
                del load_args["name"]

            dataset = load_dataset(**load_args)
            
            logger.info(f"Pomyślnie załadowano dataset: {stage}")
            return dataset

        except Exception as e:
            logger.error(f"Błąd podczas ładowania datasetu {stage}: {e}")
            raise e

    def get_available_stages(self):
        return list(self.datasets_config.keys())

# Przykład użycia (tylko jeśli uruchamiamy plik bezpośrednio)
if __name__ == "__main__":
    loader = DatasetLoader()
    try:
        # Przykładowe ładowanie (zakomentowane, aby nie pobierać ogromnych danych przy imporcie)
        # print("Dostępne etapy:", loader.get_available_stages())
        # ds_base = loader.load("base", streaming=True) # Streaming pozwala na szybki podgląd bez pobierania całości
        # print("Base item:", next(iter(ds_base)))
        pass
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
