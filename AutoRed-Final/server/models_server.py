"""
Models manager for FastAPI server — loads models on startup, keeps in memory.
"""
import torch
import time
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DistilBertForSequenceClassification,
)

from experiment.llama_3_8b_verbose import (
    LLAMA_PATH, GENERATOR_PATH, DISTILBERT_CKPT, device
)


class ServerModelsManager:
    """Load and keep models in GPU memory across runs."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.load_times: Dict[str, float] = {}
        self.load_error: Optional[str] = None
        self._loaded = False

    def load_all(self) -> Dict[str, float]:
        """Load all 4 models. Returns dict of load times."""
        if self._loaded:
            return self.load_times

        # Load victim LLM
        t0 = time.time()
        print("[SERVER] Loading victim LLM...")
        self.models["victim"] = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH, dtype=torch.float16, device_map="auto", local_files_only=True
        )
        self.tokenizers["victim"] = AutoTokenizer.from_pretrained(
            LLAMA_PATH, local_files_only=True, use_fast=False
        )
        self.load_times["victim"] = time.time() - t0
        print(f"[SERVER] ✓ Victim loaded ({self.load_times['victim']:.1f}s)")

        # Load generator
        t0 = time.time()
        print("[SERVER] Loading generator...")
        self.models["generator"] = AutoModelForCausalLM.from_pretrained(
            GENERATOR_PATH, dtype=torch.float16, device_map="auto", local_files_only=True
        )
        self.models["generator"].eval()
        self.tokenizers["generator"] = AutoTokenizer.from_pretrained(
            GENERATOR_PATH, local_files_only=True, use_fast=False
        )
        self.load_times["generator"] = time.time() - t0
        print(f"[SERVER] ✓ Generator loaded ({self.load_times['generator']:.1f}s)")

        # Load judge
        t0 = time.time()
        print("[SERVER] Loading judge...")
        self.models["judge"] = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_CKPT, local_files_only=True
        ).to(device)
        self.models["judge"].eval()
        self.tokenizers["judge"] = AutoTokenizer.from_pretrained(
            DISTILBERT_CKPT, local_files_only=True
        )
        self.load_times["judge"] = time.time() - t0
        print(f"[SERVER] ✓ Judge loaded ({self.load_times['judge']:.1f}s)")

        # Extractor uses victim model
        self.load_times["extractor"] = 0
        self._loaded = True

        return self.load_times

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_status(self) -> Dict[str, Any]:
        return {
            "ready": self._loaded,
            "error": self.load_error,
            "victim": {"loaded": self._loaded, "name": LLAMA_PATH, "load_time": self.load_times.get("victim", 0)},
            "generator": {"loaded": self._loaded, "name": GENERATOR_PATH, "load_time": self.load_times.get("generator", 0)},
            "judge": {"loaded": self._loaded, "name": DISTILBERT_CKPT, "load_time": self.load_times.get("judge", 0)},
            "extractor": {"loaded": self._loaded, "name": LLAMA_PATH, "load_time": 0},
        }

    def get_model(self, name: str):
        return self.models.get(name)

    def get_tokenizer(self, name: str):
        return self.tokenizers.get(name)


# Global singleton
server_models = ServerModelsManager()
