"""
JailGuard Reimplementation — LLM Interface
==========================================
Unified interface for querying different LLM backends:
  • Ollama            (local Ollama server)
  • HuggingFace       (local GPU via transformers, good for debugging)
  • vLLM              (local GPU via vLLM — fast, recommended for benchmarking)
  • OpenAI API        (cloud)

Offline / compute-node usage:
    Set OFFLINE=1 in config.py (or call setup_offline_mode() before build_llm())
    to prevent ANY network calls. All models must already be in the HF cache.

Usage:
    from llm_interface import build_llm, query_llm, setup_offline_mode
    setup_offline_mode()          # call before build_llm() on air-gapped nodes
    llm = build_llm()
    response = query_llm(llm, "What is gravity?")
"""

import os
import time
import sys
import json
from typing import Union, List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Offline mode helper  (call BEFORE importing transformers / vllm)
# ─────────────────────────────────────────────────────────────────────────────
# Set up offline mode and fix for multi-GPU Ray hang immediately on import.
# This must happen before vllm, torch, or ray are imported anywhere else.
import config as cfg
if getattr(cfg, "OFFLINE", False):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"

    # Fix for offline Ray multi-GPU initialization hangs on HPC/cluster nodes
    os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

    # NCCL: force loopback / shared-memory transport
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_SHM_DISABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "WARN"

    # Force vLLM to use 'spawn' and use standard V0 engine (prevents flashinfer JIT issues in vLLM 0.8+)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V1"] = "0"
    print("[offline] Network access disabled — using local HF cache only.")

# ─────────────────────────────────────────────────────────────────────────────
#  Ollama Backend
# ─────────────────────────────────────────────────────────────────────────────

class OllamaLLM:
    """Query a locally running Ollama server."""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: int = 120):
        self.model   = model
        self.host    = host.rstrip('/')
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            # Normalize: 'llama3.2' may appear as 'llama3.2:latest'
            model_base = self.model.split(':')[0]
            available  = [m.split(':')[0] for m in models]
            if model_base not in available:
                print(f"\n⚠️  Model '{self.model}' not found in Ollama.")
                print(f"   Available: {models}")
                print(f"   Pull it with: ollama pull {self.model}\n")
        except Exception as e:
            print(f"\n⚠️  Cannot connect to Ollama at {self.host}: {e}")
            print("   Make sure Ollama is running: ollama serve\n")

    def query(self, prompt: Union[str, List[Dict]], system: str = "You are a helpful assistant.") -> str:
        import requests

        if isinstance(prompt, list):
            # Multi-turn conversation (injection attacks)
            messages = prompt
        else:
            messages = [
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ]

        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": 0.7, "num_predict": 512},
        }

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            return "I cannot assist with that! [TIMEOUT]"
        except Exception as e:
            print(f"  [Ollama] Error: {e}")
            return "No response!"

    def __repr__(self):
        return f"OllamaLLM(model={self.model}, host={self.host})"


# ─────────────────────────────────────────────────────────────────────────────
#  HuggingFace Backend
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceLLM:
    """Load and query a HuggingFace model locally on GPU."""

    def __init__(self, model_id: str, device: str = "cuda",
                 max_new_tokens: int = 256, temperature: float = 0.7,
                 load_in_8bit: bool = False):
        print(f"Loading HuggingFace model '{model_id}'...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # bfloat16 is native to A100 and gives full accuracy at half memory
            kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            }
            if load_in_8bit:
                # 8-bit overrides bfloat16; only use if VRAM is constrained
                kwargs.pop("torch_dtype", None)
                kwargs["load_in_8bit"] = True

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, **kwargs
            )
            self.model.eval()
            self.max_new_tokens = max_new_tokens
            self.temperature    = temperature
            self.model_id       = model_id
            print(f"✓ Model loaded: {model_id} "
                  f"(dtype={self.model.dtype}, "
                  f"device={next(self.model.parameters()).device})")
        except ImportError:
            raise ImportError(
                "transformers not installed. Run:\n"
                "  pip install transformers accelerate sentencepiece protobuf"
            )

    def query(self, prompt: Union[str, List[Dict]],
              system: str = "You are a helpful assistant.") -> str:
        from transformers import pipeline
        import torch

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ]

        try:
            # Use chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                ) + "\nASSISTANT:"

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return decoded.strip()
        except Exception as e:
            print(f"  [HuggingFace] Error: {e}")
            return "No response!"

    def __repr__(self):
        return f"HuggingFaceLLM(model={self.model_id})"


# ─────────────────────────────────────────────────────────────────────────────
#  vLLM Backend  (recommended for benchmarking — much faster than transformers)
# ─────────────────────────────────────────────────────────────────────────────

class VllmLLM:
    """
    Query a model via vLLM — an optimised inference engine that uses
    PagedAttention and continuous batching for much higher throughput than
    plain HuggingFace Transformers.

    Key advantages for JailGuard benchmarking:
      • Generates all N variant responses in a single batched call
      • Automatically uses tensor parallelism across all available GPUs
      • 3-10× faster throughput than the HuggingFace backend

    Requires:  pip install vllm
    """

    def __init__(
        self,
        model_id:         str,
        tensor_parallel:  int   = -1,     # -1 → use all available GPUs
        max_new_tokens:   int   = 256,
        temperature:      float = 0.7,
        gpu_memory_util:  float = 0.90,   # fraction of GPU VRAM to use
        dtype:            str   = "bfloat16",  # bfloat16 is optimal on A100
    ):
        try:
            from vllm import LLM, SamplingParams
            import torch
        except ImportError:
            raise ImportError(
                "vLLM not installed. Run:\n"
                "  pip install vllm"
            )

        # ── Pre-flight CUDA driver compatibility check ───────────────────────
        # vLLM ships with a specific CUDA toolkit version compiled in.
        # If the system driver is older than that toolkit requires, vLLM will
        # silently report CUDA as unavailable and then crash at model load.
        # We detect this early and give a clear, actionable error.
        if not torch.cuda.is_available():
            torch_cuda = torch.version.cuda or "unknown"
            raise RuntimeError(
                f"CUDA is not available with this torch build (torch {torch.__version__}, "
                f"compiled for CUDA {torch_cuda}).\n"
                f"This usually means vLLM upgraded torch to a version that requires a newer "
                f"NVIDIA driver than the one on this system.\n\n"
                f"Fix — reinstall torch + vLLM pinned to CUDA 12.1:\n"
                f"  pip install torch==2.1.1+cu121 "
                f"--index-url https://download.pytorch.org/whl/cu121\n"
                f"  pip install vllm==0.4.3 "
                f"--extra-index-url https://download.pytorch.org/whl/cu121\n\n"
                f"Or use --backend huggingface to skip vLLM entirely."
            )

        n_gpus = tensor_parallel if tensor_parallel > 0 else torch.cuda.device_count()
        if n_gpus < 1:
            n_gpus = 1

        print(f"Loading vLLM model '{model_id}' "
              f"(tensor_parallel={n_gpus}, dtype={dtype}, "
              f"gpu_memory_util={gpu_memory_util})...")

        # vLLM 0.2.7: Ray is mandatory for tensor_parallel_size > 1.
        # ParallelConfig.__init__ forces worker_use_ray=True for multi-GPU.
        # The NCCL env vars set in setup_offline_mode() prevent the NCCL
        # init from hanging on HPC nodes (loopback, disable IB/P2P).
        self.llm = LLM(
            model                    = model_id,
            tensor_parallel_size     = n_gpus,
            dtype                    = dtype,
            gpu_memory_utilization   = gpu_memory_util,
            trust_remote_code        = True,
            download_dir             = None,    # uses HF_HOME / HF cache
        )
        self.sampling_params = SamplingParams(
            max_tokens  = max_new_tokens,
            temperature = temperature,
        )
        self.model_id  = model_id
        self.n_gpus    = n_gpus
        print(f"✓ vLLM model loaded: {model_id} on {n_gpus} GPU(s)")


    def _build_prompt(self, message: Union[str, List[Dict]],
                      system: str) -> str:
        """Convert messages to a plain text prompt using the chat template."""
        try:
            # Try to use the tokenizer's chat template for proper formatting
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            if isinstance(message, str):
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": message},
                ]
            else:
                msgs = message
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: plain concatenation
            if isinstance(message, str):
                return f"SYSTEM: {system}\nUSER: {message}\nASSISTANT:"
            return "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in message
            ) + "\nASSISTANT:"

    def query(self, prompt: Union[str, List[Dict]],
              system: str = "You are a helpful assistant.") -> str:
        """Query a single prompt. Calls generate() internally."""
        results = self.generate([prompt], system=system)
        return results[0]

    def generate(self, prompts: List[Union[str, List[Dict]]],
                 system: str = "You are a helpful assistant.") -> List[str]:
        """
        Batch-generate responses for a list of prompts in a single vLLM call.
        This is much more efficient than querying one at a time.

        Args:
            prompts: list of str (jailbreak) or list-of-dicts (injection)
            system:  system prompt to prepend

        Returns:
            list of response strings (same length as prompts)
        """
        formatted = [self._build_prompt(p, system) for p in prompts]
        outputs   = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def close(self):
        """Gracefully shut down vLLM workers on exit."""
        try:
            if hasattr(self, 'llm'):
                import torch
                if hasattr(self.llm, 'llm_engine'):
                    engine = self.llm.llm_engine
                    if hasattr(engine, 'model_executor') and hasattr(engine.model_executor, 'shutdown'):
                        engine.model_executor.shutdown()
                del self.llm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"VllmLLM(model={self.model_id}, gpus={self.n_gpus})"


# ─────────────────────────────────────────────────────────────────────────────
#  OpenAI Backend
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLM:
    """Query OpenAI's chat completion API."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", sleep: int = 3):
        self.model = model
        self.sleep = sleep
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def query(self, prompt: Union[str, List[Dict]],
              system: str = "You are a helpful assistant.") -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            time.sleep(self.sleep)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "BadRequest" in err or "content_filter" in err:
                return "I cannot assist with that!"
            print(f"  [OpenAI] Error: {e}")
            return "No response!"

    def __repr__(self):
        return f"OpenAILLM(model={self.model})"


# ─────────────────────────────────────────────────────────────────────────────
#  Factory — build_llm()
# ─────────────────────────────────────────────────────────────────────────────

def build_llm(
    backend:         Optional[str] = None,
    model_id:        Optional[str] = None,
    tensor_parallel: Optional[int] = None,
):
    """
    Build an LLM object based on config.py settings (or CLI overrides).

    Args:
        backend:         Override cfg.LLM_BACKEND (e.g. "vllm").
        model_id:        Override the victim model for vllm/huggingface.
                         Must be a HuggingFace model ID already cached locally
                         when OFFLINE=True.
                         Examples:
                           "mistralai/Mistral-7B-Instruct-v0.3"
                           "meta-llama/Llama-2-7b-chat-hf"
                           "internlm/internlm2-chat-7b"
        tensor_parallel: Override the number of GPUs to use for vLLM tensor
                         parallelism. e.g. 4 to use 4 A100s, 1 for single-GPU.
                         Defaults to cfg.VLLM_TENSOR_PARALLEL (currently 4).

    If OFFLINE=True in config.py, offline mode is enforced before any model
    is loaded (prevents network calls on air-gapped compute nodes).

    Returns an object with a `.query(prompt)` method.
    For the vllm backend, the object also exposes a `.generate(list)` method
    that processes a full batch of prompts in a single GPU forward pass.
    """
    import config as cfg

    # (Offline mode is now set at module load time)
    b = (backend or cfg.LLM_BACKEND).lower()

    if b == "ollama":
        return OllamaLLM(
            model   = cfg.OLLAMA_MODEL,
            host    = cfg.OLLAMA_HOST,
            timeout = cfg.OLLAMA_TIMEOUT,
        )
    elif b == "huggingface":
        return HuggingFaceLLM(
            model_id      = model_id or cfg.HF_MODEL_ID,
            device        = cfg.HF_DEVICE,
            max_new_tokens= cfg.HF_MAX_TOKENS,
            temperature   = cfg.HF_TEMPERATURE,
            load_in_8bit  = cfg.HF_LOAD_IN_8BIT,
        )
    elif b == "vllm":
        return VllmLLM(
            model_id        = model_id or cfg.VLLM_MODEL_ID,
            tensor_parallel = tensor_parallel if tensor_parallel is not None
                              else cfg.VLLM_TENSOR_PARALLEL,
            max_new_tokens  = cfg.VLLM_MAX_TOKENS,
            temperature     = cfg.VLLM_TEMPERATURE,
            gpu_memory_util = cfg.VLLM_GPU_MEMORY_UTIL,
            dtype           = cfg.VLLM_DTYPE,
        )
    elif b == "openai":
        return OpenAILLM(
            api_key = cfg.OPENAI_API_KEY,
            model   = cfg.OPENAI_MODEL,
            sleep   = cfg.OPENAI_SLEEP,
        )
    else:
        raise ValueError(
            f"Unknown backend '{b}'. Choose: ollama / huggingface / vllm / openai"
        )



def query_llm(llm, prompt: Union[str, List[Dict]],
              system: str = "You are a helpful assistant.") -> str:
    """
    Unified query function.

    Args:
        llm:    LLM object built by build_llm()
        prompt: str (jailbreak) or list-of-dicts (injection conversation)
        system: system prompt string (ignored for multi-turn inputs)

    Returns:
        str: LLM response
    """
    return llm.query(prompt, system=system)
