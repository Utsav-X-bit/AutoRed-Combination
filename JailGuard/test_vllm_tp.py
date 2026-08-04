import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"

from vllm import LLM
import time

print("Starting vLLM init...")
t0 = time.time()
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
    trust_remote_code=True,
    download_dir=None
)
print(f"Init complete in {time.time() - t0:.2f}s")
