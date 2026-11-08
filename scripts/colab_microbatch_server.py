"""OpenAI-compatible Transformers server with deterministic micro-batching.

Run this file in a GPU Colab runtime, then expose port 8000 with the tunnel of
your choice. It batches independent chat-completion requests, not ReAct steps
within the same sample.
"""

import asyncio
import gc
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

# Reduce CUDA allocator fragmentation in long-running notebook sessions. This
# must be configured before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
# PhoGPT publishes its MPT implementation with the checkpoint. Keep remote
# model code opt-in for arbitrary models, while making the supported PhoGPT
# default work without another notebook-specific branch.
TRUST_REMOTE_CODE = os.getenv(
    "TRUST_REMOTE_CODE",
    "1" if "phogpt" in MODEL_ID.lower() else "0",
).strip().lower() in {"1", "true", "yes", "on"}
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "0"))
# Keep this aligned with the default used by the evaluation client. A request
# above this value is rejected by Pydantic before it reaches the batch worker.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
# Very long ReAct transcripts can make attention/KV-cache memory dwarf the
# model weights. Keep the newest part of the transcript when a client exceeds
# this server-side safety limit.
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_INPUT_HEAD_TOKENS = int(os.getenv("MAX_INPUT_HEAD_TOKENS", "3072"))
MIN_OOM_RETRY_TOKENS = int(os.getenv("MIN_OOM_RETRY_TOKENS", "64"))
# On two 16-GiB T4s, limiting weight placement to 7 GiB per device forces a
# real layer split and leaves room for attention activations and KV cache.
GPU_MODEL_MEMORY_GIB = float(os.getenv("GPU_MODEL_MEMORY_GIB", "7"))
# Quantizing Qwen3 keeps the whole model on one GPU. This is usually faster
# than sending every generated token across two T4s, while leaving much more
# room for the KV cache used by long ReAct transcripts.
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "0").strip().lower() in {"1", "true", "yes", "on"}
ATTN_IMPLEMENTATION = os.getenv(
    "ATTN_IMPLEMENTATION",
    "eager" if "phogpt" in MODEL_ID.lower() else "sdpa",
)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS)


@dataclass
class PendingRequest:
    request: ChatCompletionRequest
    future: asyncio.Future


tokenizer: Any = None
model: Any = None
request_queue: Optional[asyncio.Queue[PendingRequest]] = None
worker_task: Optional[asyncio.Task] = None


def get_request_queue() -> asyncio.Queue[PendingRequest]:
    global request_queue
    if request_queue is None:
        request_queue = asyncio.Queue()
    return request_queue


def _model_input_device() -> torch.device:
    """Return the device holding embeddings for single- or multi-GPU models."""
    return model.get_input_embeddings().weight.device


def _clear_cuda_cache() -> None:
    gc.collect()
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "cuda out of memory" in str(exc).lower()


def generate_batch(
    pending: list[PendingRequest],
    max_new_tokens_override: Optional[int] = None,
) -> list[str]:
    """Run exactly one deterministic `generate` for a batch of prompts."""
    prompts = [
        tokenizer.apply_chat_template(
            [
                message.model_dump() if hasattr(message, "model_dump") else message.dict()
                for message in item.request.messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in pending
    ]
    encoded_prompts = tokenizer(prompts, add_special_tokens=False).input_ids
    tail_tokens = MAX_INPUT_TOKENS - MAX_INPUT_HEAD_TOKENS
    if MAX_INPUT_HEAD_TOKENS < 1 or tail_tokens < 1:
        raise RuntimeError("MAX_INPUT_HEAD_TOKENS must be between 1 and MAX_INPUT_TOKENS - 1")

    truncated_count = 0
    for index, token_ids in enumerate(encoded_prompts):
        if len(token_ids) > MAX_INPUT_TOKENS:
            # Preserve the system prompt/tool catalog at the front and the
            # newest ReAct observations at the end.
            encoded_prompts[index] = token_ids[:MAX_INPUT_HEAD_TOKENS] + token_ids[-tail_tokens:]
            truncated_count += 1

    inputs = tokenizer.pad(
        {"input_ids": encoded_prompts},
        padding=True,
        return_tensors="pt",
    ).to(_model_input_device())
    input_length = inputs.input_ids.shape[1]
    requested_max_tokens = max(item.request.max_tokens for item in pending)
    max_tokens = max_new_tokens_override or requested_max_tokens

    if truncated_count:
        print(
            f"Truncated {truncated_count}/{len(prompts)} prompt(s) to {MAX_INPUT_TOKENS} tokens "
            f"({MAX_INPUT_HEAD_TOKENS} head + {tail_tokens} tail).",
            flush=True,
        )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[:, input_length:]
    # Requests from the agent normally all have the same max_tokens. Handle a
    # different OpenAI-compatible client safely too.
    return [
        tokenizer.decode(token_ids[:item.request.max_tokens], skip_special_tokens=True).strip()
        for item, token_ids in zip(pending, generated)
    ]


def generate_batch_resilient(pending: list[PendingRequest]) -> list[str]:
    """Retry OOMs without failing unrelated requests in the same micro-batch."""
    try:
        return generate_batch(pending)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        _clear_cuda_cache()

        # A large padded batch can fail even when every request fits alone.
        if len(pending) > 1:
            midpoint = len(pending) // 2
            print(f"CUDA OOM: splitting batch {len(pending)} -> {midpoint}+{len(pending) - midpoint}", flush=True)
            return generate_batch_resilient(pending[:midpoint]) + generate_batch_resilient(pending[midpoint:])

        # Tool calls are normally short. If a single request still fails,
        # reducing only the output budget preserves the complete input prompt.
        requested = pending[0].request.max_tokens
        if requested <= 1:
            raise exc
        retry_limits = sorted(
            {
                max(1, min(requested - 1, limit))
                for limit in (requested // 2, MIN_OOM_RETRY_TOKENS)
                if limit < requested
            },
            reverse=True,
        )

        last_exc: Exception = exc
        for limit in retry_limits:
            print(f"CUDA OOM: retrying single request with max_new_tokens={limit}", flush=True)
            try:
                return generate_batch(pending, max_new_tokens_override=limit)
            except Exception as retry_exc:
                if not _is_cuda_oom(retry_exc):
                    raise
                last_exc = retry_exc
                _clear_cuda_cache()
        raise last_exc


async def batch_worker() -> None:
    """Collect requests for a tiny window, then execute one GPU batch."""
    queue = get_request_queue()
    while True:
        first = await queue.get()
        batch = [first]
        deadline = time.monotonic() + BATCH_WAIT_MS / 1000

        while len(batch) < BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), remaining))
            except asyncio.TimeoutError:
                break

        active = [item for item in batch if not item.future.cancelled()]
        if not active:
            continue
        try:
            texts = await asyncio.to_thread(generate_batch_resilient, active)
            for item, text in zip(active, texts):
                if not item.future.cancelled():
                    item.future.set_result(text)
        except Exception as exc:  # Return the model error to every affected call.
            for item in active:
                if not item.future.cancelled():
                    item.future.set_exception(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer, model, worker_task
    print(f"Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Refusing CPU inference for this server.")
    gpu_count = torch.cuda.device_count()
    quantization_config = None
    if LOAD_IN_4BIT:
        device_map: Any = {"": 0}
        max_memory = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif gpu_count > 1:
        device_map: Any = "balanced"
        max_memory = {
            index: f"{max(1, min(GPU_MODEL_MEMORY_GIB, torch.cuda.get_device_properties(index).total_memory / 1024**3 - 1)):.1f}GiB"
            for index in range(gpu_count)
        }
    else:
        device_map = {"": 0}
        max_memory = None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
    )
    hf_device_map = getattr(model, "hf_device_map", {})
    if any(device in {"cpu", "disk"} for device in hf_device_map.values()):
        raise RuntimeError(f"Model was offloaded outside CUDA: {hf_device_map}")
    model.eval()
    print(
        f"Model loaded on {gpu_count} GPU(s). device_map={hf_device_map or device_map}; "
        f"attention={ATTN_IMPLEMENTATION}; 4bit={LOAD_IN_4BIT}; max_input={MAX_INPUT_TOKENS}; "
        f"micro-batch: size={BATCH_SIZE}, wait={BATCH_WAIT_MS}ms",
        flush=True,
    )
    worker_task = asyncio.create_task(batch_worker())
    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> dict:
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict:
    if req.temperature != 0:
        raise HTTPException(status_code=400, detail="Only temperature=0 is allowed for deterministic inference.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    queue = get_request_queue()
    future = asyncio.get_running_loop().create_future()
    await queue.put(PendingRequest(request=req, future=future))
    try:
        text = await future
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
