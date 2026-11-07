"""OpenAI-compatible Qwen server for Colab with deterministic micro-batching.

Run this file in a GPU Colab runtime, then expose port 8000 with the tunnel of
your choice. It batches independent chat-completion requests, not ReAct steps
within the same sample.
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "15"))
# Keep this aligned with the default used by the evaluation client. A request
# above this value is rejected by Pydantic before it reaches the batch worker.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))


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
request_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
worker_task: Optional[asyncio.Task] = None


def generate_batch(pending: list[PendingRequest]) -> list[str]:
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
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1]
    max_tokens = max(item.request.max_tokens for item in pending)

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


async def batch_worker() -> None:
    """Collect requests for a tiny window, then execute one GPU batch."""
    while True:
        first = await request_queue.get()
        batch = [first]
        deadline = time.monotonic() + BATCH_WAIT_MS / 1000

        while len(batch) < BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(request_queue.get(), remaining))
            except asyncio.TimeoutError:
                break

        active = [item for item in batch if not item.future.cancelled()]
        if not active:
            continue
        try:
            texts = await asyncio.to_thread(generate_batch, active)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Refusing CPU inference for this server.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        # Do not silently offload layers to CPU/disk: that makes latency
        # unsuitable for the competition's 30-second per-sample limit.
        device_map={"": 0},
    )
    model.eval()
    print(f"Model loaded. Micro-batch: size={BATCH_SIZE}, wait={BATCH_WAIT_MS}ms", flush=True)
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

    future = asyncio.get_running_loop().create_future()
    await request_queue.put(PendingRequest(request=req, future=future))
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
