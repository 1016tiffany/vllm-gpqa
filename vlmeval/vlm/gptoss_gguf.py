# model_gptoss.py – GPT-OSS-20B (transformers OR llama.cpp-gguf backend)
from __future__ import annotations

import os, re, math, logging
import torch

from vlmeval.vlm.base import BaseModel
from vlmeval.smp.misc import get_rank_and_world_size, get_gpu_memory, auto_split_flag

# Optional llama.cpp
try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False

FAIL_MSG = "Failed to obtain answer via API."


def _join_chat_as_text(messages: list[dict], tokenizer=None) -> str:
    """
    Portable chat-to-prompt renderer:
    1) If tokenizer has apply_chat_template → use it.
    2) Else fallback to a simple ChatML-ish format that works broadly.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback template (ChatML-like)
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # in our pipeline we only keep text; join any leftover pieces
            content = " ".join(
                x.get("text", x.get("value", "")) if isinstance(x, dict) else str(x)
                for x in content
            )
        lines.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    lines.append("<|im_start|>assistant\n")  # generation begins
    return "\n".join(lines)


class GPTOSSChat:
    """
    Minimal local text LM wrapper for VLMEvalKit (llama.cpp GGUF).
    Designed for Custom Text MCQ datasets (like your gpqa.tsv).
    """

    # VLMEvalKit checks this to decide API vs local path
    is_api = False

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        max_tokens: int = 8,
        **_ignored,  # swallow any extra kwargs VLMEvalKit might pass
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF not found: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or max(4, (os.cpu_count() or 8) // 2),
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            embedding=False,
            verbose=False,
        )
        self.decode_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
        )
        self._dump_image = False  # no-op; kept for interface parity

    # VLMEvalKit may call this on local models; provide a no-op.
    def set_dump_image(self, flag: bool):
        self._dump_image = flag

    # If you want VLMEvalKit to let the dataset build the prompt, just return False.
    # (inference.py checks hasattr(model, 'use_custom_prompt') and calls it)
    def use_custom_prompt(self, dataset_name: str) -> bool:
        return False

    def _extract_letter(self, text: str) -> str:
        m = re.search(r"\b([A-E])\b", (text or "").strip().upper())
        return m.group(1) if m else ""

    def _build_prompt_from_struct(self, s: Dict[str, Any]) -> str:
        q = s.get("question") or s.get("text") or s.get("prompt") or ""
        opts = []
        for k in ["A", "B", "C", "D", "E"]:
            if k in s and str(s[k]).strip():
                opts.append(f"{k}) {s[k]}")
        sys_msg = (
            "You are a helpful assistant for multiple-choice questions. "
            "Respond with ONE capital letter A, B, C, D, or E only.\n"
        )
        body = "\n".join(opts)
        return f"{sys_msg}\nQuestion:\n{q}\n\nOptions:\n{body}\n\nAnswer (A-E) only:"

    def generate(self, message, dataset: str | None = None) -> str:
        """
        VLMEvalKit calls this with `message` from dataset.build_prompt(item).
        Return exactly one letter A-E; otherwise the evaluator may misparse.
        """
        if isinstance(message, dict):
            prompt = self._build_prompt_from_struct(message)
        elif isinstance(message, str):
            prompt = f"Answer with one capital letter A-E only.\n{message}\nAnswer (A-E): "
        else:
            prompt = "Answer with one capital letter A-E only."

        out = self.llm.create_completion(prompt=prompt, **self.decode_kwargs)
        txt = out["choices"][0]["text"]
        letter = self._extract_letter(txt)
        return letter if letter else FAIL_MSG
