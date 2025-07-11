# model.py  – Qwen-VL (transformers *or* llama.cpp-gguf backend)
from __future__ import annotations

import os, sys, warnings, math, logging, datetime
import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr
import re

# ---------- NEW: llama.cpp multimodal ----------
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False
# ------------------------------------------------


# --------------------- helpers ------------------
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')
# ------------------------------------------------


def split_model():
    """ Split the 72 B model across all visible GPUs (unchanged) """
    device_map = {}
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = max(1, total_gpus // world_size)
    num_layers = 80 + 8  # visual “virtual” layers
    num_layers_per_gpu = [math.ceil(num_layers / num_gpus)] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map.update(
        visual=rank,
        **{
            'model.embed_tokens': rank,
            'model.norm':        last_gpu,
            'model.rotary_emb':  last_gpu,
            'lm_head':           last_gpu,
        },
    )
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    """
    A single class that now supports **either**:
      • HuggingFace transformers checkpoints (.bin / .safetensors)
      • llama.cpp GGUF checkpoints (.gguf)

    Selection is automatic from `model_path`.
    """
    INSTALL_REQ = False
    INTERLEAVE  = True
    VIDEO_LLM   = True

    # ---------- NEW: allow overriding the mmproj-CLIP gguf -----------
    def __init__(
        self,
        model_path: str,
        *,
        clip_model_path: str | None = None,  # only used for .gguf
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 64,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        n_ctx: int = 8192,                   # llama.cpp context window
        n_gpu_layers: int = -1,              # -1 == all GPU if memory permits
        n_threads: int | None = None,
    ):
    # ---------------------------------------------------------------
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps, self.nframe, self.FRAME_FACTOR = 2.0, 64, 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        self._is_gguf = model_path.lower().endswith(".gguf")
        self._clip_model_path = (
            clip_model_path
            or os.path.join(os.path.dirname(model_path),
                            "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")
        )

        # ------------------------------------------------------------------
        # 1) GGUF pathway  (requires llama_cpp built w/ CUDA or Metal)
        # ------------------------------------------------------------------
        if self._is_gguf:
            if not _LLAMA_CPP_AVAILABLE:
                raise RuntimeError(
                    "llama_cpp not installed – `pip install llama-cpp-python[server]` "
                    "with CUDA/Metal support, then rebuild."
                )

            if not os.path.isfile(self._clip_model_path):
                raise FileNotFoundError(
                    f"Cannot find CLIP/mmproj gguf at {self._clip_model_path}. "
                    "Pass `clip_model_path=...` when constructing the model."
                )

            chat_handler = Qwen25VLChatHandler(
                clip_model_path=self._clip_model_path
            )

            # NOTE 1: n_gpu_layers = -1 → load everything on GPU
            # NOTE 2: n_threads defaults to #logical cores; override for busy boxes
            self.model = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_parts=1,
                seed=0,
                logits_all=False,
                embedding=False,
                n_threads=n_threads or max(4, os.cpu_count() // 2),
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            # self.processor = None  # not used in GGUF mode
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            torch.cuda.empty_cache()
            return  # GGUF initialisation done

        # ------------------------------------------------------------------
        # 2) Transformers pathway (unchanged from upstream)
        # ------------------------------------------------------------------
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            Qwen2VLProcessor,
        )

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems   = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0, "No visible GPU memory"

        if '72b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto',
                device_map=split_model(),
                attn_implementation='flash_attention_2')
            self.model.eval()

        elif auto_split_flag():
            assert world_size == 1, \
                'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto',
                device_map='auto',
                attn_implementation='flash_attention_2')

        else:  # default single-GPU path
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu')
            self.model.cuda().eval()

        torch.cuda.empty_cache()
    # --------------------- end __init__ -------------------------------

    # ------------------------ helpers ------------------------------
    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict]:
        """
        Converts the caller-supplied list of {'type','value'} dicts into the
        correct multimodal chat-content format **for whichever backend is active**.
        """
        content: list[dict] = []
        for s in inputs:
            if s['type'] == 'image':
                url = ensure_image_url(s['value'])
                if self._is_gguf:
                    # llama.cpp multimodal chat-format: {"type":"image_url", "image_url": "..."}
                    item = {"type": "image_url", "image_url": url}
                else:
                    item = {'type': 'image', 'image': url}
                    if dataset == 'OCRBench':
                        item['min_pixels'] = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels

            elif s['type'] == 'video':
                url = ensure_video_url(s['value'])
                if self._is_gguf:
                    item = {"type": "video_url", "video_url": url}
                else:
                    item = {'type': 'video', 'video': url,
                            'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        cap = cv2.VideoCapture(s['value'])
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if total < self.nframe:
                            total = total // self.FRAME_FACTOR * self.FRAME_FACTOR
                        item['nframes'] = total if total else self.nframe

            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}")
            content.append(item)
        return content
    # ---------------------------------------------------------------

    # ----------------------- generation ----------------------------
    def generate_inner(self, message, dataset: str | None = None):
        """
        1. Build the chat messages
        2. Call either llama_cpp.create_chat_completion or transformers.generate
        3. Post-process if requested
        """
        messages: list[dict] = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})

        user_content = self._prepare_content(message, dataset)
        messages.append({'role': 'user', 'content': user_content})

        if self.verbose:
            logging.info(f"[PROMPT] {messages}")

        # ---------- GGUF branch ----------
        if self._is_gguf:
            kwargs = dict(
                max_tokens=self.generate_kwargs['max_new_tokens'],
                temperature=self.generate_kwargs['temperature'],
                top_p=self.generate_kwargs['top_p'],
                top_k=self.generate_kwargs['top_k'],
                repeat_penalty=self.generate_kwargs['repetition_penalty'],
            )
            resp = self.model.create_chat_completion(messages=messages, **kwargs)
            response = resp["choices"][0]["message"]["content"]

        # ---------- Transformers branch ----------
        else:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            inputs = self.processor(
                text=text, images=images, videos=videos,
                padding=True, return_tensors='pt').to('cuda')

            generated = self.model.generate(**inputs, **self.generate_kwargs)
            generated = [o[len(i):] for i, o in zip(inputs.input_ids, generated)]
            response = self.processor.tokenizer.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

        # optional boxed{} extraction for math tasks
        if self.post_process and '\\boxed{' in response:
            resp = response.split('\\boxed{')[-1]
            depth = 1
            for i, ch in enumerate(resp):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        response = resp[:i]
                        break
        
        # ───────────────────────────────────────────────────────────────
        # Post-process so evaluator doesn’t get answers like "{Bowmore}"
        # or ": 10 years".
        # ----------------------------------------------------------------
        def _normalize_answer(text: str) -> str:
            # 1) Trim
            text = text.strip()
            # 2) Strip any leading “{[(” and trailing “}])”
            text = re.sub(r'^[\{\[\("]+',    '', text)
            text = re.sub(r'[\}\]\)"]+$',    '', text)
            # 3) Drop leading colons/dashes/bullets
            text = re.sub(r'^[:\-–•]+\s*',   '', text)
            # 4) Collapse internal whitespace
            text = re.sub(r'\s+', ' ', text)
            # 5) Strip trailing .,;:
            text = text.rstrip('.,;:')
            # 6) Lowercase (so “AP” → “ap”; references must be lowercased too)
            text = text.lower()
            return text

        response = _normalize_answer(response)

        if self.verbose:
            logging.info(f"[RESPONSE] {response}")
        
        # ─── Debug: dump every Q-A pair to a JSONL file ──────────────────────────
        # ─── Debug dump: only rank 0 writes out ─────────
        from ...smp import get_rank_and_world_size
        rank, _world_size = get_rank_and_world_size()
        debug_path = os.getenv("DEBUG_OUTPUT_FILE")
        if debug_path and rank == 0:                     # avoid multi-GPU duplication
            try:
                import json, pathlib, io
                pathlib.Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
                with io.open(debug_path, "a", encoding="utf-8") as f:
                    record = {
                        "dataset": dataset,
                        "prompt":  messages,             # full, tokenised prompt
                        "answer":  response.strip()
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.warning(f"[DEBUG_DUMP] failed: {e}")
        # ──────────────────────────────────────────────────────────────────────────
        return response
    # ---------------------------------------------------------------
