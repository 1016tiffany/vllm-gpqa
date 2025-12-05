# vlmeval/vlm/gpt_oss_20b.py
import warnings
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

# Optional Harmony encoding for structured prompts with vLLM
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )
except Exception:
    HarmonyEncodingName = None
    load_harmony_encoding = None
    Conversation = None
    Message = None
    Role = None
    SystemContent = None
    DeveloperContent = None

class GPTOSS20B(BaseModel):
    """
    Text-only wrapper for GPT-OSS-20B variants (BF16 or HQQ) inside VLMEvalKit.

    - Default backend: HuggingFace Transformers
    - Optional backend: vLLM (set use_vllm=True when registering in config.py)

    NOTE on vLLM + HQQ:
      vLLM's HQQ path currently supports 4-bit with group_size=64 only (HQQMarlin).
      Mixed 2/4-bit HQQ won't load via vLLM. Use the Transformers backend for those.
    """

    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = False

    def __init__(self, model_path='unsloth/gpt-oss-20b-BF16', use_vllm=False,
                 vllm_quantization=None,  # e.g. 'hqq', 'awq', 'gptq', or None for BF16
                 use_harmony=False,
                 harmony_encoding_name: str = 'HARMONY_GPT_OSS',
                 **kwargs):
        # DO NOT: super().__init__(model_path, **kwargs)  # BaseModel.__init__ takes no args
        # This pattern matches other VLMEvalKit wrappers (e.g., QwenVL, molmo)

        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.use_harmony = bool(use_harmony)
        self.kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True
        )
        self.kwargs.update(kwargs)

        # Initialize Harmony encoding if requested and available
        self._harmony_encoding = None
        if self.use_vllm and self.use_harmony:
            if load_harmony_encoding is None:
                raise RuntimeError(
                    "use_harmony=True but openai_harmony is not installed. pip install openai-harmony"
                )
            # Default to HARMONY_GPT_OSS unless overridden
            enc_name = harmony_encoding_name if harmony_encoding_name else 'HARMONY_GPT_OSS'
            # Support both enum name and explicit HarmonyEncodingName
            try:
                enc_enum = getattr(HarmonyEncodingName, enc_name) if isinstance(enc_name, str) else enc_name
            except Exception:
                enc_enum = HarmonyEncodingName.HARMONY_GPT_OSS
            self._harmony_encoding = load_harmony_encoding(enc_enum)

        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
            except Exception as err:
                raise RuntimeError(
                    "vLLM backend requested but vllm is not installed. "
                    "pip install vllm"
                ) from err

            # For BF16 full-precision, leave quantization=None.
            # For HQQ via vLLM, quantization='hqq' and checkpoint must include
            # a vLLM-compatible quantize_config.json (4-bit, group_size=64).
            self._vllm_sampling = SamplingParams(
                temperature=float(self.kwargs.get("temperature", 0.0)),
                top_p=float(self.kwargs.get("top_p", 1.0)),
                max_tokens=int(self.kwargs.get("max_new_tokens", 512)),
            )
            self._vllm = LLM(
                model=self.model_path,
                dtype="bfloat16",
                quantization=vllm_quantization,   # e.g. 'hqq' for HQQMarlin path
                trust_remote_code=True,
                # Add other engine args here if needed, e.g.,
                # tensor_parallel_size=int(self.kwargs.get("tensor_parallel_size", 1)),
            )
            # tokenizer comes from vLLM internally; still keep HF tokenizer for decoding edge cases
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            # HuggingFace Transformers path (works for BF16 and HQQ mixed 2/4‑bit)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True
            )
            # Let Transformers read quantization config from the checkpoint if present
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()

        warnings.warn(f'Generation kwargs: {self.kwargs}')

    def _build_harmony_prefill(self, prompt: str):
        """
        Build Harmony prefill token IDs and stop tokens for a single user prompt.
        We use a minimal conversation: [SYSTEM empty, USER prompt].
        VLMEvalKit provides text-only messages; for advanced roles include DeveloperContent via kwargs.
        """
        assert self._harmony_encoding is not None
        # Optional developer instructions
        dev_instructions = self.kwargs.get("developer_instructions")
        messages = [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        ]
        if dev_instructions:
            messages.append(
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(str(dev_instructions)),
                )
            )
        messages.append(Message.from_role_and_content(Role.USER, str(prompt)))

        convo = Conversation.from_messages(messages)
        prefill_ids = self._harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        stop_token_ids = self._harmony_encoding.stop_tokens_for_assistant_actions()
        return prefill_ids, stop_token_ids

    def generate_inner(self, message, dataset=None):
        prompt, _ = self.message_to_promptimg(message, dataset=dataset)

        if not self.use_vllm and self.use_harmony:
        # Build chat-style messages with reasoning level high, tools disabled
            messages = [
            {"role": "system", "content": "You are a helpful assistant. Reasoning: high. Tools: disabled."},
            {"role": "user", "content": prompt},
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            )
            prompt_text = chat_prompt
        else:
            prompt_text = prompt

        encoded = self.tokenizer([prompt_text], return_tensors="pt", padding=False)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        gen = self.model.generate(
        **encoded,
        do_sample=self.kwargs.get("do_sample", False),
        num_beams=self.kwargs.get("num_beams", 1),
        max_new_tokens=self.kwargs.get("max_new_tokens", 512),
        use_cache=self.kwargs.get("use_cache", True),
        )
        out = self.tokenizer.decode(gen[0][encoded["input_ids"].size(1):], skip_special_tokens=True)
        return out.strip()

