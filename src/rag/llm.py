"""
llm.py — Prompted LLM wrapper for grounded financial QA (Path B).

Loads Mistral 7B Instruct v0.3 in 4-bit and answers questions from
a provided context. Uses a versioned system prompt (configs/system_prompt.txt)
to steer the model without fine-tuning.

This is the counterpart to the fine-tuned adapter in Phase 5A:
same base model, no adapter, prompt engineering only.

Public API:
    llm = PromptedLLM()
    answer = llm.answer(context="...", question="...")
"""

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_SYSTEM_PROMPT_PATH = Path("configs/system_prompt.txt")


class PromptedLLM:
    """
    Grounded QA over a provided context, no fine-tuning.

    The context is expected to come from RAG retrieval — a passage or
    concatenated passages from SEC filings. The LLM is instructed to
    answer only from that context.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH,
        hf_token: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Args:
            base_model: HF Hub ID of the instruction-tuned base LLM.
            system_prompt_path: Path to plain-text system prompt.
            hf_token: HF token (falls back to HF_TOKEN env var).
            device: 'cuda', 'cpu', or 'auto'.
            load_in_4bit: True on GPU (memory savings). False on CPU (4-bit unsupported).
        """
        hf_token = hf_token or os.getenv("HF_TOKEN")

        # Load system prompt
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

        # Decide device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load tokenizer
        print(f"Loading tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model — 4-bit if GPU, fp32 if CPU
        if device == "cuda" and load_in_4bit:
            print(f"Loading {base_model} in 4-bit on GPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
            )
        else:
            print(f"Loading {base_model} on {device} (no quantization)")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float32,
                token=hf_token,
            ).to(device)
        self.model.eval()

    def _build_prompt(self, context: str, question: str) -> str:
        """
        Build Mistral's [INST]...[/INST] chat format with our system prompt.

        Mistral doesn't have a dedicated system-role slot in its base
        template — the convention is to prepend the system prompt to the
        first user message.
        """
        user_message = (
            f"{self.system_prompt}\n\n"
            f"Context: {context.strip()}\n\n"
            f"Question: {question.strip()}"
        )
        # Omit the leading <s> — the tokenizer adds BOS automatically.
        return f"[INST] {user_message} [/INST]"

    def answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate an answer for a question given a context passage.

        Args:
            context: Passage from a filing (from RAG retrieval).
            question: Natural-language question.
            max_new_tokens: Cap on generation length.
            temperature: 0.0 for deterministic (recommended for factual QA).
                        Higher for creative — not useful here.

        Returns:
            The generated answer as a string (already stripped of prompt tokens).
        """
        prompt = self._build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Strip the prompt tokens from the output — we only want the model's response
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    # Local smoke test — will run on CPU and be very slow (~1-2 min per answer)
    # if no GPU is available. Recommend running this on Kaggle for real testing.
    from dotenv import load_dotenv
    load_dotenv()

    llm = PromptedLLM()

    context = (
        "The Company designs, manufactures and markets smartphones, "
        "personal computers, tablets, wearables and accessories, and sells "
        "a variety of related services."
    )
    question = "What does the company sell?"

    print(f"\n=== Test ===")
    print(f"Context: {context}")
    print(f"Question: {question}\n")

    answer = llm.answer(context, question)
    print(f"Answer: {answer}")