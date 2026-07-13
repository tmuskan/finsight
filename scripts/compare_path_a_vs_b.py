"""
compare_path_a_vs_b.py — Head-to-head: base Mistral (Path B) vs fine-tuned adapter (Path A).

Runs the SAME test cases through:
    Path A: Mistral-7B-Instruct-v0.3 + musk1209/finsight-qlora-mistral adapter
    Path B: Mistral-7B-Instruct-v0.3 with system-prompt engineering only

Both models get identical prompts, identical generation parameters, and
identical contexts. Any output difference comes from the adapter.

Intended to run on Kaggle T4 (needs GPU for reasonable speed).
"""

import gc
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# Enable importing our llm module (for its system prompt loading pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "rag"))


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_REPO = "musk1209/finsight-qlora-mistral"
SYSTEM_PROMPT_PATH = Path("configs/system_prompt.txt")


TEST_CASES = [
    # Basic extraction
    {
        "id": "sell",
        "context": "The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services.",
        "question": "What does the company sell?",
        "expected_theme": "smartphones, computers, tablets, wearables, accessories, services",
    },
    # Numerical
    {
        "id": "revenue_growth",
        "context": "Total revenue for fiscal year 2025 increased 6% to $394.3 billion, compared to $371.5 billion in fiscal 2024. This increase was primarily driven by growth in Services revenue.",
        "question": "By how much did total revenue grow in fiscal 2025?",
        "expected_theme": "6% / $22.8B / driven by Services",
    },
    # Risk factors
    {
        "id": "risks",
        "context": "The Company's operations and financial performance may be adversely affected by geopolitical tensions, natural disasters, and public health crises that disrupt global supply chains.",
        "question": "What are the risks to the company's operations?",
        "expected_theme": "geopolitical tensions, natural disasters, health crises, supply chain",
    },
    # Adversarial: answer NOT in context
    {
        "id": "no_answer",
        "context": "The Company designs, manufactures and markets smartphones and computers.",
        "question": "What was the CEO's compensation last year?",
        "expected_theme": "should refuse — info not in context",
    },
    # Adversarial: precise number required
    {
        "id": "exact_number",
        "context": "Cash and cash equivalents totaled $50.2 billion at the end of the quarter, up from $48.7 billion a year earlier.",
        "question": "What was cash and cash equivalents this quarter?",
        "expected_theme": "exact figure $50.2 billion",
    },
    # Multi-fact synthesis
    {
        "id": "multi_fact",
        "context": "Segment A generated revenue of $120 billion, up 12% year-over-year. Segment B generated revenue of $80 billion, down 3%. Total company revenue was $200 billion.",
        "question": "Which segment performed better?",
        "expected_theme": "Segment A — grew, higher revenue",
    },
]


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_prompt(system_prompt: str, context: str, question: str) -> str:
    """Same format for both models — the only difference is adapter on/off."""
    user_message = (
        f"{system_prompt}\n\n"
        f"Context: {context.strip()}\n\n"
        f"Question: {question.strip()}"
    )
    return f"[INST] {user_message} [/INST]"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    system_prompt = load_system_prompt()

    print(f"Loading base model {BASE_MODEL} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    print(f"Attaching adapter {ADAPTER_REPO}...")
    model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=hf_token)
    model.eval()

    # Run every test case through both configurations on the SAME model instance
    # (PEFT lets us toggle the adapter on/off without reloading — saves VRAM)
    results = []
    for tc in TEST_CASES:
        prompt = build_prompt(system_prompt, tc["context"], tc["question"])

        # PATH B: base model, no adapter
        with model.disable_adapter():
            path_b_answer = generate(model, tokenizer, prompt)

        # PATH A: adapter re-enabled after context exits
        path_a_answer = generate(model, tokenizer, prompt)

        results.append({
            "id": tc["id"],
            "question": tc["question"],
            "expected_theme": tc["expected_theme"],
            "path_a_answer": path_a_answer,
            "path_b_answer": path_b_answer,
        })

        gc.collect()
        torch.cuda.empty_cache()

    # Print results
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 78)

    for r in results:
        print(f"\n### {r['id']}: {r['question']}")
        print(f"Expected: {r['expected_theme']}")
        print(f"\n  PATH A (fine-tuned):")
        print(f"    {r['path_a_answer']}")
        print(f"\n  PATH B (prompt only):")
        print(f"    {r['path_b_answer']}")
        print("-" * 78)


if __name__ == "__main__":
    main()