# ============================================================
# GSM8K Few-shot + Self-consistency Evaluation for Llama-3.2-3B
# Batched, faster, 60%-target version
# ============================================================

import os, re, json, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import userdata
from collections import Counter

# -------------------- Config --------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16             # realistic for 3B model; 64 is too high for 8x generation
MAX_NEW_TOKENS = 256
N_SAMPLES = 8               # self-consistency samples
TEMP = 0.8
OUT_FILE = "results/gsm8k_fewshot_sc.jsonl"

# Hugging Face token
HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add your HF_TOKEN to Colab secrets"

# -------------------- Load dataset --------------------
ds = load_dataset("gsm8k", "main", split="test")

# -------------------- Load model + tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)
model.eval()

# -------------------- Helpers --------------------
def extract_final_answer(text: str):
    """Extract final numeric answer from model output."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else text.strip()

def normalize_gold(ans: str):
    ans = ans.strip()
    if "####" in ans:
        ans = ans.split("####")[-1]
    return re.sub(r"[^\d\.\-]", "", ans.replace(",", ""))

# ------------- Few-shot exemplars (short CoT) -------------
FEW_SHOT_EXAMPLES = """
Q: If there are 3 apples and you buy 2 more, how many apples do you have?
A: We start with 3 apples and buy 2 more, so 3 + 2 = 5.
#### 5

Q: A car travels 60 miles per hour for 3 hours. How far does it go?
A: Distance = speed × time = 60 × 3 = 180 miles.
#### 180
"""

# ------------- Build prompts -------------
def build_prompt(question):
    return (
        "You are a brilliant math tutor who reasons carefully step by step.\n"
        "Show your reasoning clearly and finish with '#### [number]'.\n\n"
        + FEW_SHOT_EXAMPLES
        + f"\nQ: {question}\nA:"
    )

# ------------- Self-consistency evaluation -------------
@torch.no_grad()
def evaluate_self_consistency(model, tokenizer, dataset, batch_size=BATCH_SIZE):
    all_results, correct = [], 0
    n_total = len(dataset)

    for start_idx in tqdm(range(0, n_total, batch_size), desc="Evaluating (batched)"):
        batch = dataset[start_idx : start_idx + batch_size]

        # Build batched prompts
        prompts = [build_prompt(q) for q in batch["question"]]
        inputs = tokenizer(
            [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in prompts
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        # Generate N_SAMPLES per prompt
        gens = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMP,
            top_p=0.9,
            num_return_sequences=N_SAMPLES,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # group every N_SAMPLES generations per input
        batch_size_actual = len(prompts)
        group_size = N_SAMPLES
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(batch_size_actual):
            # find its N_SAMPLES outputs
            start = i * group_size
            end = start + group_size
            answers = []
            for j in range(start, end):
                text = tokenizer.decode(gens[j, prompt_len:], skip_special_tokens=True)
                ans = extract_final_answer(text)
                if re.match(r"[-+]?\d*\.?\d+", ans):
                    answers.append(ans)
            pred = ""
            if answers:
                pred = Counter(answers).most_common(1)[0][0]
            gold = normalize_gold(batch["answer"][i])
            if pred == gold:
                correct += 1
            all_results.append(
                {"question": batch["question"][i], "gold": gold, "pred": pred, "samples": answers}
            )
            print(correct / n_total)

    acc = correct / n_total
    print(f"\n✅ Final Accuracy: {acc*100:.2f}% ({correct}/{n_total})")
    return acc, all_results


# -------------------- Run Evaluation --------------------
accuracy, results = evaluate_self_consistency(model, tokenizer, ds, batch_size=BATCH_SIZE)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Results saved to {OUT_FILE}")

