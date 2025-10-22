# ============================================================
# Inference for Saved SFT Models (Vanilla or Star)
# Auto-detects latest checkpoint and evaluates on GSM8K
# ============================================================

import os, re, torch, json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
# Set these according to which SFT run you’re evaluating
# Example: "./sft_gsm8k" or "./sft_star_gsm8k"
OUTPUT_DIR = "./sft_star_gsm8k"      # change to "./sft_gsm8k" for vanilla
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128

# ---------------- CHECKPOINT LOADING ----------------
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("epoch_")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    latest = checkpoints[-1]
    return os.path.join(output_dir, latest)

latest_ckpt = get_latest_checkpoint(OUTPUT_DIR)
print(f"🔍 Using checkpoint: {latest_ckpt}")

# ---------------- LOAD MODEL + TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(latest_ckpt)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # decoder-only

model = AutoModelForCausalLM.from_pretrained(
    latest_ckpt,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.eval()

# ---------------- GSM8K TEST DATA ----------------
def preprocess_gsm8k(tokenizer, split="test", max_length=1024):
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def tokenize_fn(example):
        prompt = f"You are a helpful math tutor. Solve step by step.\n\nQ: {example['question']}\nA:"
        answer = example["answer"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attn_mask = [1] * len(input_ids)
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attn_mask += [0] * pad_len
        else:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attn_mask = attn_mask[:max_length]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)

test_dl = preprocess_gsm8k(tokenizer, split="test")

# ---------------- HELPERS ----------------
def extract_final_answer(text):
    """Get last number from text after #### or digits."""
    if "####" in text:
        text = text.split("####")[-1]
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return matches[-1] if matches else text.strip()

def normalize_num_str(s):
    s = s.strip()
    if "####" in s:
        s = s.split("####")[-1]
    return re.sub(r"[^\d\.\-]", "", s.replace(",", ""))

# ---------------- EVALUATION ----------------
@torch.no_grad()
def evaluate(model, tokenizer, dataloader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()

        # Greedy generation for accuracy
        gen = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        prompt_len = batch["input_ids"].shape[1]
        preds = []
        for i in range(gen.size(0)):
            continuation = tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True)
            preds.append(extract_final_answer(continuation))

        golds = []
        for lbl in batch["labels"]:
            ids = [t.item() for t in lbl if t.item() != -100]
            golds.append(normalize_num_str(tokenizer.decode(ids, skip_special_tokens=True)))

        for p, g in zip(preds, golds):
            if p and g:
                try:
                    if abs(float(p) - float(g)) < 1e-3:
                        correct += 1
                except ValueError:
                    if p.strip() == g.strip():
                        correct += 1
            total += 1

    avg_loss = total_loss / len(dataloader)
    acc = correct / total if total > 0 else 0.0
    print(f"\n✅ Final Results:")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return avg_loss, acc

# ---------------- RUN ----------------
test_loss, test_acc = evaluate(model, tokenizer, test_dl)

