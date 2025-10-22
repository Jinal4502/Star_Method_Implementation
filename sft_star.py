# ============================================================
# VANILLA SFT (STAR + GSM8K)
# Auto Resume + Validation Loss + Test Accuracy
# ============================================================

import os, json, re, torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login

# ---------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 4
LR = 1e-5
MAX_LENGTH = 1024
OUTPUT_DIR = "./sft_star_gsm8k"
DATA_PATH = "/home/jjvyas1/Star/results/star_bootstrapped_train_fixed.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOGIN (if private model) ----------------
login(token="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")

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------- STAR DATASET ----------------
class StarDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    self.samples.append(ex)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed line: {e}")
        print(f"✅ Loaded {len(self.samples)} samples from {path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        q = ex.get("question", "").strip()
        rationale = ex.get("rationale", "").strip()
        ans = ex.get("gold_answer", "").strip()

        prompt = f"You are a helpful math tutor. Solve step by step.\n\nQ: {q}\nA:"
        output = rationale + f"\n#### {ans}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(" " + output, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attn_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attn_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attn_mask = attn_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "labels": torch.tensor(labels)
        }

# ---------------- LOAD TRAIN / VAL ----------------
dataset = StarDataset(DATA_PATH, tokenizer, MAX_LENGTH)
n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

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
    return DataLoader(ds, batch_size=BATCH_SIZE)

test_dl = preprocess_gsm8k(tokenizer, split="test", max_length=MAX_LENGTH)

# ---------------- AUTO RESUME LOGIC ----------------
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("epoch_")]
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    latest_ckpt = checkpoints[-1]
    latest_epoch = int(latest_ckpt.split("_")[-1])
    return os.path.join(output_dir, latest_ckpt), latest_epoch

latest_ckpt_path, resume_epoch = get_latest_checkpoint(OUTPUT_DIR)

if latest_ckpt_path:
    print(f"🔄 Found checkpoint: {latest_ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        latest_ckpt_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        use_auth_token=True,
    )
else:
    print("🚀 No checkpoint found. Starting from base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        use_auth_token=True,
    )
    resume_epoch = 0

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_dl) * (EPOCHS - resume_epoch)
lr_scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

# ---------------- HELPERS ----------------
def extract_final_answer(text):
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1] if matches else text.strip()

def normalize_num_str(s):
    s = s.strip()
    if "####" in s:
        return s.split("####")[-1].strip()
    return s

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, compute_accuracy=False, desc="Evaluating"):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in tqdm(dataloader, desc=desc, colour="green"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()

        if compute_accuracy:
            gen = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = batch["input_ids"].shape[1]
            gen_only = gen[:, prompt_len:]
            preds = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            golds = []
            for lbl in batch["labels"]:
                ids = [t.item() for t in lbl if t.item() != -100]
                golds.append(tokenizer.decode(ids, skip_special_tokens=True))

            preds = [extract_final_answer(p) for p in preds]
            golds = [normalize_num_str(g) for g in golds]
            total_correct += sum(p == g for p, g in zip(preds, golds))
            total_samples += len(golds)

    avg_loss = total_loss / len(dataloader)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc

# ---------------- TRAINING LOOP ----------------
for epoch in range(resume_epoch + 1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", colour="blue")

    for step, batch in enumerate(pbar):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        if (step + 1) % 500 == 0 or (step + 1) == len(train_dl):
            print(f"[Epoch {epoch}] Step {step+1}/{len(train_dl)} | Train Loss: {avg_loss:.4f}")

    # Validation (loss only)
    val_loss, _ = evaluate(model, val_dl, tokenizer, compute_accuracy=False, desc="Validation (loss only)")
    print(f"\nEpoch {epoch} Summary:")
    print(f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}\n")

    # Save checkpoint
    save_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model saved to {save_dir}")

# ---------------- FINAL TEST (loss + accuracy) ----------------
test_loss, test_acc = evaluate(model, test_dl, tokenizer, compute_accuracy=True, desc="GSM8K Test")
print(f"\n🎯 GSM8K Test Results:\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

