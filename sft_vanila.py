# ====================================================
# Vanilla SFT on GSM8K with Auto Resume & Evaluation
# ====================================================

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from datasets import load_dataset
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 4
LR = 1e-5
MAX_LENGTH = 1024
OUTPUT_DIR = "./sft_gsm8k"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------- DATA PREP ----------------
def preprocess_dataset(tokenizer, max_length=1024):
    ds = load_dataset("openai/gsm8k", "main")

    def build_prompt(example):
        return (
            "You are a helpful math tutor. Solve the following problem step by step.\n\n"
            f"Q: {example['question']}\nA:"
        )

    def tokenize_example(example):
        prompt = build_prompt(example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(" " + example["answer"], add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Tokenize train and test splits separately
    train_ds = ds["train"].map(tokenize_example, remove_columns=ds["train"].column_names)
    test_ds = ds["test"].map(tokenize_example, remove_columns=ds["test"].column_names)

    # Optionally split train_ds into train/val
    train_val = train_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = train_val["train"]
    val_ds = train_val["test"]

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = preprocess_dataset(tokenizer, MAX_LENGTH)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
subset = torch.utils.data.Subset(val_ds, range(0, 20))   # first 200 examples
val_dl = DataLoader(subset, batch_size=1)
test_dl = DataLoader(test_ds, batch_size=16)

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
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(latest_ckpt_path)
else:
    print("🚀 No checkpoint found. Starting from scratch...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )
    resume_epoch = 0

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# ---------------- HELPERS ----------------
def extract_final_answer(text):
    import re
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1] if matches else text.strip()

def normalize_num_str(s):
    s = s.strip()
    if "####" in s:
        return s.split("####")[-1].strip()
    return s

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, compute_accuracy=False, max_new_tokens=128, desc="Evaluating"):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in tqdm(dataloader, desc=desc, colour="green"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Compute loss normally
        outputs = model(**batch)
        total_loss += outputs.loss.item()

        if compute_accuracy:
            # Build prompts (exclude answer tokens)
            prompts = []
            prompt_lengths = []
            for i in range(batch["input_ids"].size(0)):
                l = (batch["labels"][i] == -100).sum().item()
                prompt_lengths.append(l)
                prompts.append(batch["input_ids"][i, :l])

            # Pad prompt sequences to same length
            input_prompts = torch.nn.utils.rnn.pad_sequence(
                prompts, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(DEVICE)
            attn_prompts = (input_prompts != tokenizer.pad_token_id).long()

            # Generate from *only the prompt*
            gen = model.generate(
                input_ids=input_prompts,
                attention_mask=attn_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            preds = []
            for i in range(gen.size(0)):
                # Use stored prompt length before padding
                l = prompt_lengths[i]
                gen_only = gen[i, l:]  # skip prompt part
                text = tokenizer.decode(gen_only, skip_special_tokens=True)
                preds.append(text.strip())

            # Decode gold answers
            golds = []
            for lbl in batch["labels"]:
                ids = [t.item() for t in lbl if t.item() != -100]
                golds.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

            # Normalize
            preds = [extract_final_answer(p) for p in preds]
            golds = [normalize_num_str(g) for g in golds]
            print(preds, golds)
            # Compare
            for p, g in zip(preds, golds):
                try:
                    if float(p) == float(g):
                        total_correct += 1
                except ValueError:
                    if p.strip() == g.strip():
                        total_correct += 1
            total_samples += len(golds)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


# ---------------- EVALUATE BEFORE RESUME ----------------
if latest_ckpt_path:
    # val_loss, val_acc = evaluate(model, val_dl, tokenizer, compute_accuracy=True, desc="Validation before resume")
    test_loss, test_acc = evaluate(model, test_dl, tokenizer, compute_accuracy=True, desc="Test before resume")
    print(f"📊 Checkpoint ({latest_ckpt_path}) Evaluation:")
    # print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
    # print(f"Test Loss: {test_loss:.4f}")

# ---------------- TRAIN SETUP ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_dl) * (EPOCHS - resume_epoch)
lr_scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

# ---------------- TRAIN LOOP ----------------
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
            print(f"Epoch {epoch}/{EPOCHS} | Step {step+1} | Loss {avg_loss:.4f}")

    # Validation
    val_loss, val_acc = evaluate(model, val_dl, tokenizer, desc="Validation after epoch")
    print(f"\nEpoch {epoch} Summary:")
    print(f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}\n")

    # Save checkpoint
    save_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model saved to {save_dir}")

# ---------------- FINAL TEST EVAL ----------------
test_loss, test_acc = evaluate(model, test_dl, tokenizer, compute_accuracy=True, desc="Final Test Evaluation")
print(f"\n🎯 Final GSM8K Test Results:\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
