
# -------------------- Imports --------------------
import os
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Config --------------------
model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
batch_size = 128
max_new_tokens = 256
device = "cuda:0" if torch.cuda.is_available() else "cpu"
out_file = "results/star_bootstrapped_train_fixed.jsonl"
os.makedirs(os.path.dirname(out_file), exist_ok=True)

# -------------------- Load dataset --------------------
ds = load_dataset("gsm8k", "main", split="train")

# -------------------- Load model + tokenizer --------------------
print("Loading model:", model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # keep behavior consistent

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    dtype=torch.float16,
    device_map="auto",
    use_auth_token=True
)

# -------------------- Helper functions --------------------
def extract_final_answer(text):
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        return matches[-1]
    m = re.search(r"(?:####|Answer[:\s]+)\s*([-\d\.]+)", text)
    if m:
        return m.group(1)
    return text.strip()

def normalize_num_str(s):
    s = s.strip()
    if '####' in s:
        return s.split('####')[-1].strip()
    return s

def make_prompt(question):
    return (
        "You are a helpful math tutor. Solve the following problem step by step. "
        "Show your reasoning and then give the final answer.\n\n"
        f"Q: {question}\nA:"
    )

def make_hint_prompt(question, gold_answer):
    return (
        "You are a helpful math tutor. The correct final answer is already known. "
        "Explain step by step how to arrive at it, showing your reasoning clearly. "
        "Conclude again with the final answer.\n\n"
        f"Q: {question}\nCorrect Answer: {gold_answer}\nA:"
    )

def generate_single(prompt):
    msg = [[{"role": "user", "content": prompt}]]
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    prompt_len = inputs["attention_mask"].sum(dim=1).item()
    cont = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return cont

# -------------------- Load already processed questions --------------------
processed_questions = set()
if os.path.exists(out_file):
    print(f"Resuming — reading already processed questions from {out_file}")
    with open(out_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_questions.add(data["question"])
            except Exception:
                continue
    print(f"Loaded {len(processed_questions)} processed questions — will skip these.")

# -------------------- Generation config --------------------
gen_cfg = {
    "max_new_tokens": max_new_tokens,
    "do_sample": False,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

# -------------------- Bootstrapped generation (resume-safe) --------------------
model.to(device)
written = 0

# open file in append mode
with open(out_file, "a") as fout:
    for i in tqdm(range(0, len(ds), batch_size), desc="Bootstrapping"):
        batch = ds[i:i+batch_size]

        # Skip examples already processed
        questions, answers = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            if q not in processed_questions:
                questions.append(q)
                answers.append(a)
        if not questions:
            continue  # skip whole batch if already covered

        prompts = [make_prompt(q) for q in questions]
        golds = [normalize_num_str(g) for g in answers]
        conversations = [[{"role": "user", "content": p}] for p in prompts]

        try:
            inputs = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        except Exception as e:
            print("Tokenization failed; falling back to per-example generation.", e)
            inputs = None

        if inputs is None or inputs["input_ids"].shape[0] != len(prompts):
            for q, gold in zip(questions, golds):
                cont = generate_single(make_prompt(q))
                pred = extract_final_answer(cont)
                if pred != gold:
                    cont = generate_single(make_hint_prompt(q, gold))
                    pred = extract_final_answer(cont)
                fout.write(json.dumps({"question": q, "gold_answer": gold, "rationale": cont}) + "\n")
                processed_questions.add(q)
                written += 1
            continue

        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_cfg)

        if outputs.shape[0] != inputs["input_ids"].shape[0]:
            print("Batch size mismatch; falling back to per-example.")
            for q, gold in zip(questions, golds):
                cont = generate_single(make_prompt(q))
                pred = extract_final_answer(cont)
                if pred != gold:
                    cont = generate_single(make_hint_prompt(q, gold))
                    pred = extract_final_answer(cont)
                fout.write(json.dumps({"question": q, "gold_answer": gold, "rationale": cont}) + "\n")
                processed_questions.add(q)
                written += 1
            continue

        for idx in range(outputs.shape[0]):
            prompt_len_i = int(prompt_lens[idx])
            cont_tokens = outputs[idx, prompt_len_i:]
            cont = tokenizer.decode(cont_tokens, skip_special_tokens=True).strip()
            pred = extract_final_answer(cont)
            gold = golds[idx]
            q = questions[idx]

            if pred != gold:
                cont = generate_single(make_hint_prompt(q, gold))
                pred = extract_final_answer(cont)

            fout.write(json.dumps({"question": q, "gold_answer": gold, "rationale": cont}) + "\n")
            processed_questions.add(q)
            written += 1

print(f"✅ Done — wrote {written} new examples (total now {len(processed_questions)} in {out_file})")
