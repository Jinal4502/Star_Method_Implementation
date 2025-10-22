# 🌟 STaR Method Implementation (GSM8K Fine-Tuning)

This repository implements **Supervised Fine-Tuning (SFT)** and **STaR (Self-Taught Reasoner)** methods on the **GSM8K** mathematical reasoning dataset using **LLaMA-3.2-3B-Instruct**.

---

## 📂 Files Overview

| File | Description |
|------|--------------|
| **zero_shot.py** | Evaluates the base LLaMA model on GSM8K in a zero-shot setup using a reasoning-based prompt. |
| **bootstrapped_data_generation.py** | Generates the STaR bootstrapped dataset by prompting the model to produce rationales and correcting with hints if answers mismatch. |
| **sft_vanila.py** | Fine-tunes the base model directly on GSM8K (Vanilla SFT). |
| **sft_star.py** | Fine-tunes the model using the bootstrapped rationalized dataset (STaR SFT). |
| **inference.py** | Loads the latest saved checkpoint (Vanilla or STaR) and evaluates on the GSM8K test set. |
| **star_bootstrapped_train_fixed.jsonl** | The generated bootstrapped dataset containing questions, rationales, and gold answers. |

---

## 🧩 SLURM Scripts

| Script | Purpose |
|---------|----------|
| **sbatch_zero_shot.sh** | Runs `zero_shot.py` on the HPC cluster. |
| **sbatch_bootstrap.sh** | Runs `bootstrapped_data_generation.py` for dataset creation. |
| **sbatch_vanila.sh** | Launches Vanilla SFT training on GSM8K. |
| **sbatch_star.sh** | Launches STaR SFT training on bootstrapped data. |
| **sbatch_inference.sh** | Executes model evaluation/inference job. |

---

## 🧠 Quick Start

```bash
# Run zero-shot baseline
sbatch sbatch_zero_shot.sh

# Generate bootstrapped data (STaR)
sbatch sbatch_bootstrap.sh

# Train Vanilla SFT
sbatch sbatch_vanila.sh

# Train STaR SFT
sbatch sbatch_star.sh

# Evaluate latest checkpoint
sbatch sbatch_inference.sh

```

**Author:** Jinal Vyas  
**Course:** CSE 573 – Fall 2025  
**Institution:** Arizona State University  
