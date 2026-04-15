# 🧹 Dataset Cleaning & Alignment Engine

<div align="center">

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-4CAF50?style=for-the-badge&logo=checkmarx&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A production-grade OpenEnv reinforcement learning environment simulating real-world ML dataset cleaning and alignment workflows.**

[🚀 Live Demo](https://huggingface.co/spaces/madhutiwari/dataset-cleaning-env) · [📖 OpenEnv Docs](https://github.com/openenv) · [🐛 Report Bug](https://github.com/MadhuTiwari-345/dataset-cleaning-env/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why This Environment?](#-why-this-environment)
- [Architecture](#-architecture)
- [Observation Space](#-observation-space)
- [Action Space](#-action-space)
- [Tasks](#-tasks)
- [Reward Function](#-reward-function)
- [Baseline Scores](#-baseline-scores)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)

---

## 🌍 Overview 

The **Dataset Cleaning & Alignment Engine** simulates the real-world ML engineering workflow of cleaning, correcting, and aligning training datasets before model training. Every ML team performs these tasks this environment turns that workflow into a structured RL benchmark.

An AI agent receives noisy, real-world-style dataset samples and must take corrective actions over multiple steps to improve quality. The environment tracks partial progress and rewards improvement at each step, making it ideal for training and evaluating reasoning agents.

```
Raw Dataset → Text Cleaning → Label Correction → Deduplication → Alignment → Final Dataset
```

---

## 💡 Why This Environment?

| Problem | What This Env Tests |
|---|---|
| Real ML engineers clean data daily | Can an agent perform structured data QA? |
| Labels are often wrong or inconsistent | Does the agent understand semantic correctness? |
| Datasets contain duplicates and noise | Can the agent identify and remove redundancy? |
| Cleaning requires multi-step reasoning | Does the agent plan actions strategically? |

This environment is **novel in the OpenEnv ecosystem** — no prior environment models the ML data engineering pipeline. It fills a direct gap for evaluating agents on tasks that require structured reasoning over semi-structured data.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OpenEnv Interface                   │
│  POST /reset   POST /step   GET /state               │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │    FastAPI Server     │
         │    (env.py)           │
         └───────────┬───────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌───────────┐
│  Task 1  │   │  Task 2  │   │  Task 3   │
│  Text    │   │  Label   │   │  Dataset  │
│ Cleaning │   │ Correct. │   │ Alignment │
│  (Easy)  │   │ (Medium) │   │  (Hard)   │
└─────────┘   └──────────┘   └───────────┘
```

---

## 👁 Observation Space

Each step returns a structured JSON observation:

```python
class Observation(BaseModel):
    sample_id:  str          # Unique sample identifier
    text:       str          # Raw or partially cleaned text
    label:      str          # Current label ("positive" | "negative")
    issues:     List[str]    # Detected issues: ["typo", "wrong_label", "duplicate"]
    confidence: float        # Model confidence score [0.0 – 1.0]
    step:       int          # Current step number
    history:    List[str]    # Previous actions taken this episode
```

**Example:**
```json
{
  "sample_id": "1",
  "text": "Ths prodct is absolutly amazng",
  "label": "positive",
  "issues": ["typo"],
  "confidence": 0.62,
  "step": 1,
  "history": []
}
```

---

## ⚡ Action Space

The agent selects one discrete typed action per step:

```python
class Action(BaseModel):
    type:    Literal["fix_text", "relabel", "remove", "mark_clean"]
    payload: Dict[str, Any]
```

| Action | Description | Payload Fields |
|---|---|---|
| `fix_text` | Correct typos/grammar in sample text | `corrected_text`, `sample_id` |
| `relabel` | Change the sentiment label | `new_label`, `sample_id` |
| `remove` | Delete a duplicate/irrelevant sample | `sample_id` |
| `mark_clean` | Finalize sample as clean and correct | `sample_id` |

---

## 🎯 Tasks

### 🟢 Task 1 — Text Cleaning `(Easy)`

**Objective:** Fix spelling and grammar errors in the sample text without altering its meaning.

| | |
|---|---|
| **Input** | `"Ths prodct is absolutly amazng"` |
| **Expected** | `"This product is absolutely amazing"` |
| **Max Steps** | 5 |
| **Grader** | Edit distance improvement + semantic similarity |

```python
def grade(original, corrected, reference):
    score = 0.0
    if spelling_error_rate(corrected) < spelling_error_rate(original):
        score += 0.5
    if semantic_similarity(corrected, reference) > 0.85:
        score += 0.5
    return score  # [0.0 – 1.0]
```

---

### 🟡 Task 2 — Label Correction `(Medium)`

**Objective:** Detect and fix incorrectly labeled samples based on text sentiment.

| | |
|---|---|
| **Input** | `text: "I absolutely hate this product"` / `label: "positive"` |
| **Expected** | `label: "negative"` |
| **Max Steps** | 8 |
| **Grader** | Label match + confidence weighting |

```python
def grade(predicted, ground_truth, confidence):
    if predicted == ground_truth:
        return 1.0
    elif confidence < 0.4:   # close call
        return 0.4
    return 0.0
```

---

### 🔴 Task 3 — Dataset Alignment `(Hard)`

**Objective:** Given a multi-sample dataset, fix label inconsistencies, remove duplicates, and ensure class balance.

| | |
|---|---|
| **Samples** | 4 inter-related samples (3a, 3b, 3c, 3d) |
| **Challenges** | Wrong labels, duplicate entries, conflicting annotations |
| **Max Steps** | 15 |
| **Grader** | Composite score across 3 dimensions |

```python
def grade(dataset):
    return (
        uniqueness_score(dataset)        * 0.30 +
        label_consistency_score(dataset) * 0.40 +
        class_balance_score(dataset)     * 0.30
    )
```

---

## 💰 Reward Function

The reward function provides **dense, per-step signals** — not just binary episode outcomes:

```python
def compute_reward(prev_state, curr_state, action, done):
    reward = 0.0

    # Text quality delta
    reward += (prev_error_rate - curr_error_rate) * 0.4

    # Correct relabeling bonus
    if action.type == "relabel" and new_label_is_correct:
        reward += 0.5

    # Unnecessary deletion penalty
    if action.type == "remove" and sample_was_valid:
        reward -= 0.3

    # Step efficiency cost
    reward -= 0.02

    # Repeat action penalty
    if action == last_action:
        reward -= 0.15

    # Episode completion bonus
    if done and final_quality_score > 0.8:
        reward += 1.0

    return max(-1.0, min(1.0, reward))
```

**Reward range:** `[-1.0, 1.0]` per step

---

## 📊 Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Steps | Score | Success |
|---|---|---|---|---|
| `text-cleaning` | 🟢 Easy | 2 | **1.000** | ✅ |
| `label-correction` | 🟡 Medium | 2 | **1.000** | ✅ |
| `dataset-alignment` | 🔴 Hard | 3 | **1.000** | ✅ |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/MadhuTiwari-345/dataset-cleaning-env
cd dataset-cleaning-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn env:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
# Set credentials
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run baseline agent
python inference.py
```

### Expected Output

```
[START] task=text-cleaning env=dataset-cleaning-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"type":"fix_text","payload":{"corrected_text":"This product is absolutely amazing","sample_id":"1"}} reward=0.78 done=false error=null
[STEP] step=2 action={"type":"mark_clean","payload":{"sample_id":"1"}} reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.78,1.00
```

---

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t openenv-cleaner .

# Run container
docker run -d -p 7860:7860 --name openenv-container openenv-cleaner

# Verify
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "text-cleaning"}'
```

### Validate Submission

```bash
bash validate.sh http://localhost:7860 .
```

```
✅ PASSED -- HF Space is live and responds to /reset
✅ PASSED -- Docker build succeeded
✅ PASSED -- openenv validate passed
   All 3/3 checks passed!
```

### HuggingFace Spaces

Live at: **https://huggingface.co/spaces/madhutiwari/dataset-cleaning-env**

---

## 📁 Project Structure

```
dataset-cleaning-env/
├── env.py                  # Main OpenEnv FastAPI application
├── inference.py            # Baseline LLM agent script
├── openenv.yaml            # OpenEnv metadata and task definitions
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── validate.sh             # Submission validator script
├── data/
│   └── samples.json        # Pre-generated noisy dataset samples
├── tasks/
│   ├── task1_cleaning.py   # Easy: text cleaning + grader
│   ├── task2_labeling.py   # Medium: label correction + grader
│   └── task3_alignment.py  # Hard: full dataset alignment + grader
└── README.md
```

---

## ⚙️ Configuration

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token | Required |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## 📄 OpenEnv YAML

```yaml
name: dataset-cleaning-env
version: "1.0.0"
description: >
  RL environment simulating ML dataset cleaning and alignment.
tags:
  - openenv
  - data-cleaning
  - nlp
  - alignment
tasks:
  - id: text-cleaning
    difficulty: easy
    max_steps: 5
  - id: label-correction
    difficulty: medium
    max_steps: 8
  - id: dataset-alignment
    difficulty: hard
    max_steps: 15
observation_space: structured_json
action_space: discrete_typed
reward_range: [-1.0, 1.0]
```

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

Built for the **OpenEnv Hackathon** · Deployed on **HuggingFace Spaces**

**[🚀 Try it Live](https://huggingface.co/spaces/madhutiwari/dataset-cleaning-env)**

</div>
