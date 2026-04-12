# Dataset Cleaning and Alignment Engine
An OpenEnv environment simulating real-world ML dataset cleaning.

## Overview
This environment simulates how ML data engineers clean, correct, and align training datasets. It consists of three escalating tasks:
- **Easy (Text Cleaning)**: Fix typos in an input text without altering semantics.
- **Medium (Label Correction)**: Identify and remediate improperly labeled text based on content and model confidence.
- **Hard (Dataset Alignment)**: Work across a batched dataset view to eliminate contradictory labels and balance classification distribution, alongside removal of duplicates.

## Evaluation Baseline
The baseline uses `Qwen/Qwen2.5-72B-Instruct` or equivalent to reason through observation and metadata, yielding JSON-structured actions:
- `fix_text`
- `relabel`
- `remove`
- `mark_clean`

Scores reflect partial rewards across steps based on real alignment criteria!

## Action & Observation Spaces
- **Observation Space (structured_json)**: Detailed sample text, label, system-assigned 'issues', confidence score, step counter, history, and full dataset dump (for hard task).
- **Action Space (discrete_typed)**:
  - `fix_text`: (Payload: `corrected_text`, `sample_id`) Fix typos and grammatical errors.
  - `relabel`: (Payload: `new_label`, `sample_id`) Provide a corrected label.
  - `remove`: (Payload: `sample_id`) Remove samples from the dataset.
  - `mark_clean`: Finish task / pass.

## Setup & Usage Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt uv
   uv lock
   ```
2. Start the OpenEnv server:
   ```bash
   openenv server start --dir .
   ```
   Or via Docker:
   ```bash
   docker build -t openenv-cleaner .
   docker run -p 7860:7860 openenv-cleaner
   ```
3. Run inference:
   ```bash
   export HF_TOKEN="your_hf_token"
   export API_BASE_URL="https://api.openai.com/v1" # Or Hugging Face router
   export MODEL_NAME="gpt-4o" # or another model
   python inference.py
   ```

## Baseline Scores
Running the reference `inference.py` script against `gpt-4o` yields scores around:
- **Easy (Text Cleaning)**: 0.95+
- **Medium (Label Correction)**: 0.85+
- **Hard (Dataset Alignment)**: 0.70+
