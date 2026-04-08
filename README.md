# RedactionEnv: Context-Based PII Redaction 🕵️‍♂️🛡️

**RedactionEnv** is a Reinforcement Learning (RL) environment built on the **OpenEnv** framework, specifically designed to train and evaluate AI agents on context-aware Personally Identifiable Information (PII) redaction tasks.

Unlike simple regex-based or zero-context NER models, RedactionEnv challenges agents to understand the nuances of text. It requires agents to decide whether to redact an entity based on the surrounding context (e.g., redacting a witness's name but keeping a judge's name public) and enforces strict privacy constraints through an asymmetric reward system.

---

## 🌟 Key Features

* **Context-Aware Scenarios**: Tasks include metadata like `is_public_record` and explicit `context_info` instructions to guide redaction logic.
* **Tiered Difficulty Levels**:
  * 🟢 **Easy**: Direct PII mentions (e.g., SSNs, phone numbers, standard names).
  * 🟡 **Medium**: Context-sensitive roles (e.g., redacting "suspect" names but leaving "police officer" names).
  * 🔴 **Hard**: Multi-hop contextual clues and pseudonym resolutions.
* **Asymmetric Reward Function**: Designed for real-world privacy needs. It heavily penalizes *False Negatives* (missing sensitive data) while applying a smaller penalty for *False Positives* (over-redacting safe text).
* **GRPO-Ready**: Built for Group Relative Policy Optimization (GRPO), ensuring the environment easily integrates with generation-based LLM RL tuning.
* **OpenEnv Architecture**: Fully conforms to the Meta OpenEnv standard with asynchronous WebSocket/Docker client-server execution.

---

## 📁 Project Structure

```text
RedactionEnv/
â”œâ”€â”€ baselines/
â”‚   â””â”€â”€ openai_baseline.py       # Zero-shot evaluation script using GPT-4o
â”œâ”€â”€ redaction_env/
â”‚   â”œâ”€â”€ client.py                # OpenEnv client implementation
â”‚   â”œâ”€â”€ graders.py               # Dataset samples and difficulty evaluation logic
â”‚   â”œâ”€â”€ models.py                # Pydantic schemas for Actions and Observations
â”‚   â”œâ”€â”€ reward.py                # Asymmetric reward logic calculation
â”‚   â””â”€â”€ server/                  # OpenEnv server and Docker environments
â”œâ”€â”€ training/
â”‚   â””â”€â”€ grpo_train.py            # Scaffold for GRPO training loop
â”œâ”€â”€ inference.py                 # Standardized inference and evaluation runner
â””â”€â”€ .env                         # Configuration variables for models and environments
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.10+ installed, then install the necessary dependencies:

```bash
# Create and activate your virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows

# Install dependencies (assuming you're using pip or uv)
pip install -e ./redaction_env
pip install openai python-dotenv
```

### 2. Configuration

Create or modify your `.env` file in the root directory to configure the baseline scripts and inference runners:

```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
HF_TOKEN=your_api_key_here
IMAGE_NAME=redaction_env_env:latest
REDACTION_TASK=redaction
REDACTION_BENCHMARK=redaction_env
```

### 3. Usage

#### Running the OpenAI Baseline
To test the environment purely using a standard LLM (GPT-4o) with zero-shot prompting across all difficulty tiers:

```bash
python baselines/openai_baseline.py
```

#### Running Standard Inference
OpenEnv benchmarks utilize a standardized inference file that connects to the environment (locally or via Docker) and emits `[START]`, `[STEP]`, and `[END]` evaluation blocks.

```bash
python inference.py
```

---

## 📈 Reward Structure

The heart of RedactionEnv is its privacy-first reward calculation (`redaction_env/reward.py`):

* **True Positive (TP)**: Correctly redacting sensitive data. *(Reward: +1.0)*
* **True Negative (TN)**: Correctly leaving safe text as-is. *(Reward: +0.1)*
* **False Positive (FP)**: Over-redacting safe text. *(Penalty: -1.0)*
* **False Negative (FN)**: **CRITICAL.** Missing sensitive PII. *(Penalty: -5.0)*

This enforces the agent to prioritize safety over utility, leaning towards over-redaction rather than leaking PII in ambiguous cases.

---

## 🛠️ Development & Training

To train your own small language models (SLMs) on this environment:
1. Wrap the `RedactionEnv` inside a standard Gym/Gymnasium wrapper if using traditional RL frameworks.
2. For LLMs, integrate the step interactions directly inside a TRL-compatible GRPO wrapper. The scaffold is located in `training/grpo_train.py`.
