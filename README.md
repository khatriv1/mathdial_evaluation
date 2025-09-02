# MathDial Teacher Move Classification Evaluation

A comprehensive evaluation framework for classifying teacher pedagogical moves in math‑tutoring conversations using the **MathDial** dataset. This project compares five advanced LLM prompting strategies to determine which best matches expert annotations.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Teacher Move Categories](#teacher-move-categories)
* [Key Features](#key-features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Run Complete Evaluation Suite](#run-complete-evaluation-suite)
  * [Run an Individual Technique](#run-an-individual-technique)
* [Evaluation Metrics](#evaluation-metrics)
* [Prompting Techniques](#prompting-techniques)
* [Dataset Statistics](#dataset-statistics)
* [Output Files](#output-files)
* [Configuration Options](#configuration-options)
* [Troubleshooting](#troubleshooting)
* [Dependencies](#dependencies)
* [Pedagogical Background](#pedagogical-background)
* [Citation](#citation)
* [License](#license)
* [Contributing](#contributing)
* [Contact](#contact)

---

## Project Overview

This project evaluates different prompting techniques for classifying **teacher utterances** in math tutoring conversations into pedagogical move categories. The system analyzes how teachers guide students through math problems, identifying their teaching strategies and pedagogical approaches.

---

## Teacher Move Categories

| Category    | Definition                                                     | Intents                                                                                           | Example                                           |
| ----------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Generic** | General conversational elements with limited pedagogical value | Greeting/Farewell, General inquiry                                                                | “Hi, how are you doing with the word problem?”    |
| **Focus**   | Guides the student to specific aspects for direct progress     | Seek Strategy, Guiding Student Focus, Recall Relevant Information                                 | “So what should you do next?”                     |
| **Probing** | Explores underlying concepts and understanding                 | Asking for Explanation, Seeking Self Correction, Perturbing the Question, Seeking World Knowledge | “Why do you think you need to add these numbers?” |
| **Telling** | Reveals parts of the solution when the student is stuck        | Revealing Strategy, Revealing Answer                                                              | “You need to add … to … to get your answer.”      |

---

## Key Features

* Context‑aware classification using conversation history
* Multiple evaluation metrics (accuracy + 3 reliability measures)
* Five prompting techniques (zero‑shot → active prompting)
* Separate validation pool to prevent data leakage
* Automated pipeline to run all techniques from a single script

---

## Project Structure

```text
mathdial_evaluation/
├── data/
│   ├── hold_out150.csv              # Main dataset (287 conversations)
│   └── active_prompting_pool_20.csv # Separate pool (30 conversations)
├── prompting/
│   ├── zero_shot.py                 # Direct classification
│   ├── few_shot.py                  # With 12 examples
│   ├── auto_cot.py                  # Chain-of-thought reasoning
│   ├── self_consistency.py          # Multiple reasoning paths
│   └── active_prompt.py             # Uncertainty sampling
├── utils/
│   ├── data_loader.py               # Data processing utilities
│   ├── mathdial_rubric.py           # Move definitions & rubric
│   └── metrics.py                   # Evaluation metrics
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_few_shot.py
│   ├── evaluate_auto_cot.py
│   ├── evaluate_self_consistency.py
│   └── evaluate_active_prompt.py
├── results/                         # Output directory
├── config.py                        # Configuration settings
├── main.py                          # Main evaluation script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites

* Python **3.7+**
* OpenAI API key with GPT‑3.5/GPT‑4 access
* **8 GB** RAM (minimum recommended)

### Setup

```bash
# 1) Clone the repo
git clone <repository-url>
cd mathdial_evaluation

# 2) Create and activate a virtual environment
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

Configure your API key in `config.py`:

```python
# config.py
OPENAI_API_KEY = "your-api-key-here"
```

Verify data files exist in `data/`:

* `hold_out150.csv` (287 conversations)
* `active_prompting_pool_20.csv` (30 conversations)

> **Note:** Counts above reflect the current dumps; update if your local copies differ.

---

## Usage

### Run Complete Evaluation Suite

```bash
python main.py
```

Interactive prompts:

* **How many conversations to evaluate?** (press Enter for all 287)
* **Select techniques**
  1: Zero-shot
  2: Few-shot
  3: Auto-CoT
  4: Self-Consistency
  5: Active Prompting
  6: All techniques

### Run an Individual Technique

```bash
# Zero-shot
python evaluation/evaluate_zero_shot.py

# Few-shot
python evaluation/evaluate_few_shot.py

# Auto-CoT
python evaluation/evaluate_auto_cot.py

# Self-consistency
python evaluation/evaluate_self_consistency.py

# Active prompting
python evaluation/evaluate_active_prompt.py
```

---

## Evaluation Metrics

| Metric               | Range       | Description                      | Interpretation                                        |
| -------------------- | ----------- | -------------------------------- | ----------------------------------------------------- |
| **Accuracy**         | 0–100%      | Exact match % for move sequences | Higher is better                                      |
| **Cohen’s κ**        | −100 to 100 | Agreement beyond chance (×100)   | >80 Almost perfect; 60–80 Substantial; 40–60 Moderate |
| **Krippendorff’s α** | 0–100       | Reliability measure (×100)       | >80 Good; 67–80 Tentative; <67 Poor                   |
| **ICC**              | −100 to 100 | Intraclass correlation (×100)    | >75 Excellent; 60–75 Good; 40–59 Fair                 |

> All metrics except **Accuracy** are multiplied by 100 for clearer visualization.

---

## Prompting Techniques

1. **Zero-shot** — Direct classification without examples

   ```text
   "Classify this teacher utterance into: generic, focus, probing, or telling."
   ```

2. **Few-shot** — Uses 12 curated examples from MathDial Table 2

   ```text
   "Here are examples: [12 examples] ... Now classify this utterance."
   ```

3. **Auto-CoT** — Zero-shot with chain-of-thought reasoning

   ```text
   "Let's think step by step... What is the pedagogical intent?"
   ```

4. **Self-Consistency** — Multiple reasoning paths with majority voting (n=5)

   ```text
   "Generate 5 independent classifications and take the majority vote."
   ```

5. **Active Prompting** — Learns from uncertain/wrong examples via uncertainty sampling

   ```text
   "Learn from these challenging cases: [8 examples] ... Now classify."
   ```

---

## Dataset Statistics

**Main Dataset (`hold_out150.csv`)**

* Total Conversations: **287**
* Total Rows: **287**
* Columns: **13**
* Avg. Teacher Moves per Conversation: **8–12**

**Teacher Move Distribution**

* **Focus**: 37% (Most common — guiding strategies)
* **Probing**: 24% (Exploring understanding)
* **Generic**: 21% (Conversational elements)
* **Telling**: 18% (Direct instruction)

**Active Prompting Pool (`active_prompting_pool_20.csv`)**

* Total Conversations: **30**
* Separate validation set for uncertainty sampling
* No overlap with main test set

---

## Output Files

Each run creates a timestamped directory under `results/`:

```text
results/mathdial_evaluation_YYYYMMDD_HHMMSS/
├── mathdial_all_techniques_comparison.csv     # Metrics summary
├── mathdial_all_techniques_comparison.png     # Visual comparison
├── mathdial_all_detailed_results.csv          # All predictions
├── mathdial_comprehensive_report.txt          # Detailed analysis
└── [technique_name]/
    ├── detailed_results.csv                   # Per-conversation results
    └── metrics_summary.csv                    # Technique metrics
```

---

## Configuration Options

**Model Selection**

```python
# config.py
MODEL_ID = "gpt-3.5-turbo-0125"  # Default
# MODEL_ID = "gpt-4"             # Uncomment for GPT-4
```

**Rate Limiting**

```python
SLEEP_TIME = 1  # Seconds between API calls
```

**Sample Size**

```bash
# Evaluate a subset (e.g., first 50 conversations)
python main.py
# When prompted, enter: 50
```

Other options—output directories, random seeds, etc.—are configurable in `config.py`.

---

## Troubleshooting

| Issue               | Solution                                                                       |
| ------------------- | ------------------------------------------------------------------------------ |
| `FileNotFoundError` | Run from the repository root (`mathdial_evaluation/`) and check `data/` files. |
| API key errors      | Verify `OPENAI_API_KEY` in `config.py`.                                        |
| Import errors       | Re-install dependencies: `pip install -r requirements.txt`.                    |
| Rate limit errors   | Increase `SLEEP_TIME` in `config.py`.                                          |
| Memory issues       | Reduce the number of conversations evaluated.                                  |

---

## Dependencies

Core requirements (see `requirements.txt`):

* `numpy >= 1.21.0` — Numerical computations
* `pandas >= 1.3.0` — Data manipulation
* `matplotlib >= 3.5.0` — Visualizations
* `scikit-learn >= 1.0.0` — Metrics calculation
* `scipy >= 1.7.0` — Statistical functions
* `krippendorff >= 0.5.0` — Krippendorff’s alpha
* `openai >= 1.0.0` — GPT API access

---

## Pedagogical Background

* **Scaffolding vs. Direct Instruction**

  * Scaffolding moves (**Focus + Probing**): \~61% of moves
  * Direct instruction (**Telling**): \~18% of moves
  * Conversational (**Generic**): \~21% of moves
* **Educational Impact**

  * Focus and Probing promote deeper learning
  * Telling provides necessary support when students are stuck
  * A balanced mix often yields the best outcomes

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{macina2023mathdial,
  title={MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems},
  author={Macina, Jakub and Daheim, Nico and Chowdhury, Sankalan Pal and Sinha, Tanmay and Kapur, Manu and Gurevych, Iryna and Sachan, Mrinmaya},
  journal={arXiv preprint arXiv:2305.14536},
  year={2023}
}
```

---

## License

This project is for **research and educational** purposes. The MathDial dataset follows its original licensing terms.

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description

---

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
