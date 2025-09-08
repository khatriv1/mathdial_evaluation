# MathDial Teacher Move Classification System

A comprehensive evaluation and optimization framework for classifying teacher pedagogical moves in math tutoring conversations using the **MathDial** dataset. This project includes three major components:

1. **Baseline Evaluation Framework** - Traditional prompting techniques
2. **APO Optimization** - Automatic Prompt Optimization for rubric refinement
3. **DSPy Implementation** - Data-driven learning approach without predefined rubrics

---

## Table of Contents

* [Project Overview](#project-overview)
* [Teacher Move Categories](#teacher-move-categories)
* [System Components](#system-components)
  * [1. Baseline Evaluation](#1-baseline-evaluation)
  * [2. APO Optimization](#2-apo-optimization)
  * [3. DSPy Implementation](#3-dspy-implementation)
* [Installation](#installation)
* [Dataset Structure](#dataset-structure)
* [Usage Guide](#usage-guide)
* [Results & Comparison](#results--comparison)

---

## Project Overview

This project evaluates and optimizes different approaches for classifying **teacher utterances** in math tutoring conversations into pedagogical move categories. The system analyzes how teachers guide students through math problems using three distinct methodologies.

---

## Teacher Move Categories

| Category    | Definition                                                     | Example                                           |
| ----------- | -------------------------------------------------------------- | ------------------------------------------------- |
| **Generic** | General conversational elements with limited pedagogical value | "Hi, how are you doing with the word problem?"    |
| **Focus**   | Guides the student to specific aspects for direct progress     | "So what should you do next?"                     |
| **Probing** | Explores underlying concepts and understanding                 | "Why do you think you need to add these numbers?" |
| **Telling** | Reveals parts of the solution when the student is stuck        | "You need to add … to … to get your answer."      |

---

## Project Structure

```text
mathdial_evaluation/
├── data/
│   ├── hold_out150.csv                    # Test dataset (150 conversations)
│   ├── active_prompting_pool_20.csv       # Active prompting pool (30 conversations)
│   ├── apo_training_100.csv              # APO optimization set (100 conversations)
│   └── dspy_300.csv                       # DSPy training set (557 utterances)
│
├── prompting/
│   ├── zero_shot.py                       # Direct classification
│   ├── few_shot.py                        # With 12 examples
│   ├── auto_cot.py                        # Chain-of-thought reasoning
│   ├── self_consistency.py                # Multiple reasoning paths
│   └── active_prompt.py                   # Uncertainty sampling
│
├── utils/
│   ├── data_loader.py                     # Data processing utilities
│   ├── mathdial_rubric.py                 # Move definitions & rubric
│   └── metrics.py                         # Evaluation metrics
│
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_few_shot.py
│   ├── evaluate_auto_cot.py
│   ├── evaluate_self_consistency.py
│   └── evaluate_active_prompt.py
│
├── apo_optimization/                       # APO SYSTEM
│   ├── mathdial_apo_system.py            # Core APO rubric optimizer
│   ├── run_apo.py                        # Automated APO runner
│   ├── config.py                         # APO configuration
│   ├── apo_copies/                       # Generated during runtime
│   │   ├── prompting/                    # Modified prompting copies
│   │   └── utils/                        # Modified rubric files
│   └── results/                          # APO output
│       ├── optimized_rubrics_eval2.json
│       ├── optimized_rubrics_eval4.json
│       └── optimized_rubrics_eval9.json
│
├── dspy_implementation/                    # DSPY SYSTEM
│   ├── dspy_mathdial_classifier.py       # GPT-3.5 base classifier
│   ├── dspy_mathdial_classifier_gpt5.py  # GPT-5 reasoning model
│   ├── run_dspy_mathdial_experiments.py  # GPT-3.5 experiments runner
│   ├── run_dspy_mathdial_gpt5.py        # GPT-5 experiments runner
│   └── results/                          # DSPy output
│       ├── dspy_mathdial_results_*.csv
│       ├── dspy_mathdial_summary.json
│       ├── mathdial_module_*_learned.json
│       └── dspy_mathdial_*_chart.png
│
├── results/                               # Main output directory
├── config.py                              # Global configuration
├── main.py                                # Main baseline evaluation script
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## System Components

### 1. Baseline Evaluation

Traditional prompting techniques for classification using predefined rubrics.

**Key Files:**
- `prompting/` - Contains 5 prompting techniques
- `utils/mathdial_rubric.py` - Teacher move definitions
- `evaluation/` - Individual technique evaluators
- `main.py` - Run all baseline techniques

### 2. APO Optimization (`apo_optimization/`)

**Automatic Prompt Optimization** to refine rubric definitions while keeping prompts at baseline.

**Key Features:**
- Generates 5 rubric variations automatically
- Tests each variation with all 5 prompting techniques
- Uses 100 samples from `apo_training_100.csv` for optimization
- Preserves `holdout_150.csv` for final testing
- Creates modified copies in `apo_copies/` during runtime

**APO Process:**
1. Generate 5 alternative rubric definitions using GPT
2. Test each rubric with all prompting techniques
3. Select best-performing rubric based on accuracy
4. Output optimized rubrics for 2, 4, and 9 conversation samples

### 3. DSPy Implementation (`dspy_implementation/`)

**Data-driven approach** that learns from examples without predefined rubrics.

**Key Features:**
- NO predefined rubric required - learns from data
- Uses Chain-of-Thought reasoning
- Trains with 100, 200, 300 sample configurations
- Supports both GPT-3.5 and GPT-5 models
- Bootstrap few-shot optimization

**DSPy Training:**
1. Load training examples from `dspy_300.csv`
2. Learn patterns using BootstrapFewShot optimization
3. Generate reasoning chains for classification
4. Test on `holdout_150.csv`

---

## Installation

### Prerequisites

* Python **3.7+**
* OpenAI API key with GPT-3.5/GPT-4/GPT-5 access
* **8 GB** RAM minimum

### Setup

```bash
# Clone repository
git clone https://github.com/khatriv1/mathdial_evaluation.git
cd mathdial_evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For DSPy implementation
pip install dspy-ai

# Configure API key
# Edit config.py with your OpenAI API key
```

---

## Dataset Structure

```
data/
├── apo_training_100.csv           # APO optimization set (100 conversations)
├── hold_out150.csv                # Test set (150 conversations)
├── active_prompting_pool_20.csv   # Active prompting examples
└── dspy_300.csv                   # DSPy training set (557 utterances)
```

**Data Split:**
- APO uses `apo_training_100.csv` for optimization
- DSPy uses `dspy_300.csv` for training
- Both test on `hold_out150.csv`
- Active prompting uses separate `active_prompting_pool_20.csv`

---

## Usage Guide

### Running Baseline Evaluation

```bash
# Run all baseline techniques
python main.py

# Run individual technique
python evaluation/evaluate_zero_shot.py
```

### Running APO Optimization

```bash
cd apo_optimization

# Run automated APO (2-3 hours)
python run_apo.py
# Select option 1 for full optimization

# Check system requirements
python run_apo.py
# Select option 3
```

**APO Output:**
- `results/optimized_rubrics_eval2.json` - 2 conversations
- `results/optimized_rubrics_eval4.json` - 4 conversations  
- `results/optimized_rubrics_eval9.json` - 9 conversations

### Running DSPy Experiments

```bash
cd dspy_implementation

# GPT-3.5 experiments (standard)
python run_dspy_mathdial_experiments.py

# GPT-3.5 with fresh API calls (no cache)
python run_dspy_mathdial_experiments.py --clear-cache

# GPT-5 experiments (reasoning model)
python run_dspy_mathdial_gpt5.py

# GPT-5 with fresh API calls
python run_dspy_mathdial_gpt5.py --clear-cache
```

**DSPy Output:**
- `results/dspy_mathdial_results_*.csv` - Individual results
- `results/dspy_mathdial_summary.json` - Performance summary
- `results/mathdial_module_*_learned.json` - Learned patterns
- `results/dspy_mathdial_*_comparison_chart.png` - Visualizations

---

## Results & Comparison

### Performance Overview

| Approach | Best Accuracy | Key Advantage | Time Required |
|----------|--------------|---------------|---------------|
| **Baseline** | ~45-55% | Simple, interpretable | Minutes |
| **APO Optimized** | ~55-65% | Improved rubric definitions | 2-3 hours |
| **DSPy GPT-3.5** | ~60-70% | Learns from data | 30-60 minutes |
| **DSPy GPT-5** | ~70-80% | Advanced reasoning | 1-2 hours |

### Key Insights

**APO Optimization:**
- Generates context-aware rubric variations
- Best for improving existing classification systems
- Maintains interpretability while boosting accuracy

**DSPy Implementation:**
- No predefined rubrics needed
- Learns directly from conversation patterns
- Scales well with more training data
- GPT-5 shows significant improvement over GPT-3.5

### Evaluation Metrics

All approaches evaluated using:
- **Accuracy**: Exact match percentage
- **Cohen's κ**: Agreement beyond chance
- **Krippendorff's α**: Reliability measure
- **ICC**: Intraclass correlation
- **Per-move accuracy**: Performance by category

---

## Advanced Features

### APO Rubric Generation

The APO system generates variations like:

```python
# Baseline
"focus": "Seeking strategy, guiding focus, recalling information"

# Optimized variant
"focus": "Questions or statements that orient students toward 
         the next step or relevant problem information"
```

### DSPy Learning Examples

DSPy learns patterns such as:

```python
# Input
teacher_utterance: "Can you explain your thinking?"
context: "Student just provided an answer"

# DSPy reasoning
"This is a probing move because the teacher is asking 
 the student to justify their reasoning"
```

---

## Configuration Options

### Model Selection

```python
# config.py
MODEL_NAME = "gpt-3.5-turbo"  # Baseline
MODEL_NAME = "gpt-4"           # Better accuracy  
MODEL_NAME = "gpt-5"           # Best reasoning (DSPy)
```

### Sample Sizes

- APO: 2, 4, 9 conversations (~10, 20, 50 utterances)
- DSPy: 100, 200, 300 training examples

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API rate limits | Increase `SLEEP_TIME` in config.py |
| DSPy cache issues | Run with `--clear-cache` flag |
| Memory errors | Reduce sample size |
| Missing data files | Check `data/` folder contents |

---

## Citation

```bibtex
@article{macina2023mathdial,
  title={MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties},
  author={Macina, Jakub and others},
  journal={arXiv preprint arXiv:2305.14536},
  year={2023}
}
```

---

## Contact

For questions about:
- **APO implementation**: See `apo_optimization/README_APO.md`
- **DSPy implementation**: See `dspy_implementation/README_DSPY.md`
- **General issues**: Open a GitHub issue