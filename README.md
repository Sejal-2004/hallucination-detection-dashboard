# Can Small Language Models Tell When They Are Wrong?

An interactive dashboard that shows the results of my final semester research project on **hallucination detection** in small language models, with and without retrieved context.

**Live dashboard:** https://hallucination-detection-dashboard.streamlit.app/ 

## What this project does

Language models sometimes give wrong answers in a confident voice. This project tests two simple detectors that do not need the correct answer:

- **D0 (answer agreement):** ask the same question 5 more times and check how often the answer changes.
- **D1 (explanation similarity):** compare the one-sentence reasons of the 5 answers using sentence embeddings (all-MiniLM-L6-v2).

Detector quality is measured with AUROC (0.5 = chance, 1.0 = perfect) and 95% bootstrap confidence intervals.

## Setup

- **Models:** Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct
- **Data:** ARC-Challenge science questions (150 dev, 300 test, seed 42)
- **Conditions:** question only (no context) and question with retrieved passages (context)
- **Sampling:** 1 greedy main answer and 5 samples at temperature 0.7

## Main findings

- Retrieved context did not improve accuracy for either model.
- Qwen3-4B is almost always consistent, even when wrong, so D0 is close to chance.
- Phi-4-mini gives a modest detection signal (AUROC about 0.63 to 0.70).
- No detector was clearly better, and the sample size gives wide confidence intervals.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` - the Streamlit dashboard
- `data/summary.csv` - results table used by the dashboard
- `requirements.txt` - Python packages

The full code and method details are in a separate research repository: <!-- paste research repo link here -->

## Author
**Name:** Sejal Singh

**Email:** [sejalsingj8647@gmail.com](mailto:sejalsingh8647@gmail.com)

**LinkedIn:** [Connect with me](www.linkedin.com/in/sejal-singh-97669b314)
