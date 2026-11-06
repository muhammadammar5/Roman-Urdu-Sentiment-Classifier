# Roman Urdu Sentiment Classifier

Fine-tuned **DistilBERT** that classifies Roman Urdu (Urdu written in Latin script, often mixed with English) as **Negative, Neutral or Positive**. Compared against a **TF-IDF + SVM baseline**, with a custom **text normaliser** for Roman Urdu's inconsistent spelling, an **inference benchmark** (batching and FP16) and a **Gradio demo** for Hugging Face Spaces.

**Live demo:** _add your Hugging Face Space link here_

## Highlights

- **Reproducible pipeline:** download, clean, de-duplicate, stratified split and train, all through CLI commands with a fixed seed.
- **Leakage-safe data handling:** duplicates are removed on the *normalised* text, and texts with conflicting labels are dropped before splitting.
- **Roman Urdu normaliser:** collapses stretched letters, strips URLs/mentions, and maps spelling variants ("acha", "achha", "accha") to one form.
- **Honest baselines and ablations:** TF-IDF + LinearSVC, and DistilBERT trained with and without normalisation.
- **Plain PyTorch training loop:** dynamic padding, mixed-precision (FP16), warm-up schedule, early stopping on validation macro-F1.
- **Fast inference:** length-sorted batching and FP16, benchmarked in milliseconds per review.
- **Tested:** unit tests for the normaliser, data pipeline and Space sync.

## Dataset

[Roman Urdu Data Set](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set) (Sharf, 2017), UCI Machine Learning Repository, licensed **CC BY 4.0**. About 20,000 short texts (e-commerce reviews, Facebook comments, tweets) labelled Positive, Negative or Neutral.

> Sharf, Z. (2017). *Roman Urdu Data Set* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58325

Cleaning removes empty rows, fixes the known "Neative" label typo, drops texts with conflicting labels and removes duplicates, so the final count is lower than the raw count. Exact numbers are written to `data/processed/<mode>/stats.json`.

## Quick start (Google Colab, recommended)

Open `notebooks/colab_train.ipynb` in Colab, choose a **T4 GPU** runtime and run all cells.

## Quick start (local)

```bash
git clone https://github.com/muhammadammar5/Roman-Urdu-Sentiment-Classifier.git
cd Roman-Urdu-Sentiment-Classifier
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data --download --mode three          # or --mode binary
python -m src.baseline --mode three                 # TF-IDF + SVM
python -m src.train --mode three --run-name tuned   # fine-tune DistilBERT
python -m src.compare                               # results table
python -m src.benchmark --model-dir models/tuned    # speed benchmark
```

Training on CPU works but is slow; a free Colab T4 takes a few minutes per epoch at most.

## Experiments

```bash
python -m src.train --mode three --run-name no_norm --no-normalize      # normalisation ablation
python -m src.train --mode three --model distilbert-base-uncased --run-name uncased
python -m src.train --mode three --class-weights --run-name weighted
python -m src.train --mode binary --run-name binary                     # Negative vs Positive only
python -m src.train --seed 1 --run-name tuned_s1                        # repeat with other seeds
```

## Results

> Fill in from `results/*.json` after your runs (`python -m src.compare` prints the table).

**Test set, 3-class** (N = _TBD_ test texts)

| Model | Normalised | Accuracy | Macro-F1 |
|---|---|---|---|
| TF-IDF + LinearSVC | yes | _TBD_ | _TBD_ |
| DistilBERT | no | _TBD_ | _TBD_ |
| DistilBERT | yes | _TBD_ | _TBD_ |

**Inference speed** (GPU: _TBD_, ms per review)

| Setup | ms / review | Reduction |
|---|---|---|
| FP32, batch 1 | _TBD_ | baseline |
| FP32, batch 32 | _TBD_ | _TBD_ |
| FP16, batch 32 | _TBD_ | _TBD_ |

FP16 macro-F1 vs FP32: _TBD_ (the benchmark checks that FP16 does not hurt accuracy).

### Reading the results honestly

- All numbers come from a single seed unless you repeat runs. Differences under about one point are within noise.
- The test split is used once per model, after choosing the checkpoint on the validation split.
- The neutral class is the hardest (it is vague, even for human annotators), so 3-class scores are lower than binary scores.

## Project structure

```
src/
  config.py        paths and constants
  normalize.py     Roman Urdu normaliser
  data.py          download, clean, split
  baseline.py      TF-IDF + LinearSVC
  train.py         DistilBERT fine-tuning (PyTorch loop)
  predict.py       batched inference wrapper (FP16 support)
  evaluate.py      evaluate a saved model
  benchmark.py     ms-per-review benchmark
  compare.py       results table
  push_to_hub.py   upload model to Hugging Face
space/             Gradio demo for Hugging Face Spaces
notebooks/         Colab notebook
tests/             pytest suite
```

## Demo on Hugging Face Spaces

1. Push the model: `python -m src.push_to_hub --model-dir models/tuned --repo-id <hf-username>/roman-urdu-sentiment-distilbert`
2. Create a new **Gradio** Space and upload the four files in `space/` (`app.py`, `normalize.py`, `requirements.txt`, `README.md`).
3. In `space/app.py`, set `MODEL_ID` (or add a `MODEL_ID` Space variable) to your Hub repo id.

Run the demo locally with `python space/app.py`.

## Tests

```bash
pytest
```

## Limitations

- Labels are crowd-sourced and noisy; sarcasm and code-mixing are hard for any model.
- The data covers reviews and social media text, so performance on other domains is not guaranteed.
- The spelling-variant table is hand-built and conservative; extend `VARIANTS` in `src/normalize.py` using frequent tokens from your data (and copy the file to `space/`, a test checks they match).
- Roman Urdu has no standard orthography, so some variants will always be missed.

## License

MIT. The dataset is CC BY 4.0 (see above).
