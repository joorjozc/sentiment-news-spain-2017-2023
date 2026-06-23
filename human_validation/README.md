# Human Validation of the Emotion Classifier

This folder contains the human-annotation study used to validate the RoBERTa-BNE
emotion classifier against reader-perceived emotion, added in the revised version of
the manuscript (Section III-B).

## Design

- **Sample:** a representative subset of **150 headlines** drawn from the corpus
  (seed 42), stored with the model probability vectors in `_answer_key_v2.csv`.
- **Annotators (blind protocol, no access to model outputs):**
  - **A1** — 36-year-old female, Millennial cohort (`Anotacion_Humana_v2_A1_Millenial.xlsx`).
  - **A2** — 21-year-old male, Generation Z cohort (`Anotacion_Humana_v2_A2_Gen Z.xlsx`).
- **Tasks per headline:** (i) a single **dominant emotion** (7 categories) and
  (ii) the **intensity (0–4)** of the four core emotions (anger, fear, joy, sadness).

## Files

| File | Description |
|------|-------------|
| `Anotacion_Humana_v2_A1_Millenial.xlsx` | Annotations by A1 (Millennial). |
| `Anotacion_Humana_v2_A2_Gen Z.xlsx` | Annotations by A2 (Gen Z). |
| `_answer_key_v2.csv` | Sampled headlines + RoBERTa-BNE probability vectors (evaluation key). |
| `_build_validation_sample_v2.py` | Builds the representative sample and the answer key. |
| `_score_validation_v2.py` | Computes all agreement metrics. |
| `resultados_validacion_v2.json` | Computed results. |
| `protocolo_anotacion.md` | Annotation protocol given to the annotators. |

## Reproduce

```bash
python _score_validation_v2.py   # regenerates resultados_validacion_v2.json
```

Requires `numpy`, `pandas`, `openpyxl` (no `scikit-learn`/`scipy` dependency; Cohen's
κ, Gwet's AC1, PABAK and Spearman ρ are implemented directly).

## Key results

Because the corpus is dominated by neutral headlines, we report **Gwet's AC1**
(robust to high prevalence) alongside Cohen's κ.

| Comparison | Accuracy | Cohen κ | Gwet AC1 |
|------------|---------:|--------:|---------:|
| A1 (Millennial) vs model | 0.600 | 0.269 | 0.560 |
| A2 (Gen Z) vs model | 0.480 | 0.170 | 0.420 |
| Inter-annotator (A1 vs A2) | 0.613 | 0.348 | 0.572 |
| **Consensus** (92/150 agreed) **vs model** | **0.641** | **0.291** | **0.609** |

Human-perceived intensity correlated positively with the model probabilities
(consensus Spearman ρ = 0.29). The moderate inter-annotator agreement underscores the
intrinsic subjectivity of single-headline emotion perception.
