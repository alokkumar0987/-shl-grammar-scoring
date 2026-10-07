# Grammar Scoring Engine for Spoken English

SHL Hiring Assessment 2026 (private Kaggle challenge). The model takes a 45–60 s spoken answer and predicts a grammar score
on the 0–5 MOS rubric.

| | v1 (baseline) | **v3 (final)** |
|---|---|---|
| Public leaderboard RMSE | 0.3515 | **0.3416** |
| Cross-validated RMSE (speaker-grouped, 732 scored clips) | 0.5217 | **0.5108** |
| Cross-validated RMSE / Pearson (all 769 training clips) | 0.5089 / 0.912 | **0.4983 / 0.916** |
| Training RMSE (in-sample, all 769 clips) | 0.1886 | **0.1223** |

The final submission is [`submission.csv`](submission.csv) (216 rows, from `test.csv`).

## Repository

```
notebooks/
  grammar_scoring_engine_v3.ipynb           final notebook, executed on Kaggle (code, report, plots, metrics)
  grammar_scoring_engine_v1_baseline.ipynb  first version, executed on Kaggle
submission.csv                              predictions of the final notebook
requirements.txt
```

The competition audio is not included in this repository.

## Approach

Grammar shows up in two places: in **what** is said (the words and sentence structure) and in **how** it is said (fluency,
hesitation, self-correction). The pipeline therefore looks at each clip in three ways, fits small regularised models on
each, and combines them.

```
                         audio clip (16 kHz)
                                │
     preprocessing: silence trim, normalisation, speaker clusters, zero-score detector
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
   AUDIO VIEWS             TRANSCRIPT VIEWS           RUBRIC SIGNALS
   WavLM-large             Whisper large-v3,          LanguageTool, GPT-2 surprisal,
   Whisper-L3 encoder      prompted to keep           CoLA acceptability,
   w2v-BERT 2.0            fillers and errors         LLM judges (Qwen2.5-7B, Qwen3-14B),
   Qwen2-Audio-7B          RoBERTa-large, LLM states  CoEdIT corrections, CTC vs Whisper
        └───────────────────────┼─────────────────────────┘
                                ▼
       Ridge / SVR / ordinal heads per view, speaker-grouped 5-fold CV × 3 seeds
                                ▼
                 non-negative linear stacking of out-of-fold predictions
                                ▼
          speaker prior for known speakers, zero-score clips → 0 → submission.csv
```

### Key decisions

| Observation | Decision |
|---|---|
| 37 training clips are labelled 0, which is outside the 1–5 rubric. They contain normal speech but form a separate, perfectly recognisable batch. | Removed from regression; a small logistic-regression detector maps such clips to 0. It finds 37/37 in CV and flags none in the test set. |
| The same speakers appear several times with nearly identical scores (score std inside a speaker cluster 0.24 vs 1.01 overall). | Speaker x-vectors are clustered and used as groups in `StratifiedGroupKFold`, so validation always uses unseen speakers. Without this, CV is over-optimistic. |
| Whisper tends to "clean up" learner speech, hiding the errors being scored. | Whisper is prompted with a disfluent, ungrammatical example so the transcript stays verbatim. A second, letter-level CTC transcript measures how much Whisper repaired. |
| Pretrained encoders store different information in different layers. | Each encoder's best layer (or a small window of layers) is chosen by cross-validation. |
| Only 732 usable training clips. | Frozen pretrained features with strongly regularised models (Ridge, SVR, ordinal logistic heads) instead of fine-tuning. |
| About 1 in 9 test clips comes from a speaker who is also in the training data. | For those clips the prediction is blended with that speaker's known scores; the weight (0.45) is estimated on training clips of repeated speakers. |
| `sample_submission.csv` lists IDs that mostly do not match `test.csv`. | The submission is built from the 216 files in `test.csv`. |

### What helps (ablation, CV RMSE on scored clips, lower is better)

| Views used in the stack | CV RMSE |
|---|---|
| handcrafted + LLM judges + GEC | 0.629 |
| transcript views (RoBERTa + LLM states) | 0.571 |
| audio views only | 0.541 |
| all views | **0.511** |

Audio encoders carry the most signal. The LLM judges and grammatical-error-correction features are the strongest
interpretable signals: the Qwen3-14B judge, given the rubric and four scored example transcripts, reaches Spearman 0.68
with the true score on its own. The main remaining error is at the low end: speakers scored 2 who speak fluently are often
predicted around 2.5–3.

## Evaluation

- **Training RMSE** (required): all models refit on the full training set and scored on it. This number is optimistic by
  construction.
- **Cross-validated RMSE and Pearson**: out-of-fold predictions from speaker-grouped 5-fold CV, repeated with 3 seeds; the
  stacker is evaluated with a second, nested CV. This is the estimate used for every modelling decision.
- Public leaderboard RMSE has stayed close to 0.67 × CV RMSE, so CV is a reliable guide.

## Reproducing

The notebook is written for a Kaggle notebook with a T4 GPU and internet access:

1. Attach the competition data; the notebook reads `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`.
2. Settings → Accelerator **GPU T4**, Internet **on**.
3. **Save Version → Save & Run All**. The first run takes about 4.5 hours, mostly ASR and the LLMs.
4. `submission.csv` is written to `/kaggle/working`. All extracted features are saved to `/kaggle/working/features`; adding
   this output as an input to a later run reuses them, so model experiments take minutes.

Without a GPU the notebook still runs, with smaller models and without the 7B/14B LLMs.

## References

- Bannò & Matassoni (2022). *Proficiency assessment of L2 spoken English using wav2vec 2.0.* arXiv:2210.13168
- Ma et al. (2025). *Assessment of L2 Oral Proficiency using Speech Large Language Models.* arXiv:2505.21148
- *Speak & Improve Challenge 2025: Tasks and Baseline Systems.* arXiv:2412.11985
- Knill, Gales, Manakul & Caines (2019). *Automatic grammatical error detection of non-native spoken learner English.* ICASSP
- Raheja et al. (2023). *CoEdIT: Text Editing by Task-Specific Instruction Tuning.* arXiv:2305.09857
- Chen et al. (2022). *WavLM*; Radford et al. (2022). *Whisper*; Baevski et al. (2020). *wav2vec 2.0*
- Caruana et al. (2004). *Ensemble Selection from Libraries of Models.* ICML
