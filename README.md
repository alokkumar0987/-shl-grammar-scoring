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
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ INPUT   985 spoken answers, 45-60 s, 16 kHz mono                                             │
 │         769 training clips with MOS grammar labels (0-5)  +  216 test clips                  │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1  PREPROCESSING AND DATA CHECKS                                                             │
 │    DC removal -> silence trim (35 dB) -> peak normalisation                                  │
 │    speaker x-vectors -> 486 speaker clusters      => CV groups + speaker prior (step 5)      │
 │    37 clips labelled 0 = separate recording batch => kept out of regression, own detector    │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                ▼                                ▼
 ┌────────────────────────────┐   ┌────────────────────────────┐   ┌────────────────────────────┐
 │ 2a AUDIO VIEWS             │   │ 2b TRANSCRIPT VIEWS        │   │ 2c RUBRIC SIGNALS (59)     │
 │    how it is said          │   │    what is said            │   │    interpretable evidence  │
 │                            │   │                            │   │                            │
 │ WavLM-large                │   │ Whisper large-v3 ASR,      │   │ fluency: rate, pauses, um  │
 │ Whisper large-v3 encoder   │   │ verbatim prompt keeps      │   │ LanguageTool error rate    │
 │ w2v-BERT 2.0               │   │ fillers and errors         │   │ GPT-2 surprisal, CoLA      │
 │ Qwen2-Audio-7B (4-bit)     │   │  -> RoBERTa-large          │   │ CoEdIT corrections/word    │
 │                            │   │  -> Qwen2.5-7B states      │   │ CTC vs Whisper distance    │
 │ mean + std pooling of      │   │  -> Qwen3-14B states       │   │ LLM judges P(score 1..5):  │
 │ every layer over time      │   │ mean pooling, every layer  │   │  Qwen2.5-7B (zero-shot)    │
 │                            │   │                            │   │  Qwen3-14B (4 anchors)     │
 └────────────────────────────┘   └────────────────────────────┘   └────────────────────────────┘
                │                                │                                │
    best layer / window (CV)         best layer / window (CV)                     │
                │                                │                                │
                └────────────────────────────────┼────────────────────────────────┘
                                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 3  BASE MODELS   7 embedding views x {Ridge, SVR (tuned C), ordinal head, KNN} = 28          │
 │                  + rubric signals x {Ridge, gradient boosting, ordinal head}  = 3  -> 31     │
 │    speaker-grouped stratified 5-fold CV x 3 seeds -> out-of-fold predictions                 │
 │    (a speaker is never in training and validation of the same fold)                          │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 4  STACKING      non-negative linear regression on the 31 out-of-fold predictions            │
 │    chosen over Caruana ensemble selection by nested CV; calibration check: none needed       │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 5  POST-PROCESSING                                                                           │
 │    25 test clips of known speakers -> 0.55 x model + 0.45 x that speaker's mean score        │
 │    zero-score detector (p > 0.8) -> 0   (37/37 found in CV, 0 flagged in test)               │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ OUTPUT  submission.csv (216 rows)                                                            │
 │         public LB RMSE 0.3416  |  CV RMSE 0.4983, Pearson 0.916  |  training RMSE 0.1223     │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Validation scheme

```
 732 scored training clips ──► 486 speaker clusters (x-vector cosine similarity, threshold chosen from the labels)
                                        │
                                        ▼
            StratifiedGroupKFold: 5 folds, stratified on the rounded score, grouped by speaker
            fold 1   [  VAL  ][ train ][ train ][ train ][ train ]
            fold 2   [ train ][  VAL  ][ train ][ train ][ train ]
              ...       all clips of one speaker always fall in the same block
            fold 5   [ train ][ train ][ train ][ train ][  VAL  ]
                                        │   repeated with 3 seeds
                                        ▼
            every clip gets an out-of-fold prediction  ──►  CV RMSE / Pearson
            the stacker is cross-validated again on the same folds (nested), so its weights never
            see the clips they are scored on
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

## Iterations

| Version | What changed | CV RMSE (scored) | Public LB |
|---|---|---|---|
| v1 | WavLM, Whisper encoder, Qwen2-Audio and RoBERTa embeddings + handcrafted grammar/fluency features, Ridge/SVR, non-negative stacking, speaker-grouped CV, zero-score detector | 0.5217 | 0.3515 |
| v3 | + w2v-BERT 2.0, Qwen2.5-7B and Qwen3-14B rubric judges (with scored anchors), CoEdIT error-correction rate, CTC-vs-Whisper distance, layer windows, ordinal and KNN heads, tuned SVR, speaker prior, feature store | 0.5108 | **0.3416** |

Every change was kept or dropped on the basis of speaker-grouped CV, never on the public leaderboard alone.

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
